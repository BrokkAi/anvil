use anyhow::Result;
use tokio_util::sync::CancellationToken;

use crate::asgard::{CheckpointId, TrajectoryDag};
use crate::llm_client::{
    ChatMessage, FunctionDef, LlmBackend, LlmResponse, StreamChatRequest, ToolCall, ToolDefinition,
};

pub(crate) const ASGARD_MAX_IN_FLIGHT: usize = 5;
pub(crate) const ASGARD_WORKER_MAX_STEPS: usize = 10;
pub(crate) const ASGARD_SUPERVISOR_MAX_STEPS: usize = 10;
pub(crate) const ASGARD_VIEW_TOOL_CALL_MAX_HANDLES: usize = 16;

pub(crate) const SPAWN_WORKERS_TOOL: &str = "spawn_workers";
pub(crate) const SAVE_CHECKPOINT_TOOL: &str = "save_checkpoint";
pub(crate) const DISCARD_TOOL: &str = "discard";
pub(crate) const FINALIZE_TOOL: &str = "finalize";
pub(crate) const VIEW_TOOL_CALL_TOOL: &str = "view_tool_call";
pub(crate) const WAIT_TOOL: &str = "wait";
pub(crate) const UPDATE_PLAN_TOOL: &str = "update_plan";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpawnRequest {
    pub(crate) from: CheckpointId,
    pub(crate) instructions: String,
    pub(crate) model: Option<String>,
}

pub(crate) struct FinalizeRequest {
    pub(crate) checkpoint: CheckpointId,
    pub(crate) response: Option<String>,
    pub(crate) evidence: Vec<String>,
}

pub(crate) struct SupervisorTurnContext<'a> {
    pub(crate) dag: &'a TrajectoryDag,
    pub(crate) pending: Option<usize>,
    pub(crate) allowed_models: &'a [String],
}

pub(crate) struct SupervisorStreamCall<'a> {
    pub(crate) llm: &'a dyn LlmBackend,
    pub(crate) model: &'a str,
    pub(crate) request_prefix: &'a [ChatMessage],
    pub(crate) tail: &'a [ChatMessage],
    pub(crate) tools: &'a [ToolDefinition],
    pub(crate) cancel: &'a CancellationToken,
    pub(crate) idle_timeout: crate::llm_client::IdleTimeouts,
    pub(crate) text_sink: Option<crate::tool_loop::TextSink>,
    pub(crate) thought_sink: Option<crate::tool_loop::TextSink>,
}

pub(crate) fn supervisor_supplement() -> String {
    format!(
        r#"# Asgard supervision

You are the supervisor of a team of asynchronous workers solving the task. Everything above still governs the work: workers are agents operating under those instructions with the full standard agent toolset (file reading and editing, search, code intelligence, and the shell), and the standards in "How you work" and "Verification" are requirements you enforce through your workers, not suggestions. The difference is that you never touch files or run commands yourself - you act only through these tools:
- spawn_workers: fork workers from "root" (the original repository state) or any saved checkpoint like "w3". Each worker gets its own checkout of that state plus your instructions, runs up to 10 steps (one step = one batch of tool calls), then reports back for review. Workers may finish sooner; a worker that stops making tool calls is done, and its final message is its report to you.
- save_checkpoint: a reviewed trajectory you save (or spawn a worker from) becomes a permanent checkpoint you can branch from later.
- discard: permanently discard the just-reviewed trajectory.
- view_tool_call: expands compact-trace handles like "w3m5" into complete, untruncated arguments and results. Viewing is free - use it whenever a summarized line matters to your decision. Handles exist only for trajectories that have been presented for review; you cannot watch a worker that is still running. To wait for in-flight workers, call wait (or simply reply without tool calls) - you will be re-engaged when the next one finishes.
- wait: end this turn and wait. You are re-engaged automatically the moment the next worker finishes. Use this instead of polling, acknowledgment workers, or empty tool calls.
- update_plan: maintain the user-visible plan for the overall task. Workers cannot see or update it; fold their progress into it yourself.
- finalize: ends the run. The named checkpoint's repository state is delivered as the result, and that worker's final message (or the response you provide) becomes the final answer.

Reviews: you review one finished worker at a time - where it forked from, your instructions, a compact trace of each step, a diffstat, and its final message verbatim. Each of your turns allows up to 10 steps (a step = one batch of tool calls) - the same budget your workers get. Each turn you also receive an ephemeral <dag> overview of every fragment by id, including discarded ones. You may mix viewing, plan updates, spawning, and saving freely across up to {supervisor_steps} supervisor steps. Every reviewed trajectory must be resolved before your turn ends: save_checkpoint it, spawn a worker from it (which saves it), or discard it. Discarded trajectories are gone permanently and their handles die with them. At most {workers} workers can be in flight; the others appear as brief status lines and get their own review turns. Workers inherit the full conversation along their ancestor chain plus your new instructions, and know nothing about sibling workers or your plans - put everything they need into the instructions.

Finalize is where "Verification" binds you: a worker's report is a claim, not evidence. Do not finalize until the Verification requirements above have actually been discharged on the finalized checkpoint's chain - real test runs whose commands and output you have inspected via view_tool_call. A pre-existing suite that passed before the change proves nothing about the change: the evidence must include tests that exercise the new behavior the task demands - ideally the spec tests written from the task text at the start. If that evidence does not exist yet, spawn a verification worker on that checkpoint first and review what it finds.

Strategy is yours: run independent approaches in parallel, spawn short fact-finding workers to answer questions ("find how X is implemented; report file and line"), branch variants from a good checkpoint, cut losses early, and keep instructions explicit and testable. Start every task by having a worker pin the specification: identify the details of the task statement that admit more than one reading (exact formats, encodings, argument orders, boundary values, naming), and write tests derived from the task text alone - before any implementation exists - that lock in the required reading. Those spec tests are your contract: implementation workers run them, and your finalize evidence should show them passing. Delivery, branches, and commit ceremony are handled outside the run - never spend a worker on commit messages or branch bookkeeping."#,
        supervisor_steps = ASGARD_SUPERVISOR_MAX_STEPS,
        workers = ASGARD_MAX_IN_FLIGHT
    )
}

pub(crate) fn supervisor_tool_definitions(allowed_models: &[String]) -> Vec<ToolDefinition> {
    let mut model_enum: Vec<&String> = Vec::new();
    for model in allowed_models {
        if !model_enum.contains(&model) {
            model_enum.push(model);
        }
    }
    let allowed_models = model_enum;
    vec![
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: SPAWN_WORKERS_TOOL.to_string(),
                description: "Fork new workers from \"root\" or a saved checkpoint id like \"w3\" (or the just-reviewed worker's id, which saves it); each runs up to 10 steps in its own checkout, then reports back. Never spawn a worker to wait, poll, or acknowledge: call wait (or simply reply without tool calls) to wait, and you are re-engaged automatically the moment a worker finishes.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["workers"],
                    "properties": {
                        "workers": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": ASGARD_MAX_IN_FLIGHT,
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["from", "instructions"],
                                "properties": {
                                    "from": { "type": "string" },
                                    "instructions": { "type": "string", "minLength": 1 },
                                    "model": {
                                        "type": "string",
                                        "enum": allowed_models,
                                    },
                                },
                            },
                        },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: SAVE_CHECKPOINT_TOOL.to_string(),
                description: "Save the just-reviewed trajectory as a permanent checkpoint without spawning from it yet.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {},
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: DISCARD_TOOL.to_string(),
                description: "Discard the just-reviewed trajectory permanently. Every reviewed trajectory must be saved, spawned from (which saves it implicitly), or discarded.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {},
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: FINALIZE_TOOL.to_string(),
                description: "End the run delivering the named checkpoint's state; response overrides the checkpoint worker's final message as the user-facing answer. Name evidence handles for inspected test runs whenever possible.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["checkpoint"],
                    "properties": {
                        "checkpoint": { "type": "string" },
                        "response": { "type": "string" },
                        "evidence": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "the handles of the test runs you inspected that verify the finalized state",
                        },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: WAIT_TOOL.to_string(),
                description: "End this turn and wait. You are re-engaged automatically the moment the next worker finishes. Use this instead of polling, acknowledgment workers, or empty tool calls.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {},
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: VIEW_TOOL_CALL_TOOL.to_string(),
                description: "Expand compact-trace handles like \"w3m5\" into complete untruncated arguments and results; free of charge; works across every saved trajectory and the one under review.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["handles"],
                    "properties": {
                        "handles": {
                            "type": "array",
                            "maxItems": ASGARD_VIEW_TOOL_CALL_MAX_HANDLES,
                            "items": { "type": "string" },
                        },
                    },
                }),
            },
        },
        crate::tools::update_plan_tool_definition(),
    ]
}

pub(crate) async fn stream_supervisor_response(
    call: SupervisorStreamCall<'_>,
) -> Result<LlmResponse> {
    crate::llm_client::stream_chat_no_visible_output_with_retry(
        call.llm,
        "running Asgard supervisor turn",
        call.cancel,
        || {
            let mut messages = call.request_prefix.to_vec();
            messages.extend(call.tail.to_vec());
            let text_sink = call.text_sink.clone();
            let thought_sink = call.thought_sink.clone();
            StreamChatRequest {
                model: call.model.to_string(),
                messages,
                tools: Some(call.tools.to_vec()),
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(move |token| {
                    if let Some(sink) = &text_sink {
                        (sink
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner))(
                            token
                        );
                    }
                }),
                on_thought: Box::new(move |token| {
                    if let Some(sink) = &thought_sink {
                        (sink
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner))(
                            token
                        );
                    }
                }),
                cancel: call.cancel.clone(),
                idle_timeouts: call.idle_timeout,
            }
        },
    )
    .await
}

pub(crate) fn parse_view_tool_call(call: &ToolCall) -> std::result::Result<Vec<String>, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    let handles = string_array_property(&arguments, "handles")?;
    if handles.len() > ASGARD_VIEW_TOOL_CALL_MAX_HANDLES {
        return Err(format!(
            "view_tool_call accepts at most {ASGARD_VIEW_TOOL_CALL_MAX_HANDLES} handles, got {}",
            handles.len()
        ));
    }
    Ok(handles)
}

pub(crate) fn parse_spawn_workers(
    call: &ToolCall,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<Vec<SpawnRequest>, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    let workers = arguments
        .get("workers")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "workers must be an array".to_string())?;
    if workers.is_empty() {
        return Err(
            "workers must contain at least one worker. If you meant to wait for \
                    in-flight workers, call wait (or simply reply without tool calls) instead — you are \
                    re-engaged automatically when the next worker finishes."
                .to_string(),
        );
    }
    if workers.len() > ASGARD_MAX_IN_FLIGHT {
        return Err(format!(
            "workers must contain at most {ASGARD_MAX_IN_FLIGHT} workers"
        ));
    }

    workers
        .iter()
        .enumerate()
        .map(|(index, worker)| parse_spawn_worker(index, worker, context))
        .collect()
}

fn parse_spawn_worker(
    index: usize,
    worker: &serde_json::Value,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<SpawnRequest, String> {
    let from_raw = worker
        .get("from")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("workers[{index}].from must be a string"))?;
    let from = parse_available_checkpoint(from_raw, context)
        .map_err(|error| format!("workers[{index}].from {error}"))?;
    let instructions = worker
        .get("instructions")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("workers[{index}].instructions must be a string"))?
        .trim()
        .to_string();
    if instructions.is_empty() {
        return Err(format!("workers[{index}].instructions must not be empty"));
    }
    let model = worker
        .get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    if let Some(model) = &model
        && !context
            .allowed_models
            .iter()
            .any(|allowed| allowed == model)
    {
        return Err(format!("workers[{index}].model {model:?} is not allowed"));
    }
    Ok(SpawnRequest {
        from,
        instructions,
        model,
    })
}

pub(crate) fn parse_finalize(
    call: &ToolCall,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<FinalizeRequest, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    let checkpoint_raw = arguments
        .get("checkpoint")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "checkpoint must be a string".to_string())?;
    let checkpoint = parse_available_checkpoint(checkpoint_raw, context)
        .map_err(|error| format!("checkpoint {error}"))?;
    let response = arguments
        .get("response")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let evidence = match arguments.get("evidence") {
        Some(_) => string_array_property(&arguments, "evidence")?,
        None => Vec::new(),
    };
    Ok(FinalizeRequest {
        checkpoint,
        response,
        evidence,
    })
}

pub(crate) fn parse_update_plan(
    call: &ToolCall,
) -> std::result::Result<crate::plan::UpdatePlanArgs, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    serde_json::from_value::<crate::plan::UpdatePlanArgs>(arguments)
        .map_err(|error| format!("Invalid update_plan arguments: {error}"))
}

pub(crate) fn elide_view_tool_results_for_permanent_record(
    transcript: &[ChatMessage],
) -> Vec<ChatMessage> {
    let mut view_handles_by_call = std::collections::HashMap::new();
    for message in transcript {
        for call in message.tool_calls.iter().flatten() {
            if call.function.name == VIEW_TOOL_CALL_TOOL {
                let handles = parse_view_tool_call(call).unwrap_or_default();
                view_handles_by_call.insert(call.id.clone(), handles);
            }
        }
    }

    // Errors and other short results stay verbatim: eliding an error hides the
    // failure from the supervisor's permanent record, and we observed it then
    // repeating the same failing call turn after turn. Only genuinely large
    // expansions are worth eliding.
    const ELISION_MIN_BYTES: usize = 400;

    transcript
        .iter()
        .map(|message| {
            if message.role == "tool"
                && message.name.as_deref() == Some(VIEW_TOOL_CALL_TOOL)
                && message.content_text().len() >= ELISION_MIN_BYTES
                && let Some(call_id) = &message.tool_call_id
            {
                let handles = view_handles_by_call
                    .get(call_id)
                    .filter(|handles| !handles.is_empty())
                    .map(|handles| handles.join(", "))
                    .unwrap_or_else(|| "unknown handles".to_string());
                return ChatMessage::tool_result(
                    call_id,
                    VIEW_TOOL_CALL_TOOL,
                    format!("[expanded {handles} - elided from the permanent record]"),
                );
            }
            message.clone()
        })
        .collect()
}

fn parse_available_checkpoint(
    value: &str,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<CheckpointId, String> {
    let checkpoint =
        CheckpointId::parse(value).ok_or_else(|| format!("{value:?} is not a checkpoint id"))?;
    if context.dag.contains(&checkpoint) || is_pending_checkpoint(&checkpoint, context) {
        Ok(checkpoint)
    } else {
        Err(format!("{checkpoint} is not root, saved, or under review"))
    }
}

pub(crate) fn is_pending_checkpoint(
    checkpoint: &CheckpointId,
    context: &SupervisorTurnContext<'_>,
) -> bool {
    matches!(
        (checkpoint, context.pending),
        (CheckpointId::Worker(worker), Some(pending_worker)) if *worker == pending_worker
    )
}

fn normalize_arguments(raw: &str) -> std::result::Result<serde_json::Value, String> {
    crate::tool_arguments::normalize_tool_arguments(raw)
        .map(|arguments| arguments.value)
        .map_err(|error| format!("unparseable arguments: {error:#}"))
}

fn string_array_property(
    arguments: &serde_json::Value,
    property: &str,
) -> std::result::Result<Vec<String>, String> {
    arguments
        .get(property)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{property} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{property}[{index}] must be a string"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::asgard::{TrajectoryNode, TrajectoryWindow, WorkerStopReason};
    use crate::llm_client::{FunctionCall, TokenUsage};

    fn supervisor_tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn saved_dag() -> TrajectoryDag {
        let mut dag = TrajectoryDag::new(vec![ChatMessage::user("task")], "base".to_string());
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 1,
                parent: CheckpointId::Root,
                instructions: "saved".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("saved worker instructions"),
                window_messages: Vec::new(),
                compact: String::new(),
                final_response: "saved result".to_string(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: "commit-1".to_string(),
        })
        .unwrap();
        dag
    }

    fn context<'a>(dag: &'a TrajectoryDag, pending: Option<usize>) -> SupervisorTurnContext<'a> {
        static ALLOWED: &[String] = &[];
        SupervisorTurnContext {
            dag,
            pending,
            allowed_models: ALLOWED,
        }
    }

    #[test]
    fn spawn_parser_accepts_root_saved_and_pending_checkpoints() {
        let dag = saved_dag();
        let allowed = vec!["model-a".to_string()];
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: Some(4),
            allowed_models: &allowed,
        };
        let call = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [
                    { "from": "root", "instructions": "bootstrap" },
                    { "from": "w1", "instructions": "branch saved", "model": "model-a" },
                    { "from": "w4", "instructions": "branch pending" }
                ]
            }),
        );

        let spawns = parse_spawn_workers(&call, &context).expect("valid spawns");

        assert_eq!(spawns.len(), 3);
        assert_eq!(spawns[2].from, CheckpointId::Worker(4));
    }

    #[test]
    fn spawn_parser_rejects_bad_checkpoint_model_and_empty_instructions() {
        let dag = saved_dag();
        let allowed = vec!["model-a".to_string()];
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: None,
            allowed_models: &allowed,
        };

        let bad_from = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "w99", "instructions": "x" }] }),
        );
        assert!(
            parse_spawn_workers(&bad_from, &context)
                .expect_err("bad checkpoint")
                .contains("w99 is not root, saved, or under review")
        );

        let bad_model = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x", "model": "model-b" }] }),
        );
        assert!(
            parse_spawn_workers(&bad_model, &context)
                .expect_err("bad model")
                .contains("model-b")
        );

        let empty = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "  " }] }),
        );
        assert!(
            parse_spawn_workers(&empty, &context)
                .expect_err("empty instructions")
                .contains("must not be empty")
        );
    }

    #[test]
    fn finalize_parser_accepts_pending_and_response() {
        let dag = saved_dag();
        let call = supervisor_tool_call(
            "finalize",
            FINALIZE_TOOL,
            serde_json::json!({ "checkpoint": "w4", "response": "done" }),
        );

        let parsed = parse_finalize(&call, &context(&dag, Some(4))).expect("valid finalize");

        assert_eq!(parsed.checkpoint, CheckpointId::Worker(4));
        assert_eq!(parsed.response.as_deref(), Some("done"));
        assert!(parsed.evidence.is_empty());
    }

    #[test]
    fn view_parser_enforces_handle_array_and_cap() {
        let call = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": ["w1m1", "w2m3"] }),
        );
        assert_eq!(
            parse_view_tool_call(&call).expect("valid handles"),
            vec!["w1m1".to_string(), "w2m3".to_string()]
        );

        let too_many = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": vec!["x"; ASGARD_VIEW_TOOL_CALL_MAX_HANDLES + 1] }),
        );
        assert!(
            parse_view_tool_call(&too_many)
                .expect_err("too many")
                .contains("at most")
        );
    }

    #[test]
    fn update_plan_parser_reports_schema_errors() {
        let call = supervisor_tool_call(
            "plan",
            UPDATE_PLAN_TOOL,
            serde_json::json!({ "plan": [{ "step": "missing status" }] }),
        );

        assert!(
            parse_update_plan(&call)
                .expect_err("invalid plan")
                .contains("Invalid update_plan arguments")
        );
    }

    #[test]
    fn permanent_record_elides_only_view_tool_results() {
        let view_call = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": ["w1m1", "w2m3"] }),
        );
        let spawn_call = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x" }] }),
        );
        let error_view_call = supervisor_tool_call(
            "view-err",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": ["w9"] }),
        );
        let huge_payload = format!("huge payload {}", "x".repeat(500));
        let transcript = vec![
            ChatMessage::assistant_tool_calls(vec![view_call, spawn_call, error_view_call]),
            ChatMessage::tool_result("view", VIEW_TOOL_CALL_TOOL, huge_payload.clone()),
            ChatMessage::tool_result("spawn", SPAWN_WORKERS_TOOL, "spawned w3 from root"),
            ChatMessage::tool_result(
                "view-err",
                VIEW_TOOL_CALL_TOOL,
                "<tool_call handle=\"w9\" error=\"malformed handle\" />",
            ),
        ];

        let permanent = elide_view_tool_results_for_permanent_record(&transcript);
        let text = permanent
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!text.contains(&huge_payload));
        assert!(text.contains("[expanded w1m1, w2m3 - elided from the permanent record]"));
        assert!(text.contains("spawned w3 from root"));
        // Short results — errors especially — must survive verbatim so the
        // supervisor can see that a call failed and not repeat it.
        assert!(text.contains(r#"<tool_call handle="w9" error="malformed handle" />"#));
    }
}
