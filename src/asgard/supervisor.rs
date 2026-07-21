use anyhow::Result;
use tokio_util::sync::CancellationToken;

use crate::asgard::{CheckpointId, TrajectoryDag};
use crate::llm_client::{
    ChatMessage, FunctionDef, LlmBackend, LlmResponse, StreamChatRequest, ToolCall, ToolDefinition,
};

pub(crate) const ASGARD_BATCH_CAP: usize = 8;
pub(crate) const ASGARD_WORKER_MAX_STEPS: usize = 10;
pub(crate) const ASGARD_SUPERVISOR_MAX_STEPS: usize = 15;
pub(crate) const ASGARD_VIEW_TOOL_CALL_MAX_HANDLES: usize = 16;

pub(crate) const SPAWN_WORKERS_TOOL: &str = "spawn_workers";
pub(crate) const PREFINALIZE_TOOL: &str = "prefinalize";
pub(crate) const SAVE_CHECKPOINT_TOOL: &str = "save_checkpoint";
pub(crate) const MERGE_CHECKPOINT_TOOL: &str = "merge_checkpoint";
pub(crate) const DISCARD_TOOL: &str = "discard";
pub(crate) const FINALIZE_TOOL: &str = "finalize";
pub(crate) const VIEW_TOOL_CALL_TOOL: &str = "view_tool_call";
pub(crate) const UPDATE_PLAN_TOOL: &str = "update_plan";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpawnRequest {
    pub(crate) from: CheckpointId,
    pub(crate) instructions: String,
    pub(crate) model: Option<String>,
    pub(crate) injected_files: Vec<(String, Vec<u8>)>,
    pub(crate) max_steps: Option<usize>,
}

pub(crate) struct FinalizeRequest {
    pub(crate) checkpoint: CheckpointId,
    pub(crate) response: Option<String>,
    pub(crate) evidence: Vec<String>,
    pub(crate) abandoned: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct MergeCheckpointRequest {
    pub(crate) from: CheckpointId,
    pub(crate) onto: CheckpointId,
}

pub(crate) struct SupervisorTurnContext<'a> {
    pub(crate) dag: &'a TrajectoryDag,
    pub(crate) pending: &'a [usize],
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

You are the supervisor of a team of barrier-batch workers solving the task. Everything above still governs the work: workers are agents operating under those instructions with the full standard agent toolset (file reading and editing, search, code intelligence, and the shell), and the standards in "How you work" and "Verification" are requirements you enforce through your workers, not suggestions. You never touch files or run commands yourself - you act only through these tools:
- spawn_workers: fork 1 to {workers} workers from "root" (the original repository state) or any saved checkpoint like "w3". A successful spawn ends your turn; the whole batch runs concurrently to completion, then all sibling reports return together for review. Each worker gets its own checkout of the forked state plus your instructions and runs up to {worker_steps} steps (one step = one batch of tool calls); a worker that stops making tool calls sooner is done, and its final message is its report to you.
- prefinalize: spawn your final verification pass, with the same batch semantics as spawn_workers. finalize stays locked until a prefinalize batch has run and every one of its reports has been reviewed and resolved.
- save_checkpoint: a reviewed trajectory you save (or spawn a worker from) becomes a permanent checkpoint you can branch from later. When multiple siblings are under review, pass `worker` (for example "w7") to name which one.
- merge_checkpoint: combine two saved checkpoints with a real git merge, producing a new checkpoint - this is how a split batch's parallel work comes back together.
- discard: permanently discard a reviewed trajectory. Pass `worker` to name which one when multiple siblings are under review.
- view_tool_call: expand compact-trace handles like "w3m5" into complete, untruncated arguments and results. Viewing is free - use it whenever a summarized line matters to your decision. Handles exist only once a trajectory has been presented for review.
- update_plan: maintain the user-visible plan for the overall task. Workers cannot see or update it; fold their progress into it yourself.
- finalize: ends the run. The named checkpoint's repository state is delivered as the result, and that worker's final message (or the response you provide) becomes the final answer.

Reviews: you review a finished batch at a time - where each sibling forked from, your instructions, a compact trace of each step, a diffstat, and its final message verbatim. Each of your turns allows up to {supervisor_steps} steps, and you also receive an ephemeral <dag> overview of every fragment by id, including discarded ones. You may mix viewing, plan updates, and resolutions across the turn, but a successful spawn_workers or prefinalize ends it. Every reviewed sibling must be resolved before your turn ends: save_checkpoint it, spawn from it (which saves it), or discard it. Discarded trajectories are gone permanently and their handles die with them. Workers inherit the full conversation along their ancestor chain plus your new instructions, and know nothing about sibling workers or your plans - put everything they need into the instructions.

First duty: the task message contains a spec intake - two independent contracts, one written from the task text alone, one grounded in the repository. Diff them, resolve every divergence and flagged ambiguity deliberately - spawn a fact-finding worker when repository evidence is needed - and record the settled reading. Calling conventions and argument order are always ambiguities: resolve them against an analogous implementation already in the repo, never by assumption. For each numbered ambiguity (A1, A2, ...) in the intake, state your resolution and a one-line rationale in your plan before spawning implementation workers. Then have two workers in that same batch write spec tests from the settled reading independently - before implementation exists - including one test per resolution that locks in your chosen reading. Where the two suites disagree about the same behavior, you have found an ambiguity nobody flagged: resolve it explicitly before implementing. Implementation workers run the union of both suites, and your finalize evidence should show it passing. If the intake is missing, have your first worker pin the specification from the task text instead.

Batch composition: default to parallel batches, not a lone worker. Whenever the work has two or more independent parts, spawn one worker per part in the same batch and merge_checkpoint the results; a single-worker turn is justified only when the next step genuinely depends on the last result. The intake's PARALLEL PLAN is your default batch composition: when the intake enumerates separable file groups, spawn one worker per group in the same batch, land any file it flags as shared in a precursor first, and aim the redundant pair (below) at the riskiest area it names; deviate from the plan deliberately, not by drift. Plan the split so the same files are not edited twice and merges stay clean. If independent parts share a file, land the shared file first in a quick precursor worker, then split from that checkpoint. Never spawn multiple workers with identical instructions - duplicates are collapsed, and a stalled approach needs diagnosis, not repetition. Deliberate redundancy is different: for the riskiest core of the contract - the densest edge surface, the most contested ambiguities - spawn two workers in the same batch to implement that same contract independently from different vantages you name (one from the spec's data model outward, one from the test cases inward), sharing your settled ambiguity resolutions with both. Their tests and behavior will diverge precisely on the edges nobody knew were ambiguous and on slips a single worker's self-tests cannot see. When siblings have implemented the same contract, harvest that divergence before resolving the batch: spawn a differential worker from one sibling's checkpoint that checks out the other sibling's test files directly from its commit (git checkout <sha> -- <paths>) and runs them against the implementation it is standing in. Every cross-sibling failure is an ambiguity two readings resolved differently - adjudicate it against the task text, never by majority or by trusting whichever implementation looks cleaner. Fact-finding stays cheap throughout: short workers that answer questions ("find how X is implemented; report file and line"), variants branched from a good checkpoint, losses cut early, instructions explicit and testable.

Finalize is where "Verification" binds you: a worker's report is a claim, not evidence. Do not finalize until the Verification requirements above have actually been discharged on the finalized checkpoint's chain - real test runs whose commands and output you have inspected via view_tool_call. A pre-existing suite that passed before the change proves nothing about the change: the evidence must include tests that exercise the new behavior the task demands - ideally the spec tests written at the start. If that evidence does not exist yet, your prefinalize batch must produce it. A plant is only evidence after re-checking the expected behavior against the spec text or an existing reference implementation - a self-authored test that encodes your own misreading will catch the correct code as the bug. A filtered or single-file test run is progress evidence, not completion evidence: completion evidence is the project's full suite, or a stated reason it cannot be run. Delivery, branches, and commit ceremony are handled outside the run - never spend a worker on commit messages or branch bookkeeping."#,
        supervisor_steps = ASGARD_SUPERVISOR_MAX_STEPS,
        workers = ASGARD_BATCH_CAP,
        worker_steps = ASGARD_WORKER_MAX_STEPS
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
                description: "Fork a barrier batch of new workers from \"root\" or a saved checkpoint id like \"w3\" (or a pending reviewed worker's id, which saves it); each runs up to 10 steps in its own checkout, then the whole batch reports back together. A successful spawn ends your turn immediately.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["workers"],
                    "properties": {
                        "workers": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": ASGARD_BATCH_CAP,
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
                name: PREFINALIZE_TOOL.to_string(),
                description: "Spawn your final verification pass; finalize is locked until this has run and been reviewed. Verification workers should: run the project's full test suite on the checkpoint you intend to deliver; spawn one worker per plant - multi-plant protocols exhaust step budgets; plant one classic bug in the diff's critical paths - swap an argument order, invert a boundary, drop a term - and confirm the suite goes red. Each plant must be a real mutation of the SHIPPED implementation confirmed by inspecting the diff, never a synthetic stand-in function or an edit to test files/mocks. Revert the plant and report any plant the suite failed to catch (a surviving plant means a test must be strengthened before finalizing); re-check the task's obligations against the delivered state. Alongside the plant workers, include one verification worker dedicated to adversarial inputs derived from the spec text alone - empty, zero, single vs many, negative, trailing, an attribute present but valueless - exercised through the public surface; near-misses live in the edges the implementation's own tests never generated. When other checkpoints authored test files missing from the candidate, prefinalize auto-adds one sibling-test verification worker that runs them. Treat each reported failure as a divergence to adjudicate against the task text before finalizing: a defensible test the candidate fails means fix the candidate; a test encoding a wrong or implementation-specific reading is dismissed with a stated reason. Never settle this by majority or by which side wrote more tests. A hung or timed-out verification run is a blocker: identify the exact hanging test before attributing it to anything pre-existing. Never substitute a weaker ad-hoc check. If verification reveals needed work stranded in another checkpoint, use merge_checkpoint to pull it in.".to_string(),
                parameters: spawn_workers_parameters(&allowed_models),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: SAVE_CHECKPOINT_TOOL.to_string(),
                description: "Save a reviewed trajectory as a permanent checkpoint without spawning from it yet. Pass worker like \"w7\" when multiple siblings are pending.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "worker": { "type": "string" },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: MERGE_CHECKPOINT_TOOL.to_string(),
                description: "Merge one saved checkpoint into another saved checkpoint using a real git merge of the full branch, including git's rename detection, producing a new two-parent checkpoint. This is how a split batch's parallel work recombines, and how work stranded in a checkpoint outside your delivered lineage is recovered when finalize warns about it. If git reports conflicts, merge_checkpoint aborts the scratch merge cleanly and lists the conflicted files; then spawn a worker from the onto checkpoint with explicit conflict-resolution instructions.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["from", "onto"],
                    "properties": {
                        "from": { "type": "string" },
                        "onto": { "type": "string" },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: DISCARD_TOOL.to_string(),
                description: "Discard a reviewed trajectory permanently. Every reviewed trajectory must be saved, spawned from (which saves it implicitly), or discarded. Pass worker like \"w7\" when multiple siblings are pending.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "worker": { "type": "string" },
                    },
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
                        "abandoned": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "checkpoint ids you are deliberately leaving out of the delivered lineage",
                        },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: VIEW_TOOL_CALL_TOOL.to_string(),
                description: "Expand compact-trace handles like \"w3m5\" into complete untruncated arguments and results; free of charge; works across every saved trajectory and all siblings in the batch under review.".to_string(),
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

fn spawn_workers_parameters(allowed_models: &[&String]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["workers"],
        "properties": {
            "workers": {
                "type": "array",
                "minItems": 1,
                "maxItems": ASGARD_BATCH_CAP,
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
    })
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
        return Err("workers must contain at least one worker.".to_string());
    }
    if workers.len() > ASGARD_BATCH_CAP {
        return Err(format!(
            "workers must contain at most {ASGARD_BATCH_CAP} workers"
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
        injected_files: Vec::new(),
        max_steps: None,
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
    let abandoned = match arguments.get("abandoned") {
        Some(_) => string_array_property(&arguments, "abandoned")?,
        None => Vec::new(),
    };
    Ok(FinalizeRequest {
        checkpoint,
        response,
        evidence,
        abandoned,
    })
}

pub(crate) fn parse_merge_checkpoint(
    call: &ToolCall,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<MergeCheckpointRequest, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    let from_raw = arguments
        .get("from")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "from must be a string".to_string())?;
    let onto_raw = arguments
        .get("onto")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "onto must be a string".to_string())?;
    let from =
        parse_saved_checkpoint(from_raw, context).map_err(|error| format!("from {error}"))?;
    let onto =
        parse_saved_checkpoint(onto_raw, context).map_err(|error| format!("onto {error}"))?;
    if from == onto {
        return Err("from and onto must be distinct saved checkpoints".to_string());
    }
    Ok(MergeCheckpointRequest { from, onto })
}

pub(crate) fn parse_update_plan(
    call: &ToolCall,
) -> std::result::Result<crate::plan::UpdatePlanArgs, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    serde_json::from_value::<crate::plan::UpdatePlanArgs>(arguments)
        .map_err(|error| format!("Invalid update_plan arguments: {error}"))
}

/// Replace every `view_tool_call` payload in the permanent record with the
/// per-handle summary computed when the call ran. Applies to all expansions
/// regardless of size: the summary carries what the supervisor needs to
/// remember (what it looked at, what shape the answer had, and every error
/// verbatim), and the payload itself can always be re-expanded.
pub(crate) fn elide_view_tool_results_for_permanent_record(
    transcript: &[ChatMessage],
    view_summaries: &std::collections::HashMap<String, String>,
) -> Vec<ChatMessage> {
    transcript
        .iter()
        .map(|message| {
            if message.role == "tool"
                && message.name.as_deref() == Some(VIEW_TOOL_CALL_TOOL)
                && let Some(call_id) = &message.tool_call_id
                && let Some(summary) = view_summaries.get(call_id)
            {
                return ChatMessage::tool_result(call_id, VIEW_TOOL_CALL_TOOL, summary.clone());
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

fn parse_saved_checkpoint(
    value: &str,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<CheckpointId, String> {
    let checkpoint =
        CheckpointId::parse(value).ok_or_else(|| format!("{value:?} is not a checkpoint id"))?;
    match checkpoint {
        CheckpointId::Root => Err("root is not a saved checkpoint".to_string()),
        CheckpointId::Worker(worker) if context.dag.node(worker).is_some() => Ok(checkpoint),
        CheckpointId::Worker(_) => Err(format!("{checkpoint} is not a saved checkpoint")),
    }
}

pub(crate) fn is_pending_checkpoint(
    checkpoint: &CheckpointId,
    context: &SupervisorTurnContext<'_>,
) -> bool {
    matches!(checkpoint, CheckpointId::Worker(worker) if context.pending.contains(worker))
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
            merged_from: Vec::new(),
        })
        .unwrap();
        dag
    }

    fn context<'a>(dag: &'a TrajectoryDag, pending: &'a [usize]) -> SupervisorTurnContext<'a> {
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
            pending: &[4],
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
            pending: &[],
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

        let nine_workers = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": (0..9)
                    .map(|index| serde_json::json!({
                        "from": "root",
                        "instructions": format!("worker {index}")
                    }))
                    .collect::<Vec<_>>()
            }),
        );
        assert!(
            parse_spawn_workers(&nine_workers, &context)
                .expect_err("too many workers")
                .contains("workers must contain at most 8 workers")
        );
    }

    #[test]
    fn finalize_parser_accepts_pending_and_response() {
        let dag = saved_dag();
        let call = supervisor_tool_call(
            "finalize",
            FINALIZE_TOOL,
            serde_json::json!({
                "checkpoint": "w4",
                "response": "done",
                "abandoned": ["w1"]
            }),
        );

        let parsed = parse_finalize(&call, &context(&dag, &[4])).expect("valid finalize");

        assert_eq!(parsed.checkpoint, CheckpointId::Worker(4));
        assert_eq!(parsed.response.as_deref(), Some("done"));
        assert!(parsed.evidence.is_empty());
        assert_eq!(parsed.abandoned, vec!["w1".to_string()]);
    }

    #[test]
    fn merge_checkpoint_parser_requires_distinct_saved_checkpoints() {
        let mut dag = saved_dag();
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 2,
                parent: CheckpointId::Root,
                instructions: "sibling".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("sibling instructions"),
                window_messages: Vec::new(),
                compact: String::new(),
                final_response: "sibling result".to_string(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: "commit-2".to_string(),
            merged_from: Vec::new(),
        })
        .unwrap();
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 3,
                parent: CheckpointId::Worker(1),
                instructions: "child".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("child instructions"),
                window_messages: Vec::new(),
                compact: String::new(),
                final_response: "child result".to_string(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: "commit-3".to_string(),
            merged_from: Vec::new(),
        })
        .unwrap();
        let context = context(&dag, &[4]);

        let call = supervisor_tool_call(
            "merge",
            MERGE_CHECKPOINT_TOOL,
            serde_json::json!({ "from": "w2", "onto": "w1" }),
        );
        let parsed = parse_merge_checkpoint(&call, &context).expect("valid merge");
        assert_eq!(parsed.from, CheckpointId::Worker(2));
        assert_eq!(parsed.onto, CheckpointId::Worker(1));

        let root = supervisor_tool_call(
            "merge",
            MERGE_CHECKPOINT_TOOL,
            serde_json::json!({ "from": "root", "onto": "w1" }),
        );
        assert!(
            parse_merge_checkpoint(&root, &context)
                .expect_err("root rejected")
                .contains("root is not a saved checkpoint")
        );
        let same = supervisor_tool_call(
            "merge",
            MERGE_CHECKPOINT_TOOL,
            serde_json::json!({ "from": "w1", "onto": "w1" }),
        );
        assert!(
            parse_merge_checkpoint(&same, &context)
                .expect_err("same rejected")
                .contains("distinct")
        );
        let ancestor = supervisor_tool_call(
            "merge",
            MERGE_CHECKPOINT_TOOL,
            serde_json::json!({ "from": "w1", "onto": "w3" }),
        );
        let parsed = parse_merge_checkpoint(&ancestor, &context).expect("ancestor merge allowed");
        assert_eq!(parsed.from, CheckpointId::Worker(1));
        assert_eq!(parsed.onto, CheckpointId::Worker(3));
        let pending = supervisor_tool_call(
            "merge",
            MERGE_CHECKPOINT_TOOL,
            serde_json::json!({ "from": "w4", "onto": "w1" }),
        );
        assert!(
            parse_merge_checkpoint(&pending, &context)
                .expect_err("pending rejected")
                .contains("w4 is not a saved checkpoint")
        );
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
    fn permanent_record_replaces_view_results_with_their_summaries() {
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
        let payload = format!("huge payload {}", "x".repeat(500));
        let short_payload = "tiny".to_string();
        let transcript = vec![
            ChatMessage::assistant_tool_calls(vec![view_call, spawn_call, error_view_call]),
            ChatMessage::tool_result("view", VIEW_TOOL_CALL_TOOL, payload.clone()),
            ChatMessage::tool_result("spawn", SPAWN_WORKERS_TOOL, "spawned w3 from root"),
            ChatMessage::tool_result("view-err", VIEW_TOOL_CALL_TOOL, short_payload.clone()),
        ];
        let summaries = std::collections::HashMap::from([
            (
                "view".to_string(),
                "[viewed w1m1: read_file, 512 bytes]\n[viewed w2m3: \"PASS (12) FAIL (0)\"]"
                    .to_string(),
            ),
            (
                "view-err".to_string(),
                "[attempted view of w9: malformed handle]".to_string(),
            ),
        ]);

        let permanent = elide_view_tool_results_for_permanent_record(&transcript, &summaries);
        let text = permanent
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!text.contains(&payload));
        assert!(text.contains("[viewed w1m1: read_file, 512 bytes]"));
        assert!(text.contains(r#"[viewed w2m3: "PASS (12) FAIL (0)"]"#));
        // Errors are summarized too, but the summary quotes them in full so the
        // supervisor sees that a call failed and does not repeat it. Size is
        // irrelevant: even this short payload is replaced.
        assert!(!text.contains(&short_payload));
        assert!(text.contains("[attempted view of w9: malformed handle]"));
        // Non-view tool results are untouched.
        assert!(text.contains("spawned w3 from root"));
    }

    #[test]
    fn supervisor_supplement_requires_numbered_ambiguity_resolutions() {
        let supplement = supervisor_supplement();

        assert!(!supplement.contains("numbered obligations ledger"));
        assert!(supplement.contains("For each numbered ambiguity (A1, A2, ...)"));
        assert!(supplement.contains("state your resolution and a one-line rationale"));
        assert!(supplement.contains("one test per resolution"));
        assert!(supplement.contains("If independent parts share a file"));
        assert!(supplement.contains("when the intake enumerates separable file groups"));
        assert!(supplement.contains("PARALLEL PLAN is your default batch composition"));
        assert!(supplement.contains("two workers in that same batch write spec tests"));
        assert!(
            supplement.contains("Calling conventions and argument order are always ambiguities")
        );
        assert!(supplement.contains("A plant is only evidence after re-checking"));
    }

    #[test]
    fn prefinalize_tool_docs_require_real_single_plants_and_hang_diagnosis() {
        let tools = supervisor_tool_definitions(&["model-a".to_string()]);
        let prefinalize = tools
            .iter()
            .find(|tool| tool.function.name == PREFINALIZE_TOOL)
            .expect("prefinalize tool");
        let description = &prefinalize.function.description;

        assert!(description.contains("spawn one worker per plant"));
        assert!(description.contains("real mutation of the SHIPPED implementation"));
        assert!(description.contains("confirmed by inspecting the diff"));
        assert!(description.contains("hung or timed-out verification run is a blocker"));
        assert!(description.contains("Never substitute a weaker ad-hoc check"));
        assert!(description.contains("sibling-test verification"));
        assert!(description.contains("adjudicate against the task text"));
    }
}
