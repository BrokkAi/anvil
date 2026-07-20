use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionNotification, SessionUpdate, TextContent,
};
use agent_client_protocol::{Client, ConnectionTo};
use tokio_util::sync::CancellationToken;

use crate::asgard::{AsgardLiveOutput, extract_worker_final_response};
use crate::llm_client::{
    ChatMessage, IdleTimeouts, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage,
};
use crate::session::SessionStore;
use crate::structured_output::StructuredOutputRequest;
use crate::tool_loop::{LoopOutcome, NotificationMode, SpawnedCx, TextSink};

pub(crate) const ASGARD_INTAKE_MAX_STEPS: usize = 8;

pub(crate) const ASGARD_INTAKE_READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "list_directory",
    "grep_search",
    "search_symbols",
    "get_symbol_sources",
    "get_summaries",
    "scan_usages_by_reference",
];

const READER_L_PROMPT: &str = "You are a specification reader. From the task text alone - you have no repository access - write the behavioral contract a correct implementation must satisfy: every named symbol and API with its exact spelling; every enumerated set with all its members; every scoping qualifier (which commands, modes, or inputs a rule names - and where a rule is stated for one member of a set, flag whether it plausibly extends to the siblings); every exact error message or exception; every input domain (do \"numbers\" include decimals? negatives?); every ordering and formatting rule. Quote the task phrase each item derives from. End with a numbered list titled 'AMBIGUITIES', one line each ('A1:', 'A2:', ...), covering every detail that admits more than one reading, and every boundary a quantifier or edge word creates ('empty', 'each', 'all', 'zero', 'trailing', 'without a value', 'at least') - for each, state what happens on both sides of the boundary. Output the numbered contract followed by the AMBIGUITIES list and nothing else.";

const READER_G_PROMPT: &str = "You are a repository scout preparing for a task another team will implement. For each requirement in the task, find the repository evidence that constrains its interpretation: existing naming conventions, sibling APIs and how they behave, golden and fixture files, project layout. Wherever the task quantifies over a set (\"all dialects\", \"each command\"), enumerate the set's actual members from the repository and report the count and every member with its path. Report exact file paths for all evidence. Do not write code or tests. Your final message is the deliverable: a numbered report of constraints and enumerations, ending with a section titled 'PARALLEL PLAN': the separable groups of files an implementation would touch (one line per group, with paths), any file two groups would both edit (flag it: 'shared, land first'), and one line naming the riskiest contract area - the densest edge surface or the requirement with the least repository evidence. You have a hard budget of a few tool steps; your final text report is the only thing that survives this session, so stop exploring early enough to write it. An incomplete report beats no report.";

pub(crate) struct IntakeContracts {
    pub(crate) literal: Option<String>,
    pub(crate) grounded: Option<String>,
}

pub(crate) struct AsgardIntakeRun<'a> {
    pub(crate) cx: &'a ConnectionTo<Client>,
    pub(crate) sessions: &'a SessionStore,
    pub(crate) session_id: &'a str,
    pub(crate) llm: &'a Arc<dyn LlmBackend>,
    pub(crate) parent_cwd: &'a std::path::Path,
    pub(crate) config: &'a crate::asgard::Config,
    pub(crate) selected_model: &'a str,
    pub(crate) reasoning_effort: Option<&'a str>,
    pub(crate) service_tier: Option<&'a str>,
    pub(crate) structured_output: Option<&'a StructuredOutputRequest>,
    pub(crate) idle_timeout: IdleTimeouts,
    pub(crate) cancel: CancellationToken,
    pub(crate) live_output: &'a AsgardLiveOutput,
    pub(crate) original_task: &'a str,
    pub(crate) context_length: Option<u32>,
    pub(crate) context_prefix_len: usize,
}

pub(crate) async fn run_asgard_intake(
    run: AsgardIntakeRun<'_>,
) -> (IntakeContracts, Vec<(String, TokenUsage)>) {
    send_intake_thought(
        run.live_output,
        "Asgard intake: two readers examining the task.\n",
    );
    if run.cancel.is_cancelled() {
        return (
            IntakeContracts {
                literal: None,
                grounded: None,
            },
            Vec::new(),
        );
    }

    let started = Instant::now();
    let supervisor_model = run
        .config
        .supervisor_model
        .as_deref()
        .unwrap_or(run.selected_model);
    let grounded_model = run.config.candidate_models[0].as_str();
    let literal = read_literal_contract(
        run.llm,
        supervisor_model,
        run.idle_timeout,
        run.cancel.clone(),
        run.original_task,
    );
    let grounded = read_grounded_contract(&run, grounded_model);
    let (literal, grounded) = tokio::join!(literal, grounded);
    let elapsed_millis = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;

    let mut usages = Vec::new();
    let (literal_text, literal_usage) = match literal {
        Some((text, usage)) => {
            usages.push((supervisor_model.to_string(), usage));
            (Some(text), usage)
        }
        None => (None, TokenUsage::default()),
    };
    let (grounded_text, grounded_usage) = match grounded {
        Some((text, usage)) => {
            usages.push((grounded_model.to_string(), usage));
            (Some(text), usage)
        }
        None => (None, TokenUsage::default()),
    };

    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_intake",
        "literal": literal_text,
        "grounded": grounded_text,
        "literal_model": supervisor_model,
        "grounded_model": grounded_model,
        "elapsed_millis": elapsed_millis,
        "literal_usage": usage_json(literal_usage),
        "grounded_usage": usage_json(grounded_usage),
    }));

    (
        IntakeContracts {
            literal: literal_text,
            grounded: grounded_text,
        },
        usages,
    )
}

pub(crate) async fn intake_read_only_allowlist(
    registry: &Arc<crate::tools::ToolRegistry>,
) -> HashSet<String> {
    let read_only = ASGARD_INTAKE_READ_ONLY_TOOLS
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    registry
        .tool_definitions()
        .await
        .into_iter()
        .map(|tool| tool.function.name)
        .filter(|name| read_only.contains(name.as_str()))
        .collect()
}

async fn read_literal_contract(
    llm: &Arc<dyn LlmBackend>,
    model: &str,
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
    original_task: &str,
) -> Option<(String, TokenUsage)> {
    if cancel.is_cancelled() {
        return None;
    }
    let messages = vec![
        ChatMessage::system(READER_L_PROMPT),
        ChatMessage::user(original_task),
    ];
    let result = crate::llm_client::stream_chat_no_visible_output_with_retry(
        llm.as_ref(),
        "asgard_intake_literal",
        &cancel,
        || StreamChatRequest {
            model: model.to_string(),
            messages: messages.clone(),
            tools: None,
            reasoning_effort: None,
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_: &str| {}),
            on_thought: Box::new(|_: &str| {}),
            cancel: cancel.clone(),
            idle_timeouts: idle_timeout,
        },
    )
    .await;
    match result {
        Ok(response) if !cancel.is_cancelled() => response_text_and_usage(response)
            .filter(|(text, _)| !text.trim().is_empty())
            .or_else(|| {
                tracing::warn!("Asgard literal intake returned empty text");
                None
            }),
        Ok(_) => None,
        Err(error) => {
            tracing::warn!("Asgard literal intake failed: {error:#}");
            None
        }
    }
}

async fn read_grounded_contract(
    run: &AsgardIntakeRun<'_>,
    model: &str,
) -> Option<(String, TokenUsage)> {
    if run.cancel.is_cancelled() {
        return None;
    }
    let Some(registry) = run
        .sessions
        .create_trajectory_registry(run.session_id, run.parent_cwd.to_path_buf())
        .await
    else {
        tracing::warn!("failed to create Asgard grounded intake registry: unknown parent session");
        return None;
    };
    let tool_allowlist = Arc::new(intake_read_only_allowlist(&registry).await);
    let noop_text: TextSink = Arc::new(Mutex::new(|_: &str| {}));
    let noop_thought: TextSink = Arc::new(Mutex::new(|_: &str| {}));
    let messages = vec![
        ChatMessage::system(READER_G_PROMPT),
        ChatMessage::user(run.original_task),
    ];
    let outcome = crate::tool_loop::run(
        run.llm,
        &registry,
        model,
        run.reasoning_effort,
        run.service_tier,
        run.structured_output,
        messages,
        ASGARD_INTAKE_MAX_STEPS,
        run.idle_timeout,
        run.cancel.clone(),
        noop_text,
        noop_thought,
        SpawnedCx::new(run.cx),
        run.session_id.to_string(),
        run.sessions.clone(),
        run.original_task.to_string(),
        NotificationMode::Silent,
        0,
        Some(tool_allowlist),
        None,
        true,
        None,
        run.context_length,
        run.context_prefix_len,
        None,
    )
    .await;
    grounded_text_and_usage(run, model, outcome).await
}

async fn grounded_text_and_usage(
    run: &AsgardIntakeRun<'_>,
    model: &str,
    outcome: LoopOutcome,
) -> Option<(String, TokenUsage)> {
    let text = extract_worker_final_response(&outcome.continuation_messages);
    if text.trim().is_empty() {
        tracing::warn!("Asgard grounded intake returned empty text");
        return grounded_forced_report(run, model, outcome).await;
    }
    Some((text, outcome.usage))
}

async fn grounded_forced_report(
    run: &AsgardIntakeRun<'_>,
    model: &str,
    outcome: LoopOutcome,
) -> Option<(String, TokenUsage)> {
    if run.cancel.is_cancelled() {
        return None;
    }
    let mut messages = outcome.continuation_messages;
    messages.push(ChatMessage::user(
        "Your step budget is exhausted. Write your numbered report now from what you have already seen. Do not call any tools.",
    ));
    let result = crate::llm_client::stream_chat_no_visible_output_with_retry(
        run.llm.as_ref(),
        "asgard_intake_grounded_forced_report",
        &run.cancel,
        || StreamChatRequest {
            model: model.to_string(),
            messages: messages.clone(),
            tools: None,
            reasoning_effort: run.reasoning_effort.map(str::to_string),
            service_tier: run.service_tier.map(str::to_string),
            temperature: None,
            structured_output: run.structured_output.cloned(),
            on_token: Box::new(|_: &str| {}),
            on_thought: Box::new(|_: &str| {}),
            cancel: run.cancel.clone(),
            idle_timeouts: run.idle_timeout,
        },
    )
    .await;
    match result {
        Ok(response) if !run.cancel.is_cancelled() => {
            let (text, fallback_usage) = response_text_and_usage(response)?;
            if text.trim().is_empty() {
                tracing::warn!("Asgard grounded intake forced report returned empty text");
                return None;
            }
            let mut usage = outcome.usage;
            usage.add(fallback_usage);
            Some((text, usage))
        }
        Ok(_) => None,
        Err(error) => {
            tracing::warn!("Asgard grounded intake forced report failed: {error:#}");
            None
        }
    }
}

fn response_text_and_usage(response: LlmResponse) -> Option<(String, TokenUsage)> {
    match response {
        LlmResponse::Text { text, usage, .. } | LlmResponse::ToolCalls { text, usage, .. } => {
            Some((text, usage))
        }
    }
}

fn send_intake_thought(output: &AsgardLiveOutput, text: &str) {
    let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    let update = SessionUpdate::AgentThoughtChunk(chunk);
    let notification = SessionNotification::new(output.session_id.clone(), update);
    if let Err(e) = output.cx.send_notification(notification) {
        tracing::warn!("failed to send thought session update: {e}");
    }
}

fn usage_json(usage: TokenUsage) -> serde_json::Value {
    serde_json::json!({
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "thought": usage.thought_tokens,
        "cachedRead": usage.cached_read_tokens,
        "cachedWrite": usage.cached_write_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_reader_prompt_requires_numbered_ambiguities() {
        assert!(READER_L_PROMPT.contains("numbered list titled 'AMBIGUITIES'"));
        assert!(READER_L_PROMPT.contains("one line each ('A1:', 'A2:', ...)"));
        assert!(
            READER_L_PROMPT
                .contains("Output the numbered contract followed by the AMBIGUITIES list")
        );
    }
}
