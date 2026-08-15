//! Transport-neutral prompt-turn orchestration.
//!
//! One model turn = run the tool loop, account usage, validate structured
//! output, render the optional turn recap, and persist the conversation
//! turn. Both the ACP adapter (`acp::run_model_turn_in_spawn`) and the HTTP
//! runs API drive this single pipeline so turn semantics cannot drift
//! between transports (#316/#318). Transport specifics stay with the
//! caller: ACP wraps its connection in text sinks / `SpawnedCx` and sends
//! the post-turn `session/usage_update`; HTTP records SSE events and a run
//! resource instead.

use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use std::panic::AssertUnwindSafe;

use crate::llm_client::{ChatMessage, IdleTimeouts, TokenUsage};
use crate::runtime::{EventSink, PermissionBroker};
use crate::session::{ConversationTurn, SessionStore, sanitize_replay_events};
use crate::structured_output::{
    StructuredOutputRequest, StructuredOutputResult, validate_response,
};
use crate::tool_loop::{LoopOutcome, LoopStop, NotificationMode, TextSink, TurnFailure};
use crate::workspace_delta::{WorkspaceDeltaTracker, workspace_delta_for_turn};

const RECAP_SUMMARY_MIN_CHARS: usize = 280;

/// Everything one prompt turn needs. Grouped as a struct (rather than ~20
/// positional arguments) because two transports construct it.
pub(crate) struct PromptTurnRequest<'a> {
    pub sessions: &'a SessionStore,
    pub session_id: &'a str,
    /// Workspace root used for the recap's workspace delta and usage cost
    /// lookups; the session cwd in practice.
    pub fallback_cwd: &'a Path,
    pub llm: &'a Arc<dyn crate::llm_client::LlmBackend>,
    pub registry: &'a Arc<crate::tools::ToolRegistry>,
    pub model: &'a str,
    pub reasoning_effort: Option<&'a str>,
    pub service_tier: Option<&'a str>,
    pub structured_output_request: Option<&'a StructuredOutputRequest>,
    pub messages: Vec<ChatMessage>,
    /// Usage already spent preparing this turn (history compaction).
    pub initial_usage: TokenUsage,
    pub context_length: Option<u32>,
    pub context_prefix_len: usize,
    pub initial_plan: Option<crate::plan::UpdatePlanArgs>,
    pub max_turns: usize,
    pub idle_timeout: IdleTimeouts,
    pub cancel: tokio_util::sync::CancellationToken,
    pub turn_recap_enabled: bool,
    pub prompt_text_for_turn: String,
    /// Streaming assistant text; also receives the rendered turn recap and
    /// the persistence-failure warning, exactly like the ACP message stream.
    pub text_sink: TextSink,
    pub thought_sink: TextSink,
    pub event_sink: &'a dyn EventSink,
    pub permission_broker: &'a dyn PermissionBroker,
}

pub(crate) struct PromptTurnOutcome {
    pub structured_output: Option<StructuredOutputResult>,
    /// Session-cumulative usage after recording this turn.
    pub cumulative_usage: TokenUsage,
    /// This turn's usage alone (compaction included). Consumed by the
    /// HTTP runs API for run resources and terminal events; the ACP
    /// adapter reports usage through its own session-cumulative update.
    #[cfg(feature = "http-api")]
    pub turn_usage: TokenUsage,
    /// This turn's usage split by the model that actually spent it. The
    /// harness routes its own internal work (permission classification, the
    /// semantic reranker, subagents) to a utility model, so a single turn can
    /// bill two models. Empty when the turn spent nothing.
    pub usage_by_model: std::collections::BTreeMap<String, TokenUsage>,
    pub response: String,
    pub stop: LoopStop,
    pub tool_stats: crate::host_notice::ToolCallStats,
    pub persisted_fragment_id: Option<String>,
}

impl PromptTurnOutcome {
    /// The turn failure to surface in usage reports, when the loop failed.
    pub fn turn_failure(&self) -> Option<&TurnFailure> {
        match &self.stop {
            LoopStop::Failed(failure) => Some(failure),
            _ => None,
        }
    }
}

fn emit_text(sink: &TextSink, text: &str) {
    if let Ok(mut f) = sink.lock() {
        f(text);
    }
}

/// Run one complete prompt turn and persist it. Panics inside the tool loop
/// are converted into a non-retryable failed turn rather than unwinding into
/// the caller's dispatch loop.
pub(crate) async fn run_prompt_turn(request: PromptTurnRequest<'_>) -> PromptTurnOutcome {
    let PromptTurnRequest {
        sessions,
        session_id,
        fallback_cwd,
        llm,
        registry,
        model,
        reasoning_effort,
        service_tier,
        structured_output_request,
        mut messages,
        initial_usage,
        context_length,
        context_prefix_len,
        initial_plan,
        max_turns,
        idle_timeout,
        cancel,
        turn_recap_enabled,
        prompt_text_for_turn,
        text_sink,
        thought_sink,
        event_sink,
        permission_broker,
    } = request;

    if let Some(instructions) = registry.mcp_instructions().await {
        append_mcp_instructions_to_system_prompt(&mut messages, &instructions);
    }

    let cancel_status = cancel.clone();
    let workspace_delta_tracker = if turn_recap_enabled {
        Some(WorkspaceDeltaTracker::snapshot(fallback_cwd).await)
    } else {
        None
    };

    let recap_text_sink = text_sink.clone();
    let loop_future = crate::tool_loop::run(
        llm,
        registry,
        model,
        reasoning_effort,
        service_tier,
        structured_output_request,
        messages,
        max_turns,
        idle_timeout,
        cancel,
        text_sink,
        thought_sink,
        event_sink,
        permission_broker,
        session_id.to_string(),
        sessions.clone(),
        prompt_text_for_turn.clone(),
        NotificationMode::Live,
        0,
        None,
        None,
        false,
        None,
        None,
        context_length,
        context_prefix_len,
        initial_plan,
    );
    let mut outcome = match AssertUnwindSafe(loop_future).catch_unwind().await {
        Ok(outcome) => outcome,
        Err(panic) => {
            tracing::error!(session_id = %session_id, "tool loop panicked: {:?}", panic);
            // A panic is treated as fatal (non-retryable): retrying a
            // deterministic crash would just spin, so an autonomous driver
            // should stop and surface it rather than back off and retry.
            LoopOutcome {
                response: "Error: agent loop panicked. See server logs.".to_string(),
                tool_exchanges: Vec::new(),
                replay_events: Vec::new(),
                usage: TokenUsage::default(),
                usage_by_model: std::collections::BTreeMap::new(),
                stop: LoopStop::Failed(TurnFailure {
                    retryable: false,
                    message: "agent loop panicked".to_string(),
                }),
                current_plan: None,
                compaction_checkpoint: None,
            }
        }
    };
    outcome.usage.add(initial_usage);
    // Preparing the turn (history compaction) runs on the session model, so
    // its tokens belong to that model's line in the breakdown.
    if !outcome.usage_by_model.is_empty() && !initial_usage.is_zero() {
        outcome
            .usage_by_model
            .entry(model.to_string())
            .or_default()
            .add(initial_usage);
    }
    let LoopOutcome {
        response: response_text,
        tool_exchanges,
        replay_events,
        usage: turn_usage,
        usage_by_model,
        stop,
        current_plan,
        compaction_checkpoint,
    } = outcome;

    let model_metadata = sessions.available_model_metadata().await;
    let model_meta = model_metadata.iter().find(|meta| meta.id == model);
    // An empty breakdown means the loop never attributed usage (a panic, or a
    // turn that spent nothing), so fall back to pricing the whole turn on the
    // session model.
    let cost_delta_usd = if usage_by_model.is_empty() {
        model_meta.and_then(|meta| meta.estimate_cost_usd(turn_usage))
    } else {
        crate::usage_report::estimate_usage_by_model_cost(&model_metadata, &usage_by_model)
    };
    let recap_context_length = model_meta.and_then(|meta| meta.context_length);
    let cumulative_usage = sessions
        .record_usage(session_id, turn_usage, cost_delta_usd)
        .await
        .unwrap_or(turn_usage);
    let structured_output_result =
        structured_output_request.map(|request| validate_response(request, &response_text));

    // Build the turn up front so the recap summarizer can read it without
    // cloning the tool exchanges. `agent_response` holds the raw model text for
    // now (the summarizer strips host notices itself) and is replaced with the
    // recap-augmented text below before the turn is persisted.
    let mut turn = ConversationTurn {
        user_prompt: prompt_text_for_turn,
        agent_response: response_text.clone(),
        replay_events: sanitize_replay_events(&replay_events),
        tool_exchanges,
        structured_output: structured_output_result.clone(),
        summary: None,
        current_plan,
        compaction_checkpoint,
        fragment_id: None,
    };

    let workspace_delta = if let Some(tracker) = workspace_delta_tracker {
        workspace_delta_for_turn(fallback_cwd, tracker).await
    } else {
        crate::host_notice::WorkspaceDelta::default()
    };
    let tool_stats = crate::host_notice::ToolCallStats::from_exchanges(&turn.tool_exchanges)
        .with_workspace_delta(&workspace_delta);
    let visible_response =
        if turn_recap_enabled && !cancel_status.is_cancelled() && tool_stats.has_changed_files() {
            let summary = recap_work_summary(
                llm.as_ref(),
                model,
                &turn,
                recap_context_length,
                idle_timeout,
                cancel_status.clone(),
            )
            .await;
            let recap = crate::host_notice::render_turn_recap(
                summary.as_deref(),
                &turn.tool_exchanges,
                Some(&workspace_delta),
                &stop,
            );
            emit_text(&recap_text_sink, &recap);
            format!("{response_text}{recap}")
        } else {
            response_text.clone()
        };
    turn.agent_response = visible_response;

    let persisted_fragment_id = match sessions.add_turn(session_id, turn).await {
        Ok(fragment_id) => fragment_id,
        Err(e) => {
            emit_text(
                &recap_text_sink,
                &format!(
                    "\n**Warning:** failed to save this conversation turn to disk; \
                     it will not survive a session reload: {e}\n"
                ),
            );
            None
        }
    };

    PromptTurnOutcome {
        structured_output: structured_output_result,
        cumulative_usage,
        #[cfg(feature = "http-api")]
        turn_usage,
        usage_by_model,
        response: response_text,
        stop,
        tool_stats,
        persisted_fragment_id,
    }
}

pub(crate) fn append_mcp_instructions_to_system_prompt(
    messages: &mut [ChatMessage],
    instructions: &str,
) {
    let Some(system) = messages.iter_mut().find(|message| message.role == "system") else {
        return;
    };
    let Some(crate::llm_client::ChatContentPart::Text { text }) = system.content.first_mut() else {
        return;
    };
    text.push_str("\n\n");
    text.push_str(instructions);
}

/// Best-effort, user-facing summary of the work done this turn, for the
/// recap. Returns `None` when there is nothing worth summarizing or the
/// summarizer call fails or is cancelled; the recap then renders only its
/// deterministic stat lines. Failures are intentionally swallowed -- a recap
/// is a convenience, never a reason to fail the turn.
async fn recap_work_summary(
    llm: &dyn crate::llm_client::LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    context_length: Option<u32>,
    idle_timeout: IdleTimeouts,
    cancel: tokio_util::sync::CancellationToken,
) -> Option<String> {
    let visible = crate::host_notice::model_visible_assistant_text(&turn.agent_response);
    if turn.tool_exchanges.is_empty() && visible.trim().chars().count() < RECAP_SUMMARY_MIN_CHARS {
        return None;
    }
    match crate::context_manager::summarize_turn_for_recap(
        llm,
        model,
        turn,
        context_length,
        idle_timeout,
        cancel,
    )
    .await
    {
        Ok(summary) if !summary.trim().is_empty() => Some(summary),
        Ok(_) => None,
        Err(e) => {
            tracing::debug!(model, "turn recap summary failed: {e:#}");
            None
        }
    }
}
