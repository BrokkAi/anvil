//! Per-turn conversation summarization for context-window management.
//!
//! Mirrors Brokk's approach (`ContextManager.compressHistory(TaskEntry)`
//! / `SummarizerPrompts.compressHistory`): each conversation turn can
//! carry an LLM-generated summary alongside its original log. When a
//! turn has a summary, the LLM sees only the summary on subsequent
//! turns; the original is preserved on disk for replay determinism and
//! for human-readable rendering.
//!
//! This module is the pure-logic + LLM-call layer. The Session state
//! (`ConversationTurn.summary`), persistence (the `summaryContentId`
//! slot in the session zip), and the trigger (`agent.rs` deciding
//! which turns to compress and when) live elsewhere.
//!
//! Two building blocks:
//!
//! 1. [`context_budget`] -- token budget math (75% of the model's
//!    declared context length, with a conservative fallback).
//! 2. [`build_summarization_messages`] + [`run_summarization`] -- the
//!    chat-message scaffolding and async LLM driver that produces one
//!    turn's summary.

use std::time::Duration;

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use crate::llm_client::{ChatMessage, LlmBackend, LlmResponse, StreamChatRequest};
use crate::session::ConversationTurn;

// ---------------------------------------------------------------------------
// Budget math
// ---------------------------------------------------------------------------

/// Fraction of the declared context length we let the prompt occupy.
/// The remainder is headroom for the model's response (which streams
/// into the same window) plus the model's hidden bookkeeping tokens.
/// 75% matches the heuristic the OpenAI cookbook recommends for
/// sustained tool-using sessions.
const BUDGET_FRACTION: f64 = 0.75;

/// Conservative fallback when the backend doesn't publish a context
/// length (Codex, Ollama). Most modern Codex/Ollama models support
/// well over 128k tokens, but undershooting just costs us an extra
/// summarization round -- overshooting would silently drop the user's
/// prompt mid-stream, which is much worse.
pub const FALLBACK_CONTEXT_LENGTH: u32 = 128_000;

/// Token budget for the prompt portion of a request. Returns
/// `BUDGET_FRACTION * context_length`, falling back to
/// `FALLBACK_CONTEXT_LENGTH` when the catalog doesn't expose one.
pub fn context_budget(declared_context_length: Option<u32>) -> usize {
    let raw = declared_context_length.unwrap_or(FALLBACK_CONTEXT_LENGTH) as f64;
    (raw * BUDGET_FRACTION) as usize
}

// ---------------------------------------------------------------------------
// Summarization prompt + LLM call
// ---------------------------------------------------------------------------

/// Tool result text past this length gets truncated in the
/// summarization input. Keeps oversized payloads (large file reads,
/// shell output) from blowing the summarizer's own context, since the
/// summarizer is fed by the same model family that produced them.
const SUMMARIZATION_TOOL_RESULT_CAP: usize = 800;

/// Build the chat-message list that asks the LLM to produce a summary
/// of one conversation turn.
///
/// Mirrors the wording from Brokk's `SummarizerPrompts.compressHistory`:
/// the model is asked for a detailed-but-concise summary written for
/// a third party who hasn't seen the original turn. Bulleted format is
/// requested so the result lands as a compact substitute when fed
/// back into the next turn's prompt.
pub fn build_summarization_messages(turn: &ConversationTurn) -> Vec<ChatMessage> {
    let system = "You are a conversation summarizer for an AI coding assistant. \
Your output replaces a single past turn (user prompt + assistant response + \
any tool calls and results) in the assistant's working context on every \
subsequent turn, so it must preserve the operational facts the assistant \
needs to continue the conversation correctly. It is not meant to read like \
prose.\n\
\n\
Preserve in your summary:\n\
- Specific file paths, function names, line numbers, and code symbols mentioned.\n\
- Decisions made and their rationale (one line each).\n\
- Open TODOs, unresolved errors, and known failure modes.\n\
- Tool calls run and the outcome (success/failure + key output).\n\
- User preferences expressed during the turn.\n\
\n\
Drop:\n\
- Pleasantries and meta-discussion about how to collaborate.\n\
- Exact wording -- compress to bullets.\n\
- Verbose tool output already captured by a short outcome line.\n\
\n\
Output format: a bulleted list wrapped in \
<conversation_summary>...</conversation_summary>. No preamble, no closing \
remarks.";

    let mut body = String::new();
    body.push_str("Turn to summarize:\n\n");
    body.push_str("User: ");
    body.push_str(turn.user_prompt.trim());
    body.push('\n');
    for exchange in &turn.tool_exchanges {
        let result = if exchange.result.chars().count() > SUMMARIZATION_TOOL_RESULT_CAP {
            let truncated: String = exchange
                .result
                .chars()
                .take(SUMMARIZATION_TOOL_RESULT_CAP)
                .collect();
            format!("(truncated) {truncated}")
        } else {
            exchange.result.clone()
        };
        body.push_str(&format!(
            "Tool `{}` args={} -> {}\n",
            exchange.tool_name, exchange.arguments, result
        ));
    }
    if !turn.agent_response.trim().is_empty() {
        body.push_str("Assistant: ");
        body.push_str(turn.agent_response.trim());
        body.push('\n');
    }

    vec![ChatMessage::system(system), ChatMessage::user(body)]
}

/// Strip a single `<conversation_summary>...</conversation_summary>`
/// wrapper if the model produced one. Tolerant of leading/trailing
/// whitespace and of models that omit the closing tag.
pub fn strip_summary_tags(s: &str) -> String {
    let trimmed = s.trim();
    let opened = trimmed
        .strip_prefix("<conversation_summary>")
        .unwrap_or(trimmed)
        .trim();
    let closed = opened
        .strip_suffix("</conversation_summary>")
        .unwrap_or(opened)
        .trim();
    closed.to_string()
}

/// Drive the LLM to produce a summary of one conversation turn.
/// Returns the summary text with any outer tags stripped.
///
/// Errors propagate to the caller. The caller is expected to *leave
/// the turn uncompressed* on failure (matching Brokk's behavior in
/// `ContextManager.compressHistory`) rather than silently dropping
/// the turn -- a missing summary just means the LLM sees the full
/// log on the next turn.
pub async fn run_summarization(
    llm: &dyn LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    idle_timeout: Duration,
    cancel: CancellationToken,
) -> Result<String> {
    let messages = build_summarization_messages(turn);

    // Discard streamed deltas -- only the final text matters here.
    // The streaming machinery in `OpenAiClient` still drives the SSE
    // state machine; we just don't relay it to the user.
    let on_token: Box<dyn FnMut(&str) + Send> = Box::new(|_| {});
    let on_thought: Box<dyn FnMut(&str) + Send> = Box::new(|_| {});

    let response = llm
        .stream_chat(StreamChatRequest {
            model: model.to_string(),
            messages,
            tools: None,
            // Summarization is structured extraction, not deep
            // reasoning -- "low" is plenty and keeps the cost down on
            // reasoning-capable models that bill thinking tokens.
            reasoning_effort: Some("low".to_string()),
            on_token,
            on_thought,
            cancel,
            idle_timeout,
        })
        .await?;

    let text = match response {
        LlmResponse::Text(t) => t,
        // The summarizer isn't given any tools -- if a model fires
        // tool_calls anyway we ignore them and use whatever text it
        // produced alongside (some models emit a single "thinking out
        // loud" text chunk before any calls).
        LlmResponse::ToolCalls { text, .. } => text,
    };
    Ok(strip_summary_tags(&text))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use futures::future::BoxFuture;

    use super::*;
    use crate::session::ToolExchange;

    fn turn(user: &str, agent: &str) -> ConversationTurn {
        ConversationTurn {
            user_prompt: user.to_string(),
            agent_response: agent.to_string(),
            tool_exchanges: Vec::new(),
            summary: None,
            fragment_id: None,
        }
    }

    fn turn_with_tool(
        user: &str,
        agent: &str,
        tool: &str,
        args: &str,
        result: &str,
    ) -> ConversationTurn {
        ConversationTurn {
            user_prompt: user.to_string(),
            agent_response: agent.to_string(),
            tool_exchanges: vec![ToolExchange {
                call_id: "call_1".to_string(),
                tool_name: tool.to_string(),
                arguments: args.to_string(),
                result: result.to_string(),
            }],
            summary: None,
            fragment_id: None,
        }
    }

    #[test]
    fn context_budget_uses_fraction_of_declared_length() {
        assert_eq!(context_budget(Some(200_000)), 150_000);
        assert_eq!(context_budget(Some(128_000)), 96_000);
    }

    #[test]
    fn context_budget_falls_back_when_unknown() {
        assert_eq!(context_budget(None), 96_000);
    }

    #[test]
    fn build_summarization_messages_includes_user_assistant_and_tools() {
        let t = turn_with_tool("find TODOs", "found 3", "shell", r#"{"cmd":"rg"}"#, "out");
        let msgs = build_summarization_messages(&t);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        let body = msgs[1].content.as_deref().unwrap();
        assert!(body.contains("User: find TODOs"));
        assert!(body.contains("Tool `shell`"));
        assert!(body.contains("Assistant: found 3"));
    }

    /// Oversized tool results must be truncated in the summarizer
    /// input so the summarizer's own context doesn't overflow.
    #[test]
    fn build_summarization_messages_truncates_huge_tool_results() {
        let huge = "z".repeat(SUMMARIZATION_TOOL_RESULT_CAP * 4);
        let t = turn_with_tool("u", "a", "shell", "{}", &huge);
        let msgs = build_summarization_messages(&t);
        let body = msgs[1].content.as_deref().unwrap();
        assert!(body.contains("(truncated)"));
        assert!(!body.contains(&huge));
    }

    #[test]
    fn strip_summary_tags_unwraps_paired_tags() {
        let s = "<conversation_summary>\n- a\n- b\n</conversation_summary>";
        assert_eq!(strip_summary_tags(s), "- a\n- b");
    }

    #[test]
    fn strip_summary_tags_passes_through_unwrapped() {
        assert_eq!(strip_summary_tags("- a\n- b"), "- a\n- b");
    }

    #[test]
    fn strip_summary_tags_tolerates_missing_close() {
        let s = "<conversation_summary>\n- a\n- b\n";
        assert_eq!(strip_summary_tags(s), "- a\n- b");
    }

    /// Mock backend that hands back a canned response so we can
    /// verify `run_summarization` correctly stitches in the messages,
    /// reasoning effort, and tag-stripping.
    struct CannedBackend {
        last_messages: Arc<Mutex<Option<Vec<ChatMessage>>>>,
        last_reasoning_effort: Arc<Mutex<Option<String>>>,
        response: LlmResponse,
    }

    impl LlmBackend for CannedBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            async { Ok(vec!["mock".to_string()]) }.boxed()
        }

        fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            *self.last_messages.lock().unwrap() = Some(request.messages);
            *self.last_reasoning_effort.lock().unwrap() = request.reasoning_effort;
            let response = match &self.response {
                LlmResponse::Text(t) => LlmResponse::Text(t.clone()),
                LlmResponse::ToolCalls { text, calls } => LlmResponse::ToolCalls {
                    text: text.clone(),
                    calls: calls.clone(),
                },
            };
            async move { Ok(response) }.boxed()
        }
    }

    #[tokio::test]
    async fn run_summarization_sends_messages_and_strips_tags() {
        let last_messages = Arc::new(Mutex::new(None));
        let last_reasoning_effort = Arc::new(Mutex::new(None));
        let backend = CannedBackend {
            last_messages: last_messages.clone(),
            last_reasoning_effort: last_reasoning_effort.clone(),
            response: LlmResponse::Text(
                "<conversation_summary>\n- key bullet\n</conversation_summary>".to_string(),
            ),
        };
        let t = turn("hello", "hi");

        let out = run_summarization(
            &backend,
            "mock-model",
            &t,
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await
        .expect("summarization succeeds");
        assert_eq!(out, "- key bullet");

        let sent = last_messages.lock().unwrap().take().expect("called");
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[0].role, "system");
        assert_eq!(sent[1].role, "user");

        assert_eq!(
            last_reasoning_effort.lock().unwrap().as_deref(),
            Some("low")
        );
    }
}
