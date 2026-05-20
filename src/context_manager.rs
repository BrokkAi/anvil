//! Context-window management: decides how to fit a long conversation
//! into the model's context window without losing operational state.
//!
//! Three building blocks, exposed independently so each can be tested
//! and reviewed in isolation:
//!
//! 1. **Pivot planning** ([`plan_summarization_pivot`],
//!    [`sliding_window_pivot`]) -- pure functions that decide *where*
//!    the boundary between "summarized" and "verbatim" history should
//!    move to fit a token budget.
//! 2. **Summarization prompt** ([`build_summarization_messages`]) --
//!    constructs the chat message list to send the model when asking
//!    it to compress old turns into a summary block.
//! 3. **LLM call** ([`run_summarization`]) -- async wrapper that
//!    drives the LLM, strips outer tags, and surfaces just the
//!    resulting summary text.
//!
//! Nothing here mutates session state or builds the user-facing prompt;
//! integration with `agent.rs` and `session.rs` happens in the later
//! phases described in `PLANS.md`. The crate is wired in but unused for
//! now, gated under `#[allow(dead_code)]` so the unused-warnings stay
//! out of CI until Phase 4 lights it up.

#![allow(dead_code)]

use std::time::Duration;

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use crate::llm_client::{ChatMessage, LlmBackend, LlmResponse, StreamChatRequest};
use crate::session::ConversationTurn;
use crate::tokens::approximate_tokens;

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

/// Compression strategy chosen per session via `/setup advanced`.
///
/// `Hybrid` is the intended default once Phase 5 wires the setting in;
/// `None` exists for opt-out (and to keep early phases of the rollout
/// non-disruptive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextStrategy {
    /// No compression: send the full history every turn. Cheapest in
    /// CPU/latency but blows the context window once the session is
    /// long enough.
    None,
    /// Sliding window: drop oldest turns until the projected token
    /// count is under budget. Lossy but bounded; no extra LLM call.
    SlidingWindow,
    /// Hybrid (default): summarize older turns into a
    /// `<conversation_summary>` block, advance the summary pivot, and
    /// keep the rest of the history verbatim. Falls back to
    /// `SlidingWindow` on summarization failure so the user's prompt
    /// is never blocked on a meta-LLM call.
    #[default]
    Hybrid,
}

impl ContextStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SlidingWindow => "sliding_window",
            Self::Hybrid => "hybrid",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "sliding_window" => Some(Self::SlidingWindow),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Budget math
// ---------------------------------------------------------------------------

/// Fraction of the declared context length we let the prompt occupy.
/// The remainder is headroom for the model's response (which streams
/// into the same window) plus the model's hidden bookkeeping tokens.
/// 75% is the same default Brokk uses on the Java side and matches the
/// "leave a quarter for output" heuristic the OpenAI cookbook
/// recommends for sustained tool-using sessions.
const BUDGET_FRACTION: f64 = 0.75;

/// Conservative fallback when the backend doesn't publish a context
/// length (Codex, Ollama). Most modern Codex/Ollama models support
/// well over 128k tokens, but undershooting just costs us an extra
/// compression round -- overshooting would silently drop the user's
/// prompt mid-stream, which is much worse.
pub const FALLBACK_CONTEXT_LENGTH: u32 = 128_000;

/// Token budget for the prompt portion of a request. Returns
/// `BUDGET_FRACTION * context_length`, falling back to
/// `FALLBACK_CONTEXT_LENGTH` when the catalog doesn't expose one.
pub fn context_budget(declared_context_length: Option<u32>) -> usize {
    let raw = declared_context_length.unwrap_or(FALLBACK_CONTEXT_LENGTH) as f64;
    (raw * BUDGET_FRACTION) as usize
}

/// Tokens contributed by a single conversation turn when replayed
/// verbatim into the prompt. Includes the user prompt, assistant
/// response, and every tool exchange (round-tripped on each replay
/// via `build_prompt_messages`).
fn turn_tokens(turn: &ConversationTurn) -> usize {
    let mut sum = approximate_tokens(&turn.user_prompt);
    sum += approximate_tokens(&turn.agent_response);
    for exchange in &turn.tool_exchanges {
        sum += approximate_tokens(&exchange.tool_name);
        sum += approximate_tokens(&exchange.arguments);
        sum += approximate_tokens(&exchange.result);
    }
    sum
}

/// Rough estimate of how many tokens one absorbed turn occupies in the
/// summary block. The real number depends on the summarizer LLM's
/// verbosity; this is intentionally generous so the planner doesn't
/// declare victory prematurely and then overshoot the budget. If the
/// realized summary turns out smaller, the next round simply needs to
/// absorb fewer turns.
const SUMMARY_TOKENS_PER_ABSORBED_TURN: usize = 80;

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Outcome of planning a Hybrid compression round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionPlan {
    /// New summary pivot. `history[..new_pivot]` should be absorbed
    /// into the summary; `history[new_pivot..]` should be replayed
    /// verbatim. Always `>= current_pivot`.
    pub new_pivot: usize,
    /// Rough estimate of the resulting summary's size in tokens. Used
    /// by callers (and tests) to verify the budget will fit.
    pub expected_summary_tokens: usize,
    /// True when even absorbing the entire history isn't enough --
    /// callers should fall back to sliding-window on the remaining
    /// surface (e.g. truncate the AGENTS.md block, drop the skills
    /// catalog, etc.) or accept budget overrun.
    pub exhausted_history: bool,
}

/// Decide where to move the summary pivot to get the projected prompt
/// under `budget`. Pure -- does not call the LLM.
///
/// Returns `None` when the prompt already fits within budget. Returns
/// `Some(plan)` otherwise, where `plan.exhausted_history == true`
/// means even absorbing every turn won't shrink things enough (the
/// system prompt + AGENTS.md + skills catalog + the user's new prompt
/// together exceed the budget, and the caller must fall back to
/// dropping non-history surface).
pub fn plan_summarization_pivot(
    history: &[ConversationTurn],
    current_pivot: usize,
    current_summary_tokens: usize,
    overhead_tokens: usize,
    budget: usize,
) -> Option<CompressionPlan> {
    let current_pivot = current_pivot.min(history.len());
    let mut remaining: usize = history[current_pivot..].iter().map(turn_tokens).sum();

    let projected_initial = overhead_tokens
        .saturating_add(current_summary_tokens)
        .saturating_add(remaining);
    if projected_initial <= budget {
        return None;
    }

    let mut new_pivot = current_pivot;
    let mut absorbed: usize = 0;
    loop {
        let expected_summary_tokens =
            current_summary_tokens + absorbed * SUMMARY_TOKENS_PER_ABSORBED_TURN;
        let projected = overhead_tokens
            .saturating_add(expected_summary_tokens)
            .saturating_add(remaining);
        if projected <= budget {
            return Some(CompressionPlan {
                new_pivot,
                expected_summary_tokens,
                exhausted_history: false,
            });
        }
        if new_pivot >= history.len() {
            return Some(CompressionPlan {
                new_pivot: history.len(),
                expected_summary_tokens,
                exhausted_history: true,
            });
        }
        remaining = remaining.saturating_sub(turn_tokens(&history[new_pivot]));
        new_pivot += 1;
        absorbed += 1;
    }
}

/// Sliding-window fallback: advance the pivot (effectively dropping
/// turns from the prompt -- *not* from disk) until the projected
/// prompt fits the budget. Used as a fail-safe when summarization
/// fails or isn't enabled. Pure -- no LLM call.
pub fn sliding_window_pivot(
    history: &[ConversationTurn],
    current_pivot: usize,
    current_summary_tokens: usize,
    overhead_tokens: usize,
    budget: usize,
) -> usize {
    let current_pivot = current_pivot.min(history.len());
    let mut new_pivot = current_pivot;
    let mut remaining: usize = history[current_pivot..].iter().map(turn_tokens).sum();
    while new_pivot < history.len()
        && overhead_tokens
            .saturating_add(current_summary_tokens)
            .saturating_add(remaining)
            > budget
    {
        remaining = remaining.saturating_sub(turn_tokens(&history[new_pivot]));
        new_pivot += 1;
    }
    new_pivot
}

// ---------------------------------------------------------------------------
// Summarization prompt + LLM call
// ---------------------------------------------------------------------------

/// Tool result text past this length gets truncated in the
/// summarization input. Picked to keep large file dumps from blowing
/// the summarizer's own context, since this prompt is fed back into
/// the same model. Mirrors the philosophy of
/// `tool_loop::MAX_TOOL_RESULT_BYTES` (50KB at the boundary) but
/// stricter -- the summarizer rarely needs the full payload to
/// produce useful bullets.
const SUMMARIZATION_TOOL_RESULT_CAP: usize = 800;

/// Build the chat-message list that asks the LLM to produce (or
/// extend) a conversation summary covering `turns`.
///
/// The system prompt is intentionally directive about what to keep vs.
/// drop because the resulting summary is read by the same LLM on every
/// subsequent turn -- prose-style "narrative" summaries waste tokens,
/// while structured fact lists translate cleanly into the model's
/// working context.
pub fn build_summarization_messages(
    existing_summary: Option<&str>,
    turns: &[ConversationTurn],
) -> Vec<ChatMessage> {
    let system = "You are a conversation summarizer for an AI coding assistant. \
Your output is fed back to the assistant on every subsequent turn, so it must \
preserve the operational facts the assistant needs to continue the conversation \
correctly -- it is not meant to read like prose.\n\
\n\
Preserve in your summary:\n\
- Specific file paths, function names, line numbers, and code symbols mentioned.\n\
- Decisions made and their rationale (one line each).\n\
- Open TODOs, unresolved errors, and known failure modes.\n\
- Tool calls run and the outcome (success/failure + key output).\n\
- User preferences expressed during the conversation.\n\
\n\
Drop:\n\
- Pleasantries and meta-discussion about how to collaborate.\n\
- Repeated context that's still verbatim in the recent turns.\n\
- Exact wording -- compress to bullets.\n\
\n\
Output format: a bulleted list wrapped in <conversation_summary>...</conversation_summary>. \
No preamble, no closing remarks.";

    let mut body = String::new();
    if let Some(prev) = existing_summary {
        body.push_str("Existing summary (extend it; do not restart):\n\n");
        body.push_str(prev.trim());
        body.push_str("\n\nAdditional turns to absorb:\n\n");
    } else {
        body.push_str("Turns to summarize:\n\n");
    }

    for (i, turn) in turns.iter().enumerate() {
        body.push_str(&format!("--- Turn {} ---\n", i + 1));
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

/// Drive the LLM to produce a new conversation summary covering
/// `turns`, extending `existing_summary` if present. Returns the
/// summary text with any outer tags stripped.
///
/// Errors propagate to the caller, which is expected to fall back to
/// `sliding_window_pivot` rather than blocking the user's prompt.
pub async fn run_summarization(
    llm: &dyn LlmBackend,
    model: &str,
    existing_summary: Option<&str>,
    turns: &[ConversationTurn],
    idle_timeout: Duration,
    cancel: CancellationToken,
) -> Result<String> {
    if turns.is_empty() {
        anyhow::bail!("run_summarization: no turns to summarize");
    }
    let messages = build_summarization_messages(existing_summary, turns);

    // Discard streamed deltas -- only the final text matters here.
    // `TokenSink = Box<dyn FnMut(&str) + Send>` lets us hand in a
    // throw-away closure for both the assistant token and reasoning
    // chunks. The streaming machinery in `OpenAiClient` still drives
    // the SSE state machine; we just don't relay it to the user.
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
        }
    }

    #[test]
    fn context_budget_uses_fraction_of_declared_length() {
        // 200_000 * 0.75 = 150_000
        assert_eq!(context_budget(Some(200_000)), 150_000);
        // 128_000 * 0.75 = 96_000
        assert_eq!(context_budget(Some(128_000)), 96_000);
    }

    #[test]
    fn context_budget_falls_back_when_unknown() {
        // FALLBACK_CONTEXT_LENGTH (128_000) * 0.75 = 96_000
        assert_eq!(context_budget(None), 96_000);
    }

    #[test]
    fn strategy_round_trips_through_string() {
        for s in [
            ContextStrategy::None,
            ContextStrategy::SlidingWindow,
            ContextStrategy::Hybrid,
        ] {
            assert_eq!(ContextStrategy::parse(s.as_str()), Some(s));
        }
        assert_eq!(ContextStrategy::parse("garbage"), None);
    }

    #[test]
    fn strategy_default_is_hybrid() {
        // Default must match the user's chosen strategy from the
        // design phase. Changing it requires a corresponding update
        // in PLANS.md / setup advanced.
        assert_eq!(ContextStrategy::default(), ContextStrategy::Hybrid);
    }

    #[test]
    fn plan_returns_none_when_already_under_budget() {
        let history = vec![turn("hi", "hello"), turn("how are you?", "great thanks")];
        let plan = plan_summarization_pivot(&history, 0, 0, 100, 10_000);
        assert!(plan.is_none(), "should not compress when already fits");
    }

    /// When the projection exceeds budget, the planner must absorb at
    /// least one turn. With trivial history (no tool exchanges) each
    /// absorbed turn replaces real tokens with the
    /// `SUMMARY_TOKENS_PER_ABSORBED_TURN` estimate.
    #[test]
    fn plan_absorbs_turns_until_under_budget() {
        let history: Vec<ConversationTurn> = (0..10)
            .map(|i| turn(&format!("user {i} "), &format!("agent {i} ")))
            .collect();
        let total: usize = history.iter().map(turn_tokens).sum();
        // Budget that's well below the verbatim cost forces compression.
        let budget = total / 4;
        let plan =
            plan_summarization_pivot(&history, 0, 0, 0, budget).expect("budget overrun -> plan");
        assert!(
            plan.new_pivot > 0,
            "pivot must advance past 0 to absorb anything"
        );
        assert!(
            !plan.exhausted_history || plan.new_pivot == history.len(),
            "exhausted_history flag must match pivot==len"
        );
    }

    /// When even absorbing the entire history can't shrink things
    /// enough (e.g. overhead alone exceeds budget), the planner must
    /// report `exhausted_history` so the caller knows to fall back to
    /// dropping non-history surface.
    #[test]
    fn plan_flags_exhausted_history_when_overhead_alone_exceeds_budget() {
        let history = vec![turn("a", "b"), turn("c", "d")];
        // Overhead alone is way over budget. No amount of absorbing
        // history will save us.
        let plan =
            plan_summarization_pivot(&history, 0, 0, 10_000, 100).expect("over budget -> plan");
        assert!(plan.exhausted_history);
        assert_eq!(plan.new_pivot, history.len());
    }

    /// `current_pivot` must be respected -- the planner never moves
    /// the pivot backwards (which would re-expose already-summarized
    /// turns).
    #[test]
    fn plan_never_moves_pivot_backwards() {
        let history: Vec<ConversationTurn> = (0..5).map(|i| turn(&format!("u{i}"), "a")).collect();
        let total: usize = history.iter().map(turn_tokens).sum();
        // Force compression by giving zero verbatim budget.
        let plan =
            plan_summarization_pivot(&history, 3, 0, 0, total / 8).expect("over budget -> plan");
        assert!(plan.new_pivot >= 3, "pivot must not regress past 3");
    }

    /// Tool exchanges round-trip back to the LLM on each replay so
    /// the planner must charge them against the verbatim budget --
    /// otherwise a session with a single 50KB tool result wouldn't
    /// trigger compression at all.
    #[test]
    fn plan_accounts_for_tool_exchange_tokens() {
        let big_tool_result = "x".repeat(20_000); // Many tokens.
        let history = vec![turn_with_tool(
            "do the thing",
            "done",
            "shell",
            "{}",
            &big_tool_result,
        )];
        let plan = plan_summarization_pivot(&history, 0, 0, 0, 1_000)
            .expect("tool-heavy turn exceeds budget -> plan");
        assert!(plan.new_pivot > 0, "tool-heavy turn must be absorbed");
    }

    /// Sliding-window fallback drops turns until under budget; it
    /// must respect the current pivot just like the summarizing
    /// planner does.
    #[test]
    fn sliding_window_drops_turns_until_under_budget() {
        let history: Vec<ConversationTurn> =
            (0..6).map(|i| turn(&format!("u{i} "), "a ")).collect();
        let total: usize = history.iter().map(turn_tokens).sum();
        let new_pivot = sliding_window_pivot(&history, 0, 0, 0, total / 3);
        assert!(new_pivot > 0 && new_pivot < history.len());
    }

    #[test]
    fn sliding_window_returns_current_pivot_when_already_fits() {
        let history = vec![turn("hi", "hello")];
        let new_pivot = sliding_window_pivot(&history, 0, 0, 50, 10_000);
        assert_eq!(new_pivot, 0);
    }

    #[test]
    fn build_summarization_messages_uses_existing_summary() {
        let turns = vec![turn("u", "a")];
        let msgs = build_summarization_messages(Some("- prior bullet"), &turns);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        let user_content = msgs[1].content.as_deref().unwrap();
        assert!(user_content.contains("Existing summary"));
        assert!(user_content.contains("- prior bullet"));
        assert!(user_content.contains("Additional turns to absorb"));
    }

    #[test]
    fn build_summarization_messages_when_no_prior_summary() {
        let turns = vec![turn("hello", "hi back")];
        let msgs = build_summarization_messages(None, &turns);
        let user_content = msgs[1].content.as_deref().unwrap();
        assert!(user_content.contains("Turns to summarize"));
        assert!(!user_content.contains("Existing summary"));
        assert!(user_content.contains("User: hello"));
        assert!(user_content.contains("Assistant: hi back"));
    }

    /// Oversized tool results must be truncated in the summarizer
    /// input so the summarizer's own context doesn't overflow.
    #[test]
    fn build_summarization_messages_truncates_huge_tool_results() {
        let huge = "z".repeat(SUMMARIZATION_TOOL_RESULT_CAP * 4);
        let turns = vec![turn_with_tool("u", "a", "shell", "{}", &huge)];
        let msgs = build_summarization_messages(None, &turns);
        let user_content = msgs[1].content.as_deref().unwrap();
        assert!(user_content.contains("(truncated)"));
        // The full payload must NOT have made it through verbatim.
        assert!(!user_content.contains(&huge));
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
        let turns = vec![turn("hello", "hi")];

        let out = run_summarization(
            &backend,
            "mock-model",
            None,
            &turns,
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await
        .expect("summarization succeeds");
        assert_eq!(out, "- key bullet");

        // The backend must have seen the system+user messages we
        // build, not the conversation history verbatim.
        let sent = last_messages.lock().unwrap().take().expect("called");
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[0].role, "system");
        assert_eq!(sent[1].role, "user");

        // Reasoning effort defaults to "low" -- summarization is
        // structured extraction, not deep reasoning.
        assert_eq!(
            last_reasoning_effort.lock().unwrap().as_deref(),
            Some("low")
        );
    }

    #[tokio::test]
    async fn run_summarization_rejects_empty_history() {
        let backend = CannedBackend {
            last_messages: Arc::new(Mutex::new(None)),
            last_reasoning_effort: Arc::new(Mutex::new(None)),
            response: LlmResponse::Text(String::new()),
        };
        let err = run_summarization(
            &backend,
            "mock-model",
            None,
            &[],
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await
        .expect_err("empty turns must error");
        assert!(
            err.to_string().contains("no turns to summarize"),
            "unexpected error message: {err}"
        );
    }
}
