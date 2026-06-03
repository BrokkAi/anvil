//! Per-turn conversation summarization, with hierarchical fallback
//! for turns too large to fit in the summarizer's context window.
//!
//! Single-call path mirrors Brokk's `ContextManager.compressHistory`:
//! one LLM call per turn, system prompt asks for a bulleted summary of
//! a single task entry. Brokk relies on `ContextSizeGuard` (a UI-side
//! pre-flight at file/content add time) to keep entries from growing
//! large enough to trip up `compressHistory` downstream, which has no
//! defense of its own.
//!
//! Anvil takes the downstream-defense approach instead: when a turn
//! exceeds the summarizer's input budget, split the turn into chunks,
//! summarize each chunk independently, then run a meta-summarization
//! pass over the chunk summaries to produce one coherent summary.
//! Recursive if even the combined chunk summaries don't fit in one
//! meta call. We could also gate at input time (ACP exposes
//! `session/prompt` server-side, and tool output is already bounded
//! per call by `MAX_TOOL_RESULT_BYTES`), but the hierarchical
//! summarizer guarantees the wedge can't form regardless of what
//! upstream gates exist -- so it's safe to skip them for now.
//!
//! The result is a guarantee: every turn we attempt to summarize
//! either gets a summary (lossy if the turn was monstrous, but never
//! silently dropped) or returns a clean `Err` that the caller
//! (`handle_compress`) reports to the user. Combined with the fact
//! that `set_turn_summary` only flips the in-memory + summaryContentId
//! state and leaves the verbatim log on disk, the session is
//! recoverable from any failure mode.

use std::time::Duration;

use anyhow::Result;
use futures::stream::{self, StreamExt, TryStreamExt};
use tokio_util::sync::CancellationToken;

use crate::llm_client::{ChatMessage, LlmBackend, LlmResponse, StreamChatRequest};
use crate::session::ConversationTurn;
use crate::tokens::{approximate_tokens, approximate_tokens_messages};

/// Maximum number of summarization LLM calls in flight at one time.
/// Keeps `/compress` and the auto-trigger from saturating provider
/// rate limits when a turn fans out into many chunks. Two is a
/// conservative default that avoids `429`s on the common providers
/// without forcing a per-backend rate-limit story; raise once Anvil
/// has provider-aware throttling.
const MAX_CONCURRENT_CHUNK_REQUESTS: usize = 2;

// ---------------------------------------------------------------------------
// Budget math
// ---------------------------------------------------------------------------

/// Fraction of declared context window we let the *prompt* portion of
/// a regular chat request occupy. 75% leaves room for the model's
/// response plus its own bookkeeping. Used by `/context` and the
/// per-turn-summarization trigger threshold.
const BUDGET_FRACTION: f64 = 0.75;

/// Conservative fallback used only for internal compression budgeting
/// when a backend does not publish a reliable context length. This is
/// not a provider-reported maximum and must not be presented to users
/// as the model's context window.
pub const FALLBACK_CONTEXT_LENGTH: u32 = 128_000;

/// Per-call input budget for the *summarizer*. Smaller than
/// `context_budget` because we want headroom for the system prompt
/// plus the summary the model will produce in the response (~25% of
/// the window is generous for the output bullet list).
const SUMMARIZER_INPUT_FRACTION: f64 = 0.65;

/// Token budget for a regular chat request's prompt. Used in the
/// `/context` report and as the threshold the per-prompt compression
/// trigger compares against.
pub fn context_budget(declared_context_length: Option<u32>) -> usize {
    let raw = declared_context_length.unwrap_or(FALLBACK_CONTEXT_LENGTH) as f64;
    (raw * BUDGET_FRACTION) as usize
}

/// Token budget for one *summarization* LLM call's input. The actual
/// summary the model writes in the response goes against the
/// remaining ~35% of the window.
fn summarizer_input_budget(declared_context_length: Option<u32>) -> usize {
    let raw = declared_context_length.unwrap_or(FALLBACK_CONTEXT_LENGTH) as f64;
    (raw * SUMMARIZER_INPUT_FRACTION) as usize
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Produce a summary for one conversation turn. Single LLM call when
/// the turn fits in the summarizer's input budget; otherwise splits
/// the turn into chunks, summarizes each, then runs a meta pass over
/// the chunk summaries (recursive if even the meta input doesn't
/// fit).
///
/// Errors propagate -- the caller is expected to leave the turn
/// uncompressed (matching Brokk's behavior in
/// `ContextManager.compressHistory`) rather than silently dropping
/// the turn. The persisted log is unaffected on every failure path.
pub async fn summarize_turn(
    llm: &dyn LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    context_length: Option<u32>,
    idle_timeout: Duration,
    cancel: CancellationToken,
) -> Result<String> {
    let budget = summarizer_input_budget(context_length);

    // Fast path: the whole turn fits in one summarization call.
    let single_call_messages = build_turn_summarization_messages(turn);
    if approximate_tokens_messages(&single_call_messages) <= budget {
        return run_summarization_request(llm, model, single_call_messages, idle_timeout, cancel)
            .await;
    }

    // Hierarchical path. The turn body is split into chunks, each
    // small enough to summarize independently; then the chunk
    // summaries are combined via a meta-summarization pass. Total
    // LLM calls: N (one per chunk) + 1 (meta) at the minimum, plus
    // any extra rounds the recursion needs if the combined chunk
    // summaries themselves overrun the meta budget.
    summarize_turn_hierarchical(llm, model, turn, budget, idle_timeout, cancel).await
}

async fn summarize_turn_hierarchical(
    llm: &dyn LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    budget: usize,
    idle_timeout: Duration,
    cancel: CancellationToken,
) -> Result<String> {
    let chunks = split_turn_to_chunks(turn, budget);
    if chunks.is_empty() {
        anyhow::bail!("split_turn_to_chunks returned no chunks (turn had no content?)");
    }

    let chunk_summaries =
        summarize_chunks_parallel(llm, model, &chunks, idle_timeout, cancel.clone()).await?;

    // Combine via a meta-summarization pass. Each chunk summary is
    // small (~bullets-only output), so the combined input is usually
    // far below budget. The fallback recursion handles the case
    // where even the combined summaries overrun -- e.g. 100+ chunks
    // each producing several KB of bullets.
    combine_chunk_summaries(llm, model, &chunk_summaries, budget, idle_timeout, cancel).await
}

/// Drive `MAX_CONCURRENT_CHUNK_REQUESTS` chunk summarizations
/// concurrently against the LLM, preserving submission order in the
/// returned vec. `buffered(N)` polls at most N futures at a time and
/// yields results in input order; `try_collect` short-circuits on
/// the first error so a 429 / network fail aborts the rest of the
/// run rather than burning credits on doomed work.
async fn summarize_chunks_parallel(
    llm: &dyn LlmBackend,
    model: &str,
    chunks: &[String],
    idle_timeout: Duration,
    cancel: CancellationToken,
) -> Result<Vec<String>> {
    let chunk_count = chunks.len();
    // Pre-build the per-chunk messages synchronously so the futures
    // dispatched to `buffered` carry only owned data. Capturing `&str`
    // model and `&[String]` chunks across the closure boundary makes
    // the compiler's auto-trait inference of `Send` on the resulting
    // future too narrow (it picks a concrete lifetime instead of a
    // higher-ranked one, breaking downstream `Send` bounds in the
    // ACP dispatch path).
    let prepared: Vec<Vec<ChatMessage>> = chunks
        .iter()
        .enumerate()
        .map(|(i, chunk_text)| {
            let part_label = format!("{} of {}", i + 1, chunk_count);
            build_chunk_summarization_messages(chunk_text, &part_label)
        })
        .collect();
    let model = model.to_string();
    let summaries: Vec<String> = stream::iter(prepared)
        .map(|messages| {
            let cancel = cancel.clone();
            let model = model.clone();
            async move {
                if cancel.is_cancelled() {
                    anyhow::bail!("summarization cancelled");
                }
                run_summarization_request(llm, &model, messages, idle_timeout, cancel).await
            }
        })
        .buffered(MAX_CONCURRENT_CHUNK_REQUESTS)
        .try_collect()
        .await?;
    Ok(summaries)
}

/// Run one meta-summarization pass over a list of chunk summaries.
/// Recurses (chunked) if the combined input itself overruns budget.
/// Used by `summarize_turn_hierarchical` as the join step.
async fn combine_chunk_summaries(
    llm: &dyn LlmBackend,
    model: &str,
    chunk_summaries: &[String],
    budget: usize,
    idle_timeout: Duration,
    cancel: CancellationToken,
) -> Result<String> {
    let combined = format_chunk_summaries(chunk_summaries);
    let messages = build_meta_summarization_messages(&combined);
    if approximate_tokens_messages(&messages) <= budget {
        return run_summarization_request(llm, model, messages, idle_timeout, cancel).await;
    }
    // Even the combined chunk summaries are too big. Recurse: treat
    // the combined text as a body that needs splitting and
    // summarizing, the same way we treat a too-big turn.
    if cancel.is_cancelled() {
        anyhow::bail!("summarization cancelled");
    }
    let sub_chunks = split_plain_text_to_chunks(&combined, budget);
    if sub_chunks.len() <= 1 {
        // Couldn't reduce further -- the combined text is dense and
        // every chunk is at the floor. Surface a clean error so the
        // turn stays uncompressed rather than looping forever.
        anyhow::bail!("combined chunk summaries do not fit in budget and cannot be split further");
    }
    let sub_summaries =
        summarize_chunks_parallel(llm, model, &sub_chunks, idle_timeout, cancel.clone()).await?;
    // Box the recursive future so the compiler doesn't try to
    // construct an infinite-size Future type. The recursion depth is
    // bounded by how aggressively `split_plain_text_to_chunks`
    // shrinks the input, so this terminates quickly in practice.
    Box::pin(combine_chunk_summaries(
        llm,
        model,
        &sub_summaries,
        budget,
        idle_timeout,
        cancel,
    ))
    .await
}

// ---------------------------------------------------------------------------
// LLM request driver
// ---------------------------------------------------------------------------

/// Drive a summarization stream and return the response text with any
/// `<conversation_summary>...</conversation_summary>` wrapper stripped.
/// Token deltas are discarded -- only the final body matters here.
async fn run_summarization_request(
    llm: &dyn LlmBackend,
    model: &str,
    messages: Vec<ChatMessage>,
    idle_timeout: Duration,
    cancel: CancellationToken,
) -> Result<String> {
    let on_token: Box<dyn FnMut(&str) + Send> = Box::new(|_| {});
    let on_thought: Box<dyn FnMut(&str) + Send> = Box::new(|_| {});
    let response = llm
        .stream_chat(StreamChatRequest {
            model: model.to_string(),
            messages,
            tools: None,
            // Summarization is structured extraction, not deep
            // reasoning -- "low" keeps cost down on reasoning-capable
            // models that bill thinking tokens.
            reasoning_effort: Some("low".to_string()),
            structured_output: None,
            on_token,
            on_thought,
            cancel,
            idle_timeout,
        })
        .await?;
    let text = match response {
        LlmResponse::Text { text, .. } => text,
        LlmResponse::ToolCalls { text, .. } => text,
    };
    Ok(strip_summary_tags(&text))
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

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/// System prompt for summarizing a single complete turn. Mirrors the
/// directive style of Brokk's `SummarizerPrompts.compressHistory`.
const SYSTEM_PROMPT_TURN: &str = "You are a conversation summarizer for an AI coding assistant. \
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

/// System prompt for summarizing one CHUNK of a turn that's been
/// split because it didn't fit in one call. The model is told that
/// another step will combine its output with summaries of the other
/// chunks, so it should focus on extraction rather than coherent
/// narrative.
const SYSTEM_PROMPT_CHUNK: &str = "You are summarizing PART of a single conversation turn that \
was too large to summarize in one pass. A later step will combine your \
output with summaries of the other parts into a single coherent summary. \
Your job is extraction, not narrative.\n\
\n\
Preserve, from this part only:\n\
- Specific file paths, function names, line numbers, code symbols.\n\
- Tool calls run and the outcome.\n\
- Decisions, TODOs, errors, user preferences.\n\
\n\
Drop:\n\
- Pleasantries.\n\
- Exact wording (compress to bullets).\n\
- Verbose tool output already captured by a short outcome line.\n\
\n\
Output format: a bulleted list, no preamble, no closing remarks, \
no <conversation_summary> tags (those go on the final combined output).";

/// System prompt for the META-summarization pass that joins chunk
/// summaries into one coherent turn summary. Emphasizes deduplication
/// (the same file path might appear in multiple chunks) and
/// preserving ordering of decisions/tool calls.
const SYSTEM_PROMPT_META: &str = "You will receive several bulleted summaries, each covering one \
part of a single conversation turn that was too large to summarize at \
once. Combine them into ONE coherent summary of the entire turn:\n\
\n\
- Preserve every distinct fact (file paths, function names, tool calls, \
  decisions, errors).\n\
- Deduplicate facts that appear in multiple parts.\n\
- Preserve the order in which decisions/tool calls happened across parts.\n\
- Keep it concise: bullets, not paragraphs.\n\
\n\
Output format: a bulleted list wrapped in \
<conversation_summary>...</conversation_summary>. No preamble, no closing \
remarks.";

/// Build the chat-message list for a single-call (whole-turn) summarization.
pub fn build_turn_summarization_messages(turn: &ConversationTurn) -> Vec<ChatMessage> {
    let mut body = String::new();
    body.push_str("Turn to summarize:\n\n");
    body.push_str("User: ");
    body.push_str(turn.user_prompt.trim());
    body.push('\n');
    for exchange in &turn.tool_exchanges {
        body.push_str(&format!(
            "Tool `{}` args={} -> {}\n",
            exchange.tool_name, exchange.arguments, exchange.result
        ));
    }
    if !turn.agent_response.trim().is_empty() {
        body.push_str("Assistant: ");
        body.push_str(turn.agent_response.trim());
        body.push('\n');
    }
    vec![
        ChatMessage::system(SYSTEM_PROMPT_TURN),
        ChatMessage::user(body),
    ]
}

/// Build the chat-message list for summarizing one chunk of a turn.
/// `part_label` is woven into the user message so the model knows
/// which part it's looking at (occasionally useful for ordering hints).
fn build_chunk_summarization_messages(chunk_text: &str, part_label: &str) -> Vec<ChatMessage> {
    let body = format!("Turn fragment ({part_label}):\n\n{chunk_text}");
    vec![
        ChatMessage::system(SYSTEM_PROMPT_CHUNK),
        ChatMessage::user(body),
    ]
}

/// Build the chat-message list for joining chunk summaries.
fn build_meta_summarization_messages(combined: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(SYSTEM_PROMPT_META),
        ChatMessage::user(combined.to_string()),
    ]
}

fn format_chunk_summaries(chunk_summaries: &[String]) -> String {
    let mut out = String::new();
    for (i, s) in chunk_summaries.iter().enumerate() {
        out.push_str(&format!("Part {} of {}:\n", i + 1, chunk_summaries.len()));
        out.push_str(s.trim());
        out.push_str("\n\n");
    }
    out
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

/// Rough overhead in tokens reserved for the chunk's system prompt
/// plus framing inside `build_chunk_summarization_messages`. Subtracted
/// from the budget when sizing chunks so the full request body fits.
const CHUNK_OVERHEAD_TOKENS: usize = 800;

/// Hard floor for any single chunk's input. Even if `budget - overhead`
/// resolves to a tiny number for a degenerate small-context model, we
/// keep at least this many tokens per chunk so the LLM has something
/// meaningful to summarize. The recursion in
/// `combine_chunk_summaries` uses this floor to detect "can't split
/// further" and bail out cleanly.
const MIN_CHUNK_TOKENS: usize = 1_000;

/// Split one turn into a list of chunk-body strings, each small
/// enough that wrapping it with `SYSTEM_PROMPT_CHUNK` produces an
/// under-budget request.
///
/// Atoms (in chronological order):
/// 1. `User: <user_prompt>`
/// 2. For each tool exchange: `Tool <name> args=... -> <result>`
/// 3. `Assistant: <agent_response>`
///
/// Atoms are greedily packed into chunks; if an atom alone exceeds
/// the per-chunk budget, it gets split internally (by line, then by
/// character as a last resort) before packing continues.
fn split_turn_to_chunks(turn: &ConversationTurn, budget: usize) -> Vec<String> {
    let per_chunk_budget = budget
        .saturating_sub(CHUNK_OVERHEAD_TOKENS)
        .max(MIN_CHUNK_TOKENS);
    let mut atoms: Vec<String> = Vec::new();
    if !turn.user_prompt.trim().is_empty() {
        atoms.push(format!("User: {}", turn.user_prompt.trim()));
    }
    for exchange in &turn.tool_exchanges {
        atoms.push(format!(
            "Tool `{}` args={} -> {}",
            exchange.tool_name, exchange.arguments, exchange.result
        ));
    }
    if !turn.agent_response.trim().is_empty() {
        atoms.push(format!("Assistant: {}", turn.agent_response.trim()));
    }

    pack_atoms_into_chunks(atoms, per_chunk_budget)
}

/// Split a plain string into chunk-body strings. Used by the meta-pass
/// recursion when even the combined chunk summaries overrun budget.
/// Operates by line, falling back to character split for monstrous
/// single lines.
fn split_plain_text_to_chunks(text: &str, budget: usize) -> Vec<String> {
    let per_chunk_budget = budget
        .saturating_sub(CHUNK_OVERHEAD_TOKENS)
        .max(MIN_CHUNK_TOKENS);
    let atom = text.to_string();
    pack_atoms_into_chunks(vec![atom], per_chunk_budget)
}

/// Pack atoms greedily into chunks. Any atom larger than the budget
/// is internally split (by line, then character) before being
/// packed. Returns at least one chunk even when the input is tiny.
fn pack_atoms_into_chunks(atoms: Vec<String>, per_chunk_budget: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;
    let separator = "\n\n";
    let separator_tokens = approximate_tokens(separator);

    let push_current =
        |current: &mut String, current_tokens: &mut usize, chunks: &mut Vec<String>| {
            if !current.is_empty() {
                chunks.push(std::mem::take(current));
                *current_tokens = 0;
            }
        };

    for atom in atoms {
        let atom_tokens = approximate_tokens(&atom);
        if atom_tokens <= per_chunk_budget {
            let extra_tokens = if current.is_empty() {
                atom_tokens
            } else {
                separator_tokens + atom_tokens
            };
            if current_tokens + extra_tokens > per_chunk_budget && !current.is_empty() {
                push_current(&mut current, &mut current_tokens, &mut chunks);
            }
            if !current.is_empty() {
                current.push_str(separator);
                current_tokens += separator_tokens;
            }
            current.push_str(&atom);
            current_tokens += atom_tokens;
        } else {
            // Atom too big for one chunk on its own. Flush whatever's
            // pending, then split this atom and emit each piece as
            // its own chunk (packing fills back in on the trailing
            // piece if it has room for more).
            push_current(&mut current, &mut current_tokens, &mut chunks);
            for piece in split_single_atom(&atom, per_chunk_budget) {
                chunks.push(piece);
            }
        }
    }
    push_current(&mut current, &mut current_tokens, &mut chunks);

    if chunks.is_empty() {
        // Degenerate: caller passed an empty atom list. Emit one
        // empty chunk so downstream loops don't have to special-case.
        chunks.push(String::new());
    }
    chunks
}

/// Split a single oversized atom by lines; if a single line itself
/// exceeds the budget, fall back to character split. The result is a
/// list of chunk bodies, each ≤ per_chunk_budget tokens.
fn split_single_atom(atom: &str, per_chunk_budget: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;
    for line in atom.split_inclusive('\n') {
        let line_tokens = approximate_tokens(line);
        if line_tokens > per_chunk_budget {
            // Single line too big -- flush, then char-split this line.
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
                current_tokens = 0;
            }
            for piece in split_string_by_chars(line, per_chunk_budget) {
                out.push(piece);
            }
            continue;
        }
        if current_tokens + line_tokens > per_chunk_budget && !current.is_empty() {
            out.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        current.push_str(line);
        current_tokens += line_tokens;
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Last-resort split for a single line longer than the budget: cut
/// by characters at a budget-shaped boundary. The "4 chars per token"
/// estimate matches OpenAI's rule-of-thumb and is close enough to
/// `o200k_base`'s real ratio for English / source code.
fn split_string_by_chars(s: &str, per_chunk_budget: usize) -> Vec<String> {
    if s.is_empty() {
        return vec![String::new()];
    }
    let target_chars = per_chunk_budget.saturating_mul(4).max(256);
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut buf_chars = 0usize;
    for ch in s.chars() {
        buf.push(ch);
        buf_chars += 1;
        if buf_chars >= target_chars {
            out.push(std::mem::take(&mut buf));
            buf_chars = 0;
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
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
            structured_output: None,
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
            structured_output: None,
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
    fn summarizer_input_budget_is_smaller_than_chat_budget() {
        // Summarizer leaves room for its own response inside the
        // window, so its input budget must be tighter than the chat
        // budget computed by `context_budget`.
        let chat = context_budget(Some(200_000));
        let summ = summarizer_input_budget(Some(200_000));
        assert!(
            summ < chat,
            "summarizer budget {summ} should be < chat budget {chat}"
        );
    }

    #[test]
    fn build_turn_summarization_messages_includes_user_tool_assistant() {
        let t = turn_with_tool("find TODOs", "found 3", "shell", r#"{"cmd":"rg"}"#, "out");
        let msgs = build_turn_summarization_messages(&t);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        let body = msgs[1].text_content().unwrap();
        assert!(body.contains("User: find TODOs"));
        assert!(body.contains("Tool `shell`"));
        assert!(body.contains("Assistant: found 3"));
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

    /// A small turn produces exactly one chunk -- no split needed.
    /// This is the "fast path" the public API takes when the turn
    /// fits in one summarization call.
    #[test]
    fn split_turn_to_chunks_keeps_small_turn_in_one_chunk() {
        let t = turn("u", "a");
        let chunks = split_turn_to_chunks(&t, 10_000);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("User: u"));
        assert!(chunks[0].contains("Assistant: a"));
    }

    /// A turn with a single monstrously-large tool result must split
    /// into multiple chunks, each under budget. The character split
    /// fallback handles the case where one tool result has no line
    /// breaks.
    #[test]
    fn split_turn_to_chunks_breaks_huge_tool_result() {
        // ~50k tokens of diverse content. BPE tokenizers compress
        // long runs of identical characters very aggressively
        // (`"x".repeat(N)` is almost free), so we use varied tokens
        // instead to actually stress the splitter.
        let huge: String = (0..6_000)
            .map(|i| format!("line {i}: foo bar baz qux quux corge grault garply\n"))
            .collect();
        let t = turn_with_tool("u", "a", "shell", "{}", &huge);
        let chunks = split_turn_to_chunks(&t, 8_000);
        assert!(
            chunks.len() > 1,
            "expected multi-chunk output for huge tool result, got {}",
            chunks.len()
        );
        for chunk in &chunks {
            // Each chunk fits inside the per-chunk budget (which is
            // budget - CHUNK_OVERHEAD_TOKENS, with a floor of
            // MIN_CHUNK_TOKENS). Use the raw budget as the loose
            // upper bound -- the check is "under budget", not
            // "exactly at floor".
            assert!(
                approximate_tokens(chunk) <= 8_000,
                "chunk exceeds budget: ~{} tokens",
                approximate_tokens(chunk)
            );
        }
    }

    /// A turn with many small atoms (lots of short tool calls) packs
    /// greedily -- the chunker doesn't waste a chunk per atom.
    #[test]
    fn split_turn_to_chunks_packs_small_atoms_greedily() {
        let mut t = turn("u", "a");
        for i in 0..40 {
            t.tool_exchanges.push(ToolExchange {
                call_id: format!("c{i}"),
                tool_name: "noop".into(),
                arguments: format!(r#"{{"i":{i}}}"#),
                result: format!("result {i}"),
            });
        }
        let chunks = split_turn_to_chunks(&t, 8_000);
        assert!(
            chunks.len() < 40,
            "greedy packing should yield far fewer chunks than atoms"
        );
    }

    /// `split_plain_text_to_chunks` is the meta-pass recursion's
    /// splitter. It must produce multiple chunks when the input
    /// exceeds budget, and at least one chunk when the input is
    /// trivial.
    #[test]
    fn split_plain_text_handles_empty_and_huge() {
        assert_eq!(split_plain_text_to_chunks("", 5_000).len(), 1);
        let huge = "line\n".repeat(40_000); // ~200_000 chars
        let chunks = split_plain_text_to_chunks(&huge, 5_000);
        assert!(chunks.len() > 1);
    }

    /// Mock backend that dispatches its response based on which
    /// system prompt the caller sent (chunk vs. meta vs. whole-turn).
    /// Pop-from-vec ordering would couple the test to the splitter's
    /// exact chunk count; matching on prompt keeps the test robust to
    /// splitter changes.
    struct ScriptedBackend {
        turn_response: String,
        chunk_response: String,
        meta_response: String,
        call_count: Arc<Mutex<usize>>,
        seen_system_prompts: Arc<Mutex<Vec<String>>>,
    }

    impl ScriptedBackend {
        #[allow(clippy::type_complexity)]
        fn new(
            turn_response: impl Into<String>,
            chunk_response: impl Into<String>,
            meta_response: impl Into<String>,
        ) -> (Self, Arc<Mutex<usize>>, Arc<Mutex<Vec<String>>>) {
            let call_count = Arc::new(Mutex::new(0usize));
            let seen = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    turn_response: turn_response.into(),
                    chunk_response: chunk_response.into(),
                    meta_response: meta_response.into(),
                    call_count: call_count.clone(),
                    seen_system_prompts: seen.clone(),
                },
                call_count,
                seen,
            )
        }
    }

    impl LlmBackend for ScriptedBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            async { Ok(vec!["mock".into()]) }.boxed()
        }
        fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            *self.call_count.lock().unwrap() += 1;
            let system = request
                .messages
                .first()
                .and_then(|m| m.text_content())
                .unwrap_or("")
                .to_string();
            let response = if system.contains("Combine them into ONE coherent summary") {
                self.meta_response.clone()
            } else if system.contains("summarizing PART") {
                self.chunk_response.clone()
            } else {
                self.turn_response.clone()
            };
            self.seen_system_prompts.lock().unwrap().push(system);
            async move {
                Ok(LlmResponse::Text {
                    text: response,
                    usage: crate::llm_client::TokenUsage::default(),
                })
            }
            .boxed()
        }
    }

    /// The fast path: a small turn produces exactly one LLM call and
    /// that call uses the single-turn system prompt.
    #[tokio::test]
    async fn summarize_turn_takes_single_call_path_for_small_turn() {
        let (backend, call_count, seen) = ScriptedBackend::new(
            "<conversation_summary>\n- bullet\n</conversation_summary>",
            "- chunk (unexpected)",
            "- meta (unexpected)",
        );
        let t = turn("hello", "hi");
        let out = summarize_turn(
            &backend,
            "mock",
            &t,
            Some(200_000),
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await
        .expect("succeeds");
        assert_eq!(out, "- bullet");
        assert_eq!(
            *call_count.lock().unwrap(),
            1,
            "expected exactly one LLM call"
        );
        let prompts = seen.lock().unwrap();
        assert!(
            prompts[0].contains("replaces a single past turn"),
            "should use turn-level system prompt; got: {}",
            &prompts[0][..prompts[0].len().min(120)]
        );
    }

    /// The hierarchical path: a turn that overflows budget triggers
    /// per-chunk calls plus a meta call. Verifies both prompt kinds
    /// are invoked and the meta result is returned.
    #[tokio::test]
    async fn summarize_turn_takes_hierarchical_path_for_oversized_turn() {
        let (backend, call_count, seen) = ScriptedBackend::new(
            "- turn (unexpected)",
            "- chunk bullet",
            "<conversation_summary>\n- final combined\n</conversation_summary>",
        );
        // ~15k tokens of varied content with a 16k declared context
        // → ~10.4k summarizer budget → hierarchical path fires.
        // BPE compresses long runs of identical characters
        // aggressively (`"x".repeat(N)` tokenizes to almost
        // nothing), so we use varied tokens that actually weigh in.
        let huge: String = (0..1_500)
            .map(|i| format!("line {i} word1 word2 word3 word4 word5 word6 word7\n"))
            .collect();
        let t = turn_with_tool("user prompt", "agent reply", "shell", "{}", &huge);
        let out = summarize_turn(
            &backend,
            "mock",
            &t,
            Some(16_000),
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await
        .expect("succeeds");
        assert_eq!(out, "- final combined");
        let count = *call_count.lock().unwrap();
        assert!(
            count >= 2,
            "expected ≥2 LLM calls (≥1 chunk + meta), got {count}"
        );
        let prompts = seen.lock().unwrap();
        assert!(
            prompts
                .last()
                .unwrap()
                .contains("Combine them into ONE coherent summary"),
            "last call should use meta system prompt"
        );
        assert!(
            prompts
                .iter()
                .take(prompts.len() - 1)
                .all(|p| p.contains("summarizing PART")),
            "non-final calls should use chunk system prompt"
        );
    }

    /// Cancellation propagates through the chunked path: if the token
    /// fires between chunks, the function bails with an error rather
    /// than continuing to make LLM calls.
    #[tokio::test]
    async fn summarize_turn_honors_cancellation_between_chunks() {
        struct AlwaysOkBackend {
            call_count: Arc<Mutex<usize>>,
            cancel: CancellationToken,
        }
        impl LlmBackend for AlwaysOkBackend {
            fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
                async { Ok(vec!["mock".into()]) }.boxed()
            }
            fn stream_chat(
                &self,
                _request: StreamChatRequest,
            ) -> BoxFuture<'_, Result<LlmResponse>> {
                *self.call_count.lock().unwrap() += 1;
                // Cancel after the first call returns -- the next
                // iteration of the chunk loop should observe the
                // token and bail out.
                self.cancel.cancel();
                async move {
                    Ok(LlmResponse::Text {
                        text: "- partial".into(),
                        usage: crate::llm_client::TokenUsage::default(),
                    })
                }
                .boxed()
            }
        }
        let cancel = CancellationToken::new();
        let call_count = Arc::new(Mutex::new(0usize));
        let backend = AlwaysOkBackend {
            call_count: call_count.clone(),
            cancel: cancel.clone(),
        };
        // Diverse content so the BPE tokenizer doesn't compress it
        // down to a single chunk -- we need the hierarchical path
        // to fire for the cancel-between-chunks check to be
        // meaningful.
        let huge: String = (0..5_000)
            .map(|i| format!("line {i}: foo bar baz qux quux corge grault garply\n"))
            .collect();
        let t = turn_with_tool("u", "a", "shell", "{}", &huge);
        let result = summarize_turn(
            &backend,
            "mock",
            &t,
            Some(16_000),
            Duration::from_secs(60),
            cancel.clone(),
        )
        .await;
        assert!(result.is_err(), "cancelled run should return Err");
        // We expect the first chunk to have been issued before the
        // cancel was observed; subsequent chunks should be skipped.
        let count = *call_count.lock().unwrap();
        assert!(
            (1..10).contains(&count),
            "should have made some calls but stopped early, got {count}"
        );
    }

    /// Parallel chunk summarization must not exceed
    /// `MAX_CONCURRENT_CHUNK_REQUESTS` in-flight requests at any
    /// instant. Without the cap a long compress run would fan out
    /// to N parallel calls and trip provider rate limits (`429`s).
    /// The test backend tracks active call count via an
    /// atomic-style counter and asserts the high-water mark equals
    /// the cap (not just ≤; with enough chunks we expect the cap to
    /// be saturated).
    #[tokio::test]
    async fn parallel_chunk_summarization_honors_concurrency_cap() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ConcurrencyTrackingBackend {
            in_flight: Arc<AtomicUsize>,
            high_water: Arc<AtomicUsize>,
            meta_response: String,
            chunk_response: String,
        }
        impl LlmBackend for ConcurrencyTrackingBackend {
            fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
                async { Ok(vec!["mock".into()]) }.boxed()
            }
            fn stream_chat(
                &self,
                request: StreamChatRequest,
            ) -> BoxFuture<'_, Result<LlmResponse>> {
                // Increment in-flight on entry, bump the high-water
                // mark, sleep to leave the slot held long enough
                // for any over-cap futures to be visible, then
                // decrement.
                let in_flight = self.in_flight.clone();
                let high_water = self.high_water.clone();
                let system = request
                    .messages
                    .first()
                    .and_then(|m| m.text_content())
                    .unwrap_or("")
                    .to_string();
                let response = if system.contains("Combine them into ONE coherent summary") {
                    self.meta_response.clone()
                } else {
                    self.chunk_response.clone()
                };
                async move {
                    let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    high_water.fetch_max(current, Ordering::SeqCst);
                    // Sleep is short but long enough that concurrent
                    // futures overlap deterministically under any
                    // tokio scheduler order.
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    Ok(LlmResponse::Text {
                        text: response,
                        usage: crate::llm_client::TokenUsage::default(),
                    })
                }
                .boxed()
            }
        }

        let in_flight = Arc::new(AtomicUsize::new(0));
        let high_water = Arc::new(AtomicUsize::new(0));
        let backend = ConcurrencyTrackingBackend {
            in_flight: in_flight.clone(),
            high_water: high_water.clone(),
            meta_response: "<conversation_summary>\n- meta\n</conversation_summary>".into(),
            chunk_response: "- chunk".into(),
        };
        // Enough varied content to fan out into several chunks --
        // 4+ ensures the cap can actually be saturated.
        let huge: String = (0..2_500)
            .map(|i| format!("line {i} word1 word2 word3 word4 word5 word6 word7\n"))
            .collect();
        let t = turn_with_tool("u", "a", "shell", "{}", &huge);
        let out = summarize_turn(
            &backend,
            "mock",
            &t,
            Some(16_000),
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await
        .expect("succeeds");
        assert_eq!(out, "- meta");
        let max_observed = high_water.load(Ordering::SeqCst);
        assert!(
            max_observed <= MAX_CONCURRENT_CHUNK_REQUESTS,
            "concurrency cap exceeded: observed {max_observed} in-flight (cap is {MAX_CONCURRENT_CHUNK_REQUESTS})"
        );
        // Saturation check: with enough chunks the cap should
        // actually be hit. If `max_observed` were stuck at 1, the
        // parallelization isn't actually engaging.
        assert!(
            max_observed >= 2,
            "parallelization not engaging: max in-flight was {max_observed}"
        );
    }
}
