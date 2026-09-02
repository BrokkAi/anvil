//! Full-history model-context compaction and turn-recap summarization.
//!
//! Model history is compacted cumulatively at completed tool-exchange
//! boundaries. The canonical system/project/skill prefix stays verbatim; an
//! older dynamic prefix becomes a structured `<state_snapshot>`, the recent
//! tail stays exact, and the current `update_plan` value is pinned separately.
//! The checkpoint is provider-neutral and persisted on its anchor turn, while
//! raw turns remain untouched for ACP replay, rewind, and Brokk compatibility.
//!
//! Compaction tries a native, in-conversation path first: the full native
//! message list (canonical prefix + dynamic history) plus the advertised
//! tool schemas is sent back to the model as-is, with one trailing user
//! instruction asking it to write the `<state_snapshot>` in place. This
//! preserves provider prompt-cache reuse and loses no structure, unlike
//! flattening to text. When that attempt is skipped (the input is already
//! too close to the context window to risk it), fails, or produces an
//! unusable or non-reducing snapshot, compaction falls back to rendering
//! history as plain text and summarizing it in a fresh request, splitting,
//! extracting, and folding hierarchically until the final state-snapshot
//! request fits. The same chunking machinery is retained for the
//! independent user-facing turn-recap feature.

use anyhow::Result;
use futures::stream::{self, StreamExt, TryStreamExt};
use std::collections::HashMap;
#[cfg(test)]
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::session::ConversationTurn;
use crate::tokens::{approximate_tokens, approximate_tokens_messages};
use anvil_llm::llm_client::{
    ChatContentPart, ChatMessage, IdleTimeouts, LlmBackend, LlmResponse, StreamChatRequest,
    TokenUsage, ToolDefinition, stream_chat_no_visible_output_with_retry,
};

/// Maximum number of summarization LLM calls in flight at one time.
/// Keeps oversized recap summarization from saturating provider rate
/// limits when a turn fans out into many chunks. Two is a
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
/// full-history compaction trigger threshold.
const BUDGET_FRACTION: f64 = 0.75;

/// Conservative fallback when the backend doesn't publish a context
/// length (Codex, Ollama). Undershooting just costs us extra
/// summarization rounds; overshooting silently drops the user's
/// prompt mid-stream, which is much worse.
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

#[derive(Debug)]
pub struct HistoryCompaction {
    pub checkpoint_messages: Vec<ChatMessage>,
    pub usage: TokenUsage,
    pub before_tokens: usize,
    pub after_tokens: usize,
}

/// Exact messages that compaction must retain outside the generated snapshot.
///
/// A long active turn must keep its user request verbatim. The request can
/// contain an output schema or another exact contract that a summary can lose.
#[derive(Default)]
pub struct HistoryPins<'a> {
    pub current_plan: Option<&'a crate::plan::UpdatePlanArgs>,
    pub active_user_message: Option<&'a ChatMessage>,
}

/// The `<state_snapshot>` XML schema, shared verbatim by the rendered
/// fallback prompt and the native in-conversation instruction so the tag
/// list has exactly one source of truth.
const STATE_SNAPSHOT_SCHEMA: &str = "<state_snapshot>\n\
<primary_request_and_intent>...</primary_request_and_intent>\n\
<explicit_requirements>...</explicit_requirements>\n\
<all_user_messages>...</all_user_messages>\n\
<decisions_and_rationale>...</decisions_and_rationale>\n\
<files_and_code_state>...</files_and_code_state>\n\
<commands_tests_and_evidence>...</commands_tests_and_evidence>\n\
<errors_and_fixes>...</errors_and_fixes>\n\
<pending_tasks>...</pending_tasks>\n\
<current_work>...</current_work>\n\
<next_step>...</next_step>\n\
</state_snapshot>";

const HISTORY_SNAPSHOT_PROMPT_PREFIX: &str = "You maintain the working memory of an AI coding \
agent. Rewrite the supplied conversation history as a precise state snapshot that another agent \
can continue from immediately. Preserve facts and evidence, not prose. Never claim work or \
verification that the history does not show. Include every user message or request, explicit \
requirements and preferences, decisions and rationale, important files/symbols/patch state, \
commands and results, errors and attempted fixes, current work, pending work, and the best next \
step. Distinguish verified facts from hypotheses. Never present an action that already failed as \
current or next work without stating that it failed and why. Tool calls may contain the only \
evidence, so read them carefully. Omit private analysis/reasoning and redundant chatter.\n\n\
Return only this XML structure:\n";

/// System prompt for the rendered-fallback compaction request. Built (rather
/// than a plain `const`) so it can splice in [`STATE_SNAPSHOT_SCHEMA`]
/// without duplicating the tag list -- `concat!` cannot reference a named
/// `const` string, only literals.
fn history_snapshot_prompt() -> String {
    format!("{HISTORY_SNAPSHOT_PROMPT_PREFIX}{STATE_SNAPSHOT_SCHEMA}")
}

/// Trailing user instruction appended to the native in-conversation
/// compaction request (Phase 2's primary path). Unlike
/// [`HISTORY_SNAPSHOT_PROMPT_PREFIX`], which introduces a flattened,
/// out-of-band history dump, this is spoken directly into the ongoing
/// conversation -- it announces the checkpoint, tells the model not to act
/// on anything (no tool calls), and asks for the same snapshot schema.
const NATIVE_SNAPSHOT_INSTRUCTION_PREFIX: &str = "The conversation above is this session's actual \
history so far and is about to exceed its context window. Before we continue, checkpoint it: \
rewrite everything above as a precise state snapshot that another instance of you can continue \
from immediately, using only the conversation itself as source material. Preserve facts and \
evidence, not prose. Never claim work or verification that the history does not show. Include \
every user message or request, explicit requirements and preferences, decisions and rationale, \
important files/symbols/patch state, commands and results, errors and attempted fixes, current \
work, pending work, and the best next step. Distinguish verified facts from hypotheses. Never \
present an action that already failed as current or next work without stating that it failed and \
why. Tool calls may contain the only evidence, so read them carefully. Omit private \
analysis/reasoning and redundant chatter. Do NOT call any tools for this; respond with text \
only.\n\n\
Return only this XML structure:\n";

fn native_snapshot_instruction() -> String {
    format!("{NATIVE_SNAPSHOT_INSTRUCTION_PREFIX}{STATE_SNAPSHOT_SCHEMA}")
}

/// Above this fraction of the declared context window, `all_messages` alone
/// (before the trailing compaction instruction is even added) is judged too
/// close to the limit to risk the native in-conversation attempt, so
/// compaction skips straight to the rendered fallback. The trigger that
/// calls `compact_history` at all fires at [`BUDGET_FRACTION`] (75%), so
/// this only trips when the caller's token estimate ran far behind the
/// real prompt size.
const NATIVE_ATTEMPT_GUARD_FRACTION: f64 = 0.90;

const HISTORY_CHUNK_PROMPT: &str = "Extract durable working-memory facts from this fragment of \
an AI coding-agent history. Preserve user requests, requirements, decisions, paths and symbols, \
patch state, commands and their actual results, errors, unfinished work, and chronology. Do not \
infer success. Omit private analysis and redundant chatter. Output concise labeled bullets only; \
a later pass will assemble the final state snapshot.";

/// Frames the checkpoint for the restarted agent that reads it next. Issue
/// #326: without this framing, a fresh session treated the checkpoint as
/// just more history to react to, so it re-read files and re-issued tool
/// calls whose outcomes were already recorded (and often already failed).
const CHECKPOINT_PREAMBLE: &str = "A previous session of this same agent ran out of context while \
working on the task described below. The state snapshot and digests that follow summarize work \
that is ALREADY DONE. Build on it instead of repeating it: do not re-issue tool calls whose \
outcomes are recorded here, do not re-read files listed in <files_already_read> unless you need \
content that is not summarized here, and never repeat a call listed in <failed_tool_calls> with \
the same arguments -- those exact calls already failed. Fix the arguments or take a different \
approach.";

/// Compact complete provider-neutral model history into one cumulative state
/// checkpoint. The caller keeps the canonical system/instruction prefix and
/// any post-checkpoint tail verbatim.
///
/// `all_messages` is the full conversation (canonical prefix + dynamic
/// history) in native form; `dynamic_start` is the index where the
/// compactable dynamic history begins. Everything below operates on
/// `history = &all_messages[dynamic_start..]`, exactly as if that slice had
/// been passed directly -- `all_messages` and `tools` exist only so the
/// native attempt can send the model its own real conversation (plus
/// advertised tool schemas) instead of a flattened rendering of it.
#[allow(clippy::too_many_arguments)]
pub async fn compact_history(
    llm: &dyn LlmBackend,
    model: &str,
    all_messages: &[ChatMessage],
    dynamic_start: usize,
    tools: Option<&[ToolDefinition]>,
    pins: HistoryPins<'_>,
    reasoning_effort: Option<String>,
    context_length: Option<u32>,
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
) -> Result<HistoryCompaction> {
    let history = &all_messages[dynamic_start..];
    if history.is_empty() {
        anyhow::bail!("cannot compact empty history");
    }
    let before_tokens = approximate_tokens_messages(history);
    let mut compactable_history = history.to_vec();
    if let Some(active_user_message) = pins.active_user_message {
        let index = compactable_history
            .iter()
            .rposition(|message| message == active_user_message)
            .ok_or_else(|| anyhow::anyhow!("active user message is not in compacted history"))?;
        compactable_history.remove(index);
    }
    if compactable_history.is_empty() {
        anyhow::bail!("cannot compact history with only a pinned user message");
    }
    let budget = summarizer_input_budget(context_length);
    let mut usage = TokenUsage::default();
    let tail_start = exact_tail_start(&compactable_history, context_length);
    let (history_to_summarize, exact_tail) = compactable_history.split_at(tail_start);
    let history_to_summarize = if history_to_summarize.is_empty() {
        &compactable_history[..]
    } else {
        history_to_summarize
    };
    let exact_tail = if history_to_summarize.len() == compactable_history.len() {
        &[][..]
    } else {
        exact_tail
    };
    let digests = build_digests(history);

    // Primary path: ask the model to checkpoint its own real conversation
    // in place. Skipped when the raw conversation is already close enough
    // to the window that adding the trailing instruction risks overflowing
    // it outright -- the rendered fallback below is smaller by
    // construction (flattened text plus one instruction, not the full
    // native message list).
    let window_tokens = context_length.unwrap_or(FALLBACK_CONTEXT_LENGTH) as f64;
    let all_messages_tokens = approximate_tokens_messages(all_messages) as f64;
    if all_messages_tokens <= window_tokens * NATIVE_ATTEMPT_GUARD_FRACTION {
        if cancel.is_cancelled() {
            anyhow::bail!("compaction cancelled");
        }
        let mut native_messages = all_messages.to_vec();
        native_messages.push(ChatMessage::user(native_snapshot_instruction()));
        match run_compaction_request(
            llm,
            model,
            native_messages,
            tools.map(<[_]>::to_vec),
            reasoning_effort.clone(),
            idle_timeout,
            cancel.clone(),
        )
        .await
        {
            Ok((snapshot, call_usage)) => {
                usage.add(call_usage);
                match build_checkpoint(&snapshot, &digests, &pins, exact_tail, before_tokens) {
                    Ok((checkpoint_messages, after_tokens)) => {
                        return Ok(HistoryCompaction {
                            checkpoint_messages,
                            usage,
                            before_tokens,
                            after_tokens,
                        });
                    }
                    Err(error) => {
                        tracing::warn!(
                            "native in-conversation compaction produced an unusable checkpoint; \
                             falling back to rendered summarization: {error:#}"
                        );
                    }
                }
            }
            Err(error) => {
                if cancel.is_cancelled() {
                    return Err(error);
                }
                tracing::warn!(
                    "native in-conversation compaction failed; falling back to rendered \
                     summarization: {error:#}"
                );
            }
        }
    }

    // Fallback: flatten `history` to plain text and summarize it in a fresh
    // request, splitting hierarchically when even the flattened text
    // doesn't fit. Unchanged from the pre-native-path behavior; also used
    // directly (never via the native attempt above) by the turn-recap
    // feature's chunking helpers below.
    let mut body = render_history_for_compaction(history_to_summarize);

    loop {
        let final_messages = vec![
            ChatMessage::system(history_snapshot_prompt()),
            ChatMessage::user(format!("History to compact:\n\n{body}")),
        ];
        if approximate_tokens_messages(&final_messages) <= budget {
            let (snapshot, call_usage) = run_compaction_request(
                llm,
                model,
                final_messages,
                None,
                reasoning_effort.clone(),
                idle_timeout,
                cancel.clone(),
            )
            .await?;
            usage.add(call_usage);
            let (checkpoint_messages, after_tokens) =
                build_checkpoint(&snapshot, &digests, &pins, exact_tail, before_tokens)?;
            return Ok(HistoryCompaction {
                checkpoint_messages,
                usage,
                before_tokens,
                after_tokens,
            });
        }

        let chunks = split_plain_text_to_chunks(&body, budget);
        if chunks.len() <= 1 {
            anyhow::bail!("history exceeds compaction budget and cannot be split further");
        }
        let prior_body_tokens = approximate_tokens(&body);
        let mut summaries = Vec::with_capacity(chunks.len());
        for (index, chunk) in chunks.iter().enumerate() {
            if cancel.is_cancelled() {
                anyhow::bail!("compaction cancelled");
            }
            let messages = vec![
                ChatMessage::system(HISTORY_CHUNK_PROMPT),
                ChatMessage::user(format!(
                    "History fragment {} of {}:\n\n{}",
                    index + 1,
                    chunks.len(),
                    chunk
                )),
            ];
            let (summary, call_usage) = run_compaction_request(
                llm,
                model,
                messages,
                None,
                reasoning_effort.clone(),
                idle_timeout,
                cancel.clone(),
            )
            .await?;
            usage.add(call_usage);
            summaries.push(summary);
        }
        let reduced = format_chunk_summaries(&summaries);
        let reduced_tokens = approximate_tokens(&reduced);
        if reduced_tokens >= prior_body_tokens {
            anyhow::bail!(
                "hierarchical compaction made no progress (~{prior_body_tokens} -> ~{reduced_tokens} tokens)"
            );
        }
        body = reduced;
    }
}

/// Normalize the compactor's raw response into checkpoint messages and
/// verify the checkpoint actually shrank history. Shared by the native and
/// rendered-fallback paths so both apply the exact same
/// preamble/digest/plan/tail assembly and the same not-reduced check; the
/// native path treats any `Err` here as a fallback trigger, while the
/// rendered path (the last resort) propagates it as a hard error.
fn build_checkpoint(
    raw_snapshot: &str,
    digests: &str,
    pins: &HistoryPins<'_>,
    exact_tail: &[ChatMessage],
    before_tokens: usize,
) -> Result<(Vec<ChatMessage>, usize)> {
    let snapshot = normalize_state_snapshot(raw_snapshot)?;
    let mut checkpoint_messages = Vec::new();
    // The active user request is replayed verbatim ahead of the snapshot so
    // an exact contract in it (an output schema, for example) survives a
    // compaction that happens in the middle of the turn serving it.
    if let Some(active_user_message) = pins.active_user_message {
        checkpoint_messages.push(active_user_message.clone());
    }
    checkpoint_messages.push(ChatMessage::user(format!(
        "{CHECKPOINT_PREAMBLE}\n\n{snapshot}{digests}"
    )));
    if let Some(plan) = pins.current_plan {
        checkpoint_messages.push(ChatMessage::user(format!(
            "<current_plan>\n{}\n</current_plan>",
            serde_json::to_string_pretty(plan)?
        )));
    }
    checkpoint_messages.extend_from_slice(exact_tail);
    let after_tokens = approximate_tokens_messages(&checkpoint_messages);
    if after_tokens >= before_tokens {
        anyhow::bail!(
            "compaction did not reduce history (~{before_tokens} -> ~{after_tokens} tokens)"
        );
    }
    Ok((checkpoint_messages, after_tokens))
}

// ---------------------------------------------------------------------------
// Checkpoint digests
// ---------------------------------------------------------------------------
//
// Deterministic (non-LLM) summaries appended after the compactor's prose
// snapshot. The model-written snapshot can lose or blur exact tool-call
// outcomes when it summarizes; these digests are computed straight from the
// full history so a restarted agent has an unambiguous record of what not
// to repeat.

/// Build the digest suffix appended after the state snapshot in the first
/// checkpoint message: `<files_already_read>` then `<failed_tool_calls>`,
/// each preceded by a blank line, in that order. Empty string when neither
/// digest has anything to report.
fn build_digests(history: &[ChatMessage]) -> String {
    let mut out = String::new();
    if let Some(files) = files_already_read_digest(history) {
        out.push_str("\n\n");
        out.push_str(&files);
    }
    if let Some(failed) = failed_tool_calls_digest(history) {
        out.push_str("\n\n");
        out.push_str(&failed);
    }
    out
}

/// Cap on distinct paths listed in the `<files_already_read>` digest. Past
/// this, later paths are dropped in favor of one summary line -- a session
/// that read hundreds of files has already blown well past what's useful to
/// enumerate.
const MAX_DIGEST_PATHS: usize = 100;

/// How many of the most recent failed tool calls the `<failed_tool_calls>`
/// digest keeps in full. Older failures are dropped with a count note --
/// only the failures the next agent is likely to still be relevant to
/// (i.e. recent ones) are worth the tokens.
const MAX_DIGEST_FAILURES: usize = 10;

/// One `read_file` call's `(offset, limit)` pair, in the tool's raw
/// 0-based-offset / max-line-count schema.
type ReadRange = (Option<usize>, Option<usize>);

/// Scan `history` for `read_file` tool calls and render one line per
/// distinct path (first-seen order): just the path if any call read the
/// whole file, otherwise the merged 1-based inclusive line ranges actually
/// read (`src/foo.rs (lines 1-200, 400-450)`). Calls with unparseable
/// arguments or a missing/non-string `file_path` are ignored. `None` when no
/// `read_file` calls are present.
fn files_already_read_digest(history: &[ChatMessage]) -> Option<String> {
    let mut order: Vec<String> = Vec::new();
    let mut reads: HashMap<String, Vec<ReadRange>> = HashMap::new();

    for message in history {
        if message.role != "assistant" {
            continue;
        }
        let Some(calls) = &message.tool_calls else {
            continue;
        };
        for call in calls {
            if call.function.name != "read_file" {
                continue;
            }
            let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
            else {
                continue;
            };
            let Some(file_path) = args.get("file_path").and_then(|v| v.as_str()) else {
                continue;
            };
            let offset = args
                .get("offset")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            reads
                .entry(file_path.to_string())
                .or_insert_with(|| {
                    order.push(file_path.to_string());
                    Vec::new()
                })
                .push((offset, limit));
        }
    }

    if order.is_empty() {
        return None;
    }

    let total_paths = order.len();
    let mut lines: Vec<String> = Vec::new();
    for path in order.iter().take(MAX_DIGEST_PATHS) {
        let calls = &reads[path];
        let whole_file = calls
            .iter()
            .any(|(offset, limit)| offset.is_none() && limit.is_none());
        if whole_file {
            lines.push(path.clone());
            continue;
        }
        lines.push(format!("{path} ({})", render_read_ranges(calls)));
    }
    if total_paths > MAX_DIGEST_PATHS {
        lines.push(format!("(+{} more files)", total_paths - MAX_DIGEST_PATHS));
    }

    Some(format!(
        "<files_already_read>\n{}\n</files_already_read>",
        lines.join("\n")
    ))
}

/// Merge one path's `(offset, limit)` reads into "lines A-B, C-D" text.
/// `offset` is the tool's 0-based start line; `limit` is the max line
/// count, absent meaning "to end of file". Overlapping/adjacent ranges are
/// merged; an unbounded (to-end-of-file) range renders its upper bound as
/// `end` and absorbs any range that starts after it.
fn render_read_ranges(calls: &[ReadRange]) -> String {
    let mut ranges: Vec<(usize, usize)> = calls
        .iter()
        .map(|(offset, limit)| {
            let start = offset.unwrap_or(0) + 1; // 0-based -> 1-based
            let end = limit
                .map(|limit| start.saturating_add(limit.saturating_sub(1)))
                .unwrap_or(usize::MAX);
            (start, end)
        })
        .collect();
    ranges.sort_unstable_by_key(|&(start, _)| start);

    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges.drain(..) {
        match merged.last_mut() {
            Some(last) if start <= last.1.saturating_add(1) => {
                last.1 = last.1.max(end);
            }
            _ => merged.push((start, end)),
        }
    }

    let rendered = merged
        .iter()
        .map(|&(start, end)| {
            if end == usize::MAX {
                format!("{start}-end")
            } else {
                format!("{start}-{end}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("lines {rendered}")
}

/// Scan `history` for failed tool results (per
/// `crate::tools::tool_result_failed`) and render the most recent
/// [`MAX_DIGEST_FAILURES`] as a numbered list: `` N. `tool` args=... -> result
/// first line ``. Arguments and the result's first line are truncated
/// (char-boundary-safe) to keep the digest bounded. `None` when no tool
/// result failed.
fn failed_tool_calls_digest(history: &[ChatMessage]) -> Option<String> {
    const MAX_ARGS_CHARS: usize = 500;
    const MAX_RESULT_CHARS: usize = 200;

    let mut calls: HashMap<&str, (&str, &str)> = HashMap::new();
    for message in history {
        if message.role != "assistant" {
            continue;
        }
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for call in tool_calls {
            calls.insert(
                call.id.as_str(),
                (
                    call.function.name.as_str(),
                    call.function.arguments.as_str(),
                ),
            );
        }
    }

    struct Failure {
        tool_name: String,
        arguments: String,
        result_first_line: String,
    }

    let mut failures: Vec<Failure> = Vec::new();
    for message in history {
        if message.role != "tool" {
            continue;
        }
        let Some(call_id) = &message.tool_call_id else {
            continue;
        };
        let text = message.text_content().unwrap_or("");
        if !crate::tools::tool_result_failed(text) {
            continue;
        }
        let (tool_name, arguments) = calls
            .get(call_id.as_str())
            .map(|&(name, args)| (name.to_string(), args.to_string()))
            .unwrap_or_else(|| {
                (
                    message
                        .name
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                    String::new(),
                )
            });
        failures.push(Failure {
            tool_name,
            arguments,
            result_first_line: text.lines().next().unwrap_or("").to_string(),
        });
    }

    if failures.is_empty() {
        return None;
    }

    let total = failures.len();
    let kept: &[Failure] = if total > MAX_DIGEST_FAILURES {
        &failures[total - MAX_DIGEST_FAILURES..]
    } else {
        &failures[..]
    };

    let mut lines: Vec<String> = Vec::new();
    if total > MAX_DIGEST_FAILURES {
        lines.push(format!(
            "({} earlier failed calls omitted)",
            total - MAX_DIGEST_FAILURES
        ));
    }
    for (index, failure) in kept.iter().enumerate() {
        lines.push(format!(
            "{}. `{}` args={} -> {}",
            index + 1,
            failure.tool_name,
            truncate_chars(&failure.arguments, MAX_ARGS_CHARS),
            truncate_chars(&failure.result_first_line, MAX_RESULT_CHARS),
        ));
    }

    Some(format!(
        "<failed_tool_calls>\n{}\n</failed_tool_calls>",
        lines.join("\n")
    ))
}

/// Truncate `s` to at most `max` chars (char-boundary-safe, unlike a byte
/// slice), appending `…` when truncation actually happened.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut truncated: String = s.chars().take(max).collect();
    truncated.push('…');
    truncated
}

/// Retain a recent provider-neutral tail verbatim while summarizing the older
/// prefix. The budget is token-based rather than message-count based, and the
/// boundary backs up over tool results to keep an assistant tool-call batch
/// paired with all of its results.
fn exact_tail_start(history: &[ChatMessage], context_length: Option<u32>) -> usize {
    let tail_budget = (context_length.unwrap_or(FALLBACK_CONTEXT_LENGTH) as usize / 10).max(1_000);
    let mut start = history.len();
    let mut tokens = 0usize;
    while start > 0 {
        let next = approximate_tokens_messages(&history[start - 1..start]);
        if tokens > 0 && tokens.saturating_add(next) > tail_budget {
            break;
        }
        start -= 1;
        tokens = tokens.saturating_add(next);
    }
    if start == 0 {
        return history.len();
    }
    while start > 0 && history[start].role == "tool" {
        start -= 1;
    }
    start
}

fn render_history_for_compaction(history: &[ChatMessage]) -> String {
    let mut rendered = String::new();
    for (index, message) in history.iter().enumerate() {
        rendered.push_str(&format!("Message {} [{}]:\n", index + 1, message.role));
        for part in &message.content {
            match part {
                ChatContentPart::Text { text } => rendered.push_str(text),
                ChatContentPart::Image { .. } => rendered.push_str("[image omitted]"),
            }
            rendered.push('\n');
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                rendered.push_str(&format!(
                    "Tool call {} `{}` args={}\n",
                    call.id, call.function.name, call.function.arguments
                ));
            }
        }
        if let Some(call_id) = &message.tool_call_id {
            rendered.push_str(&format!("Tool result for {call_id}"));
            if let Some(name) = &message.name {
                rendered.push_str(&format!(" (`{name}`)"));
            }
            rendered.push('\n');
        }
        rendered.push('\n');
    }
    rendered
}

async fn run_compaction_request(
    llm: &dyn LlmBackend,
    model: &str,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<ToolDefinition>>,
    reasoning_effort: Option<String>,
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
) -> Result<(String, TokenUsage)> {
    let response = stream_chat_no_visible_output_with_retry(
        llm,
        "compacting conversation history",
        &cancel,
        || StreamChatRequest {
            model: model.to_string(),
            messages: messages.clone(),
            tools: tools.clone(),
            reasoning_effort: reasoning_effort.clone(),
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: cancel.clone(),
            idle_timeouts: idle_timeout,
        },
    )
    .await?;
    let usage = response.usage();
    let text = match response {
        LlmResponse::Text { text, .. } | LlmResponse::ToolCalls { text, .. } => text,
    };
    Ok((text, usage))
}

fn normalize_state_snapshot(text: &str) -> Result<String> {
    let without_analysis = strip_xml_blocks(text, "analysis");
    let trimmed = without_analysis.trim();
    let start = trimmed.find("<state_snapshot>");
    let end = trimmed.rfind("</state_snapshot>");
    match (start, end) {
        (Some(start), Some(end)) if end >= start => {
            let end = end + "</state_snapshot>".len();
            Ok(trimmed[start..end].to_string())
        }
        _ if !trimmed.is_empty() => Ok(format!("<state_snapshot>\n{}\n</state_snapshot>", trimmed)),
        _ => anyhow::bail!("compactor returned an empty state snapshot"),
    }
}

fn strip_xml_blocks(input: &str, tag: &str) -> String {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut output = input.to_string();
    while let Some(start) = output.find(&open) {
        let end = output[start + open.len()..]
            .find(&close)
            .map(|offset| start + open.len() + offset + close.len())
            .unwrap_or(output.len());
        output.replace_range(start..end, "");
    }
    output
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
#[cfg(test)]
pub async fn summarize_turn(
    llm: &dyn LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    context_length: Option<u32>,
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
) -> Result<String> {
    summarize_turn_styled(
        llm,
        model,
        turn,
        context_length,
        idle_timeout,
        cancel,
        SummaryStyle::Compression,
    )
    .await
}

/// Produce a short, user-facing recap summary of one turn -- "what the
/// assistant did this turn", in prose, for the host's turn recap. Same
/// hierarchical fallback as [`summarize_turn`]; only the framing prompts
/// differ. The caller treats failure as non-fatal (the recap still shows
/// its deterministic stats), so errors propagate for the caller to ignore.
pub async fn summarize_turn_for_recap(
    llm: &dyn LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    context_length: Option<u32>,
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
) -> Result<String> {
    summarize_turn_styled(
        llm,
        model,
        turn,
        context_length,
        idle_timeout,
        cancel,
        SummaryStyle::Recap,
    )
    .await
}

async fn summarize_turn_styled(
    llm: &dyn LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    context_length: Option<u32>,
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
    style: SummaryStyle,
) -> Result<String> {
    let budget = summarizer_input_budget(context_length);

    // Fast path: the whole turn fits in one summarization call.
    let single_call_messages = build_turn_summarization_messages_styled(turn, style);
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
    summarize_turn_hierarchical(llm, model, turn, budget, idle_timeout, cancel, style).await
}

async fn summarize_turn_hierarchical(
    llm: &dyn LlmBackend,
    model: &str,
    turn: &ConversationTurn,
    budget: usize,
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
    style: SummaryStyle,
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
    combine_chunk_summaries(
        llm,
        model,
        &chunk_summaries,
        budget,
        idle_timeout,
        cancel,
        style,
    )
    .await
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
    idle_timeout: IdleTimeouts,
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
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
    style: SummaryStyle,
) -> Result<String> {
    let combined = format_chunk_summaries(chunk_summaries);
    let messages = build_meta_summarization_messages(&combined, style);
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
        style,
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
    idle_timeout: IdleTimeouts,
    cancel: CancellationToken,
) -> Result<String> {
    let response = stream_chat_no_visible_output_with_retry(
        llm,
        "summarizing conversation turn",
        &cancel,
        || StreamChatRequest {
            model: model.to_string(),
            messages: messages.clone(),
            tools: None,
            // Summarization is structured extraction, not deep
            // reasoning -- "low" keeps cost down on reasoning-capable
            // models that bill thinking tokens.
            reasoning_effort: Some("low".to_string()),
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: cancel.clone(),
            idle_timeouts: idle_timeout,
        },
    )
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
#[cfg(test)]
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
#[cfg(test)]
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

/// System prompt for a user-facing turn *recap* summary. Unlike
/// `SYSTEM_PROMPT_TURN` (whose output replaces a turn in the model's
/// working context and is deliberately not prose), this output is shown
/// to the human at the end of a turn, so it describes what the assistant
/// just did in plain language.
const SYSTEM_PROMPT_TURN_RECAP: &str = "You are writing a short recap for the user of an AI coding \
assistant. You will receive one turn -- the user's last message, the \
assistant's response, and any tool calls and results. Summarize, for the \
user, what the assistant actually did this turn.\n\
\n\
Write at most two concise Markdown bullet points (one line each), in plain \
past tense. Only mention facts directly supported by the turn:\n\
- Concrete work actually performed: files created or edited, commands run, \
  things investigated or decisions made.\n\
- Notable results, errors, test outcomes, or unfinished work.\n\
\n\
Do not infer fixes, changes, tests, or conclusions that are not explicit in \
the turn. Do not restate the user's request, add preamble or a closing remark, \
or wrap the output in any tags or headings. If no substantive work happened, \
reply with one short bullet saying so.";

/// Meta-join prompt for the recap style: combine per-part summaries of an
/// oversized turn into one short user-facing recap.
const SYSTEM_PROMPT_META_RECAP: &str = "You will receive several bulleted summaries, each covering \
part of a single assistant turn that was too large to summarize at once. \
Combine them into ONE short user-facing recap of what the assistant did \
this turn.\n\
\n\
- Keep only concrete work, results, errors, and unfinished items explicit in \
  the part summaries.\n\
- Deduplicate facts that appear in multiple parts.\n\
- Preserve the order in which the work happened.\n\
- Keep it very short: at most two Markdown bullet points, one line each.\n\
\n\
Do not infer fixes or outcomes that are not explicit. Output only the bullet \
points -- no preamble, no closing remark, no heading, and no surrounding tags.";

/// Which framing the turn and meta summarization prompts use. The chunk
/// (extraction) prompt is shared across styles -- only the single-call
/// turn prompt and the meta-join prompt differ between a context-
/// compression summary (operational facts for the model) and a
/// user-facing recap (prose for the human).
#[derive(Clone, Copy)]
enum SummaryStyle {
    #[cfg(test)]
    Compression,
    Recap,
}

impl SummaryStyle {
    fn turn_prompt(self) -> &'static str {
        match self {
            #[cfg(test)]
            SummaryStyle::Compression => SYSTEM_PROMPT_TURN,
            SummaryStyle::Recap => SYSTEM_PROMPT_TURN_RECAP,
        }
    }

    fn meta_prompt(self) -> &'static str {
        match self {
            #[cfg(test)]
            SummaryStyle::Compression => SYSTEM_PROMPT_META,
            SummaryStyle::Recap => SYSTEM_PROMPT_META_RECAP,
        }
    }
}

/// Build the chat-message list for a single-call (whole-turn)
/// compression summarization. Test-only convenience wrapper; production
/// goes through the styled variant below.
#[cfg(test)]
fn build_turn_summarization_messages(turn: &ConversationTurn) -> Vec<ChatMessage> {
    build_turn_summarization_messages_styled(turn, SummaryStyle::Compression)
}

fn build_turn_summarization_messages_styled(
    turn: &ConversationTurn,
    style: SummaryStyle,
) -> Vec<ChatMessage> {
    let mut body = String::from("Turn to summarize:\n\n");
    for atom in turn_summary_atoms(turn) {
        body.push_str(&atom);
        body.push('\n');
    }
    vec![
        ChatMessage::system(style.turn_prompt()),
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
fn build_meta_summarization_messages(combined: &str, style: SummaryStyle) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(style.meta_prompt()),
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
    let atoms = turn_summary_atoms(turn);

    pack_atoms_into_chunks(atoms, per_chunk_budget)
}

fn turn_summary_atoms(turn: &ConversationTurn) -> Vec<String> {
    let mut atoms: Vec<String> = Vec::new();
    if !turn.user_prompt.trim().is_empty() {
        atoms.push(format!("User: {}", turn.user_prompt.trim()));
    }

    let replay_events = crate::session::sanitize_replay_events(&turn.replay_events);
    if !replay_events.is_empty() {
        for event in &replay_events {
            match event {
                crate::session::TurnReplayEvent::AssistantToolCalls { text, calls } => {
                    if !text.trim().is_empty() {
                        atoms.push(format!("Assistant: {}", text.trim()));
                    }
                    for call in calls {
                        atoms.push(format!(
                            "Tool call `{}` args={}",
                            call.tool_name, call.arguments
                        ));
                    }
                }
                crate::session::TurnReplayEvent::ToolResult(exchange) => {
                    atoms.push(format!(
                        "Tool result `{}` -> {}",
                        exchange.tool_name, exchange.result
                    ));
                }
                crate::session::TurnReplayEvent::AssistantText { text } => {
                    if !text.trim().is_empty() {
                        atoms.push(format!("Assistant: {}", text.trim()));
                    }
                }
            }
        }
    } else {
        for exchange in &turn.tool_exchanges {
            atoms.push(format!(
                "Tool `{}` args={} -> {}",
                exchange.tool_name, exchange.arguments, exchange.result
            ));
        }
    }

    let assistant_text = crate::host_notice::model_visible_assistant_text(&turn.agent_response);
    if replay_events.is_empty() && !assistant_text.trim().is_empty() {
        atoms.push(format!("Assistant: {}", assistant_text.trim()));
    }
    atoms
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use futures::future::BoxFuture;

    use super::*;
    use crate::session::ToolExchange;

    fn turn(user: &str, agent: &str) -> ConversationTurn {
        ConversationTurn {
            user_prompt: user.to_string(),
            agent_response: agent.to_string(),
            replay_events: Vec::new(),
            tool_exchanges: Vec::new(),
            structured_output: None,
            summary: None,
            current_plan: None,
            compaction_checkpoint: None,
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
            replay_events: Vec::new(),
            tool_exchanges: vec![ToolExchange {
                call_id: "call_1".to_string(),
                tool_name: tool.to_string(),
                arguments: args.to_string(),
                result: result.to_string(),
                ..ToolExchange::default()
            }],
            structured_output: None,
            summary: None,
            current_plan: None,
            compaction_checkpoint: None,
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
    fn build_turn_summarization_messages_uses_ordered_replay_events() {
        use crate::session::{ToolCallReplay, TurnReplayEvent};

        let mut t = turn("find TODOs", "aggregate text should not be appended");
        t.replay_events = vec![
            TurnReplayEvent::AssistantToolCalls {
                text: "I will search.".into(),
                calls: vec![ToolCallReplay {
                    call_id: "c1".into(),
                    tool_name: "grep_search".into(),
                    arguments: r#"{"pattern":"TODO"}"#.into(),
                }],
            },
            TurnReplayEvent::ToolResult(ToolExchange {
                call_id: "c1".into(),
                tool_name: "grep_search".into(),
                arguments: r#"{"pattern":"TODO"}"#.into(),
                result: "src/lib.rs:42".into(),
                ..ToolExchange::default()
            }),
            TurnReplayEvent::AssistantText {
                text: "Done.".into(),
            },
        ];

        let msgs = build_turn_summarization_messages(&t);
        let body = msgs[1].text_content().unwrap();
        let search_pos = body.find("Assistant: I will search.").unwrap();
        let call_pos = body.find("Tool call `grep_search`").unwrap();
        let result_pos = body.find("Tool result `grep_search`").unwrap();
        let done_pos = body.find("Assistant: Done.").unwrap();
        assert!(search_pos < call_pos);
        assert!(call_pos < result_pos);
        assert!(result_pos < done_pos);
        assert!(!body.contains("aggregate text should not be appended"));
    }

    #[test]
    fn build_turn_summarization_messages_strips_host_notices_from_text_only_turn() {
        let recap = crate::host_notice::render_turn_recap(
            Some("- Investigated the foo path and edited `bar.rs`."),
            &[],
            None,
            &crate::tool_loop::LoopStop::Completed { had_text: true },
        );
        let t = turn("what happened?", &format!("The model answer.{recap}"));
        let msgs = build_turn_summarization_messages(&t);
        let body = msgs[1].text_content().unwrap();
        assert!(body.contains("Assistant: The model answer."));
        assert!(!body.contains("Anvil Recap"));
        assert!(!body.contains("Files changed"));
        // The host-written work summary must not leak into the model's history.
        assert!(!body.contains("Investigated the foo path"));
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

    #[test]
    fn state_snapshot_normalization_removes_analysis_scratch() {
        let normalized = normalize_state_snapshot(
            "<analysis>private scratch</analysis>\n<state_snapshot>kept</state_snapshot>",
        )
        .unwrap();
        assert_eq!(normalized, "<state_snapshot>kept</state_snapshot>");
        assert!(!normalized.contains("private scratch"));
    }

    #[test]
    fn compaction_render_omits_reasoning_and_image_payloads() {
        let mut message = ChatMessage::assistant_with_reasoning(
            "visible evidence",
            Some("private reasoning".to_string()),
        );
        message.content.push(ChatContentPart::image_data(
            "very-large-secret-base64",
            "image/png",
        ));
        let rendered = render_history_for_compaction(&[message]);
        assert!(rendered.contains("visible evidence"));
        assert!(rendered.contains("[image omitted]"));
        assert!(!rendered.contains("private reasoning"));
        assert!(!rendered.contains("very-large-secret-base64"));
    }

    #[test]
    fn exact_tail_keeps_tool_call_and_result_together() {
        use anvil_llm::llm_client::{FunctionCall, ToolCall};

        let messages = vec![
            ChatMessage::user("old".repeat(8_000)),
            ChatMessage::assistant_tool_calls(vec![ToolCall {
                id: "c1".into(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: r#"{"path":"src/lib.rs"}"#.into(),
                },
            }]),
            ChatMessage::tool_result("c1", "read_file", "result"),
        ];
        let start = exact_tail_start(&messages, Some(8_000));
        assert_eq!(start, 1);
        assert_eq!(messages[start].role, "assistant");
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
                ..ToolExchange::default()
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
                    reasoning_content: None,
                    usage: anvil_llm::llm_client::TokenUsage::default(),
                    codex_reasoning: None,
                })
            }
            .boxed()
        }
    }

    struct IncompleteThenSummaryBackend {
        attempts: Arc<AtomicUsize>,
    }

    impl LlmBackend for IncompleteThenSummaryBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            async { Ok(vec!["mock".into()]) }.boxed()
        }

        fn stream_chat(&self, _request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            let attempts = self.attempts.clone();
            async move {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt == 1 {
                    return Err(anyhow::Error::new(
                        anvil_llm::llm_client::IncompleteStreamError::new(
                            "test SSE",
                            "response.completed",
                        ),
                    ));
                }
                Ok(LlmResponse::Text {
                    text: "<conversation_summary>\n- recovered\n</conversation_summary>"
                        .to_string(),
                    reasoning_content: None,
                    usage: anvil_llm::llm_client::TokenUsage::default(),
                    codex_reasoning: None,
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
            IdleTimeouts::uniform(Duration::from_secs(60)),
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

    #[tokio::test]
    async fn compact_history_emits_snapshot_then_exact_tail() {
        use anvil_llm::llm_client::{FunctionCall, ToolCall};

        let (backend, _, seen) = ScriptedBackend::new(
            "<state_snapshot><pending_tasks>finish parser</pending_tasks></state_snapshot>",
            "- extracted history",
            "- combined history",
        );
        let history = vec![
            ChatMessage::user("older evidence ".repeat(6_000)),
            ChatMessage::user("Return only {\"files\": []}"),
            ChatMessage::assistant_tool_calls(vec![ToolCall {
                id: "c1".into(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: r#"{"file_path":"src/lib.rs"}"#.into(),
                },
            }]),
            ChatMessage::tool_result("c1", "read_file", "exact recent result"),
        ];
        let compacted = compact_history(
            &backend,
            "mock",
            &history,
            0,
            None,
            HistoryPins {
                current_plan: None,
                active_user_message: Some(&history[1]),
            },
            None,
            Some(8_000),
            IdleTimeouts::uniform(Duration::from_secs(60)),
            CancellationToken::new(),
        )
        .await
        .expect("compaction succeeds");

        assert_eq!(
            compacted.checkpoint_messages[0].text_content(),
            Some("Return only {\"files\": []}")
        );
        let checkpoint = compacted.checkpoint_messages[1].text_content().unwrap();
        assert!(checkpoint.starts_with(CHECKPOINT_PREAMBLE));
        assert!(checkpoint.contains("<state_snapshot>"));
        assert!(checkpoint.contains("<files_already_read>"));
        assert_eq!(compacted.checkpoint_messages[2].role, "assistant");
        assert_eq!(compacted.checkpoint_messages[3].role, "tool");
        assert_eq!(
            compacted.checkpoint_messages[3].text_content(),
            Some("exact recent result")
        );
        assert!(
            seen.lock()
                .unwrap()
                .iter()
                .all(|prompt| !prompt.contains("{\"files\": []}")),
            "the pinned user contract must not enter the generated snapshot"
        );
        assert!(compacted.after_tokens < compacted.before_tokens);
    }

    /// Backend for exercising the native in-conversation compaction path.
    /// `ScriptedBackend` dispatches on the FIRST message, which works for
    /// the rendered fallback (a fresh `[system, user]` request) but not for
    /// the native attempt, whose first message is the conversation's own
    /// system message. This dispatches on whether the LAST message carries
    /// the native instruction's marker text instead, and records every
    /// request's messages/tools for inspection.
    type CapturedCompactionRequest = (
        Vec<ChatMessage>,
        Option<Vec<anvil_llm::llm_client::ToolDefinition>>,
    );

    struct NativeAwareBackend {
        native_fails: bool,
        native_response: String,
        fallback_response: String,
        chunk_response: String,
        captured: Arc<Mutex<Vec<CapturedCompactionRequest>>>,
    }

    impl NativeAwareBackend {
        fn new(native_fails: bool, native_response: &str, fallback_response: &str) -> Self {
            Self {
                native_fails,
                native_response: native_response.to_string(),
                fallback_response: fallback_response.to_string(),
                chunk_response: "- extracted history".to_string(),
                captured: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn is_native_request(messages: &[ChatMessage]) -> bool {
            messages
                .last()
                .and_then(|m| m.text_content())
                .is_some_and(|text| text.contains(NATIVE_SNAPSHOT_INSTRUCTION_PREFIX))
        }
    }

    impl LlmBackend for NativeAwareBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            async { Ok(vec!["mock".into()]) }.boxed()
        }
        fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            self.captured
                .lock()
                .unwrap()
                .push((request.messages.clone(), request.tools.clone()));
            if Self::is_native_request(&request.messages) {
                if self.native_fails {
                    return async {
                        Err(anyhow::anyhow!("native compaction intentionally failed"))
                    }
                    .boxed();
                }
                let response = self.native_response.clone();
                return async move {
                    Ok(LlmResponse::Text {
                        text: response,
                        reasoning_content: None,
                        usage: anvil_llm::llm_client::TokenUsage::default(),
                        codex_reasoning: None,
                    })
                }
                .boxed();
            }
            let system = request
                .messages
                .first()
                .and_then(|m| m.text_content())
                .unwrap_or("");
            // `HISTORY_CHUNK_PROMPT` (extraction pass) vs. the final
            // rendered-snapshot request built from `history_snapshot_prompt()`
            // -- compact_history's fallback has no separate meta/combine
            // stage, unlike turn summarization.
            let response = if system.contains("Extract durable working-memory facts") {
                self.chunk_response.clone()
            } else {
                self.fallback_response.clone()
            };
            async move {
                Ok(LlmResponse::Text {
                    text: response,
                    reasoning_content: None,
                    usage: anvil_llm::llm_client::TokenUsage::default(),
                    codex_reasoning: None,
                })
            }
            .boxed()
        }
    }

    fn read_file_tool_definition() -> anvil_llm::llm_client::ToolDefinition {
        anvil_llm::llm_client::ToolDefinition {
            r#type: "function".to_string(),
            function: anvil_llm::llm_client::FunctionDef {
                name: "read_file".to_string(),
                description: "Read a file.".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
        }
    }

    /// The native attempt sends the conversation's own native messages
    /// (assistant `tool_calls` included, un-flattened), the trailing
    /// instruction, and the advertised tools -- exactly what Phase 2
    /// promises over the old flatten-to-text request.
    #[tokio::test]
    async fn compact_history_native_path_sends_native_messages_and_tools() {
        use anvil_llm::llm_client::{FunctionCall, ToolCall};

        let backend = NativeAwareBackend::new(
            false,
            "<state_snapshot><pending_tasks>done</pending_tasks></state_snapshot>",
            "unused",
        );
        let captured = backend.captured.clone();
        let all_messages = vec![
            ChatMessage::system("canonical system prompt"),
            ChatMessage::user("please read the file and summarize its contents".repeat(50)),
            ChatMessage::assistant_tool_calls(vec![ToolCall {
                id: "c1".into(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: r#"{"file_path":"src/lib.rs"}"#.into(),
                },
            }]),
            ChatMessage::tool_result("c1", "read_file", "file contents ".repeat(200)),
        ];
        let tools = vec![read_file_tool_definition()];
        let compacted = compact_history(
            &backend,
            "mock",
            &all_messages,
            1,
            Some(&tools),
            HistoryPins::default(),
            None,
            Some(200_000),
            IdleTimeouts::uniform(Duration::from_secs(60)),
            CancellationToken::new(),
        )
        .await
        .expect("compaction succeeds");
        assert!(compacted.after_tokens < compacted.before_tokens);

        let requests = captured.lock().unwrap();
        assert_eq!(
            requests.len(),
            1,
            "native attempt should succeed on the first call"
        );
        let (messages, request_tools) = &requests[0];

        // The original native messages -- including the un-flattened
        // assistant `tool_calls` message -- are present verbatim.
        assert_eq!(messages[0].text_content(), Some("canonical system prompt"));
        assert!(
            messages[1]
                .text_content()
                .unwrap()
                .starts_with("please read the file")
        );
        assert!(
            messages[2].tool_calls.is_some(),
            "assistant tool_calls must survive un-flattened"
        );
        assert_eq!(
            messages[2].tool_calls.as_ref().unwrap()[0].function.name,
            "read_file"
        );
        assert_eq!(messages[3].role, "tool");

        // Trailing instruction asks for a state snapshot in text only.
        let last = messages.last().unwrap().text_content().unwrap();
        assert!(last.contains("state_snapshot"));
        assert!(last.contains("Do NOT call any tools"));

        // Advertised tool schemas are passed through.
        let request_tools = request_tools.as_ref().expect("tools should be forwarded");
        assert!(request_tools.iter().any(|t| t.function.name == "read_file"));
    }

    /// When the native attempt fails, compaction falls back to the
    /// flatten+chunk rendered path and still produces a valid checkpoint.
    #[tokio::test]
    async fn compact_history_falls_back_when_native_attempt_fails() {
        let backend = NativeAwareBackend::new(
            true,
            "unused",
            "<state_snapshot><pending_tasks>fallback done</pending_tasks></state_snapshot>",
        );
        let captured = backend.captured.clone();
        let all_messages = vec![
            ChatMessage::system("canonical system prompt"),
            ChatMessage::user("old context ".repeat(2_000)),
            ChatMessage::assistant("acknowledged"),
        ];
        let compacted = compact_history(
            &backend,
            "mock",
            &all_messages,
            1,
            None,
            HistoryPins::default(),
            None,
            Some(200_000),
            IdleTimeouts::uniform(Duration::from_secs(60)),
            CancellationToken::new(),
        )
        .await
        .expect("compaction succeeds via fallback");

        let first_message = compacted.checkpoint_messages[0].text_content().unwrap();
        assert!(first_message.starts_with(CHECKPOINT_PREAMBLE));
        assert!(first_message.contains("<state_snapshot>"));
        assert!(first_message.contains("fallback done"));

        let requests = captured.lock().unwrap();
        assert!(
            requests.len() >= 2,
            "expected a failed native call plus at least one fallback call, got {}",
            requests.len()
        );
        assert!(
            NativeAwareBackend::is_native_request(&requests[0].0),
            "first call should be the native attempt"
        );
        assert!(
            !NativeAwareBackend::is_native_request(&requests[1].0),
            "second call should be the rendered fallback, not another native attempt"
        );
    }

    /// When the raw conversation is already too close to the declared
    /// context window, the native attempt is skipped entirely -- the
    /// backend never sees the native instruction, and the very first
    /// request is already the rendered fallback.
    #[tokio::test]
    async fn compact_history_guard_skips_native_when_conversation_too_large() {
        let backend = NativeAwareBackend::new(
            false,
            "unused",
            "<state_snapshot><pending_tasks>guarded fallback</pending_tasks></state_snapshot>",
        );
        let captured = backend.captured.clone();
        // A tiny declared window (1_000 tokens) with history that alone
        // exceeds 90% of it forces the guard to trip.
        let all_messages = vec![
            ChatMessage::system("canonical system prompt"),
            ChatMessage::user("word ".repeat(2_000)),
        ];
        let compacted = compact_history(
            &backend,
            "mock",
            &all_messages,
            1,
            None,
            HistoryPins::default(),
            None,
            Some(1_000),
            IdleTimeouts::uniform(Duration::from_secs(60)),
            CancellationToken::new(),
        )
        .await
        .expect("compaction succeeds via fallback");
        assert!(
            compacted.checkpoint_messages[0]
                .text_content()
                .unwrap()
                .contains("guarded fallback")
        );

        let requests = captured.lock().unwrap();
        assert!(!requests.is_empty());
        for (messages, _) in requests.iter() {
            assert!(
                !NativeAwareBackend::is_native_request(messages),
                "native instruction must never be sent once the guard trips"
            );
        }
    }

    /// Build an assistant message issuing one `read_file` tool call, using
    /// the real schema fields (`file_path`, `offset`, `limit`; see
    /// `tools/mod.rs` `ReadFileArgs`).
    fn read_file_call(
        call_id: &str,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> ChatMessage {
        use anvil_llm::llm_client::{FunctionCall, ToolCall};

        let mut args = serde_json::Map::new();
        args.insert(
            "file_path".to_string(),
            serde_json::Value::String(path.to_string()),
        );
        if let Some(offset) = offset {
            args.insert("offset".to_string(), serde_json::Value::from(offset));
        }
        if let Some(limit) = limit {
            args.insert("limit".to_string(), serde_json::Value::from(limit));
        }
        ChatMessage::assistant_tool_calls(vec![ToolCall {
            id: call_id.to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: serde_json::Value::Object(args).to_string(),
            },
        }])
    }

    #[test]
    fn files_already_read_digest_is_none_without_read_file_calls() {
        let history = vec![ChatMessage::user("hello"), ChatMessage::assistant("hi")];
        assert!(files_already_read_digest(&history).is_none());
    }

    #[test]
    fn files_already_read_digest_dedupes_merges_ranges_and_prefers_whole_file() {
        let history = vec![
            read_file_call("c1", "src/foo.rs", Some(0), Some(200)), // lines 1-200
            read_file_call("c2", "src/foo.rs", Some(399), Some(51)), // lines 400-450
            read_file_call("c3", "src/foo.rs", Some(150), Some(100)), // lines 151-250, overlaps c1
            read_file_call("c4", "src/bar.rs", Some(10), Some(5)),  // lines 11-15
            read_file_call("c5", "src/bar.rs", None, None),         // whole file wins
            read_file_call("c6", "src/baz.rs", Some(20), None),     // offset-to-end
        ];
        let digest = files_already_read_digest(&history).expect("digest present");
        assert!(digest.starts_with("<files_already_read>\n"));
        assert!(digest.ends_with("\n</files_already_read>"));
        // c1 and c3 overlap (1-200 and 151-250) and merge into one range;
        // c2 (400-450) stays separate.
        assert!(
            digest.contains("src/foo.rs (lines 1-250, 400-450)"),
            "unexpected digest: {digest}"
        );
        // A later whole-file read overrides the earlier ranged reads.
        assert!(digest.contains("src/bar.rs\n") || digest.contains("src/bar.rs\n</"));
        assert!(!digest.contains("src/bar.rs (lines"));
        // offset present, no limit -> "to end of file".
        assert!(digest.contains("src/baz.rs (lines 21-end)"), "{digest}");
        // First-seen order: foo.rs before bar.rs before baz.rs.
        let foo_pos = digest.find("src/foo.rs").unwrap();
        let bar_pos = digest.find("src/bar.rs").unwrap();
        let baz_pos = digest.find("src/baz.rs").unwrap();
        assert!(foo_pos < bar_pos && bar_pos < baz_pos);
    }

    #[test]
    fn files_already_read_digest_ignores_unparseable_and_missing_path() {
        use anvil_llm::llm_client::{FunctionCall, ToolCall};

        let bad_json = ChatMessage::assistant_tool_calls(vec![ToolCall {
            id: "c1".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: "not json".into(),
            },
        }]);
        let missing_path = ChatMessage::assistant_tool_calls(vec![ToolCall {
            id: "c2".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: r#"{"offset":0}"#.into(),
            },
        }]);
        let history = vec![bad_json, missing_path];
        assert!(files_already_read_digest(&history).is_none());
    }

    #[test]
    fn files_already_read_digest_caps_paths_and_notes_the_rest() {
        let history: Vec<ChatMessage> = (0..101)
            .map(|i| read_file_call(&format!("c{i}"), &format!("src/file_{i}.rs"), None, None))
            .collect();
        let digest = files_already_read_digest(&history).expect("digest present");
        assert!(digest.contains("(+1 more files)"), "{digest}");
        assert!(digest.contains("src/file_0.rs"));
        assert!(!digest.contains("src/file_100.rs"));
    }

    /// Build a `role: "tool"` message paired with an assistant tool call so
    /// the digest's call-id join has something to match. `result` becomes
    /// the tool message's text content.
    fn tool_exchange_messages(
        call_id: &str,
        tool_name: &str,
        args: &str,
        result: &str,
    ) -> Vec<ChatMessage> {
        use anvil_llm::llm_client::{FunctionCall, ToolCall};

        vec![
            ChatMessage::assistant_tool_calls(vec![ToolCall {
                id: call_id.to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: tool_name.to_string(),
                    arguments: args.to_string(),
                },
            }]),
            ChatMessage::tool_result(call_id, tool_name, result),
        ]
    }

    #[test]
    fn failed_tool_calls_digest_is_none_when_nothing_failed() {
        let mut history = Vec::new();
        history.extend(tool_exchange_messages(
            "c1",
            "read_file",
            "{}",
            "file contents",
        ));
        assert!(failed_tool_calls_digest(&history).is_none());
    }

    #[test]
    fn failed_tool_calls_digest_catches_malformed_args_error() {
        let mut history = Vec::new();
        history.extend(tool_exchange_messages(
            "c1",
            "read_file",
            r#"{"file_path": "src/lib.rs""#,
            "Error: tool arguments are not valid JSON (EOF while parsing an object at line 1 column 27)",
        ));
        // A successful call must not show up alongside the failure.
        history.extend(tool_exchange_messages(
            "c2",
            "read_file",
            "{}",
            "ok contents",
        ));
        let digest = failed_tool_calls_digest(&history).expect("digest present");
        assert!(digest.starts_with("<failed_tool_calls>\n"));
        assert!(digest.contains("1. `read_file` args="));
        assert!(digest.contains("Error: tool arguments are not valid JSON"));
        assert!(!digest.contains("ok contents"));
    }

    #[test]
    fn failed_tool_calls_digest_catches_permission_denials() {
        let mut history = Vec::new();
        history.extend(tool_exchange_messages(
            "c1",
            "edit",
            r#"{"file_path":"src/lib.rs","old_string":"a","new_string":"b"}"#,
            "Tool use denied by user.",
        ));
        history.extend(tool_exchange_messages(
            "c2",
            "run_shell_command",
            r#"{"command":"rm -rf target"}"#,
            "Tool use denied by auto permissions: destructive command outside approved scope",
        ));
        let digest = failed_tool_calls_digest(&history).expect("digest present");
        assert!(digest.contains("1. `edit` args="), "{digest}");
        assert!(digest.contains("Tool use denied by user."), "{digest}");
        assert!(digest.contains("2. `run_shell_command` args="), "{digest}");
    }

    #[test]
    fn failed_tool_calls_digest_truncates_long_arguments() {
        let long_args = format!(r#"{{"pattern":"{}"}}"#, "x".repeat(600));
        let mut history = Vec::new();
        history.extend(tool_exchange_messages(
            "c1",
            "grep_search",
            &long_args,
            "Error: no matches",
        ));
        let digest = failed_tool_calls_digest(&history).expect("digest present");
        assert!(digest.contains('…'), "expected truncation marker: {digest}");
        // 500 chars of args plus the marker, well under the full argument length.
        assert!(digest.len() < long_args.len());
    }

    #[test]
    fn failed_tool_calls_digest_keeps_last_ten_and_notes_omitted_count() {
        let mut history = Vec::new();
        for i in 0..13 {
            history.extend(tool_exchange_messages(
                &format!("c{i}"),
                "run_shell_command",
                &format!(r#"{{"command":"step{i}"}}"#),
                &format!("Error: step {i} failed"),
            ));
        }
        let digest = failed_tool_calls_digest(&history).expect("digest present");
        assert!(
            digest.contains("(3 earlier failed calls omitted)"),
            "{digest}"
        );
        // Only the last 10 (steps 3..=12) survive, renumbered from 1.
        assert!(digest.contains("step 3 failed"), "{digest}");
        assert!(!digest.contains("step 2 failed"), "{digest}");
        assert!(digest.contains("step 12 failed"), "{digest}");
        assert!(digest.contains("10. `run_shell_command`"), "{digest}");
        assert!(!digest.contains("11. `run_shell_command`"), "{digest}");
    }

    /// The recap entry point reuses the same machinery but drives the
    /// user-facing recap prompt rather than the compression prompt.
    #[tokio::test]
    async fn summarize_turn_for_recap_uses_recap_prompt() {
        let (backend, call_count, seen) = ScriptedBackend::new(
            "- Edited `lib.rs` and ran the tests.",
            "- chunk (unexpected)",
            "- meta (unexpected)",
        );
        let t = turn("fix the bug", "Done.");
        let out = summarize_turn_for_recap(
            &backend,
            "mock",
            &t,
            Some(200_000),
            IdleTimeouts::uniform(Duration::from_secs(60)),
            CancellationToken::new(),
        )
        .await
        .expect("succeeds");
        assert_eq!(out, "- Edited `lib.rs` and ran the tests.");
        assert_eq!(*call_count.lock().unwrap(), 1);
        let prompts = seen.lock().unwrap();
        assert!(
            prompts[0].contains("writing a short recap for the user"),
            "should use the recap system prompt; got: {}",
            &prompts[0][..prompts[0].len().min(120)]
        );
        // And it must NOT be the compression prompt.
        assert!(!prompts[0].contains("replaces a single past turn"));
    }

    #[tokio::test]
    async fn summarize_turn_retries_incomplete_stream_without_visible_output() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend = IncompleteThenSummaryBackend {
            attempts: attempts.clone(),
        };
        let t = turn("hello", "hi");

        let out = summarize_turn(
            &backend,
            "mock",
            &t,
            Some(200_000),
            IdleTimeouts::uniform(Duration::from_secs(60)),
            CancellationToken::new(),
        )
        .await
        .expect("retry should recover summarization");

        assert_eq!(out, "- recovered");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
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
            IdleTimeouts::uniform(Duration::from_secs(60)),
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
                        reasoning_content: None,
                        usage: anvil_llm::llm_client::TokenUsage::default(),
                        codex_reasoning: None,
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
            IdleTimeouts::uniform(Duration::from_secs(60)),
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
                        reasoning_content: None,
                        usage: anvil_llm::llm_client::TokenUsage::default(),
                        codex_reasoning: None,
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
            IdleTimeouts::uniform(Duration::from_secs(60)),
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
