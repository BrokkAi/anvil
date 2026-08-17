//! LLM relevance reranker wrapped transparently around bifrost's
//! `semantic_search` MCP tool.
//!
//! `semantic_search` returns independent, un-fused ranked lists
//! (`vector_ranked` over symbols, `coedit_ranked` over files) and explicitly
//! leaves fusion to the caller. Rather than dump the raw lists into the
//! model's context, the harness runs one *disposable*
//! LLM turn on top of the live conversation: it shows the selected utility model each
//! candidate together with its structured declaration signatures (and, for a
//! file candidate, its summary) and asks it to return just the relevant ones,
//! best-first. The model then sees a single clean, relevance-ordered hit list;
//! the bulky candidate context lives only in the disposable turn and never
//! pollutes the main conversation.
//!
//! The prompt is signatures-first: no candidate arrives with its source. In a
//! reproduced pool 97.9 percent of gold candidates were decidable from name,
//! path, signals, and signatures alone, and pre-fetching every body cost up to
//! 120KB per query to serve the other 2.1 percent. Those are reachable through
//! the one tool the disposable turn offers, `fetch_candidate_sources`, in a
//! bounded exchange of at most `MAX_BODY_FETCH_ROUNDS` rounds before the
//! selection must come back.
//!
//! The disposable turn takes the *task* half of the conversation as its prefix:
//! the system, user, and assistant text up to the trailing assistant message
//! that carries the in-flight `tool_calls`. Tool results and `tool_calls` are
//! left out (see `prefix_for_rerank`), so the utility model sees what the task
//! is and what the assistant has said about it, and none of the retrieved bulk
//! it has already read.
//!
//! Provider failures return a tool error. Invalid structured output gets one
//! corrective retry, then returns a tool error. A bifrost failure inside a
//! body-fetch round returns a tool error too. A failed reranker must not
//! silently change the result into a different retrieval treatment.
//! Bifrost's raw payload is never exposed to the model.
//!
//! One bifrost answer is not a failure at all: analyzer-pool saturation. Bifrost
//! bounds admission to its analyzer execution pool and refuses an overflowing
//! call with an explicitly retryable message, which this module's own fan-out
//! provokes (see `ANALYZER_SATURATION_MARKER`). Every bifrost call here goes
//! through `call_bifrost_tool_with_backpressure`, which waits and retries on
//! that signal within a bounded budget before it gives up and fails as any other
//! bifrost error does.
//!
//! Bifrost used to return a third list, `bm25_ranked`, from a lexical arm fused
//! alongside the dense one, and a `retrieval_profile` naming the leg budget the
//! `BIFROST_SEMANTIC_SEARCH_PROFILE` sweep had selected. Both were deleted in
//! bifrost c353c862 after the hybrid A/B lost on tokens and money with no
//! measurable task benefit; the winning budget (dense `2*k`, co-edit `k`) is now
//! the server's only behavior, so a current server never sends either key. This
//! module still reads both where they appear: replaying an archived response, or
//! pointing the harness at an older bifrost build, are ordinary inputs, and the
//! absence is normal rather than an error. Nothing branches on which shape
//! arrived.

use anyhow::Context;
use futures::future::join_all;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::llm_client::{
    ChatContentPart, ChatMessage, FunctionDef, IdleTimeouts, LlmBackend, LlmResponse,
    StreamChatRequest, TokenUsage, ToolCall, ToolDefinition,
    stream_chat_no_visible_output_with_retry,
};
use crate::structured_output::StructuredOutputRequest;
use crate::tools::ToolRegistry;
use crate::trace_logging::{append_trace_record, tool_timing_record};

const DEFAULT_FINAL_K: usize = 20;
const MAX_FINAL_K: usize = 20;
const OVERFETCH_MULTIPLIER: usize = 2;
const MAX_CANDIDATE_CONTEXT_BYTES: usize = 8_000;
const MAX_TOTAL_CONTEXT_BYTES: usize = 120_000;
const CONTEXT_FETCH_BATCH: usize = 8;
const MAX_SELECTED_DECLARATIONS: usize = 5;
const MAX_UTILITY_OUTPUT_ATTEMPTS: usize = 2;
const RRF_RANK_CONSTANT: f64 = 60.0;

/// Declarations rendered per candidate in the rerank prompt. Bifrost returns
/// every declaration it has for a target, and a reproduced pool held a
/// gtest-macro candidate with 290 of them that alone was 27 percent of the
/// prompt. Relevance is decided by the first few signatures; the tail is
/// repetition.
const MAX_PROMPT_DECLARATIONS: usize = 8;

/// The one tool the rerank turn offers the utility model.
const BODY_FETCH_TOOL: &str = "fetch_candidate_sources";
/// Tool rounds allowed before the turn must answer. Two rounds cover "fetch,
/// then fetch what the first fetch pointed at"; more is a budget, not a
/// decision procedure.
const MAX_BODY_FETCH_ROUNDS: usize = 2;
/// Candidate ids one `fetch_candidate_sources` call may ask for.
const MAX_BODY_FETCH_IDS: usize = 10;
/// Source bytes the whole rerank may hand back through the tool. In a
/// reproduced pool 97.9 percent of gold candidates were decidable from name,
/// path, signals, and signatures alone, so this is an exception budget.
const MAX_BODY_FETCH_TOTAL_BYTES: usize = 40_000;

/// First-progress idle timeout for this reranker's utility requests, and for
/// nothing else.
///
/// The measured failure mode is the provider stalling before its first token.
/// At the session's 120-second first-progress timeout, with up to four attempts
/// from `stream_chat_no_visible_output_with_retry`, one stalled rerank call can
/// spend more than eight minutes producing nothing. In the 2026-08-14
/// remeasure, 49 such stalls at 120 seconds each accounted for approximately
/// the entire rerank phase, and a body-fetch round multiplies the exposure by
/// putting more requests behind the same stall.
///
/// Only the first-progress bound moves. Once the utility model has started
/// streaming it is making progress, so the inter-chunk bound stays whatever the
/// session set. A session configured tighter than this keeps its own value.
const UTILITY_FIRST_PROGRESS_TIMEOUT: Duration = Duration::from_secs(30);

/// The stable part of bifrost's analyzer-pool overflow message.
///
/// Bifrost bounds admission to its analyzer execution pool -- four executing
/// calls plus a 32-deep queue -- and refuses anything past that with
/// `-32603 too many analyzer requests are queued; retry <tool> once earlier
/// calls complete`. That is backpressure, not a failed request: the server is
/// telling this client to come back, and the tool name in the message varies
/// while this phrase does not.
const ANALYZER_SATURATION_MARKER: &str = "too many analyzer requests are queued";

/// Waits between attempts at a saturated bifrost, and with them the whole retry
/// budget for one call: four retries spread over 15.25 seconds.
///
/// The fan-out that provokes the signal is this module's own. Enrichment issues
/// one `get_summaries` per candidate in batches of `CONTEXT_FETCH_BATCH`, for
/// every query in the batch, and CIM step zero multiplies that by the number of
/// workspaces: four workspaces with three queries put roughly 96 calls in flight
/// against 36 admission slots at one server. Each of those calls asks for a
/// single target's summary, so the work queued ahead of a refused call is small
/// and drains in well under a second -- which is why the first retry is 250 ms
/// rather than a full second. The quadrupling steps then back off for a server
/// that is genuinely congested rather than momentarily full.
///
/// The 15-second ceiling is what makes this safe to do at every call site: it is
/// under 1 percent of the 1800-second CIM MCP tool timeout, and the worst case
/// -- every sequential batch of a large query exhausting the whole budget --
/// costs tens of seconds against a task that today scores zero. Past the ceiling
/// the call fails exactly as it did before.
const SATURATION_RETRY_DELAYS: [Duration; 4] = [
    Duration::from_millis(250),
    Duration::from_secs(1),
    Duration::from_secs(4),
    Duration::from_secs(10),
];

const MAX_LINE_CHARS: usize = 2048;

/// What the reranker hands back to the tool loop, mirroring the fields the loop
/// needs from a `ToolExecution`.
pub(crate) struct RerankOutcome {
    pub output: String,
    pub failed: bool,
    pub usage: TokenUsage,
    pub usage_model: Option<String>,
}

impl RerankOutcome {
    fn passthrough(output: String, usage: TokenUsage) -> Self {
        Self {
            output,
            failed: false,
            usage,
            usage_model: None,
        }
    }

    fn error(message: String) -> Self {
        Self {
            output: message,
            failed: true,
            usage: TokenUsage::default(),
            usage_model: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CandidateKind {
    Symbol,
    File,
}

#[derive(Debug, Clone)]
struct Candidate {
    /// Short stable id shown to the model (`s1`, `f1`, ...).
    id: String,
    kind: CandidateKind,
    /// Fully-qualified symbol name or workspace-relative file path.
    name: String,
    /// Which retrieval legs surfaced it (`vector` for symbols, `coedit` for
    /// files; `bm25` only when replaying a pre-c353c862 bifrost response).
    signals: Vec<&'static str>,
    /// Where the retrieved chunk sits, for symbols (from `get_symbol_sources`).
    /// Kept structured so the agent-facing renderer can ask whether a
    /// declaration is in the same file without re-parsing a rendered string.
    location: Option<ChunkLocation>,
    /// Truncated source (symbols) or rendered summary (files), if fetched.
    context: Option<String>,
    /// Structured declaration signatures supplied by bifrost. These are
    /// prompt-local choices for the utility model and compact locators for the
    /// calling agent; they are never reconstructed from source text.
    declarations: Vec<DeclarationLocator>,
    /// Reciprocal-rank score accumulated across the active retrieval legs.
    rrf_score: f64,
    /// First raw-list position, used to make equal RRF scores deterministic.
    first_seen: usize,
}

/// The chunk range bifrost retrieved, before any declaration refines it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChunkLocation {
    path: String,
    lines: Option<(u64, u64)>,
}

impl std::fmt::Display for ChunkLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.lines {
            Some((start, end)) => write!(f, "{}:{start}-{end}", self.path),
            None => write!(f, "{}", self.path),
        }
    }
}

#[derive(Debug, Clone)]
struct DeclarationLocator {
    id: String,
    symbol: String,
    kind: String,
    path: String,
    start_line: u64,
    end_line: u64,
    signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Selection {
    id: String,
    declarations: Vec<String>,
}

struct RankedCandidate<'a> {
    candidate: &'a Candidate,
    declarations: Vec<&'a DeclarationLocator>,
    declaration_fallback: bool,
}

/// Run `semantic_search`, then rerank its candidates with a disposable LLM turn
/// and render a unified, relevance-ordered hit list. Reranker failures return
/// an explicit tool error and never substitute reciprocal-rank fusion.
///
/// The tool loop routes `semantic_search` here instead of through
/// `execute_tool`, so this wrapper -- not the loop -- owns the call's
/// `tool_timing` trace record. `duration` spans the whole model-visible call
/// (every query in the batch: search, candidate context fetch, and the
/// disposable rerank turn), which is what `execute_tool` records for every
/// other tool: the time the model waited.
///
/// Note that this counts calls the `semantic_search_batch` records cannot: the
/// batch-start record is written only after `parse_final_k` and `parse_queries`
/// accept the arguments, so a rejected call never opens a batch. In the r26
/// CodeScaleBench arm the traces hold 55 batch starts against 64
/// `semantic_search` calls the model issued; argument rejection is one
/// mechanism for that gap, and it is not yet established that it is the only
/// one. Do not read batch-start counts as a call census.
pub(crate) async fn rerank_semantic_search(
    llm: &Arc<dyn LlmBackend>,
    model: &str,
    registry: &ToolRegistry,
    prior_messages: &[ChatMessage],
    args: &Value,
    idle_timeout: IdleTimeouts,
    cancel: &CancellationToken,
) -> RerankOutcome {
    let started = std::time::Instant::now();
    let outcome = run_rerank(
        llm,
        model,
        registry,
        prior_messages,
        args,
        idle_timeout,
        cancel,
    )
    .await;
    append_trace_record(tool_timing_record(
        "semantic_search",
        None,
        started.elapsed(),
        !outcome.failed,
    ));
    outcome
}

/// Fan the batch out: every query in `queries` is retrieved and reranked
/// independently, then the per-query sections are concatenated in request
/// order. Split from the wrapper above so the `tool_timing` record covers the
/// whole model-visible call.
async fn run_rerank(
    llm: &Arc<dyn LlmBackend>,
    model: &str,
    registry: &ToolRegistry,
    prior_messages: &[ChatMessage],
    args: &Value,
    idle_timeout: IdleTimeouts,
    cancel: &CancellationToken,
) -> RerankOutcome {
    let started = Instant::now();
    let final_k = match parse_final_k(args) {
        Ok(k) => k,
        Err(message) => return RerankOutcome::error(message),
    };
    let queries = match parse_queries(args) {
        Ok(queries) => queries,
        Err(message) => return RerankOutcome::error(message),
    };
    let workspace = args.get("workspace").and_then(Value::as_str);
    append_trace_record(json!({
        "type": "semantic_search_batch",
        "phase": "start",
        "query_count": queries.len(),
        "requested_final_k_per_query": final_k,
    }));
    let query_count = queries.len();
    let outcomes = join_all(queries.iter().enumerate().map(|(query_index, query)| {
        rerank_one_semantic_search(
            llm,
            model,
            registry,
            prior_messages,
            query,
            query_index,
            query_count,
            final_k,
            workspace,
            idle_timeout,
            cancel,
        )
    }))
    .await;

    let mut usage = TokenUsage::default();
    let mut usage_model: Option<String> = None;
    let mut failed = false;
    let mut sections = Vec::with_capacity(outcomes.len());
    for ((query_index, query), outcome) in queries.iter().enumerate().zip(outcomes) {
        usage.add(outcome.usage);
        if let Some(model) = outcome.usage_model {
            if let Some(existing) = &usage_model {
                assert_eq!(
                    existing, &model,
                    "one semantic-search batch must use one utility model"
                );
            } else {
                usage_model = Some(model);
            }
        }
        failed |= outcome.failed;
        sections.push(format!(
            "Query {} of {}: \"{}\"\n{}",
            query_index + 1,
            query_count,
            query,
            outcome.output
        ));
    }
    append_trace_record(json!({
        "type": "semantic_search_batch",
        "phase": "complete",
        "query_count": query_count,
        "requested_final_k_per_query": final_k,
        "failed": failed,
        "elapsed_millis": started.elapsed().as_millis(),
    }));
    RerankOutcome {
        output: sections.join("\n\n"),
        failed,
        usage,
        usage_model,
    }
}

#[allow(clippy::too_many_arguments)]
async fn rerank_one_semantic_search(
    llm: &Arc<dyn LlmBackend>,
    model: &str,
    registry: &ToolRegistry,
    prior_messages: &[ChatMessage],
    query: &str,
    query_index: usize,
    query_count: usize,
    final_k: usize,
    workspace: Option<&str>,
    idle_timeout: IdleTimeouts,
    cancel: &CancellationToken,
) -> RerankOutcome {
    let utility = crate::utility_model::select(model);
    // Every request this function makes is a utility request, so the shorter
    // first-progress bound is applied once, here. See
    // `UTILITY_FIRST_PROGRESS_TIMEOUT`.
    let idle_timeout = IdleTimeouts {
        first_progress: idle_timeout
            .first_progress
            .min(UTILITY_FIRST_PROGRESS_TIMEOUT),
        inter_chunk: idle_timeout.inter_chunk,
    };
    let started = Instant::now();
    let phases = PhaseTrace {
        query,
        query_index,
        query_count,
        utility: &utility,
        started,
    };
    let (bifrost_args, base_k) = prepare_bifrost_args(query, final_k, workspace);

    // 1. Underlying search. A hard failure here is the model's to see.
    phases.record("retrieval_start", None);
    let raw = match call_bifrost_tool_with_backpressure(registry, "semantic_search", bifrost_args)
        .await
    {
        Ok(value) => {
            phases.record("retrieval_complete", None);
            value
        }
        Err(err) => {
            phases.record("retrieval_error", Some(&format!("{err:#}")));
            return RerankOutcome::error(format!("Error: {err}"));
        }
    };

    // 2. Parse the entire realized pool. An empty search is a valid final result.
    let mut candidates = parse_candidates(&raw);
    if candidates.is_empty() {
        trace_rerank(RerankTrace {
            query,
            query_index,
            query_count,
            raw: &raw,
            final_k,
            base_k,
            deduplicated_count: 0,
            context_bytes: 0,
            selected_count: 0,
            final_count: 0,
            signature_candidate_count: 0,
            signature_locator_count: 0,
            final_with_signature_count: 0,
            declaration_fallback_count: 0,
            body_fetch_rounds: 0,
            bodies_fetched: 0,
            failure_reason: None,
            usage: TokenUsage::default(),
            utility: &utility,
        });
        return RerankOutcome::passthrough(
            render_unified(query, &[], 0, notes(&raw)),
            TokenUsage::default(),
        );
    }

    // 3. Fetch the structured declaration records for the candidates.
    phases.record("context_fetch_start", None);
    if let Err(err) = fetch_context(registry, &mut candidates, workspace).await {
        phases.record("context_fetch_error", Some(&format!("{err:#}")));
        return RerankOutcome::error(format!(
            "Error: semantic_search context fetch failed: {err:#}"
        ));
    }
    phases.record("context_fetch_complete", None);
    let context_bytes = bound_candidate_context(&mut candidates);

    // 4. Disposable relevance turn on top of the live conversation.
    let mut messages = prefix_for_rerank(prior_messages);
    messages.push(ChatMessage::user(build_rerank_prompt(
        query,
        &candidates,
        final_k,
    )));
    let structured = StructuredOutputRequest {
        schema_name: "semantic_rerank".to_string(),
        schema: rerank_schema(final_k),
        allow_coercion: true,
        prefer_json_object: false,
    };

    phases.record("utility_request_start", None);
    let body_fetch_tool = vec![body_fetch_tool_definition()];
    let mut usage = TokenUsage::default();
    let mut rejection = String::new();
    let mut attempts_made = 0;
    let mut last_selected_count = 0;
    let mut budget = BodyFetchBudget::default();
    'attempts: for attempt in 1..=MAX_UTILITY_OUTPUT_ATTEMPTS {
        attempts_made = attempt;
        // Body-fetch rounds run inside the attempt. Tools and the
        // structured-output schema ride on the same request, so a turn that
        // needs no body still answers in a single call; once the round budget
        // is spent the request goes out with the schema and no tools.
        let text = 'exchange: loop {
            let tools_open = budget.rounds < MAX_BODY_FETCH_ROUNDS;
            let response = stream_chat_no_visible_output_with_retry(
                llm.as_ref(),
                "semantic_search rerank",
                cancel,
                || StreamChatRequest {
                    model: utility.model.clone(),
                    messages: messages.clone(),
                    tools: tools_open.then(|| body_fetch_tool.clone()),
                    reasoning_effort: utility.reasoning_effort.clone(),
                    service_tier: None,
                    temperature: None,
                    structured_output: Some(structured.clone()),
                    on_token: Box::new(|_| {}),
                    on_thought: Box::new(|_| {}),
                    cancel: cancel.clone(),
                    idle_timeouts: idle_timeout,
                },
            )
            .await;

            let response = match response {
                Ok(response) => response,
                Err(err) => {
                    phases.record("utility_request_error", Some(&format!("{err:#}")));
                    rejection = format!("provider request failed: {err:#}");
                    break 'attempts;
                }
            };
            phases.record("utility_request_complete", None);
            usage.add(response.usage());
            let (text, reasoning_content, calls) = match response {
                LlmResponse::Text {
                    text,
                    reasoning_content,
                    ..
                } => (text, reasoning_content, Vec::new()),
                LlmResponse::ToolCalls {
                    text,
                    reasoning_content,
                    calls,
                    ..
                } => (text, reasoning_content, calls),
            };
            if !tools_open || calls.is_empty() {
                break 'exchange text;
            }

            // A round that resolves to nothing useful still spends a round: the
            // budget bounds the exchange, not the model's hit rate.
            budget.rounds += 1;
            let mut results = Vec::with_capacity(calls.len());
            let mut round_ids = 0;
            for call in &calls {
                match run_body_fetch_call(registry, &mut candidates, call, workspace, &mut budget)
                    .await
                {
                    Ok((result, ids)) => {
                        round_ids += ids;
                        results.push(result);
                    }
                    Err(err) => {
                        phases.record("utility_tool_error", Some(&format!("{err:#}")));
                        return RerankOutcome::error(format!(
                            "Error: semantic_search source fetch failed: {err:#}"
                        ));
                    }
                }
            }
            phases.record_tool_round(budget.rounds, round_ids);
            messages.push(
                ChatMessage::assistant_tool_calls_with_content_and_reasoning(
                    text,
                    calls.clone(),
                    reasoning_content,
                ),
            );
            for (call, result) in calls.iter().zip(results) {
                messages.push(ChatMessage::tool_result(
                    &call.id,
                    &call.function.name,
                    result,
                ));
            }
            if budget.rounds == MAX_BODY_FETCH_ROUNDS {
                messages.push(ChatMessage::user(format!(
                    "Tool access is finished: `{BODY_FETCH_TOOL}` is no longer available. Decide \
                     from what you already have and return only the selection object."
                )));
            }
        };

        let selected = parse_selections(&text);
        let selected_count = selected.as_ref().map_or(0, Vec::len);
        last_selected_count = selected_count;
        let ordered = selected
            .as_ref()
            .ok_or_else(|| "malformed structured output".to_string())
            .and_then(|selected| order_candidates(&candidates, selected, final_k));
        match ordered {
            Ok(ordered) => {
                tracing::debug!(
                    query = %query,
                    candidates = candidates.len(),
                    selected = ordered.len(),
                    cached_read_tokens = usage.cached_read_tokens,
                    input_tokens = usage.input_tokens,
                    attempts = attempt,
                    "semantic_search reranked"
                );
                trace_rerank(RerankTrace {
                    query,
                    query_index,
                    query_count,
                    raw: &raw,
                    final_k,
                    base_k,
                    deduplicated_count: candidates.len(),
                    context_bytes,
                    selected_count,
                    final_count: ordered.len(),
                    signature_candidate_count: candidates
                        .iter()
                        .filter(|candidate| !candidate.declarations.is_empty())
                        .count(),
                    signature_locator_count: candidates
                        .iter()
                        .map(|candidate| candidate.declarations.len())
                        .sum(),
                    final_with_signature_count: ordered
                        .iter()
                        .filter(|selected| !selected.declarations.is_empty())
                        .count(),
                    declaration_fallback_count: ordered
                        .iter()
                        .filter(|selected| selected.declaration_fallback)
                        .count(),
                    body_fetch_rounds: budget.rounds,
                    bodies_fetched: budget.bodies,
                    failure_reason: None,
                    usage,
                    utility: &utility,
                });
                let output = render_unified(query, &ordered, candidates.len(), notes(&raw));
                let mut outcome = RerankOutcome::passthrough(output, usage);
                outcome.usage_model = Some(utility.model.clone());
                return outcome;
            }
            Err(error) => rejection = error,
        }

        let will_retry = attempt < MAX_UTILITY_OUTPUT_ATTEMPTS;
        append_trace_record(json!({
            "type": "semantic_search_utility_output_rejected",
            "query": query,
            "query_index": query_index,
            "query_count": query_count,
            "attempt": attempt,
            "will_retry": will_retry,
            "reason": rejection,
            "output": text,
            "utility_model": utility.model,
        }));
        if will_retry {
            messages.push(ChatMessage::assistant(text));
            messages.push(ChatMessage::user(format!(
                "That response was invalid because {rejection}. Return only an object matching \
                 the requested JSON schema. Use exact candidate and declaration ids."
            )));
        }
    }

    trace_rerank(RerankTrace {
        query,
        query_index,
        query_count,
        raw: &raw,
        final_k,
        base_k,
        deduplicated_count: candidates.len(),
        context_bytes,
        selected_count: last_selected_count,
        final_count: 0,
        signature_candidate_count: candidates
            .iter()
            .filter(|candidate| !candidate.declarations.is_empty())
            .count(),
        signature_locator_count: candidates
            .iter()
            .map(|candidate| candidate.declarations.len())
            .sum(),
        final_with_signature_count: 0,
        declaration_fallback_count: 0,
        body_fetch_rounds: budget.rounds,
        bodies_fetched: budget.bodies,
        failure_reason: Some(&rejection),
        usage,
        utility: &utility,
    });
    RerankOutcome {
        output: format!(
            "Error: semantic_search reranker failed after {attempts_made} attempt(s): {rejection}"
        ),
        failed: true,
        usage,
        usage_model: Some(utility.model.clone()),
    }
}

/// Symbol legs, in the order they are folded together. `vector_ranked` is the
/// only one a current bifrost sends; `bm25_ranked` is read for replayed and
/// older-server responses (see the module docs). A leg the response omits
/// simply contributes nothing.
const SYMBOL_LEGS: [(&str, &str); 2] = [("vector_ranked", "vector"), ("bm25_ranked", "bm25")];

fn parse_final_k(args: &Value) -> Result<usize, String> {
    let Some(object) = args.as_object() else {
        return Err("Invalid arguments for `semantic_search`: expected an object".to_string());
    };
    let Some(value) = object.get("k") else {
        return Ok(DEFAULT_FINAL_K);
    };
    let Some(k) = value.as_u64() else {
        return Err(
            "Invalid arguments for `semantic_search`: k must be an integer from 1 through 20"
                .to_string(),
        );
    };
    if !(1..=MAX_FINAL_K as u64).contains(&k) {
        return Err(
            "Invalid arguments for `semantic_search`: k must be an integer from 1 through 20"
                .to_string(),
        );
    }
    Ok(k as usize)
}

fn parse_queries(args: &Value) -> Result<Vec<String>, String> {
    let Some(object) = args.as_object() else {
        return Err("Invalid arguments for `semantic_search`: expected an object".to_string());
    };
    let Some(values) = object.get("queries").and_then(Value::as_array) else {
        return Err(
            "Invalid arguments for `semantic_search`: queries must be an array of 1 through 3 strings"
                .to_string(),
        );
    };
    if !(1..=3).contains(&values.len()) {
        return Err(
            "Invalid arguments for `semantic_search`: queries must contain 1 through 3 strings"
                .to_string(),
        );
    }
    let mut seen = HashSet::new();
    let mut queries = Vec::with_capacity(values.len());
    for value in values {
        let Some(query) = value.as_str() else {
            return Err(
                "Invalid arguments for `semantic_search`: every query must be a string".to_string(),
            );
        };
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() || normalized != query {
            return Err(
                "Invalid arguments for `semantic_search`: queries must be nonempty and have normalized whitespace"
                    .to_string(),
            );
        }
        if !seen.insert(query.to_lowercase()) {
            return Err(
                "Invalid arguments for `semantic_search`: queries must be unique ignoring case"
                    .to_string(),
            );
        }
        queries.push(query.to_string());
    }
    Ok(queries)
}

fn prepare_bifrost_args(query: &str, final_k: usize, workspace: Option<&str>) -> (Value, usize) {
    let base_k = final_k * OVERFETCH_MULTIPLIER;
    (
        bifrost_args_with_workspace(json!({ "query": query, "k": base_k }), workspace),
        base_k,
    )
}

/// Call a bifrost tool on the harness's behalf, waiting out analyzer-pool
/// saturation.
///
/// Saturation is the one bifrost error that says "not now" rather than "no": it
/// reports that admission to the analyzer pool is full and names retrying as the
/// remedy (`ANALYZER_SATURATION_MARKER`). Treating it as a failure is what turned
/// a busy server into a failed retrieval, and a failed step-zero retrieval into a
/// scored zero. So a saturated call waits and repeats itself on
/// `SATURATION_RETRY_DELAYS`; every other error, and saturation that outlives the
/// budget, returns unchanged. Handling it here, at the point the signal arrives,
/// keeps each caller's own concurrency intact -- the server already bounds how
/// much work it admits, so a client-side limiter would only duplicate a decision
/// bifrost has already made.
async fn call_bifrost_tool_with_backpressure(
    registry: &ToolRegistry,
    tool: &str,
    args: Value,
) -> anyhow::Result<Value> {
    let mut retries = 0usize;
    loop {
        let error = match registry.call_bifrost_tool_raw(tool, args.clone()).await {
            Ok(value) => {
                if retries > 0 {
                    append_trace_record(json!({
                        "type": "bifrost_analyzer_saturation",
                        "phase": "recovered",
                        "tool": tool,
                        "retries": retries,
                    }));
                }
                return Ok(value);
            }
            Err(error) => error,
        };
        let message = format!("{error:#}");
        if !message.contains(ANALYZER_SATURATION_MARKER) {
            return Err(error);
        }
        let Some(delay) = SATURATION_RETRY_DELAYS.get(retries).copied() else {
            tracing::warn!(
                tool,
                retries,
                %message,
                "bifrost analyzer pool stayed saturated for the whole retry budget"
            );
            append_trace_record(json!({
                "type": "bifrost_analyzer_saturation",
                "phase": "exhausted",
                "tool": tool,
                "retries": retries,
                "error": message,
            }));
            return Err(error);
        };
        retries += 1;
        tracing::debug!(
            tool,
            retry = retries,
            delay_millis = delay.as_millis(),
            "bifrost analyzer pool is saturated; retrying"
        );
        append_trace_record(json!({
            "type": "bifrost_analyzer_saturation",
            "phase": "retry",
            "tool": tool,
            "retry": retries,
            "delay_millis": delay.as_millis(),
            "error": message,
        }));
        tokio::time::sleep(delay).await;
    }
}

/// Name the workspace in a bifrost tool call's arguments, or leave them alone
/// in the single-root shape that has no router. Shared with `cim`, which builds
/// step zero's forced `semantic_search` call the same way.
pub(crate) fn bifrost_args_with_workspace(mut args: Value, workspace: Option<&str>) -> Value {
    if let Some(workspace) = workspace {
        args.as_object_mut()
            .expect("Bifrost arguments must be an object")
            .insert("workspace".to_string(), json!(workspace));
    }
    args
}

/// Parse every realized item from `semantic_search`'s legs into one
/// identity-deduplicated candidate list. The order is deterministic RRF order,
/// which is also the provider-failure fallback order.
fn parse_candidates(raw: &Value) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut index: HashMap<(CandidateKind, String), usize> = HashMap::new();
    let mut first_seen = 0;
    for (key, signal) in SYMBOL_LEGS {
        let Some(array) = raw.get(key).and_then(Value::as_array) else {
            continue;
        };
        for (rank, item) in array.iter().enumerate() {
            let Some(fqfn) = item.get("fqfn").and_then(Value::as_str) else {
                continue;
            };
            let identity = (CandidateKind::Symbol, fqfn.to_string());
            match index.get(&identity) {
                Some(&i) => {
                    candidates[i].rrf_score += 1.0 / (RRF_RANK_CONSTANT + rank as f64 + 1.0);
                    if !candidates[i].signals.contains(&signal) {
                        candidates[i].signals.push(signal);
                    }
                }
                None => {
                    index.insert(identity, candidates.len());
                    candidates.push(Candidate {
                        id: String::new(),
                        kind: CandidateKind::Symbol,
                        name: fqfn.to_string(),
                        signals: vec![signal],
                        location: None,
                        context: None,
                        declarations: Vec::new(),
                        rrf_score: 1.0 / (RRF_RANK_CONSTANT + rank as f64 + 1.0),
                        first_seen,
                    });
                    first_seen += 1;
                }
            }
        }
    }
    if let Some(array) = raw.get("coedit_ranked").and_then(Value::as_array) {
        for (rank, item) in array.iter().enumerate() {
            let Some(path) = item.get("path").and_then(Value::as_str) else {
                continue;
            };
            let identity = (CandidateKind::File, path.to_string());
            match index.get(&identity) {
                Some(&i) => {
                    candidates[i].rrf_score += 1.0 / (RRF_RANK_CONSTANT + rank as f64 + 1.0)
                }
                None => {
                    index.insert(identity, candidates.len());
                    candidates.push(Candidate {
                        id: String::new(),
                        kind: CandidateKind::File,
                        name: path.to_string(),
                        signals: vec!["coedit"],
                        location: None,
                        context: None,
                        declarations: Vec::new(),
                        rrf_score: 1.0 / (RRF_RANK_CONSTANT + rank as f64 + 1.0),
                        first_seen,
                    });
                    first_seen += 1;
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.first_seen.cmp(&b.first_seen))
            .then_with(|| a.name.cmp(&b.name))
    });
    let mut symbol_id = 0;
    let mut file_id = 0;
    for candidate in &mut candidates {
        candidate.id = match candidate.kind {
            CandidateKind::Symbol => {
                symbol_id += 1;
                format!("s{symbol_id}")
            }
            CandidateKind::File => {
                file_id += 1;
                format!("f{file_id}")
            }
        };
    }
    candidates
}

/// Batch-fetch file summaries (`get_summaries`) for the candidates and attach
/// the structured declaration records they carry. A Bifrost error fails the
/// semantic-search call -- except analyzer-pool saturation, which is waited out
/// per call (`call_bifrost_tool_with_backpressure`), because this sweep's own
/// concurrency is what causes it. Name-only results are a different tool
/// contract, not a valid fallback for failed context enrichment.
///
/// This sweep no longer pre-fetches `get_symbol_sources` for every candidate.
/// Bodies are fetched on demand by the utility model through
/// `fetch_candidate_sources`; the summaries are the only source of the
/// signatures and `path:lines` records that both the prompt and the agent-facing
/// card are built from, so they are still fetched for every candidate.
async fn fetch_context(
    registry: &ToolRegistry,
    candidates: &mut [Candidate],
    workspace: Option<&str>,
) -> anyhow::Result<()> {
    for start in (0..candidates.len()).step_by(CONTEXT_FETCH_BATCH) {
        let end = (start + CONTEXT_FETCH_BATCH).min(candidates.len());
        let targets: Vec<String> = candidates[start..end]
            .iter()
            .map(|candidate| candidate.name.clone())
            .collect();
        fetch_summary_context(registry, candidates, &targets, workspace).await?;
    }
    Ok(())
}

async fn fetch_summary_context(
    registry: &ToolRegistry,
    candidates: &mut [Candidate],
    targets: &[String],
    workspace: Option<&str>,
) -> anyhow::Result<()> {
    if targets.is_empty() {
        return Ok(());
    }
    // Bifrost deliberately degrades an oversized aggregate summary to compact
    // file outlines. That is useful for an interactive caller, but discards the
    // structured declaration records this adapter needs for locator cards.
    // Keep every request intrinsically small while retaining parallelism within
    // the already bounded candidate batch.
    let summaries = join_all(targets.iter().map(|target| {
        call_bifrost_tool_with_backpressure(
            registry,
            "get_summaries",
            bifrost_args_with_workspace(json!({ "targets": [target] }), workspace),
        )
    }))
    .await;
    for (target, value) in targets.iter().zip(summaries) {
        let value = value.with_context(|| {
            format!("get_summaries failed for '{target}' during semantic-search enrichment")
        })?;
        attach_summaries(candidates, &value);
    }
    Ok(())
}

/// What the utility model may do with `fetch_candidate_sources`, spent across
/// the whole rerank of one query.
#[derive(Default)]
struct BodyFetchBudget {
    /// Tool rounds spent, whether or not they produced anything.
    rounds: usize,
    /// Candidates whose source reached the model, counting one whose body was
    /// reported as identical to an earlier candidate's: the model has that
    /// candidate's source either way.
    bodies: usize,
    /// Source bytes printed, against `MAX_BODY_FETCH_TOTAL_BYTES`.
    bytes: usize,
}

/// The one tool the rerank turn offers. It exists because 2.1 percent of gold
/// candidates in a reproduced pool were not decidable from name, path, signals,
/// and signatures: generic-named methods whose deciding evidence was in the
/// first lines of the body.
fn body_fetch_tool_definition() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: BODY_FETCH_TOOL.to_string(),
            description: format!(
                "Fetch the source of candidates whose relevance the signatures leave undecidable. \
                 Pass candidate ids exactly as shown in the prompt, for example \"s3\", at most \
                 {MAX_BODY_FETCH_IDS} per call. Source is available for symbol candidates only."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "maxItems": MAX_BODY_FETCH_IDS,
                        "description": "Candidate ids to fetch source for."
                    }
                },
                "required": ["ids"],
                "additionalProperties": false
            }),
        },
    }
}

/// Run one `fetch_candidate_sources` call and render the tool result, returning
/// it with the number of distinct ids the call asked for.
///
/// A bifrost failure propagates: the rerank fails closed on it, exactly as it
/// does for the search and the summary sweep. Everything the model can get
/// wrong -- another tool's name, unparsable arguments, an id that names no
/// candidate or names a file -- is reported inside the result text so the
/// exchange can continue.
async fn run_body_fetch_call(
    registry: &ToolRegistry,
    candidates: &mut [Candidate],
    call: &ToolCall,
    workspace: Option<&str>,
    budget: &mut BodyFetchBudget,
) -> anyhow::Result<(String, usize)> {
    if call.function.name != BODY_FETCH_TOOL {
        return Ok((
            format!(
                "Error: unknown tool '{}'. The only tool available here is `{BODY_FETCH_TOOL}`.",
                call.function.name
            ),
            0,
        ));
    }
    let arguments: Value = match serde_json::from_str(&call.function.arguments) {
        Ok(arguments) => arguments,
        Err(err) => {
            return Ok((
                format!("Error: could not parse the arguments of `{BODY_FETCH_TOOL}`: {err}"),
                0,
            ));
        }
    };
    let Some(requested) = arguments.get("ids").and_then(Value::as_array) else {
        return Ok((
            format!("Error: `{BODY_FETCH_TOOL}` needs an \"ids\" array of candidate ids."),
            0,
        ));
    };
    let mut ids: Vec<&str> = Vec::new();
    for value in requested {
        let Some(id) = value.as_str() else {
            return Ok((
                "Error: every entry of \"ids\" must be a candidate id string.".to_string(),
                0,
            ));
        };
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let mut notes: Vec<String> = Vec::new();
    if ids.len() > MAX_BODY_FETCH_IDS {
        notes.push(format!(
            "Only the first {MAX_BODY_FETCH_IDS} of {} requested ids were fetched.",
            ids.len()
        ));
        ids.truncate(MAX_BODY_FETCH_IDS);
    }

    let mut resolved: Vec<(&str, String)> = Vec::new();
    for id in &ids {
        match candidates.iter().find(|candidate| candidate.id == *id) {
            None => notes.push(format!("{id}: no candidate has that id.")),
            Some(candidate) if candidate.kind == CandidateKind::File => notes.push(format!(
                "{id}: a file candidate; source is available for symbol candidates only."
            )),
            Some(candidate) => resolved.push((id, candidate.name.clone())),
        }
    }
    let id_count = ids.len();
    if resolved.is_empty() {
        return Ok((render_body_fetch_result(String::new(), &notes), id_count));
    }

    let symbols: Vec<&str> = resolved.iter().map(|(_, name)| name.as_str()).collect();
    let sources = call_bifrost_tool_with_backpressure(
        registry,
        "get_symbol_sources",
        bifrost_args_with_workspace(json!({ "symbols": symbols }), workspace),
    )
    .await
    .context("get_symbol_sources failed during semantic-search body fetch")?;
    let bodies = take_symbol_sources(candidates, &sources);

    let mut out = String::new();
    let mut first_by_body: HashMap<&str, &str> = HashMap::new();
    for (id, name) in &resolved {
        let Some(body) = bodies.get(name) else {
            out.push_str(&format!("[{id}] {name}\nno source available\n\n"));
            continue;
        };
        if let Some(first) = first_by_body.get(body.as_str()) {
            out.push_str(&format!("[{id}] {name}\nidentical source to {first}\n\n"));
            budget.bodies += 1;
            continue;
        }
        if budget.bytes + body.len() > MAX_BODY_FETCH_TOTAL_BYTES {
            notes.push(format!(
                "{id}: source omitted, this search's {MAX_BODY_FETCH_TOTAL_BYTES}-byte source \
                 budget is spent."
            ));
            continue;
        }
        budget.bytes += body.len();
        budget.bodies += 1;
        first_by_body.insert(body.as_str(), id);
        out.push_str(&format!("[{id}] {name}\n```\n{body}\n```\n\n"));
    }
    Ok((render_body_fetch_result(out, &notes), id_count))
}

fn render_body_fetch_result(mut out: String, notes: &[String]) -> String {
    for note in notes {
        out.push_str(note);
        out.push('\n');
    }
    let out = out.trim_end();
    if out.is_empty() {
        return "No sources were returned.".to_string();
    }
    out.to_string()
}

/// Enforce the disposable-turn declaration/summary budget without ever removing
/// a candidate identity. Earlier RRF candidates consume the shared budget first.
fn bound_candidate_context(candidates: &mut [Candidate]) -> usize {
    let mut remaining = MAX_TOTAL_CONTEXT_BYTES;
    let mut used = 0;
    for candidate in candidates {
        let mut candidate_remaining = remaining.min(MAX_CANDIDATE_CONTEXT_BYTES);
        let mut declarations = Vec::new();
        for mut declaration in std::mem::take(&mut candidate.declarations) {
            if declarations.len() == MAX_PROMPT_DECLARATIONS {
                break;
            }
            declaration.id = format!("{}d{}", candidate.id, declarations.len() + 1);
            let bytes = render_declaration_for_prompt(&declaration).len();
            if bytes > candidate_remaining {
                break;
            }
            candidate_remaining -= bytes;
            remaining -= bytes;
            used += bytes;
            declarations.push(declaration);
        }
        candidate.declarations = declarations;
        let Some(context) = candidate.context.take() else {
            continue;
        };
        let limit = candidate_remaining.min(remaining);
        if limit == 0 {
            continue;
        }
        let bounded = truncate_utf8(&context, limit);
        used += bounded.len();
        remaining -= bounded.len();
        candidate.context = Some(bounded);
    }
    used
}

fn truncate_utf8(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// Read a `get_symbol_sources` result
/// (`{ sources: [{label,path,start_line,end_line,text}] }`): record each
/// matching symbol candidate's chunk location, and return the bounded source
/// text keyed by candidate name.
///
/// The location is kept on the candidate because the agent-facing card needs it
/// (a C++ header/implementation split is the one case where the chunk names a
/// file no declaration names). The source text is not: bodies live in the tool
/// result the utility model asked for and nowhere else.
fn take_symbol_sources(candidates: &mut [Candidate], result: &Value) -> HashMap<String, String> {
    let mut bodies = HashMap::new();
    let Some(sources) = result.get("sources").and_then(Value::as_array) else {
        return bodies;
    };
    let mut by_label: HashMap<&str, &Value> = HashMap::new();
    for block in sources {
        if let Some(label) = block.get("label").and_then(Value::as_str) {
            by_label.entry(label).or_insert(block);
        }
    }
    for candidate in candidates
        .iter_mut()
        .filter(|c| c.kind == CandidateKind::Symbol)
    {
        let Some(block) = by_label.get(candidate.name.as_str()) else {
            continue;
        };
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            bodies.insert(
                candidate.name.clone(),
                truncate_utf8(&sample_source(text), MAX_CANDIDATE_CONTEXT_BYTES),
            );
        }
        let path = block.get("path").and_then(Value::as_str);
        let start = block.get("start_line").and_then(Value::as_u64);
        let end = block.get("end_line").and_then(Value::as_u64);
        candidate.location = path.map(|path| ChunkLocation {
            path: path.to_string(),
            lines: start.zip(end),
        });
    }
    bodies
}

/// Attach structured declaration signatures to every matching candidate and
/// rendered summaries as private context for file candidates.
fn attach_summaries(candidates: &mut [Candidate], result: &Value) {
    if let Some(summaries) = result.get("summaries").and_then(Value::as_array) {
        for candidate in candidates.iter_mut() {
            let matching: Vec<&Value> = summaries
                .iter()
                .filter(|block| match candidate.kind {
                    CandidateKind::Symbol => {
                        block.get("label").and_then(Value::as_str) == Some(candidate.name.as_str())
                    }
                    CandidateKind::File => {
                        block.get("path").and_then(Value::as_str) == Some(candidate.name.as_str())
                            || block.get("label").and_then(Value::as_str)
                                == Some(candidate.name.as_str())
                    }
                })
                .collect();
            if candidate.kind == CandidateKind::File
                && let Some(block) = matching.first()
            {
                candidate.context = Some(sample_summary(&render_summary_block(block)));
            }
            for block in matching {
                attach_declarations(candidate, block);
            }
        }
    }

    let Some(files) = result
        .get("compact_symbols")
        .and_then(|compact| compact.get("files"))
        .and_then(Value::as_array)
    else {
        return;
    };
    for file in files {
        let Some(path) = file.get("path").and_then(Value::as_str) else {
            continue;
        };
        let Some(candidate) = candidates
            .iter_mut()
            .find(|candidate| candidate.kind == CandidateKind::File && candidate.name == path)
        else {
            continue;
        };
        if candidate.context.is_some() {
            continue;
        }
        let loc = file.get("loc").and_then(Value::as_u64);
        let lines = file
            .get("lines")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        if !lines.is_empty() {
            candidate.context = Some(match loc {
                Some(loc) => format!("{path} ({loc} lines)\n{}", lines.join("\n")),
                None => lines.join("\n"),
            });
        }
    }
}

fn attach_declarations(candidate: &mut Candidate, block: &Value) {
    let Some(elements) = block.get("elements").and_then(Value::as_array) else {
        return;
    };
    for element in elements {
        if element.get("presentation").and_then(Value::as_str) == Some("sampled_excerpt") {
            continue;
        }
        let Some(signature) = element.get("text").and_then(Value::as_str) else {
            continue;
        };
        let signature = signature.trim();
        if signature.is_empty() {
            continue;
        }
        let Some(symbol) = element.get("symbol").and_then(Value::as_str) else {
            continue;
        };
        let Some(kind) = element.get("kind").and_then(Value::as_str) else {
            continue;
        };
        let Some(path) = element.get("path").and_then(Value::as_str) else {
            continue;
        };
        let Some(start_line) = element.get("start_line").and_then(Value::as_u64) else {
            continue;
        };
        let Some(end_line) = element.get("end_line").and_then(Value::as_u64) else {
            continue;
        };
        let duplicate = candidate.declarations.iter().any(|declaration| {
            declaration.symbol == symbol
                && declaration.path == path
                && declaration.start_line == start_line
                && declaration.end_line == end_line
                && declaration.signature == signature
        });
        if !duplicate {
            candidate.declarations.push(DeclarationLocator {
                id: String::new(),
                symbol: symbol.to_string(),
                kind: kind.to_string(),
                path: path.to_string(),
                start_line,
                end_line,
                signature: signature.to_string(),
            });
        }
    }
}

/// Flatten a summary block into text: preamble followed by each element's
/// declaration text.
fn render_summary_block(block: &Value) -> String {
    let mut out = String::new();
    if let Some(preamble) = block.get("preamble").and_then(Value::as_str) {
        let preamble = preamble.trim();
        if !preamble.is_empty() {
            out.push_str(preamble);
            out.push('\n');
        }
    }
    if let Some(elements) = block.get("elements").and_then(Value::as_array) {
        for element in elements {
            if let Some(text) = element.get("text").and_then(Value::as_str) {
                out.push_str(text.trim_end());
                out.push('\n');
            }
        }
    }
    out.trim_end().to_string()
}

/// Build the conversation prefix for the disposable turn: the system, user, and
/// assistant text of the live history, in order, up to the trailing assistant
/// `tool_calls` message (and any sibling tool results already appended this
/// step). Every tool-role message and every `tool_calls` field is dropped; an
/// assistant message that has nothing left but its `tool_calls` is dropped with
/// them.
///
/// What the reranker needs from the conversation is the task and the
/// assistant's own account of it. It does not need the retrieved bulk, which it
/// has no way to judge candidates against and which dominated the prompt:
/// across a reproduced trace pool the resent conversation ran 101KB at the
/// median and 345KB at p90, 83 percent of it prior tool results.
///
/// Dropping them also ends the provider-cache-hit property the old prefix had
/// -- this is no longer a copy of the conversation, so it cannot match one --
/// which is the trade the size makes worth taking.
///
/// Adjacent assistant messages are merged. The tool loop pushes exactly one
/// assistant message per assistant turn, so removing the tool results between
/// two of them would otherwise hand a backend a run of same-role messages this
/// harness has never sent (`bedrock_client::convert_messages` and the
/// OpenAI-compatible client both forward the list as given).
fn prefix_for_rerank(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let cut = messages
        .iter()
        .rposition(|m| m.role == "assistant" && m.tool_calls.is_some())
        .unwrap_or(messages.len());
    let mut prefix: Vec<ChatMessage> = Vec::new();
    for message in &messages[..cut] {
        match message.role.as_str() {
            "system" | "user" => prefix.push(message.clone()),
            "assistant" => {
                let text = message.content_text();
                if text.trim().is_empty() {
                    continue;
                }
                match prefix.last_mut() {
                    Some(previous) if previous.role == "assistant" => {
                        previous.content = vec![ChatContentPart::text(format!(
                            "{}\n\n{text}",
                            previous.content_text()
                        ))];
                        if let Some(reasoning) = &message.reasoning_content {
                            previous.reasoning_content =
                                Some(match previous.reasoning_content.take() {
                                    Some(existing) => format!("{existing}\n\n{reasoning}"),
                                    None => reasoning.clone(),
                                });
                        }
                    }
                    _ => prefix.push(ChatMessage::assistant_with_reasoning(
                        text,
                        message.reasoning_content.clone(),
                    )),
                }
            }
            _ => {}
        }
    }
    prefix
}

fn build_rerank_prompt(query: &str, candidates: &[Candidate], final_k: usize) -> String {
    let mut out = String::new();
    out.push_str(
        "A code search just ran for the query below and returned these candidate results. \
Each candidate has an id, the symbol or file it refers to, which retrieval signals surfaced it, \
and its structured declaration signatures with ids (a file candidate also carries a summary). \
Decide which candidates are genuinely relevant to the query and the task in this conversation. \
For every selected candidate that has declaration ids, choose the one through five declarations \
most useful for locating the relevant implementation. Omit irrelevant candidates. Order all \
selected candidates from strongest direct evidence to weakest supporting evidence.\n\n",
    );
    out.push_str(&format!(
        "Source bodies are not included. Most candidates are decidable from the name, path, \
signals, and signatures alone, so decide from those. When a candidate's relevance is genuinely \
undecidable that way -- a generic name whose signature says nothing about what it does -- call \
the `{BODY_FETCH_TOOL}` tool with the candidate ids you need, up to {MAX_BODY_FETCH_IDS} per \
call, and it returns their source. Ask only for the candidates you cannot decide.\n\n"
    ));
    out.push_str(&format!(
        "Respond with ONLY a JSON object of the form {{\"relevant\": \
[{{\"id\":\"<candidate-id>\",\"declarations\":[\"<declaration-id>\", ...]}}]}} using \
the exact ids shown. Select at most {final_k} candidates and at most \
{MAX_SELECTED_DECLARATIONS} declarations per candidate. A selected candidate with declaration \
ids must select at least one; a candidate with no declaration ids must use an empty array. \
If nothing is relevant, return \
{{\"relevant\": []}}.\n\n"
    ));
    out.push_str("<query>\n");
    out.push_str(query.trim());
    out.push_str("\n</query>\n\n<candidates>\n");
    for candidate in candidates {
        out.push_str(&render_candidate_for_prompt(candidate));
        out.push('\n');
    }
    out.push_str("</candidates>\n");
    out
}

fn render_candidate_for_prompt(candidate: &Candidate) -> String {
    let kind = match candidate.kind {
        CandidateKind::Symbol => "symbol",
        CandidateKind::File => "file",
    };
    let signals = candidate.signals.join(", ");
    let mut out = format!("[{}] {kind} {} ({signals})\n", candidate.id, candidate.name);
    if let Some(location) = &candidate.location {
        out.push_str(&location.to_string());
        out.push('\n');
    }
    if candidate.declarations.is_empty() {
        out.push_str("(no structured signatures available)\n");
    } else {
        out.push_str("Structured signatures:\n");
        for declaration in &candidate.declarations {
            out.push_str(&render_declaration_for_prompt(declaration));
        }
    }
    if let Some(context) = &candidate.context {
        out.push_str("```\n");
        out.push_str(context);
        out.push_str("\n```\n");
    }
    out
}

fn render_declaration_for_prompt(declaration: &DeclarationLocator) -> String {
    format!(
        "[{}] {} {} at {}:{}-{}\n{}\n",
        declaration.id,
        declaration.kind,
        declaration.symbol,
        declaration.path,
        declaration.start_line,
        declaration.end_line,
        declaration.signature
    )
}

/// Resolve the model's selected ids back to candidates, preserving model order
/// and rejecting unknown or duplicate candidate/declaration ids.
fn order_candidates<'a>(
    candidates: &'a [Candidate],
    selected: &[Selection],
    final_k: usize,
) -> Result<Vec<RankedCandidate<'a>>, String> {
    let by_id: HashMap<&str, &Candidate> = candidates.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    if selected.len() > final_k {
        return Err(format!(
            "selected {} candidates, exceeding k={final_k}",
            selected.len()
        ));
    }
    for selection in selected {
        if !seen.insert(selection.id.as_str()) {
            return Err(format!("duplicate candidate id {}", selection.id));
        }
        let Some(candidate) = by_id.get(selection.id.as_str()).copied() else {
            return Err(format!("unknown candidate id {}", selection.id));
        };
        let declarations_by_id: HashMap<&str, &DeclarationLocator> = candidate
            .declarations
            .iter()
            .map(|declaration| (declaration.id.as_str(), declaration))
            .collect();
        let mut declaration_seen = HashSet::new();
        let mut declarations =
            Vec::with_capacity(selection.declarations.len().min(MAX_SELECTED_DECLARATIONS));
        for id in &selection.declarations {
            if !declaration_seen.insert(id.as_str()) {
                continue;
            }
            if let Some(declaration) = declarations_by_id.get(id.as_str()).copied() {
                declarations.push(declaration);
                if declarations.len() == MAX_SELECTED_DECLARATIONS {
                    break;
                }
            }
        }
        let declaration_fallback = declarations.is_empty() && !candidate.declarations.is_empty();
        if declaration_fallback {
            declarations.extend(
                candidate
                    .declarations
                    .iter()
                    .take(MAX_SELECTED_DECLARATIONS),
            );
        }
        ordered.push(RankedCandidate {
            candidate,
            declarations,
            declaration_fallback,
        });
    }
    Ok(ordered)
}

struct RerankTrace<'a> {
    query: &'a str,
    query_index: usize,
    query_count: usize,
    raw: &'a Value,
    final_k: usize,
    base_k: usize,
    deduplicated_count: usize,
    context_bytes: usize,
    selected_count: usize,
    final_count: usize,
    signature_candidate_count: usize,
    signature_locator_count: usize,
    final_with_signature_count: usize,
    declaration_fallback_count: usize,
    body_fetch_rounds: usize,
    bodies_fetched: usize,
    failure_reason: Option<&'a str>,
    usage: TokenUsage,
    utility: &'a crate::utility_model::UtilityModelSelection,
}

/// The identity every `semantic_search_phase` record carries for one query.
struct PhaseTrace<'a> {
    query: &'a str,
    query_index: usize,
    query_count: usize,
    utility: &'a crate::utility_model::UtilityModelSelection,
    started: Instant,
}

impl PhaseTrace<'_> {
    fn record(&self, phase: &str, error: Option<&str>) {
        append_trace_record(self.base(phase, error));
    }

    /// One body-fetch round. Same record type as every other phase, so a reader
    /// sees the rounds in sequence with the requests around them.
    fn record_tool_round(&self, round: usize, id_count: usize) {
        let mut record = self.base("utility_tool_round", None);
        record["round"] = json!(round);
        record["id_count"] = json!(id_count);
        append_trace_record(record);
    }

    fn base(&self, phase: &str, error: Option<&str>) -> Value {
        json!({
            "type": "semantic_search_phase",
            "phase": phase,
            "query": self.query,
            "query_index": self.query_index,
            "query_count": self.query_count,
            "elapsed_millis": self.started.elapsed().as_millis(),
            "utility_model": self.utility.model,
            "utility_reasoning_effort": self.utility.reasoning_effort,
            "utility_model_source": self.utility.source,
            "error": error,
        })
    }
}

fn trace_rerank(trace: RerankTrace<'_>) {
    // A leg the response does not carry is reported as null, not as zero. Zero
    // is a real measurement -- r26 ran the `semantic-coedit-2-1` budget, whose
    // bm25 leg was present with a zero depth -- and a dense-only server that
    // has no lexical leg at all must not be averaged in as though it had asked
    // and got nothing. cimeval's `mean_realized_*` skips nulls and keeps its
    // r26 numbers.
    let realized =
        |key: &str| -> Option<usize> { trace.raw.get(key).and_then(Value::as_array).map(Vec::len) };
    append_trace_record(json!({
        "type": "semantic_search_rerank",
        "query": trace.query,
        "query_index": trace.query_index,
        "query_count": trace.query_count,
        "requested_final_k": trace.final_k,
        "forwarded_base_k": trace.base_k,
        // Null against a current bifrost: c353c862 removed the profile knob
        // and fixed the budget at the arm the sweep chose. The field stays so a
        // reader can tell a dense-only run from an r26 one that named its arm.
        "retrieval_profile": diagnostic(trace.raw, "retrieval_profile"),
        "requested_leg_counts": diagnostic(trace.raw, "requested_leg_counts"),
        "retrieval_timings": diagnostic(trace.raw, "timings"),
        "realized_vector": realized("vector_ranked"),
        "realized_bm25": realized("bm25_ranked"),
        "realized_coedit": realized("coedit_ranked"),
        "realized_leg_counts": {
            "vector": realized("vector_ranked"),
            "bm25": realized("bm25_ranked"),
            "coedit": realized("coedit_ranked"),
        },
        "realized_dedup_candidates": trace.deduplicated_count,
        "context_bytes": trace.context_bytes,
        "reranker_selected_count": trace.selected_count,
        "selected_final_count": trace.final_count,
        "structured_signature_candidate_count": trace.signature_candidate_count,
        "structured_signature_locator_count": trace.signature_locator_count,
        "selected_with_signature_count": trace.final_with_signature_count,
        "declaration_selection_fallback_count": trace.declaration_fallback_count,
        // On-demand source: how many rounds the utility model spent on
        // `fetch_candidate_sources` and how many candidate bodies it received.
        // Zero on both is the expected shape; the prompt is signatures-first.
        "body_fetch_rounds": trace.body_fetch_rounds,
        "bodies_fetched": trace.bodies_fetched,
        // Keep the old fields so trace readers can distinguish this behavior
        // from old runs that silently used reciprocal-rank fusion.
        "fallback": false,
        "fallback_reason": null,
        "failed": trace.failure_reason.is_some(),
        "failure_reason": trace.failure_reason,
        "utility_model": trace.utility.model,
        "utility_reasoning_effort": trace.utility.reasoning_effort,
        "utility_model_source": trace.utility.source,
        "reranker_usage": {
            "input_tokens": trace.usage.input_tokens,
            "output_tokens": trace.usage.output_tokens,
            "thought_tokens": trace.usage.thought_tokens,
            "cached_read_tokens": trace.usage.cached_read_tokens,
            "cached_write_tokens": trace.usage.cached_write_tokens,
            "total_tokens": trace.usage.total_tokens(),
        },
    }));
}

fn diagnostic<'a>(raw: &'a Value, key: &str) -> Option<&'a Value> {
    raw.get(key)
        .or_else(|| raw.get("diagnostics").and_then(|value| value.get(key)))
}

fn render_unified(
    query: &str,
    ordered: &[RankedCandidate<'_>],
    candidate_count: usize,
    notes: Option<String>,
) -> String {
    let mut out = format!(
        "Reranked {} relevant result(s) (from {} candidates) for query: \"{}\"\n",
        ordered.len(),
        candidate_count,
        query.trim()
    );
    if let Some(notes) = notes {
        out.push_str(&notes);
        out.push('\n');
    }
    out.push('\n');
    for (rank, selected) in ordered.iter().enumerate() {
        let candidate = selected.candidate;
        let kind = match candidate.kind {
            CandidateKind::Symbol => "symbol",
            CandidateKind::File => "file",
        };
        out.push_str(&format!(
            "{}. {} {} [{}]\n",
            rank + 1,
            kind,
            candidate.name,
            candidate.signals.join(", ")
        ));
        // A declaration carries its own path and a range that is tighter than
        // the chunk's, so the chunk location only earns a line when it names a
        // file no declaration names. That happens on a C++ header/implementation
        // split, where the chunk is the .cpp body and the declaration the .h.
        let location_names_another_file = match &candidate.location {
            None => false,
            Some(location) => {
                selected.declarations.is_empty()
                    || selected
                        .declarations
                        .iter()
                        .any(|declaration| declaration.path != location.path)
            }
        };
        if let Some(location) = &candidate.location
            && location_names_another_file
        {
            out.push_str(&format!("   {location}\n"));
        }
        if selected.declarations.is_empty() {
            out.push_str("   signature unavailable\n");
        } else {
            for declaration in &selected.declarations {
                // For a symbol candidate bifrost returns the declaration of the
                // symbol that was asked for, so repeating the name here says
                // nothing. A file candidate's declarations name the symbols
                // inside that file, which the header line does not, so they stay.
                let named = if declaration.symbol == candidate.name {
                    String::new()
                } else {
                    format!(" {}", declaration.symbol)
                };
                out.push_str(&format!(
                    "   {}{} at {}:{}-{}\n",
                    declaration.kind,
                    named,
                    declaration.path,
                    declaration.start_line,
                    declaration.end_line
                ));
                for line in declaration.signature.lines() {
                    out.push_str("      ");
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

fn notes(raw: &Value) -> Option<String> {
    let array = raw.get("notes").and_then(Value::as_array)?;
    let joined: Vec<&str> = array.iter().filter_map(Value::as_str).collect();
    if joined.is_empty() {
        None
    } else {
        Some(joined.join("\n"))
    }
}

fn rerank_schema(final_k: usize) -> Value {
    json!({
        "type": "object",
        "properties": {
            "relevant": {
                "type": "array",
                "maxItems": final_k,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "declarations": {
                            "type": "array",
                            "maxItems": MAX_SELECTED_DECLARATIONS,
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["id", "declarations"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["relevant"],
        "additionalProperties": false
    })
}

/// Parse the strict candidate/declaration selection object from a model response. Native
/// structured-output backends return clean JSON by construction; the prompt
/// instructs the rest to do the same. Anything that doesn't parse yields `None`,
/// which invokes deterministic RRF fallback -- no bespoke extraction needed.
fn parse_selections(text: &str) -> Option<Vec<Selection>> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    let object = value.as_object()?;
    if object.len() != 1 {
        return None;
    }
    let array = object.get("relevant")?.as_array()?;
    array
        .iter()
        .map(|value| {
            let object = value.as_object()?;
            if object.len() != 2 {
                return None;
            }
            let id = object.get("id")?.as_str()?.to_string();
            let declarations = object
                .get("declarations")?
                .as_array()?
                .iter()
                .map(|value| value.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()?;
            Some(Selection { id, declarations })
        })
        .collect()
}

fn sample_source(text: &str) -> String {
    head_tail_sample(text, 50, 30, 20)
}

fn sample_summary(text: &str) -> String {
    head_tail_sample(text, 120, 90, 30)
}

/// Show the first `head` and last `tail` lines when `content` exceeds
/// `max_lines`, with an elision marker between; clamp pathologically wide lines
/// to `MAX_LINE_CHARS` on a UTF-8 boundary. Mirrors bifrost's `model_context::cap`
/// (which is in a separate process and so not reusable here), with the 30/20
/// split this reranker wants.
fn head_tail_sample(content: &str, max_lines: usize, head: usize, tail: usize) -> String {
    let clamped: Vec<String> = content.lines().map(clamp_line).collect();
    let total = clamped.len();
    if total <= max_lines {
        return clamped.join("\n");
    }
    let head = head.min(total);
    let tail = tail.min(total - head);
    let omitted = total - head - tail;
    let mut out: Vec<String> = Vec::with_capacity(head + tail + 1);
    out.extend(clamped[..head].iter().cloned());
    out.push(format!("… {omitted} lines omitted …"));
    out.extend(clamped[total - tail..].iter().cloned());
    out.join("\n")
}

fn clamp_line(line: &str) -> String {
    if line.len() <= MAX_LINE_CHARS {
        return line.to_string();
    }
    let mut cut = MAX_LINE_CHARS;
    while !line.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}… [line truncated]", &line[..cut])
}

/// A fake bifrost that serves the three tools this reranker uses. Two of its
/// symbols share a body, so a fetch of both exercises deduplication.
///
/// Three argv switches shape how it fails:
///
/// - `fail_sources` makes `get_symbol_sources` return a JSON-RPC error while
///   search and summaries keep working, which is the fail-closed case.
/// - `saturate_summaries=<count>` makes the first `count` `get_summaries` calls
///   answer with bifrost's analyzer-pool overflow error, then serve normally.
///   That is the backpressure signal, so a count of one exercises recovery and a
///   count past the retry budget exercises the give-up path.
/// - `fail_workspace=<name>`, repeatable, makes every call that names that
///   workspace fail. One failing workspace out of several is what a forced CIM
///   step zero has to survive.
///
/// It takes its workspace shape from its own command line, the way
/// `McpServerConfig::rendered_args` writes it and the readiness fake
/// (`semantic_readiness::FAKE_BIFROST`) reads it:
///
/// - No `--workspace` argument: single-root mode, and a `workspace` argument is
///   rejected because there is no router to route it.
/// - One or more `--workspace <name>=<path>`: named mode, where every call must
///   name a configured workspace or be refused with bifrost's own
///   `invalid_params` message. Each workspace answers under its own symbol
///   prefix, so a caller can tell which workspace served a result.
///
/// Module scope rather than inside `mod tests` so another module's tests can
/// drive a whole `semantic_search` call against it; `cim`'s step zero does.
#[cfg(test)]
#[cfg(unix)]
pub(crate) const FAKE_BIFROST_WITH_SOURCES: &str = r#"
import json, sys

argv = sys.argv[1:]
fail_sources = "fail_sources" in argv

named = {}
saturate_summaries = 0
fail_workspaces = set()
i = 0
while i < len(argv):
    if argv[i] == "--workspace" and i + 1 < len(argv):
        name, _, path = argv[i + 1].partition("=")
        named[name] = path
        i += 2
        continue
    if argv[i].startswith("saturate_summaries="):
        saturate_summaries = int(argv[i].split("=", 1)[1])
    elif argv[i].startswith("fail_workspace="):
        fail_workspaces.add(argv[i].split("=", 1)[1])
    i += 1

PATHS = {
    "app.alpha.run": "app/alpha.py",
    "app.beta.run": "app/beta.py",
    "app.gamma.helper": "app/gamma.py",
}
BODIES = {
    "app.alpha.run": "def run(self):\n    return self.value",
    "app.beta.run": "def run(self):\n    return self.value",
    "app.gamma.helper": "def helper(self):\n    return 42",
}

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    method = msg.get("method")
    if mid is None:
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "bifrost", "version": "0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": n, "description": "fake",
             "inputSchema": {"type": "object", "properties": {}}}
            for n in ["semantic_search", "get_summaries", "get_symbol_sources"]]}})
    elif method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments", {})
        workspace = args.get("workspace")
        if named:
            if not isinstance(workspace, str) or workspace not in named:
                send({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32602, "message": "workspace must be one configured name"}})
                continue
        elif workspace is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32602,
                "message": "workspace is not accepted without configured workspaces"}})
            continue
        if workspace in fail_workspaces:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32603,
                "message": "workspace " + str(workspace) + " index is unavailable"}})
            continue
        prefix = (workspace + ".") if named else ""
        if name == "semantic_search":
            send({"jsonrpc": "2.0", "id": mid, "result": {"structuredContent": {
                "vector_ranked": [{"fqfn": prefix + s, "score": 1.0} for s in
                                  ["app.alpha.run", "app.beta.run", "app.gamma.helper"]],
                "coedit_ranked": []}}})
        elif name == "get_summaries":
            if saturate_summaries > 0:
                saturate_summaries -= 1
                send({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32603,
                    "message": "too many analyzer requests are queued; retry "
                               "get_summaries once earlier calls complete"}})
                continue
            summaries = []
            for target in args.get("targets", []):
                symbol = target[len(prefix):] if target.startswith(prefix) else target
                if symbol not in PATHS:
                    continue
                summaries.append({"label": target, "path": PATHS[symbol], "elements": [
                    {"path": PATHS[symbol], "symbol": target, "kind": "function",
                     "start_line": 1, "end_line": 2,
                     "text": "def " + symbol.split(".")[-1] + "(self):"}]})
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"structuredContent": {"summaries": summaries}}})
        elif name == "get_symbol_sources":
            if fail_sources:
                send({"jsonrpc": "2.0", "id": mid,
                      "error": {"code": -32000, "message": "symbol source index unavailable"}})
                continue
            sources = []
            for target in args.get("symbols", []):
                symbol = target[len(prefix):] if target.startswith(prefix) else target
                if symbol not in BODIES:
                    continue
                sources.append({"label": target, "path": PATHS[symbol],
                                "start_line": 1, "end_line": 2, "text": BODIES[symbol]})
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"structuredContent": {"sources": sources}}})
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "unexpected tool " + name}})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": method}})
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn context_fetch_reports_bifrost_errors() {
        let temp = tempfile::tempdir().expect("temp dir");
        let registry = ToolRegistry::new(
            temp.path().to_path_buf(),
            Vec::new(),
            Vec::new(),
            Arc::new(crate::skills::SkillRegistry::default()),
            Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces: None,
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await;
        let mut candidates = parse_candidates(&json!({
            "vector_ranked": [
                { "fqfn": "example.Type.method", "score": 1.0 }
            ],
            "coedit_ranked": []
        }));

        let error = fetch_context(&registry, &mut candidates, Some("backend"))
            .await
            .expect_err("missing Bifrost context service must fail enrichment");
        assert!(
            format!("{error:#}").contains("get_summaries failed"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn short_source_is_unchanged() {
        let text = "line 1\nline 2\nline 3";
        assert_eq!(head_tail_sample(text, 50, 30, 20), text);
    }

    #[test]
    fn long_source_keeps_head_and_tail_with_marker() {
        let lines: Vec<String> = (1..=100).map(|n| format!("line {n}")).collect();
        let text = lines.join("\n");
        let sampled = head_tail_sample(&text, 50, 30, 20);
        let out: Vec<&str> = sampled.lines().collect();
        assert_eq!(out.len(), 30 + 1 + 20);
        assert_eq!(out[0], "line 1");
        assert_eq!(out[29], "line 30");
        assert_eq!(out[30], "… 50 lines omitted …");
        assert_eq!(out[31], "line 81");
        assert_eq!(out[50], "line 100");
    }

    #[test]
    fn wide_lines_are_clamped_on_char_boundary() {
        let wide = "α".repeat(2000); // 2000 chars, 4000 bytes > MAX_LINE_CHARS
        let clamped = clamp_line(&wide);
        assert!(clamped.ends_with("… [line truncated]"));
        // Truncation must not panic and must land on a boundary.
        assert!(clamped.len() < wide.len());
    }

    /// The shape a current bifrost sends: dense symbols plus co-edited files,
    /// no `bm25_ranked` key at all. Its absence must read as "that leg does not
    /// exist", not as a parse failure or an empty result.
    #[test]
    fn dense_only_response_parses_without_the_removed_bm25_leg() {
        let raw = json!({
            "vector_ranked": [
                { "fqfn": "a.B.c", "score": 0.9 },
                { "fqfn": "a.B.d", "score": 0.5 }
            ],
            "coedit_ranked": [
                { "path": "src/x.rs", "score": 0.3 }
            ],
            "requested_leg_counts": { "vector": 20, "coedit": 10 },
            "timings": { "total_ms": 12 },
            "notes": []
        });
        let candidates = parse_candidates(&raw);
        let names: Vec<&str> = candidates.iter().map(|c| c.name.as_str()).collect();
        // Deterministic RRF order across the legs, not "all symbols, then all
        // files": the top co-edited file ties the second dense symbol on
        // reciprocal rank and wins the tie by first appearance.
        assert_eq!(names, vec!["a.B.c", "src/x.rs", "a.B.d"]);
        for candidate in &candidates {
            let expected = match candidate.kind {
                CandidateKind::Symbol => vec!["vector"],
                CandidateKind::File => vec!["coedit"],
            };
            assert_eq!(candidate.signals, expected, "{}", candidate.name);
        }
        // The rerank prompt is built from exactly these candidates, so a
        // dense-only response still produces a full, model-ready turn.
        let prompt = build_rerank_prompt("find the retry budget", &candidates, DEFAULT_FINAL_K);
        for candidate in &candidates {
            assert!(prompt.contains(&format!("[{}]", candidate.id)));
            assert!(prompt.contains(&candidate.name));
        }
        assert!(
            !prompt.contains("bm25"),
            "no lexical leg exists to advertise to the model: {prompt}"
        );
        // The full rerank path over these candidates: prompt, selection,
        // ordering. A dense-only response reranks like any other.
        let ordered = order_candidates(
            &candidates,
            &[selection("f1"), selection("s2")],
            DEFAULT_FINAL_K,
        )
        .expect("a dense-only response reranks");
        let ordered_names: Vec<&str> = ordered
            .iter()
            .map(|selected| selected.candidate.name.as_str())
            .collect();
        assert_eq!(ordered_names, vec!["src/x.rs", "a.B.d"]);

        // The diagnostics the trace record reads are absent, not zero: a
        // campaign that averages `realized_bm25` must not count a leg that no
        // longer exists as a leg that returned nothing.
        assert!(diagnostic(&raw, "retrieval_profile").is_none());
        assert!(diagnostic(&raw, "requested_leg_counts").is_some());
        assert!(diagnostic(&raw, "timings").is_some());
        assert!(raw.get("bm25_ranked").is_none());
    }

    async fn traced(raw: &Value) -> Value {
        let utility = crate::utility_model::UtilityModelSelection {
            model: "test-model".to_string(),
            reasoning_effort: Some("low".to_string()),
            source: "test",
        };
        let cwd = tempfile::tempdir().expect("temp cwd");
        let path = cwd.path().join("anvil-trace.jsonl");
        crate::trace_logging::with_trace_path(&path, async {
            trace_rerank(RerankTrace {
                query: "q",
                query_index: 0,
                query_count: 1,
                raw,
                final_k: 20,
                base_k: 40,
                deduplicated_count: 0,
                context_bytes: 0,
                selected_count: 0,
                final_count: 0,
                signature_candidate_count: 0,
                signature_locator_count: 0,
                final_with_signature_count: 0,
                declaration_fallback_count: 0,
                body_fetch_rounds: 0,
                bodies_fetched: 0,
                failure_reason: None,
                usage: TokenUsage::default(),
                utility: &utility,
            })
        })
        .await;
        let lines = std::fs::read_to_string(&path).expect("trace written");
        serde_json::from_str(lines.lines().next().expect("one record")).expect("valid json")
    }

    /// A campaign that averages `realized_bm25` skips nulls. A leg the response
    /// never carried must therefore be null, not zero: zero is a real
    /// measurement (r26 ran the `semantic-coedit-2-1` budget, whose bm25 leg
    /// was present at depth zero) and conflating the two would silently fold
    /// dense-only runs into the r26 mean.
    #[tokio::test]
    async fn a_removed_leg_traces_as_absent_and_an_empty_one_as_zero() {
        let dense_only = traced(&json!({
            "vector_ranked": [{ "fqfn": "a.B.c", "score": 0.9 }],
            "coedit_ranked": [],
            "requested_leg_counts": { "vector": 40, "coedit": 20 },
        }))
        .await;
        assert_eq!(dense_only["realized_vector"], 1);
        assert_eq!(dense_only["realized_coedit"], 0);
        assert!(
            dense_only["realized_bm25"].is_null(),
            "a leg bifrost no longer has must not be reported as an empty one: {dense_only}"
        );
        assert!(dense_only["retrieval_profile"].is_null());
        assert_eq!(dense_only["requested_leg_counts"]["vector"], 40);

        let r26_shape = traced(&json!({
            "vector_ranked": [{ "fqfn": "a.B.c", "score": 0.9 }],
            "bm25_ranked": [],
            "coedit_ranked": [],
            "retrieval_profile": "semantic-coedit-2-1",
            "requested_leg_counts": { "vector": 40, "bm25": 0, "coedit": 20 },
        }))
        .await;
        assert_eq!(r26_shape["realized_bm25"], 0);
        assert_eq!(r26_shape["retrieval_profile"], "semantic-coedit-2-1");
    }

    /// An archived response, or one from a bifrost older than c353c862, still
    /// carries the lexical leg. Replay must keep folding it in.
    #[test]
    fn candidates_dedup_symbols_and_union_signals() {
        let raw = json!({
            "vector_ranked": [
                { "fqfn": "a.B.c", "score": 0.9 },
                { "fqfn": "a.B.d", "score": 0.5 }
            ],
            "bm25_ranked": [
                { "fqfn": "a.B.c", "score": 0.7 }
            ],
            "coedit_ranked": [
                { "path": "src/x.rs", "score": 0.3 }
            ]
        });
        let candidates = parse_candidates(&raw);
        assert_eq!(candidates.len(), 3);

        let c = candidates.iter().find(|c| c.name == "a.B.c").unwrap();
        assert_eq!(c.id, "s1"); // top score after sort
        assert_eq!(c.signals, vec!["vector", "bm25"]);

        let file = candidates.iter().find(|c| c.name == "src/x.rs").unwrap();
        assert_eq!(file.id, "f1");
        assert_eq!(file.kind, CandidateKind::File);
        assert_eq!(file.signals, vec!["coedit"]);
    }

    #[test]
    fn final_k_defaults_and_validates_model_boundary() {
        assert_eq!(parse_final_k(&json!({ "queries": ["x"] })), Ok(20));
        assert_eq!(parse_final_k(&json!({ "queries": ["x"], "k": 1 })), Ok(1));
        assert_eq!(parse_final_k(&json!({ "queries": ["x"], "k": 20 })), Ok(20));
        for invalid in [json!(0), json!(21), json!(1.5), json!("20"), Value::Null] {
            assert!(parse_final_k(&json!({ "k": invalid })).is_err());
        }
    }

    #[test]
    fn queries_validate_normalization_uniqueness_and_batch_limit() {
        assert_eq!(
            parse_queries(&json!({ "queries": ["find auth", "locate refresh"] })),
            Ok(vec!["find auth".to_string(), "locate refresh".to_string()])
        );
        for invalid in [
            json!({ "query": "old scalar" }),
            json!({ "queries": [] }),
            json!({ "queries": ["one", "two", "three", "four"] }),
            json!({ "queries": [" leading"] }),
            json!({ "queries": ["Same", "same"] }),
        ] {
            assert!(parse_queries(&invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn bifrost_receives_twice_the_final_k() {
        for final_k in [1, 7, 20] {
            let (forwarded, base_k) = prepare_bifrost_args("needle", final_k, None);
            assert_eq!(base_k, 2 * final_k);
            assert_eq!(forwarded["k"], 2 * final_k);
            assert_eq!(forwarded["query"], "needle");
            assert!(forwarded.get("queries").is_none());
        }
    }

    #[test]
    fn bifrost_receives_the_selected_workspace() {
        let (forwarded, _) = prepare_bifrost_args("needle", 10, Some("backend"));
        assert_eq!(forwarded["workspace"], "backend");
    }

    #[test]
    fn parses_and_prompts_entire_120_candidate_pool() {
        let vector: Vec<Value> = (0..40)
            .map(|i| json!({ "fqfn": format!("vector::{i}"), "score": 1.0 }))
            .collect();
        let bm25: Vec<Value> = (0..40)
            .map(|i| json!({ "fqfn": format!("bm25::{i}"), "score": 1.0 }))
            .collect();
        let coedit: Vec<Value> = (0..40)
            .map(|i| json!({ "path": format!("src/{i}.rs"), "score": 1.0 }))
            .collect();
        let candidates = parse_candidates(&json!({
            "vector_ranked": vector,
            "bm25_ranked": bm25,
            "coedit_ranked": coedit,
        }));
        assert_eq!(candidates.len(), 120);
        let prompt = build_rerank_prompt("query", &candidates, 20);
        for candidate in &candidates {
            assert!(prompt.contains(&format!("[{}]", candidate.id)));
            assert!(prompt.contains(&candidate.name));
        }
        assert!(prompt.contains("strongest direct evidence to weakest supporting evidence"));
    }

    #[test]
    fn parses_relevant_object_and_rejects_non_json() {
        assert_eq!(
            parse_selections(
                r#"{"relevant":[{"id":"s1","declarations":["s1d1"]},{"id":"f2","declarations":[]}]}"#
            ),
            Some(vec![
                Selection {
                    id: "s1".to_string(),
                    declarations: vec!["s1d1".to_string()]
                },
                Selection {
                    id: "f2".to_string(),
                    declarations: Vec::new()
                }
            ])
        );
        assert_eq!(parse_selections("  {\"relevant\": []}  "), Some(Vec::new()));
        // Non-JSON / fenced output does not parse; the caller falls back to
        // raw passthrough.
        assert_eq!(parse_selections("```json\n{\"relevant\": []}\n```"), None);
        assert_eq!(parse_selections("no json here"), None);
        assert_eq!(parse_selections(r#"{"relevant":["s1",3]}"#), None);
        assert_eq!(
            parse_selections(r#"{"relevant":[],"unexpected":true}"#),
            None
        );
    }

    #[test]
    fn order_candidates_preserves_model_order_and_rejects_unknown() {
        let candidates = parse_candidates(&json!({
            "vector_ranked": [
                { "fqfn": "a.B.c", "score": 0.9 },
                { "fqfn": "a.B.d", "score": 0.5 }
            ]
        }));
        let selected = vec![selection("s2"), selection("s1")];
        let ordered = order_candidates(&candidates, &selected, 20).unwrap();
        let names: Vec<&str> = ordered
            .iter()
            .map(|selected| selected.candidate.name.as_str())
            .collect();
        assert_eq!(names, vec!["a.B.d", "a.B.c"]);
        assert!(order_candidates(&candidates, &[selection("nope")], 20).is_err());
        assert!(order_candidates(&candidates, &[selection("s1"), selection("s1")], 20).is_err());
    }

    #[test]
    fn valid_empty_and_fewer_selections_are_final_but_overlong_is_invalid() {
        let candidates = parse_candidates(&json!({
            "vector_ranked": (0..6)
                .map(|i| json!({ "fqfn": format!("item::{i}") }))
                .collect::<Vec<_>>()
        }));
        assert!(order_candidates(&candidates, &[], 3).unwrap().is_empty());

        let fewer = order_candidates(&candidates, &[selection("s2")], 3).unwrap();
        assert_eq!(fewer.len(), 1);

        let selected: Vec<Selection> = (1..=6).map(|i| selection(&format!("s{i}"))).collect();
        assert!(order_candidates(&candidates, &selected, 3).is_err());
    }

    #[test]
    fn context_budget_is_utf8_safe_and_preserves_all_identities() {
        let context = "α".repeat(5_000);
        let mut candidates: Vec<Candidate> = (0..20)
            .map(|i| {
                live_candidate(
                    &format!("s{}", i + 1),
                    CandidateKind::Symbol,
                    &format!("item::{i}"),
                    vec!["vector"],
                    &context,
                )
            })
            .collect();
        let used = bound_candidate_context(&mut candidates);
        assert_eq!(used, MAX_TOTAL_CONTEXT_BYTES);
        assert!(
            candidates
                .iter()
                .filter_map(|candidate| candidate.context.as_ref())
                .all(|context| context.len() <= MAX_CANDIDATE_CONTEXT_BYTES)
        );
        let prompt = build_rerank_prompt("query", &candidates, 20);
        for candidate in &candidates {
            assert!(prompt.contains(&candidate.name));
        }
    }

    /// A macro-heavy candidate can carry hundreds of declarations. One with 290
    /// was 27 percent of a reproduced prompt on its own, so only the first few
    /// are rendered and only those can be cited back.
    #[test]
    fn a_candidate_with_many_declarations_renders_only_the_first_few() {
        let elements: Vec<Value> = (1..=40)
            .map(|i| {
                json!({
                    "path": "test/suite.cc",
                    "symbol": format!("Suite_Case{i}_Test"),
                    "kind": "class",
                    "start_line": i,
                    "end_line": i + 1,
                    "text": format!("class Suite_Case{i}_Test")
                })
            })
            .collect();
        let mut candidates = parse_candidates(&json!({
            "coedit_ranked": [{ "path": "test/suite.cc", "score": 1.0 }]
        }));
        attach_summaries(
            &mut candidates,
            &json!({ "summaries": [{ "label": "test/suite.cc", "path": "test/suite.cc", "elements": elements }] }),
        );
        assert_eq!(candidates[0].declarations.len(), 40);

        bound_candidate_context(&mut candidates);
        assert_eq!(candidates[0].declarations.len(), MAX_PROMPT_DECLARATIONS);
        let prompt = build_rerank_prompt("gtest cases", &candidates, 20);
        assert!(prompt.contains("[f1d8]"), "{prompt}");
        assert!(!prompt.contains("[f1d9]"), "{prompt}");
        assert!(order_candidates(&candidates, &[selection("f1")], 20).is_ok());
    }

    #[test]
    fn compact_file_outlines_are_valid_reranker_context() {
        let mut candidates = parse_candidates(&json!({
            "coedit_ranked": [{ "path": "src/config.rs", "score": 1.0 }]
        }));
        attach_summaries(
            &mut candidates,
            &json!({
                "summaries": [],
                "compact_symbols": {
                    "files": [{
                        "path": "src/config.rs",
                        "loc": 42,
                        "lines": ["- Config", "  - load"]
                    }]
                },
                "degraded": true
            }),
        );

        let context = candidates[0]
            .context
            .as_deref()
            .expect("compact outline is attached");
        assert_eq!(context, "src/config.rs (42 lines)\n- Config\n  - load");
    }

    #[test]
    fn public_locator_cards_use_classifier_selected_bifrost_signatures_only() {
        let mut candidates = parse_candidates(&json!({
            "coedit_ranked": [{ "path": "src/config.rs", "score": 1.0 }]
        }));
        attach_summaries(
            &mut candidates,
            &json!({
                "summaries": [{
                    "label": "src/config.rs",
                    "path": "src/config.rs",
                    "preamble": "PRIVATE_BODY_SENTINEL",
                    "elements": [
                        {"path":"src/config.rs","symbol":"load","kind":"function","start_line":10,"end_line":12,"text":"fn load(path: &Path) -> Config"},
                        {"path":"src/config.rs","symbol":"save","kind":"function","start_line":20,"end_line":23,"text":"fn save(config: &Config)"}
                    ]
                }]
            }),
        );
        bound_candidate_context(&mut candidates);
        let selected = vec![Selection {
            id: "f1".to_string(),
            declarations: vec!["f1d2".to_string()],
        }];
        let ordered = order_candidates(&candidates, &selected, 20).unwrap();
        let rendered = render_unified("configuration persistence", &ordered, 1, None);

        assert!(rendered.contains("fn save(config: &Config)"));
        assert!(!rendered.contains("fn load(path: &Path)"));
        assert!(!rendered.contains("PRIVATE_BODY_SENTINEL"));
        assert!(!rendered.contains("```"));

        let fallback = order_candidates(&candidates, &[selection("f1")], 20).unwrap();
        assert!(fallback[0].declaration_fallback);
        assert_eq!(fallback[0].declarations.len(), 2);
    }

    /// The agent card must not say the same thing twice. Measured over a
    /// 16-task CodeScale arm, the declaration line repeated the candidate name
    /// on all 1,376 declarations, and repeated the chunk's file on 88 percent
    /// of them. The remaining 12 percent are a C++ header/implementation split,
    /// where the chunk is the .cpp body and the declaration the .h, and there
    /// the chunk location is the only place the .cpp appears. The chunk
    /// location now reaches the card only for a candidate the utility model
    /// fetched the source of, which is what `take_symbol_sources` records.
    #[test]
    fn agent_card_drops_the_repeated_name_but_keeps_a_second_file() {
        let card = |raw: Value, summaries: Value, id: &str| {
            let mut candidates = parse_candidates(&raw);
            take_symbol_sources(&mut candidates, &summaries);
            attach_summaries(&mut candidates, &summaries);
            let ordered = order_candidates(&candidates, &[selection(id)], 20).unwrap();
            render_unified("q", &ordered, 1, None)
        };

        // Declaration in the same file as the chunk: the name and the file are
        // stated once each.
        let same_file = card(
            json!({ "vector_ranked": [{ "fqfn": "app.handlers.load", "score": 1.0 }] }),
            json!({
                "sources": [{"label":"app.handlers.load","path":"app/handlers.py","start_line":27,"end_line":104,"text":"body"}],
                "summaries": [{"label":"app.handlers.load","path":"app/handlers.py","elements":[
                    {"path":"app/handlers.py","symbol":"app.handlers.load","kind":"function","start_line":27,"end_line":104,"text":"def load(self):"}
                ]}]
            }),
            "s1",
        );
        assert!(same_file.contains("1. symbol app.handlers.load [vector]"));
        assert!(same_file.contains("   function at app/handlers.py:27-104"));
        assert!(same_file.contains("def load(self):"));
        assert_eq!(same_file.matches("app.handlers.load").count(), 1);
        assert_eq!(same_file.matches("app/handlers.py").count(), 1);

        // Declaration in a different file: both files survive, the name still
        // appears once.
        let split = card(
            json!({ "vector_ranked": [{ "fqfn": "llvm.IntegerType.get", "score": 1.0 }] }),
            json!({
                "sources": [{"label":"llvm.IntegerType.get","path":"lib/IR/Type.cpp","start_line":318,"end_line":340,"text":"body"}],
                "summaries": [{"label":"llvm.IntegerType.get","path":"lib/IR/Type.cpp","elements":[
                    {"path":"include/llvm/IR/DerivedTypes.h","symbol":"llvm.IntegerType.get","kind":"function","start_line":66,"end_line":66,"text":"static IntegerType *get();"}
                ]}]
            }),
            "s1",
        );
        assert!(split.contains("   lib/IR/Type.cpp:318-340"));
        assert!(split.contains("   function at include/llvm/IR/DerivedTypes.h:66-66"));
        assert_eq!(split.matches("llvm.IntegerType.get").count(), 1);

        // A file candidate's declarations name symbols the header line does not,
        // so those names must stay.
        let file = card(
            json!({ "coedit_ranked": [{ "path": "src/config.rs", "score": 1.0 }] }),
            json!({
                "summaries": [{"label":"src/config.rs","path":"src/config.rs","elements":[
                    {"path":"src/config.rs","symbol":"load","kind":"function","start_line":10,"end_line":12,"text":"fn load()"}
                ]}]
            }),
            "f1",
        );
        assert!(file.contains("1. file src/config.rs [coedit]"));
        assert!(file.contains("   function load at src/config.rs:10-12"));
    }

    fn live_candidate(
        id: &str,
        kind: CandidateKind,
        name: &str,
        signals: Vec<&'static str>,
        context: &str,
    ) -> Candidate {
        Candidate {
            id: id.to_string(),
            kind,
            name: name.to_string(),
            signals,
            location: None,
            context: Some(context.to_string()),
            declarations: Vec::new(),
            rrf_score: 0.0,
            first_seen: 0,
        }
    }

    /// Live end-to-end of the disposable rerank turn against hosted DeepSeek
    /// `deepseek-v4-flash`. DeepSeek is a non-native structured-output backend,
    /// so `response_format` is dropped and the model must produce clean
    /// `{"relevant": [...]}` JSON from the prompt alone -- exactly the path this
    /// reranker depends on. Exercises the real prompt, model call, and parser.
    #[tokio::test]
    #[ignore = "live network test; requires DEEPSEEK_API_KEY"]
    async fn rerank_turn_with_deepseek_v4_flash_live() {
        let key = std::env::var(crate::discovery::DEEPSEEK_API_KEY_ENV)
            .expect("DEEPSEEK_API_KEY must be set for the live rerank test");
        let llm = crate::deepseek_backend_from_key(&key).expect("deepseek backend builds");

        let candidates = vec![
            live_candidate(
                "s1",
                CandidateKind::Symbol,
                "config::parse_config",
                vec!["vector", "bm25"],
                "pub fn parse_config(path: &Path) -> Result<Config> {\n    let raw = std::fs::read_to_string(path)?;\n    let config: Config = serde_json::from_str(&raw)?;\n    Ok(config)\n}",
            ),
            live_candidate(
                "s2",
                CandidateKind::Symbol,
                "math::add",
                vec!["vector"],
                "pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}",
            ),
            live_candidate(
                "f1",
                CandidateKind::File,
                "src/config.rs",
                vec!["coedit"],
                "Loads and parses application configuration from JSON files; defines Config and parse_config.",
            ),
            live_candidate(
                "f2",
                CandidateKind::File,
                "src/ui/button.rs",
                vec!["coedit"],
                "Renders a clickable UI button widget and handles hover/press visual state.",
            ),
        ];

        let prompt = build_rerank_prompt(
            "where is JSON config file parsing implemented?",
            &candidates,
            20,
        );
        let messages = vec![ChatMessage::user(prompt)];
        let structured = StructuredOutputRequest {
            schema_name: "semantic_rerank".to_string(),
            schema: rerank_schema(20),
            allow_coercion: true,
            prefer_json_object: false,
        };
        let cancel = CancellationToken::new();

        let response = stream_chat_no_visible_output_with_retry(
            llm.as_ref(),
            "semantic_search rerank live test",
            &cancel,
            || StreamChatRequest {
                model: "deepseek-v4-flash".to_string(),
                messages: messages.clone(),
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: Some(structured.clone()),
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeouts: IdleTimeouts {
                    first_progress: std::time::Duration::from_secs(60),
                    inter_chunk: std::time::Duration::from_secs(60),
                },
            },
        )
        .await
        .expect("deepseek rerank request succeeds");

        let text = match response {
            LlmResponse::Text { text, .. } | LlmResponse::ToolCalls { text, .. } => text,
        };
        eprintln!("deepseek-v4-flash raw rerank response:\n{text}");

        let selected = parse_selections(&text).expect("response parses as {\"relevant\": [...]}");
        let valid: std::collections::HashSet<&str> =
            candidates.iter().map(|c| c.id.as_str()).collect();
        assert!(!selected.is_empty(), "expected at least one relevant id");
        assert!(
            selected
                .iter()
                .all(|selection| valid.contains(selection.id.as_str())),
            "all returned ids must be known candidate ids, got {selected:?}"
        );
        assert!(
            selected
                .iter()
                .any(|selection| selection.id == "s1" || selection.id == "f1"),
            "expected a config-parsing candidate (s1/f1) to be judged relevant, got {selected:?}"
        );
        // The trivially-irrelevant UI button should not be selected.
        assert!(
            !selected.iter().any(|selection| selection.id == "f2"),
            "did not expect the UI button file to be relevant, got {selected:?}"
        );
    }

    /// The tool loop dispatches `semantic_search` into the reranker instead of
    /// `execute_tool`, so the reranker owns the `tool_timing` record. Without
    /// it the tool is invisible to every trace consumer that counts calls per
    /// tool. This drives the reranker down its no-bifrost error path, which
    /// needs no MCP server and never reaches the LLM.
    #[tokio::test]
    async fn rerank_records_tool_timing_like_every_other_tool() {
        struct UnusedBackend;
        impl LlmBackend for UnusedBackend {
            fn list_models(&self) -> futures::future::BoxFuture<'_, anyhow::Result<Vec<String>>> {
                unimplemented!("the rerank turn is never reached in this test")
            }

            fn stream_chat(
                &self,
                _request: StreamChatRequest,
            ) -> futures::future::BoxFuture<'_, anyhow::Result<LlmResponse>> {
                unimplemented!("the rerank turn is never reached in this test")
            }
        }

        let cwd = tempfile::tempdir().expect("temp cwd");
        let trace = cwd.path().join("anvil-trace.jsonl");
        let registry = ToolRegistry::new(
            cwd.path().to_path_buf(),
            Vec::new(),
            Vec::new(),
            Arc::new(crate::skills::SkillRegistry::default()),
            Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces: None,
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await;
        let llm: Arc<dyn LlmBackend> = Arc::new(UnusedBackend);

        let outcome = crate::trace_logging::with_trace_path(
            &trace,
            rerank_semantic_search(
                &llm,
                "test-model",
                &registry,
                &[],
                &json!({"queries": ["where is the retry budget applied"]}),
                IdleTimeouts {
                    first_progress: std::time::Duration::from_secs(1),
                    inter_chunk: std::time::Duration::from_secs(1),
                },
                &CancellationToken::new(),
            ),
        )
        .await;
        assert!(
            outcome.failed,
            "no bifrost server is registered, so the search must fail: {}",
            outcome.output
        );

        let lines = std::fs::read_to_string(&trace).unwrap_or_default();
        let timings: Vec<Value> = lines
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|record| record.get("type").and_then(Value::as_str) == Some("tool_timing"))
            .collect();
        assert_eq!(
            timings.len(),
            1,
            "expected exactly one tool_timing record, got {timings:?} from {lines:?}"
        );
        let timing = &timings[0];
        assert_eq!(
            timing.get("tool").and_then(Value::as_str),
            Some("semantic_search")
        );
        assert_eq!(timing.get("success").and_then(Value::as_bool), Some(false));
        assert!(
            timing.get("duration_ms").is_some_and(Value::is_number),
            "timing record needs the same duration_ms field every other tool writes: {timing:?}"
        );
        assert!(
            timing.get("timestamp").and_then(Value::as_str).is_some(),
            "timing record needs the timestamp the trace writer stamps: {timing:?}"
        );
    }

    /// The call the `semantic_search_batch` records cannot see. Rejected
    /// arguments never open a batch, so without the wrapper's own record this
    /// call would be invisible to a trace consumer counting calls per tool.
    /// It is also the cheapest path through the wrapper: no MCP server, no LLM.
    #[tokio::test]
    async fn a_rejected_semantic_search_still_records_its_tool_timing() {
        struct UnusedBackend;
        impl LlmBackend for UnusedBackend {
            fn list_models(&self) -> futures::future::BoxFuture<'_, anyhow::Result<Vec<String>>> {
                unimplemented!("rejected arguments never reach the rerank turn")
            }

            fn stream_chat(
                &self,
                _request: StreamChatRequest,
            ) -> futures::future::BoxFuture<'_, anyhow::Result<LlmResponse>> {
                unimplemented!("rejected arguments never reach the rerank turn")
            }
        }

        let cwd = tempfile::tempdir().expect("temp cwd");
        let trace = cwd.path().join("anvil-trace.jsonl");
        let registry = ToolRegistry::new(
            cwd.path().to_path_buf(),
            Vec::new(),
            Vec::new(),
            Arc::new(crate::skills::SkillRegistry::default()),
            Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces: None,
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await;
        let llm: Arc<dyn LlmBackend> = Arc::new(UnusedBackend);

        let outcome = crate::trace_logging::with_trace_path(
            &trace,
            rerank_semantic_search(
                &llm,
                "test-model",
                &registry,
                &[],
                &json!({"queries": []}),
                IdleTimeouts {
                    first_progress: std::time::Duration::from_secs(1),
                    inter_chunk: std::time::Duration::from_secs(1),
                },
                &CancellationToken::new(),
            ),
        )
        .await;
        assert!(
            outcome.failed,
            "an empty query list is rejected: {}",
            outcome.output
        );

        let lines = std::fs::read_to_string(&trace).unwrap_or_default();
        let records: Vec<Value> = lines
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect();
        let timings: Vec<&Value> = records
            .iter()
            .filter(|record| record.get("type").and_then(Value::as_str) == Some("tool_timing"))
            .collect();
        assert_eq!(
            timings.len(),
            1,
            "expected exactly one tool_timing record, got {records:?}"
        );
        assert_eq!(
            timings[0].get("tool").and_then(Value::as_str),
            Some("semantic_search")
        );
        assert_eq!(
            timings[0].get("success").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            records
                .iter()
                .all(|record| record.get("type").and_then(Value::as_str)
                    != Some("semantic_search_batch")),
            "a rejected call must not have opened a batch: {records:?}"
        );
    }

    fn selection(id: &str) -> Selection {
        Selection {
            id: id.to_string(),
            declarations: Vec::new(),
        }
    }

    fn tool_call(id: &str, name: &str) -> crate::llm_client::ToolCall {
        crate::llm_client::ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: crate::llm_client::FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    /// The disposable turn resends the task, not the retrieval. A realistic
    /// mixed history must come back as system/user/assistant text only, in
    /// order, with the tool results and every `tool_calls` field gone.
    #[test]
    fn the_rerank_prefix_keeps_only_system_user_and_assistant_text() {
        let history = vec![
            ChatMessage::system("You are Anvil."),
            ChatMessage::user("Where is the retry budget applied?"),
            ChatMessage::assistant_tool_calls_with_content_and_reasoning(
                "Let me search the retry path.",
                vec![tool_call("call-1", "grep_search")],
                Some("the budget is probably per tier".to_string()),
            ),
            ChatMessage::tool_result("call-1", "grep_search", "PRIOR_TOOL_OUTPUT_SENTINEL"),
            ChatMessage::assistant_with_reasoning(
                "Those hits look like the wrong layer.",
                Some("try the client instead".to_string()),
            ),
            // An assistant turn that was nothing but tool calls: dropping the
            // calls leaves nothing, so the message goes too.
            ChatMessage::assistant_tool_calls(vec![tool_call("call-2", "read_file")]),
            ChatMessage::tool_result("call-2", "read_file", "ANOTHER_TOOL_OUTPUT_SENTINEL"),
            ChatMessage::user("Also check the backoff."),
            // The in-flight semantic_search call: the cut point, as before.
            ChatMessage::assistant_tool_calls_with_content_and_reasoning(
                "Searching.",
                vec![tool_call("call-3", "semantic_search")],
                None,
            ),
        ];

        let prefix = prefix_for_rerank(&history);
        let roles: Vec<&str> = prefix.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
        assert!(
            prefix.iter().all(|m| m.tool_calls.is_none()),
            "no tool_calls may survive: {prefix:?}"
        );
        // The two assistant texts around the dropped tool result merge into one
        // message rather than becoming a same-role run.
        assert_eq!(
            prefix[2].content_text(),
            "Let me search the retry path.\n\nThose hits look like the wrong layer."
        );
        assert_eq!(
            prefix[2].reasoning_content.as_deref(),
            Some("the budget is probably per tier\n\ntry the client instead")
        );
        assert_eq!(prefix[3].content_text(), "Also check the backoff.");
        let rendered = prefix
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !rendered.contains("PRIOR_TOOL_OUTPUT_SENTINEL"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("ANOTHER_TOOL_OUTPUT_SENTINEL"),
            "{rendered}"
        );
        assert!(!rendered.contains("Searching."), "{rendered}");
    }

    /// With no in-flight tool call there is no cut point, and the same
    /// projection still applies to the whole history.
    #[test]
    fn a_history_without_tool_calls_keeps_its_whole_projection() {
        let history = vec![
            ChatMessage::system("You are Anvil."),
            ChatMessage::user("Find the parser."),
            ChatMessage::assistant("Here it is."),
        ];
        let prefix = prefix_for_rerank(&history);
        assert_eq!(prefix.len(), 3);
        assert_eq!(prefix[2].content_text(), "Here it is.");
    }

    /// What one rerank request looked like, kept because the shape of the
    /// request is the behavior under test: which tools it offered and what the
    /// model had been told by then.
    #[cfg(unix)]
    #[derive(Debug)]
    struct CapturedRequest {
        messages: Vec<ChatMessage>,
        tool_names: Vec<String>,
        structured: bool,
        idle_timeouts: IdleTimeouts,
    }

    /// A utility model scripted by request index. Hand-written rather than
    /// mocked: every test below drives the real rerank turn through it.
    #[cfg(unix)]
    struct ScriptedUtilityBackend {
        respond: Box<dyn Fn(usize) -> LlmResponse + Send + Sync>,
        requests: Arc<std::sync::Mutex<Vec<CapturedRequest>>>,
    }

    #[cfg(unix)]
    impl LlmBackend for ScriptedUtilityBackend {
        fn list_models(&self) -> futures::future::BoxFuture<'_, anyhow::Result<Vec<String>>> {
            use futures::FutureExt;
            async { Ok(vec!["test-model".to_string()]) }.boxed()
        }

        fn stream_chat(
            &self,
            request: StreamChatRequest,
        ) -> futures::future::BoxFuture<'_, anyhow::Result<LlmResponse>> {
            use futures::FutureExt;
            let index = {
                let mut requests = self.requests.lock().expect("captured requests");
                requests.push(CapturedRequest {
                    messages: request.messages.clone(),
                    tool_names: request
                        .tools
                        .iter()
                        .flatten()
                        .map(|tool| tool.function.name.clone())
                        .collect(),
                    structured: request.structured_output.is_some(),
                    idle_timeouts: request.idle_timeouts,
                });
                requests.len() - 1
            };
            let response = (self.respond)(index);
            async move { Ok(response) }.boxed()
        }
    }

    #[cfg(unix)]
    fn body_fetch_response(call_id: &str, ids: &[&str]) -> LlmResponse {
        LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: None,
            calls: vec![crate::llm_client::ToolCall {
                id: call_id.to_string(),
                r#type: "function".to_string(),
                function: crate::llm_client::FunctionCall {
                    name: BODY_FETCH_TOOL.to_string(),
                    arguments: json!({ "ids": ids }).to_string(),
                },
            }],
            usage: TokenUsage::default(),
        }
    }

    #[cfg(unix)]
    fn selection_response(selection: Value) -> LlmResponse {
        LlmResponse::Text {
            text: selection.to_string(),
            reasoning_content: None,
            usage: TokenUsage::default(),
        }
    }

    /// Drive one whole `semantic_search` call against the fake bifrost and a
    /// scripted utility model, returning the outcome, the requests the utility
    /// model saw, and the trace records the call wrote.
    #[cfg(unix)]
    async fn scripted_rerank(
        script_args: &[&str],
        respond: impl Fn(usize) -> LlmResponse + Send + Sync + 'static,
    ) -> (RerankOutcome, Vec<CapturedRequest>, Vec<Value>) {
        use crate::mcp::{McpFraming, McpServerConfig};

        let cwd = tempfile::tempdir().expect("temp cwd");
        let script = cwd.path().join("fake_bifrost_sources.py");
        std::fs::write(&script, FAKE_BIFROST_WITH_SOURCES).expect("write fake server");
        let trace = cwd.path().join("anvil-trace.jsonl");
        let mut args = vec![script.to_string_lossy().into_owned()];
        args.extend(script_args.iter().map(|arg| (*arg).to_string()));

        let registry = ToolRegistry::new(
            cwd.path().to_path_buf(),
            Vec::new(),
            vec![McpServerConfig {
                name: "bifrost".to_string(),
                transport: Default::default(),
                command: std::env::var("ANVIL_PYTHON").unwrap_or_else(|_| "python3".to_string()),
                url: None,
                headers: Vec::new(),
                args,
                env: Vec::new(),
                framing: McpFraming::Line,
                enabled: true,
            }],
            Arc::new(crate::skills::SkillRegistry::default()),
            Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces: None,
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await;
        assert!(
            registry.is_bifrost_tool("semantic_search"),
            "the fake bifrost must advertise semantic_search"
        );

        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let llm: Arc<dyn LlmBackend> = Arc::new(ScriptedUtilityBackend {
            respond: Box::new(respond),
            requests: requests.clone(),
        });
        let outcome = crate::trace_logging::with_trace_path(
            &trace,
            rerank_semantic_search(
                &llm,
                "test-model",
                &registry,
                &[ChatMessage::user("where does run() come from?")],
                &json!({ "queries": ["where does run come from"] }),
                // The session's shape, so the utility override is visible.
                IdleTimeouts {
                    first_progress: Duration::from_secs(120),
                    inter_chunk: Duration::from_secs(60),
                },
                &CancellationToken::new(),
            ),
        )
        .await;

        let records = std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect();
        let requests = std::mem::take(&mut *requests.lock().expect("captured requests"));
        (outcome, requests, records)
    }

    #[cfg(unix)]
    fn rerank_record(records: &[Value]) -> &Value {
        records
            .iter()
            .find(|record| {
                record.get("type").and_then(Value::as_str) == Some("semantic_search_rerank")
            })
            .unwrap_or_else(|| panic!("expected a semantic_search_rerank record in {records:?}"))
    }

    /// The ordinary case: signatures decide it, no body is fetched, and the
    /// prompt never carried a source body to begin with.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_rerank_that_needs_no_source_selects_in_one_request() {
        let (outcome, requests, records) = scripted_rerank(&[], |_| {
            selection_response(json!({ "relevant": [{ "id": "s1", "declarations": ["s1d1"] }] }))
        })
        .await;

        assert!(!outcome.failed, "{}", outcome.output);
        assert!(
            outcome.output.contains("app.alpha.run"),
            "{}",
            outcome.output
        );
        assert_eq!(requests.len(), 1);
        let prompt = requests[0].messages.last().expect("prompt").content_text();
        assert!(prompt.contains("[s1] symbol app.alpha.run"), "{prompt}");
        assert!(prompt.contains("def run(self):"), "{prompt}");
        // Signatures only: the body bifrost holds for this symbol is not in the
        // prompt, and the tool is advertised instead.
        assert!(!prompt.contains("return self.value"), "{prompt}");
        assert!(prompt.contains(BODY_FETCH_TOOL), "{prompt}");
        assert_eq!(requests[0].tool_names, vec![BODY_FETCH_TOOL.to_string()]);
        assert!(requests[0].structured);
        // A stalled utility request is abandoned well before the session's own
        // first-progress budget; a streaming one keeps it.
        assert_eq!(
            requests[0].idle_timeouts,
            IdleTimeouts {
                first_progress: UTILITY_FIRST_PROGRESS_TIMEOUT,
                inter_chunk: Duration::from_secs(60),
            }
        );

        let record = rerank_record(&records);
        assert_eq!(record["body_fetch_rounds"], 0);
        assert_eq!(record["bodies_fetched"], 0);
        assert!(
            !records
                .iter()
                .any(|record| record["phase"] == "utility_tool_round"),
            "no round was needed: {records:?}"
        );
    }

    /// The exception path: the model asks for two bodies that turn out to be
    /// identical and one id that does not exist, then selects. The duplicate is
    /// stated by reference, the unknown id is rejected in the result text
    /// rather than failing the search, and the selection still round trips.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_body_fetch_round_dedupes_bodies_and_rejects_unknown_ids() {
        let (outcome, requests, records) = scripted_rerank(&[], |index| match index {
            0 => body_fetch_response("call-1", &["s1", "s2", "s9"]),
            _ => selection_response(
                json!({ "relevant": [{ "id": "s2", "declarations": ["s2d1"] }] }),
            ),
        })
        .await;

        assert!(!outcome.failed, "{}", outcome.output);
        assert!(
            outcome.output.contains("app.beta.run"),
            "{}",
            outcome.output
        );
        assert_eq!(requests.len(), 2);

        let tool_results: Vec<String> = requests[1]
            .messages
            .iter()
            .filter(|message| message.role == "tool")
            .map(ChatMessage::content_text)
            .collect();
        assert_eq!(tool_results.len(), 1, "{:?}", requests[1].messages);
        let result = &tool_results[0];
        assert!(result.contains("[s1] app.alpha.run"), "{result}");
        assert!(result.contains("return self.value"), "{result}");
        assert!(result.contains("[s2] app.beta.run"), "{result}");
        assert!(result.contains("identical source to s1"), "{result}");
        assert_eq!(result.matches("return self.value").count(), 1, "{result}");
        assert!(result.contains("s9: no candidate has that id."), "{result}");
        // The assistant turn that made the call is replayed with it, so the
        // tool result has something to attach to.
        assert!(
            requests[1]
                .messages
                .iter()
                .any(|message| message.role == "assistant" && message.tool_calls.is_some()),
            "{:?}",
            requests[1].messages
        );

        let round = records
            .iter()
            .find(|record| record["phase"] == "utility_tool_round")
            .unwrap_or_else(|| panic!("expected a utility_tool_round record in {records:?}"));
        assert_eq!(round["round"], 1);
        assert_eq!(round["id_count"], 3);
        let record = rerank_record(&records);
        assert_eq!(record["body_fetch_rounds"], 1);
        assert_eq!(record["bodies_fetched"], 2);
    }

    /// Every `bifrost_analyzer_saturation` record the call wrote.
    #[cfg(unix)]
    fn saturation_records(records: &[Value]) -> Vec<&Value> {
        records
            .iter()
            .filter(|record| record["type"] == "bifrost_analyzer_saturation")
            .collect()
    }

    /// The defect this fixes. Bifrost refuses an enrichment call because its
    /// analyzer pool is full -- a signal that names retrying as the remedy -- and
    /// the whole rerank used to fail on it. Now the call waits, repeats itself,
    /// and the search comes back normal, with the retry on the record rather
    /// than hidden.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_saturated_summary_call_retries_and_reranks_normally() {
        let (outcome, requests, records) = scripted_rerank(&["saturate_summaries=1"], |_| {
            selection_response(json!({ "relevant": [{ "id": "s1", "declarations": ["s1d1"] }] }))
        })
        .await;

        assert!(!outcome.failed, "{}", outcome.output);
        assert!(
            outcome.output.contains("app.alpha.run"),
            "{}",
            outcome.output
        );
        assert_eq!(requests.len(), 1);
        // The retry restored the enrichment rather than reranking without it:
        // every candidate reaches the prompt with its signatures.
        let prompt = requests[0].messages.last().expect("prompt").content_text();
        assert!(
            !prompt.contains("(no structured signatures available)"),
            "{prompt}"
        );

        let saturation = saturation_records(&records);
        assert_eq!(saturation.len(), 2, "{saturation:?}");
        assert_eq!(saturation[0]["phase"], "retry");
        assert_eq!(saturation[0]["tool"], "get_summaries");
        assert_eq!(saturation[0]["retry"], 1);
        assert_eq!(
            saturation[0]["delay_millis"],
            json!(SATURATION_RETRY_DELAYS[0].as_millis())
        );
        assert!(
            saturation[0]["error"]
                .as_str()
                .expect("the refusal text is recorded")
                .contains(ANALYZER_SATURATION_MARKER),
            "{saturation:?}"
        );
        assert_eq!(saturation[1]["phase"], "recovered");
        assert_eq!(saturation[1]["retries"], 1);
    }

    /// Saturation that outlives the budget is a failure again: a rerank that
    /// quietly proceeded on partial enrichment would be a different retrieval
    /// treatment. This is the one test that pays the whole retry budget, which
    /// is the point of having one -- a permanently saturated server bounds the
    /// query instead of hanging it.
    #[cfg(unix)]
    #[tokio::test]
    async fn saturation_past_the_retry_budget_fails_the_rerank_closed() {
        let (outcome, requests, records) = scripted_rerank(&["saturate_summaries=99"], |_| {
            selection_response(json!({ "relevant": [] }))
        })
        .await;

        assert!(outcome.failed, "{}", outcome.output);
        assert!(
            outcome.output.contains("context fetch failed"),
            "{}",
            outcome.output
        );
        assert!(
            outcome.output.contains(ANALYZER_SATURATION_MARKER),
            "{}",
            outcome.output
        );
        assert!(
            requests.is_empty(),
            "the utility model is never asked to rerank an unenriched pool: {requests:?}"
        );

        // The three candidates of one batch are enriched concurrently, so each
        // of them spends the whole budget before the batch reports the failure.
        let saturation = saturation_records(&records);
        let retries = saturation
            .iter()
            .filter(|record| record["phase"] == "retry")
            .count();
        let exhausted = saturation
            .iter()
            .filter(|record| record["phase"] == "exhausted")
            .count();
        assert_eq!(exhausted, 3, "{saturation:?}");
        assert_eq!(retries, 3 * SATURATION_RETRY_DELAYS.len(), "{saturation:?}");
    }

    /// A model that keeps calling the tool runs out of rounds. The final
    /// request must still go out -- with the schema, without the tool, and with
    /// the instruction that tool access is over.
    #[cfg(unix)]
    #[tokio::test]
    async fn exhausted_rounds_still_issue_the_final_structured_request() {
        let (outcome, requests, records) = scripted_rerank(&[], |index| match index {
            0 => body_fetch_response("call-1", &["s1"]),
            1 => body_fetch_response("call-2", &["s3"]),
            _ => selection_response(json!({ "relevant": [] })),
        })
        .await;

        assert!(!outcome.failed, "{}", outcome.output);
        assert_eq!(requests.len(), MAX_BODY_FETCH_ROUNDS + 1);
        let final_request = requests.last().expect("final request");
        assert!(
            final_request.tool_names.is_empty(),
            "{:?}",
            final_request.tool_names
        );
        assert!(final_request.structured);
        assert!(
            final_request
                .messages
                .last()
                .expect("instruction")
                .content_text()
                .contains("Tool access is finished"),
            "{:?}",
            final_request.messages.last()
        );

        let record = rerank_record(&records);
        assert_eq!(record["body_fetch_rounds"], MAX_BODY_FETCH_ROUNDS);
        assert_eq!(record["bodies_fetched"], 2);
    }

    /// A bifrost failure inside a round fails the search, exactly as one during
    /// retrieval or the summary sweep does. Silently reranking without the
    /// source the model asked for would be a different retrieval treatment.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_bifrost_failure_during_a_round_fails_closed() {
        let (outcome, requests, records) =
            scripted_rerank(&["fail_sources"], |index| match index {
                0 => body_fetch_response("call-1", &["s1"]),
                _ => selection_response(json!({ "relevant": [] })),
            })
            .await;

        assert!(outcome.failed, "{}", outcome.output);
        assert!(
            outcome.output.contains("source fetch failed"),
            "{}",
            outcome.output
        );
        assert!(
            outcome.output.contains("symbol source index unavailable"),
            "{}",
            outcome.output
        );
        assert_eq!(requests.len(), 1, "the turn stops at the failed round");
        assert!(
            records
                .iter()
                .any(|record| record["phase"] == "utility_tool_error"),
            "{records:?}"
        );
    }
}
