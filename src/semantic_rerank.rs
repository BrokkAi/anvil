//! LLM relevance reranker wrapped transparently around bifrost's
//! `semantic_search` MCP tool.
//!
//! `semantic_search` returns three independent, un-fused ranked lists
//! (`vector_ranked` and `bm25_ranked` over symbols, `coedit_ranked` over
//! files) and explicitly leaves fusion to the caller. Rather than dump all
//! three raw lists into the model's context, the harness runs one *disposable*
//! LLM turn on top of the live conversation: it shows the active model each
//! candidate together with its (truncated) source or file summary and asks it
//! to return just the relevant ones, best-first. The model then sees a single
//! clean, relevance-ordered hit list; the bulky candidate context lives only
//! in the disposable turn and never pollutes the main conversation.
//!
//! The disposable turn reuses the conversation history as its prefix (minus the
//! trailing assistant message that carries the in-flight `tool_calls`), so the
//! long history stays a provider-cache hit and the only genuinely new tokens
//! are the candidate sources themselves.
//!
//! Provider and structured-output failures degrade to deterministic reciprocal
//! rank fusion. Bifrost's raw three-list payload is never exposed to the model.

use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::llm_client::{
    ChatMessage, IdleTimeouts, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage,
    stream_chat_no_visible_output_with_retry,
};
use crate::structured_output::StructuredOutputRequest;
use crate::tools::ToolRegistry;
use crate::trace_logging::append_trace_record;

const DEFAULT_FINAL_K: usize = 20;
const MAX_FINAL_K: usize = 20;
const OVERFETCH_MULTIPLIER: usize = 2;
const MAX_CANDIDATE_CONTEXT_BYTES: usize = 8_000;
const MAX_TOTAL_CONTEXT_BYTES: usize = 120_000;
const CONTEXT_FETCH_BATCH: usize = 8;
const RRF_RANK_CONSTANT: f64 = 60.0;

/// Reasoning effort for the disposable turn. "low" (not "minimal", which some
/// backends don't support and which is too aggressive where they do) keeps the
/// relevance call cheap and fast.
const RERANK_REASONING_EFFORT: &str = "low";

const MAX_LINE_CHARS: usize = 2048;

/// What the reranker hands back to the tool loop, mirroring the fields the loop
/// needs from a `ToolExecution`.
pub(crate) struct RerankOutcome {
    pub output: String,
    pub failed: bool,
    pub usage: TokenUsage,
}

impl RerankOutcome {
    fn passthrough(output: String, usage: TokenUsage) -> Self {
        Self {
            output,
            failed: false,
            usage,
        }
    }

    fn error(message: String) -> Self {
        Self {
            output: message,
            failed: true,
            usage: TokenUsage::default(),
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
    /// Which retrieval legs surfaced it (`vector`/`bm25` for symbols,
    /// `coedit` for files).
    signals: Vec<&'static str>,
    /// `path:start-end` for symbols (from `get_symbol_sources`), if known.
    location: Option<String>,
    /// Truncated source (symbols) or rendered summary (files), if fetched.
    context: Option<String>,
    /// Reciprocal-rank score accumulated across the active retrieval legs.
    rrf_score: f64,
    /// First raw-list position, used to make equal RRF scores deterministic.
    first_seen: usize,
}

/// Run `semantic_search`, then rerank its candidates with a disposable LLM turn
/// and render a unified, relevance-ordered hit list. Falls back to bifrost's raw
/// payload on any error or empty selection.
pub(crate) async fn rerank_semantic_search(
    llm: &Arc<dyn LlmBackend>,
    model: &str,
    registry: &ToolRegistry,
    prior_messages: &[ChatMessage],
    args: &Value,
    idle_timeout: IdleTimeouts,
    cancel: &CancellationToken,
) -> RerankOutcome {
    let final_k = match parse_final_k(args) {
        Ok(k) => k,
        Err(message) => return RerankOutcome::error(message),
    };
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let (bifrost_args, base_k) = prepare_bifrost_args(args, final_k);

    // 1. Underlying search. A hard failure here is the model's to see.
    let raw = match registry
        .call_bifrost_tool_raw("semantic_search", bifrost_args)
        .await
    {
        Ok(value) => value,
        Err(err) => return RerankOutcome::error(format!("Error: {err}")),
    };

    // 2. Parse the entire realized pool. An empty search is a valid final result.
    let mut candidates = parse_candidates(&raw);
    if candidates.is_empty() {
        trace_rerank(RerankTrace {
            raw: &raw,
            final_k,
            base_k,
            deduplicated_count: 0,
            context_bytes: 0,
            selected_count: 0,
            final_count: 0,
            fallback_reason: None,
            usage: TokenUsage::default(),
        });
        return RerankOutcome::passthrough(
            render_unified(&query, &[], 0, notes(&raw)),
            TokenUsage::default(),
        );
    }

    // 3. Fetch source / summaries for the candidates (best effort).
    fetch_context(registry, &mut candidates).await;
    let context_bytes = bound_candidate_context(&mut candidates);

    // 4. Disposable relevance turn on top of the live conversation.
    let mut messages = prefix_for_rerank(prior_messages);
    messages.push(ChatMessage::user(build_rerank_prompt(
        &query,
        &candidates,
        final_k,
    )));
    let structured = StructuredOutputRequest {
        schema_name: "semantic_rerank".to_string(),
        schema: rerank_schema(),
        allow_coercion: true,
        prefer_json_object: false,
    };

    let response = stream_chat_no_visible_output_with_retry(
        llm.as_ref(),
        "semantic_search rerank",
        cancel,
        || StreamChatRequest {
            model: model.to_string(),
            messages: messages.clone(),
            tools: None,
            reasoning_effort: Some(RERANK_REASONING_EFFORT.to_string()),
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
            tracing::warn!(
                error = format!("{err:#}"),
                "semantic_search rerank turn failed; using reciprocal-rank fusion"
            );
            let ordered = rrf_fallback(&candidates, final_k);
            trace_rerank(RerankTrace {
                raw: &raw,
                final_k,
                base_k,
                deduplicated_count: candidates.len(),
                context_bytes,
                selected_count: 0,
                final_count: ordered.len(),
                fallback_reason: Some("provider_failure"),
                usage: TokenUsage::default(),
            });
            return RerankOutcome::passthrough(
                render_unified(&query, &ordered, candidates.len(), notes(&raw)),
                TokenUsage::default(),
            );
        }
    };
    let usage = response.usage();
    let text = match response {
        LlmResponse::Text { text, .. } | LlmResponse::ToolCalls { text, .. } => text,
    };

    // 5. A well-formed empty or all-unknown selection is valid and final. Only
    // malformed structured output invokes fallback.
    let selected = match parse_selected_ids(&text) {
        Some(selected) => selected,
        None => {
            tracing::warn!(
                "semantic_search rerank returned malformed output; using reciprocal-rank fusion"
            );
            let ordered = rrf_fallback(&candidates, final_k);
            trace_rerank(RerankTrace {
                raw: &raw,
                final_k,
                base_k,
                deduplicated_count: candidates.len(),
                context_bytes,
                selected_count: 0,
                final_count: ordered.len(),
                fallback_reason: Some("malformed_output"),
                usage,
            });
            return RerankOutcome::passthrough(
                render_unified(&query, &ordered, candidates.len(), notes(&raw)),
                usage,
            );
        }
    };
    let ordered = order_candidates(&candidates, &selected, final_k);

    tracing::debug!(
        query = %query,
        candidates = candidates.len(),
        selected = ordered.len(),
        cached_read_tokens = usage.cached_read_tokens,
        input_tokens = usage.input_tokens,
        "semantic_search reranked"
    );
    trace_rerank(RerankTrace {
        raw: &raw,
        final_k,
        base_k,
        deduplicated_count: candidates.len(),
        context_bytes,
        selected_count: selected.len(),
        final_count: ordered.len(),
        fallback_reason: None,
        usage,
    });
    let output = render_unified(&query, &ordered, candidates.len(), notes(&raw));
    RerankOutcome::passthrough(output, usage)
}

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

fn prepare_bifrost_args(args: &Value, final_k: usize) -> (Value, usize) {
    let base_k = final_k * OVERFETCH_MULTIPLIER;
    let mut bifrost_args = args.clone();
    let object = bifrost_args
        .as_object_mut()
        .expect("parse_final_k established that semantic_search arguments are an object");
    object.insert("k".to_string(), json!(base_k));
    (bifrost_args, base_k)
}

/// Parse every realized item from `semantic_search`'s three legs into one
/// identity-deduplicated candidate list. The order is deterministic RRF order,
/// which is also the provider-failure fallback order.
fn parse_candidates(raw: &Value) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut index: HashMap<(CandidateKind, String), usize> = HashMap::new();
    let mut first_seen = 0;
    for (key, signal) in [("vector_ranked", "vector"), ("bm25_ranked", "bm25")] {
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

/// Batch-fetch symbol source (`get_symbol_sources`) and file summaries
/// (`get_summaries`) for the candidates and attach truncated context. Partial
/// or total failure is non-fatal: a candidate without context still appears by
/// name in the rerank prompt.
async fn fetch_context(registry: &ToolRegistry, candidates: &mut [Candidate]) {
    for start in (0..candidates.len()).step_by(CONTEXT_FETCH_BATCH) {
        let end = (start + CONTEXT_FETCH_BATCH).min(candidates.len());
        let symbols: Vec<String> = candidates[start..end]
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::Symbol)
            .map(|candidate| candidate.name.clone())
            .collect();
        let files: Vec<String> = candidates[start..end]
            .iter()
            .filter(|candidate| candidate.kind == CandidateKind::File)
            .map(|candidate| candidate.name.clone())
            .collect();

        fetch_symbol_context(registry, candidates, &symbols).await;
        fetch_file_context(registry, candidates, &files).await;
        if available_context_bytes(candidates) >= MAX_TOTAL_CONTEXT_BYTES {
            break;
        }
    }
}

async fn fetch_symbol_context(
    registry: &ToolRegistry,
    candidates: &mut [Candidate],
    symbols: &[String],
) {
    if symbols.is_empty() {
        return;
    }
    match registry
        .call_bifrost_tool_raw("get_symbol_sources", json!({ "symbols": symbols }))
        .await
    {
        Ok(value) => attach_symbol_sources(candidates, &value),
        Err(_) if symbols.len() > 1 => {
            for symbol in symbols {
                if let Ok(value) = registry
                    .call_bifrost_tool_raw("get_symbol_sources", json!({ "symbols": [symbol] }))
                    .await
                {
                    attach_symbol_sources(candidates, &value);
                }
            }
        }
        Err(_) => {}
    }
}

async fn fetch_file_context(
    registry: &ToolRegistry,
    candidates: &mut [Candidate],
    files: &[String],
) {
    if files.is_empty() {
        return;
    }
    match registry
        .call_bifrost_tool_raw("get_summaries", json!({ "targets": files }))
        .await
    {
        Ok(value) => attach_summaries(candidates, &value),
        Err(_) if files.len() > 1 => {
            for file in files {
                if let Ok(value) = registry
                    .call_bifrost_tool_raw("get_summaries", json!({ "targets": [file] }))
                    .await
                {
                    attach_summaries(candidates, &value);
                }
            }
        }
        Err(_) => {}
    }
}

fn available_context_bytes(candidates: &[Candidate]) -> usize {
    candidates
        .iter()
        .filter_map(|candidate| candidate.context.as_ref())
        .map(|context| context.len().min(MAX_CANDIDATE_CONTEXT_BYTES))
        .sum()
}

/// Enforce the disposable-turn source/summary budget without ever removing a
/// candidate identity. Earlier RRF candidates consume the shared budget first.
fn bound_candidate_context(candidates: &mut [Candidate]) -> usize {
    let mut remaining = MAX_TOTAL_CONTEXT_BYTES;
    let mut used = 0;
    for candidate in candidates {
        let Some(context) = candidate.context.take() else {
            continue;
        };
        let limit = remaining.min(MAX_CANDIDATE_CONTEXT_BYTES);
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

/// Attach truncated source text to symbol candidates from a
/// `get_symbol_sources` result (`{ sources: [{label,path,start_line,end_line,text}] }`).
fn attach_symbol_sources(candidates: &mut [Candidate], result: &Value) {
    let Some(sources) = result.get("sources").and_then(Value::as_array) else {
        return;
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
            candidate.context = Some(sample_source(text));
        }
        let path = block.get("path").and_then(Value::as_str);
        let start = block.get("start_line").and_then(Value::as_u64);
        let end = block.get("end_line").and_then(Value::as_u64);
        candidate.location = match (path, start, end) {
            (Some(path), Some(start), Some(end)) => Some(format!("{path}:{start}-{end}")),
            (Some(path), _, _) => Some(path.to_string()),
            _ => None,
        };
    }
}

/// Attach rendered, truncated summaries to file candidates from a
/// `get_summaries` result (`{ summaries: [{label,path,preamble,elements:[{text}]}] }`).
fn attach_summaries(candidates: &mut [Candidate], result: &Value) {
    let mut by_path: HashMap<&str, &Value> = HashMap::new();
    if let Some(summaries) = result.get("summaries").and_then(Value::as_array) {
        for block in summaries {
            let key = block
                .get("path")
                .and_then(Value::as_str)
                .or_else(|| block.get("label").and_then(Value::as_str));
            if let Some(key) = key {
                by_path.entry(key).or_insert(block);
            }
        }
    }
    for candidate in candidates
        .iter_mut()
        .filter(|c| c.kind == CandidateKind::File)
    {
        if let Some(block) = by_path.get(candidate.name.as_str()) {
            candidate.context = Some(sample_summary(&render_summary_block(block)));
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

/// Build the conversation prefix for the disposable turn: the live history with
/// the trailing assistant `tool_calls` message (and any sibling tool results
/// already appended this step) dropped, so the turn ends cleanly and the prior
/// history stays an identical, cache-hit prefix.
fn prefix_for_rerank(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let cut = messages
        .iter()
        .rposition(|m| m.role == "assistant" && m.tool_calls.is_some())
        .unwrap_or(messages.len());
    messages[..cut].to_vec()
}

fn build_rerank_prompt(query: &str, candidates: &[Candidate], final_k: usize) -> String {
    let mut out = String::new();
    out.push_str(
        "A code search just ran for the query below and returned these candidate results. \
Each candidate has an id, the symbol or file it refers to, and (when available) its source \
or file summary. Decide which candidates are genuinely relevant to the query and the task \
in this conversation, then return their ids ordered most-relevant first. Omit irrelevant ones.\n\n",
    );
    out.push_str(&format!(
        "Respond with ONLY a JSON object of the form {{\"relevant\": [\"<id>\", ...]}} using \
the exact ids shown. Select at most {final_k}. If nothing is relevant, return \
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
        out.push_str(location);
        out.push('\n');
    }
    match &candidate.context {
        Some(context) => {
            out.push_str("```\n");
            out.push_str(context);
            out.push_str("\n```\n");
        }
        None => out.push_str("(no source available)\n"),
    }
    out
}

/// Resolve the model's selected ids back to candidates, preserving the model's
/// order and dropping unknown/duplicate ids.
fn order_candidates<'a>(
    candidates: &'a [Candidate],
    selected: &[String],
    final_k: usize,
) -> Vec<&'a Candidate> {
    let by_id: HashMap<&str, &Candidate> = candidates.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for id in selected {
        if !seen.insert(id.as_str()) {
            continue;
        }
        if let Some(candidate) = by_id.get(id.as_str()) {
            ordered.push(*candidate);
            if ordered.len() == final_k {
                break;
            }
        }
    }
    ordered
}

fn rrf_fallback(candidates: &[Candidate], final_k: usize) -> Vec<&Candidate> {
    candidates.iter().take(final_k).collect()
}

struct RerankTrace<'a> {
    raw: &'a Value,
    final_k: usize,
    base_k: usize,
    deduplicated_count: usize,
    context_bytes: usize,
    selected_count: usize,
    final_count: usize,
    fallback_reason: Option<&'a str>,
    usage: TokenUsage,
}

fn trace_rerank(trace: RerankTrace<'_>) {
    let realized = |key: &str| {
        trace
            .raw
            .get(key)
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
    };
    append_trace_record(json!({
        "type": "semantic_search_rerank",
        "requested_final_k": trace.final_k,
        "forwarded_base_k": trace.base_k,
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
        "fallback": trace.fallback_reason.is_some(),
        "fallback_reason": trace.fallback_reason,
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
    ordered: &[&Candidate],
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
    for (rank, candidate) in ordered.iter().enumerate() {
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
        if let Some(location) = &candidate.location {
            out.push_str(&format!("   {location}\n"));
        }
        if let Some(context) = &candidate.context {
            out.push_str("```\n");
            out.push_str(context);
            out.push_str("\n```\n");
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

fn rerank_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "relevant": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["relevant"],
        "additionalProperties": false
    })
}

/// Parse `{"relevant": ["id", ...]}` from a model response. Native
/// structured-output backends return clean JSON by construction; the prompt
/// instructs the rest to do the same. Anything that doesn't parse yields `None`,
/// which invokes deterministic RRF fallback -- no bespoke extraction needed.
fn parse_selected_ids(text: &str) -> Option<Vec<String>> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    let object = value.as_object()?;
    if object.len() != 1 {
        return None;
    }
    let array = object.get("relevant")?.as_array()?;
    array
        .iter()
        .map(|value| value.as_str().map(str::to_string))
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(parse_final_k(&json!({ "query": "x" })), Ok(20));
        assert_eq!(parse_final_k(&json!({ "query": "x", "k": 1 })), Ok(1));
        assert_eq!(parse_final_k(&json!({ "query": "x", "k": 20 })), Ok(20));
        for invalid in [json!(0), json!(21), json!(1.5), json!("20"), Value::Null] {
            assert!(parse_final_k(&json!({ "k": invalid })).is_err());
        }
    }

    #[test]
    fn bifrost_receives_twice_the_final_k() {
        for final_k in [1, 7, 20] {
            let args = json!({ "query": "needle", "k": final_k });
            let (forwarded, base_k) = prepare_bifrost_args(&args, final_k);
            assert_eq!(base_k, 2 * final_k);
            assert_eq!(forwarded["k"], 2 * final_k);
            assert_eq!(forwarded["query"], "needle");
            assert_eq!(args["k"], final_k);
        }
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
    }

    #[test]
    fn parses_relevant_object_and_rejects_non_json() {
        assert_eq!(
            parse_selected_ids(r#"{"relevant":["s1","f2"]}"#),
            Some(vec!["s1".to_string(), "f2".to_string()])
        );
        assert_eq!(
            parse_selected_ids("  {\"relevant\": []}  "),
            Some(Vec::new())
        );
        // Non-JSON / fenced output does not parse; the caller falls back to
        // raw passthrough.
        assert_eq!(
            parse_selected_ids("```json\n{\"relevant\": [\"s3\"]}\n```"),
            None
        );
        assert_eq!(parse_selected_ids("no json here"), None);
        assert_eq!(parse_selected_ids(r#"{"relevant":["s1",3]}"#), None);
        assert_eq!(
            parse_selected_ids(r#"{"relevant":[],"unexpected":true}"#),
            None
        );
    }

    #[test]
    fn order_candidates_preserves_model_order_and_drops_unknown() {
        let candidates = parse_candidates(&json!({
            "vector_ranked": [
                { "fqfn": "a.B.c", "score": 0.9 },
                { "fqfn": "a.B.d", "score": 0.5 }
            ]
        }));
        let selected = vec![
            "s2".to_string(),
            "nope".to_string(),
            "s1".to_string(),
            "s2".to_string(),
        ];
        let ordered = order_candidates(&candidates, &selected, 20);
        let names: Vec<&str> = ordered.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["a.B.d", "a.B.c"]);
    }

    #[test]
    fn valid_empty_fewer_and_overlong_selections_are_final() {
        let candidates = parse_candidates(&json!({
            "vector_ranked": (0..6)
                .map(|i| json!({ "fqfn": format!("item::{i}") }))
                .collect::<Vec<_>>()
        }));
        assert!(order_candidates(&candidates, &[], 3).is_empty());

        let fewer = order_candidates(&candidates, &["s2".to_string()], 3);
        assert_eq!(fewer.len(), 1);

        let selected: Vec<String> = (1..=6).map(|i| format!("s{i}")).collect();
        let capped = order_candidates(&candidates, &selected, 3);
        assert_eq!(capped.len(), 3);
    }

    #[test]
    fn reciprocal_rank_fallback_is_deterministic_and_bounded() {
        let candidates = parse_candidates(&json!({
            "vector_ranked": [
                { "fqfn": "vector_only" },
                { "fqfn": "both" }
            ],
            "bm25_ranked": [
                { "fqfn": "bm25_only" },
                { "fqfn": "both" }
            ],
            "coedit_ranked": [
                { "path": "src/file.rs" }
            ]
        }));
        let first: Vec<&str> = rrf_fallback(&candidates, 2)
            .iter()
            .map(|candidate| candidate.name.as_str())
            .collect();
        let second: Vec<&str> = rrf_fallback(&candidates, 2)
            .iter()
            .map(|candidate| candidate.name.as_str())
            .collect();
        assert_eq!(first, vec!["both", "vector_only"]);
        assert_eq!(second, first);
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
        assert_eq!(available_context_bytes(&candidates), context.len());
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
            schema: rerank_schema(),
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
                reasoning_effort: Some(RERANK_REASONING_EFFORT.to_string()),
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

        let selected = parse_selected_ids(&text).expect("response parses as {\"relevant\": [...]}");
        let valid: std::collections::HashSet<&str> =
            candidates.iter().map(|c| c.id.as_str()).collect();
        assert!(!selected.is_empty(), "expected at least one relevant id");
        assert!(
            selected.iter().all(|id| valid.contains(id.as_str())),
            "all returned ids must be known candidate ids, got {selected:?}"
        );
        assert!(
            selected.iter().any(|id| id == "s1" || id == "f1"),
            "expected a config-parsing candidate (s1/f1) to be judged relevant, got {selected:?}"
        );
        // The trivially-irrelevant UI button should not be selected.
        assert!(
            !selected.iter().any(|id| id == "f2"),
            "did not expect the UI button file to be relevant, got {selected:?}"
        );
    }
}
