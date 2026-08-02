//! LLM relevance reranker wrapped transparently around bifrost's
//! `semantic_search` MCP tool.
//!
//! `semantic_search` returns three independent, un-fused ranked lists
//! (`vector_ranked` and `bm25_ranked` over symbols, `coedit_ranked` over
//! files) and explicitly leaves fusion to the caller. Rather than dump all
//! three raw lists into the model's context, the harness runs one *disposable*
//! LLM turn on top of the live conversation: it shows the selected utility model each
//! candidate together with its (truncated) source or file summary and asks it
//! to return just the relevant ones, best-first. The model then sees a single
//! clean, relevance-ordered hit list; the bulky candidate context lives only
//! in the disposable turn and never pollutes the main conversation.
//!
//! The disposable turn reuses the conversation history as its prefix (minus the
//! trailing assistant message that carries the in-flight `tool_calls`). When
//! utility routing resolves to the session provider, that prefix can remain a
//! provider-cache hit; an explicitly separate utility provider receives the
//! same relevance context without changing the main conversation.
//!
//! In ordinary operation, provider and structured-output failures degrade to
//! deterministic reciprocal-rank fusion. CIM evaluation mode fails closed so a
//! provider failure cannot silently change the experimental treatment.
//! Bifrost's raw three-list payload is never exposed to the model.

use futures::future::join_all;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
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
const MAX_SELECTED_DECLARATIONS: usize = 5;
const RRF_RANK_CONSTANT: f64 = 60.0;

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
    /// Which retrieval legs surfaced it (`vector`/`bm25` for symbols,
    /// `coedit` for files).
    signals: Vec<&'static str>,
    /// `path:start-end` for symbols (from `get_symbol_sources`), if known.
    location: Option<String>,
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
}

/// Run `semantic_search`, then rerank its candidates with a disposable LLM turn
/// and render a unified, relevance-ordered hit list. Ordinary operation falls
/// back to deterministic RRF on reranker failure; CIM evaluation fails closed.
pub(crate) async fn rerank_semantic_search(
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
    idle_timeout: IdleTimeouts,
    cancel: &CancellationToken,
) -> RerankOutcome {
    let utility = crate::utility_model::select(model);
    let started = Instant::now();
    let (bifrost_args, base_k) = prepare_bifrost_args(query, final_k);

    // 1. Underlying search. A hard failure here is the model's to see.
    trace_phase(
        "retrieval_start",
        query,
        query_index,
        query_count,
        &utility,
        started,
        None,
    );
    let raw = match registry
        .call_bifrost_tool_raw("semantic_search", bifrost_args)
        .await
    {
        Ok(value) => {
            trace_phase(
                "retrieval_complete",
                query,
                query_index,
                query_count,
                &utility,
                started,
                None,
            );
            value
        }
        Err(err) => {
            trace_phase(
                "retrieval_error",
                query,
                query_index,
                query_count,
                &utility,
                started,
                Some(&format!("{err:#}")),
            );
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
            fallback_reason: None,
            usage: TokenUsage::default(),
            utility: &utility,
        });
        return RerankOutcome::passthrough(
            render_unified(query, &[], 0, notes(&raw), false),
            TokenUsage::default(),
        );
    }

    // 3. Fetch source / summaries for the candidates (best effort).
    trace_phase(
        "context_fetch_start",
        query,
        query_index,
        query_count,
        &utility,
        started,
        None,
    );
    fetch_context(registry, &mut candidates).await;
    trace_phase(
        "context_fetch_complete",
        query,
        query_index,
        query_count,
        &utility,
        started,
        None,
    );
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

    trace_phase(
        "utility_request_start",
        query,
        query_index,
        query_count,
        &utility,
        started,
        None,
    );
    let response = stream_chat_no_visible_output_with_retry(
        llm.as_ref(),
        "semantic_search rerank",
        cancel,
        || StreamChatRequest {
            model: utility.model.clone(),
            messages: messages.clone(),
            tools: None,
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
            trace_phase(
                "utility_request_error",
                query,
                query_index,
                query_count,
                &utility,
                started,
                Some(&format!("{err:#}")),
            );
            if crate::cim::enabled() {
                return RerankOutcome::error(format!(
                    "Error: semantic_search reranker failed in CIM mode: {err:#}"
                ));
            }
            tracing::warn!(
                error = format!("{err:#}"),
                "semantic_search rerank turn failed; using reciprocal-rank fusion"
            );
            let ordered = rrf_fallback(&candidates, final_k);
            trace_rerank(RerankTrace {
                query,
                query_index,
                query_count,
                raw: &raw,
                final_k,
                base_k,
                deduplicated_count: candidates.len(),
                context_bytes,
                selected_count: 0,
                final_count: ordered.len(),
                fallback_reason: Some("provider_failure"),
                usage: TokenUsage::default(),
                utility: &utility,
            });
            return RerankOutcome::passthrough(
                render_unified(query, &ordered, candidates.len(), notes(&raw), true),
                TokenUsage::default(),
            );
        }
    };
    trace_phase(
        "utility_request_complete",
        query,
        query_index,
        query_count,
        &utility,
        started,
        None,
    );
    let usage = response.usage();
    let text = match response {
        LlmResponse::Text { text, .. } | LlmResponse::ToolCalls { text, .. } => text,
    };

    // 5. A well-formed empty selection is valid and final. Malformed JSON,
    // unknown candidate ids, and invalid declaration choices invoke the
    // ordinary fallback or fail closed in CIM mode.
    let selected = match parse_selections(&text) {
        Some(selected) => selected,
        None => {
            if crate::cim::enabled() {
                trace_phase(
                    "utility_output_error",
                    query,
                    query_index,
                    query_count,
                    &utility,
                    started,
                    Some("malformed structured output"),
                );
                return RerankOutcome {
                    output: "Error: semantic_search reranker returned malformed output in CIM mode"
                        .to_string(),
                    failed: true,
                    usage,
                    usage_model: Some(utility.model.clone()),
                };
            }
            tracing::warn!(
                "semantic_search rerank returned malformed output; using reciprocal-rank fusion"
            );
            let ordered = rrf_fallback(&candidates, final_k);
            trace_rerank(RerankTrace {
                query,
                query_index,
                query_count,
                raw: &raw,
                final_k,
                base_k,
                deduplicated_count: candidates.len(),
                context_bytes,
                selected_count: 0,
                final_count: ordered.len(),
                fallback_reason: Some("malformed_output"),
                usage,
                utility: &utility,
            });
            let mut outcome = RerankOutcome::passthrough(
                render_unified(query, &ordered, candidates.len(), notes(&raw), true),
                usage,
            );
            outcome.usage_model = Some(utility.model.clone());
            return outcome;
        }
    };
    let ordered = match order_candidates(&candidates, &selected, final_k) {
        Ok(ordered) => ordered,
        Err(error) => {
            if crate::cim::enabled() {
                trace_phase(
                    "utility_output_error",
                    query,
                    query_index,
                    query_count,
                    &utility,
                    started,
                    Some(&error),
                );
                return RerankOutcome {
                    output: format!(
                        "Error: semantic_search reranker returned invalid selections in CIM mode: {error}"
                    ),
                    failed: true,
                    usage,
                    usage_model: Some(utility.model.clone()),
                };
            }
            tracing::warn!(%error, "semantic_search rerank returned invalid selections; using reciprocal-rank fusion");
            let ordered = rrf_fallback(&candidates, final_k);
            trace_rerank(RerankTrace {
                query,
                query_index,
                query_count,
                raw: &raw,
                final_k,
                base_k,
                deduplicated_count: candidates.len(),
                context_bytes,
                selected_count: selected.len(),
                final_count: ordered.len(),
                fallback_reason: Some("invalid_selection"),
                usage,
                utility: &utility,
            });
            let mut outcome = RerankOutcome::passthrough(
                render_unified(query, &ordered, candidates.len(), notes(&raw), true),
                usage,
            );
            outcome.usage_model = Some(utility.model.clone());
            return outcome;
        }
    };

    tracing::debug!(
            query = %query,
        candidates = candidates.len(),
        selected = ordered.len(),
        cached_read_tokens = usage.cached_read_tokens,
        input_tokens = usage.input_tokens,
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
        selected_count: selected.len(),
        final_count: ordered.len(),
        fallback_reason: None,
        usage,
        utility: &utility,
    });
    let output = render_unified(query, &ordered, candidates.len(), notes(&raw), false);
    let mut outcome = RerankOutcome::passthrough(output, usage);
    outcome.usage_model = Some(utility.model.clone());
    outcome
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

fn prepare_bifrost_args(query: &str, final_k: usize) -> (Value, usize) {
    let base_k = final_k * OVERFETCH_MULTIPLIER;
    (json!({ "query": query, "k": base_k }), base_k)
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
        let summary_targets: Vec<String> = symbols.iter().chain(&files).cloned().collect();

        fetch_symbol_context(registry, candidates, &symbols).await;
        fetch_summary_context(registry, candidates, &summary_targets).await;
        if available_prompt_bytes(candidates) >= MAX_TOTAL_CONTEXT_BYTES {
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

async fn fetch_summary_context(
    registry: &ToolRegistry,
    candidates: &mut [Candidate],
    targets: &[String],
) {
    if targets.is_empty() {
        return;
    }
    match registry
        .call_bifrost_tool_raw("get_summaries", json!({ "targets": targets }))
        .await
    {
        Ok(value) => attach_summaries(candidates, &value),
        Err(_) if targets.len() > 1 => {
            for target in targets {
                if let Ok(value) = registry
                    .call_bifrost_tool_raw("get_summaries", json!({ "targets": [target] }))
                    .await
                {
                    attach_summaries(candidates, &value);
                }
            }
        }
        Err(_) => {}
    }
}

fn available_prompt_bytes(candidates: &[Candidate]) -> usize {
    candidates
        .iter()
        .map(|candidate| {
            let declarations = candidate
                .declarations
                .iter()
                .map(|declaration| render_declaration_for_prompt(declaration).len())
                .sum::<usize>();
            let context = candidate.context.as_ref().map_or(0, String::len);
            declarations
                .saturating_add(context)
                .min(MAX_CANDIDATE_CONTEXT_BYTES)
        })
        .sum()
}

/// Enforce the disposable-turn source/summary budget without ever removing a
/// candidate identity. Earlier RRF candidates consume the shared budget first.
fn bound_candidate_context(candidates: &mut [Candidate]) -> usize {
    let mut remaining = MAX_TOTAL_CONTEXT_BYTES;
    let mut used = 0;
    for candidate in candidates {
        let mut candidate_remaining = remaining.min(MAX_CANDIDATE_CONTEXT_BYTES);
        let mut declarations = Vec::new();
        for mut declaration in std::mem::take(&mut candidate.declarations) {
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
Each candidate has an id, the symbol or file it refers to, structured declaration signatures \
with ids, and (when available) private source or file-summary context. Decide which candidates \
are genuinely relevant to the query and the task in this conversation. For every selected \
candidate that has declaration ids, choose the one through five declarations most useful for \
locating the relevant implementation. Omit irrelevant candidates.\n\n",
    );
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
        out.push_str(location);
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
        if selection.declarations.len() > MAX_SELECTED_DECLARATIONS {
            return Err(format!(
                "candidate {} selected more than {MAX_SELECTED_DECLARATIONS} declarations",
                selection.id
            ));
        }
        if candidate.declarations.is_empty() && !selection.declarations.is_empty() {
            return Err(format!("candidate {} has no declaration ids", selection.id));
        }
        if !candidate.declarations.is_empty() && selection.declarations.is_empty() {
            return Err(format!(
                "candidate {} must select at least one declaration",
                selection.id
            ));
        }
        let declarations_by_id: HashMap<&str, &DeclarationLocator> = candidate
            .declarations
            .iter()
            .map(|declaration| (declaration.id.as_str(), declaration))
            .collect();
        let mut declaration_seen = HashSet::new();
        let mut declarations = Vec::with_capacity(selection.declarations.len());
        for id in &selection.declarations {
            if !declaration_seen.insert(id.as_str()) {
                return Err(format!(
                    "candidate {} repeated declaration id {id}",
                    selection.id
                ));
            }
            let Some(declaration) = declarations_by_id.get(id.as_str()).copied() else {
                return Err(format!(
                    "candidate {} selected unknown declaration id {id}",
                    selection.id
                ));
            };
            declarations.push(declaration);
        }
        ordered.push(RankedCandidate {
            candidate,
            declarations,
        });
    }
    Ok(ordered)
}

fn rrf_fallback(candidates: &[Candidate], final_k: usize) -> Vec<RankedCandidate<'_>> {
    candidates
        .iter()
        .take(final_k)
        .map(|candidate| RankedCandidate {
            candidate,
            declarations: candidate
                .declarations
                .iter()
                .take(MAX_SELECTED_DECLARATIONS)
                .collect(),
        })
        .collect()
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
    fallback_reason: Option<&'a str>,
    usage: TokenUsage,
    utility: &'a crate::utility_model::UtilityModelSelection,
}

fn trace_phase(
    phase: &str,
    query: &str,
    query_index: usize,
    query_count: usize,
    utility: &crate::utility_model::UtilityModelSelection,
    started: Instant,
    error: Option<&str>,
) {
    append_trace_record(json!({
        "type": "semantic_search_phase",
        "phase": phase,
        "query": query,
        "query_index": query_index,
        "query_count": query_count,
        "elapsed_millis": started.elapsed().as_millis(),
        "utility_model": utility.model,
        "utility_reasoning_effort": utility.reasoning_effort,
        "utility_model_source": utility.source,
        "error": error,
    }));
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
        "query": trace.query,
        "query_index": trace.query_index,
        "query_count": trace.query_count,
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
    fallback: bool,
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
    if fallback {
        out.push_str(
            "Reranker fallback: results use reciprocal-rank fusion and Bifrost-order signatures.\n",
        );
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
        if let Some(location) = &candidate.location {
            out.push_str(&format!("   {location}\n"));
        }
        if selected.declarations.is_empty() {
            out.push_str("   signature unavailable\n");
        } else {
            for declaration in &selected.declarations {
                out.push_str(&format!(
                    "   {} {} at {}:{}-{}\n",
                    declaration.kind,
                    declaration.symbol,
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
            let (forwarded, base_k) = prepare_bifrost_args("needle", final_k);
            assert_eq!(base_k, 2 * final_k);
            assert_eq!(forwarded["k"], 2 * final_k);
            assert_eq!(forwarded["query"], "needle");
            assert!(forwarded.get("queries").is_none());
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
            .map(|candidate| candidate.candidate.name.as_str())
            .collect();
        let second: Vec<&str> = rrf_fallback(&candidates, 2)
            .iter()
            .map(|candidate| candidate.candidate.name.as_str())
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
        assert_eq!(available_prompt_bytes(&candidates), context.len());
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
        let rendered = render_unified("configuration persistence", &ordered, 1, None, false);

        assert!(rendered.contains("fn save(config: &Config)"));
        assert!(!rendered.contains("fn load(path: &Path)"));
        assert!(!rendered.contains("PRIVATE_BODY_SENTINEL"));
        assert!(!rendered.contains("```"));
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

    fn selection(id: &str) -> Selection {
        Selection {
            id: id.to_string(),
            declarations: Vec::new(),
        }
    }
}
