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
//! Every failure mode degrades to passing through bifrost's raw result, so the
//! wrapper can only improve search, never break it.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::llm_client::{
    ChatMessage, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage,
    stream_chat_no_visible_output_with_retry,
};
use crate::structured_output::StructuredOutputRequest;
use crate::tools::ToolRegistry;

/// Cap on how many symbol / file candidates we fetch context for and feed to
/// the rerank turn. `semantic_search` returns at most `k` per leg (model
/// chooses `k`, up to 100); these bound the disposable-turn cost regardless.
const MAX_SYMBOL_CANDIDATES: usize = 30;
const MAX_FILE_CANDIDATES: usize = 20;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    idle_timeout: Duration,
    cancel: &CancellationToken,
) -> RerankOutcome {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    // 1. Underlying search. A hard failure here is the model's to see.
    let raw = match registry
        .call_bifrost_tool_raw("semantic_search", args.clone())
        .await
    {
        Ok(value) => value,
        Err(err) => return RerankOutcome::error(format!("Error: {err}")),
    };
    let raw_pretty = serde_json::to_string_pretty(&raw).unwrap_or_else(|_| raw.to_string());

    // 2. Candidates. Nothing to rerank -> pass the raw payload through.
    let mut candidates = parse_candidates(&raw);
    if candidates.is_empty() {
        return RerankOutcome::passthrough(raw_pretty, TokenUsage::default());
    }

    // 3. Fetch source / summaries for the candidates (best effort).
    fetch_context(registry, &mut candidates).await;

    // 4. Disposable relevance turn on top of the live conversation.
    let mut messages = prefix_for_rerank(prior_messages);
    messages.push(ChatMessage::user(build_rerank_prompt(&query, &candidates)));
    let structured = StructuredOutputRequest {
        schema_name: "semantic_rerank".to_string(),
        schema: rerank_schema(),
        allow_coercion: true,
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
            temperature: None,
            structured_output: Some(structured.clone()),
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: cancel.clone(),
            idle_timeout,
        },
    )
    .await;

    let response = match response {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!(
                error = format!("{err:#}"),
                "semantic_search rerank turn failed; passing through raw results"
            );
            return RerankOutcome::passthrough(raw_pretty, TokenUsage::default());
        }
    };
    let usage = response.usage();
    let text = match response {
        LlmResponse::Text { text, .. } | LlmResponse::ToolCalls { text, .. } => text,
    };

    // 5. Map the selection back to candidates; empty/garbage -> raw passthrough.
    let selected = parse_selected_ids(&text).unwrap_or_default();
    let ordered = order_candidates(&candidates, &selected);
    if ordered.is_empty() {
        tracing::warn!(
            "semantic_search rerank returned no usable selection; passing through raw results"
        );
        return RerankOutcome::passthrough(raw_pretty, usage);
    }

    tracing::debug!(
        query = %query,
        candidates = candidates.len(),
        selected = ordered.len(),
        cached_read_tokens = usage.cached_read_tokens,
        input_tokens = usage.input_tokens,
        "semantic_search reranked"
    );
    let output = render_unified(&query, &ordered, candidates.len(), notes(&raw));
    RerankOutcome::passthrough(output, usage)
}

/// Parse `semantic_search`'s three legs into a deduplicated candidate list:
/// symbols (vector ∪ bm25, max score, unioned signals) then files (coedit).
fn parse_candidates(raw: &Value) -> Vec<Candidate> {
    let mut symbols: Vec<(String, f32, Vec<&'static str>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for (key, signal) in [("vector_ranked", "vector"), ("bm25_ranked", "bm25")] {
        let Some(array) = raw.get(key).and_then(Value::as_array) else {
            continue;
        };
        for item in array {
            let Some(fqfn) = item.get("fqfn").and_then(Value::as_str) else {
                continue;
            };
            let score = item.get("score").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            match index.get(fqfn) {
                Some(&i) => {
                    symbols[i].1 = symbols[i].1.max(score);
                    if !symbols[i].2.contains(&signal) {
                        symbols[i].2.push(signal);
                    }
                }
                None => {
                    index.insert(fqfn.to_string(), symbols.len());
                    symbols.push((fqfn.to_string(), score, vec![signal]));
                }
            }
        }
    }
    symbols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    symbols.truncate(MAX_SYMBOL_CANDIDATES);

    let mut files: Vec<(String, f32)> = Vec::new();
    if let Some(array) = raw.get("coedit_ranked").and_then(Value::as_array) {
        for item in array {
            let Some(path) = item.get("path").and_then(Value::as_str) else {
                continue;
            };
            let score = item.get("score").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            files.push((path.to_string(), score));
        }
    }
    files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    files.truncate(MAX_FILE_CANDIDATES);

    let mut candidates = Vec::with_capacity(symbols.len() + files.len());
    for (i, (name, _score, signals)) in symbols.into_iter().enumerate() {
        candidates.push(Candidate {
            id: format!("s{}", i + 1),
            kind: CandidateKind::Symbol,
            name,
            signals,
            location: None,
            context: None,
        });
    }
    for (i, (name, _score)) in files.into_iter().enumerate() {
        candidates.push(Candidate {
            id: format!("f{}", i + 1),
            kind: CandidateKind::File,
            name,
            signals: vec!["coedit"],
            location: None,
            context: None,
        });
    }
    candidates
}

/// Batch-fetch symbol source (`get_symbol_sources`) and file summaries
/// (`get_summaries`) for the candidates and attach truncated context. Partial
/// or total failure is non-fatal: a candidate without context still appears by
/// name in the rerank prompt.
async fn fetch_context(registry: &ToolRegistry, candidates: &mut [Candidate]) {
    let symbols: Vec<String> = candidates
        .iter()
        .filter(|c| c.kind == CandidateKind::Symbol)
        .map(|c| c.name.clone())
        .collect();
    let files: Vec<String> = candidates
        .iter()
        .filter(|c| c.kind == CandidateKind::File)
        .map(|c| c.name.clone())
        .collect();

    if !symbols.is_empty()
        && let Ok(value) = registry
            .call_bifrost_tool_raw("get_symbol_sources", json!({ "symbols": symbols }))
            .await
    {
        attach_symbol_sources(candidates, &value);
    }

    if !files.is_empty()
        && let Ok(value) = registry
            .call_bifrost_tool_raw("get_summaries", json!({ "targets": files }))
            .await
    {
        attach_summaries(candidates, &value);
    }
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
    let Some(summaries) = result.get("summaries").and_then(Value::as_array) else {
        return;
    };
    let mut by_path: HashMap<&str, &Value> = HashMap::new();
    for block in summaries {
        let key = block
            .get("path")
            .and_then(Value::as_str)
            .or_else(|| block.get("label").and_then(Value::as_str));
        if let Some(key) = key {
            by_path.entry(key).or_insert(block);
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

fn build_rerank_prompt(query: &str, candidates: &[Candidate]) -> String {
    let mut out = String::new();
    out.push_str(
        "A code search just ran for the query below and returned these candidate results. \
Each candidate has an id, the symbol or file it refers to, and (when available) its source \
or file summary. Decide which candidates are genuinely relevant to the query and the task \
in this conversation, then return their ids ordered most-relevant first. Omit irrelevant ones.\n\n",
    );
    out.push_str(
        "Respond with ONLY a JSON object of the form {\"relevant\": [\"<id>\", ...]} using the \
exact ids shown. If nothing is relevant, return {\"relevant\": []}.\n\n",
    );
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
fn order_candidates<'a>(candidates: &'a [Candidate], selected: &[String]) -> Vec<&'a Candidate> {
    let by_id: HashMap<&str, &Candidate> = candidates.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut ordered = Vec::new();
    for id in selected {
        if seen.contains_key(id.as_str()) {
            continue;
        }
        if let Some(candidate) = by_id.get(id.as_str()) {
            seen.insert(id.as_str(), ());
            ordered.push(*candidate);
        }
    }
    ordered
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
/// which degrades to raw passthrough -- no bespoke extraction needed.
fn parse_selected_ids(text: &str) -> Option<Vec<String>> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    let array = value.get("relevant")?.as_array()?;
    Some(
        array
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
    )
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
        let ordered = order_candidates(&candidates, &selected);
        let names: Vec<&str> = ordered.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["a.B.d", "a.B.c"]);
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
        );
        let messages = vec![ChatMessage::user(prompt)];
        let structured = StructuredOutputRequest {
            schema_name: "semantic_rerank".to_string(),
            schema: rerank_schema(),
            allow_coercion: true,
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
                temperature: None,
                structured_output: Some(structured.clone()),
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeout: Duration::from_secs(60),
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
