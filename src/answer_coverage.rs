//! Answer-coverage check: a selection nudge at answer-writing time.
//!
//! CodeScale trace audits found the agent surfaces roughly 95 percent of the
//! gold files while it investigates, but only 72-74 percent of what it surfaced
//! reaches the final answer -- and that shortfall is the same under every tool
//! configuration, so it is a selection failure rather than a retrieval one.
//! Timing is what moved inclusion in the same audits: gold injected before the
//! first turn reached the answer 85.6 percent of the time against 50 percent for
//! gold a late `grep_search` surfaced.
//!
//! So this check runs at the moment the answer is written, and only over files
//! the agent has already seen. It never introduces a file the agent did not
//! visit: it ranks the agent's own visited set against the task and names the
//! highly-ranked files the answer left out. The write always succeeds. The
//! checklist is appended to the write's tool result, and every failure -- no
//! bifrost, no semantic index, a refused call, an unreadable artifact --
//! degrades to appending nothing.
//!
//! `BRK_ANSWER_COVERAGE` is the whole gate. Unset, `ToolRegistry` holds no
//! coverage state, nothing is tracked, and nothing is appended.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{Value, json};

use crate::tools::ToolRegistry;
use crate::trace_logging::append_trace_record;

/// Filename glob naming the answer artifact, e.g. `answer.json`. Unset turns the
/// whole feature off.
pub(crate) const ANSWER_COVERAGE_ENV: &str = "BRK_ANSWER_COVERAGE";

/// Candidates asked of each workspace, per query. Deliberately wider than the
/// 20 a model-visible `semantic_search` returns: nothing retrieved here reaches
/// the model except the handful that survive the intersection with the visited
/// set, so a wider pool costs a longer list to filter and nothing else.
const SEMANTIC_SEARCH_K: usize = 40;

/// Files the addendum may name. The checklist is a nudge to re-read a short
/// list, not a second answer.
const MAX_CHECKLIST: usize = 10;

/// Cap on the query derived from the user's own request when no CIM config
/// supplies one. Long enough to carry a whole CodeScale task statement, short
/// enough that the embedder sees a query rather than a document.
const MAX_DERIVED_QUERY_CHARS: usize = 1_000;

/// Per-session coverage state, owned by `ToolRegistry` because the registry
/// lives exactly as long as the session's MCP connections. A task-local would
/// have scoped this to one prompt turn instead.
pub(crate) struct AnswerCoverage {
    /// The configured glob, compiled once, matched against the written file's
    /// name and not its directory.
    artifact_glob: regex::Regex,
    /// Kept as configured, for the trace record.
    artifact_pattern: String,
    state: Mutex<CoverageState>,
}

#[derive(Default)]
struct CoverageState {
    /// Repository-relative (or absolute, when outside every configured root)
    /// paths the agent has actually seen, accumulated at tool-execution time.
    visited: BTreeSet<String>,
    /// Set by the first write that matches the glob, so the check fires once per
    /// session however many times the answer is rewritten.
    fired: bool,
}

impl AnswerCoverage {
    /// The session's coverage state, or `None` when `BRK_ANSWER_COVERAGE` is
    /// unset, empty, or not a usable glob.
    pub(crate) fn from_env() -> Option<Self> {
        let pattern = std::env::var(ANSWER_COVERAGE_ENV).ok()?;
        let pattern = pattern.trim().to_string();
        if pattern.is_empty() {
            return None;
        }
        match Self::new(&pattern) {
            Some(coverage) => Some(coverage),
            None => {
                tracing::warn!(
                    pattern,
                    "{ANSWER_COVERAGE_ENV} is not a usable filename glob; the answer-coverage \
                     check is off for this session"
                );
                None
            }
        }
    }

    pub(crate) fn new(pattern: &str) -> Option<Self> {
        Some(Self {
            artifact_glob: filename_glob(pattern)?,
            artifact_pattern: pattern.to_string(),
            state: Mutex::new(CoverageState::default()),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CoverageState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn record(&self, paths: Vec<String>) {
        self.lock().visited.extend(paths);
    }

    /// Take the one-shot trigger. `true` for the first caller only.
    fn claim(&self) -> bool {
        !std::mem::replace(&mut self.lock().fired, true)
    }

    fn visited(&self) -> Vec<String> {
        self.lock().visited.iter().cloned().collect()
    }

    #[cfg(test)]
    pub(crate) fn visited_snapshot(&self) -> Vec<String> {
        self.visited()
    }
}

/// Compile a filename glob (`*` and `?`) into an anchored regex. Everything else
/// matches literally, so a plain `answer.json` matches that name and no other.
fn filename_glob(pattern: &str) -> Option<regex::Regex> {
    let mut expression = String::from("^");
    for character in pattern.chars() {
        match character {
            '*' => expression.push_str(".*"),
            '?' => expression.push('.'),
            other => expression.push_str(&regex::escape(&other.to_string())),
        }
    }
    expression.push('$');
    regex::Regex::new(&expression).ok()
}

/// Accumulate the workspace files this tool result put in front of the agent.
///
/// Called for every successful tool execution on both dispatch paths. Extraction
/// is a cheap read of arguments and result text the loop already holds; nothing
/// re-parses a trace and nothing touches the filesystem.
pub(crate) fn record_visited(registry: &ToolRegistry, tool_name: &str, args: &Value, output: &str) {
    let Some(coverage) = registry.answer_coverage() else {
        return;
    };
    let paths = visited_paths(
        registry.cwd(),
        registry.additional_roots(),
        tool_name,
        args,
        output,
    );
    if !paths.is_empty() {
        coverage.record(paths);
    }
}

/// The files one tool result showed the agent.
///
/// The three tools carry their paths in three different places, so each is read
/// where its paths actually are:
///
/// - `read_file` names its one file in `file_path`; the result is the content.
/// - `grep_search` prints `path:line_number: text` per match, where `path` is
///   relative to the searched root. That path is recorded as printed, without
///   joining the `path` argument onto it, because a search rooted at a single
///   file reports that file's own display path rather than a path beneath it.
///   Matching is suffix-based (`same_file`), so a search under a subdirectory
///   still resolves against the repository-relative form.
/// - `list_directory` prints bare entry names with no directory at all, so here
///   the `path` argument *is* joined on -- otherwise every entry would degrade
///   into a basename that matches any file of that name anywhere. Names ending
///   in `/` are subdirectories rather than files, and the truncation note is not
///   an entry.
fn visited_paths(
    cwd: &Path,
    additional_roots: &[PathBuf],
    tool_name: &str,
    args: &Value,
    output: &str,
) -> Vec<String> {
    let argument = |key: &str| args.get(key).and_then(Value::as_str);
    let normalize = |raw: &str| normalize_workspace_path(cwd, additional_roots, raw);
    match tool_name {
        "read_file" => argument("file_path")
            .and_then(normalize)
            .into_iter()
            .collect(),
        "grep_search" => output
            .lines()
            .filter_map(grep_match_path)
            .filter_map(normalize)
            .collect(),
        "list_directory" => {
            let directory = argument("path").unwrap_or(".");
            output
                .lines()
                .filter(|line| {
                    !line.is_empty() && !line.ends_with('/') && !line.starts_with("... truncated")
                })
                .filter_map(|entry| normalize(&Path::new(directory).join(entry).to_string_lossy()))
                .collect()
        }
        _ => Vec::new(),
    }
}

/// The path of one `grep_search` result line, rendered as
/// `path:line_number: matched text`.
///
/// The split is at the first `:<digits>:` rather than at the first colon: a
/// Windows drive letter carries one before the path is over, and matched source
/// text usually carries several after it. A line with no such separator is one
/// of the tool's own notes (the truncation marker, the coverage warning, "No
/// matches found") and names no file.
fn grep_match_path(line: &str) -> Option<&str> {
    let mut from = 0;
    while let Some(offset) = line[from..].find(':') {
        let colon = from + offset;
        let rest = &line[colon + 1..];
        let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if colon > 0 && digits > 0 && rest[digits..].starts_with(':') {
            return Some(&line[..colon]);
        }
        from = colon + 1;
    }
    None
}

/// Reduce a path the agent used to the shape bifrost reports: relative to the
/// repository it belongs to, forward slashes, no `./` prefix.
///
/// An absolute path under a configured root -- the session cwd, or one of the
/// ACP additional directories that carry the analysis workspaces -- loses that
/// root. An absolute path under none of them is kept whole: `same_file` compares
/// by path suffix, so it still matches the repository-relative form.
fn normalize_workspace_path(cwd: &Path, additional_roots: &[PathBuf], raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let candidate = PathBuf::from(raw);
    let relative = additional_roots
        .iter()
        .map(PathBuf::as_path)
        .chain(std::iter::once(cwd))
        .find_map(|root| candidate.strip_prefix(root).ok())
        .unwrap_or(candidate.as_path());
    let normalized = relative
        .to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_string();
    (!normalized.is_empty() && normalized != ".").then_some(normalized)
}

/// Whether two paths name the same file, allowing either to be a repository- or
/// search-root-relative tail of the other. The relation is anchored on a path
/// separator, so `pkg/foo.go` matches `repo/pkg/foo.go` and never `mypkg/foo.go`.
fn same_file(left: &str, right: &str) -> bool {
    left == right || left.ends_with(&format!("/{right}")) || right.ends_with(&format!("/{left}"))
}

fn file_name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The coverage addendum for a successful write, or `None` when this write is
/// not the answer artifact, the feature is off, the check already fired, or
/// nothing survived the intersection.
///
/// `task_text` is the user's request for this turn, read only when no CIM config
/// supplies queries.
pub(crate) async fn addendum_for_write(
    registry: &ToolRegistry,
    tool_name: &str,
    args: &Value,
    task_text: &str,
) -> Option<String> {
    if !matches!(tool_name, "write_file" | "edit") {
        return None;
    }
    let coverage = registry.answer_coverage()?;
    let artifact = args.get("file_path").and_then(Value::as_str)?;
    if !coverage
        .artifact_glob
        .is_match(file_name_of(&artifact.replace('\\', "/")))
    {
        return None;
    }
    if !coverage.claim() {
        return None;
    }

    let started = Instant::now();
    let visited = coverage.visited();
    let outcome = evaluate(registry, &visited, artifact, task_text).await;
    append_trace_record(json!({
        "type": "answer_coverage_check",
        "status": outcome.status,
        "artifact_pattern": coverage.artifact_pattern,
        "artifact_path": artifact,
        "queries": outcome.queries,
        "query_source": outcome.query_source,
        "visited_count": visited.len(),
        "workspaces": outcome
            .per_workspace
            .iter()
            .map(|workspace| json!({
                "workspace": workspace.workspace,
                "candidate_count": workspace.paths.len(),
                "errors": workspace.errors,
            }))
            .collect::<Vec<Value>>(),
        "checklist": outcome.checklist,
        "elapsed_millis": started.elapsed().as_millis(),
    }));
    (!outcome.checklist.is_empty()).then(|| render_addendum(&outcome.checklist))
}

/// One coverage check, as the trace records it.
struct CheckOutcome {
    /// `delivered` when the model was given a checklist, `empty` when the
    /// retrieval ran and left nothing to say, `skipped_error` when there was no
    /// usable query or no workspace answered.
    status: &'static str,
    queries: Vec<String>,
    query_source: &'static str,
    per_workspace: Vec<WorkspaceCandidates>,
    checklist: Vec<String>,
}

impl CheckOutcome {
    fn skipped(queries: Vec<String>, query_source: &'static str) -> Self {
        Self {
            status: "skipped_error",
            queries,
            query_source,
            per_workspace: Vec::new(),
            checklist: Vec::new(),
        }
    }
}

/// What one workspace's retrieval contributed.
struct WorkspaceCandidates {
    workspace: Option<String>,
    /// Candidate file paths in retrieval order, deduplicated.
    paths: Vec<String>,
    /// Failures against this workspace, kept so a silent check can be explained
    /// from the trace.
    errors: Vec<String>,
}

async fn evaluate(
    registry: &ToolRegistry,
    visited: &[String],
    artifact: &str,
    task_text: &str,
) -> CheckOutcome {
    let Some((queries, query_source)) = queries_for_check(task_text) else {
        return CheckOutcome::skipped(Vec::new(), "none");
    };

    // Every harness-issued bifrost call must name a configured workspace, so the
    // fan-out is over the same names the MCP command line was built from (anvil
    // AGENTS.md). `None` is the single-root shape, which has no router to name a
    // workspace to.
    let names: Vec<Option<String>> = match registry
        .analysis_workspaces()
        .filter(|workspaces| !workspaces.is_empty())
    {
        Some(workspaces) => workspaces
            .iter()
            .map(|workspace| Some(workspace.name.clone()))
            .collect(),
        None => vec![None],
    };
    let mut per_workspace = Vec::with_capacity(names.len());
    for name in names {
        per_workspace.push(candidates_for_workspace(registry, name, &queries).await);
    }
    if per_workspace
        .iter()
        .all(|workspace| workspace.paths.is_empty() && !workspace.errors.is_empty())
    {
        return CheckOutcome {
            status: "skipped_error",
            queries,
            query_source,
            per_workspace,
            checklist: Vec::new(),
        };
    }

    let answered = answer_paths(registry, artifact);
    let checklist = checklist(&per_workspace, visited, &answered);
    CheckOutcome {
        status: if checklist.is_empty() {
            "empty"
        } else {
            "delivered"
        },
        queries,
        query_source,
        per_workspace,
        checklist,
    }
}

/// The exact note the model reads, appended to the write's own result so the
/// write reports success and then says what it may have left out.
fn render_addendum(checklist: &[String]) -> String {
    format!(
        "\n\nCoverage check: you examined these files and they rank highly against the task, but \
         they are not in your answer: {}. Review whether any belong before finalizing.",
        checklist.join(", ")
    )
}

/// Resolve the queries this check ranks with, and say where they came from.
///
/// A CIM config is read whenever `BRK_CIM_CONFIG` names one, deliberately
/// without requiring `BRK_CIM_EVAL`: this check is a separate treatment from
/// step zero and has to work in an arm that runs no step zero at all.
fn queries_for_check(task_text: &str) -> Option<(Vec<String>, &'static str)> {
    match crate::cim::configured_queries() {
        Some(Ok(queries)) if !queries.is_empty() => return Some((queries, "cim_config")),
        Some(Ok(_)) => {}
        Some(Err(error)) => {
            tracing::warn!("answer-coverage check could not read the CIM query config: {error:#}");
        }
        None => {}
    }
    derived_query(task_text).map(|query| (vec![query], "user_request"))
}

/// The fallback query: the user's own request, whitespace-normalized and cut
/// back to a whole word at `MAX_DERIVED_QUERY_CHARS`.
fn derived_query(task_text: &str) -> Option<String> {
    let normalized = task_text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return None;
    }
    if normalized.chars().count() <= MAX_DERIVED_QUERY_CHARS {
        return Some(normalized);
    }
    let cut = normalized
        .char_indices()
        .nth(MAX_DERIVED_QUERY_CHARS)
        .map_or(normalized.len(), |(index, _)| index);
    let head = &normalized[..cut];
    Some(
        head.rsplit_once(' ')
            .map_or(head, |(head, _)| head)
            .to_string(),
    )
}

/// Rank one workspace's files against every query.
///
/// Two legs of `semantic_search` carry file identity, and only one carries it
/// directly. `coedit_ranked` is a list of paths, but bifrost seeds that leg with
/// the dense top files and then removes the seeds from its own result (bifrost
/// `relevance.rs`: `let excluded: HashSet<_> = seed_weights.keys()...`), so on
/// its own it names everything except the files the query matched best.
/// `vector_ranked` is that dense leg, and it identifies symbols by
/// fully-qualified name with no path at all. So each search is followed by one
/// batched `get_summaries` that resolves those names to files: one extra call
/// per query per workspace, which is the enrichment the reranker already does
/// per candidate, done once for the whole list because only the path is wanted.
async fn candidates_for_workspace(
    registry: &ToolRegistry,
    workspace: Option<String>,
    queries: &[String],
) -> WorkspaceCandidates {
    let mut paths: Vec<String> = Vec::new();
    let mut errors = Vec::new();
    for query in queries {
        let arguments = crate::semantic_rerank::bifrost_args_with_workspace(
            json!({ "query": query, "k": SEMANTIC_SEARCH_K }),
            workspace.as_deref(),
        );
        let raw = match crate::semantic_rerank::call_bifrost_tool_with_backpressure(
            registry,
            "semantic_search",
            arguments,
        )
        .await
        {
            Ok(raw) => raw,
            Err(error) => {
                errors.push(format!("{error:#}"));
                continue;
            }
        };
        let symbols = ranked_symbols(&raw);
        let resolved = resolve_symbol_paths(registry, &symbols, workspace.as_deref()).await;
        for symbol in &symbols {
            if let Some(path) = resolved.by_symbol.get(symbol) {
                push_unique(&mut paths, path);
            }
        }
        for path in &resolved.compact {
            push_unique(&mut paths, path);
        }
        for path in ranked_files(&raw) {
            push_unique(&mut paths, &path);
        }
    }
    WorkspaceCandidates {
        workspace,
        paths,
        errors,
    }
}

fn push_unique(paths: &mut Vec<String>, path: &str) {
    if !paths.iter().any(|existing| existing == path) {
        paths.push(path.to_string());
    }
}

fn ranked_symbols(raw: &Value) -> Vec<String> {
    raw.get("vector_ranked")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("fqfn").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn ranked_files(raw: &Value) -> Vec<String> {
    raw.get("coedit_ranked")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("path").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Files behind a batch of ranked symbols.
#[derive(Default)]
struct SymbolPaths {
    /// Fully-qualified name to repository-relative path, from the full summary
    /// shape. This is what preserves the dense ranking's order.
    by_symbol: HashMap<String, String>,
    /// Paths from the compact shape bifrost degrades an oversized aggregate
    /// summary to. It names files without saying which symbol asked for them, so
    /// these keep the order bifrost listed them in and follow the ranked ones.
    compact: Vec<String>,
}

async fn resolve_symbol_paths(
    registry: &ToolRegistry,
    symbols: &[String],
    workspace: Option<&str>,
) -> SymbolPaths {
    if symbols.is_empty() {
        return SymbolPaths::default();
    }
    let arguments = crate::semantic_rerank::bifrost_args_with_workspace(
        json!({ "targets": symbols }),
        workspace,
    );
    let raw = match crate::semantic_rerank::call_bifrost_tool_with_backpressure(
        registry,
        "get_summaries",
        arguments,
    )
    .await
    {
        Ok(raw) => raw,
        Err(error) => {
            // The dense leg is lost for this query and the co-edit leg is not. A
            // partial ranking over the agent's own files is still a ranking.
            tracing::debug!(
                "answer-coverage check could not resolve ranked symbols to files: {error:#}"
            );
            return SymbolPaths::default();
        }
    };
    let mut resolved = SymbolPaths::default();
    for block in raw
        .get("summaries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let (Some(label), Some(path)) = (
            block.get("label").and_then(Value::as_str),
            block.get("path").and_then(Value::as_str),
        ) else {
            continue;
        };
        resolved
            .by_symbol
            .entry(label.to_string())
            .or_insert_with(|| path.to_string());
    }
    for file in raw
        .get("compact_symbols")
        .and_then(|compact| compact.get("files"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(path) = file.get("path").and_then(Value::as_str) {
            push_unique(&mut resolved.compact, path);
        }
    }
    resolved
}

/// Path-shaped tokens in the artifact the write just produced.
///
/// The file is read back from disk rather than taken from the write arguments,
/// so an `edit` that changed part of an answer is read the same way a
/// `write_file` that replaced all of it is. A token counts as a path when it
/// contains a separator; bare words are not treated as filenames, because an
/// answer is prose as well as paths.
fn answer_paths(registry: &ToolRegistry, artifact: &str) -> Vec<String> {
    let Ok(resolved) =
        crate::tools::safe_resolve_in_roots(registry.cwd(), registry.additional_roots(), artifact)
    else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&resolved) else {
        return Vec::new();
    };
    text.split(|c: char| c.is_whitespace() || "\"'`,[]{}()<>|".contains(c))
        .map(|token| token.trim_matches(|c: char| matches!(c, '.' | ':' | ';' | '#')))
        .filter(|token| token.contains('/'))
        .filter_map(|token| {
            normalize_workspace_path(registry.cwd(), registry.additional_roots(), token)
        })
        .collect()
}

/// The checklist: retrieved candidates the agent visited and the answer omits,
/// in retrieval order, capped at `MAX_CHECKLIST`.
///
/// Workspaces are walked in configured order and, within a workspace, queries in
/// configured order, so a single-workspace single-query session sees exactly
/// bifrost's own ranking.
fn checklist(
    per_workspace: &[WorkspaceCandidates],
    visited: &[String],
    answered: &[String],
) -> Vec<String> {
    // Both arms of `same_file` imply equal file names, so grouping by name turns
    // the scan over every visited path into a lookup.
    fn by_file_name(paths: &[String]) -> HashMap<&str, Vec<&str>> {
        let mut index: HashMap<&str, Vec<&str>> = HashMap::new();
        for path in paths {
            index.entry(file_name_of(path)).or_default().push(path);
        }
        index
    }
    let visited_by_name = by_file_name(visited);
    let answered_by_name = by_file_name(answered);
    let contains = |index: &HashMap<&str, Vec<&str>>, candidate: &str| {
        index
            .get(file_name_of(candidate))
            .is_some_and(|paths| paths.iter().any(|path| same_file(candidate, path)))
    };

    let mut checklist: Vec<String> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for workspace in per_workspace {
        for candidate in &workspace.paths {
            if checklist.len() == MAX_CHECKLIST {
                return checklist;
            }
            if seen.insert(candidate.as_str())
                && contains(&visited_by_name, candidate)
                && !contains(&answered_by_name, candidate)
            {
                checklist.push(candidate.clone());
            }
        }
    }
    checklist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coverage(pattern: &str) -> AnswerCoverage {
        AnswerCoverage::new(pattern).expect("a usable glob")
    }

    #[test]
    fn a_plain_name_glob_matches_that_name_only() {
        let glob = coverage("answer.json").artifact_glob;
        assert!(glob.is_match("answer.json"));
        assert!(!glob.is_match("answerXjson"));
        assert!(!glob.is_match("my-answer.json"));
    }

    #[test]
    fn a_wildcard_glob_matches_a_family_of_names() {
        let glob = coverage("answer*.json").artifact_glob;
        assert!(glob.is_match("answer.json"));
        assert!(glob.is_match("answer-final.json"));
        assert!(!glob.is_match("answer.txt"));
    }

    #[test]
    fn the_trigger_is_claimed_once() {
        let coverage = coverage("answer.json");
        assert!(coverage.claim());
        assert!(!coverage.claim());
        assert!(!coverage.claim());
    }

    #[test]
    fn grep_lines_give_up_their_paths_and_notes_do_not() {
        assert_eq!(
            grep_match_path("pkg/storage/cacher.go:412: func (c *Cacher) Watch("),
            Some("pkg/storage/cacher.go")
        );
        // A matched line that itself contains colons still splits at the line
        // number, which is the only colon-delimited integer field.
        assert_eq!(
            grep_match_path("a/b.rs:7: let x: usize = 1;"),
            Some("a/b.rs")
        );
        assert_eq!(grep_match_path("... truncated at 50 results"), None);
        assert_eq!(grep_match_path("No matches found for 'x'"), None);
        assert_eq!(
            grep_match_path("warning: this search could not cover everything:"),
            None
        );
    }

    #[test]
    fn visited_paths_read_each_tool_where_its_paths_actually_are() {
        let cwd = PathBuf::from("/work/repo");
        let roots = vec![PathBuf::from("/workspace/api")];

        assert_eq!(
            visited_paths(
                &cwd,
                &roots,
                "read_file",
                &json!({ "file_path": "src/main.rs" }),
                "fn main() {}"
            ),
            vec!["src/main.rs".to_string()]
        );
        // An absolute path under a configured additional root is reported the way
        // bifrost reports it: relative to that repository.
        assert_eq!(
            visited_paths(
                &cwd,
                &roots,
                "read_file",
                &json!({ "file_path": "/workspace/api/pkg/server.go" }),
                ""
            ),
            vec!["pkg/server.go".to_string()]
        );
        // grep paths are already rooted at the search path, so they are kept as
        // printed and the tool's own notes are skipped.
        assert_eq!(
            visited_paths(
                &cwd,
                &roots,
                "grep_search",
                &json!({ "pattern": "Watch", "path": "pkg" }),
                "storage/cacher.go:412: func Watch()\nstorage/etcd.go:8: Watch\n\
                 ... truncated at 50 results"
            ),
            vec![
                "storage/cacher.go".to_string(),
                "storage/etcd.go".to_string()
            ]
        );
        // list_directory prints bare names, so the directory argument is joined
        // back on and subdirectories are skipped.
        assert_eq!(
            visited_paths(
                &cwd,
                &roots,
                "list_directory",
                &json!({ "path": "src/tools" }),
                "filesystem.rs\nmod.rs\nnested/"
            ),
            vec![
                "src/tools/filesystem.rs".to_string(),
                "src/tools/mod.rs".to_string()
            ]
        );
        // A tool that shows no files contributes none, whatever its output looks
        // like.
        assert!(
            visited_paths(
                &cwd,
                &roots,
                "run_shell_command",
                &json!({ "command": "ls src" }),
                "src/main.rs"
            )
            .is_empty()
        );
    }

    #[test]
    fn a_partial_prefix_still_names_the_same_file() {
        assert!(same_file("pkg/foo.go", "pkg/foo.go"));
        assert!(same_file("repo/pkg/foo.go", "pkg/foo.go"));
        assert!(same_file("pkg/foo.go", "repo/pkg/foo.go"));
        assert!(!same_file("mypkg/foo.go", "pkg/foo.go"));
        assert!(!same_file("pkg/foo.go", "pkg/bar.go"));
    }

    #[test]
    fn the_derived_query_is_normalized_and_cut_at_a_word() {
        assert_eq!(
            derived_query("  find   the\nwatch cache\t"),
            Some("find the watch cache".to_string())
        );
        assert_eq!(derived_query("   "), None);

        let long = "alpha ".repeat(400);
        let derived = derived_query(&long).expect("a long request still derives a query");
        assert!(
            derived.chars().count() <= MAX_DERIVED_QUERY_CHARS,
            "{derived}"
        );
        assert!(
            derived.ends_with("alpha"),
            "cut back to a word boundary: {derived}"
        );
    }

    #[test]
    fn the_checklist_is_visited_minus_answered_in_retrieval_order_and_capped() {
        let per_workspace = vec![
            WorkspaceCandidates {
                workspace: Some("api".to_string()),
                paths: (1..=8).map(|n| format!("api/f{n}.go")).collect(),
                errors: Vec::new(),
            },
            WorkspaceCandidates {
                workspace: Some("ui".to_string()),
                paths: (1..=8).map(|n| format!("ui/f{n}.ts")).collect(),
                errors: Vec::new(),
            },
        ];
        let visited: Vec<String> = (1..=8)
            .map(|n| format!("api/f{n}.go"))
            .chain((1..=8).map(|n| format!("ui/f{n}.ts")))
            .collect();
        // The answer already names two of them, one written with a repository
        // prefix the retrieval does not use.
        let answered = vec!["api/f1.go".to_string(), "checkout/ui/f2.ts".to_string()];

        assert_eq!(
            checklist(&per_workspace, &visited, &answered),
            vec![
                "api/f2.go",
                "api/f3.go",
                "api/f4.go",
                "api/f5.go",
                "api/f6.go",
                "api/f7.go",
                "api/f8.go",
                "ui/f1.ts",
                "ui/f3.ts",
                "ui/f4.ts",
            ],
            "retrieval order, answered files removed, capped at {MAX_CHECKLIST}"
        );
    }

    #[test]
    fn a_candidate_the_agent_never_visited_is_not_on_the_checklist() {
        let per_workspace = vec![WorkspaceCandidates {
            workspace: None,
            paths: vec!["seen.go".to_string(), "unseen.go".to_string()],
            errors: Vec::new(),
        }];

        assert_eq!(
            checklist(&per_workspace, &["repo/seen.go".to_string()], &[]),
            vec!["seen.go".to_string()],
            "the check is selection support over the agent's own files, not discovery"
        );
    }

    #[test]
    fn the_addendum_names_the_files_and_asks_for_a_review() {
        assert_eq!(
            render_addendum(&["a/b.go".to_string(), "c/d.go".to_string()]),
            "\n\nCoverage check: you examined these files and they rank highly against the task, \
             but they are not in your answer: a/b.go, c/d.go. Review whether any belong before \
             finalizing."
        );
    }
}
