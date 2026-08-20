//! CIM evaluation-only synthetic semantic-search configuration.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::p2t::{ForcedStep, PrefixToolCall};

pub(crate) const CIM_EVAL_ENV: &str = "BRK_CIM_EVAL";
const CIM_CONFIG_ENV: &str = "BRK_CIM_CONFIG";
const CIM_SCHEMA_VERSION: u32 = 1;
const CIM_FINAL_K: usize = 20;
const CIM_MCP_TOOL_CALL_TIMEOUT: Duration = Duration::from_secs(1_800);

pub(crate) fn mcp_tool_call_timeout() -> Option<Duration> {
    enabled().then_some(CIM_MCP_TOOL_CALL_TIMEOUT)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CimConfig {
    pub query_manifest_sha256: String,
    pub k: usize,
    pub queries: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CimConfigFile {
    schema_version: u32,
    query_manifest_sha256: String,
    k: usize,
    queries: Vec<String>,
}

// Test-only stand-in for the two environment variables that configure step
// zero: the parsed queries, and the config path the once-per-eval claim file is
// derived from (the file itself is never read).
//
// It is a task-local, scoped like `trace_logging::with_trace_path`, because
// `std::env::set_var` is not a safe way to configure a test in this crate. The
// unit tests share one binary and run in parallel, so one test's env write races
// every other test's env read: setting `BRK_CIM_EVAL` and `BRK_CIM_CONFIG`
// around a session test made two unrelated tests fail and made the CIM tests
// read each other's config path. A task-local is visible to the task under test
// and to nothing else.
#[cfg(test)]
tokio::task_local! {
    static TEST_EVAL: (CimConfig, PathBuf);
}

/// Run `future` as though `BRK_CIM_EVAL=1` and `BRK_CIM_CONFIG=config_path` were
/// set, with `config_path` already parsed into `config`. Give each test its own
/// `config_path` so it claims step zero for itself.
///
/// Unix-gated with its only consumers: the tool_loop step-zero tests drive a
/// fake bifrost shell script, so on Windows this helper would compile unused
/// and fail the deny-warnings build.
#[cfg(all(test, unix))]
pub(crate) async fn with_test_eval<F: std::future::Future>(
    config: CimConfig,
    config_path: PathBuf,
    future: F,
) -> F::Output {
    TEST_EVAL.scope((config, config_path), future).await
}

#[cfg(test)]
fn test_eval() -> Option<(CimConfig, PathBuf)> {
    TEST_EVAL.try_with(Clone::clone).ok()
}

#[cfg(not(test))]
fn test_eval() -> Option<(CimConfig, PathBuf)> {
    None
}

pub(crate) fn enabled() -> bool {
    test_eval().is_some() || crate::p2t::env_var_truthy(CIM_EVAL_ENV)
}

pub(crate) fn load_config_from_env(
    p2t_enabled: bool,
    train_bifrost_enabled: bool,
) -> Result<Option<CimConfig>> {
    if !enabled() {
        return Ok(None);
    }
    if p2t_enabled || train_bifrost_enabled {
        bail!("{CIM_EVAL_ENV} cannot be combined with BRK_PATCHES_TO_TRACES or BRK_TRAIN_BIFROST");
    }
    if let Some((config, _)) = test_eval() {
        return Ok(Some(config));
    }
    let path = config_path_from_env()?;
    load_config(&path).map(Some)
}

/// The configured step-zero queries, for a consumer that is not step zero.
///
/// `None` when `BRK_CIM_CONFIG` names no config. Deliberately independent of
/// `BRK_CIM_EVAL`: the answer-coverage check reuses the same per-task query list
/// in an arm that runs no forced step at all, so requiring the step-zero switch
/// would make the query list unreachable exactly where it is wanted.
pub(crate) fn configured_queries() -> Option<Result<Vec<String>>> {
    if let Some((config, _)) = test_eval() {
        return Some(Ok(config.queries));
    }
    if std::env::var(CIM_CONFIG_ENV).is_err() {
        return None;
    }
    Some(
        config_path_from_env()
            .and_then(|path| load_config(&path))
            .map(|config| config.queries),
    )
}

fn config_path_from_env() -> Result<PathBuf> {
    let path = PathBuf::from(
        std::env::var(CIM_CONFIG_ENV)
            .with_context(|| format!("{CIM_CONFIG_ENV} must be set when {CIM_EVAL_ENV}=1"))?,
    );
    if !path.is_absolute() {
        bail!("{CIM_CONFIG_ENV} must be an absolute path");
    }
    Ok(path)
}

pub(crate) fn claim_synthetic_step() -> Result<bool> {
    match test_eval() {
        Some((_, config_path)) => claim_synthetic_step_at(&config_path),
        None => claim_synthetic_step_at(&config_path_from_env()?),
    }
}

fn claim_synthetic_step_at(config_path: &Path) -> Result<bool> {
    let claim_path = config_path.with_extension("step-0.claimed");
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&claim_path)
    {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error)
            .with_context(|| format!("failed to claim CIM step zero at {}", claim_path.display())),
    }
}

fn load_config(path: &Path) -> Result<CimConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed: CimConfigFile = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if parsed.schema_version != CIM_SCHEMA_VERSION {
        bail!(
            "unsupported CIM config schema version {}; expected {CIM_SCHEMA_VERSION}",
            parsed.schema_version
        );
    }
    if parsed.query_manifest_sha256.trim().is_empty() {
        bail!("query_manifest_sha256 must not be empty");
    }
    if parsed.k != CIM_FINAL_K {
        bail!("CIM semantic-search k must be exactly {CIM_FINAL_K}");
    }
    if parsed.queries.len() > 3 {
        bail!("CIM semantic queries must contain at most 3 entries");
    }
    let mut seen = HashSet::new();
    for query in &parsed.queries {
        if query.trim().is_empty() {
            bail!("CIM semantic queries must not be empty");
        }
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized != *query {
            bail!("CIM semantic queries must have normalized whitespace: {query:?}");
        }
        if !seen.insert(query.to_lowercase()) {
            bail!("CIM semantic queries must be unique ignoring case: {query:?}");
        }
    }
    Ok(CimConfig {
        query_manifest_sha256: parsed.query_manifest_sha256,
        k: parsed.k,
        queries: parsed.queries,
    })
}

/// Call id of the workspace-less step-zero call, and the prefix every
/// per-workspace id extends. Stable so a trace consumer can find step zero
/// without reading the arguments.
const STEP_ZERO_CALL_ID: &str = "cim-step-0-semantic-search";

/// Step zero: one forced assistant step that retrieves before the model's
/// first turn.
///
/// The step has to name its workspaces whenever the session has named ones.
/// Anvil starts bifrost either as `--root <cwd>` or as one
/// `--workspace <name>=<path>` per repository (`McpServerConfig::rendered_args`),
/// and in the second shape bifrost's router rejects any tool call without a
/// configured `workspace` argument ("workspace must be one configured name").
/// Every CodeScale eval runs the named shape, so a workspace-less forced call
/// came back as an immediate `retrieval_error` and killed the session before the
/// first real turn -- the same failure the readiness probe hit in
/// `semantic_readiness`.
///
/// So the names come from the same session configuration that built bifrost's
/// command line, and the step asks every configured workspace the same queries:
/// one `semantic_search` call per workspace, each with its own stable id, all in
/// this one step. Cross-repository coverage is the point of the arm, and
/// retrieving from a single workspace would forfeit the multi-repo tasks. The
/// CIM config itself stays workspace-agnostic (queries and k only), because the
/// query manifest is shared across tasks whose repository sets differ.
pub(crate) fn synthetic_step(
    config: &CimConfig,
    analysis_workspaces: Option<&[crate::session::AnalysisWorkspace]>,
) -> ForcedStep {
    let named = analysis_workspaces.filter(|workspaces| !workspaces.is_empty());
    ForcedStep {
        assistant_text: String::new(),
        tool_calls: if config.queries.is_empty() {
            Vec::new()
        } else {
            match named {
                Some(workspaces) => workspaces
                    .iter()
                    .map(|workspace| semantic_search_call(config, Some(&workspace.name)))
                    .collect(),
                None => vec![semantic_search_call(config, None)],
            }
        },
        message: None,
    }
}

/// One step-zero `semantic_search` call. `workspace` is `None` only in the
/// single-root shape, where bifrost has no router to name a workspace to.
fn semantic_search_call(config: &CimConfig, workspace: Option<&str>) -> PrefixToolCall {
    PrefixToolCall {
        // Workspace names are unique and restricted to `[A-Za-z0-9._-]`
        // (`acp::validate_analysis_workspaces`, `mcp::workspace_slug`), so the
        // suffixed ids stay distinct and readable.
        id: match workspace {
            Some(name) => format!("{STEP_ZERO_CALL_ID}-{name}"),
            None => STEP_ZERO_CALL_ID.to_string(),
        },
        name: "semantic_search".to_string(),
        arguments: crate::semantic_rerank::bifrost_args_with_workspace(
            serde_json::json!({ "queries": config.queries, "k": config.k }),
            workspace,
        )
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn config_file(payload: serde_json::Value) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(
            &mut file,
            serde_json::to_string(&payload).unwrap().as_bytes(),
        )
        .unwrap();
        file
    }

    #[test]
    fn config_accepts_variable_and_empty_query_lists() {
        for queries in [
            serde_json::json!([]),
            serde_json::json!(["find cache invalidation"]),
        ] {
            let file = config_file(serde_json::json!({
                "schema_version": 1,
                "query_manifest_sha256": "abc",
                "k": 20,
                "queries": queries,
            }));
            assert_eq!(load_config(file.path()).unwrap().k, 20);
        }
    }

    #[test]
    fn config_rejects_wrong_k_duplicate_and_overlong_queries() {
        let wrong_k = config_file(serde_json::json!({
            "schema_version": 1,
            "query_manifest_sha256": "abc",
            "k": 10,
            "queries": [],
        }));
        assert!(
            load_config(wrong_k.path())
                .unwrap_err()
                .to_string()
                .contains("exactly 20")
        );

        let duplicates = config_file(serde_json::json!({
            "schema_version": 1,
            "query_manifest_sha256": "abc",
            "k": 20,
            "queries": ["Find auth flow", "find auth flow"],
        }));
        assert!(
            load_config(duplicates.path())
                .unwrap_err()
                .to_string()
                .contains("unique")
        );

        let too_many = config_file(serde_json::json!({
            "schema_version": 1,
            "query_manifest_sha256": "abc",
            "k": 20,
            "queries": ["one", "two", "three", "four"],
        }));
        assert!(
            load_config(too_many.path())
                .unwrap_err()
                .to_string()
                .contains("at most 3")
        );
    }

    fn two_query_config() -> CimConfig {
        CimConfig {
            query_manifest_sha256: "abc".to_string(),
            k: 20,
            queries: vec![
                "find auth flow".to_string(),
                "locate token refresh".to_string(),
            ],
        }
    }

    fn workspaces(names: &[&str]) -> Vec<crate::session::AnalysisWorkspace> {
        names
            .iter()
            .map(|name| crate::session::AnalysisWorkspace {
                name: (*name).to_string(),
                path: PathBuf::from("/repos").join(name),
            })
            .collect()
    }

    fn arguments(call: &PrefixToolCall) -> serde_json::Value {
        serde_json::from_str(&call.arguments).expect("arguments are JSON")
    }

    /// A single-root session has no name to send, so the call carries no
    /// `workspace` argument and keeps the original id.
    #[test]
    fn synthetic_step_without_named_workspaces_keeps_one_workspaceless_call() {
        let step = synthetic_step(&two_query_config(), None);
        assert_eq!(step.tool_calls.len(), 1);
        assert_eq!(step.tool_calls[0].id, "cim-step-0-semantic-search");
        assert_eq!(step.tool_calls[0].name, "semantic_search");
        assert_eq!(
            arguments(&step.tool_calls[0]),
            serde_json::json!({
                "queries": ["find auth flow", "locate token refresh"],
                "k": 20
            })
        );
        // An empty configured list is the single-root shape too: bifrost was
        // started with `--root`, so there is still no name to send.
        assert_eq!(
            synthetic_step(&two_query_config(), Some(&[])).tool_calls,
            step.tool_calls
        );
    }

    /// The bug this step was failing on. Every CodeScale eval starts bifrost
    /// with `--workspace NAME=PATH`, where the router rejects a call that names
    /// no workspace, so step zero asks each configured workspace the same
    /// queries under its own id.
    #[test]
    fn synthetic_step_names_every_configured_workspace() {
        let config = two_query_config();
        let step = synthetic_step(&config, Some(&workspaces(&["backend", "frontend"])));

        let ids: Vec<&str> = step
            .tool_calls
            .iter()
            .map(|call| call.id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec![
                "cim-step-0-semantic-search-backend",
                "cim-step-0-semantic-search-frontend"
            ]
        );
        assert_eq!(
            step.tool_calls
                .iter()
                .map(arguments)
                .collect::<Vec<serde_json::Value>>(),
            vec![
                serde_json::json!({
                    "queries": ["find auth flow", "locate token refresh"],
                    "k": 20,
                    "workspace": "backend"
                }),
                serde_json::json!({
                    "queries": ["find auth flow", "locate token refresh"],
                    "k": 20,
                    "workspace": "frontend"
                }),
            ]
        );
        assert!(
            step.tool_calls
                .iter()
                .all(|call| call.name == "semantic_search")
        );
    }

    #[test]
    fn synthetic_step_with_no_queries_has_no_tool_call() {
        let config = CimConfig {
            query_manifest_sha256: "abc".to_string(),
            k: 20,
            queries: Vec::new(),
        };
        assert!(synthetic_step(&config, None).tool_calls.is_empty());
        assert!(
            synthetic_step(&config, Some(&workspaces(&["backend", "frontend"])))
                .tool_calls
                .is_empty(),
            "no queries means no calls, however many workspaces there are"
        );
    }

    #[test]
    fn synthetic_step_claim_is_shared_across_fresh_sessions() {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("cim-config.json");
        std::fs::write(&config, "{}").unwrap();

        assert!(claim_synthetic_step_at(&config).unwrap());
        assert!(!claim_synthetic_step_at(&config).unwrap());
    }

    #[test]
    fn cim_mcp_timeout_covers_the_cell_deadline() {
        assert_eq!(CIM_MCP_TOOL_CALL_TIMEOUT, Duration::from_secs(1_800));
    }
}
