//! CIM evaluation-only synthetic semantic-search configuration.

use std::collections::HashSet;
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

pub(crate) fn enabled() -> bool {
    crate::p2t::env_var_truthy(CIM_EVAL_ENV)
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
    let path = PathBuf::from(
        std::env::var(CIM_CONFIG_ENV)
            .with_context(|| format!("{CIM_CONFIG_ENV} must be set when {CIM_EVAL_ENV}=1"))?,
    );
    if !path.is_absolute() {
        bail!("{CIM_CONFIG_ENV} must be an absolute path");
    }
    load_config(&path).map(Some)
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

pub(crate) fn synthetic_step(config: &CimConfig) -> ForcedStep {
    ForcedStep {
        assistant_text: String::new(),
        tool_calls: if config.queries.is_empty() {
            Vec::new()
        } else {
            vec![PrefixToolCall {
                id: "cim-step-0-semantic-search".to_string(),
                name: "semantic_search".to_string(),
                arguments: serde_json::json!({ "queries": config.queries, "k": config.k })
                    .to_string(),
            }]
        },
        message: None,
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

    #[test]
    fn synthetic_step_uses_stable_ids_and_k_twenty() {
        let step = synthetic_step(&CimConfig {
            query_manifest_sha256: "abc".to_string(),
            k: 20,
            queries: vec![
                "find auth flow".to_string(),
                "locate token refresh".to_string(),
            ],
        });
        assert_eq!(step.tool_calls.len(), 1);
        assert_eq!(step.tool_calls[0].id, "cim-step-0-semantic-search");
        assert_eq!(step.tool_calls[0].name, "semantic_search");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&step.tool_calls[0].arguments).unwrap(),
            serde_json::json!({
                "queries": ["find auth flow", "locate token refresh"],
                "k": 20
            })
        );
    }

    #[test]
    fn synthetic_step_with_no_queries_has_no_tool_call() {
        let step = synthetic_step(&CimConfig {
            query_manifest_sha256: "abc".to_string(),
            k: 20,
            queries: Vec::new(),
        });
        assert!(step.tool_calls.is_empty());
    }

    #[test]
    fn cim_mcp_timeout_covers_the_cell_deadline() {
        assert_eq!(CIM_MCP_TOOL_CALL_TIMEOUT, Duration::from_secs(1_800));
    }
}
