//! Tiny persisted state for first-run setup nudges and per-install preferences.
//!
//! This is intentionally not the source of truth for whether models work.
//! Model readiness is re-derived from the live session/catalog every time.
//! The file only records whether the user has already seen the first-run
//! setup screen and the last selected model/reasoning effort/sandbox mode so
//! configured installs get a short hint instead of the full welcome on every
//! new session. It also stores user-configured MCP servers; when that field is
//! absent, Anvil seeds the config with its preinstalled servers.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SetupState {
    #[serde(default)]
    pub first_run_seen: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<Vec<crate::mcp::McpServerConfig>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_defaults_version: Option<u32>,
}

static WRITE_LOCK: Mutex<()> = Mutex::new(());
const CURRENT_MCP_DEFAULTS_VERSION: u32 = 2;

#[cfg(test)]
thread_local! {
    static TEST_CONFIG_HOME: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct TestConfigHomeScope {
    prev: Option<PathBuf>,
}

#[cfg(test)]
impl TestConfigHomeScope {
    pub(crate) fn set(path: PathBuf) -> Self {
        let prev = TEST_CONFIG_HOME.with(|slot| slot.borrow().clone());
        TEST_CONFIG_HOME.with(|slot| *slot.borrow_mut() = Some(path));
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for TestConfigHomeScope {
    fn drop(&mut self) {
        TEST_CONFIG_HOME.with(|slot| *slot.borrow_mut() = self.prev.take());
    }
}

pub fn path() -> Result<PathBuf> {
    #[cfg(test)]
    {
        if let Some(custom) = TEST_CONFIG_HOME.with(|slot| slot.borrow().clone()) {
            Ok(custom.join("setup.json"))
        } else {
            Err(anyhow::anyhow!(
                "test setup state path is unset; use TestConfigHomeScope"
            ))
        }
    }
    #[cfg(not(test))]
    {
        if let Ok(custom) = std::env::var("BROKK_CONFIG_HOME")
            && !custom.trim().is_empty()
        {
            Ok(PathBuf::from(custom).join("setup.json"))
        } else {
            let base = dirs::config_dir().ok_or_else(|| {
                anyhow::anyhow!("could not resolve OS config directory for setup state")
            })?;
            Ok(base.join("brokk").join("setup.json"))
        }
    }
}

pub fn read() -> SetupState {
    read_inner()
}

pub fn read_sandbox_mode_preference() -> Option<Option<crate::sandbox_backend::SandboxMode>> {
    let path = path().ok()?;
    let bytes = std::fs::read(&path).ok()?;
    serde_json::from_slice::<SetupState>(&bytes)
        .ok()
        .map(|state| state.last_sandbox_mode)
}

pub fn mark_first_run_seen() -> Result<()> {
    update(|state| state.first_run_seen = true)
}

pub fn remember_last_reasoning_effort(reasoning_effort: Option<String>) -> Result<()> {
    update(|state| state.last_reasoning_effort = reasoning_effort)
}

pub fn remember_sandbox_mode(mode: Option<crate::sandbox_backend::SandboxMode>) -> Result<()> {
    update(|state| state.last_sandbox_mode = mode)
}

pub fn read_mcp_servers() -> Vec<crate::mcp::McpServerConfig> {
    #[cfg(test)]
    if path().is_err() {
        return Vec::new();
    }
    let mut state = read();
    let (servers, changed) = migrate_mcp_servers(&mut state);
    if changed && write_inner(&state).is_err() {
        tracing::warn!("failed to persist migrated MCP setup state");
    }
    servers
}

fn migrate_mcp_servers(state: &mut SetupState) -> (Vec<crate::mcp::McpServerConfig>, bool) {
    let mut servers = state
        .mcp_servers
        .clone()
        .unwrap_or_else(crate::mcp::default_servers);
    let bifrost = crate::mcp::McpServerConfig::bifrost_core();
    let bifrost_code_quality = crate::mcp::McpServerConfig::bifrost_code_quality();
    let mut changed = normalize_builtin_mcp_servers(
        &mut servers,
        &[bifrost.clone(), bifrost_code_quality.clone()],
    );

    let legacy_defaults_need_upgrade = state.mcp_servers.is_some()
        && state.mcp_defaults_version.unwrap_or(0) < CURRENT_MCP_DEFAULTS_VERSION;
    if legacy_defaults_need_upgrade
        && servers
            .iter()
            .any(|server| matches_builtin_server(server, &bifrost))
        && !servers
            .iter()
            .any(|server| server.name == bifrost_code_quality.name)
    {
        servers.push(bifrost_code_quality);
        changed = true;
    }

    if state.mcp_servers.is_some()
        && (changed || state.mcp_defaults_version != Some(CURRENT_MCP_DEFAULTS_VERSION))
    {
        state.mcp_servers = Some(servers.clone());
        state.mcp_defaults_version = Some(CURRENT_MCP_DEFAULTS_VERSION);
        changed = true;
    }

    (servers, changed)
}

fn normalize_builtin_mcp_servers(
    servers: &mut [crate::mcp::McpServerConfig],
    builtins: &[crate::mcp::McpServerConfig],
) -> bool {
    let mut changed = false;
    for server in servers {
        for builtin in builtins {
            if matches_builtin_server(server, builtin)
                && server.framing == crate::mcp::McpFraming::ContentLength
            {
                server.framing = builtin.framing;
                changed = true;
            }
        }
    }
    changed
}

fn matches_builtin_server(
    server: &crate::mcp::McpServerConfig,
    builtin: &crate::mcp::McpServerConfig,
) -> bool {
    server.name == builtin.name && server.command == builtin.command && server.args == builtin.args
}

pub fn remember_mcp_servers(servers: Vec<crate::mcp::McpServerConfig>) -> Result<()> {
    update(|state| {
        state.mcp_servers = Some(servers);
        state.mcp_defaults_version = Some(CURRENT_MCP_DEFAULTS_VERSION);
    })
}

pub fn remember_last_selection(
    model: Option<String>,
    reasoning_effort: Option<String>,
) -> Result<()> {
    update(|state| {
        state.last_model = model;
        state.last_reasoning_effort = reasoning_effort;
    })
}

fn update(mutator: impl FnOnce(&mut SetupState)) -> Result<()> {
    let _guard = WRITE_LOCK.lock().expect("setup state write mutex poisoned");
    let mut state = read_inner();
    mutator(&mut state);
    write_inner(&state)
}

fn read_inner() -> SetupState {
    let Ok(path) = path() else {
        return SetupState::default();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return SetupState::default();
    };
    serde_json::from_slice::<SetupState>(&bytes).unwrap_or_default()
}

fn write_inner(state: &SetupState) -> Result<()> {
    let path = match path() {
        Ok(path) => path,
        Err(_e) => {
            #[cfg(test)]
            {
                return Ok(());
            }
            #[cfg(not(test))]
            {
                return Err(_e);
            }
        }
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating setup state dir {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("setup.json");
    let tmp = path.with_file_name(format!(".{file_name}.tmp.{}", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(state).context("serializing setup state")?;
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn custom_server() -> crate::mcp::McpServerConfig {
        crate::mcp::McpServerConfig {
            name: "custom-search".to_string(),
            command: "custom-search".to_string(),
            args: vec!["--serve".to_string()],
            framing: crate::mcp::McpFraming::ContentLength,
            enabled: true,
        }
    }

    #[test]
    fn read_mcp_servers_defaults_to_core_and_code_quality_bifrost() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());

        assert_eq!(read_mcp_servers(), crate::mcp::default_servers());
    }

    #[test]
    fn read_mcp_servers_migrates_legacy_single_bifrost_default() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());

        write_inner(&SetupState {
            mcp_servers: Some(vec![crate::mcp::McpServerConfig {
                framing: crate::mcp::McpFraming::ContentLength,
                ..crate::mcp::McpServerConfig::bifrost_core()
            }]),
            ..SetupState::default()
        })
        .expect("write setup state");

        assert_eq!(
            read_mcp_servers(),
            vec![
                crate::mcp::McpServerConfig::bifrost_core(),
                crate::mcp::McpServerConfig::bifrost_code_quality(),
            ]
        );
    }

    #[test]
    fn read_mcp_servers_migrates_legacy_core_plus_custom_config() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());

        write_inner(&SetupState {
            mcp_servers: Some(vec![
                crate::mcp::McpServerConfig {
                    framing: crate::mcp::McpFraming::ContentLength,
                    ..crate::mcp::McpServerConfig::bifrost_core()
                },
                custom_server(),
            ]),
            ..SetupState::default()
        })
        .expect("write setup state");

        assert_eq!(
            read_mcp_servers(),
            vec![
                crate::mcp::McpServerConfig::bifrost_core(),
                custom_server(),
                crate::mcp::McpServerConfig::bifrost_code_quality(),
            ]
        );
    }

    #[test]
    fn migrated_code_quality_server_can_be_removed_persistently() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());

        write_inner(&SetupState {
            mcp_servers: Some(vec![crate::mcp::McpServerConfig::bifrost_core()]),
            ..SetupState::default()
        })
        .expect("write setup state");

        let migrated = read_mcp_servers();
        assert_eq!(
            migrated,
            vec![
                crate::mcp::McpServerConfig::bifrost_core(),
                crate::mcp::McpServerConfig::bifrost_code_quality(),
            ]
        );

        remember_mcp_servers(vec![crate::mcp::McpServerConfig::bifrost_core()])
            .expect("persist explicit removal");

        assert_eq!(
            read_mcp_servers(),
            vec![crate::mcp::McpServerConfig::bifrost_core()]
        );
    }
}
