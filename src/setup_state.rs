//! Tiny persisted state for first-run setup nudges and per-install preferences.
//!
//! This is intentionally not the source of truth for whether models work.
//! Model readiness is re-derived from the live session/catalog every time.
//! The file only records whether the user has already seen the first-run
//! setup screen and the last selected sandbox plus the `/setup recap`
//! preference so configured installs get a short hint instead of the full
//! welcome on every new session. ACP session config options such as model,
//! reasoning effort, behavior mode, and permission mode are intentionally not
//! stored here; clients must send them for each session. It also stores
//! user-configured MCP servers; when that field is absent, Anvil seeds the
//! config with its preinstalled servers.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SetupState {
    #[serde(default)]
    pub first_run_seen: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_lenient_optional"
    )]
    pub last_sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_recap_enabled: Option<bool>,
    /// Legacy install-wide approvals from older builds. Current builds use
    /// repo-local `.brokk/permissions.json` instead, but we still deserialize
    /// this field for backward compatibility.
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "alwaysAllow")]
    pub always_allow: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<Vec<crate::mcp::McpServerConfig>>,
}

/// Deserialize an optional enum field leniently: a value this build does not
/// recognize degrades to `None` instead of failing the whole `SetupState`
/// deserialization. Without this, `read_inner`'s `unwrap_or_default()` would
/// reset every preference (model, sandbox, MCP servers, ...) to default on
/// encountering a single unknown enum value -- e.g. after a downgrade that
/// no longer knows a newer mode. Absent fields are handled by
/// `#[serde(default)]` and never reach this function.
fn deserialize_lenient_optional<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| serde_json::from_value(value).ok()))
}

static WRITE_LOCK: Mutex<()> = Mutex::new(());

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

pub(crate) fn config_home() -> Result<PathBuf> {
    #[cfg(test)]
    {
        if let Some(custom) = TEST_CONFIG_HOME.with(|slot| slot.borrow().clone()) {
            Ok(custom)
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
            Ok(PathBuf::from(custom))
        } else {
            let base = dirs::config_dir().ok_or_else(|| {
                anyhow::anyhow!("could not resolve OS config directory for setup state")
            })?;
            Ok(base.join("brokk"))
        }
    }
}

pub fn path() -> Result<PathBuf> {
    Ok(config_home()?.join("setup.json"))
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

pub fn remember_sandbox_mode(mode: Option<crate::sandbox_backend::SandboxMode>) -> Result<()> {
    update(|state| state.last_sandbox_mode = mode)
}

pub fn remember_turn_recap_enabled(enabled: bool) -> Result<()> {
    update(|state| state.turn_recap_enabled = Some(enabled))
}

pub fn read_mcp_servers() -> Vec<crate::mcp::McpServerConfig> {
    #[cfg(test)]
    if path().is_err() {
        return Vec::new();
    }
    let mut servers = read()
        .mcp_servers
        .unwrap_or_else(crate::mcp::default_servers);
    for server in &mut servers {
        crate::mcp::normalize_preinstalled_bifrost_server(server);
    }
    servers
}

pub fn remember_mcp_servers(servers: Vec<crate::mcp::McpServerConfig>) -> Result<()> {
    update(|state| state.mcp_servers = Some(servers))
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

    /// An unrecognized value for one enum preference must degrade to `None`
    /// without discarding the rest of the setup state. Before the lenient
    /// deserializer, a single unknown enum value failed the whole-struct parse,
    /// and `read_inner`'s `unwrap_or_default()` then wiped sibling setup
    /// preferences and the user's MCP server list.
    #[test]
    fn unknown_enum_value_degrades_to_none_and_preserves_siblings() {
        use crate::sandbox_backend::SandboxMode;

        let state = SetupState {
            first_run_seen: true,
            last_sandbox_mode: Some(SandboxMode::Wasm),
            mcp_servers: Some(vec![crate::mcp::McpServerConfig {
                name: "bifrost".to_string(),
                command: "bifrost".to_string(),
                args: Vec::new(),
                env: Vec::new(),
                framing: crate::mcp::McpFraming::Line,
                enabled: true,
            }]),
            ..SetupState::default()
        };

        let mut json = serde_json::to_value(&state).expect("serialize setup state");
        json["last_sandbox_mode"] = serde_json::Value::String("quantum".to_string());
        let parsed: SetupState =
            serde_json::from_value(json).expect("unknown enum must not fail the whole struct");
        assert!(parsed.first_run_seen);
        assert_eq!(parsed.last_sandbox_mode, None);
        assert_eq!(parsed.mcp_servers.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn legacy_session_config_keys_are_ignored_and_dropped_on_write() {
        use crate::sandbox_backend::SandboxMode;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        std::fs::create_dir_all(config_dir.path()).expect("create config dir");
        std::fs::write(
            config_dir.path().join("setup.json"),
            serde_json::json!({
                "last_model": "model-b",
                "last_reasoning_effort": "high",
                "last_behavior_mode": "PLAN",
                "last_permission_mode": "readOnly",
                "last_sandbox_mode": "wasm"
            })
            .to_string(),
        )
        .expect("write legacy setup state");

        assert_eq!(read().last_sandbox_mode, Some(SandboxMode::Wasm));
        remember_turn_recap_enabled(true).expect("write setup state");

        let rewritten: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path().expect("setup path")).expect("read setup"),
        )
        .expect("setup json");
        for key in [
            "last_model",
            "last_reasoning_effort",
            "last_behavior_mode",
            "last_permission_mode",
        ] {
            assert!(
                rewritten.get(key).is_none(),
                "legacy session config key {key} must be dropped"
            );
        }
        assert_eq!(rewritten["last_sandbox_mode"], "wasm");
        assert_eq!(rewritten["turn_recap_enabled"], true);
    }

    #[test]
    fn read_mcp_servers_migrates_default_bifrost_command_to_managed_binary() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        remember_mcp_servers(vec![crate::mcp::McpServerConfig {
            name: "bifrost".to_string(),
            command: "bifrost".to_string(),
            args: crate::mcp::McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: crate::mcp::McpFraming::ContentLength,
            enabled: true,
        }])
        .expect("remember mcp servers");

        let servers = read_mcp_servers();
        let bifrost = servers
            .into_iter()
            .find(|server| server.name == "bifrost")
            .expect("bifrost server");

        assert_eq!(
            bifrost.command,
            crate::mcp::McpServerConfig::bifrost().command
        );
        assert!(
            bifrost
                .command
                .contains(crate::mcp::BUNDLED_BIFROST_VERSION),
            "expected managed bifrost command to contain pinned version '{}', got: '{}'",
            crate::mcp::BUNDLED_BIFROST_VERSION,
            bifrost.command,
        );
        assert_eq!(bifrost.framing, crate::mcp::McpFraming::Line);
    }

    #[test]
    fn read_mcp_servers_preserves_custom_bifrost_command() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        remember_mcp_servers(vec![crate::mcp::McpServerConfig {
            name: "bifrost".to_string(),
            command: "/tmp/custom-bifrost".to_string(),
            args: crate::mcp::McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: crate::mcp::McpFraming::Line,
            enabled: true,
        }])
        .expect("remember mcp servers");

        let servers = read_mcp_servers();
        let bifrost = servers
            .into_iter()
            .find(|server| server.name == "bifrost")
            .expect("bifrost server");

        assert_eq!(bifrost.command, "/tmp/custom-bifrost");
    }
}
