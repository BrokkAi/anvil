//! On-disk credential store for AWS Bedrock.
//!
//! Bedrock authentication uses a bearer token (AWS SSO token or similar)
//! plus an optional region and default model override. Unlike Codex,
//! Bedrock has no OAuth flow -- the user pastes a static bearer token
//! once and we reuse it forever (until they rotate or disconnect).
//! Persistence is opt-in: users who export `AWS_BEARER_TOKEN_BEDROCK`
//! in their shell get the existing zero-config behaviour, and this file
//! is only created when `/setup bedrock key <token>` is invoked from a
//! session.
//!
//! Storage location follows OS conventions via `dirs::config_dir()`:
//! `~/.config/brokk/bedrock.json` on Linux (or `$XDG_CONFIG_HOME`),
//! `~/Library/Application Support/brokk/bedrock.json` on macOS,
//! `%APPDATA%\brokk\bedrock.json` on Windows. The file is written
//! atomically (stage `.tmp` then rename) and chmod'd to 0600 on Unix so
//! other local users can't read the credentials.
//!
//! Legacy fallback: `~/.secrets/bedrock_api_key` and
//! `~/.secrets/aws_bearer_token_bedrock` are still read as a fallback
//! for users who configured Bedrock before this file existed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::bedrock_client::{BEDROCK_API_KEY_ENV, BEDROCK_DEFAULT_MODEL, BEDROCK_DEFAULT_REGION};

/// Persisted Bedrock credentials. The bearer_token is the only required
/// field; region and default_model are optional overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockAuth {
    pub bearer_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
}

/// Snapshot of where Bedrock credentials currently come from.
/// Single source of truth for the "env owns" contract: whenever
/// `AWS_BEARER_TOKEN_BEDROCK` is non-empty the environment owns the
/// credential lifecycle, `/setup bedrock key` is hidden from
/// suggestions, and the slash command returns an explanation rather
/// than mutating state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CredentialState {
    pub env_set: bool,
    pub file_present: bool,
    /// A token exists in the legacy `~/.secrets/` fallback the backend
    /// still resolves. Tracked separately from `file_present` because the
    /// managed setup commands (`key`/`region`/`model`/`disconnect`) only
    /// touch `brokk/bedrock.json`, never the secrets files.
    pub secrets_present: bool,
}

impl CredentialState {
    /// Read the current env+file+secrets state. Cheap: a single env
    /// lookup and (when needed) a couple of small disk reads; safe to
    /// call from `available_commands_update` paths and per-request
    /// handlers. Must enumerate every source
    /// `bedrock_client::bearer_token_from_env_or_secrets` resolves, or
    /// detection reports "not configured" for a working backend.
    pub fn snapshot() -> Self {
        let env_set = std::env::var(BEDROCK_API_KEY_ENV)
            .ok()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        let file_present = match read() {
            Ok(Some(auth)) => !auth.bearer_token.trim().is_empty(),
            _ => false,
        };
        let secrets_present = matches!(
            crate::bedrock_client::bearer_token_from_secrets(),
            Ok(Some(_))
        );
        Self {
            env_set,
            file_present,
            secrets_present,
        }
    }

    /// Where the active credential, if any, is being read from.
    /// Mirrors the precedence in `bearer_token_from_env_or_secrets`: env
    /// wins over the managed file, the file wins over the legacy
    /// `~/.secrets/` fallback, and that wins over nothing.
    pub fn active_source(&self) -> &'static str {
        if self.env_set {
            "env"
        } else if self.file_present {
            "file"
        } else if self.secrets_present {
            "secrets"
        } else {
            "none"
        }
    }

    /// True when the environment owns the credential lifecycle.
    pub fn env_owns(&self) -> bool {
        self.env_set
    }
}

/// Resolve `<config>/brokk/bedrock.json`. Honours `$BROKK_CONFIG_HOME`
/// if set so tests (and power users) can redirect the credential file.
pub fn auth_path() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("BROKK_CONFIG_HOME") {
        return Ok(PathBuf::from(custom).join("bedrock.json"));
    }
    let base = dirs::config_dir()
        .ok_or_else(|| anyhow!("could not resolve OS config directory for Bedrock credentials"))?;
    Ok(base.join("brokk").join("bedrock.json"))
}

pub fn read() -> Result<Option<BedrockAuth>> {
    let path = auth_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed = serde_json::from_slice::<BedrockAuth>(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some(parsed))
}

/// Atomic write: stage to `bedrock.json.tmp` in the same directory,
/// chmod to 0600, then rename. Mirrors `openrouter_auth::write` so a
/// crash mid-write never leaves a half-written credential file.
pub fn write(auth: &BedrockAuth) -> Result<()> {
    let path = auth_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("tmp.{}", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(auth).context("serializing BedrockAuth")?;
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    set_user_only_perms(&tmp)?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Best-effort logout: delete the stored credentials. Missing file is
/// not an error -- `/setup bedrock disconnect` is idempotent.
pub fn logout() -> Result<()> {
    let path = auth_path()?;
    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_user_only_perms(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("chmod 600 {}", path.display()))
}

#[cfg(not(unix))]
fn set_user_only_perms(_path: &Path) -> Result<()> {
    Ok(())
}

/// Read the region from all configured sources.
/// Precedence: AWS_REGION > AWS_DEFAULT_REGION > BEDROCK_REGION > brokk config > default.
pub fn region_from_any_source() -> String {
    // AWS standard vars
    for var in ["AWS_REGION", "AWS_DEFAULT_REGION"] {
        if let Ok(val) = std::env::var(var) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    // Bedrock-specific env
    if let Ok(val) = std::env::var(crate::bedrock_client::BEDROCK_REGION_ENV) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    // Brokk config
    if let Ok(Some(auth)) = read()
        && let Some(region) = auth.region
    {
        let trimmed = region.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    BEDROCK_DEFAULT_REGION.to_string()
}

/// Read the default model from all configured sources.
/// Precedence: ANVIL_BEDROCK_MODEL env > brokk config > hardcoded default.
pub fn model_from_any_source() -> String {
    if let Ok(val) = std::env::var(crate::bedrock_client::BEDROCK_MODEL_ENV) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Ok(Some(auth)) = read()
        && let Some(model) = auth.default_model
    {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    BEDROCK_DEFAULT_MODEL.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_VARS: &[&str] = &[
        "BROKK_CONFIG_HOME",
        "BROKK_SECRETS_HOME",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "BEDROCK_REGION",
        "ANVIL_BEDROCK_MODEL",
    ];

    struct EnvScope {
        restore: Vec<(&'static str, Option<String>)>,
    }

    impl EnvScope {
        fn new(config_home: &std::path::Path) -> Self {
            let mut env = Self {
                restore: Vec::new(),
            };
            for var in TEST_VARS {
                let prev = std::env::var(var).ok();
                env.restore.push((var, prev));
            }
            // SAFETY: single-threaded test, no concurrent env access.
            unsafe {
                for var in TEST_VARS {
                    std::env::remove_var(var);
                }
                std::env::set_var("BROKK_CONFIG_HOME", config_home);
                // Redirect the legacy secrets dir into the temp config
                // home too, so detection never reads the developer's real
                // ~/.secrets/. It holds no secret-named files unless a
                // test writes one, so `secrets_present` stays false by
                // default.
                std::env::set_var("BROKK_SECRETS_HOME", config_home);
            }
            env
        }

        fn set_env(&mut self, var: &'static str, value: &str) {
            // SAFETY: single-threaded test, no concurrent env access.
            unsafe {
                std::env::set_var(var, value);
            }
        }
    }

    impl Drop for EnvScope {
        fn drop(&mut self) {
            // SAFETY: single-threaded test, no concurrent env access.
            unsafe {
                for (var, prev) in self.restore.iter() {
                    match prev {
                        Some(v) => std::env::set_var(var, v),
                        None => std::env::remove_var(var),
                    }
                }
            }
        }
    }

    #[test]
    fn round_trip_writes_then_reads_same_data() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        assert!(read().unwrap().is_none(), "no auth before write");
        write(&BedrockAuth {
            bearer_token: "test-token-123".to_string(),
            region: Some("eu-west-1".to_string()),
            default_model: Some("us.anthropic.claude-opus-4-8".to_string()),
        })
        .unwrap();
        let got = read().unwrap().expect("auth present after write");
        assert_eq!(got.bearer_token, "test-token-123");
        assert_eq!(got.region.as_deref(), Some("eu-west-1"));
        assert_eq!(
            got.default_model.as_deref(),
            Some("us.anthropic.claude-opus-4-8")
        );
    }

    #[test]
    fn logout_removes_file_and_is_idempotent() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        write(&BedrockAuth {
            bearer_token: "test-token".to_string(),
            region: None,
            default_model: None,
        })
        .unwrap();
        assert!(auth_path().unwrap().exists());
        logout().unwrap();
        assert!(!auth_path().unwrap().exists());
        logout().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn write_sets_user_only_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        write(&BedrockAuth {
            bearer_token: "test-token".to_string(),
            region: None,
            default_model: None,
        })
        .unwrap();
        let perms = std::fs::metadata(auth_path().unwrap())
            .unwrap()
            .permissions();
        assert_eq!(
            perms.mode() & 0o777,
            0o600,
            "credential file must be readable only by the owner"
        );
    }

    #[test]
    fn credential_state_reports_env_when_env_set() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let mut env = EnvScope::new(tmp.path());
        env.set_env("AWS_BEARER_TOKEN_BEDROCK", "token-from-env");

        let state = CredentialState::snapshot();
        assert!(state.env_set);
        assert!(state.env_owns());
        assert_eq!(state.active_source(), "env");
    }

    #[test]
    fn credential_state_reports_file_when_only_file_set() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        write(&BedrockAuth {
            bearer_token: "token-from-file".to_string(),
            region: None,
            default_model: None,
        })
        .unwrap();

        let state = CredentialState::snapshot();
        assert!(!state.env_set);
        assert!(state.file_present);
        assert!(!state.secrets_present);
        assert!(!state.env_owns());
        assert_eq!(state.active_source(), "file");
    }

    #[test]
    fn credential_state_reports_none_when_nothing_set() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        let state = CredentialState::snapshot();
        assert!(!state.env_set);
        assert!(!state.file_present);
        assert!(!state.secrets_present);
        assert!(!state.env_owns());
        assert_eq!(state.active_source(), "none");
    }

    #[test]
    fn credential_state_reports_secrets_when_only_secrets_file_set() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());
        // Legacy fallback: a token in ~/.secrets/ (redirected to tmp via
        // BROKK_SECRETS_HOME) with no env var and no managed config file.
        // This is the regression: the backend resolves this token (models
        // load) but detection used to report "none".
        std::fs::write(tmp.path().join("bedrock_api_key"), "secret-token\n").unwrap();

        let state = CredentialState::snapshot();
        assert!(!state.env_set);
        assert!(!state.file_present);
        assert!(state.secrets_present);
        assert!(!state.env_owns());
        assert_eq!(state.active_source(), "secrets");
    }

    #[test]
    fn credential_state_file_wins_over_secrets() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());
        std::fs::write(tmp.path().join("bedrock_api_key"), "secret-token\n").unwrap();
        write(&BedrockAuth {
            bearer_token: "token-from-file".to_string(),
            region: None,
            default_model: None,
        })
        .unwrap();

        let state = CredentialState::snapshot();
        assert!(state.file_present);
        assert!(state.secrets_present);
        assert_eq!(state.active_source(), "file");
    }

    #[test]
    fn bearer_token_precedence_env_wins_over_file() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let mut env = EnvScope::new(tmp.path());
        env.set_env("AWS_BEARER_TOKEN_BEDROCK", "env-token");

        write(&BedrockAuth {
            bearer_token: "file-token".to_string(),
            region: None,
            default_model: None,
        })
        .unwrap();

        let token = crate::bedrock_client::bearer_token_from_env_or_secrets()
            .unwrap()
            .expect("token found");
        assert_eq!(token, "env-token");
    }

    #[test]
    fn bearer_token_falls_back_to_file_when_env_unset() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        write(&BedrockAuth {
            bearer_token: "file-token".to_string(),
            region: None,
            default_model: None,
        })
        .unwrap();

        let token = crate::bedrock_client::bearer_token_from_env_or_secrets()
            .unwrap()
            .expect("token found");
        assert_eq!(token, "file-token");
    }

    #[test]
    fn region_precedence_aws_region_wins() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let mut env = EnvScope::new(tmp.path());
        env.set_env("AWS_REGION", "us-west-2");
        env.set_env("BEDROCK_REGION", "eu-central-1");

        write(&BedrockAuth {
            bearer_token: "t".to_string(),
            region: Some("ap-southeast-1".to_string()),
            default_model: None,
        })
        .unwrap();

        assert_eq!(region_from_any_source(), "us-west-2");
    }

    #[test]
    fn region_falls_back_to_brokk_config() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        write(&BedrockAuth {
            bearer_token: "t".to_string(),
            region: Some("ap-southeast-1".to_string()),
            default_model: None,
        })
        .unwrap();

        assert_eq!(region_from_any_source(), "ap-southeast-1");
    }

    #[test]
    fn region_defaults_to_us_east_1() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        assert_eq!(region_from_any_source(), "us-east-1");
    }

    #[test]
    fn model_precedence_env_wins() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let mut env = EnvScope::new(tmp.path());
        env.set_env("ANVIL_BEDROCK_MODEL", "env-model");

        write(&BedrockAuth {
            bearer_token: "t".to_string(),
            region: None,
            default_model: Some("file-model".to_string()),
        })
        .unwrap();

        assert_eq!(model_from_any_source(), "env-model");
    }

    #[test]
    fn model_falls_back_to_brokk_config() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        write(&BedrockAuth {
            bearer_token: "t".to_string(),
            region: None,
            default_model: Some("file-model".to_string()),
        })
        .unwrap();

        assert_eq!(model_from_any_source(), "file-model");
    }

    #[test]
    fn model_defaults_to_claude_sonnet() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _env = EnvScope::new(tmp.path());

        assert!(model_from_any_source().contains("claude-sonnet"));
    }
}
