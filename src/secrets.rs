//! Consolidated on-disk store for provider setup secrets.
//!
//! Historically every provider grew its own credential file --
//! `openrouter.json`, `bedrock.json`, and DeepSeek had no file at all
//! (env var only). This module gives them one home:
//!
//! - `~/.config/brokk/secrets.json` on Linux (or `$XDG_CONFIG_HOME`),
//! - `~/Library/Application Support/brokk/secrets.json` on macOS,
//! - `%APPDATA%\brokk\secrets.json` on Windows,
//! - `$BROKK_CONFIG_HOME/secrets.json` when that override is set.
//!
//! The file is written atomically (stage `.tmp` then rename) and chmod'd
//! to 0600 on Unix so other local users can't read the keys, matching the
//! per-provider files it replaces.
//!
//! Division of responsibility:
//! - Env vars (`DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`,
//!   `AWS_BEARER_TOKEN_BEDROCK`) always win at the call sites; this file
//!   is the persistent fallback written by the `/setup` flows.
//! - Codex stays in `~/.codex/auth.json`: that file is owned by the
//!   third-party Codex CLI integration and must remain compatible with it.
//! - The legacy `~/.secrets/` Bedrock token files are still read as a
//!   fallback by `bedrock_client` but are never migrated or deleted here:
//!   that directory is shared with other tools, so Anvil only removes
//!   those files on an explicit `/setup bedrock disconnect`.
//!
//! [`migrate_legacy_files`] folds the Anvil-owned per-provider files into
//! `secrets.json` once at startup (copy, then delete only after the
//! consolidated file is safely on disk). The provider modules keep a
//! read-only fallback to their legacy file so a failed or skipped
//! migration never locks a user out.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::bedrock_auth::BedrockAuth;
use crate::openrouter_auth::OpenRouterAuth;

/// Flat one-field record for hosted DeepSeek. Like OpenRouter, DeepSeek
/// keys are static (no refresh, no expiry) so there's nothing more to
/// persist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekAuth {
    pub api_key: String,
}

/// The consolidated secrets file: one optional section per provider.
/// Sections are omitted from the JSON entirely when absent so the file
/// stays readable and diff-friendly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SetupSecrets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deepseek: Option<DeepSeekAuth>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openrouter: Option<OpenRouterAuth>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bedrock: Option<BedrockAuth>,
}

/// Resolve `<config>/brokk/secrets.json`. Honours `$BROKK_CONFIG_HOME`
/// if set so tests (and power users) can redirect the credential file
/// without touching the real one.
pub fn secrets_path() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("BROKK_CONFIG_HOME") {
        return Ok(PathBuf::from(custom).join("secrets.json"));
    }
    let base = dirs::config_dir()
        .ok_or_else(|| anyhow!("could not resolve OS config directory for setup secrets"))?;
    Ok(base.join("brokk").join("secrets.json"))
}

pub fn read() -> Result<Option<SetupSecrets>> {
    let path = secrets_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed = serde_json::from_slice::<SetupSecrets>(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some(parsed))
}

/// Atomic write: stage to a unique `.tmp` in the same directory, chmod to
/// 0600, then rename. Mirrors the per-provider files this store replaces
/// so a crash mid-write never leaves a half-written credential file.
pub fn write(secrets: &SetupSecrets) -> Result<()> {
    let path = secrets_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("tmp.{}", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(secrets).context("serializing SetupSecrets")?;
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    set_user_only_perms(&tmp)?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Read-modify-write helper for the provider modules: load the current
/// secrets (or an empty default), apply `mutate`, and persist.
pub fn update(mutate: impl FnOnce(&mut SetupSecrets)) -> Result<()> {
    let mut secrets = read()?.unwrap_or_default();
    mutate(&mut secrets);
    write(&secrets)
}

/// One-time consolidation of the Anvil-owned legacy per-provider
/// credential files (`openrouter.json`, `bedrock.json`) into
/// `secrets.json`, called once at startup.
///
/// Copy-then-delete: a legacy file is only removed after the consolidated
/// file has been written successfully, so a crash or write failure can
/// never lose a credential. A section already present in `secrets.json`
/// wins over legacy content (never clobber newer state); the stale legacy
/// file is still deleted in that case since its contents are shadowed. A
/// malformed legacy file is left in place with a warning -- the
/// per-provider read fallback keeps ignoring it exactly as before.
///
/// Best-effort by design: this runs before any backend is built, and a
/// failure must degrade to the pre-consolidation behaviour (per-provider
/// files keep working through the read fallbacks) rather than block
/// startup.
pub fn migrate_legacy_files() {
    let mut secrets = match read() {
        Ok(secrets) => secrets.unwrap_or_default(),
        Err(e) => {
            tracing::warn!("skipping secrets migration: cannot read secrets.json: {e:#}");
            return;
        }
    };

    let mut migrated_paths: Vec<PathBuf> = Vec::new();
    let mut changed = false;

    match crate::openrouter_auth::read_legacy_file() {
        Ok(Some((path, auth))) => {
            if secrets.openrouter.is_none() {
                secrets.openrouter = Some(auth);
                changed = true;
            }
            migrated_paths.push(path);
        }
        Ok(None) => {}
        Err(e) => tracing::warn!("leaving legacy openrouter.json in place: {e:#}"),
    }
    match crate::bedrock_auth::read_legacy_file() {
        Ok(Some((path, auth))) => {
            if secrets.bedrock.is_none() {
                secrets.bedrock = Some(auth);
                changed = true;
            }
            migrated_paths.push(path);
        }
        Ok(None) => {}
        Err(e) => tracing::warn!("leaving legacy bedrock.json in place: {e:#}"),
    }

    if changed {
        if let Err(e) = write(&secrets) {
            tracing::warn!("secrets migration failed to write secrets.json: {e:#}");
            return;
        }
        tracing::info!(
            "migrated legacy provider credential files into {}",
            secrets_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "secrets.json".to_string())
        );
    }
    for path in migrated_paths {
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!("removed legacy credential file {}", path.display()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                "failed to remove legacy credential file {}: {e}",
                path.display()
            ),
        }
    }
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

/// Snapshot of where DeepSeek credentials currently come from. Single
/// source of truth for the "env owns" contract, mirroring the OpenRouter
/// and Bedrock `CredentialState` types: whenever `DEEPSEEK_API_KEY` is
/// non-empty the environment owns the credential lifecycle and `/setup
/// deepseek key` explains rather than mutating state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekCredentialState {
    pub env_set: bool,
    pub file_present: bool,
}

impl DeepSeekCredentialState {
    pub fn snapshot() -> Self {
        let env_set = std::env::var(crate::discovery::DEEPSEEK_API_KEY_ENV)
            .ok()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        let file_present = match read() {
            Ok(Some(secrets)) => secrets
                .deepseek
                .is_some_and(|auth| !auth.api_key.trim().is_empty()),
            _ => false,
        };
        Self {
            env_set,
            file_present,
        }
    }

    /// Where the active credential, if any, is being read from. Mirrors
    /// the precedence in `build_deepseek_backend`: env wins over file,
    /// file wins over nothing.
    pub fn active_source(&self) -> &'static str {
        if self.env_set {
            "env"
        } else if self.file_present {
            "file"
        } else {
            "none"
        }
    }

    /// True when the environment owns the credential lifecycle.
    pub fn env_owns(&self) -> bool {
        self.env_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};

    #[test]
    fn round_trip_writes_then_reads_all_sections() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        assert!(read().unwrap().is_none(), "no secrets before write");
        write(&SetupSecrets {
            deepseek: Some(DeepSeekAuth {
                api_key: "sk-ds".into(),
            }),
            openrouter: Some(OpenRouterAuth {
                api_key: "sk-or".into(),
            }),
            bedrock: Some(BedrockAuth {
                bearer_token: "aws-token".into(),
                region: Some("us-east-1".into()),
                default_model: None,
            }),
        })
        .unwrap();

        let got = read().unwrap().expect("secrets present after write");
        assert_eq!(got.deepseek.unwrap().api_key, "sk-ds");
        assert_eq!(got.openrouter.unwrap().api_key, "sk-or");
        let bedrock = got.bedrock.unwrap();
        assert_eq!(bedrock.bearer_token, "aws-token");
        assert_eq!(bedrock.region.as_deref(), Some("us-east-1"));
    }

    #[cfg(unix)]
    #[test]
    fn write_sets_user_only_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        write(&SetupSecrets::default()).unwrap();
        let perms = std::fs::metadata(secrets_path().unwrap())
            .unwrap()
            .permissions();
        assert_eq!(
            perms.mode() & 0o777,
            0o600,
            "secrets file must be readable only by the owner"
        );
    }

    #[test]
    fn update_preserves_unrelated_sections() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        update(|s| {
            s.openrouter = Some(OpenRouterAuth {
                api_key: "sk-or".into(),
            })
        })
        .unwrap();
        update(|s| {
            s.deepseek = Some(DeepSeekAuth {
                api_key: "sk-ds".into(),
            })
        })
        .unwrap();

        let got = read().unwrap().expect("secrets present");
        assert_eq!(
            got.openrouter.expect("openrouter survives").api_key,
            "sk-or"
        );
        assert_eq!(got.deepseek.expect("deepseek written").api_key, "sk-ds");
    }

    #[test]
    fn migrate_legacy_files_consolidates_and_removes_legacy() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        // Seed both legacy files the way the pre-consolidation code wrote them.
        let openrouter_legacy = tmp.path().join("openrouter.json");
        std::fs::write(&openrouter_legacy, r#"{"api_key":"sk-or-legacy"}"#).unwrap();
        let bedrock_legacy = tmp.path().join("bedrock.json");
        std::fs::write(
            &bedrock_legacy,
            r#"{"bearer_token":"aws-legacy","region":"eu-west-1"}"#,
        )
        .unwrap();

        migrate_legacy_files();

        let secrets = read().unwrap().expect("secrets.json created");
        assert_eq!(secrets.openrouter.unwrap().api_key, "sk-or-legacy");
        let bedrock = secrets.bedrock.unwrap();
        assert_eq!(bedrock.bearer_token, "aws-legacy");
        assert_eq!(bedrock.region.as_deref(), Some("eu-west-1"));
        assert!(
            !openrouter_legacy.exists(),
            "legacy openrouter.json removed after successful migration"
        );
        assert!(
            !bedrock_legacy.exists(),
            "legacy bedrock.json removed after successful migration"
        );
    }

    #[test]
    fn migrate_legacy_files_never_clobbers_existing_sections() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        write(&SetupSecrets {
            openrouter: Some(OpenRouterAuth {
                api_key: "sk-or-current".into(),
            }),
            ..SetupSecrets::default()
        })
        .unwrap();
        let openrouter_legacy = tmp.path().join("openrouter.json");
        std::fs::write(&openrouter_legacy, r#"{"api_key":"sk-or-stale"}"#).unwrap();

        migrate_legacy_files();

        let secrets = read().unwrap().expect("secrets present");
        assert_eq!(
            secrets.openrouter.unwrap().api_key,
            "sk-or-current",
            "existing section wins over legacy content"
        );
        assert!(
            !openrouter_legacy.exists(),
            "shadowed legacy file still removed"
        );
    }

    #[test]
    fn migrate_legacy_files_leaves_malformed_legacy_in_place() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        let openrouter_legacy = tmp.path().join("openrouter.json");
        std::fs::write(&openrouter_legacy, "not json").unwrap();

        migrate_legacy_files();

        assert!(
            openrouter_legacy.exists(),
            "malformed legacy file must not be deleted"
        );
        assert!(
            read().unwrap().is_none(),
            "nothing to migrate, secrets.json not created"
        );
    }

    #[test]
    fn deepseek_credential_state_reports_sources() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        {
            let _env = EnvScope::set("DEEPSEEK_API_KEY", "sk-ds-env");
            let state = DeepSeekCredentialState::snapshot();
            assert!(state.env_set && state.env_owns());
            assert_eq!(state.active_source(), "env");
        }

        let _env = EnvScope::remove("DEEPSEEK_API_KEY");
        let state = DeepSeekCredentialState::snapshot();
        assert_eq!(state.active_source(), "none");

        update(|s| {
            s.deepseek = Some(DeepSeekAuth {
                api_key: "sk-ds-file".into(),
            })
        })
        .unwrap();
        let state = DeepSeekCredentialState::snapshot();
        assert!(state.file_present && !state.env_owns());
        assert_eq!(state.active_source(), "file");
    }
}
