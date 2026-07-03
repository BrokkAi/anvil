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
//!
//! The migration is ONE-WAY: once the legacy files are folded in and
//! removed, downgrading to an Anvil release that predates `secrets.json`
//! shows every provider as "not connected" until the user re-runs the
//! login/setup commands (env vars keep working). A malformed
//! `secrets.json` never locks anything: reads fall back to the legacy
//! files, and the first setup command that needs to write quarantines the
//! corrupt file aside (`secrets.json.corrupt-<id>`) so its contents stay
//! recoverable by hand.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::bedrock_auth::BedrockAuth;
use crate::deepseek_auth::DeepSeekAuth;
use crate::openrouter_auth::OpenRouterAuth;

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

/// Serializes every read-modify-write of `secrets.json` within this
/// process. Consolidating the per-provider files into one document made
/// concurrent setup commands (separate sessions, or setup racing the
/// startup migration) able to silently drop each other's sections via
/// stale reads; the per-provider files could never interfere across
/// providers. Cross-process writers remain unsynchronized -- writes only
/// happen on interactive setup commands, and the atomic rename keeps the
/// file itself intact either way.
static STORE_LOCK: Mutex<()> = Mutex::new(());

fn store_guard() -> std::sync::MutexGuard<'static, ()> {
    // The lock only brackets small fs reads/writes; a poisoned guard means
    // a previous writer panicked mid-update, which the atomic rename makes
    // safe to continue past.
    STORE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Atomic write: stage to a unique `.tmp` in the same directory (created
/// 0600 on Unix *before* any secret bytes land on disk), then rename.
/// A crash mid-write never leaves a half-written or loosely-permissioned
/// credential file.
pub fn write(secrets: &SetupSecrets) -> Result<()> {
    let path = secrets_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("tmp.{}", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(secrets).context("serializing SetupSecrets")?;
    write_user_only(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Read-modify-write helper for the provider modules: load the current
/// secrets (or an empty default), apply `mutate`, and persist -- all under
/// [`STORE_LOCK`]. A malformed store is quarantined rather than parsed
/// over: the corrupt file moves aside so the write can proceed on a fresh
/// document without destroying possibly-recoverable key material.
pub fn update(mutate: impl FnOnce(&mut SetupSecrets)) -> Result<()> {
    let _guard = store_guard();
    let mut secrets = match read() {
        Ok(secrets) => secrets.unwrap_or_default(),
        Err(read_err) => {
            quarantine_corrupt_store(read_err)?;
            SetupSecrets::default()
        }
    };
    mutate(&mut secrets);
    write(&secrets)
}

/// Move a malformed `secrets.json` aside so a fresh one can be written.
/// Renaming (not deleting) preserves whatever key material the corrupt
/// file still holds for manual recovery. Errors if the rename fails --
/// writing over the corrupt file would destroy that material.
fn quarantine_corrupt_store(read_err: anyhow::Error) -> Result<()> {
    let path = secrets_path()?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("secrets.json");
    let quarantine = path.with_file_name(format!("{file_name}.corrupt-{}", uuid::Uuid::new_v4()));
    std::fs::rename(&path, &quarantine).with_context(|| {
        format!(
            "quarantining malformed {} (original error: {read_err:#})",
            path.display()
        )
    })?;
    tracing::warn!(
        "malformed secrets store moved aside to {}; starting a fresh one ({read_err:#})",
        quarantine.display()
    );
    Ok(())
}

/// Read and parse a legacy per-provider credential file, returning its
/// path alongside the record so the migration can delete the file once
/// its contents are safely consolidated. Shared by the provider modules'
/// `read_legacy_file` wrappers.
pub(crate) fn read_legacy_json<T: serde::de::DeserializeOwned>(
    path: PathBuf,
) -> Result<Option<(PathBuf, T)>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed = serde_json::from_slice::<T>(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some((path, parsed)))
}

/// Best-effort removal of a superseded legacy credential file. Missing is
/// fine; other failures only warn -- a leftover legacy file is shadowed by
/// the consolidated store, never read back in preference to it.
pub(crate) fn remove_legacy_credential_file(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => tracing::info!("removed legacy credential file {}", path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(
            "failed to remove legacy credential file {}: {e}",
            path.display()
        ),
    }
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
    let _guard = store_guard();
    let mut secrets = match read() {
        Ok(secrets) => secrets.unwrap_or_default(),
        Err(e) => {
            // Do not quarantine here: startup is a passive moment and the
            // provider read fallbacks keep working. The first explicit
            // setup write quarantines instead.
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
            } else {
                tracing::info!(
                    "legacy {} is shadowed by the consolidated store; discarding its contents",
                    path.display()
                );
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
            } else {
                tracing::info!(
                    "legacy {} is shadowed by the consolidated store; discarding its contents",
                    path.display()
                );
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
            "migrated legacy provider credential files into {}; note this is one-way -- \
             a downgraded Anvil will need its login/setup commands re-run",
            secrets_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "secrets.json".to_string())
        );
    }
    for path in migrated_paths {
        remove_legacy_credential_file(&path);
    }
}

/// Create `path` and write `bytes` such that the content never touches
/// disk with looser-than-owner-only permissions: on Unix the file is
/// created 0600 *before* any bytes are written (umask can only tighten
/// that), rather than written first and chmod'd after.
#[cfg(unix)]
fn write_user_only(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)
}

#[cfg(not(unix))]
fn write_user_only(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
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
    fn update_quarantines_a_malformed_store_instead_of_failing() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        std::fs::write(secrets_path().unwrap(), "{not json").unwrap();

        update(|s| {
            s.deepseek = Some(DeepSeekAuth {
                api_key: "sk-ds".into(),
            })
        })
        .expect("update must recover from a malformed store");

        let got = read().unwrap().expect("fresh store written");
        assert_eq!(got.deepseek.unwrap().api_key, "sk-ds");

        // The corrupt content is preserved aside, not destroyed.
        let quarantined: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("secrets.json.corrupt-")
            })
            .collect();
        assert_eq!(quarantined.len(), 1, "corrupt store must be moved aside");
        assert_eq!(
            std::fs::read_to_string(quarantined[0].path()).unwrap(),
            "{not json"
        );
    }

    #[cfg(unix)]
    #[test]
    fn migrate_legacy_files_preserves_legacy_when_write_fails() {
        use std::os::unix::fs::PermissionsExt;
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        let openrouter_legacy = tmp.path().join("openrouter.json");
        std::fs::write(&openrouter_legacy, r#"{"api_key":"sk-or-legacy"}"#).unwrap();

        // Make the config dir read-only so the consolidated write fails.
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
        migrate_legacy_files();
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(
            openrouter_legacy.exists(),
            "a failed consolidated write must never delete the legacy credential"
        );
        assert_eq!(
            crate::openrouter_auth::read()
                .unwrap()
                .expect("legacy still readable")
                .api_key,
            "sk-or-legacy"
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
}
