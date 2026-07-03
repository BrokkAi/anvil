//! Credential access for OpenRouter.
//!
//! Unlike Codex, OpenRouter has no OAuth flow -- the user pastes a static
//! `sk-or-...` key once and we reuse it forever (until they rotate or
//! disconnect). Persistence is opt-in: users who export
//! `OPENROUTER_API_KEY` in their shell get the existing zero-config
//! behaviour, and on-disk state is only created when `/openrouter-login
//! <key>` is invoked from a session.
//!
//! Storage lives in the consolidated [`crate::secrets`] store
//! (`<config>/brokk/secrets.json`, 0600, atomic). The pre-consolidation
//! per-provider file (`<config>/brokk/openrouter.json`) is still read as
//! a fallback and is folded into the consolidated store by
//! [`crate::secrets::migrate_legacy_files`] at startup.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::discovery::OPENROUTER_API_KEY_ENV;

/// Flat one-field record. OpenRouter keys are static (no refresh, no
/// expiry, no auth_mode) so there's nothing more to persist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterAuth {
    pub api_key: String,
}

/// Snapshot of where OpenRouter credentials currently come from.
/// Single source of truth for the "env owns" contract: whenever
/// `OPENROUTER_API_KEY` is non-empty the environment owns the
/// credential lifecycle, `/openrouter-login` is hidden from
/// autocomplete, and the slash command returns an explanation rather
/// than mutating state.
///
/// Both reads (`env_set`, `file_present`) treat any failure as
/// "absent" so callers can render a consistent UI even when the file
/// is malformed or the env var is unset -- diagnostic output should
/// never panic, and a broken on-disk file is functionally equivalent
/// to no file at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CredentialState {
    pub env_set: bool,
    pub file_present: bool,
}

impl CredentialState {
    /// Read the current env+file state. Cheap: a single env lookup and
    /// (when needed) a small disk read; safe to call from
    /// `available_commands_update` paths and per-request handlers.
    pub fn snapshot() -> Self {
        let env_set = std::env::var(OPENROUTER_API_KEY_ENV)
            .ok()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        let file_present = match read() {
            Ok(Some(auth)) => !auth.api_key.trim().is_empty(),
            _ => false,
        };
        Self {
            env_set,
            file_present,
        }
    }

    /// Where the active credential, if any, is being read from.
    /// Mirrors the precedence in `build_openrouter_backend`: env wins
    /// over file, file wins over nothing.
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
    /// Callers should hide `/openrouter-login` from autocomplete and
    /// have the handler explain rather than mutate state.
    pub fn env_owns(&self) -> bool {
        self.env_set
    }
}

/// Resolve the legacy `<config>/brokk/openrouter.json` path. Honours
/// `$BROKK_CONFIG_HOME` if set so tests (and power users) can redirect the
/// credential files without touching the real ones. New writes go to the
/// consolidated store; this path survives as the read/migration fallback
/// and as the anchor for `refresh_log_path`.
pub fn auth_path() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("BROKK_CONFIG_HOME") {
        return Ok(PathBuf::from(custom).join("openrouter.json"));
    }
    let base = dirs::config_dir().ok_or_else(|| {
        anyhow!("could not resolve OS config directory for OpenRouter credentials")
    })?;
    Ok(base.join("brokk").join("openrouter.json"))
}

/// Resolve a best-effort debug log path beside `openrouter.json` so
/// refresh instrumentation can be inspected even when transcript
/// notifications are dropped by the client.
pub fn refresh_log_path() -> Result<PathBuf> {
    let auth = auth_path()?;
    let parent = auth
        .parent()
        .ok_or_else(|| anyhow!("openrouter auth path has no parent directory"))?;
    Ok(parent.join("openrouter-refresh.log"))
}

/// Append one line to the refresh trace file. Best effort: callers use
/// this for debugging, so failures are intentionally swallowed after a
/// warning rather than disrupting the main flow.
pub fn append_refresh_log(line: &str) {
    if let Err(e) = append_refresh_log_inner(line) {
        tracing::warn!("failed to append OpenRouter refresh log: {e:#}");
    }
}

fn append_refresh_log_inner(line: &str) -> Result<()> {
    use std::io::Write as _;

    let path = refresh_log_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    writeln!(file, "{line}").with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Read the stored OpenRouter credentials: the consolidated secrets store
/// first, then the legacy per-provider file (pre-migration installs, or a
/// migration that could not complete).
pub fn read() -> Result<Option<OpenRouterAuth>> {
    if let Some(secrets) = crate::secrets::read()?
        && let Some(auth) = secrets.openrouter
    {
        return Ok(Some(auth));
    }
    Ok(read_legacy_file()?.map(|(_, auth)| auth))
}

/// Read the legacy `openrouter.json`, returning its path alongside the
/// parsed record so `secrets::migrate_legacy_files` can delete the file
/// once its contents are safely consolidated.
pub(crate) fn read_legacy_file() -> Result<Option<(PathBuf, OpenRouterAuth)>> {
    let path = auth_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed = serde_json::from_slice::<OpenRouterAuth>(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some((path, parsed)))
}

/// Persist the key into the consolidated secrets store, then drop the
/// superseded legacy file (best-effort: a leftover legacy file is merely
/// shadowed, never read back in preference to the store).
pub fn write(auth: &OpenRouterAuth) -> Result<()> {
    crate::secrets::update(|secrets| secrets.openrouter = Some(auth.clone()))?;
    remove_legacy_file();
    Ok(())
}

/// Best-effort logout: delete the stored credentials. Missing state is
/// not an error -- `/openrouter-login disconnect` is idempotent.
pub fn logout() -> Result<()> {
    if crate::secrets::read()?.is_some_and(|secrets| secrets.openrouter.is_some()) {
        crate::secrets::update(|secrets| secrets.openrouter = None)?;
    }
    let path = auth_path()?;
    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
    }
    Ok(())
}

fn remove_legacy_file() {
    let Ok(path) = auth_path() else {
        return;
    };
    match std::fs::remove_file(&path) {
        Ok(()) => tracing::info!("removed legacy credential file {}", path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(
            "failed to remove legacy credential file {}: {e}",
            path.display()
        ),
    }
}

/// Test-only helpers shared with sibling modules so any test that
/// mutates `OPENROUTER_API_KEY` or `BROKK_CONFIG_HOME` serialises on a
/// single process-wide mutex. Env mutation in multi-threaded Rust is
/// `unsafe` (POSIX `getenv` is not atomic), so one guard for all
/// env-touching tests is the minimum-friction safe pattern.
#[cfg(test)]
pub(crate) mod test_support {
    use std::ffi::OsStr;
    use tokio::sync::Mutex;

    /// Acquire this before mutating any env var read by either
    /// `openrouter_auth` or any caller of `CredentialState::snapshot`.
    /// Holding it during the whole test (until `EnvScope` drops) keeps
    /// concurrent tests from observing partial state.
    ///
    /// Uses `tokio::sync::Mutex` instead of `std::sync::Mutex` so the
    /// guard can be held across `.await` points in `#[tokio::test]`
    /// cases without tripping `clippy::await_holding_lock`. Sync
    /// `#[test]` cases acquire it via `blocking_lock()`; async cases
    /// use `.lock().await`. The mutex is constructed via `const_new`
    /// so it fits in a plain `static`.
    pub(crate) static ENV_GUARD: Mutex<()> = Mutex::const_new(());

    /// RAII guard that sets (or removes) an env var on construction and
    /// restores the previous value on drop. Pair with a held lock on
    /// `ENV_GUARD` for cross-test safety.
    pub(crate) struct EnvScope {
        var: &'static str,
        prev: Option<String>,
    }

    impl EnvScope {
        pub(crate) fn set(var: &'static str, value: impl AsRef<OsStr>) -> Self {
            let prev = std::env::var(var).ok();
            // SAFETY: callers hold `ENV_GUARD` so no concurrent thread
            // is reading or writing this process's env table.
            unsafe {
                std::env::set_var(var, value);
            }
            Self { var, prev }
        }

        pub(crate) fn remove(var: &'static str) -> Self {
            let prev = std::env::var(var).ok();
            // SAFETY: see `set`.
            unsafe {
                std::env::remove_var(var);
            }
            Self { var, prev }
        }
    }

    impl Drop for EnvScope {
        fn drop(&mut self) {
            // SAFETY: see `EnvScope::set`.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(self.var, v),
                    None => std::env::remove_var(self.var),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};

    #[test]
    fn round_trip_writes_then_reads_same_key() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        assert!(read().unwrap().is_none(), "no key before write");
        write(&OpenRouterAuth {
            api_key: "sk-or-test-key".to_string(),
        })
        .unwrap();
        let got = read().unwrap().expect("key present after write");
        assert_eq!(got.api_key, "sk-or-test-key");
    }

    #[test]
    fn logout_clears_stored_key_and_is_idempotent() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        write(&OpenRouterAuth {
            api_key: "sk-or-test".to_string(),
        })
        .unwrap();
        assert!(read().unwrap().is_some());
        logout().unwrap();
        assert!(read().unwrap().is_none());
        // second call must not error
        logout().unwrap();
    }

    #[test]
    fn write_goes_to_consolidated_store_and_supersedes_legacy() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        // Simulate a pre-consolidation install.
        std::fs::write(auth_path().unwrap(), r#"{"api_key":"sk-or-legacy"}"#).unwrap();
        assert_eq!(
            read().unwrap().expect("legacy fallback readable").api_key,
            "sk-or-legacy"
        );

        write(&OpenRouterAuth {
            api_key: "sk-or-new".to_string(),
        })
        .unwrap();
        assert!(
            !auth_path().unwrap().exists(),
            "legacy file removed once superseded"
        );
        assert!(crate::secrets::secrets_path().unwrap().exists());
        assert_eq!(read().unwrap().expect("key present").api_key, "sk-or-new");
    }

    #[test]
    fn logout_also_removes_legacy_file() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _scope = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        std::fs::write(auth_path().unwrap(), r#"{"api_key":"sk-or-legacy"}"#).unwrap();
        logout().unwrap();
        assert!(!auth_path().unwrap().exists());
        assert!(read().unwrap().is_none());
    }

    #[test]
    fn credential_state_reports_env_when_env_set() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());
        let _env = EnvScope::set("OPENROUTER_API_KEY", "sk-or-from-env");

        let state = CredentialState::snapshot();
        assert!(state.env_set);
        assert!(!state.file_present);
        assert!(state.env_owns());
        assert_eq!(state.active_source(), "env");
    }

    #[test]
    fn credential_state_reports_file_when_only_file_set() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());
        let _env = EnvScope::remove("OPENROUTER_API_KEY");
        write(&OpenRouterAuth {
            api_key: "sk-or-from-file".to_string(),
        })
        .unwrap();

        let state = CredentialState::snapshot();
        assert!(!state.env_set);
        assert!(state.file_present);
        assert!(!state.env_owns());
        assert_eq!(state.active_source(), "file");
    }

    #[test]
    fn credential_state_reports_none_when_nothing_set() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());
        let _env = EnvScope::remove("OPENROUTER_API_KEY");

        let state = CredentialState::snapshot();
        assert!(!state.env_set);
        assert!(!state.file_present);
        assert!(!state.env_owns());
        assert_eq!(state.active_source(), "none");
    }

    #[test]
    fn credential_state_treats_blank_env_as_unset() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());
        let _env = EnvScope::set("OPENROUTER_API_KEY", "   ");

        let state = CredentialState::snapshot();
        // Trim-empty env var must NOT take ownership: matches the
        // startup parser in `build_openrouter_backend`, which falls
        // through to the file when the env is whitespace-only.
        assert!(!state.env_set);
        assert!(!state.env_owns());
    }
}
