//! Credential access for hosted DeepSeek.
//!
//! DeepSeek keys are static (no refresh, no expiry): the user pastes one
//! once via `/setup deepseek key <key>` and we reuse it until they rotate
//! or disconnect. Persistence is opt-in: users who export
//! `DEEPSEEK_API_KEY` in their shell get the zero-config behaviour, and
//! on-disk state is only created by the setup command.
//!
//! Storage lives in the consolidated [`crate::secrets`] store
//! (`<config>/brokk/secrets.json`, 0600, atomic), like OpenRouter and
//! Bedrock. Unlike those two there is no legacy per-provider file to fall
//! back to or migrate -- DeepSeek predates no store but the env var.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::discovery::DEEPSEEK_API_KEY_ENV;

/// Flat one-field record. Like OpenRouter, DeepSeek keys are static so
/// there's nothing more to persist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekAuth {
    pub api_key: String,
}

/// Snapshot of where DeepSeek credentials currently come from. Single
/// source of truth for the "env owns" contract, mirroring the OpenRouter
/// and Bedrock `CredentialState` types: whenever `DEEPSEEK_API_KEY` is
/// non-empty the environment owns the credential lifecycle and `/setup
/// deepseek key` explains rather than mutating state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CredentialState {
    pub env_set: bool,
    pub file_present: bool,
}

impl CredentialState {
    pub fn snapshot() -> Self {
        let env_set = std::env::var(DEEPSEEK_API_KEY_ENV)
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

/// Read the stored DeepSeek credentials from the consolidated store. A
/// malformed store degrades to "no stored key" with a warning (matching
/// the sibling providers' resilience posture); the env var keeps working
/// regardless.
pub fn read() -> Result<Option<DeepSeekAuth>> {
    match crate::secrets::read() {
        Ok(Some(secrets)) => Ok(secrets.deepseek),
        Ok(None) => Ok(None),
        Err(e) => {
            tracing::warn!("secrets store unreadable; treating DeepSeek as not stored: {e:#}");
            Ok(None)
        }
    }
}

/// Persist the key into the consolidated secrets store.
pub fn write(auth: &DeepSeekAuth) -> Result<()> {
    crate::secrets::update(|secrets| secrets.deepseek = Some(auth.clone()))
}

/// Best-effort logout: clear the stored key. Missing state is not an
/// error -- `/setup deepseek disconnect` is idempotent and, mirroring the
/// sibling providers, must not create a store when nothing was ever
/// saved. A malformed store is quarantined by `update` so disconnect
/// self-heals broken state instead of reporting failure.
pub fn logout() -> Result<()> {
    match crate::secrets::read() {
        Ok(Some(secrets)) if secrets.deepseek.is_some() => {
            crate::secrets::update(|secrets| secrets.deepseek = None)
        }
        Ok(_) => Ok(()),
        Err(_) => crate::secrets::update(|secrets| secrets.deepseek = None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};

    #[test]
    fn credential_state_reports_sources() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        {
            let _env = EnvScope::set("DEEPSEEK_API_KEY", "sk-ds-env");
            let state = CredentialState::snapshot();
            assert!(state.env_set && state.env_owns());
            assert_eq!(state.active_source(), "env");
        }

        let _env = EnvScope::remove("DEEPSEEK_API_KEY");
        let state = CredentialState::snapshot();
        assert_eq!(state.active_source(), "none");

        write(&DeepSeekAuth {
            api_key: "sk-ds-file".into(),
        })
        .unwrap();
        let state = CredentialState::snapshot();
        assert!(state.file_present && !state.env_owns());
        assert_eq!(state.active_source(), "file");
    }

    #[test]
    fn logout_is_idempotent_and_never_creates_a_store() {
        let _lock = ENV_GUARD.blocking_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp.path());

        logout().unwrap();
        assert!(
            !crate::secrets::secrets_path().unwrap().exists(),
            "disconnect with nothing saved must not create secrets.json"
        );

        write(&DeepSeekAuth {
            api_key: "sk-ds".into(),
        })
        .unwrap();
        logout().unwrap();
        assert!(read().unwrap().is_none());
        logout().unwrap();
    }
}
