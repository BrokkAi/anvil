//! Tiny persisted state for first-run setup nudges and per-install preferences.
//!
//! This is intentionally not the source of truth for whether models work.
//! Model readiness is re-derived from the live session/catalog every time.
//! The file only records whether the user has already seen the first-run
//! setup screen and the last selected model/reasoning effort so configured
//! installs get a short hint instead of the full welcome on every new session.

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

pub fn mark_first_run_seen() -> Result<()> {
    update(|state| state.first_run_seen = true)
}

pub fn remember_last_reasoning_effort(reasoning_effort: Option<String>) -> Result<()> {
    update(|state| state.last_reasoning_effort = reasoning_effort)
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
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(state).context("serializing setup state")?;
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}
