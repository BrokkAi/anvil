//! Tiny persisted state for first-run setup nudges.
//!
//! This is intentionally not the source of truth for whether models work.
//! Model readiness is re-derived from the live session/catalog every time.
//! The file only records whether the user has already seen the first-run
//! setup screen so configured installs get a short hint instead of the full
//! welcome on every new session.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SetupState {
    #[serde(default)]
    pub first_run_seen: bool,
}

pub fn path() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("BROKK_CONFIG_HOME")
        && !custom.trim().is_empty()
    {
        return Ok(PathBuf::from(custom).join("setup.json"));
    }
    let base = dirs::config_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve OS config directory for setup state"))?;
    Ok(base.join("brokk").join("setup.json"))
}

pub fn read() -> SetupState {
    let Ok(path) = path() else {
        return SetupState::default();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return SetupState::default();
    };
    serde_json::from_slice::<SetupState>(&bytes).unwrap_or_default()
}

pub fn mark_first_run_seen() -> Result<()> {
    let mut state = read();
    state.first_run_seen = true;
    write(&state)
}

fn write(state: &SetupState) -> Result<()> {
    let path = path()?;
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
