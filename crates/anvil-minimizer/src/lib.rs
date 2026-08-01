//! Shell-output minimizer, vendored from oh-my-pi `crates/pi-shell` at commit
//! `09a7c865636457c50ed75fc3b1a7cc21ef72c105`.
//!
//! Do not hand-edit files under `src/minimizer*` -- re-vendor from upstream
//! instead, then run `cargo fmt -p anvil-minimizer` (upstream formats with
//! nightly rustfmt options; this repo reformats the copy with stable rustfmt,
//! so diff against upstream with `--ignore-all-space`). See `NOTICE` for RTK
//! (rtk-ai/rtk) attribution.

// Anvil only exercises the whole-command path; upstream helpers for the
// segmented-chain path (e.g. `chain_output`) are intentionally unused here,
// and CI's `-D warnings` must not fail on vendored code.
#![allow(dead_code)]

pub mod minimizer;

pub use minimizer::{
    MinimizerConfig, MinimizerCtx, MinimizerOptions, MinimizerOutput, apply, engine,
};
