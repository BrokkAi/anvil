//! When the default `wasm-sandbox` feature is enabled, compiles
//! `brokk-acp-sandbox` to `wasm32-wasip2` and exposes the resulting
//! artifact path via the `BROKK_ACP_SANDBOX_WASM` env var so
//! `src/sandbox_backend.rs` can `include_bytes!` it.
//!
//! Why a build script rather than a checked-in `.wasm`:
//!   - The wasm artifact must stay in lockstep with the library half of
//!     `brokk-acp-sandbox`. Checking in a `.wasm` invites drift between
//!     the native fallback (linked code) and the sandbox path (loaded
//!     bytes), where a parser fix lands in one but not the other.
//!   - It keeps the repository binary-free, so every install rebuilds
//!     the sandbox artifact from the resolved crate version.
//!
//! Why we invoke `cargo build` recursively instead of letting cargo
//! resolve the wasm target on its own:
//!   - The host crate (`brokk-acp-rust`) targets the build machine's
//!     architecture, not wasm. Cross-targeting from one cargo invocation
//!     requires per-target dependency tables and would push us into
//!     resolver hell, especially with `wasmtime` (host-only) and
//!     `brokk-acp-sandbox` (both host and wasm). Two invocations keep
//!     the dependency graphs cleanly separated.
//!
//! Failure modes:
//!   - `rustup target add wasm32-wasip2` not run on this host: the build
//!     script fails with a clear message asking the user to install it.
//!   - The sandbox sub-crate fails to compile (e.g. a non-portable dep
//!     was added): the error propagates with the offending stderr from
//!     the nested cargo invocation.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use serde::Deserialize;

const SANDBOX_CRATE: &str = "brokk-acp-sandbox";
const WASM_TARGET: &str = "wasm32-wasip2";

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_WASM_SANDBOX");

    if env::var_os("CARGO_FEATURE_WASM_SANDBOX").is_none() {
        return;
    }

    let sandbox_manifest = find_dependency_manifest(SANDBOX_CRATE);
    if let Some(parent) = sandbox_manifest.parent() {
        println!("cargo:rerun-if-changed={}", parent.join("src").display());
    }

    // Build to a dedicated target dir under OUT_DIR so the registry source
    // remains read-only and the artifact cannot collide with the host build.
    let target_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("sandbox-target");

    let status = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
        .args([
            "build",
            "--release",
            "--bin",
            SANDBOX_CRATE,
            "--target",
            WASM_TARGET,
            "--manifest-path",
        ])
        .arg(&sandbox_manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        // Clear cargo env vars inherited from the outer build so the
        // nested invocation picks its own target/profile.
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTFLAGS")
        .status()
        .expect("invoke `cargo build` for brokk-acp-sandbox wasm target");

    if !status.success() {
        eprintln!(
            "cargo:warning=building {SANDBOX_CRATE} for {WASM_TARGET} failed. \
             Make sure `rustup target add {WASM_TARGET}` has been run on this host."
        );
        std::process::exit(1);
    }

    let wasm_path = target_dir
        .join(WASM_TARGET)
        .join("release")
        .join(format!("{SANDBOX_CRATE}.wasm"));

    if !wasm_path.exists() {
        eprintln!(
            "cargo:warning=expected wasm artifact at {} but it was not produced",
            wasm_path.display()
        );
        std::process::exit(1);
    }

    println!(
        "cargo:rustc-env=BROKK_ACP_SANDBOX_WASM={}",
        wasm_path.display()
    );
}

fn find_dependency_manifest(name: &str) -> PathBuf {
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1"])
        // Clear cargo env vars inherited from the outer build so metadata
        // describes the host build graph, not the nested wasm build.
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTFLAGS")
        .output()
        .expect("invoke `cargo metadata`");

    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        panic!("cargo metadata failed");
    }

    let metadata: CargoMetadata =
        serde_json::from_slice(&output.stdout).expect("parse cargo metadata JSON");
    let package = metadata
        .packages
        .into_iter()
        .find(|package| package.name == name && package.source.is_some())
        .unwrap_or_else(|| panic!("could not find resolved dependency package `{name}`"));

    PathBuf::from(package.manifest_path)
}

#[derive(Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
}

#[derive(Deserialize)]
struct CargoPackage {
    name: String,
    manifest_path: String,
    source: Option<String>,
}
