---
title: Install Anvil
description: Install a released Anvil binary, use Cargo, or build from source.
---

Anvil runs as a subprocess launched by an ACP client. Prefer a released binary when evaluating it: this avoids a large Rust and Wasmtime compile.

## Prebuilt Release

Download the archive and matching `.sha256` sidecar from the [latest GitHub release](https://github.com/BrokkAi/anvil/releases/latest).

| Platform | Release asset suffix |
| --- | --- |
| Linux x86-64 | `x86_64-unknown-linux-gnu.zip` |
| Linux ARM64 | `aarch64-unknown-linux-gnu.zip` |
| Android ARM64 | `aarch64-linux-android.zip` |
| Windows x86-64 | `x86_64-pc-windows-msvc.zip` |
| macOS Intel and Apple Silicon | `universal-apple-darwin.zip` |

Verify the downloaded archive before extracting it:

```bash
# Linux x86-64 example; substitute the version and target you downloaded.
sha256sum -c brokk-anvil-v0.23.0-x86_64-unknown-linux-gnu.zip.sha256

# macOS
shasum -a 256 -c brokk-anvil-vX.Y.Z-universal-apple-darwin.zip.sha256
```

On Windows, compare the sidecar with:

```powershell
$archive = ".\brokk-anvil-vX.Y.Z-x86_64-pc-windows-msvc.zip"
$expected = (Get-Content "$archive.sha256").Split()[0].ToLower()
$actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLower()
if ($actual -ne $expected) { throw "SHA-256 mismatch" }
```

Extract the archive, place `anvil` or `anvil.exe` somewhere stable, and record its absolute path for the client configuration. Confirm the binary:

```bash
/absolute/path/to/anvil --version
```

If a Unix extraction tool drops the executable bit, run `chmod +x /absolute/path/to/anvil`. macOS releases are not currently notarized, so Gatekeeper may require you to approve the downloaded binary through the normal system security UI.

## Install With Cargo

The published crate is named `brokk-anvil`; the executable is `anvil`. Install a current stable Rust toolchain with [rustup](https://rustup.rs/) first. The default build embeds a Wasm sandbox and therefore needs the WASI Preview 2 target.

```bash
rustup toolchain install stable
rustup default stable
rustup target add wasm32-wasip2
cargo install brokk-anvil --locked --force
anvil --version
```

A cold Cargo install can take longer than the evaluation itself. Use the prebuilt archive when time matters.

## Build This Checkout

```bash
rustup target add wasm32-wasip2
cargo build --release --bin anvil
./target/release/anvil --version
```

To omit the embedded Wasm parser sandbox:

```bash
cargo build --release --no-default-features --bin anvil
```

The source checkout also provides `cargo xtask build-acp-for-zed` and `cargo xtask build-acp-for-jetbrains`. These helpers are not included by `cargo install`; installed users configure the absolute binary path manually.

## Linux Sandbox Prerequisite

Install Bubblewrap (`bwrap`) to use Anvil's Linux OS-level shell sandbox. Without it, Anvil can use its Wasm parsing fallback, but that fallback does **not** provide equivalent containment for shell commands. See [Permissions and Sandboxing](/permissions-sandboxing/).

Continue with [Zed](/zed/), [JetBrains](/jetbrains/), or [another ACP client](/other-acp-clients/).
