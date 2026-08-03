---
title: Install Anvil
description: Install a released Anvil binary, use Cargo, or build from source.
---

Anvil runs as a subprocess launched by an ACP client. Prefer a released binary when evaluating it: this avoids a large Rust and Wasmtime compile.

## Homebrew

Install from the [BrokkAi Homebrew tap](https://github.com/BrokkAi/homebrew-tap)
on macOS (Apple Silicon and Intel) or Linux (x86-64 and ARM64 glibc):

```bash
brew tap brokkai/tap
brew install anvil
anvil --version
```

The formula downloads the release archive for your platform and verifies its
published SHA-256 checksum. Upgrade with `brew upgrade anvil` and uninstall
with `brew uninstall anvil`. The tap regenerates its formulae from tagged
releases on a schedule, so upgrades follow new Anvil releases automatically.
For Windows, musl-based Linux, or Android, use the methods below.

## Install Script

Install the released binary with the install script:

```bash
curl -fsSL https://raw.githubusercontent.com/BrokkAi/anvil/refs/heads/master/install.sh | bash
```

The script detects your platform, downloads the matching release archive from
GitHub, requires and verifies its published SHA-256 checksum, and installs
`anvil` into `~/.local/bin`. It offers to add that directory to your `PATH`
when it is missing and the terminal is interactive.

### Supported Platforms

| Platform | Architecture | Install script | Release target |
| --- | --- | --- | --- |
| macOS | Apple Silicon and Intel | Yes | `universal-apple-darwin` |
| Linux (glibc) | x86-64 | Yes | `x86_64-unknown-linux-gnu` |
| Linux (glibc) | ARM64 | Yes | `aarch64-unknown-linux-gnu` |
| Linux (musl, such as Alpine) | x86-64 and ARM64 | No, use Cargo | none published |
| WSL 1 and WSL 2 | x86-64 and ARM64 | Yes, as Linux | Linux targets above |
| Android (Termux) | ARM64 | Yes | `aarch64-linux-android` |
| Windows | x86-64 | No, use Cargo or the release archive | `x86_64-pc-windows-msvc` |

The script stops with an explanation on musl-based Linux rather than installing
a glibc binary that cannot run.

### WSL

WSL is Linux, so run the same command inside your WSL shell. It installs the
Linux binary, which runs inside WSL only. Windows-native ACP clients cannot
execute it; install the Windows build separately when a client running outside
WSL needs to launch Anvil.

### Windows

The install script does not cover Windows. Use [Cargo](#install-with-cargo), or
download the `.zip` archive and matching `.sha256` sidecar from the
[release page](https://github.com/BrokkAi/anvil/releases) and place `anvil.exe`
on your `PATH`. Running the script from Git Bash, MSYS2, or Cygwin does not
install the Windows binary; use WSL only when you specifically want Anvil to
run inside WSL.

Pipe-to-shell installs run remote code. To read the script before running it,
download it first:

```bash
curl -fsSL -O https://raw.githubusercontent.com/BrokkAi/anvil/refs/heads/master/install.sh
less install.sh
bash install.sh
```

The script accepts these environment variables:

| Variable | Purpose |
| --- | --- |
| `INSTALL_DIR` | Install directory. Defaults to `~/.local/bin`. |
| `ANVIL_INSTALL_DIR` | Same as `INSTALL_DIR`, with higher precedence. |
| `ANVIL_VERSION` | Release tag to install, for example `v0.24.2`. Defaults to the latest release. |
| `ANVIL_GITHUB_OWNER` | GitHub owner to download from. Defaults to `BrokkAi`. |
| `GITHUB_TOKEN` | Token used for GitHub API rate limits. |
| `PROFILE` | Shell profile to update when the install directory is not on `PATH`. |

Pin a version and choose the directory like this:

```bash
ANVIL_VERSION=v0.24.2 INSTALL_DIR=/usr/local/bin \
  bash -c "$(curl -fsSL https://raw.githubusercontent.com/BrokkAi/anvil/refs/heads/master/install.sh)"
```

Re-running the script installs over the existing binary, so it also serves as
the upgrade path.

## Verify the Install

```bash
anvil --version
```

## Manual Prebuilt Release

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

Extract the archive and place `anvil` or `anvil.exe` somewhere stable. Confirm the binary:

```bash
/absolute/path/to/anvil --version
```

If a Unix extraction tool drops the executable bit, run `chmod +x /absolute/path/to/anvil`. macOS releases are not currently notarized, so Gatekeeper may require you to approve the downloaded binary through the normal system security UI.

Anvil can configure supported ACP clients with the absolute path of the
currently running executable:

```bash
anvil install zed
anvil install jetbrains
anvil install neovim --plugin codecompanion
anvil install neovim --plugin avante
```

Use `--force` to replace an existing Anvil entry or generated Neovim module.
When `--plugin` is omitted for Neovim, Anvil prompts in an interactive terminal
and defaults to CodeCompanion in non-interactive use. Move the executable to its
stable location before running an installer; editor settings retain that
detected absolute path.

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

The source checkout also provides `cargo xtask build-acp-for-zed` and `cargo xtask build-acp-for-jetbrains`. These developer helpers build first and then configure the resulting checkout binary. Installed users should use `anvil install`.

## Linux Sandbox Prerequisite

Install Bubblewrap (`bwrap`) to use Anvil's Linux OS-level shell sandbox. Without it, Anvil can use its Wasm parsing fallback, but that fallback does **not** provide equivalent containment for shell commands. See [Permissions and Sandboxing](/permissions-sandboxing/).

Continue with [Zed](/zed/), [JetBrains](/jetbrains/), or [another ACP client](/other-acp-clients/).
