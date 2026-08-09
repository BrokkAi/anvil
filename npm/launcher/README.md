# @brokkai/anvil

[Anvil](https://github.com/BrokkAi/anvil) is a Rust ACP (Agent Client
Protocol) server with first-run setup for Codex, Ollama, and OpenRouter. This
package installs the released native `anvil` binary through npm — no Rust
toolchain and no first-run download.

## Install

```bash
npm install -g @brokkai/anvil
anvil --version
```

Or run it one-shot without a persistent install:

```bash
npx -y @brokkai/anvil --version
npx -y @brokkai/anvil@<version> --version
```

## How it works

`@brokkai/anvil` contains a small launcher and pins one exact optional
dependency per supported platform. npm installs only the package matching your
operating system and CPU; each platform package contains the complete
checksum-verified GitHub release bundle for that Anvil version. The launcher
executes the native binary by absolute path and forwards arguments, stdio,
signals, and the exit status unchanged.

| Platform | Package |
| --- | --- |
| macOS (Apple Silicon and Intel) | `@brokkai/anvil-darwin-universal` |
| Linux x86-64 (glibc) | `@brokkai/anvil-linux-x64` |
| Linux ARM64 (glibc) | `@brokkai/anvil-linux-arm64` |
| Android ARM64 (Termux) | `@brokkai/anvil-android-arm64` |
| Windows x86-64 | `@brokkai/anvil-win32-x64` |

Unsupported platforms (for example musl-based Linux such as Alpine) fail with
a clear error; use [another install method](https://anvil.brokk.ai/install/)
there. Do not install with `--no-optional` / `--omit=optional`: the platform
packages are optional dependencies, and skipping them leaves the launcher with
no binary to run.

## Pin, upgrade, uninstall

```bash
npm install -g @brokkai/anvil@0.24.3   # pin an exact Anvil release
npm install -g @brokkai/anvil@latest   # upgrade
npm uninstall -g @brokkai/anvil        # uninstall
```

## Next steps

Running `anvil` directly starts a stdio JSON-RPC server; use it through an ACP
client. Configure a supported client from the installed binary:

```bash
anvil install zed
anvil install jetbrains
anvil install neovim --plugin codecompanion
```

Documentation: <https://anvil.brokk.ai/> · Source and issues:
<https://github.com/BrokkAi/anvil>
