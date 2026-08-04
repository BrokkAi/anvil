<h1 align="center">Anvil</h1>

<p align="center">
  <a href="https://anvil.brokk.ai/">
    <img src="docs/public/anvil-social-card.png" alt="Anvil — one portable agent runtime for your ACP interface" width="720">
  </a>
</p>

<p align="center">
  <a href="https://github.com/BrokkAi/anvil/actions/workflows/ci.yml"><img src="https://github.com/BrokkAi/anvil/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/BrokkAi/anvil/releases/latest"><img src="https://img.shields.io/github/v/release/BrokkAi/anvil" alt="Latest release"></a>
  <a href="https://crates.io/crates/brokk-anvil"><img src="https://img.shields.io/crates/v/brokk-anvil" alt="crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/BrokkAi/anvil" alt="LGPL-3.0-only"></a>
</p>

<p align="center">
  <a href="#install-and-evaluate">Quickstart</a> ·
  <a href="https://anvil.brokk.ai/">Documentation</a> ·
  <a href="https://anvil.brokk.ai/evaluate-anvil/">Ten-minute evaluation</a> ·
  <a href="https://discord.gg/geYkWUeH">Discord</a>
</p>

Anvil is a Rust [Agent Client Protocol](https://agentclientprotocol.com/) server: one reusable agent engine for editors, bots, TUIs, and internal tools.

It is intentionally **just the ACP server**. Anvil owns model routing, the tool loop, permission enforcement, sessions, context compaction, sandboxing, and MCP integration. The ACP client owns the user experience.

```text
ACP client              stdio / JSON-RPC              Anvil
----------              ----------------              -----
Zed        ----------------------------------------->  agent loop
JetBrains   --------------------------------------->  model routing
issue bot   --------------------------------------->  permission gate
custom TUI  --------------------------------------->  tools + sessions
```

## Why Anvil

- **One engine, many frontends.** Reuse the same execution semantics from a supported editor or any ACP-over-stdio client.
- **Model routing built in.** Connect Codex/ChatGPT, Bedrock, Ollama, ds4, DeepSeek, Kimi, OpenAI-compatible providers, or OpenRouter.
- **Real agent tooling.** Filesystem and shell tools, managed Bifrost code intelligence, MCP servers, skills, plugins, and subagents.
- **Explicit safety boundaries.** Clients choose permission behavior while Anvil applies permission gates, workspace path checks, and the configured sandbox strategy.
- **Persistent work.** Load and resume sessions with durable history, usage reporting, and context compaction.

## Install and Evaluate

Install with Homebrew on macOS (Apple Silicon and Intel) or Linux (x86-64 and
ARM64 glibc):

```bash
brew install brokkai/tap/anvil
anvil --version
```

Or install through npm on any supported platform (macOS, Linux glibc, Windows, Android/Termux) — the package installs the released native binary, no Rust toolchain needed:

```bash
npm install -g @brokkai/anvil
anvil --version

# or run one-shot without installing
npx -y @brokkai/anvil --version
```

Or install the latest checksum-verified release on macOS, Linux, WSL, or Android/Termux:

```bash
curl -fsSL https://raw.githubusercontent.com/BrokkAi/anvil/refs/heads/master/install.sh | bash
anvil --version
```

To review the installer before running it, download `install.sh`, inspect it, then run it with `bash`. You can also download an archive from [GitHub Releases](https://github.com/BrokkAi/anvil/releases/latest), or install from crates.io:

```bash
rustup target add wasm32-wasip2
cargo install brokk-anvil --locked --force
anvil --version
```

Running `anvil` directly starts a stdio JSON-RPC server; use it through an ACP client. Continue with the [installation guide](https://anvil.brokk.ai/install/) or the reproducible [ten-minute evaluation](https://anvil.brokk.ai/evaluate-anvil/).

`anvil serve` (included in all official binaries; source builds can opt out with `cargo build --no-default-features --features wasm-sandbox`) starts the same runtime as an HTTP daemon on a loopback listener (default `127.0.0.1:26845`), exposing versioned REST endpoints for session lifecycle, model discovery, the tool catalog, and asynchronous prompt runs with Server-Sent-Event streaming and cancellation (`/health`, `/v1/models`, `/v1/tools`, `/v1/sessions`, `/v1/runs`). The daemon supports bearer-token authentication (required for non-loopback binding), server-enforced workspace roots, and interactive permission approval endpoints. The wire contract is versioned and checked in under [`openapi/`](openapi/) (OpenAPI document, SSE event schema, and compatibility policy), with a black-box conformance suite holding the implementation to it; generated SDKs are under active development ([#315](https://github.com/BrokkAi/anvil/issues/315)).

Configure a supported ACP client directly from the installed binary:

```bash
anvil install zed
anvil install jetbrains
anvil install neovim --plugin codecompanion
```

## Supported Clients

- [Zed](https://anvil.brokk.ai/zed/), [JetBrains](https://anvil.brokk.ai/jetbrains/), and [Neovim](https://anvil.brokk.ai/neovim/) have direct installers and verified configuration shapes.
- [Other ACP clients](https://anvil.brokk.ai/other-acp-clients/) can launch Anvil as a custom stdio agent.
- The [client-building guide](https://anvil.brokk.ai/build-acp-client/) and `examples/` show issue triage, review, and issue-drafting automations.

## Documentation and Development

The [Anvil documentation](https://anvil.brokk.ai/) is the canonical user reference for providers, commands, tools, permissions, sessions, extensibility, and trust boundaries.

See [CONTRIBUTING.md](CONTRIBUTING.md) for source development, runtime invariants, tests, dependency-license policy, pull requests, and releases.

## License

Anvil is licensed under `LGPL-3.0-only`. See the practical [License and Use Cases](https://anvil.brokk.ai/license-use-cases/) guide, the controlling [LICENSE](LICENSE), and [third-party notices](https://anvil.brokk.ai/third-party-notices/).
