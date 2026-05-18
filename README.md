# Anvil

`anvil` is a high-performance Agent Client Protocol (ACP) server implemented in Rust. It acts as an agentic bridge between ACP-compatible IDEs (like Zed) and Ollama or OpenAI-compatible LLM backends, providing Brokk-style agentic capabilities directly within the editor.

## Features

- **Standardized Protocol**: Full support for ACP over stdio, including session lifecycle management.
- **Agentic Tool Loop**: Implements a multi-turn autonomous loop with a configurable turn limit.
- **Rich Feedback**: Streams text responses and tool-call lifecycle notifications (Pending, InProgress, Completed/Failed) with inline diffs for file writes.
- **Permission Gating**: Configurable security policies (Default, Accept Edits, Read-only, Bypass) to control tool execution.
- **Code Intelligence**: Optional integration with Bifrost for symbol search, cross-references, and structural analysis.
- **Session Persistence**: Saves and resumes conversation history and session state from disk.

## Architecture

The server is composed of several specialized modules:

- **`agent`**: The entry point for the ACP protocol; handles JSON-RPC dispatching for sessions, configuration, and prompts.
- **`tool_loop`**: The core agentic engine. Orchestrates LLM streaming, tool execution, and the permission gate.
- **`llm_client`**: Handles communication with OpenAI-compatible APIs, including tool-calling SSE stream parsing.
- **`session`**: Manages session state, conversation history persistence, and model/mode selection.
- **`tools`**: Implementation of built-in filesystem tools (read, write, list) and shell execution.
- **`bifrost_client`**: Manages the lifecycle of the Bifrost subprocess for advanced code analysis tools.

## Configuration / CLI Options

The server binary is named `anvil`. It is **zero-config by design**: at startup it reads `~/.codex/auth.json` for Codex credentials and probes `http://localhost:11434/v1/models` for Ollama, presenting whatever responds as a single combined picker. Models are tagged on the wire as `codex::<id>` and `ollama::<id>` so identical names from different sources stay distinct.

There are no flags to point at a different Ollama URL or restrict the picker. If your daemon listens elsewhere, run `ollama serve` on the default port.

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--default-model` | - | - | Override the default model id for new sessions. Accepts wire form (`codex::gpt-5-codex`) or a bare id. |
| `--max-turns` | - | `25` | Max tool-calling iterations per prompt before forcing a final response. |
| `--bifrost-binary`| `BROKK_BIFROST_BINARY` | - | Path to the `bifrost` executable to enable code-intel tools. |
| `--llm-idle-timeout-secs` | `ANVIL_LLM_IDLE_TIMEOUT_SECS` | (see source) | Seconds of SSE inactivity before aborting a streaming LLM response. |
| `--no-wasm-sandbox` | `ANVIL_NO_WASM_SANDBOX` | `false` | Disable the wasmtime-hosted parser sandbox. |

## Running Locally

### Quickest path: build + wire into your editor in one step

The crate ships with two `cargo xtask` subcommands that build the release
binary **and** rewrite an agent-server entry under `agent_servers`
in your editor's config:

```bash
# Wire into Zed   (~/.config/zed/settings.json)
cargo xtask build-acp-for-zed

# Wire into JetBrains   (~/.jetbrains/acp.json)
cargo xtask build-acp-for-jetbrains
```

Each task runs `cargo build --release --bin anvil`, then writes a
single agent-server entry pointing at `target/release/anvil`. Other
entries in the file are preserved verbatim. Re-run any time the binary
changes — the entry is rewritten in place.

Both subcommands accept:
- `--config <path>` — override the editor config path (mostly for tests).
- `--bifrost-binary <name|path>` — value passed via `--bifrost-binary`
  in the entry's args. Defaults to the literal `bifrost` (assumed to be
  on the editor's `PATH`); pass an absolute path if Bifrost lives
  somewhere `PATH` does not reach.

### Manual / advanced

If you prefer to wire things up by hand (or just want to run the binary
standalone):

```bash
# Build the release binary
cargo build --release --bin anvil

# Run against a local Ollama instance (auto-discovers models)
./target/release/anvil

# Or pin a default model on startup (wire id or bare id):
./target/release/anvil --default-model ollama::llama3.1
```

Then add the binary path to your editor's agent server config:
- Zed: `~/.config/zed/settings.json` under `agent_servers`, with
  `"type": "custom"` and `"command"` set to the absolute path.
- JetBrains: `~/.jetbrains/acp.json` under `agent_servers`, same shape
  minus the `type` field.

## Tool Calling and Permissions

The server supports a variety of tools, including `readFile`, `writeFile`, `listDirectory`, and `runShellCommand`. Execution is governed by a **Permission Mode** selectable in the client:

- **Default**: Prompts the user for approval before every mutating tool call.
- **Accept Edits**: Automatically allows file modifications but prompts for shell commands.
- **Read-only**: Strictly forbids any tool that modifies the filesystem or executes commands.
- **Bypass Permissions**: Trust the agent to execute all tools without interruption.

## Bifrost Integration

When configured with `--bifrost-binary`, the server spawns a Bifrost subprocess to provide structural code intelligence. This enables advanced tools such as:
- `search_symbols`: Find definitions across the workspace.
- `get_symbol_sources`: Fetch source code for specific symbols.
- `most_relevant_files`: Identify related files using import analysis and git history.

## Releases

Release binaries are published on GitHub Releases. Tags follow `vX.Y.Z`; the workflow validates that the tag matches `Cargo.toml`'s `version` before publishing.

Asset names per platform:
- `anvil-linux-x86_64`
- `anvil-macos-aarch64`
- `anvil-windows-x86_64.exe`
