# Anvil

`anvil` is a high-performance Agent Client Protocol (ACP) server implemented in Rust. It acts as an agentic bridge between ACP-compatible IDEs (like Zed), Codex/ChatGPT sign-in, Ollama, OpenRouter, and OpenAI-compatible LLM backends, providing Brokk-style agentic capabilities directly within the editor.

## Features

- **Standardized Protocol**: Full support for ACP over stdio, including session lifecycle management.
- **Agentic Tool Loop**: Implements a multi-turn autonomous loop with a configurable turn limit.
- **Rich Feedback**: Streams text responses and tool-call lifecycle notifications (Pending, InProgress, Completed/Failed) with inline diffs for file writes.
- **First-Run Setup**: Starts sessions with setup guidance and keeps model/provider configuration behind one `/setup` command.
- **Permission Gating**: Configurable security policies (Default, Accept Edits, Read-only, Bypass) to control tool execution.
- **Code Intelligence**: Optional integration with Bifrost for symbol search, cross-references, and structural analysis.
- **Session Persistence**: Saves and resumes conversation history and session state from disk.
- **Context Management**: `/context` reports current token usage against the model's window; `/compress` summarizes prior turns via the LLM (with hierarchical fallback for turns too large for one summarization call) and persists the result so reloads send the same compressed prompt. Compression also fires automatically when a prompt would overflow the configured budget.

## Architecture

The server is composed of several specialized modules:

- **`agent`**: The entry point for the ACP protocol; handles JSON-RPC dispatching for sessions, configuration, and prompts.
- **`tool_loop`**: The core agentic engine. Orchestrates LLM streaming, tool execution, and the permission gate.
- **`llm_client`**: Handles communication with OpenAI-compatible APIs, including tool-calling SSE stream parsing.
- **`session`**: Manages session state, conversation history persistence, and model/mode selection.
- **`tools`**: Implementation of built-in filesystem tools (read, write, list) and shell execution.
- **`bifrost_client`**: Manages the lifecycle of the Bifrost subprocess for advanced code analysis tools.

## Configuration / CLI Options

The server binary is named `anvil`. It is **zero-config by design**: at startup it reads `~/.codex/auth.json` for Codex credentials, probes `http://localhost:11434/v1/models` for Ollama, and checks OpenRouter when credentials are available. New sessions always include a short setup hint; run `/setup` in the editor to choose automatically, sign in to Codex, use local models, connect OpenRouter, or change advanced settings.

Provider priority for "Choose for me" is Codex first, local Ollama second, OpenRouter last. Models are tagged on the wire as `codex::<id>`, `ollama::<id>`, and `openrouter::<id>` so identical names from different sources stay distinct.

There are no flags to point at a different Ollama URL or restrict the picker. If your daemon listens elsewhere, run `ollama serve` on the default port.

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--default-model` | - | - | Override the default model id for new sessions. Accepts wire form (`codex::<id>`, `ollama::<id>`, `openrouter::<id>`) or a bare id. |
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

# Run and let /setup guide model/provider configuration
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

## Subagents

A subagent is a markdown file that the LLM can delegate a focused task to via the `task` tool. Each delegation runs in an isolated tool loop with its own message history; only the subagent's final assistant text comes back to the parent conversation. Use them to keep noisy multi-step work (deep research, long greps, repetitive refactors) out of the main transcript.

The `task` tool is only advertised to the model when at least one subagent has been discovered. If no subagents exist, the feature is invisible.

### File format

A subagent file is plain markdown with YAML frontmatter. `name` and `description` are required; the body is the subagent's system prompt.

```markdown
---
name: bug-hunter
description: Adversarially review a diff for off-by-one errors, missing null checks, and lock-order issues. Returns findings as a numbered list.
---

You are a meticulous code reviewer. When given a diff:

1. Read every changed file in full, not just the hunks.
2. Flag concrete bugs only — no style nits, no speculation.
3. Return findings as `<path>:<line> — <one-line description>`.
```

The frontmatter `name` should match the filename stem (`bug-hunter.md` → `name: bug-hunter`). Mismatches load with a warning. Files above 256 KiB are skipped.

### Discovery order

Subagents are loaded from the following roots, in order. Later entries override earlier ones with the same `name`:

1. `~/.claude/agents/` (user scope, Claude Code compatible)
2. `~/.agents/agents/` (user scope, cross-client)
3. `<git-root>/.claude/agents/` walking down to `cwd` (project scope, Claude Code compatible)
4. `<git-root>/.agents/agents/` walking down to `cwd` (project scope, cross-client)

Layout is flat — one `.md` per subagent, no subdirectories. Project-scope subagents win over user-scope; within a scope, `.agents/` wins over `.claude/`.

### How the model invokes a subagent

The model calls the `task` tool with three arguments:

- `subagent_type`: name from the catalog (enum-constrained, so the model cannot invent names).
- `description`: a 3-5 word label for the UI card.
- `prompt`: the self-contained task. The subagent does **not** see the parent conversation, so the prompt must include any required context.

### Constraints

- **Same permission gate.** The subagent runs against the parent session's mode and "always allow" set. `readOnly` blocks writes inside subagents too; mutating tools still prompt under `default`.
- **No nested delegation.** A subagent cannot itself call `task`. Nesting depth is capped at 1.
- **Bounded turns.** Each delegation runs at most 25 tool-calling iterations, regardless of the parent's `--max-turns`.
- **Silent execution.** The subagent's intermediate tool calls and streamed tokens are not sent to the client. Permission prompts still surface when required.
- **Cancellation propagates.** Cancelling the parent prompt cancels the active subagent.

## Bifrost Integration

When configured with `--bifrost-binary`, the server spawns a Bifrost subprocess to provide structural code intelligence. This enables advanced tools such as:
- `search_symbols`: Find definitions across the workspace.
- `get_symbol_sources`: Fetch source code for specific symbols.
- `most_relevant_files`: Identify related files using import analysis and git history.

## Releases

Release binaries are published on GitHub Releases. The `anvil` crate is published to crates.io through trusted publishing from `.github/workflows/publish-crate.yml` using the `release` environment. Tags follow `vX.Y.Z`; the workflows validate that the tag matches `Cargo.toml`'s `version` before publishing.

Asset names per platform:
- `anvil-linux-x86_64`
- `anvil-macos-aarch64`
- `anvil-windows-x86_64.exe`
