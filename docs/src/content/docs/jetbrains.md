---
title: JetBrains
description: Install, configure, validate, and troubleshoot Anvil in JetBrains ACP clients.
---

JetBrains ACP clients can launch Anvil as a custom agent server. Use an absolute path to the installed or locally built executable.

## Automatic Configuration

Run the installed Anvil binary:

```bash
anvil install jetbrains
```

Anvil detects its own executable path and merges an `Anvil` agent into
`~/.jetbrains/acp.json`. If an `Anvil` entry already exists, review it and then
pass `--force` to replace only that entry.

## Manual Configuration

Add an entry to `~/.jetbrains/acp.json`:

```json
{
  "agent_servers": {
    "Anvil": {
      "command": "/absolute/path/to/anvil",
      "args": [],
      "env": {}
    }
  }
}
```

Preserve unrelated entries already present in the file. Use `anvil.exe` on Windows. Restart the agent UI or start a fresh session after changing this configuration.

## Source-Checkout Helper

From an Anvil checkout:

```bash
rustup target add wasm32-wasip2
cargo xtask build-acp-for-jetbrains
```

The helper builds `target/release/anvil` and writes a `Brokk Code (Rust Local)` entry. Pass `--config <path>` when the active JetBrains ACP configuration is elsewhere. The helper is development tooling and is not packaged by `cargo install`.

## Setup and Validation

Open a new Anvil session and run `/setup` if no model is ready. Use the editor's session controls to select Permission, behavior, model, reasoning effort, and service tier.

Validate with a source question that requires managed Bifrost:

```text
Use Bifrost search_symbols to find main and get_symbol_sources to read it. Name the tools you called and do not edit files.
```

A visible Bifrost tool card plus repository-specific source output proves the code-intelligence path. If the agent falls back to reading files, confirm the session is using Anvil and inspect stderr for MCP startup failures.

The [Zed-led ten-minute evaluation](/evaluate-anvil/) uses the same Anvil behavior. Substitute this configuration and JetBrains' permission controls for the editor-specific steps.
