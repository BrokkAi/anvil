---
title: Zed
description: Install, configure, validate, and troubleshoot Anvil in Zed.
---

Zed can launch Anvil as a custom ACP agent. Use an absolute executable path so editor startup does not depend on shell `PATH` configuration.

## Manual Configuration

Install Anvil first, then add an entry to `~/.config/zed/settings.json`:

```json
{
  "agent_servers": {
    "Anvil": {
      "type": "custom",
      "command": "/absolute/path/to/anvil",
      "args": [],
      "env": {}
    }
  }
}
```

Merge `agent_servers` into the existing object rather than replacing unrelated settings. Restart the Agent Panel or create a fresh session after changing the command.

## Source-Checkout Helper

From an Anvil checkout:

```bash
rustup target add wasm32-wasip2
cargo xtask build-acp-for-zed
```

The helper builds `target/release/anvil` and writes a `Brokk Code (Rust Local)` entry while preserving unrelated JSON properties. Override the config location with `--config <path>`. This helper belongs to the source checkout and is not installed with the `brokk-anvil` crate.

## First Session

Choose the Anvil agent in Zed. If a provider is already usable, Anvil reports that it is ready; otherwise setup begins. Run `/setup` at any time. For Codex/ChatGPT:

```text
/setup codex
```

Browser callback authentication works on a local desktop. Use `/setup codex device` when localhost callbacks are unavailable.

Zed owns the current session selectors for model, behavior, reasoning, service tier, and Permission. Start a new session after changing the server command; resubmit desired client-owned selections when creating or restoring sessions.

## Validate the Integration

Open a source repository, select `readOnly`, and use a prompt that proves a Bifrost tool ran:

```text
Use the Bifrost get_summaries tool on src/main.rs. Name the symbols returned and do not edit files.
```

Success requires a Bifrost tool card and source-specific results. A generic answer or ordinary file read does not prove managed Bifrost. For the complete permission and edit journey, run the [ten-minute evaluation](/evaluate-anvil/).

## Troubleshooting

- Confirm `command` is absolute and the file is executable.
- Run the same path with `--version` in a terminal.
- Start a fresh Agent Panel session after configuration changes.
- If setup cannot show credential forms, the client may fall back to text commands; read the secret-handling warning in [Providers](/providers/#credentials-and-setup-forms).
- Inspect Anvil's stderr logs; stdout is reserved for ACP JSON-RPC.
