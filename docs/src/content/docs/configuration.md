---
title: Configuration and CLI
description: Understand configuration ownership, files, environment variables, and process flags.
---

Anvil separates configuration by ownership and lifetime. This prevents an editor's per-session choice from silently becoming an installation-wide default.

## Provider configuration

Provider credentials and discovery settings are shared by sessions in an Anvil installation. Use `/setup` inside a session to configure them; see [Model Providers and Setup](../providers/).

## Session controls

The ACP client owns current-session controls such as model selection, behavior mode, permission mode, reasoning effort, and service tier. Clients must send these values for each new or loaded session. Anvil does not write them to setup state or session manifests as user preferences.

## Install defaults

Sandbox behavior and automatic turn recaps are persisted as installation defaults and seed future sessions. `--transient-setup` keeps sandbox, recap, and first-run changes process-local, but provider credentials, `allowed_tools`, and MCP configuration remain persistent.

## Tool allowlist

`setup.json` accepts an optional `allowed_tools` array of exact model-callable tool names. It is applied after built-in, MCP, and dynamic tools such as `activate_skill` and `task` are assembled.

```json
{
  "allowed_tools": ["read_file", "grep_search", "search_symbols", "task"]
}
```

Unknown names are ignored. Omitting the field exposes the full assembled catalog; an empty list exposes no model-callable tools.

## CLI reference

```text
anvil [OPTIONS]
```

| Flag | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `--default-model` | — | unset | Default wire id or bare provider model id for new sessions. |
| `--reasoning-effort` | — | provider default | Seed new sessions with a supported reasoning effort, or `off`. |
| `--max-turns` | — | `0` | Prompt tool-loop ceiling; `0` is unbounded. |
| `--max-sessions` | — | `50` | Resident-session LRU limit; `0` disables it. |
| `--max-history-turns` | — | `50` | In-memory history-turn limit; `0` disables it. |
| `--llm-idle-timeout-secs` | `ANVIL_LLM_IDLE_TIMEOUT_SECS` | `300` | Wait for first meaningful stream progress. |
| `--llm-stall-timeout-secs` | `ANVIL_LLM_STALL_TIMEOUT_SECS` | `60` | Allowed gap after streaming begins. |
| `--transient-setup` | `ANVIL_TRANSIENT_SETUP` | `false` | Keep selected installation defaults process-local. |
| `--no-wasm-sandbox` | `ANVIL_NO_WASM_SANDBOX` | `false` | Disable the Wasmtime parser/search fallback; without an OS sandbox, shell commands then have no execution sandbox. |
| `--no-shell-minimizer` | `ANVIL_NO_SHELL_MINIMIZER` | `false` | Disable post-capture condensing of shell output for well-known tools (git, cargo, pytest, npm, ...). When condensing is enabled (the default), condensed results reference the raw output preserved under `.brokk/shell-output/` in the workspace. |

`ANVIL_TRACE_JSONL=<path>` writes diagnostic LLM and step events to a local JSONL trace. Treat it as sensitive: prompts, responses, and operational metadata can contain project information.

Run `anvil --help` for the exact flags supported by the installed version.

## Where secrets live

Provider integrations may reuse external credential stores or save credentials through Anvil's secret/setup mechanisms. Prefer elicitation forms such as `/setup codex` or `/setup openrouter`; a key pasted directly into chat becomes session content. See [Data and Trust Boundaries](../data-boundaries/).
