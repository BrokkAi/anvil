---
title: Permissions and Sandboxing
description: Distinguish Anvil permission policy, OS sandboxing, Wasm isolation, and escalation.
---

Permission mode decides whether a tool call is allowed or requires approval. Sandbox mode decides where and with what operating-system access certain work executes. They are separate controls.

## Permission Modes

| Mode | Behavior |
| --- | --- |
| `default` | Ask before edits and most shell commands; conservatively safe sandboxed read-only commands may run automatically. |
| `auto` | Send promptable calls to the permission classifier; calls it does not approve are denied without a human prompt. |
| `acceptEdits` | Allow edits; continue to ask before most shell commands. |
| `readOnly` | Block edits and shell commands. |
| `bypassPermissions` | Allow tool calls without prompting. |

The ACP client owns the live Permission selector. Remembered **Always allow** decisions can be inspected or removed:

```text
/permissions list
/permissions revoke <number-or-key>
/permissions clear
```

Shell commands request explicit outside-sandbox escalation with `sandbox_permissions: "require_escalated"`. In interactive modes this produces a **Run outside sandbox** choice. An outside-sandbox approval is never stored as an Always allow rule.

## Sandbox Strategies

| Strategy | Parsing and bounded file work | Shell commands |
| --- | --- | --- |
| `os` | Native | Wrapped by macOS Seatbelt or Linux Bubblewrap. |
| `wasm` | Selected parsing, search, instruction, and archive work runs in the embedded Wasmtime component. | Permission-gated, but **not** contained by Seatbelt or Bubblewrap. |
| `off` | Native in-process | Permission-gated only. |

Anvil defaults to `os` when the platform sandbox is available. Otherwise a default-feature build can use the Wasm fallback. A build made with `--no-default-features`, or a process started with `--no-wasm-sandbox`, cannot use that fallback.

```text
/setup sandbox default
/setup sandbox os
/setup sandbox wasm
/setup sandbox off
/setup sandbox status
```

Wasm mode isolates selected untrusted parsing and archive operations with per-request stores, CPU fuel, memory caps, bounded output, and scoped read-only preopens. It does not replace workspace path validation, and it does not turn arbitrary shell commands into Wasm programs.

## Workspace Paths

Built-in reads use `safe_resolve`; writes use `safe_resolve_for_write`. Both keep operations under the session working directory and reject traversal through missing or symlinked path segments. Session archive reads and writes are bounded and route through the sandbox backend.

## What These Controls Do Not Prove

- A permission prompt is not an OS containment boundary.
- Wasm parsing isolation is not shell isolation.
- `sandbox off` intentionally leaves only permission controls and path validation.
- A third-party MCP server, plugin hook, or provider has its own trust boundary.
- No mode makes model output correct or makes every zero-result analysis complete.

Continue with [Data and Trust Boundaries](/data-boundaries/) for data flow, credentials, persistence, and external processes.
