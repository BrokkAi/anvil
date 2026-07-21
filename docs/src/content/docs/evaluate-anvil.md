---
title: Evaluate Anvil in Ten Minutes
description: Verify ACP startup, provider routing, managed Bifrost, permissions, and context from Zed.
---

> Documentation and fixture expectations checked against Anvil 0.23.0 and managed Bifrost 0.8.6 on 2026-07-21. Complete the journey below to verify your editor, provider, and platform.

This journey uses Zed and a checked-in Rust fixture. It proves that an ACP client can start Anvil, select a model, stream a managed-Bifrost tool call, enforce a permission mode, apply one approved edit, and report session context.

It does **not** measure model quality, large-repository performance, every provider, session recovery in every client, or complete sandbox security.

## 1. Get Anvil and the Fixture

Follow [Install Anvil](/install/) and confirm:

```bash
/absolute/path/to/anvil --version
```

Clone the repository and open the fixture:

```bash
git clone https://github.com/BrokkAi/anvil.git
cd anvil/docs/fixtures/ten-minute-evaluation
zed . # if the Zed CLI is installed
```

Alternatively, open that fixture directory from Zed's graphical interface.

The fixture is intentionally small:

```rust
pub fn greeting(name: &str) -> String {
    format!("Hello, {name}!")
}

pub fn welcome(name: &str) -> String {
    greeting(name)
}
```

The `greeting` definition starts at `src/lib.rs:1`; its call from `welcome` is at `src/lib.rs:6`.

## 2. Connect Zed

Add an agent server entry using the absolute released-binary path from [the Zed guide](/zed/#manual-configuration). Select the newly added Anvil agent, then start a **fresh** Agent Panel session after changing the configuration.

If Anvil finds a working model, it reports that it is ready. Otherwise it starts setup. Existing Codex credentials are discovered automatically; run this if sign-in is needed:

```text
/setup codex
```

Use `/setup choose` if more than one provider is ready. Other providers are covered in [Model Providers and Setup](/providers/).

The first Bifrost-backed prompt may download the Anvil release's pinned, checksum-verified Bifrost binary. That cold start requires network access and adds installation time; later sessions use the local cache.

## 3. Prove Read-Only Code Intelligence

Set the session **Permission** selector to `readOnly`, then send this exact prompt:

```text
Use Bifrost search_symbols to find greeting, get_symbol_sources to read its definition, and scan_usages_by_reference to find its caller. Report the exact paths and lines. Do not edit files and do not substitute text search.
```

Success requires all of these signals:

- the Agent Panel shows Bifrost tool cards for the requested operations;
- the answer identifies `greeting` at `src/lib.rs:1`;
- the answer identifies the call from `welcome` at `src/lib.rs:6`; and
- `git diff -- src/lib.rs` is empty.

A plausible answer without visible Bifrost calls does not prove the managed MCP path. A file read or grep result does not substitute for the requested symbol and usage tools.

## 4. Prove the Permission Boundary

Change the Permission selector to `default`, then send:

```text
Change only the greeting text from `Hello, {name}!` to `Welcome, {name}!`. Show the resulting diff and make no other changes.
```

Anvil should request permission for the edit tool. Read/search tools may be auto-approved; approve only the constrained edit request once, then verify:

```bash
git diff -- src/lib.rs
```

The only semantic change should be `Hello, {name}!` to `Welcome, {name}!`. Rejecting the request should leave the file unchanged. Restore the fixture when finished:

```bash
git restore src/lib.rs
```

## 5. Inspect Context

Run:

```text
/context
```

The report should describe the current session's context and token estimate. If the client exposes session history, close and resume the session and confirm the prior conversation is present; support for that UI is client-specific.

## Troubleshooting

- **No agent appears:** verify that the command is an absolute path and start a fresh Zed session.
- **No model is ready:** run `/setup codex`, start Ollama and use `/setup local`, or configure another [provider](/providers/).
- **Bifrost download fails:** check network access and the Anvil config/cache directory described in [Data and Trust Boundaries](/data-boundaries/).
- **No Bifrost tool cards appear:** confirm the prompt names the tools and that the managed server started; ordinary source reading is not the same test.
- **The edit happens without a prompt:** re-check the live Permission selector. `acceptEdits` and `bypassPermissions` intentionally behave differently from `default`.

## Interpret the Result

The evaluation separates runtime facts from model-dependent prose. Tool-card presence, source locations, the permission request, and the resulting diff are the evidence. A polished answer alone is not.

Continue with [Permissions and Sandboxing](/permissions-sandboxing/) before relying on the evaluation as a security claim, and [Tools and Managed Bifrost](/tools-bifrost/) before treating tool output as complete analysis.
