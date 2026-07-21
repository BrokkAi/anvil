---
title: Data and Trust Boundaries
description: Understand which Anvil components receive data and what the security controls do not guarantee.
---

Anvil connects an ACP client, a model provider, local tools, and optional extensions. The useful security model is therefore a set of boundaries rather than a single "sandboxed" label.

## Boundary map

| Component | Data or authority it may receive |
| --- | --- |
| ACP client | Streamed model output, tool activity, permission requests, session identifiers, and any content it submits. |
| Anvil process | Prompts, model responses, local session history, provider configuration, workspace paths, and tool results. |
| Hosted model provider | The prompt assembled for a turn, which can include user text, selected files, tool results, summaries, and system instructions. Provider retention and training terms are controlled by that provider. |
| MCP server or plugin | Whatever its process, hooks, commands, inherited environment, and filesystem permissions allow. |
| Built-in filesystem and shell tools | Workspace files; shell authority depends on permission mode, effective sandbox strategy, and any approved escalation. |

## Local persistence and credentials

Anvil stores ACP sessions under `<repository>/.brokk/sessions/` so they can be loaded or resumed. `BROKK_SESSION_STORAGE_ROOT` can redirect the session-storage root. Compaction changes the model-facing context but does not erase raw turns, so archives can contain prompts, source excerpts, tool results, and summaries.

Remembered approvals are separate: they live in `<permission-scope-root>/.brokk/permissions.json`. Linked Git worktrees resolve to the main repository's permission scope, so an approval can survive a worktree's removal and be shared across linked worktrees. Anvil's own `.gitignore` excludes `.brokk/`; add the same rule in repositories that do not already ignore it.

Provider credentials may be reused from an existing provider-specific store or saved through setup. Prefer structured setup forms. A secret pasted into a normal prompt or text-mode setup path becomes conversation content and may be persisted or sent to the selected provider.

Protect the account and filesystem that run Anvil, and do not share session archives or trace files as if they were harmless logs.

## Permission is not sandboxing

Permission mode decides whether a tool call is allowed or needs approval. Sandbox mode determines how execution is isolated.

- `readOnly` blocks edit and shell tools but is not a separate operating-system container.
- `default` and `acceptEdits` can prompt, while `bypassPermissions` removes prompts.
- `os` wraps shell commands with Seatbelt on macOS or Bubblewrap on Linux.
- `wasm` isolates selected parsing, search, and archive operations; it does **not** OS-sandbox shell commands.
- `off` relies on permission decisions and path validation rather than an execution sandbox.

Built-in file tools validate paths against the session workspace. That does not constrain separately launched plugins, hooks, MCP servers, or an approved shell command in the same way.

MCP subprocesses and plugin hooks inherit Anvil's process environment unless their launcher narrows it. They can therefore see provider keys or other ambient credentials and may execute at startup or hook time without an individual model tool-call approval. Launch Anvil with a minimized environment, scope credentials narrowly, and enable only reviewed extensions.

## Network boundaries

Hosted providers receive model requests over the network. Provider discovery, authentication, usage or balance checks, plugin installation, web tools, and the first managed-Bifrost download may also make outbound requests. The exact destinations depend on enabled providers and extensions.

Client-supplied MCP may be a local stdio subprocess or a remote HTTP/SSE endpoint. A stdio subprocess may itself use the network, while remote transports send MCP traffic and configured headers directly to their endpoint.

## History and compaction

Automatic and manual compaction replace older model-facing context with a cumulative snapshot while retaining a recent exact tail. The canonical system, workspace instructions, and skill prefix remain separate from that summary. Raw turns remain available for ACP replay, rewind, and local recovery.

Compaction is a context-management feature, not deletion, redaction, encryption, or a provider-retention control.

## Unsupported guarantees

Anvil does not claim that:

- model output is correct, safe, or free from prompt injection;
- a permission prompt makes an approved command harmless;
- WASM mode contains shell execution;
- plugins or MCP servers are reviewed or isolated by Anvil;
- compaction deletes historical data;
- path validation or OS sandboxing proves complete host security;
- every platform offers the same sandbox strength;
- a ten-minute evaluation proves large-repository performance or complete security.

Choose the provider, client, extensions, permission mode, and sandbox strategy appropriate to the sensitivity of the workspace. For control details, see [Permissions and Sandboxing](../permissions-sandboxing/).
