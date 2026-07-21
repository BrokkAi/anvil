---
title: Overview
description: Understand Anvil's ACP boundary, responsibilities, and supported surfaces.
---

Anvil is a Rust [Agent Client Protocol](https://agentclientprotocol.com/) server. It is the portable agent backend behind an editor, bot, TUI, or automation rather than a user interface of its own.

## The Boundary

An ACP client launches Anvil as a subprocess and communicates over stdio using JSON-RPC.

```text
ACP client           stdio / JSON-RPC           Anvil
----------           ----------------           -----
editor        --------------------------------> agent loop
issue bot     --------------------------------> model routing
custom TUI    --------------------------------> permissions
automation    --------------------------------> tools + sessions
```

The client owns conversation presentation, controls, and permission UI. Anvil owns model routing, tool execution, permission enforcement, session persistence, context management, sandbox selection, and MCP subprocesses.

ACP session options are client-owned. Model, behavior, reasoning effort, service tier, and permission selections apply to the live session and must be resubmitted by a client when appropriate. Anvil persists durable history and session metadata, not a universal UI preference profile. Stream timeouts are separate process defaults with an in-memory session override.

## Supported Surfaces

- [Zed](/zed/) and [JetBrains](/jetbrains/) have repository-backed configuration helpers.
- Any compatible client can use the [custom ACP client contract](/other-acp-clients/).
- The Rust [client examples](/build-acp-client/) demonstrate issue triage, code review, and issue drafting.

Codex/ChatGPT, Bedrock, Ollama, ds4, DeepSeek, Kimi, OpenAI-compatible endpoints, and OpenRouter are [model providers](/providers/), not ACP clients.

## Start Here

Use [Install Anvil](/install/) for release and source options, then complete the [ten-minute evaluation](/evaluate-anvil/) to verify one client, one provider, Bifrost-backed code intelligence, permission behavior, and session context.

Before connecting private source code or third-party extensions, review [Data and Trust Boundaries](/data-boundaries/) and [Permissions and Sandboxing](/permissions-sandboxing/).
