---
title: Other ACP Clients
description: Connect Anvil to a custom or third-party ACP-over-stdio client.
---

Anvil can work with clients that implement Agent Client Protocol v1 over stdio. Zed and JetBrains have repository-backed setup helpers; other clients are treated as custom integrations until their current setup and control surfaces are manually verified.

## Launch Contract

Configure the client to spawn:

```text
/absolute/path/to/anvil
```

- ACP JSON-RPC is written to stdout. Do not mix Anvil logs with that stream.
- Logs and diagnostics go to stderr. `RUST_LOG` controls filtering.
- The session working directory must be absolute.
- Client-supplied MCP servers may use stdio, streamable HTTP, or legacy SSE. `/mcp add` manages Anvil's persisted stdio subprocess entries.

## Client Responsibilities

A client should:

1. initialize ACP v1 and advertise filesystem, terminal, and elicitation capabilities honestly;
2. create, load, or resume a session with an absolute working directory;
3. render streamed agent text, thought, plan, tool-call, usage, command, and session-info updates;
4. render the config options Anvil advertises and resubmit live session selections when appropriate;
5. answer permission requests without silently broadening the selected policy;
6. support cancellation; and
7. preserve the session identifier needed for later lifecycle requests.

Clients with ACP elicitation forms can keep OpenRouter, Bedrock, and DeepSeek credential fields out of the transcript. Text-only clients retain command fallbacks that explicitly warn when a pasted secret will become transcript content.

## Configuration Ownership

Model selection, behavior, reasoning effort, service tier, and permission mode are client-owned live session controls. Anvil does not turn them into install-wide defaults. Sandbox strategy and automatic turn recaps are install preferences; provider credentials and MCP configuration have their own persistence. `/setup timeout` changes an in-memory override for the current session rather than an ACP selector.

Continue with [Build an ACP Client](/build-acp-client/) for the message flow and the checked-in Rust bot examples.
