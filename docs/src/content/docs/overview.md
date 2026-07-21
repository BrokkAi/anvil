---
title: Overview
description: What Anvil is, what it owns, and where it fits.
---

Anvil is a Rust [Agent Client Protocol](https://agentclientprotocol.com/) server. It is intentionally the agent runtime rather than the user interface: model routing, the tool loop, permissions, sessions, context management, sandboxing, and MCP integration live here.

## Architecture

An ACP client launches Anvil as a subprocess and communicates over stdio using JSON-RPC. The client owns the experience; Anvil owns agent execution semantics.

```text
ACP client           stdio / JSON-RPC           Anvil
----------           ----------------           -----
editor        --------------------------------> agent loop
issue bot     --------------------------------> model routing
custom TUI    --------------------------------> permissions
automation    --------------------------------> tools + sessions
```

## Status

These documentation pages are an initial structure. For the complete current reference, see the [project README](https://github.com/BrokkAi/anvil#readme).
