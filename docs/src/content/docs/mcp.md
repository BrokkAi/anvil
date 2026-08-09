---
title: MCP Servers
description: Add local Model Context Protocol servers to an Anvil session.
---

Anvil starts configured MCP servers as child processes and adds their tools to the model-facing catalog. MCP processes can read whatever their command, arguments, environment, and operating-system permissions allow, so treat each server as trusted local software.

## Manage servers from a session

```text
/mcp list
/mcp add [--framing content-length|line] <name> <command> [args...]
/mcp enable <name>
/mcp disable <name>
/mcp remove <name>
/mcp reset
```

New servers use standard `Content-Length` framing. Select `--framing line` only when the server speaks newline-delimited JSON. Shell-style quoting preserves spaces in commands and arguments, and `{cwd}` expands to the session workspace root.

Configuration changes take effect on the next tool-capable prompt. Start a fresh session when you need to prove the complete startup path rather than reuse an already-running subprocess.

## Transport support

Servers added with Anvil's `/mcp add` command are local stdio processes. An ACP client may also supply stdio, streamable HTTP, or legacy SSE servers during `session/new`; Anvil advertises and implements all three lifecycle transports. HTTP and SSE entries can include the headers defined by ACP.

Remote MCP sends tool schemas, arguments, and results across another network boundary. Confirm the endpoint and credential scope before attaching it to a sensitive session.

## Managed Bifrost

Bifrost is installed as Anvil's default code-intelligence MCP server and is equivalent to:

```text
<managed-bifrost> --root {cwd} --mcp core --no-line-numbers
```

Anvil downloads the pinned, checksum-verified Bifrost release on first use. See [Tools and Managed Bifrost](../tools-bifrost/) for its tool contract and evidence boundaries.

## Operational checks

- Use `/mcp list` to confirm that a server is enabled.
- Inspect Anvil's stderr logs for process startup or framing errors; stdout is reserved for ACP JSON-RPC.
- If tools are absent after a change, submit another prompt or open a fresh session.
- Keep secrets in the server environment or its own credential store. Avoid pasting them into chat, where they become part of session history.
