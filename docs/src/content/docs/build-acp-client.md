---
title: Build an ACP Client and Example Bots
description: Integrate a custom ACP v1 client with Anvil over stdio JSON-RPC.
---

Anvil is an ACP v1 agent process. A custom editor, bot, or orchestrator launches `anvil`, speaks newline-delimited JSON-RPC over stdin/stdout, renders streamed session updates, and owns the user-facing controls. Each protocol message occupies one JSON line.

## Process contract

1. Launch the absolute `anvil` executable with the workspace as the intended session directory.
2. Treat stdout as protocol-only JSON-RPC. Capture stderr separately for logs and diagnosis.
3. Send ACP initialization with protocol version `1` and the capabilities the client genuinely supports.
4. Create a session for an absolute workspace path.
5. Submit prompts and consume `session/update` notifications until the prompt response supplies a stop reason.

Do not parse stderr as protocol output, and do not allow logging from wrappers to leak onto stdout.

| Direction | ACP method or notification | Client responsibility |
| --- | --- | --- |
| Client → Anvil | `initialize` | Negotiate v1 and advertise real capabilities. |
| Client → Anvil | `session/new` | Supply an absolute workspace and optional MCP servers. |
| Client → Anvil | `session/set_config_option` | Send each live model, reasoning, behavior, service-tier, or permission selection. |
| Client → Anvil | `session/prompt` | Submit content and await its stop reason. |
| Anvil → Client | `session/update` | Render message chunks, plans, tool lifecycles, usage, and session information. |
| Anvil → Client | `session/request_permission` | Present the options and return the user's exact decision. |
| Client → Anvil | `session/cancel` | Cooperatively stop the active prompt and its delegated work. |

## Capabilities and configuration

Advertise only filesystem, terminal, and elicitation features you can service. Independently, implement ACP permission requests and cancellation. Anvil may use the advertised capabilities to request supported client interactions.

The client owns live session selections such as:

- `model_selection`
- `reasoning_effort`
- `service_tier`
- `behavior_mode`
- `permission_mode`

Send the desired values for each new or loaded session. Anvil does not persist these client-owned selections as installation preferences, so a reconnecting client must resubmit them.

## Streamed interaction

Render text and thought chunks incrementally and maintain tool-call cards from their pending, in-progress, completed, or failed updates. When Anvil requests permission, show the choices and return the user's decision without silently broadening it. A remembered approval is stored in repository-local permission state; outside-sandbox escalation is never remembered.

Use `session/cancel` to cancel the active prompt. Cancellation is cooperative and shared by the parent turn and its in-flight subagents. A client-side overall deadline should send cancellation because arbitrary tools do not have a universal wall-clock deadline.

## Load, resume, and close

Anvil stores sessions locally. `session/load` replays stored user, agent, and tool updates so a client can reconstruct its transcript. `session/resume` reopens the same durable session without replay, for a client that already has the UI history. Both paths validate the original workspace, accept a fresh lifecycle MCP set, and return current modes/config options.

After either path, restore client UI state as needed and resubmit the desired client-owned session selections. Closing or deleting a session cancels active work before resources are released. Raw conversation history is retained even when compacted summaries change what is sent to the model.

## MCP from a client

An ACP client may attach stdio, streamable HTTP, or legacy SSE MCP servers in new, load, resume, and fork lifecycle requests. Load and resume clients should resupply their complete client-owned MCP set; an empty set clears that in-memory layer, while global `/mcp` entries and managed Bifrost are merged separately. A fork with no supplied set inherits the source session's lifecycle servers.

Send only transports the client has actually configured, including the required URL, headers, command, arguments, or environment. See [MCP Servers](../mcp/).

## Minimal acceptance test

A custom client is ready for practical evaluation when it can:

1. initialize and create a session;
2. select a model and permission mode;
3. display streamed text and tool lifecycle updates;
4. service an edit permission request;
5. cancel an active prompt;
6. load or resume a session and resubmit its live configuration.

The checked-in fixture and success criteria in [Ten-Minute Evaluation](../evaluate-anvil/) can serve as an end-to-end example bot scenario.

## Checked-in Rust bots

Build the debug agent once, then run the examples with literal input so the first pass is read-only and does not depend on GitHub:

```bash
cargo build --bin anvil

cargo run --example issue_bot -- --repo BrokkAi/anvil \
  --issue-text "Users report that /setup timeout rejects 300 seconds."

cargo run --example review_bot -- --repo BrokkAi/anvil \
  --diff-text "diff --git a/README.md b/README.md"

cargo run --example issue_writer_tui -- --repo BrokkAi/anvil --dry-run \
  --prompt "The docs should explain ACP client examples."
```

The examples find the agent through `ANVIL_AGENT`, then `target/debug/anvil`, then `cargo run --quiet --bin anvil --`. Override it explicitly with `--agent "./target/debug/anvil"`.

`issue_bot` can fetch an issue and `review_bot` can fetch a pull request through `gh`; their `--post-comment` modes write to GitHub. `issue_writer_tui` calls `gh issue create` only after confirmation, while `--dry-run` stops at the draft. These programs demonstrate the integration shape, not a production bot framework.
