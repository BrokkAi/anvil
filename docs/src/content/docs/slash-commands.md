---
title: Slash Commands
description: Built-in session commands and dynamically discovered command sources.
---

Slash commands are submitted through the active ACP session. Availability can depend on the client and installed extensions.

| Command | Purpose |
| --- | --- |
| `/setup` | Configure providers, behavior, sandbox defaults, recaps, timeouts, and advanced settings. |
| `/fast [on\|off\|status]` | Set or inspect the fast Codex service tier for the current session. |
| `/permissions` | List, revoke, or clear remembered approvals. |
| `/context` | Show the current prompt shape, retained snapshot, and token estimate. |
| `/compact` | Summarize older uncompressed turns without deleting the raw log. |
| `/rewind` | Remove the latest completed conversation turn. |
| `/usage` | Show session token totals, USD cost, and available OpenRouter credit balance. |
| `/loop <seconds> <command-or-prompt>` | Repeat work until the session is cancelled. Nested loops are rejected. |
| `/goal [--max-turns N] <objective>` | Continue across turns until verified completion, a repeated blocker, cancellation, or an optional ceiling. |
| `/mcp` | List and configure stdio MCP servers. |
| `/plugin` | Install and manage Claude Code-format plugins. |
| `/pr-create [title]` | Create a GitHub pull request from the current branch. |

:::caution
`/pr-create` is deliberately mutating. If the worktree is dirty, it stages all changes and commits them before opening the pull request. Review the exact diff and branch first.
:::

`/goal` has no turn ceiling by default. It retries transient model or network outages with capped backoff and stops when the model verifies the objective, reports the same blocker for three consecutive turns, encounters a fatal model request, is cancelled, or reaches an explicit `--max-turns` limit.

Skills and plugin command Markdown files also become slash commands. Built-ins win name collisions; user and project definitions take precedence over plugin-provided entries. See [Skills and Plugins](../skills-plugins/).

Use `/setup timeout`, `/setup timeout <seconds>`, or `/setup timeout default` for the current session's in-memory stream-timeout override. The compatibility command `/idle-timeout` is accepted but intentionally omitted from autocomplete.
