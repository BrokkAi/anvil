---
title: Subagent Concurrency
description: Ordering, isolation, cancellation, observability, and result guarantees for delegated work.
---

Anvil executes mutations deterministically and can batch independent reads. The same rules cover built-in, MCP, and delegated `task` calls.

## Execution order

When a model emits multiple tool calls in one step, Anvil divides them by concurrency safety:

1. Mutating, promptable, or otherwise unsafe calls run serially in their original order.
2. Concurrency-safe calls then run in a bounded batch of at most six.

Running mutations first ensures that reads in the same step observe the updated workspace. Results are still appended in deterministic dispatch order, not completion order.

Built-in tool metadata marks read-only file, search, and web operations plus explicitly classified Bifrost tools as concurrency-safe. `update_plan` and unknown MCP tools remain serial; MCP `readOnlyHint` annotations are not currently used to opt a tool into batching. A call without a trusted static classification remains serial.

## Subagent lanes

The dynamic `task` tool follows the same batching path:

- `permission_mode: readOnly` is concurrency-safe and is the default when the field is omitted.
- Read-only lanes in the same model step can join the bounded safe batch even when unsafe calls appeared between them originally; unsafe calls are moved ahead and completed first.
- `permission_mode: inherit` remains serial because it may use the parent's mutating or promptable authority.

Each lane receives a fresh transcript but shares the parent workspace, session id, tool registry, MCP processes, and permission gate. Delegation depth is capped at one. The subagent's intermediate tokens and tool calls are silent; only its final text is returned as the parent `task` result.

Read-only lanes are appropriate for review, exploration, triage, log analysis, and summarization. They do not get separate worktrees or operating-system containers.

## Cancellation and limits

One cancellation token is shared by the prompt, its tool calls, and all subagents. `session/cancel`, `session/close`, and `session/delete` signal the whole in-flight set; operations stop cooperatively at their next cancellation checkpoint. There is no per-lane cancellation API.

| Control | Default | Scope |
| --- | --- | --- |
| `--max-turns` | `0` (unbounded) | Agent-loop turns per prompt; subagents inherit it unless their definition sets a lower value. |
| `--llm-idle-timeout-secs` | `300` | Wait for first meaningful LLM stream progress. |
| `--llm-stall-timeout-secs` | `60` | Wait between meaningful chunks after progress begins. |
| Shell timeout | `120s` | One shell call; `timeout_seconds` is clamped to 10-3600 seconds. Lower the ceiling with `ANVIL_SHELL_TIMEOUT_CAP_SECONDS`. |

These controls are not a universal wall-clock deadline. A client or bot that needs an overall deadline must send `session/cancel` when it expires.

## Observability and results

Parent-level tool calls emit pending, in-progress, and completed or failed `session/update` states. A subagent is visible as one `task` card; its internal stream is deliberately not forwarded to the client.

Per-lane results are plain text. A structured-output request applies to the top-level prompt, so the parent or client must aggregate lane text into the requested schema. Parallel results are recorded in deterministic dispatch order.

Set `ANVIL_TRACE_JSONL=<path>` for local LLM and step diagnostics. Trace files can contain sensitive prompt and project data; protect them accordingly.

## Current non-guarantees

- no parallel mutating or inherited subagent lanes;
- no per-lane cancellation, elapsed-time deadline, or token allowance;
- no per-call dynamic tool allowlist beyond an agent definition's static `tools` frontmatter;
- no ACP stream of a subagent's internal steps; and
- no separate filesystem or process isolation between lanes.

These boundaries are part of Anvil's current public behavior. Consumers should not infer stronger isolation or scheduling guarantees from calls that happen to finish concurrently.
