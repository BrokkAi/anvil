# Concurrency model for delegated work

This document defines the concurrency contract Anvil exposes to ACP consumers
(for example SlopCop's audit-lane orchestration), per issue
[#53](https://github.com/BrokkAi/anvil/issues/53). It describes the model as it
is **actually implemented today**, so consumers can build orchestration policy
on a designed contract rather than accidental behavior. Where a capability does
not exist yet, that is stated explicitly along with the issue tracking it.

## TL;DR

- **Execution is sequential, not parallel.** Within a turn, multiple tool calls
  run one at a time, in order. A delegated "lane" is an **explicit subagent**
  (the `task` tool), and subagents run **inline and serially** inside the
  parent's tool loop — there is no fan-out across lanes today.
- **One cancellation token per prompt** is shared by the parent turn, every tool
  call, and every subagent. `session/cancel` aborts all in-flight delegated
  work atomically.
- **Bounded runtime** comes by default from the model's own completion signal,
  the LLM stream timeouts (`--llm-idle-timeout-secs` and
  `--llm-stall-timeout-secs`), and per-shell-command
  timeouts; an optional per-prompt turn ceiling (`--max-turns N`) can be set to
  bound cost/time. There is no separate per-tool-call wall-clock timeout.
- **Observability** for the parent's own tool calls is rich
  (`Pending → InProgress → Completed/Failed` via `session/update`), but a
  subagent's *internal* steps are intentionally **not** surfaced to the client.

## 1. What "delegated work" means in Anvil

There are two distinct mechanisms, and they have different guarantees:

1. **Tool calls within one turn.** When the model emits several tool calls in a
   single assistant message, those are the unit of work for that turn.
2. **Subagents (`task` tool).** The model can delegate a self-contained
   sub-task to a named subagent. This is the "delegated lane" an audit-style
   workflow would use to get role-separated analysis.

Both run on the **same session**, on the **same thread of execution**, driven by
the same tool loop (`src/tool_loop.rs`).

## 2. Concurrency model: sequential by design (today)

### Tool calls within a turn — sequential

Multiple tool calls in one turn are dispatched in a `for` loop
(`execute_step_tool_calls`, `src/tool_loop.rs`): each call is awaited to
completion before the next begins. There is **no parallel dispatch** (no
`join_all`, no task fan-out).

Ordering is deterministic and slightly reshaped: built-in/non-Bifrost tools are
ordered **before** Bifrost tools within a step (to avoid analyzer-context
shadowing), and relative order is otherwise preserved. Results are appended in
execution order, so there is no result-ordering ambiguity.

### Subagents — serial, inline, depth-limited

The `task` tool runs a subagent **synchronously inline** (`Box::pin(run(...))
.await`) as one step of the parent's tool loop. Consequences:

- If a turn emits several `task` calls, they execute **one after another**, not
  concurrently — there is no lane-level parallelism.
- **Isolation:** each subagent gets a **fresh conversation** (its own
  system + user prompt) but **shares** the parent's tool registry, working
  directory, and session id. Its streamed tokens and thoughts are discarded.
- **Nesting depth is capped at 1:** a subagent cannot spawn its own subagent.
  At max depth the `task` tool is stripped from the catalog.
- **Result:** the subagent's final assistant text is returned verbatim as the
  `task` tool's result to the parent.

> **Contract for consumers:** treat delegated lanes as **logically isolated but
> physically serialized**. You get role separation and conversation isolation,
> but you do **not** get wall-clock parallelism or reduced latency from running
> lanes "at the same time." Design prompts/policy for correctness under serial
> execution; do not assume two lanes make progress simultaneously.

## 3. Ordering and isolation guarantees

- **Ordering:** tool calls and subagents complete in a deterministic order
  (dispatch order, with built-ins before Bifrost tools). Because execution is
  serial, a later lane observes the side effects (e.g. filesystem writes) of an
  earlier one in the same turn.
- **Conversation isolation:** a subagent never sees the parent's message history
  and vice-versa; only the final text crosses the boundary.
- **Shared state:** subagents share the session's tool registry (and therefore
  the same Bifrost/MCP subprocesses), cwd, and permission scope. They are **not**
  sandboxed into separate workspaces. Lanes that mutate shared state do so
  against the same workspace, in series.

## 4. Cancellation and timeouts

### Cancellation

Each prompt gets a single `CancellationToken` (`start_prompt`/`cancel_prompt`,
`src/session.rs`). That **same token is cloned into every tool call and every
subagent**. Therefore:

- `session/cancel` cancels the token, which aborts the parent turn, the current
  tool call, and any in-flight subagent — all of them, atomically.
- The tool loop checks the token between steps and between tool calls, so
  cancellation takes effect promptly at the next checkpoint rather than mid-LLM-
  stream only.
- `session/close` and `session/delete` also cancel the token before tearing
  down resources.
- The prompt then resolves with `StopReason::Cancelled` (see ACP prompt
  handling), so a consumer can distinguish a cancelled lane-set from a completed
  one.

> **Contract for consumers:** cancellation is **all-or-nothing per prompt**.
> There is no API to cancel a single lane while letting its siblings continue —
> cancelling cancels the whole delegated set for that prompt.

### Timeouts and bounded runtime

| Knob | Default | Scope | Notes |
|---|---|---|---|
| `--max-turns` | `0` (unbounded) | per prompt | Optional cap on agent turns. Default unbounded: the loop exits when the model answers without a tool call; idle-timeout + no-progress nudges catch stalls. Pass `N>0` to deliberately bound cost/time. Subagents inherit the parent budget unless their definition sets a lower `max_turns:`; recursion is bounded by depth (`MAX_SUBAGENT_DEPTH = 1`). |
| `--llm-idle-timeout-secs` | 300 | per LLM stream | Aborts a stream when no first *meaningful* progress arrives before this timeout. Useful for reasoning models that may take a while before their first streamed chunk. |
| `--llm-stall-timeout-secs` | 60 | per LLM stream | After first progress, aborts a stream when the gap between meaningful chunks exceeds this timeout. Keepalives do not count. |
| Shell command timeout | 60s | per `run_shell_command` | Optional `timeout` values are milliseconds, rounded up to seconds, and capped at 600s. Tool output reports when a request is clamped. |

There is **no separate per-tool-call wall-clock timeout** beyond the LLM idle
timeout and (when set) the turn ceiling. With the default unbounded
`--max-turns`, a long-running non-shell tool is bounded only by cancellation and
the LLM first-progress and stall timeouts; set a positive `--max-turns` to add
a turn-count ceiling. The per-session `/idle-timeout N` override sets both LLM
stream timeout phases to `N`, preserving its historical "tolerate gaps up to N"
meaning.

> **Contract for consumers:** the turn loop terminates on the model's own
> completion signal by default. To impose a hard cost ceiling on an unattended
> or delegated workflow, set a positive `--max-turns` (total agent turns) and
> `--llm-idle-timeout-secs` and `--llm-stall-timeout-secs` (stream stall
> detection). Do not rely on a per-lane
> deadline — model an overall budget instead, and cancel if it is exceeded.

## 5. Observability

For the **parent** session's own tool calls, Anvil emits `session/update`
notifications with a full lifecycle:
`ToolCallStatus::Pending → InProgress → Completed | Failed`, plus the tool's
title, kind, and result content. A consumer can drive a live UI / queue view
from these.

A **subagent's internal** tool calls and tokens are intentionally **silent** —
they are not forwarded to the client. The client sees only the `task` tool's
own call card (pending → completed/failed) and its final text result. So today
there is **no per-lane "queued / running / completed" stream for sub-lanes**;
the lane is observable as a single tool call from the parent's perspective.

For deeper introspection, set `ANVIL_TRACE_JSONL=<path>` to capture a JSONL
trace of `llm_request` / `llm_response` / `llm_error` events (tagged with the
turn index) plus per-step records. This is the recommended channel for
debugging/observing delegated work that is not surfaced over ACP.

> **Contract for consumers:** rely on parent-level `session/update` tool-call
> status for the lanes you launch directly (one `task` call == one observable
> lane). For sub-lane detail, use the trace log, not ACP notifications.

## 6. Structured result collection

- **Structured output is parent-level only.** A `session/prompt` may request a
  validated structured output (JSON-schema) for the *top-level* turn. That
  request is **not** propagated into subagents — a subagent returns plain text,
  and the parent is responsible for assembling/validating any structured result
  from the lanes' text outputs.
- Because execution is serial, result aggregation is order-deterministic: lane
  outputs are collected in dispatch order with no interleaving.

> **Contract for consumers:** collect per-lane results as the **text** returned
> by each `task` call, and do the structured aggregation/validation at the
> orchestration layer (or in a final parent turn that requests structured
> output over the collected lane results).

## 7. What is intentionally *not* guaranteed yet

These are real limitations, not oversights — they are called out so consumers
do not build on behavior that does not exist:

- **No parallel lane execution.** Subagents and multi-tool-call steps are
  serial. Wall-clock parallel fan-out across lanes is not implemented.
- **No per-lane cancellation/timeout.** Cancellation and budgets are
  per-prompt, not per-lane.
- **No per-subagent tool allowlist** — subagents inherit the full parent tool
  catalog (tracked by [#29](https://github.com/BrokkAi/anvil/issues/29)).
- **No per-subagent token budget** beyond the turn cap
  ([#30](https://github.com/BrokkAi/anvil/issues/30)).
- **No nested-loop integration test coverage** for subagents
  ([#31](https://github.com/BrokkAi/anvil/issues/31)).
- **No sub-lane observability over ACP** — sub-lane steps are silent by design.

If/when parallel delegated execution is introduced, this document is the place
to define the added guarantees (max concurrency, per-lane isolation, per-lane
cancellation, and the observability surface for queued/running/completed
lanes).
