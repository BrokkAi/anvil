# Concurrency model for delegated work

This document defines the concurrency contract Anvil exposes to ACP consumers
(for example SlopCop's audit-lane orchestration), per issue
[#53](https://github.com/BrokkAi/anvil/issues/53). It describes the model as it
is **actually implemented today**, so consumers can build orchestration policy
on a designed contract rather than accidental behavior. Where a capability does
not exist yet, that is stated explicitly along with the issue tracking it.

## TL;DR

- **Execution is mostly sequential, with bounded read-only subagent fan-out.**
  Within a turn, ordinary tool calls run one at a time. A delegated "lane" is an
  **explicit subagent** (the `task` tool). Consecutive task lanes whose
  `permission_mode` is omitted/defaulted to `readOnly` or explicitly
  `readOnly` run concurrently, capped at six lanes; `inherit` lanes remain
  serial.
- **One cancellation token per prompt** is shared by the parent turn, every tool
  call, and every subagent. `session/cancel` signals all in-flight delegated
  work through that shared token; each operation stops at its next cancellation
  checkpoint.
- **Runtime controls are scoped, not an overall deadline.** LLM streams and
  shell commands have their own timeouts, and an optional per-prompt turn
  ceiling (`--max-turns N`) can limit agent-loop iterations. An arbitrary
  non-shell tool call and the prompt as a whole have no built-in wall-clock
  deadline; the consumer must cancel the prompt to enforce one.
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

Both run on the **same session** and are driven by the same tool loop
(`src/tool_loop.rs`). Read-only task lanes may be polled concurrently inside
that loop; other tools and inherited lanes are awaited serially.

## 2. Concurrency model: serial by default, read-only fan-out

### Tool calls within a turn — serial except read-only task batches

Multiple tool calls in one turn are dispatched in an ordered loop
(`execute_step_tool_calls`, `src/tool_loop.rs`): each call is awaited to
completion before the next begins, except for a consecutive batch of read-only
`task` calls.

Ordering is deterministic and slightly reshaped: built-in/non-Bifrost tools are
ordered **before** Bifrost tools within a step (to avoid analyzer-context
shadowing), and relative order is otherwise preserved. Results are appended in
deterministic dispatch order, not completion order, so there is no
result-ordering ambiguity.

### Subagents — isolated, depth-limited, read-only lanes parallel

The `task` tool runs a nested `run(...)` with a fresh transcript. Consequences:

- If a turn emits several consecutive `task` calls whose `permission_mode` is
  omitted/defaulted to `readOnly` or explicitly `readOnly`, Anvil runs those
  lanes concurrently with a cap of `MAX_PARALLEL_READ_ONLY_SUBAGENTS = 6`.
- `task` calls with `permission_mode: "inherit"` and any non-task calls remain
  serial. This keeps write-heavy/promptable workflows deterministic.
- **Isolation:** each subagent gets a **fresh conversation** (its own
  system + user prompt) but **shares** the parent's tool registry, working
  directory, and session id. Its streamed tokens and thoughts are discarded.
- **Nesting depth is capped at 1:** a subagent cannot spawn its own subagent.
  At max depth the `task` tool is stripped from the catalog.
- **Result:** the subagent's final assistant text is returned verbatim as the
  `task` tool's result to the parent.

> **Contract for consumers:** use read-only task lanes for parallel review,
> exploration, triage, log/test analysis, and summarization. Use inherited lanes
> for implementation/fixes, and assume those run serially.

## 3. Ordering and isolation guarantees

- **Ordering:** tool results are appended in deterministic dispatch order
  (with built-ins before Bifrost tools), even when read-only task lanes finish
  out of order internally.
- **Conversation isolation:** a subagent never sees the parent's message history
  and vice-versa; only the final text crosses the boundary.
- **Shared state:** subagents share the session's tool registry (and therefore
  the same Bifrost/MCP subprocesses), cwd, and permission scope. They are **not**
  sandboxed into separate workspaces. Read-only lanes are prevented from edits
  and shell execution by the lane-local permission override. Inherited lanes
  that can mutate shared state run against the same workspace in series.

## 4. Cancellation and timeouts

### Cancellation

Each prompt gets a single `CancellationToken` (`start_prompt`/`cancel_prompt`,
`src/session.rs`). That **same token is cloned into every tool call and every
subagent**. Therefore:

- `session/cancel` cancels the shared token for the parent turn, the current tool
  call, and any in-flight subagents. Shutdown is cooperative: each operation
  stops when it next observes the cancellation signal.
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
> cancelling signals the whole delegated set through the prompt's shared token.

### Timeout scopes and runtime controls

| Knob | Default | Scope | Notes |
|---|---|---|---|
| `--max-turns` | `0` (unbounded) | per prompt | Optional cap on agent turns. Default unbounded: the loop exits when the model answers without a tool call; idle-timeout + no-progress nudges catch stalls. Pass `N>0` to deliberately bound cost/time. Subagents inherit the parent budget unless their definition sets a lower `max_turns:`; recursion is bounded by depth (`MAX_SUBAGENT_DEPTH = 1`). |
| `--llm-idle-timeout-secs` | 300 | per LLM stream | Aborts a stream when no first *meaningful* progress arrives before this timeout. Useful for reasoning models that may take a while before their first streamed chunk. |
| `--llm-stall-timeout-secs` | 60 | per LLM stream | After first progress, aborts a stream when the gap between meaningful chunks exceeds this timeout. Keepalives do not count. |
| Shell command timeout | 60s | per `run_shell_command` | Optional `timeout` values are milliseconds, rounded up to seconds, and capped at 600s. Tool output reports when a request is clamped. |

The LLM timeouts apply only while Anvil is waiting for an LLM stream; they do
not bound a tool that is already executing. Likewise, `--max-turns` limits the
number of agent-loop iterations, not the wall-clock duration of an individual
tool call. Shell commands have their own timeout, but a non-shell tool without
an internal timeout is bounded only by prompt cancellation. The per-session
`/idle-timeout N` override sets both LLM stream timeout phases to `N`, preserving
its historical "tolerate gaps up to N" meaning.

> **Contract for consumers:** the turn loop terminates on the model's own
> completion signal by default. To limit iterations for an unattended or
> delegated workflow, set a positive `--max-turns` (total agent turns), plus
> `--llm-idle-timeout-secs` and `--llm-stall-timeout-secs` for stream-stall
> detection. These controls do not create a wall-clock deadline for arbitrary
> tool execution or a token-denominated cost ceiling. Model an overall deadline
> in the consumer and cancel the prompt if it is exceeded.

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
- Result aggregation is order-deterministic: lane outputs are collected in
  dispatch order with no parent-message interleaving, even when read-only lanes
  run concurrently.

> **Contract for consumers:** collect per-lane results as the **text** returned
> by each `task` call, and do the structured aggregation/validation at the
> orchestration layer (or in a final parent turn that requests structured
> output over the collected lane results).

## 7. Compatibility commitment

For Anvil 1.x, consumers may rely on the concurrency, ordering, isolation,
cancellation, observability, and result-collection guarantees documented here.
Changes that invalidate one of these guarantees are compatibility changes and
must be called out explicitly in release notes. Additive capabilities may
extend the contract without changing the guarantees already documented.

## 8. What is intentionally *not* guaranteed yet

These are real limitations, not oversights — they are called out so consumers
do not build on behavior that does not exist:

- **No parallel mutating lane execution.** Only read-only `task` lanes fan out.
  Inherited/promptable lanes and ordinary tool calls remain serial.
- **No per-lane cancellation/timeout.** Cancellation and budgets are
  per-prompt, not per-lane.
- **No per-call runtime tool allowlist.** Subagents can narrow their catalog
  with `tools` frontmatter, but a single `task` call cannot dynamically choose
  a different tool list.
- **No token-denominated or wall-clock subagent budget.** A subagent can set a
  lower `max_turns` value in its frontmatter, but there is no per-subagent token
  allowance or elapsed-time deadline.
- **No sub-lane observability over ACP** — sub-lane steps are silent by design.

## 9. Validation coverage

The task schema, read-only batching, permission-mode handling, tool allowlists,
and turn-cap inheritance are covered by unit tests. Anvil does not currently
have a full ACP integration test that drives a real nested subagent loop end to
end. That gap is tracked as a validation limitation rather than a runtime
behavior consumers must account for; the historical decision to rely on the
surrounding unit coverage is recorded in closed issue
[#31](https://github.com/BrokkAi/anvil/issues/31).
