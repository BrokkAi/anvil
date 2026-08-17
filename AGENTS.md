# AGENTS.md

Guidance for AI agents working in this repository.

## Build & test

```bash
cargo build --release          # default build: compiles sandbox crate to wasm32-wasip2 + host binary
cargo test
cargo fmt
cargo clippy --all-targets -- -D warnings
```

Prerequisite: `rustup target add wasm32-wasip2` (needed by `build.rs` when the default `wasm-sandbox` feature is enabled).

## Key conventions

- **Logging goes to stderr only.** stdout is reserved for JSON-RPC. Use `tracing::info!`/`warn!`/`debug!`; default filter is `info`, overridable via `RUST_LOG`.
- **Wire IDs.** Models are tagged `<source>::<id>` (e.g. `codex::gpt-5-codex`, `ollama::llama3:latest`). The double-colon avoids collisions with Ollama tags (`:`) and OpenRouter ids (`/`). Parse with `split_wire_id`.
- **Error handling.** `anyhow::Result` throughout. Discovery failures (dead Ollama, missing auth.json) are logged and treated as "no models from this source" — never fatal.
- **`block_task()`** (ACP `send_request`) must only be called inside `cx.spawn()` — never from a request handler. `SpawnedCx<'_>` encodes this requirement.
- **Session zip reads/writes** all route through `SandboxBackend` to prevent untrusted archives from OOM-ing or panicking the host. Writes use atomic temp-then-rename.
- **MCP stdio calls are demultiplexed.** In `src/mcp.rs`, one `StdioConn` per subprocess owns a reader task that is the sole reader of the child's stdout and routes each response to a per-request waiter by JSON-RPC id. `call_tool_with_timeout` must take `McpClient::state` only to check liveness, respawn, and clone the `Arc<StdioConn>` — never across the response wait, or the parallel tool batches from `execute_parallel_safe_calls` serialize client-side again. Writers hold `StdioConn::writer` only for one serialize-and-flush. Register the waiter before writing the request; deregister on timeout or cancellation.
- **Permission gate** logic lives in `pure_gate_decision()` (unit-testable without a live ACP connection). Four modes: `default`, `acceptEdits`, `readOnly`, `bypassPermissions`. `runShellCommand` always re-prompts regardless of "Always allow" sticky approval.
- **ACP session config options are client-owned.** Add `SessionConfigOption` ids only for client-visible per-call controls such as behavior mode, permission mode, model selection, or reasoning effort. Anvil must not persist them to `SetupState` or session manifests; clients resubmit the desired values for each session. Do not put host/UI/install-wide preferences in `all_config_options`, `CONFIGURE_KNOWN_KEYS`, or `apply_config_option`; use `/setup`, `SetupState`, or a dedicated config file instead.
- **Path validation**: `safe_resolve` (reads) and `safe_resolve_for_write` (writes) ensure filesystem operations stay within the session's `cwd`. The write variant walks up to the first existing ancestor, canonicalizes it, and rejects `..` in the missing tail.
- **Context compaction** lives in `src/context_manager.rs`. `compact_history` replaces the older dynamic model-history prefix with a cumulative `<state_snapshot>`, retains a recent exact tail, and pins the current `update_plan` value. The canonical system/AGENTS/skills prefix is never summarized. Oversized compactor input is chunked and reduced before the final snapshot pass. Checkpoints persist through the Anvil-only `anvilCompactionContentId` task field; raw turns and legacy `summaryContentId` data remain untouched for ACP/Brokk replay and rewind. Automatic compaction may run before a new user turn or between completed tool exchanges.
- **Every model-visible tool call writes one `tool_timing` trace record.** `execute_tool` writes it for registry-dispatched tools. A tool the loop intercepts before `execute_tool` must write its own: today `semantic_search` (in `rerank_semantic_search`), `update_plan` (in `execute_update_plan`), and `task` (in `execute_subagent`). Put the record inside the handler, not at the dispatch site, so a second dispatch path cannot lose it. Trace consumers count calls per tool from these records; a missing one reads as "the tool was never used", which is how 64 `semantic_search` calls in the r26 CodeScaleBench arm were reported as zero.
- **Semantic index prehydration is not tool time.** Bifrost hydrates its semantic index asynchronously and blocks the first call that needs it (bifrost AGENTS.md, "Index readiness design"). `semantic_readiness::wait_for_semantic_index` waits for the index after MCP connect and before the first turn, and records the wait as `semantic_index_prehydration`. Do not fold that wait into a `tool_timing` duration or nest it under a `retrieval_timings` key. **The probe must name its workspace.** When the session configures analysis workspaces, anvil starts bifrost as `--workspace <name>=<path>` and bifrost's router rejects any call without a configured `workspace` argument, so the wait probes each configured name and writes one record per workspace. A nameless probe in that mode returns `invalid_params` on the first poll; reading that as "nothing to wait for" is what let firefox charge 500-724 s of hydration to its first `semantic_search` in the 2026-08-16 remeasure. Every probe failure is logged at `warn`.
- **Open question: batch starts under-count issued semantic calls.** In the r26 arm the eval branch's `semantic_search_batch` starts (55) trail the `semantic_search` calls the model issued (64) across the same traces. The 9-call gap is unexplained -- calls rejected before the batch opens, or an emission path that skips the event. Not chased yet; do not treat batch-start counts as a call census until it is.
- **Lint suppressions**: do not add `#[allow(...)]` to get around linting. Prefer refactoring the code so the lint passes; if a suppression is truly necessary, document the invariant or external constraint that makes it safe.

## Release workflow

Releases are driven by a `vX.Y.Z` tag on master. The tag fans out to three
workflows at once (GitHub Release, Publish crate, Docs), so **everything below
must be true before the tag is pushed** — a red master or a stale generated
file turns into a failed publication, not just a failed check.

