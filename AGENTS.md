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

**Run the full `cargo test` suite with localhost-bind permission / outside any restricted execution sandbox.** Wiremock-based tests start local HTTP servers; a sandbox that denies socket binds causes dozens of unrelated failures with `Failed to bind an OS port for a mock server` / `PermissionDenied: Operation not permitted`. Do not treat those as product failures and do not first run the full suite in a sandbox known to deny localhost binds. Targeted tests that do not use Wiremock may still run sandboxed.

On Linux, `bubblewrap` (`bwrap`) must be installed for `runShellCommand` OS-level sandboxing.

## Key conventions

- **Logging goes to stderr only.** stdout is reserved for JSON-RPC. Use `tracing::info!`/`warn!`/`debug!`; default filter is `info`, overridable via `RUST_LOG`.
- **Wire IDs.** Models are tagged `<source>::<id>` (e.g. `codex::gpt-5-codex`, `ollama::llama3:latest`). The double-colon avoids collisions with Ollama tags (`:`) and OpenRouter ids (`/`). Parse with `split_wire_id`.
- **Error handling.** `anyhow::Result` throughout. Discovery failures (dead Ollama, missing auth.json) are logged and treated as "no models from this source" — never fatal.
- **`block_task()`** (ACP `send_request`) must only be called inside `cx.spawn()` — never from a request handler. `SpawnedCx<'_>` encodes this requirement.
- **Wasm sandbox calls** use `std::thread::scope` to escape the tokio runtime (wasmtime-wasi 27's sync API calls `tokio::Handle::block_on`, which panics if already on a tokio thread).
- **Session zip reads/writes** all route through `SandboxBackend` to prevent untrusted archives from OOM-ing or panicking the host. Writes use atomic temp-then-rename.
- **MCP stdio calls are demultiplexed.** In `src/mcp.rs`, one `StdioConn` per subprocess owns a reader task that is the sole reader of the child's stdout and routes each response to a per-request waiter by JSON-RPC id. `call_tool_with_timeout` must take `McpClient::state` only to check liveness, respawn, and clone the `Arc<StdioConn>` — never across the response wait, or the parallel tool batches from `execute_parallel_safe_calls` serialize client-side again. Writers hold `StdioConn::writer` only for one serialize-and-flush. Register the waiter before writing the request; deregister on timeout or cancellation.
- **Permission gate** logic lives in `pure_gate_decision()` (unit-testable without a live ACP connection). Four modes: `default`, `acceptEdits`, `readOnly`, `bypassPermissions`. `runShellCommand` always re-prompts regardless of "Always allow" sticky approval.
- **ACP session config options are client-owned.** Add `SessionConfigOption` ids only for client-visible per-call controls such as behavior mode, permission mode, model selection, or reasoning effort. Anvil must not persist them to `SetupState` or session manifests; clients resubmit the desired values for each session. Do not put host/UI/install-wide preferences in `all_config_options`, `CONFIGURE_KNOWN_KEYS`, or `apply_config_option`; use `/setup`, `SetupState`, or a dedicated config file instead.
- **Path validation**: `safe_resolve` (reads) and `safe_resolve_for_write` (writes) ensure filesystem operations stay within the session's `cwd`. The write variant walks up to the first existing ancestor, canonicalizes it, and rejects `..` in the missing tail.
- **Context compaction** lives in `src/context_manager.rs`. `compact_history` replaces the older dynamic model-history prefix with a cumulative `<state_snapshot>`, retains a recent exact tail, and pins the current `update_plan` value. The canonical system/AGENTS/skills prefix is never summarized. Oversized compactor input is chunked and reduced before the final snapshot pass. Checkpoints persist through the Anvil-only `anvilCompactionContentId` task field; raw turns and legacy `summaryContentId` data remain untouched for ACP/Brokk replay and rewind. Automatic compaction may run before a new user turn or between completed tool exchanges.
- **Lint suppressions**: do not add `#[allow(...)]` to get around linting. Prefer refactoring the code so the lint passes; if a suppression is truly necessary, document the invariant or external constraint that makes it safe.

## Release workflow

Releases are driven by a `vX.Y.Z` tag on master. The tag fans out to three
workflows at once (GitHub Release, Publish crate, Docs), so **everything below
must be true before the tag is pushed** — a red master or a stale generated
file turns into a failed publication, not just a failed check.

### 1. Preflight — master must be green

Check the latest master CI run (`gh run list --workflow=ci.yml --branch master`),
including the Windows job and the `dependency licenses` job. Never tag on a red
master.

### 2. Version bump (one commit/PR, all of it together)

1. Bump `version` in the root `Cargo.toml`.
2. Refresh the lockfile: `cargo update --workspace --offline`.
3. **Regenerate the shipped license reports — they embed the crate version**,
   so every version bump changes them and CI diffs them against the committed
   files (pinned tools: `cargo-about` 0.9.1, `cargo-deny` 0.20.2, same as CI):

   ```bash
   cargo about generate --offline --config licenses/about.toml --locked --fail licenses/about.hbs -o licenses/THIRD_PARTY_LICENSES.html
   node scripts/generate-supplemental-third-party-notices.mjs licenses/SUPPLEMENTAL_THIRD_PARTY_NOTICES.txt
   cargo deny --config licenses/deny.toml --locked check licenses
   ```

4. Update version examples in the docs
   (`docs/src/content/docs/install.md`, `npm/README.md`,
   `npm/launcher/README.md`, `npm/build-npm-packages.mjs` usage comments).
5. Verify crates.io publishability locally (add `--allow-dirty` while
   uncommitted):

   ```bash
   cargo publish --dry-run -p brokk-anvil-minimizer -p brokk-anvil --locked
   ```

   Caveat: the root `build.rs` runs a nested `cargo metadata` during publish
   verification, and that nested call resolves dependencies from crates.io —
   not from the local workspace. Every workspace dependency of `brokk-anvil`
   (currently `brokk-anvil-minimizer`) must therefore already be published at
   the exact pinned version. If the minimizer changed, bump its version in
   `crates/anvil-minimizer/Cargo.toml` **and** in the root `Cargo.toml`
   dependency line; the publish workflow publishes it before `brokk-anvil`.
   The `anvil-minimizer` dependency is dependency-renamed: the package on
   crates.io is `brokk-anvil-minimizer`, imports stay `anvil_minimizer::`.
6. `cargo clippy --all-targets -- -D warnings` — note that Windows CI compiles
   test targets too: a `#[cfg(test)]` helper whose callers are all inside a
   `#[cfg(all(test, unix))]` module is dead code on Windows and fails the
   build there even when local (unix) clippy is clean. Gate helpers exactly as
   their callers are gated.
7. Merge to master and wait for CI to be fully green again.

### 3. Tag

```bash
git tag -a vX.Y.Z -m "Anvil X.Y.Z: <summary>" && git push origin vX.Y.Z
```

The tag must match the root `Cargo.toml` version (both tag workflows enforce
this). Pushing it triggers:

- **GitHub Release** — builds the five platform zips + `.sha256` sidecars and
  creates the release (~20 minutes).
- **Publish crate** — dry-run gate, then publishes `brokk-anvil-minimizer`
  (only when its version is new) and `brokk-anvil` to crates.io. Retry-safe:
  already-published versions are skipped.
- **Docs** — deploys the website.

If a tag workflow fails: fix master first. Delete and re-push the tag only if
the GitHub release object was never created; otherwise re-run via each
workflow's `workflow_dispatch` (Publish crate has a `publish` input).

### 4. npm (after the GitHub release exists)

Run the manual `publish-npm.yml` workflow with the tag — first with the
`publish` input unchecked (build + validate + smoke test only), then checked
to publish via npm trusted publishing. The six npm packages are built from the
checksum-verified GitHub release zips; the root `@brokkai/anvil` is only ever
published after all five platform packages are publicly visible. First-time
setup and the manual bootstrap runbook live in `npm/README.md`. The Homebrew
tap follows tagged releases automatically on a schedule.

## Adding a new built-in tool

1. Add a `ToolMeta` row in `src/tools/mod.rs` (`TOOLS` constant).
2. Add the dispatch arm in `ToolRegistry::execute()`.
3. Add the tool definition in `ToolRegistry::tool_definitions()`.
4. Add the name to `BUILTIN_TOOL_NAMES` (test constant).
5. Add a title builder in `src/tool_loop/announce.rs` if the card should show path/command/pattern.
6. Run `cargo test` — the anti-drift test `builtin_tools_have_metadata_and_are_advertised` will catch missing metadata.

## Subagents (`task` meta-tool)

- Discovery lives in `src/agents.rs` (`discover` / `AgentRegistry`). Layout is flat: one `<name>.md` per file under `.claude/agents/` or `.agents/agents/`. Last-wins ordering mirrors skills.
- The `task` tool is registered dynamically in `ToolRegistry::tool_definitions()` and only when the registry is non-empty; its `subagent_type` is enum-constrained to discovered names so the model cannot invent one.
- Dispatch is in `execute_subagent` in `src/tool_loop.rs`, **not** `ToolRegistry::execute` — it needs `llm`, `spawned_cx`, and `sessions`. It calls `run()` recursively (boxed) with `NotificationMode::Silent`, a fresh transcript, and no-op text/thought sinks.
- The nested run shares the parent's permission gate (same `session_id` + `sessions`), so `readOnly` and always-allow stickiness are inherited. Depth is capped by `MAX_SUBAGENT_DEPTH` (1); turns are **not** blanket-capped — a subagent inherits the parent's `max_turns` budget (`subagent_max_turns`) unless its own definition opts into a lower `max_turns:`.
- User-facing docs for authoring subagents live under "Subagents" in `README.md`. Keep them in sync when changing discovery roots, scope precedence, or limits.

## Adding a new LLM backend

1. Implement `LlmBackend` (trait in `src/llm_client.rs`).
2. Wire it into `src/main.rs` (like `build_codex_backend` / `build_ollama_backend`).
3. Add a `ModelSource` variant in `src/discovery.rs` and update `split_wire_id`.
4. Register it in `MultiBackend::new`.
