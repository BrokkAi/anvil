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

On Linux, `bubblewrap` (`bwrap`) must be installed for `runShellCommand` OS-level sandboxing.

## Key conventions

- **Logging goes to stderr only.** stdout is reserved for JSON-RPC. Use `tracing::info!`/`warn!`/`debug!`; default filter is `info`, overridable via `RUST_LOG`.
- **Wire IDs.** Models are tagged `<source>::<id>` (e.g. `codex::gpt-5-codex`, `ollama::llama3:latest`). The double-colon avoids collisions with Ollama tags (`:`) and OpenRouter ids (`/`). Parse with `split_wire_id`.
- **Error handling.** `anyhow::Result` throughout. Discovery failures (dead Ollama, missing auth.json) are logged and treated as "no models from this source" — never fatal.
- **`block_task()`** (ACP `send_request`) must only be called inside `cx.spawn()` — never from a request handler. `SpawnedCx<'_>` encodes this requirement.
- **Wasm sandbox calls** use `std::thread::scope` to escape the tokio runtime (wasmtime-wasi 27's sync API calls `tokio::Handle::block_on`, which panics if already on a tokio thread).
- **Session zip reads/writes** all route through `SandboxBackend` to prevent untrusted archives from OOM-ing or panicking the host. Writes use atomic temp-then-rename.
- **Permission gate** logic lives in `pure_gate_decision()` (unit-testable without a live ACP connection). Four modes: `default`, `acceptEdits`, `readOnly`, `bypassPermissions`. `runShellCommand` always re-prompts regardless of "Always allow" sticky approval.
- **ACP session config options are client-owned.** Add `SessionConfigOption` ids only for client-visible per-call controls such as behavior mode, permission mode, model selection, or reasoning effort. Anvil must not persist them to `SetupState` or session manifests; clients resubmit the desired values for each session. Do not put host/UI/install-wide preferences in `all_config_options`, `CONFIGURE_KNOWN_KEYS`, or `apply_config_option`; use `/setup`, `SetupState`, or a dedicated config file instead.
- **Path validation**: `safe_resolve` (reads) and `safe_resolve_for_write` (writes) ensure filesystem operations stay within the session's `cwd`. The write variant walks up to the first existing ancestor, canonicalizes it, and rejects `..` in the missing tail.
- **Context compression** lives in `src/context_manager.rs`. The single public entry point `summarize_turn` fast-paths when the turn fits the summarizer budget (`SUMMARIZER_INPUT_FRACTION = 0.65 * context_length`) and otherwise atomizes → packs into chunks → summarizes each chunk → meta-summarizes the joined output, recursing if the meta input still overruns. Per-turn summaries persist via the existing `summaryContentId` slot in `contexts.jsonl` (Brokk-compatible — same field the Java side reads). `build_prompt_messages` substitutes a single `<conversation_summary>` user message for any turn whose `ConversationTurn.summary` is `Some`. The verbatim log stays on disk regardless, so a reload reproduces the same compressed prompt.
- **Lint suppressions**: do not add `#[allow(...)]` to get around linting. Prefer refactoring the code so the lint passes; if a suppression is truly necessary, document the invariant or external constraint that makes it safe.

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
