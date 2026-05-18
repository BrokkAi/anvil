# AGENTS.md

Guidance for AI agents working in this repository.

## Build & test

```bash
cargo build --release          # compiles sandbox crate to wasm32-wasip2 + host binary
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Prerequisite: `rustup target add wasm32-wasip2` (needed by `build.rs`).

On Linux, `bubblewrap` (`bwrap`) must be installed for `runShellCommand` OS-level sandboxing.

## Key conventions

- **Logging goes to stderr only.** stdout is reserved for JSON-RPC. Use `tracing::info!`/`warn!`/`debug!`; default filter is `info`, overridable via `RUST_LOG`.
- **Wire IDs.** Models are tagged `<source>::<id>` (e.g. `codex::gpt-5-codex`, `ollama::llama3:latest`). The double-colon avoids collisions with Ollama tags (`:`) and OpenRouter ids (`/`). Parse with `split_wire_id`.
- **Error handling.** `anyhow::Result` throughout. Discovery failures (dead Ollama, missing auth.json) are logged and treated as "no models from this source" — never fatal.
- **`block_task()`** (ACP `send_request`) must only be called inside `cx.spawn()` — never from a request handler. `SpawnedCx<'_>` encodes this requirement.
- **Wasm sandbox calls** use `std::thread::scope` to escape the tokio runtime (wasmtime-wasi 27's sync API calls `tokio::Handle::block_on`, which panics if already on a tokio thread).
- **Session zip reads/writes** all route through `SandboxBackend` to prevent untrusted archives from OOM-ing or panicking the host. Writes use atomic temp-then-rename.
- **Permission gate** logic lives in `pure_gate_decision()` (unit-testable without a live ACP connection). Four modes: `default`, `acceptEdits`, `readOnly`, `bypassPermissions`. `runShellCommand` always re-prompts regardless of "Always allow" sticky approval.
- **Path validation**: `safe_resolve` (reads) and `safe_resolve_for_write` (writes) ensure filesystem operations stay within the session's `cwd`. The write variant walks up to the first existing ancestor, canonicalizes it, and rejects `..` in the missing tail.

## Adding a new built-in tool

1. Add a `ToolMeta` row in `src/tools/mod.rs` (`TOOLS` constant).
2. Add the dispatch arm in `ToolRegistry::execute()`.
3. Add the tool definition in `ToolRegistry::tool_definitions()`.
4. Add the name to `BUILTIN_TOOL_NAMES` (test constant).
5. Add a title builder in `src/tool_loop/announce.rs` if the card should show path/command/pattern.
6. Run `cargo test` — the anti-drift test `builtin_tools_have_metadata_and_are_advertised` will catch missing metadata.

## Adding a new LLM backend

1. Implement `LlmBackend` (trait in `src/llm_client.rs`).
2. Wire it into `src/main.rs` (like `build_codex_backend` / `build_ollama_backend`).
3. Add a `ModelSource` variant in `src/discovery.rs` and update `split_wire_id`.
4. Register it in `MultiBackend::new`.
