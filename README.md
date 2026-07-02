# Anvil

Anvil is a Rust Agent Client Protocol (ACP) server.

It is intentionally **just the ACP server**: the agent runtime, model routing,
tool loop, permission gate, session store, context compression, sandboxing, and
MCP tool bridge live here, while the UI can be anything that speaks ACP over
stdio.

That separation is the point. Instead of rebuilding an agent loop inside every
editor, bot, TUI, review workflow, or internal automation, you can run Anvil as
one reusable ACP subprocess and put a small client in front of it.

## Why Anvil

- **One agent engine, many frontends.** Use the same ACP server from Zed,
  JetBrains, a Rust TUI, a GitHub issue bot, a review bot, or your own internal
  automation.
- **No giant UX surface required.** ACP gives the client a clean protocol
  boundary: initialize, create or resume a session, send prompts, receive
  streamed updates, answer permission requests.
- **Model routing built in.** Anvil discovers Codex/ChatGPT credentials,
  Ollama, a local ds4-server (antirez/ds4), and OpenRouter, then exposes
  collision-free wire ids such as `codex::gpt-5-codex`,
  `ollama::llama3:latest`, `ds4::deepseek-v4-flash`, and
  `openrouter::anthropic/claude-sonnet-4.5`.
- **MCP-native extensibility.** Anvil manages stdio MCP servers per session.
  Bifrost is preinstalled as a pinned, Anvil-managed local MCP server for
  symbol search, cross-references, and structural analysis.
- **Real agent tooling.** Built-in file reads/writes, grep, directory listing,
  shell execution, MCP tools, subagents, and skill slash commands.
- **Designed for unattended and attended flows.** Clients can run read-only,
  ask before edits, auto-accept edits, or trust all tool calls. Permission
  prompts are protocol messages, not hardcoded UI.
- **Terminal attention hooks for interactive runs.** When `stderr` is a real
  terminal, Anvil rings the terminal bell when it opens a permission prompt and
  when a turn finishes. Set `BROKK_TERMINAL_NOTIFICATIONS=off` to disable it,
  or choose events with `prompt`, `turn-ended`, or `prompt,turn-ended`.
- **Persistent working memory.** Sessions are stored on disk, can be loaded or
  resumed, and support context reports plus automatic or manual compression.

## What It Is

Anvil is not an editor plugin, a web app, or a terminal UI. It is the portable
agent backend those things can share.

```text
ACP client              stdio / JSON-RPC              Anvil
----------              ----------------              -----
Zed        ----------------------------------------->  agent loop
JetBrains   --------------------------------------->  model routing
issue bot   --------------------------------------->  permission gate
review bot  --------------------------------------->  tools + MCP
custom TUI  --------------------------------------->  session store
```

The client owns the experience. Anvil owns the agent protocol and execution
semantics.

Core modules:

- `agent`: ACP JSON-RPC dispatch, session lifecycle, slash commands, and prompt
  handling.
- `tool_loop`: the agentic engine that streams LLM output, executes tools, and
  applies the permission gate.
- `llm_client`: OpenAI-compatible API calls, tool-calling SSE parsing, and idle
  timeouts.
- `session`: persisted session state, conversation history, model/mode
  selection, and context compression storage.
- `tools`: built-in filesystem, grep, edit, and shell tools.
- `mcp`: persisted MCP server configuration and stdio MCP subprocess
  lifecycles.

## Quick Start

Build the server:

```bash
rustup target add wasm32-wasip2
cargo build --bin anvil
```

That default build includes the embedded `wasm-sandbox` feature. For targets
where the nested `wasm32-wasip2` sandbox build is undesirable, use:

```bash
cargo build --no-default-features --bin anvil
```

Run it directly if you want to see the stdio server start:

```bash
./target/debug/anvil
```

Most users connect Anvil through an ACP client rather than running it directly.
The first session shows setup guidance; use `/setup` to choose a model provider,
log in, refresh model discovery, or adjust advanced settings. Change permission
mode through the session `Permission` selector.

## Editor Setup

The repo includes `xtask` helpers that build a release binary and write an
agent-server entry into your editor config:

```bash
# Zed: ~/.config/zed/settings.json
cargo xtask build-acp-for-zed

# JetBrains: ~/.jetbrains/acp.json
cargo xtask build-acp-for-jetbrains
```

Both commands run `cargo build --release --bin anvil` and point the editor at
`target/release/anvil`. Existing unrelated config entries are preserved.

Both subcommands accept `--config <path>` to override the editor config path.

Manual wiring is also simple: configure your ACP client to run the absolute
path to `target/release/anvil` as a stdio server.

## Client Examples

The `examples/` directory shows why a "just an ACP server" architecture is
useful. Each example is a small Rust client that launches Anvil, opens an ACP
session, sends a task, streams the response, and handles permission requests.

Build the Anvil binary first so the examples can launch it without nesting
another Cargo build:

```bash
cargo build --bin anvil
```

Then run one of the clients:

```bash
# Analyze an issue from literal text and print a triage note.
cargo run --example issue_bot -- --repo BrokkAi/anvil \
  --issue-text "Users report that /setup timeout rejects 300 seconds."

# Review a supplied diff snippet and print findings.
cargo run --example review_bot -- --repo BrokkAi/anvil \
  --diff-text "diff --git a/README.md b/README.md"

# Draft an issue without creating it.
cargo run --example issue_writer_tui -- --repo BrokkAi/anvil --dry-run \
  --prompt "The README does not explain how to run ACP client examples."
```

The examples default to `ANVIL_AGENT`, then `target/debug/anvil`, then
`cargo run --quiet --bin anvil --`. You can always override the server command:

```bash
cargo run --example review_bot -- --repo BrokkAi/anvil \
  --agent "./target/debug/anvil" \
  --diff-text "diff --git a/src/main.rs b/src/main.rs"
```

Example capabilities:

- `issue_bot`: reads issue text from `--issue-text`, `--issue-file`, or
  `gh issue view`; optionally posts a comment with `--post-comment`.
- `review_bot`: reads a diff from `--diff-text`, `--diff-file`, `gh pr diff`,
  or the local git diff; optionally posts a PR comment with `--post-comment`.
- `issue_writer_tui`: collects a prompt interactively or via `--prompt`, asks
  Anvil to draft a GitHub issue, and only calls `gh issue create` after user
  confirmation. Use `--dry-run` to stop after the draft.

These are deliberately small. They are meant to show the integration shape, not
hide it behind a framework.

## Model Providers

Anvil is zero-config by default:

- **Codex / ChatGPT**: reads `~/.codex/auth.json` when present and can refresh
  stale credentials. Run `/setup codex` from a session to sign in.
- **Ollama**: probes `http://localhost:11434/v1/models`. Run Ollama on the
  default port, then use `/setup local refresh`.
- **ds4** (antirez/ds4): when a `ds4-server` process is running, Anvil detects
  it and resolves the TCP port it is listening on (no fixed port assumed),
  then probes its OpenAI-compatible `/v1/models`. macOS and Linux only. Set
  `DS4_BASE_URL` (e.g. `http://127.0.0.1:9000`) to point at a non-default host,
  a remote box, or a reverse proxy instead. Discovered only while ds4-server
  is up; start it and `/setup local refresh` to bring it online.
- **OpenRouter**: uses `OPENROUTER_API_KEY` or credentials saved through
  `/setup openrouter key <key>`.
- **Bedrock**: uses `AWS_BEARER_TOKEN_BEDROCK` or `~/.secrets/bedrock_api_key`.
  Native Bedrock invoke remains in place for Anthropic-style model ids such as
  `bedrock::us.anthropic.claude-sonnet-4-6`. Bedrock-hosted `openai.*` models
  route through the OpenAI-compatible `Responses` API at
  `https://bedrock-mantle.<region>.api.aws/v1` when the selected model supports
  that API on Bedrock. On startup and refresh, Anvil discovers Bedrock
  inference profiles for the current region and publishes invocable model ids
  in the picker. `ANVIL_BEDROCK_MODEL` may be either a foundation model id or
  an inference profile id/ARN; when a base model requires an inference profile,
  Anvil normalizes it to the matching profile automatically. Bedrock models
  expose model-family-specific reasoning presets in the effort picker. OpenAI
  GPT-5.4 and GPT-5.5 expose `none`/`low`/`medium`/`high`/`xhigh`; `gpt-oss`
  models expose `low`/`medium`/`high`; current Anthropic effort-capable models
  expose the documented subset of `low`/`medium`/`high`/`xhigh`/`max`, while
  manual extended-thinking Claude models expose `low`/`medium`/`high` budget
  presets. Other Bedrock entries retain the generic `low`/`medium`/`high`
  fallback.
  Choose `Off` in that picker (or `/setup reasoning off`) to omit reasoning
  controls for a session. Native Anthropic requests send the corresponding
  `thinking`/`output_config.effort` shape for the selected model family.
  If a manual-thinking model responds that it requires
  `thinking.type=adaptive` + `output_config.effort`, Anvil retries with that
  shape and remembers it for later turns.

Provider discovery is non-fatal. If one source is unavailable, Anvil logs it and
continues with the providers that work.

Provider priority for `/setup choose` is Codex first, then local models (Ollama,
then ds4), with OpenRouter last. You can also select a specific model with
`/setup model <wire id>`.

## Slash Commands

Built-in commands:

- `/setup`: model login, provider selection, behavior mode, sandbox mode,
  turn recaps, timeout, and advanced settings.
- `/permissions`: list or revoke remembered Always allow entries.
- `/context`: show the current session context snapshot and token estimate.
- `/loop <seconds> <slash-command-or-prompt>`: repeat a slash command or
  prompt on an interval until the session is cancelled.
- `/goal [--max-turns N] <objective>`: work autonomously across turns until
  the objective is verifiably met. Anvil re-prompts itself each turn with the
  objective plus a completion audit; it stops only when the model verifies
  every requirement against the actual worktree/command state and emits the
  `GOAL_COMPLETE` sentinel, when the model reports a blocker on three
  consecutive turns, when a model request fails fatally (e.g. auth or an
  invalid request), or when the session is cancelled. A transient model or
  network outage does **not** stop the goal: it is retried with a capped
  backoff so the goal survives the outage and resumes when it clears. There is
  **no turn limit by default** — the stopping condition is the verified
  completion, not a turn count. Pass `--max-turns N` to set an optional ceiling
  (a wrap-up turn runs when it is reached). Inspired by Codex's `/goal`, whose
  token budget is likewise optional.
- `/compress`: summarize uncompressed history turns to free context window.
- `/mcp`: list and configure stdio MCP servers.
- `/pr-create [title]`: create a GitHub pull request from the current branch.

Skill slash commands are discovered from `SKILL.md` files under
`$CODEX_HOME/skills` (or `~/.codex/skills`), `~/.claude/skills`,
`~/.agents/skills`, and project `.claude/skills` / `.agents/skills` roots. If a
skill name collides with a built-in command, the built-in command wins and the
skill slash entry is hidden from autocomplete.

## MCP Servers

Anvil reads MCP server configuration from its config file and manages stdio MCP
subprocesses per session. New servers default to standard `Content-Length` MCP
stdio framing; use `--framing line` only for NDJSON-speaking servers.

Bifrost is preinstalled as an enabled MCP server backed by Anvil's managed
local binary:

```text
<managed-bifrost> --root {cwd} --mcp searchtools --no-line-numbers  # line framing
```

Use `/mcp` in the editor to list or change MCP servers:

- `/mcp list`
- `/mcp add [--framing content-length|line] <name> <command> [args...]`
- `/mcp enable <name>`
- `/mcp disable <name>`
- `/mcp remove <name>`
- `/mcp reset`

The `{cwd}` placeholder in arguments expands to the session workspace root. Use
shell-style quoting for commands or args that contain spaces. Changes take
effect on the next tool-capable prompt.

Bifrost provides structural code-intelligence tools such as:

- `search_symbols`: find definitions across the workspace.
- `get_symbol_sources`: fetch source code for specific symbols.
- `most_relevant_files`: identify related files using imports and git history.

## Permissions And Sandboxing

Permission mode controls whether Anvil asks before tool calls:

- `default`: ask before edits, and before shell commands except for a conservative auto-approved subset of sandboxed read-only commands (for example `rg`, `git diff`, `git show`, `git status`).
- `auto`: like `default`, but promptable tool calls may be approved by the permission classifier when clearly inside the user's request.
- `acceptEdits`: allow file edits, and ask before shell commands except for a conservative auto-approved subset of sandboxed read-only commands (for example `rg`, `git diff`, `git show`, `git status`).
- `readOnly`: block edits and shell commands.
- `bypassPermissions`: allow tool calls without prompting.

Change permission mode with the session `Permission` selector exposed by the
ACP client.

Approvals remembered through an **Always allow** prompt are stored in Anvil's
setup state on disk and can be inspected or revoked with:

```text
/permissions list
/permissions revoke <number-or-key>
/permissions clear
```

`run_shell_command` always advertises a `sandbox_permissions` field
(Codex-style explicit escalation). The model may set it to `require_escalated`
up front when a command genuinely needs access the sandbox blocks — network or
DNS access, package downloads, `git push`, attaching to or debugging host
processes, or writes outside the working directory — without first triggering a
sandbox failure. An escalation request always prompts the user with a **Run
outside sandbox** choice (never auto-allowed and never remembered as an
**Always allow** rule), and Anvil rejects it deterministically when there is no
active OS sandbox to escape or when the session is in read-only mode.

When a sandboxed shell command fails with output that looks like a sandbox
boundary issue (for example `permission denied`, `operation not permitted`, a
read-only filesystem, process/namespace isolation, or blocked network/DNS
access), Anvil also appends guidance to the tool result reminding the model that
it can retry with `sandbox_permissions: "require_escalated"` if — and only if —
the sandbox is the actual blocker.

Sandbox mode is a separate execution boundary. This matters because Anvil runs
in more places than a single blessed desktop environment: macOS has the Seatbelt
sandbox exposed through `sandbox-exec`, many Linux installs have Bubblewrap
(`bwrap`), and some environments have neither.

Anvil chooses one effective sandbox strategy per session:

- **`os`: OS sandbox for shell commands.** On macOS Anvil uses Seatbelt through
  `sandbox-exec -f <profile.sb> -- sh -c <cmd>`. On Linux it uses Bubblewrap
  through `bwrap`. This is the strongest boundary for `run_shell_command`:
  filesystem access is restricted according to the session permission mode, and
  parsing runs natively for speed.
- **`wasm`: WASM fallback for untrusted project data.** When the OS sandbox is
  not available, Anvil can still run parsing and search work inside an embedded
  Wasmtime component, `brokk-acp-sandbox.wasm`. This is the `WasmFallback`
  strategy in the code. In this mode, shell commands are permission-gated but
  not wrapped in Seatbelt or Bubblewrap.
- **`off`: no sandboxing.** Parsing runs in-process and shell commands are only
  controlled by the ACP permission gate.

These strategies are mutually exclusive. WASM is not layered on top of
Seatbelt/Bubblewrap; it is the fallback Anvil uses when those OS-level shell
sandboxes are unavailable or when `/setup sandbox wasm` is selected.

WASM-sandboxed work includes:

- `SKILL.md` YAML frontmatter parsing.
- `AGENTS.md` / `CLAUDE.md` and subagent file reads with byte caps.
- Session zip reads and rewrites, including bounded entry reads and prefix
  reads for `content/*.txt`.
- `grep_search` / file-content search, including user-controlled regexes and
  bounded file reads.

The WASM sandbox gives these operations a fresh Wasmtime store per request, a
fuel budget for CPU, a 64 MiB linear-memory cap, bounded stdout/stderr pipes,
and read-only per-call preopens only when a file or directory must be inspected.
If the guest traps, exhausts fuel, or hits the memory limit, Anvil reports a
sandbox error instead of silently retrying natively.

In `wasm` mode:

- `run_shell_command` is not OS-sandboxed because Seatbelt/Bubblewrap are not
  active in this strategy; the ACP permission gate still applies.
- WASM does not replace path validation for file tools. Built-in file reads,
  writes, and edits still use `safe_resolve` / `safe_resolve_for_write` to keep
  paths inside the session `cwd`.

Sandbox selection:

- If an OS sandbox is available, Anvil defaults to `os`: shell commands are
  wrapped by Seatbelt/Bubblewrap and parsing runs natively for speed.
- If no OS sandbox is available, Anvil defaults to `wasm`: shell commands are
  not OS-sandboxed, but parser/search/archive work goes through the WASM
  sandbox.
- If the binary was built without the default `wasm-sandbox` feature, `wasm`
  mode is unavailable and Anvil falls back to native parsing when no OS
  sandbox exists.
- `--no-wasm-sandbox` disables the WASM fallback and forces native parsing.

You can override the effective mode per session:

```text
/setup sandbox default
/setup sandbox os
/setup sandbox wasm
/setup sandbox off
```

`/setup sandbox off` disables sandboxing; permission prompts remain controlled
by the session `Permission` selector.

## Tools

Core tools include:

- `read_file`
- `write_file`
- `edit`
- `list_directory`
- `grep_search`
- `web_search`
- `run_shell_command`

MCP servers can add more tools at runtime. The default Bifrost MCP server adds
workspace-aware code intelligence such as symbol search, source lookup, usage
scanning, relevant-file discovery, file search, git-log inspection, JSON
querying, and XML skimming.

## Sessions And Context

Anvil persists ACP sessions to disk so clients can load or resume previous
work. A session stores conversation turns, tool exchanges, model selection,
behavior mode, and per-turn summaries.

Context management is built in:

- `/context` reports the current prompt shape and token estimate.
- `/compress` summarizes uncompressed turns and stores summaries beside the
  original verbatim log.
- Automatic compression runs when a prompt would overflow the configured model
  context budget.

The verbatim history remains on disk; summaries affect what gets sent to the
model, not whether history is retained.

## Subagents

Subagents are markdown files that Anvil exposes to the model through the dynamic
`task` tool. They are useful for focused work such as deep code review, targeted
research, or repetitive analysis.

File format:

```markdown
---
name: bug-hunter
description: Review a diff for concrete correctness bugs.
max_turns: 8
tools: Read, Grep, Glob
---

You are a meticulous code reviewer. Flag concrete bugs only.
Return findings as `<path>:<line> - <one-line description>`.
```

`max_turns` is optional. When present, it lowers that subagent's tool-calling
turns below the inherited parent budget. `maxTurns` is accepted as an alias for
compatibility with camelCase frontmatter.

`tools` is optional. When omitted, a subagent inherits the parent tool catalog.
When present, it is an allowlist. Claude-style tool names such as `Read`,
`Write`, `Edit`, `Bash`, `Grep`, `Glob`, `WebSearch`, `WebFetch`, and `LS`
are mapped to Anvil's built-in tool names; Anvil and MCP tool names can also
be listed directly.

Discovery order:

1. `~/.claude/agents/`
2. `~/.agents/agents/`
3. `<git-root>/.claude/agents/` walking down to `cwd`
4. `<git-root>/.agents/agents/` walking down to `cwd`

Layout is flat: one `<name>.md` file per subagent. Later scopes override earlier
ones. Project-level agents override user-level agents.

Constraints:

- Subagents inherit the parent session permission gate.
- Nested delegation is disabled.
- Each subagent inherits the parent's turn budget unless the subagent
  frontmatter sets a lower `max_turns` value.
- A subagent with `tools` frontmatter sees and can call only those tools.
- Intermediate subagent updates are silent; only the final answer returns to
  the parent conversation.
- Cancelling the parent prompt cancels the active subagent.

For the full concurrency contract that delegated/audit-lane workflows can rely
on — sequential vs. parallel execution, ordering/isolation, cancellation and
timeout scope, observability, and structured-result collection — see
[`docs/concurrency.md`](docs/concurrency.md).

## CLI Reference

```text
anvil [OPTIONS]
```

| Flag | Env Var | Default | Description |
| ---- | ------- | ------- | ----------- |
| `--default-model` | - | - | Default model for new sessions. Accepts a wire id or bare provider id. |
| `--max-turns` | - | `0` (unbounded) | Optional cap on tool-calling iterations per prompt. Default unbounded: the loop exits when the model answers without a tool call. Pass `N>0` to bound cost/time; `0` disables. |
| `--max-sessions` | - | `50` | Maximum resident sessions before LRU eviction. `0` disables the cap. |
| `--max-history-turns` | - | `50` | Maximum in-memory history turns per session. `0` disables the cap. |
| `--llm-idle-timeout-secs` | `ANVIL_LLM_IDLE_TIMEOUT_SECS` | `300` | SSE inactivity timeout for LLM streaming. |
| `--transient-setup` | `ANVIL_TRANSIENT_SETUP` | `false` | Keep setup preferences process-local; model/reasoning/sandbox choices for this run do not read or update the global setup file. |
| `--no-wasm-sandbox` | `ANVIL_NO_WASM_SANDBOX` | `false` | Disable the wasmtime-hosted parser sandbox. |

## Build And Test

First-time prerequisite:

```bash
rustup target add wasm32-wasip2
```

This prerequisite applies to the default build, which enables the
`wasm-sandbox` feature and embeds `brokk-acp-sandbox.wasm`. Builds that opt out
with `--no-default-features` skip the nested `wasm32-wasip2` compile entirely.

Then run the usual checks:

```bash
cargo build --release
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

On Linux, install Bubblewrap (`bwrap`) for OS-level `run_shell_command`
sandboxing.

## Releases

Release binaries are published as zipped archives with `.sha256` sidecars.
Tags follow `vX.Y.Z`; release workflows validate that the tag matches
`Cargo.toml`.

Asset names:

- `brokk-anvil-vX.Y.Z-x86_64-unknown-linux-gnu.zip`
- `brokk-anvil-vX.Y.Z-aarch64-unknown-linux-gnu.zip`
- `brokk-anvil-vX.Y.Z-aarch64-linux-android.zip`
- `brokk-anvil-vX.Y.Z-x86_64-pc-windows-msvc.zip`
- `brokk-anvil-vX.Y.Z-universal-apple-darwin.zip`
