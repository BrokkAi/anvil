# Contributing to Anvil

Thanks for helping improve Anvil. Contributions from people using AI tools are
welcome; everyone remains responsible for the accuracy, safety, licensing, and
relevance of what they submit. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before You Start

- Search existing issues and pull requests before opening a new one.
- Use the agent/runtime bug form for incorrect behavior during an ACP session,
  and the other-bug form for installation, setup, integration, packaging, or
  documentation problems. Blank issues remain available when neither form fits.
- Keep changes focused on one problem or capability. For a large protocol,
  permission, sandbox, session-format, or provider change, open an issue or
  discuss the direction on [Discord](https://discord.gg/geYkWUeH) first.
- Do not put credentials, private source code, or unredacted private transcripts
  in issues, tests, logs, or pull requests. Report suspected vulnerabilities
  privately to [feedback@brokk.ai](mailto:feedback@brokk.ai).

An issue is useful but not mandatory for a well-scoped pull request. Use
`Fixes #123` or `Closes #123` when a pull request resolves an existing issue.

## Development Setup

Anvil is a Rust workspace whose default feature set embeds a
`wasm32-wasip2` sandbox guest in the host binary. Install the stable Rust
toolchain and the guest target before building:

```bash
rustup target add wasm32-wasip2
cargo build --release
```

On Linux, install Bubblewrap (`bwrap`) to exercise the OS-level sandbox used by
`runShellCommand`.

For a quick host-only build on a platform where the nested Wasm build is not
available, you can disable the default feature:

```bash
cargo build --no-default-features --bin anvil
```

That is useful while iterating, but it does not replace validating the default
feature set before submitting a change.

## Understand the Runtime Boundaries

Anvil is an ACP server over stdio. The client owns the user interface and
per-session controls; Anvil owns model routing, the tool loop, permission and
sandbox enforcement, context management, and session storage.

The detailed implementation contracts are maintained in [AGENTS.md](AGENTS.md).
The most important contribution boundaries are:

- Standard output is reserved for JSON-RPC. Send logs to standard error through
  `tracing`.
- ACP session configuration is client-owned and live-only. Do not persist model,
  reasoning, behavior, permission, or service-tier selections as Anvil defaults.
- Permission, path-validation, archive, MCP, and sandbox changes must fail
  closed. Cover both the allowed behavior and relevant denial or malformed-input
  paths.
- Provider discovery failures should normally be logged and treated as an
  unavailable provider rather than preventing Anvil from starting.
- Do not add lint suppressions to make CI pass. Fix the underlying code; if a
  suppression is genuinely required by an external constraint, document the
  invariant that makes it safe.

When changing a built-in tool, model provider, subagent behavior, or context
compaction, follow the component-specific checklist in `AGENTS.md` and keep its
contract synchronized with the implementation.

## Tests and Documentation

Add the smallest regression test that would have caught the problem:

- Put focused unit tests beside the implementation when possible.
- Use `tests/acp_smoke.rs` for behavior that must cross the ACP process or
  JSON-RPC boundary.
- Include negative controls for permission, sandbox, path, protocol, and
  persistence changes.
- Update `README.md` when a user-visible command, setup flow, provider, tool,
  configuration option, or limitation changes.
- Update `AGENTS.md` when an implementation invariant or extension checklist
  changes.

During development, run targeted tests by name or module. Before submitting,
run the same core checks as CI:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --release
cargo test
```

Run the full `cargo test` suite outside restricted execution sandboxes and with
localhost binding permitted. Wiremock-backed tests start local HTTP servers; an
environment that denies socket binds can produce many unrelated failures with
messages such as `Failed to bind an OS port for a mock server` or
`PermissionDenied: Operation not permitted`. Targeted tests that do not use
Wiremock can still run in a restricted sandbox.

CI repeats the checks across Ubuntu x86-64, Ubuntu ARM64, macOS, and Windows,
and also builds the Android ARM64 target. You do not need to reproduce every
runner locally, but consider path syntax, filesystem behavior, subprocesses,
and sandbox availability when changing platform-sensitive code.

## Dependency and License Changes

Commit `Cargo.lock` when dependency resolution changes. Anvil uses a reviewed,
deny-by-default dependency-license policy and ships generated third-party
notices in release archives. Do not broaden an allowed license or add an
exception without explaining and reviewing the obligation it introduces.

After changing dependencies, license policy, or vendored notice material, use
Node.js 24 and the tool versions pinned by CI to refresh the reports:

```bash
cargo install --locked cargo-about --version 0.9.1 --features cli
cargo install --locked cargo-deny --version 0.20.2

cargo deny --config licenses/deny.toml --locked check licenses
cargo about generate --offline --config licenses/about.toml --locked --fail \
  licenses/about.hbs -o licenses/THIRD_PARTY_LICENSES.html
node scripts/generate-supplemental-third-party-notices.mjs
```

Review the generated diff rather than assuming regeneration is sufficient. CI
recreates both notice reports and fails when committed output is stale.

## Pull Requests

A useful pull request description lets a reviewer understand the behavioral
change without reconstructing it from the file diff. The repository template
asks for:

- A concise description of what changed, why, and the observable effect.
- Key semantic changes rather than a list of edited files.
- Root cause for bug fixes when it is known.
- Before/after evidence and capability or safety boundaries for agent/runtime
  changes.
- Important touch points for broad or cross-cutting changes.
- Exact test, lint, build, benchmark, and manual-validation commands actually
  run.

If a relevant check could not be run or failed because of an environment
constraint, say so explicitly and include any narrower validation that did
pass. Do not report a check as passing based only on an expected outcome.

Reviewers will pay particular attention to:

- ACP compatibility and separation between client-owned and server-owned state.
- Permission, sandbox, path, archive, and secret-handling boundaries.
- Deterministic transcript, tool-result, and session behavior.
- Regression tests and negative controls.
- Documentation and implementation-contract drift.
- Cross-platform behavior and dependency-license obligations.

## Releases

Releases are maintainer-driven. `Cargo.toml` is the version source of truth;
release-preparation commits update the `brokk-anvil` version in both
`Cargo.toml` and `Cargo.lock`.

A `vX.Y.Z` tag triggers the GitHub release and crates.io workflows.
The workflows refuse to publish when the tag and `Cargo.toml` version differ.
They build the supported Linux, Android, Windows, and macOS archives, attach
SHA-256 sidecars, include the required license and source-notice files, and
publish the `brokk-anvil` crate.

Before tagging, maintainers should confirm that:

1. The version in `Cargo.toml` and `Cargo.lock` matches the intended tag.
2. Formatting, Clippy, the release build, and the full test suite pass.
3. Dependency-license policy and generated notice reports are current.
4. User-facing setup and release documentation reflects the shipped behavior.
5. The release commit is merged and the tagged commit is the exact commit meant
   to be published.
