# Plan: Codex-Style Sandbox Escalation for Anvil

## Problem

Anvil currently hides `sandbox_permissions: "require_escalated"` from `run_shell_command` until after a sandbox-looking failure. This was intended to prevent models from overusing escalation, but in practice it creates a hard failure mode:

- the model cannot proactively request the permissions a command obviously needs;
- retry availability depends on fragile stderr/stdout string matching;
- the retry must be the exact same command and directory, which blocks legitimate follow-up fixes;
- users see the agent stall, ask for manual workarounds, or take unsafe alternate paths.

Codex CLI exposes escalation in the shell tool schema up front, then relies on server-side policy, approval prompts, and auditability to prevent abuse. Anvil should move to that model.

## Goal

Make `run_shell_command` support Codex-style explicit sandbox escalation:

- expose `sandbox_permissions` in the normal shell tool schema from the start;
- keep default behavior sandboxed;
- require approval for outside-sandbox execution;
- never persist outside-sandbox approval as an always-allow rule;
- reject escalation when the current session cannot or should not honor it;
- preserve Anvil's permission gate and scoped approval model.

## Current Behavior To Replace

Relevant implementation points:

- `src/tools/mod.rs`: `tool_definitions_with_shell_escalation(false)` hides `sandbox_permissions` initially.
- `src/tool_loop.rs`: `ShellSandboxRetryState` only permits escalation after a matching failed command.
- `src/tool_loop.rs`: `is_likely_sandbox_limitation()` decides whether retry becomes available by matching output strings.
- `src/tool_loop.rs`: `permission_options_for_request()` only shows "Run outside sandbox" when retry escalation was requested.
- `src/tool_loop.rs`: `deterministic_gate_rejection()` rejects escalation unless retry state exists.

These pieces should be simplified so the model can request escalation directly and the gate decides whether to ask the user.

## Proposed Design

### 1. Always Advertise The Escalation Field

Change `ToolRegistry::tool_definitions()` so `run_shell_command` always includes:

```json
{
  "sandbox_permissions": {
    "type": "string",
    "enum": ["use_default", "require_escalated"],
    "description": "Per-command sandbox override. Defaults to `use_default`. Use `require_escalated` only when the command needs access outside the active sandbox, such as network access, writes outside the workspace, attaching to host processes, or other operations likely blocked by sandbox policy."
  },
  "justification": {
    "type": "string",
    "description": "User-facing approval reason for `require_escalated`; omit otherwise."
  }
}
```

Notes:

- `use_default` should be optional in practice; omitted means `use_default`.
- Keep `require_escalated` as the only behavior-changing value.
- If keeping compatibility with current deserialization is easier, accept `use_default` without storing it.

### 2. Parse Explicit Sandbox Permission Intent

Update `RunShellCommandArgs`:

- rename `_sandbox_permissions` to `sandbox_permissions`;
- add `UseDefault` to `ShellSandboxPermissionArg`;
- add optional `justification: Option<String>` or reuse `description` if the UI should not grow a new field yet.

Add helper methods:

```rust
impl RunShellCommandArgs {
    fn requests_outside_sandbox(&self) -> bool;
}
```

This keeps raw JSON parsing out of the gate logic and makes tests simpler.

### 3. Remove Retry-State As A Prerequisite

Delete or bypass this requirement:

- no `shell_sandbox_retry_states.is_empty()` rejection;
- no exact command/directory match requirement;
- no need to refresh the tool catalog after sandbox-looking failures.

The retry hint can remain as UX, but it should no longer be the only way to unlock escalation. If retained, make it a recommendation, not a gate.

### 4. Gate Escalation Through Approval

Update `deterministic_gate_rejection()` and `consult_gate()` so:

- `sandbox_permissions: "require_escalated"` is rejected in `ReadOnly` mode with a clear message;
- it is rejected if OS sandboxing is not active or `SandboxPolicy::resolve(...)` already returns `None`;
- it always bypasses shell auto-allow and always-allow caches;
- it always prompts the user or the configured approval reviewer;
- it never calls the auto-permission classifier, because that classifier explicitly treats outside-sandbox execution as not auto-approvable.

Approval prompt options for escalated shell commands should be only:

- `Run outside sandbox`
- `Reject`

Do not include `Always allow`.

### 5. Execute With One-Time Sandbox Override

When approved:

- pass `SandboxPolicy::None` through `sandbox_policy_override`;
- set `outside_sandbox_once = true`;
- preserve the existing explicit outside-sandbox notice in command output;
- do not add any always-allow entry.

This behavior already mostly exists through `PermissionGrant { sandbox_policy_override: Some(SandboxPolicy::None), allow_always: false }`.

### 6. Improve Model Instructions

Update the `run_shell_command` tool description:

- say commands run sandboxed by default;
- say `require_escalated` asks for one-time approval;
- give concrete valid cases: network/DNS, package downloads, `git push`, host process attach/debugging, writing outside cwd when requested;
- say not to use it for ordinary reads, searches, builds, tests, or workspace writes that should work inside the sandbox.

This mirrors Codex: the model can ask, but the platform validates and prompts.

## Implementation Steps

1. Update `src/tools/mod.rs`
   - Always include `sandbox_permissions`.
   - Add optional `justification` if desired.
   - Remove or deprecate `tool_definitions_with_shell_escalation(bool)`.
   - Keep a compatibility wrapper if many call sites expect it.

2. Update `src/tool_loop.rs`
   - Remove `ShellSandboxRetryState` gating from `deterministic_gate_rejection()`.
   - Ensure escalation disables auto-allow and always-allow.
   - Ensure escalation prompts with outside-sandbox-only options.
   - Keep `is_likely_sandbox_limitation()` only for adding helpful failure text, or delete it if no longer useful.

3. Update `src/tools/mod.rs` execution args
   - Parse `use_default` and `require_escalated`.
   - Avoid underscore-prefixed fields for values that drive policy.

4. Update tests
   - Schema includes `sandbox_permissions` initially.
   - Omitted or `use_default` remains sandboxed.
   - `require_escalated` prompts immediately in editable modes.
   - `require_escalated` is rejected in read-only mode.
   - `require_escalated` is rejected when OS sandboxing is inactive.
   - `require_escalated` never uses always-allow.
   - `require_escalated` does not require an earlier matching failed command.

5. Run validation
   - `cargo fmt`
   - focused tests around `tool_loop` and `tools`
   - broader `cargo test` if focused tests pass

## Compatibility And Safety

This change makes escalation easier for the model to request, but not easier to execute without authorization. The safety boundary should be:

- default sandboxed execution;
- deterministic server-side rejection where escalation is invalid;
- explicit one-time user approval where escalation is valid;
- no persistent allow rule for outside-sandbox commands;
- clear audit notice on outside-sandbox execution.

This is closer to Codex CLI's model and should reduce dead ends without weakening the permission gate.

## Migration Notes

Existing sessions may have tool definitions persisted in turn history. Treat this as forward compatible:

- old model turns without `sandbox_permissions` continue to default to sandboxed execution;
- old retry-state hints are harmless if left in historical tool output;
- new turns should receive the new schema immediately.

## Open Questions

- Should Anvil add a distinct `approval_policy` concept like Codex, or map this directly to existing `PermissionMode`?
- Should `require_escalated` be permitted in `Default`, `Auto`, and `AcceptEdits`, or only a subset?
- Should the UI copy say "Run outside sandbox" or "Run without sandbox" consistently?
- Should `justification` be required for escalation, or is the command itself sufficient context?
