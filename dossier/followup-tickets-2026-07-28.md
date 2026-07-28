# Follow-up tickets from the permission-classifier discovery (2026-07-28)

Filed on GitHub: mjolnir#506 (handshake, mj half), anvil#311 (effective-mode
reporting), anvil#312 (classifier hygiene), anvil#313 (cheaper classifier
model). Item 1 (mj flag plumbing) landed as mjolnir 83b35b8.

Context: mjolnir's `--permission-mode bypassPermissions` never reached anvil
over ACP; every benchmark session ran in Auto mode and paid an untraced
classifier LLM call per gated tool call (~52% of wall-clock, 19% spurious
denials). Deployment fix landed: `BROKK_ACP_PERMISSION_MODE` env default in
anvil + staged by brokkbench. These tickets are the rest of the story.

## 1. Fix the mj flag plumbing (protocol-correct fix) — DONE

Landed as mjolnir 4d71315 (source found at ~/Projects/mjolnir; the /mnt/optane
copy was the deleted one). True dead-end was TWO-fold: interactive sessions
built RuntimeRoleConfig with permission: None, and configure_permissions
mapped (AdapterKind::Custom, _) to None — benchmark runs are always
custom/bpr-agent/<wire>, so they were in the one config hole. Custom now
reuses the Anvil permission_mode mapping, gated on the server advertising the
option; failures warn loudly. Unpushed; benchmark binaries still pinned to
mj-6147059 (harmless — the anvil env default covers benchmarks regardless).

## 2. Classifier hygiene for Auto-mode users

The env default makes these irrelevant for benchmarks, but every interactive
Auto-mode user still pays them:
- Effort floor: classifier calls inherit the session's reasoning effort
  (tool_loop.rs ~4452). A trivial allow/deny prompt at xhigh costs ~7s/call.
  Force the cheapest effort for classifier calls.
- Strict output schema: `prefer_json_object` (tool_loop.rs ~4564) yields a
  1.2x empty-completion retry storm at high effort. Use strict json_schema.
- Fail-open-with-notice: `Unavailable` currently maps to Reject
  (tool_loop.rs ~4208-4223) — 19% of decisions denied legitimate calls and
  forced a wasted main-loop turn. Fail open with a visible notice (or prompt).
- Memoize decisions on (tool_name, normalized input): repeated edits to the
  same file currently re-classify from scratch every time.
- Usage accounting: classifier retry-attempt usage and unavailable-response
  usage are dropped from turn/session usage (only successful-decision usage
  is added). The permission_classifier trace record now captures the true
  sums; the session accounting should too.

## 3. Cheaper classifier model (Jonathan, 2026-07-28)

Allow configuring a distinct, cheaper model for the auto-permission
classifier instead of the session model. Default to deepseek flash
(`deepseek::deepseek-v4-flash`) when deepseek credentials/provider are
available; fall back to the session model otherwise. Composes with the
effort-floor item above (a flash-class model at low effort makes Auto mode
nearly free). Needs: config surface (env/config option), provider-availability
probe, and a trace record naming the classifier model so cost attribution
stays honest.

## 4. Permission-mode handshake assertion

The failure class was "client believes it set a mode; nothing verified it."
Anvil should report the effective permission mode in the session/new response
(or an early log line the harness asserts on), and bpr's smoke path should
fail loudly when the effective mode differs from the requested one.
