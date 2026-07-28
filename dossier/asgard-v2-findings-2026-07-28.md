# Asgard v2: full-corpus findings, fixes, and remaining gaps

*Prepared 2026-07-28 for design review. Covers the two 113-task DeepSWE sweeps
(`fullLuna`, `fullDs`), the failure diagnosis, fixes landed or in flight, and
the honest gap accounting vs vanilla single-agent baselines.*

---

## 1. What ran

Two concurrent full-corpus sweeps on the local box (30 threads total, 120m/attempt
cap, binary `anvil-7f54968`, supervisor `bedrock::openai.gpt-5.6-sol+high`):

| Sweep | Worker | Attempts | Solved | TimedOut |
|---|---|---|---|---|
| fullLuna | `bedrock::openai.gpt-5.6-luna+xhigh` | 112 (1 lost) | 9 (8.0%) | 4 |
| fullDs | `deepseek::deepseek-v4-pro` | 113 | 4 (3.5%) | 2 |

Published vanilla baselines (mini-swe-agent via Pier, 4 trials/task,
`deep-swe/published-results/deepswe-v1.1/per-task-by-model-effort.csv`):

| Model | Full corpus |
|---|---|
| gpt-5.6-sol @high | **69.2%** (313/452) |
| gpt-5.6-luna @xhigh | **56.9%** (257/452) |

Headline: asgard (sol supervising luna) at 8% vs its own worker model solo at 57%.
Most of that gap is explained below — but not all of it.

## 2. Contamination: the sol daily-quota incident

Partway through, the account's Bedrock **daily** input-token quota for sol died:

```
HTTP 429: quota input-tpd:842609633142:openai.gpt-5.6-sol (InputTokens) exceeded
```

Anvil classified all 429s as retryable-transient; retries exhausted; the asgard
fallback (`auto_save` → `finalize_latest`) then silently finalized arbitrary
checkpoints with no supervisor judgment for the rest of the day. Each poisoned
attempt completed looking like a normal `TESTS_FAILED`.

| | attempts | solved | patchBytes=0 |
|---|---|---|---|
| luna clean | 26* | 5 (19%) | 4 |
| luna quota-hit | 86 | 4 (5%) | 55 |
| ds clean | 13* | 3 (23%) | 3 |
| ds quota-hit | 100 | 1 (1%) | 79 |

*The "clean" screen (no 429 fallback in trace) later proved leaky: 2 of 9
hand-diagnosed "clean" attempts were actually first-turn supervisor deaths via
other error strings (a Bedrock 500 storm; an `invalid_prompt` mid-quota-storm).
True clean-n is ~24/11.*

**186 of 225 attempts ran with a dead supervisor.** The sweep is unusable for
absolute rates; the luna>ds ordering survives (paired McNemar on 112 common
tasks: both=3, luna-only=6, ds-only=1, p=0.125 — suggestive only).

Token-rate finding while investigating (matters for any future fleet run):
**luna workers, not the sol supervisor, are the token hog** — ~727k input
tok/attempt-minute (34M/attempt) vs sol's ~54k. At 113 concurrent attempts
luna needs ~82M input TPM against a 20M quota. Sol's TPM is fine; sol's
*daily* quota is the binding constraint and is not exposed in Service Quotas
(only per-minute quotas are, all adjustable).

## 3. Diagnosis: 9 clean failures, hand-audited (3 parallel Opus agents, 3 with local repros)

Cases chosen where vanilla luna solves the task 3–4/4, i.e. purest
"asgard lost a task its own worker wins solo."

| Case | f2p | Proximate cause | Class |
|---|---|---|---|
| fullDs actionlint | 0/55 | **Perfect solution scored zero.** Agent test file shares the filename the verifier's hidden `test.patch` claims; grader `reset_paths` deletes it, stranding helper symbols → package build failure. Repro: minus 3 stray test files → 89/89 + 111/111. | DELIVERY |
| fullDs arcane | 64/82 | Agent test declares `TestComplianceHandler_GetBaseline_404`; hidden test declares the same symbol, same package → 18 tests lost to one collision. | DELIVERY |
| fullLuna dateutil | 67/67 (+1 p2p) | Feature perfect. Worker relaxed parser validation and rewrote the pre-existing test guarding it ("obsolete legacy expectation"); supervisor ratified. Grader resets test files → original test restored → caught the change. | JUDGMENT |
| fullLuna arcane | 81/82 | Ambiguity (scalar encoding) resolved wrong, then frozen by self-authored contract test. Intake item 17 had the correct reading; the phrase appears in **zero** model responses. | ENTRENCHMENT |
| fullLuna actionlint | 54/55 | One wrong line (max-strictness merge vs per-path override). Intake item 23 correct. Bad line inherited through 32 windows; 5 attack workers, 7 attacks, the concept `relax` appears 0 times in 122MB of trace. | ENTRENCHMENT |
| fullLuna aiomonitor | 50/53 | Route-shape ambiguity. Intake Reader L listed both readings (A48/A51); **intake Reader G froze the wrong one as spec**; all 27 windows inherited it. | ENTRENCHMENT (intake-caused) |
| fullLuna bandit | 40/66 | Worker's taint analyzer only walks *called* functions; its own fixtures were all module-level driver code — the one shape that works. 6 attackers probed with the same idiom. Verifier fixtures are bare `def`s (real-world linter shape) → 0/26 on those. | CAPABILITY + self-confirming tests |
| fullLuna kcp-go | 0-byte | Not clean: quota storm + terminal `invalid_prompt` at supervisor turn 1. Empty DAG → `finalize_latest` finalized root → 0-byte patch delivered as `end_turn`/`error: null`. | INFRA (silent) |
| fullLuna abs | 0-byte | Not clean: Bedrock 500 survived all retries (Fast tier ≈ **1.4s total backoff**) at minute 4 of 90. Same silent empty finalize. | INFRA (silent) |

**ENV_LOAD ruled out in all 9** (longest shell 152s, zero kills/timeouts) despite
host load 125–235. Step-cap churn is real for DeepSeek (38–43 of ~50–68 windows
hit the 10-step cap vs ~0 for luna) but cost time, not correctness.

### The entrenchment mechanism (the residual design problem)

Same shape in 4 cases: an early worker silently resolves a spec ambiguity →
resolution enters the lineage as code **and self-authored tests** → descendants
inherit the full ancestor trajectory (framing included) → supervisor and
prefinalize attackers review the *diff* using artifacts generated under the same
premise → the wrong reading becomes unfalsifiable. 57–74% of worker wall-clock
went to `head_moved: false` confirmation passes. Diff-oriented review detects
internal inconsistency; a wrong reading is externally inconsistent and
internally perfect.

Intake's role, precisely: actively harmful in 1 of 5 semantic cases (Reader G
froze a wrong reading as authoritative spec), dead weight in 4 (correct
registers, consumed by nobody). The n=11 A/B (no measurable effect) had no
power to see either.

## 4. Fixes landed / in flight

### Anvil (`anvil-checkpoints` branch)

| Change | Status | What |
|---|---|---|
| Fatal TPD-quota classification | **landed `13a3634`** | `-tpd:` 429 bodies → non-retryable `FatalLlmQuotaError`; supervisor fallback aborts (`abort_quota_exhausted`) instead of laundering via finalize_latest. `-tpm:`/generic 429s stay retryable. 1276 tests pass. |
| A: intake removed entirely | **landed `2edf686`** | Both readers, gate, trace record, prompt injection — deleted, not gated. |
| B: test-file delivery guard | **landed `9666926`** | `ASGARD_TEST_FILE_GUARD=1`: final patch excludes worker-authored test files (language-generic pattern list) + defense-in-depth exclusion of pre-existing-test edits. Checkpoints unchanged; delivery-only; off by default. Audit trace of excluded paths. |
| ~~C: refuse test edits at tool layer~~ | **dropped by owner** | Benchmark overfitting — real-world use must allow editing tests. |
| Prompt norm | **landed `dea93d8`** | Base anvil system prompt (all sessions, unconditional): "Prefer fixing production code over weakening an existing test; if a test is genuinely obsolete, say so explicitly." |
| D: 5xx retry tier | **landed `0ab3358`** | LLM 5xx: Fast (4 attempts, ~1.4s total) → GatewayTransient (12 attempts, ~3.5min envelope). A 2-second Bedrock blip should not kill a 90-minute attempt. |
| E: empty-DAG loud-fail | in flight (last remaining) | Supervisor failure + no real checkpoint → abort loudly; never finalize root into a 0-byte "success." |

Counterfactual yield on the diagnosed set: B flips both DELIVERY cases (2
would-be solves), E+D convert both INFRA cases to honest retryable failures,
prompt norm *maybe* helps dateutil. The 4 ENTRENCHMENT/CAPABILITY cases are
untouched by any harness fix.

### Harness (`brokkbench-asgard-live`)

- `a369290` — runs cap {1,2} lifted to ≥1; `_BEDROCK_REGION` import fixed
  (deepswe engine owns its us-east-2 constant).
- AWS sweep tooling `81f1370` + follow-ups `fc02ff5`, `9a24bca`, `d630852`:
  per-task-sharded spot-VM orchestration for us-east-2 (113 VMs, 1 task/VM,
  sequential attempts), resumable state, smoke-tested live (VM up 29s,
  bootstrap 8s, rootless podman verified, terminated clean). Instance chain
  m7i.2xlarge → m7a.2xlarge → m5.2xlarge (all 8 vCPU/32 GiB — RAM held constant;
  flex types throttle sustained load and scored 1/10 on spot placement), no AZ
  pinning (EC2 placement chooses). AppArmor userns sysctl on VMs approved by owner.
  Remaining validation: one real single-task run end-to-end (blocked only on
  the fixed anvil binary).

## 5. Remaining gap vs vanilla — the honest accounting

Post-fix optimistic projection:

| Config | Clean solve rate |
|---|---|
| asgard-luna today (clean attempts) | ~19–21% |
| asgard-luna with all harness fixes | **~25–30% (projected)** |
| vanilla luna @xhigh | **57%** (51–67% on our clean subsets) |
| vanilla sol @high | **69%** (77% on our clean subsets) |

Perfect harness fixes close perhaps a third of the gap. **A ~2× deficit vs
asgard's own worker model remains**, and the diagnosed mechanism is
entrenchment, not capability: vanilla luna solves these tasks because each solo
run is an independent read of the instruction; asgard converts its first read
into an inheritance.

Sharper framing of "is asgard making sol dumber": sol never writes code in
asgard. Its 69% coding ability is spent reviewing fragments. All solving is
done by luna under worse conditions than solo luna enjoys (10-step windows,
frozen lineage, ~768k–1.5M cached tokens re-read per window, instructions
filtered through the supervisor). Current architecture ≈ paying sol prices to
run luna at half of luna's solo rate.

**Confound to kill first:** vanilla numbers come from a different scaffold
(mini-swe-agent) than asgard workers (anvil tool loop). Gap = asgard structure
+ scaffold delta, currently inseparable. The deconfounder is cheap: plain
single-agent anvil, luna@xhigh, same 113 tasks. If it lands ~57%, the deficit
is all asgard structure; if ~25%, the scaffold is the problem and supervisor
redesign is premature.

## 6. Design questions for Fable

1. **Entrenchment fixes** — proposed, in cost order:
   a. *Worker interpretive commitments* (prompt-only): final responses must state
      ambiguity resolutions explicitly, making them supervisor-reviewable objects.
   b. *Repo-idiomatic adversarial probes + spec-first finalize checklist*
      (prompt-only): attackers draw fixture shapes from the repo's test corpus,
      not the candidate's; supervisor re-reads the instruction before finalize.
   c. *`fresh: true` spawn option* (small code): worker gets instruction + tree
      only, no ancestor trajectory — an independent read that can surface
      lineage anchoring.
   d. *Interpretation audit at prefinalize* (protocol change): 1–2 fresh
      workers given only instruction + candidate diff, asked which alternative
      readings the diff forecloses; surfaced alternatives must be refuted from
      instruction text or get a rival implementation (diagnosed rivals were
      ~1–8 lines). Rationale: vanilla luna's 57% *is* the base rate of an
      independent read landing correctly — the audit samples that
      distribution and uses disagreement as the alarm. Should replace, not
      add to, the existing confirmation passes (57–74% of worker time,
      producing nothing).
2. **Is the ×3 rollout still the right spend?** Proposed instead: AWS pass 1 =
   deconfounder (vanilla-anvil luna ×1–2) + post-fix asgard-luna ×1, then decide.
3. **Step-cap economics for DeepSeek-class workers** — 10 steps binds hard for
   dsv4pro (~2/3 of windows truncated) and not at all for luna. Per-model cap?
   Supervisor already sets `max_steps` 12–30 in places; worth a policy.
4. **Quota/fleet ops** (FYI): luna input TPM increase (20M→~100M) needed for
   113-wide; sol daily TPD not self-service; consider concurrency shaping or
   staggered launch waves instead.

## Appendix: evidence pointers

- Diagnosed archives: `/tmp/claude-1000/-mnt-optane-anvil-checkpoints/6769d5b7-4894-4ff5-8bd6-daf155b418bc/scratchpad/diag/<sweep>--<task>/`
- Sweep results: `/mnt/optane/{fullLuna,fullDs}/results/`
- Grader reset behavior: `~/Projects/deep-swe/tasks/<task>/tests/grader.py` (`reset_paths(patch_paths(test_patch))`)
- Vanilla per-task data: `~/Projects/deep-swe/published-results/deepswe-v1.1/per-task-by-model-effort.csv`
- Quota evidence: `asgard_supervisor_fallback` records in poisoned traces; token rates from `asgard_usage_by_model`.
