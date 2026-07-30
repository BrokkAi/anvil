# Asgard v2: provisional report

*Written 2026-07-29/30 as a pick-up-in-two-weeks document. Covers what was
measured, what was fixed, what the architecture appears to be, and the open
questions with the evidence needed to answer them. Companion dossiers in
`dossier/`: `probe-sl3-postmortem-2026-07-29.md` (seven failure audits),
`heldout-a-findings-2026-07-29.md` (generalization result),
`followup-tickets-2026-07-28.md` (filed as mjolnir#506, anvil#311-313).*

---

## 1. The bottom line

Asgard = a supervisor model (sol) directing worker models (luna) through a
barrier-batch DAG of git checkpoints. Over four days it went from ~20% to
80% on a 35-task tuning set, then measured **63% on 27 never-iterated
held-out tasks: 1.15x vanilla luna, 0.88x vanilla sol.**

The architecture, as measured: **worker-bound, not supervisor-lifted.**
Grouping held-out tasks by how well vanilla sol does alone:

| vanilla sol | n | asgard | vanilla luna | vanilla sol |
|---|---|---|---|---|
| 4/4 | 13 | 69% | 58% | 100% |
| 3/4 | 4 | 100% | 69% | 75% |
| <=2/4 | 10 | 40% | 45% | 32% |

Supervision adds ~11 points over the worker alone and gives up ~31 points
against the supervisor alone. Paired: 4 tasks lost that sol solves 4/4
alone, 2 won where sol scores <=1/4. It behaves like *the worker plus a
small boost*, uniformly — not like sol's judgment executing through luna's
hands. **Sol's competence does not transfer downward through direction.**

That is the central finding to pick back up from. Everything below is either
evidence for it, or work that had to happen before it could be measured.

## 2. Measurement history (all on the same 113-task DeepSWE corpus)

| sweep | config | result | note |
|---|---|---|---|
| fullLuna/fullDs | pre-session | 8% / 3.5% | **unusable**: 186/225 attempts ran with a dead supervisor (sol daily TPD quota exhausted; 429s were classified retryable and the fallback laundered arbitrary checkpoints) |
| vluna-aws | vanilla anvil, luna | 61% raw / 66% valid | the deconfounder: anvil scaffold is *not* the problem (published mini-swe-agent luna = 72.2% on same tasks) |
| vluna-rerun | vanilla, 11 timed-out tasks, 3h cap | 6/21 = 29% | those tasks are anvil-hard, not merely slow; corrected full-set vanilla ~61% vs 72.2% published (~11pt scaffold gap, concentrated in that cluster — **open item**) |
| probe-sl (35 tasks) | asgard, trajectory mgmt + classifier fix | 22/35 = 63% | 71% valid; 9 failures audited |
| probe-sl2 (35) | + supervisor-epistemics layer | 21/35 = 60% | flat; 3 in / 4 out — flip-band |
| probe-sl3 (35) | + simplification (register deleted) | **28/35 = 80%** | all 3 chronic timeouts converted |
| mini-attractor (7x2) | same binary | 8/14 = 57% | vs 21% pre-redesign on identical tasks |
| **heldout-A (27)** | + time leases | **17/27 = 63%** | **1.15x luna, 0.88x sol** — the generalization number |
| cds2 (35) | codex-sol + deepseek-v4-pro | 13/35 = 37% | 59% excluding timeouts; **37% timeout rate vs luna's 6%** |

Task sets (in `~/Projects/brokkbench/`):
- `deepswe-probe-sl.tasks` (35) — **retired to regression duty**, 3 iterations of contamination
- `deepswe-heldout-A.tasks` (27) — used once, above
- `deepswe-heldout-B.tasks` (27) — **SEALED**, stratified sibling of A (expected vanilla luna 56.5% vs A's 54.6%)
- `deepswe-luna-unsolvable.tasks` (24) — **SEALED**, the beat-vanilla-sol population; vanilla luna 0/4 by construction, vanilla sol 33/96 = **34% per-attempt**, paired per-task data in the published CSV

## 3. The falsifiable prediction (the next experiment)

If asgard is "worker + small boost", then on the 24 luna-unsolvable tasks
(luna 0/4, sol 34%) asgard should land **well below 34%** — the
beat-vanilla-sol test fails for structural reasons, not fixable bugs.

This discriminates two worlds:
- **(a) fixable**: supervision should transfer competence and doesn't yet;
  the remaining postmortem fix families are worth building.
- **(b) structural**: supervision cannot transfer competence across a
  capability gap, and the value proposition must be restated — parallel
  breadth, cost (luna workers are far cheaper than sol), throughput — with
  the sol-4/4 deficit as the price of admission.

Run the 24 before investing further in supervisor mechanisms.

## 4. What the failure audits found (12 audits across two sweeps)

Three laws, each confirmed repeatedly and each violated by something we
shipped:

1. **Restatement survives its prohibition at pressure points.** Doctrine
   says briefs never restate task requirements. Three of four real sl3
   losses trace to supervisor prose overriding task text anyway:
   arcane (a **fabricated** task-text citation — "task says...uses active
   baseline"; it does not), superjson (the wording tension resolved inside
   *both* implementers' briefs, destroying pair independence, then
   re-affirmed against two review workers' runnable repros), ofetch (a
   paraphrase that silently *added* `onRequest` to the spec's enumerated
   hook list; the worker then inverted a sibling's correct passing test
   citing "the audit requirement").
2. **Visibility without obligation is ignored under pressure.**
   Schema-required acknowledgments get 100% uptake (mutations arrays,
   max_steps). Display-only surfaces get skipped at the decisive moment:
   dateutil's `changed_oracles` rendered a deleted pre-existing assertion
   into sol's context and drew an *empty* `save_checkpoint`; both of
   arcane's spec authors hit CAPPED and were never continued.
3. **A step lease is not a time lease.** One step can be `sleep 500`. numba
   burned 43 minutes inside 9 of 75 steps; query-persist 38 inside 31 of 32.
   Both died holding verified, mutation-checked candidates (at 46 and 56
   minutes) while an *environment-priced* verification obligation consumed
   the back half — numba's suite cannot finish on 2 CPUs; TanStack's root
   `pnpm test` is a CI gate (lint + four TS matrices) that never exited
   clean. Both supervisors escalated the timeout instead of taking the
   `full_suite_skipped` hatch (`skipped_reason_present: false` in all four
   coverage records).

Also: attractors are **model-level and shared**. aiomonitor failed 5x
identically across every config (a URL-shape reading; the correct rendering
appears in-trace and loses anyway); dateutil's UNTIL-validation belief
recurred in independent runs including as an over-specified spec-test
assertion *the reference solution also fails*. Both spec-test authors,
spawned at deliberately contrasting vantages, independently wrote the same
wrong reading — dual reading cannot break a prior both models hold.

## 5. What shipped (anvil, branch `anvil-checkpoints`)

Chronologically, with the evidence that motivated each:

- **Permission-classifier fix** (`51b803d`, `3d3d492`, brokkbench `4449f26`).
  mjolnir's `--permission-mode bypassPermissions` never reached anvil, so
  every benchmark session ran in Auto mode and paid an **untraced** classifier
  LLM call per gated tool call — profiled at **52% of attempt wall-clock,
  65% of all LLM calls, plus a 19% spurious-denial rate**. Fixed via
  `BROKK_ACP_PERMISSION_MODE` env default; classifier decisions now traced.
  This retroactively reframes every pre-fix benchmark number in the project.
- **Trajectory management** (`2da6cb5`, `867969f`). `prefix_from` = full |
  checkpoint | "none": supervisor-controlled context inheritance, with
  cache-safe assembly order, merge-base briefings, per-window rendered
  tokens, prefinalize defaulting to fresh. Closed the inherited-context
  channel of entrenchment completely (0 failures via trajectory framing
  since; 5-6x token reduction; one live save from a 296k-token overflow).
- **Delivery guard `pre_existing` fix** (`63007b6`). The guard stripped
  *pre-existing* files matching test patterns: mobly's `base_test.py` (a core
  production module) and meriyah's task-mandated snapshot update — two probe
  solves converted to losses by our own guard.
- **Dirty-baseline absorption + startup loud-fail** (`9c5a09f`). A task image
  shipping a pre-modified file made asgard abort, and the abort *laundered*
  as a normal `end_turn` with the env diff graded as the agent's patch.
- **Simplification** (`c28a738`, `99be76e`). Deleted the ambiguity register
  and the spec-pinning doctrine (net -172 production lines). Replaced with:
  briefs-never-restate, adjudication-of-divergences as the seat of authority,
  targeted dual reading, `max_steps` **required** (ceiling 75 = half measured
  vanilla p75 of 147), continuation-on-cap, CAPPED markers, measured step
  bands. Motivation: the register *formalized* the restatement channel —
  aiomonitor's frozen wrong reading carried a verified quote about a
  different route.
- **Time leases** (`e82c8ea`). Optional `max_minutes` (default 15, ceiling
  30), enforced between turns; in-flight tool calls never interrupted (shell
  clamps at 600s, so worst case ~25min); `CAPPED (time)` handoff; spend line;
  full-suite doctrine reworded; finalize requires
  `modified_pre_existing_tests` (harness computes truth, bounces once).

Deliberately **not** built, as overfit-risky content classification:
verbatim-span checking of "requirement-shaped" brief clauses,
assertion-inversion detection, `red_pre_existing[]` parsed from test output,
mutation provenance tracking. Each would need prose classification aimed at
three single-test residuals on a thrice-iterated set.

## 6. Harness bugs fixed (brokkbench + mjolnir)

A recurring shape worth naming: **a failed observation collapsed into a
definite negative.** Three instances:
- quota 429s classified retryable → fallback laundered arbitrary checkpoints
  as normal `TESTS_FAILED` (found pre-session, fixed `13a3634`)
- a region-mismatched `describe-instances` read as "VM terminated" → the
  orchestrator terminated four of its own healthy VMs (`258a192`)
- a **throttled** `describe-instances` (22-VM width) read the same way →
  three healthy VMs killed mid-attempt, tasks restarted (`10a7e737`)

General lesson: "unknown" must be a representable state callers handle. This
is the same principle the sl3 postmortem reached about supervisor evidence.

Also fixed: `extra_secrets` TypeError that broke *every* deepswe attempt
(`fbcb8d3`); AWS staging parent-vs-content semantics + preflight (`504c2ed`);
SSH launch backgrounding that held the channel for the harness's lifetime
(`527e996`); `ASGARD_TEST_FILE_GUARD` forwarding (`1e21ca7`); mjolnir
permission-mode plumbing for interactive and **custom** adapters — benchmark
runs are always `custom/bpr-agent/...`, which was the one config hole
(mjolnir `83b35b8`, pushed to origin/master).

Operational notes for whoever picks this up:
- luna input TPM 20M caps concurrency at ~27 attempts (asgard burns ~727k
  tok/attempt-minute pre-trajectory-mgmt, ~6.3M total/attempt after)
- sol daily TPD is **not** in Service Quotas and killed a full sweep once;
  ~1.8M/attempt after the fixes, so ~35 attempts/day is safe
- podman containers carry **no per-sweep label** — only `Job:BPR-<image-hash>`
  and an epoch in the container name. Cleanup is time-window-based only;
  adding the sweep label would make it a one-liner filter (**small open
  ticket**)
- the orchestrator's `state.json` persists a sticky `attempts` counter that
  blocks resume with "FAILED after 4 attempts: unknown"; clear those entries
  to re-run failed tasks

## 7. Open questions, ranked by information value

1. **The 24 luna-unsolvable tasks** — tests the section-3 prediction. Highest
   information per token in the project right now.
2. **Held-out B (27)** — confirmation that A's 63% / 1.15x wasn't itself a
   lucky draw. Cheap, uncontaminated.
3. **Does the attractor class yield to external oracles?** Repo-idiomatic
   fixtures (bandit's `examples/` corpus was read *zero* times while ~50
   agent-authored fixtures all used the one shape that worked), differential
   references (go-git's `git merge-file` found what five verification workers
   could not — and the system then destroyed the evidence via step budget,
   compaction, and discard). This is the only demonstrated attractor-killer
   and it is unbuilt.
4. **The ~11-point vanilla-anvil vs mini-swe-agent scaffold gap**,
   concentrated in ~11 identifiable tasks. Leading suspect: anvil's step
   granularity (~151 small steps vs mini-swe's ~64) interacting with those
   tasks' shapes.
5. **Does the time lease recover ds-class workers?** cds2 timed out on 37%
   of attempts *without* the lease (luna: 6%). A ds rerun on the lease binary
   is the cheap test; ds excluding timeouts was already 59%.
6. **Alternative architectures.** If (b) in section 3 holds, the question
   becomes what *does* lift a weaker model. Candidates with evidence behind
   them: the replay-matrix result (independent runs + adjudication recovered
   87% of failures, but selection alone is non-discriminating — 94%
   contamination), and mjolnir's primary/subagent split (see section 8).

## 8. In flight at time of writing

Investigating mjolnir's **thor/eitri split** as an alternative lifting
design, per owner request, to be run on the 24. Findings so far:
- The pinned benchmark binary `mj-6147059` exposes `--thor` / `--loki` /
  `--eitri` role flags. Current mjolnir master **removed them** (`ad517c8`
  "replace the council with claude-code-style subagents") in favour of
  `--model MODEL[+EFFORT]` (primary agent) and
  `--subagent-model MODEL[+EFFORT]|disabled|none`.
- Eitri in current master is a **read-only intent analyst** for discrete
  review ("Do not modify the workspace or delegate"), plus parallel explores
  (`max_parallel_explores`). Config surface only, no CLI flag.
- Therefore running this through bpr needs (i) a current-master mj build and
  (ii) `mjolnir_command()` in `bpr_agent_engine.py` updated from
  `--thor/--loki/--eitri` to `--model/--subagent-model`.
- The interesting configuration for *lifting luna* is luna as the primary
  agent holding the pen with sol as the subagent model — inverting asgard's
  authority structure (weak model works, strong model advises on demand),
  which directly tests the section-3 finding that competence does not
  transfer downward through direction.
