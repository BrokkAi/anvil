# Structure probe (probe-sl): results, nine-failure audit, and the residual design problem

*2026-07-28. First asgard sweep with trajectory management (prefix_from,
prefinalize-fresh default, token observability), the permission-classifier fix,
and the delivery guard, on binary anvil-77bc359. Supervisor sol@high, workers
luna@xhigh, runs=1, 35 tasks = the 32 vanilla-luna-4/4 subset + diagnosed
entrenchment tasks (dateutil, aiomonitor, bandit). Per-task AWS spot VMs.*

## Headline

| | |
|---|---|
| SUCCESS | 22/35 |
| TESTS_FAILED (valid) | 9 |
| Wall-clock timeouts (censoring) | 3 (kcp-go, dynamodb-lazy, +straggler) |
| Infra-invalid | 1 (numba dirty-tree; both bugs since fixed) |
| **Valid rate** | **22/31 = 71%** |
| **True structure rate** (after guard-bug reclassification, below) | **24/31 = 77%** |
| Reference points | vanilla ceiling ~98% on this set; historical clean asgard ~20% |

Economics: canary attempt 6.3M luna input+cached vs the 34M/attempt fullLuna
baseline (5.4x); sol ~1.8M/attempt (probe total ~63M, far under TPD); typical
attempt 23-67 min vs cap-scraping. Sol supervising luna is no longer paying
sol prices to run luna at half rate — roughly 3/4 of the structure deficit
closed in one iteration.

Diagnosed-task scoreboard: actionlint FLIPPED (the 32-window entrenchment
case), dateutil FLIPPED (the test-rewrite judgment case); arcane, aiomonitor,
bandit did not (all three audited below).

## The nine failures (one Opus audit each; full reports in session transcripts)

| Task | Score | Class | One-line mechanism |
|---|---|---|---|
| mobly | 0/79 f2p, patch stripped | DELIVERY (guard bug) | Guard excluded pre-existing `mobly/base_test.py` (basename matched `*_test.py`); 549-line verified implementation silently deleted at finalize. FIXED (63007b6). |
| meriyah | 51468/51469 | DELIVERY (guard bug) | Guard stripped the task-mandated update to a pre-existing `__snapshots__/*.snap`; supervisor's finalize report *named* the update it believed it shipped. FIXED (63007b6). |
| aiomonitor | 51/53 | ENTRENCHMENT | URL-path ambiguity (same as historical). Recon worker's "Likely..." guess laundered into supervisor spawn briefs 44s later; correct rendering appears 0 times in 74MB. 34/38 spawns fresh — irrelevant, the frozen reading rode the instruction channel and the artifact. |
| arcane | 67/82 | ENTRENCHMENT | Spec-test author baked wrong RegisterRoutes caller-contract before implementation; worker overrode the supervisor's CORRECT instruction citing the self-authored test; supervisor let it pass. (Historical collision failure did NOT recur — guard excluded both colliding files, fix B field-confirmed.) |
| query-persist | 49/50 | ENTRENCHMENT | Supervisor ordered deletion of "obsolete" timestamp setState while node_modules was absent (nothing runnable); prefinalize surfaced the red pre-existing test; supervisor offered "minimally adapt the test" escape hatch; guard stripped the adaptation at delivery; nothing re-verified post-strip. |
| go-git | 15/17 | ENTRENCHMENT | LCS-interval merge misses overlapping edits in repeated lines. Fresh auditor w23 FOUND it (differential fuzz vs git merge-file, 3x FALSE CLEAN) → step budget forced "no gap" report → compactor elided the evidence ("(67 chars omitted)"; FALSE CLEAN: 11 trace hits, 0 supervisor-visible) → discarded. Surviving mutant closed by commissioning a test asserting current behavior. Sol reasoning tokens = 0 on turns 6/9/10/13 incl. finalize. |
| valibot | 218/219 | JUDGMENT | After 3x "do not weaken contract tests", supervisor at min 62 briefed a FRESH worker to "narrow only the assertion machinery"; worker (no memory of the prohibitions — fresh cuts both ways) deleted the exact repo-idiomatic toEqualTypeOf the hidden test uses. No prefinalize attack targets test edits. |
| superjson | 193/196 | JUDGMENT | 20 interpretive resolutions frozen at turn 1 into commissioned spec suites = self-confirming oracle; turn-7 "hardening" (obligation the instruction never states) BROKE a passing behavior; all three defects more elaborate than the reference. |
| bandit | 41/66 | CAPABILITY | Same called-functions-only reachability as history, narrower. All 7 prefinalize workers genuinely fresh — and every one read the shipped suites before writing fixtures; repo's own examples/ corpus read ZERO times; 0 of ~50 fixtures used the bare-def shape. Fix (b) confirmed unimplemented and decisive. |

## What trajectory management did and did not do

Worked, with field evidence:
- Mechanized uptake: prefinalize-fresh 100% across all nine runs (0-token
  prefixes confirmed in tool results); spawn-level fresh 60-90%; explicit
  partial prefixes used for continuation chains.
- Economics: 5-6x input reduction; attempts finish in half the budget.
- Direct save: query-persist — an omitted-prefix spawn on a deep lineage died
  at 296k tokens > model max; the supervisor re-spawned with prefix_from:"w29"
  (31k) and it completed. The dial, used as designed, under real pressure.
- Zero failures attributable to inherited-trajectory framing. The context
  channel of entrenchment is closed.

Did not work — entrenchment relocated to three channels fresh context cannot
touch:
1. **The instruction channel**: the supervisor's spawn briefs re-inject the
   frozen reading into every fresh worker (aiomonitor), and can reverse its
   own standing directives without any check (valibot).
2. **The artifact channel**: fresh verifiers read the shipped code/tests
   before probing and re-anchor on them (bandit fixtures, aiomonitor w33
   probing app.py's routes, arcane's spec-only worker importing the frozen
   test helper).
3. **The evidence-destruction channel**: step budgets force reports
   mid-disagreement; window compaction elides oracle output the worker never
   cited; discard erases the one worker that knew (go-git w23 — all three at
   once).

Fresh context also has a measured COST: the valibot weakening worker had no
memory of the protections that full inheritance would have carried. With
prefix_from, the supervisor's brief is the sole knowledge channel — its
quality is now the binding constraint. Sharper: sol's late-turn rigor
(0 reasoning tokens at finalize decisions) is where the remaining losses
concentrate.

## Fix list (mechanisms, from the audits; roughly ordered by yield/cost)

Landed already:
- Guard: never exclude pre-existing files (63007b6) — flips mobly + meriyah.
- Dirty-baseline absorption + startup loud-fail (9c5a09f) — fixes numba class.

High-yield, small:
1. **Guard-before-prefinalize** (meriyah, query-persist, mobly audits all
   converge): apply the exclusion set BEFORE the final verification batch so
   the full-suite worker tests exactly what the grader will see, and feed the
   exclusion list to the supervisor before finalize. Today the guard is an
   invisible post-verification mutation of the artifact.
2. **Assertion-downgrade gate** (valibot, query-persist): edits weakening or
   deleting assertions in run-authored or pre-existing tests become
   structured events; prefinalize bounces a candidate whose delivered tests
   are strictly weaker than tests the same run authored earlier
   ("you deleted your own oracle" — both versions are in checkpoint history).
   Red pre-existing test => mandatory attack entry on the belief that made it
   red; no test edit until discharged.
3. **Surviving-mutant closure rules** (go-git): a survived mutation emits
   machine-readable coverage_gap state; closing it with "add a test asserting
   current behavior" is refused; admissible closures are out-of-band oracle
   validation or deleting the mutated logic.
4. **Evidence preservation** (go-git): compactor preserves verbatim the
   stdout tail of a worker's last K tool calls and any output the report
   never cited; a prefinalize worker whose last tool call produced a nonzero
   oracle disagreement gets a step extension, and its report schema includes
   oracle_disagreements so "no gap" cannot be written over counterexamples.
5. **Repo-corpus probe sampling** (bandit — the mechanized fix (b)):
   prefinalize coverage gate bounces unless >=1 probe worker read pre-existing
   fixture/test files from the repo and re-expressed the new capability in
   their idiom.
6. **Assumption tagging** (superjson, aiomonitor): supervisor resolutions not
   backed by an instruction quotation are tagged assumptions; tests locking
   assumptions are non-evidence at prefinalize; unmandated "hardening" that
   changes behavior on a green tree requires an instruction citation or
   reverts.

Watch items, not yet mechanized: worker-contradicts-instruction as a bounce
condition (arcane); sol late-turn reasoning collapse (consider effort floor on
finalize turns); barrier straggler idle (80 worker-min in go-git).

## Verdict

The architecture question from the deconfounder dossier is answered: with the
scaffold vindicated, trajectory management landed, and the classifier tax
removed, supervised asgard runs at ~77% true structure rate on tasks its
worker solves ~98% solo — up from ~20%. The residual is no longer "workers
drowning in inherited context"; it is supervisor epistemics: briefs that
launder guesses into specs, evidence that dies in compaction and step budgets,
and late-turn decisions made without thinking. Every fix in the list above
targets that layer, and none of them is a prompt.
