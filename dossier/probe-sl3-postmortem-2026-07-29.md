# probe-sl3 postmortem: 28/35, the simplification validated, and the residual named

*2026-07-29. Five Opus audits over the seven non-successes of probe-sl3
(binary anvil-99be76e: no register, briefs-never-restate, adjudication rule,
targeted dual reading, required max_steps<=75, continuation-on-cap, CAPPED
markers). Full reports in session transcripts; this is the synthesis.*

## Result

28/35 raw (80%), 28/33 valid (85%) vs 22 and 21 on the identical set.
Chronic-timeout family fully converted (dynamodb-lazy, pebble, kcp: AF->S).
aiomonitor fell after five identical failures. Attractors killed this
binary: query setState (mini 2/2), superjson presence-semantics (adjudicated
correctly in its sl3 run), aiomonitor routes (once).

## The seven, classified

| Task | Score | Verdict |
|---|---|---|
| arcane | 81/82 | JUDGMENT: supervisor promoted an audit's "policy is missing" note to a binding order with a FABRICATED task-text citation ("task says...uses active baseline" — it does not); worker implemented, deleted the conflicting spec assertion; a commissioned mutation then "caught" the invented predicate. Third distinct single-point failure on this task — no repetition, fresh over-specification each time. |
| superjson | 195/196 | JUDGMENT: old attractor KILLED; new one-clause defect from a turn-1 spec-author coin flip that the supervisor (a) saw as a wording tension, (b) resolved inside BOTH implementers' briefs (independence destroyed), then (c) re-affirmed against TWO review workers' runnable repros by re-quoting the same clause. w10 discarded. |
| dateutil | 2101/2102 | THE attractor, full chain: entered as spec-A's over-specified round-trip assertion (reference solution fails it too); regression shipped from a session that never ran the legacy suite; first red declared "intentional"; supervisor ordered the test rewrite without ever seeing the failure; changed_oracles rendered the deletion into its context -> empty save_checkpoint; prefinalize's 8 attacks missed the removed oracle; guard retained the file without content diff. Five enumerated adjudication misses. |
| ofetch | 59/60 | REGRESSION (redesign-attributable): supervisor brief paraphrased the spec's hook enumeration and silently ADDED onRequest; worker cited "the audit requirement" over the task text AND a sibling's correct passing test, inverted the assertion (test name still contradicts its body), locked with regression test + mutation. |
| yjs | 239/240 | BAND NOISE, clean bill: triple-independent authorship all wrote delete->set, none set->delete; every doctrine element behaved correctly. |
| numba | AF | ENVIRONMENT-COST timeout: verified mutation-checked candidate at 46 min; remaining 72 min consumed by an unfinishable full-suite obligation (numba suite >45min-and-running on 2 CPUs). Supervisor saw one 590s timeout and ESCALATED ("use at least a 30-minute timeout") instead of taking full_suite_skipped. Delivered patch = 348 bytes = the baseline commit only. |
| query-persist | AF | Same class: candidate (reference-equivalent file set, packages green, 2 mutants caught) at 56 min; root `pnpm test` is a CI GATE (lint + 4 TS matrices) that never once exited clean — the full-suite worker's own lint fix damaged the candidate; 38 min single-straggler supervisor idle; died polling. 0-byte patch. |

## Cross-cutting findings

1. **Restatement survived its prohibition, at the pressure points.** Three of
four real losses trace causally to supervisor prose overriding task text:
fabricated citation (arcane), pre-resolution in briefs (superjson),
paraphrase-widened enumeration (ofetch). The doctrine measurably reduced
restatement (aiomonitor's briefs were clean and it passed) — but prompt-level
prohibition leaks exactly where every prompt-level rule here has leaked.

2. **Visibility without obligation is ignored under pressure.** The pattern
is now exact: schema-REQUIRED acknowledgments get 100% uptake (mutations
arrays, max_steps); display-only surfaces get ignored at the decisive moment
(changed_oracles diff -> empty save; CAPPED markers on both arcane spec
authors -> no continuation; runnable repros -> re-affirmed over). The
half-measure of "make it visible" is spent; what remains needs to be typed
and blocking.

3. **A step lease is not a time lease.** One step can be `sleep 500`. w18
burned 43 minutes inside 9 of 75 steps; w21 38 minutes inside 31 of 32. And
the full-suite doctrine turns a repo-priced suite into a guaranteed loss
because its escape hatch is phrased as a concession — all four coverage
records this sweep: skipped_reason_present false after observed timeouts.

## Fix families (per-audit convergent; design-stage, owner call)

A. **Citation integrity (mechanical, event-driven — not a register):**
   - A brief clause that constrains observable behavior must quote a verbatim
     task-text span; harness rejects the spawn otherwise (kills arcane's
     fabricated citation and ofetch's widened enumeration by construction).
   - An order to edit a pre-existing test carries a mandatory quotation
     naming the behavior superseded; rejected otherwise (dateutil).
   - A ruling contradicted by a runnable repro cannot be re-affirmed by
     restating the same clause — new evidence required (superjson).

B. **Typed blocking artifacts replacing display-only surfaces:**
   - red_pre_existing[] on window records from actual test output; a save
     touching one requires an explicit ruling (dateutil's empty save).
   - Assertion inversion/deletion in any existing test = mandatory
     adjudication event (ofetch, valibot-class).
   - Suite-vs-repo cross-run when spec authors finish: an in-run suite that
     requires a pre-existing test to go red is a turn-1 divergence,
     adjudicated before implementation exists (kills dateutil at root).
   - Contested reading => two branches + discriminating test; supervisor may
     not resolve in prose (superjson; the run's own turn-5 brief proves it
     could articulate both branches).
   - Mutations cannot certify predicates introduced by same-run supervisor
     order (arcane's circular "caught").

C. **Time/cost leases (prod-appropriate; no deadline disclosure):**
   - Per-window wall-clock lease enforced by harness -> CAPPED(time) handoff,
     same semantics as the step cap.
   - A full-suite run that has timed out once is discharged as
     full_suite_skipped with the observed evidence; never retried longer.
   - "Full suite" = changed packages' test targets, not the repo CI gate.
   - Aggregate spend line beside the budget line.
   - Rejected as benchmark-shaped: telling the supervisor remaining time.

D. **Guard content-awareness:** diff retained pre-existing test files against
   base_commit; removed/weakened assertions bounce (dateutil's deleted
   assertion inside a retained file).

## What worked and should not be touched

Adjudication with honest citations (3 correct rulings in arcane's own run;
superjson's attractor killed by one); targeted dual reading (multiple
surfaces saved across runs); continuation machinery; baseline absorption;
loud failures; the guard's pre_existing file-level fix; band noise now
consists almost entirely of single-test misses at 0.98+ partials.
