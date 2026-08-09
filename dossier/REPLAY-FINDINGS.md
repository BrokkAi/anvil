# Replay matrix: offline cross-sibling test transplantation — findings

Follow-up to WRITEUP.md §7 ("open directions"). We built the offline replay proposed as
the cheap first experiment for the cross-sibling divergence direction, ran it over the
four recent sweeps, and adversarially adjudicated a stratified sample of the results.
Everything here is reproducible from `/mnt/optane/replay-matrix/` (SPEC.md documents the
harness; `metrics.json` has per-impl rows; `analysis/*/verdicts.json` the adjudications).

## Method

- Every finalized `model.patch` from asgard24-ff, asgard25-ff, asgard26a-fp, asgard26b-fp
  (160 implementations, 20 tasks, hidden outcomes known) split into `impl.patch` +
  `tests.patch` by path classification (`*_test.go`, `tests/`, `*.test.*`, `test_*.py`,
  `testdata/`, …).
- Every run's test set executed against every sibling implementation of the same task,
  inside the task's own container image (1,224 cells). Per-language precise selection:
  Go = extracted `TestXxx` names with the task's build tags; vitest/jest/pytest = the
  author's test files; yjs = whole lib0 suite with baseline-delta attribution.
- vitest/jest exit codes reclassified from log summaries (vitest exits nonzero on
  unhandled errors even when all tests pass).
- Cell classes: pass / fail / surface_error (build, import, collection failures) /
  timeout / no_tests.
- 61 divergence cells (9 tasks, stratified) adjudicated by independent readers against
  task text + the impl's official verifier failures.

## Mechanical results

| metric | value |
|---|---|
| recoverable-failure rate (hidden-fail impls rejected by ≥1 sibling test set) | 106/122 = **86.9%** |
| oracle contamination (hidden-PASS impls rejected by ≥1 sibling test set) | 32/34 = **94.1%** |
| common-mode gap (hidden-fail impls with no rejection and no surface error) | 0/122 |
| authors whose own tests pass their own finalized impl | 124/156 (**~20% finalize red**) |

No operating point separates the populations mechanically: restricting to self-passing
authors and requiring ≥3 rejections gives 57% recovery vs 68% contamination. Mean
author-suite fail-rate per impl (magnitude) medians 1.00 (hidden-fail) vs 0.29
(hidden-pass) but quartiles overlap. Go single-package tasks (actionlint, goreleaser)
are surface-error-dominated: candidate tests reference impl-specific symbols and
cross-compilation fails, so almost no assertion-level signal there.

**Conclusion 1: automatic test-union or majority-gating is dead on arrival.** Divergence
is ubiquitous and pass/fail alone does not discriminate. Any production mechanism must
be adjudication-led.

## Adjudicated results (61 cells, 9 tasks)

**Recovery cells — is the rejection pointing at the graded defect?** (35 verdicts)

- 22 same_defect, 3 related, 6 unrelated, 4 coupling_noise.
- would_flip: **15 likely, 8 possible, 12 no**. Per implementation: 11/18 sampled
  hidden-failing impls had ≥1 rejection whose adoption would likely have flipped the
  official outcome; 4 more had a possible flip; 3 had none.
- The likely-flips reproduce graded failures with striking precision: narwhals' invalid
  `percentile_cont` window aggregate and broken pyarrow `over(order_by=)`; true-myth's
  `zipWith` Nothing-crash; happy-dom's abort handler that sets a flag but never calls
  `reader.cancel()`; ofetch's process-wide singleton state map; returns' inverted
  `apply` convention and decorator placement.
- Complementarity is real: in psd-tools, two different authors each reproduced a
  different half of the impl's four official failures — the union covered all of them.
  In returns, one sibling's tests isolated all three real defects of a failing impl.

**Contamination cells — why do sibling tests reject impls that passed grading?** (26 verdicts)

- 12/26 impl_coupling (46%): tests hard-code author-private conventions — invented
  marker symbols/attributes, TS generic-parameter order, module paths, exact error
  strings. Pure noise, but with a recognizable *surface-shaped* signature
  (AttributeError / ImportError / cannot-find-export dominate).
- 7/26 stricter_defensible (27%): the tests caught **genuine violations of the literal
  task text that the official suite never exercises** (happy-dom's bodyUsed-ordering
  bug and unguarded Request path; httpx case-insensitive domain matching on accessors).
  On the benchmark these don't move reward; in real deployments these are exactly the
  catches you want.
- 7/26 wrong_reading (27%): underspecified boundary edges resolved differently by
  different authors (`halfOpenMaxRequests=0`, string-truthy option values, tuple-vs-
  scalar argument shapes) — the same quantifier/boundary family the intake ambiguity
  directive targets. Task-text adjudication resolves these correctly; in one case the
  wrong-reading author itself failed official grading on exactly that reading.

**Blind spots (where cross-testing cannot help):**

- Shared-convention blindness: both failing returns runs independently invented the
  same inverted `apply` convention; within that pair, cross-testing is silent. Only an
  author holding the *other* convention exposes it.
- Common-mode gaps: yjs's shared merge-path defect (conflict detection never checks
  pre-existing doc state during `applyUpdate`) is untested by every sibling suite; the
  one test aimed at it used tautological soft assertions. query-persist's graded
  infinite-query defect likewise untouched by any sibling's executed tests.

## Design implications for the production mechanism

1. **Transplantation is mechanically cheap and the true-positive density is high**, but
   the discriminator must be adjudication of each divergence against the task text —
   never pass/fail counts, never majority.
2. **Coupling noise is auto-triageable**: most of it fails with missing-symbol /
   missing-export / import-error signatures rather than assertion failures. A
   surface-vs-assertion split (which the replay already computes) sends most noise to a
   cheap "incomparable — rewrite through the public surface or dismiss" path.
3. **The gate must be bounded.** ~20% of trajectories finalize with their own tests red,
   contamination is dense, and the pro arms already demonstrated never-finalize
   wall-clock deaths. One adjudication round with a hard cap, then proceed.
4. **The harness cannot know how to run tests in production** (no per-task config
   exists outside the benchmark). The portable shape: the harness does the git-level
   transplantation (universal) and spawns a verification worker per sibling pair with
   the transplanted files and the test commands *observed in that sibling's trajectory*
   (structured tool-call history, not content parsing); the worker runs them and
   reports failing assertions into the batch review.
5. **Cross-testing needs genuine redundancy to bite.** The replay's richness came from
   independent runs. In-run, sibling checkpoints implementing the same contract exist
   only when the supervisor spawns redundant pairs / differential workers — doctrine
   that currently gets ~0 uptake as prose. The mechanism makes those spawns meaningful
   (the harness performs the comparison the prose asked the supervisor to arrange), and
   the shared-convention and common-mode blind spots above are the argument for pairing
   it with convention-diverse instructions rather than identical duplicate prompts.
