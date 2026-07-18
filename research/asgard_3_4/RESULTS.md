# Asgard 3/4 results

Status: the 40-call Q3 captured-context replay and the first 20-run live pilot are
complete. The Q3 promotion decision is **no-go**. Q4's shadow-survivor calibration
also reached an early **no-go** after 12 valid live studies made the 90% gate
mathematically unattainable.

## Direct verdicts

- Stateful/delta supervision has a large structural opportunity, but none of the four
  compact representations met the combined protected-endpoint, protected-live-result,
  and cache-economics gates. Keep `full` as the default.
- Explicitly naming a short window a probe is not a new execution mechanism. The
  archived decisions seen so far use three lanes throughout and provide no survivor
  counterfactuals, so they cannot estimate shallow-probe recall.
- A true survivor tournament is not justified. Prototype B retained the eventual
  winner after two probe steps in only 7/12 valid live studies (58.3%). Five late
  bloomers were killed; even eight perfect remaining studies could reach only 75%,
  so the pre-registered 90% gate failed by mathematical futility.
- `recent-exact-tail` had the best live score (3/4), but it lost the protected Returns
  solution, failed the replay endpoint gate, and raised uncached input per routing
  window by 18.4%. It is an interesting baseline, not a production candidate.
- `checkpoint-plus-delta` was the strongest structured mode (2/4) and almost held
  uncached input per window flat (+0.6%), but it also lost protected Returns and failed
  the replay endpoint gate. The frozen-checkpoint hypothesis is not supported strongly
  enough to promote.

## Facts

### Existing 40-run v9 cohort

The committed research brief reports that v9 reduced total raw tokens by 24.5% and
candidate tool-loop raw tokens by 31.9% versus v6, while the mixed non-candidate
residual rose 14.9%. Both cohorts succeeded on 9/40 runs, with only three common
successes and six flips in each direction. These totals motivate compression but do
not isolate supervisor usage.

Retrospective parsing of the same 40 v9 archives found 861 ordinary supervisor dossier
assemblies. Across the logged variable dossier components:

- selected initial history: 23,092,560 bytes (33.5%);
- accumulated selected windows: 29,631,176 bytes (43.0%);
- current candidates: 16,146,950 bytes (23.4%);
- old selected trajectory total: 76.6% of variable bytes.

The historical fraction rises from 37.7% at window 1 to 90.2% after window 20. This
is an impossible removal ceiling, not a savings estimate: it excludes the fixed
system/task/checklist/tool schema, audit-turn additions, compact replacement state,
output, and provider cache pricing.

### Eight-archive reproducible corpus

`extract_archive_corpus.py` processed all eight cases in
`scripts/asgard_live_regressions.json`:

- 110 ordinary dossier telemetry windows;
- 19 windows with aligned structured ordinary decisions;
- 91 older telemetry-only windows, retained with `decision_missing`;
- 8,165,022 measured component bytes;
- 4,834,357 bytes (59.21%) in selected initial plus selected windows;
- 3,191,010 bytes (39.08%) in accumulated selected windows alone;
- 3,330,665 bytes in current candidates.

All 19 aligned decisions used three lanes. Seventeen expose a next-window step count:
one chose 3 steps, ten chose 5, three chose 6, two chose 7, and one chose 8. This
sample shows neither spontaneous one-lane routing nor 1-2-step probes.

### Replayability gap

Old v9 archives trace candidate requests/responses and final structured supervisor
decisions, but not the supervisor's exact requests, audit exchanges, responses, or
per-call usage. Therefore old decisions cannot support faithful same-state replay.
The new opt-in capture records those missing fields for fresh runs.

### Protected-endpoint replay stop gate

The first fresh captured-context replay exercised window 14 of the successful
Returns control run, a protected endpoint at which the full supervisor selected
lane 0 and declared completion. All four compact prompts selected the same lane and
removed 44.4%--58.6% of rendered prompt bytes (35.7%--49.0% of raw input tokens),
but only `decision-log-only` reproduced the terminal decision. `latest-state`,
`checkpoint-plus-delta`, and `recent-exact-tail` instead funded one more lane for
two, two, and three steps respectively. Those three modes therefore scored 0% on
the required protected endpoint-agreement gate despite 100% winner agreement.

The pre-registered stop rule fired after four calls. The confirmatory result
therefore remains a stop/no-go. After an explicit later approval, the remaining
36 calls were run as exploratory follow-up rather than silently redefining the
gate. This is evidence that winner agreement alone is too weak for routing
compression: a false continuation changes cost and can expose a correct endpoint
to additional edits. Obligation preservation still requires manual review.

### Complete 40-call captured-context replay

The completed batch covers ten captured ordinary-routing states from PSD Tools
and Returns, with one call per state for each compact mode. It contains 40 unique
records: 21 direct selections and 19 fail-closed fallbacks where the compact
supervisor requested unavailable repository audit tools. All 21 direct selections
matched the full supervisor's winner, but only 7 matched the entire recorded
winner/completion/funding tuple. Nine direct selections flipped completion: eight
continued after the full control stopped and one stopped before the full control.

Treating every fallback as an execution of the captured full-control decision,
the mode-level results were:

| Mode | Fallback | Winner agreement | Winner + completion | Candidate-count agreement | Step agreement | Raw-input reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint + delta | 50% | 100% | 70% | 70% | 60% | 40.7% |
| decision log | 40% | 100% | 80% | 80% | 60% | 34.3% |
| latest state | 40% | 100% | 80% | 80% | 70% | 43.6% |
| recent exact tail | 60% | 100% | 80% | 70% | 70% | 34.6% |

Every mode failed the automated protected-decision gate: each reached only 80%
effective winner-plus-completion agreement across the five sampled decisions from
the protected Returns identity, versus the required 100%. No mode is
recommendation-ready, independent of the still-required human obligation review.
The complete machine-readable analysis, including disagreement and fallback rows
and usage vectors, is in `supervisor_replay_40_analysis.json`.

## Implemented experiment controls

- `full` remains the byte-identical default control.
- `latest-state`, `checkpoint-plus-delta`, `decision-log-only`, and
  `recent-exact-tail` are research-only environment-selected modes.
- Compact modes make the next `state_summary` an explicitly cumulative replacement.
- Canonical execution evidence is mechanically rendered with its surrounding window,
  so repeated `L1` identifiers remain window-qualified in context.
- Checkpoint mode freezes both summary and ledger through a global checkpoint window;
  exact later selected windows are deltas. A unit test proves the checkpoint message
  is byte-identical between intervening windows.
- Every routing turn records chosen versus counterfactual full prompt bytes and token
  estimates. Opt-in capture records exact request/response/tool/usage data.
- Completion-review construction is not routed through the compression switch.
- A conservative serial fast path now provides the separately required compact,
  one-response selector with no audit catalog. It is opt-in, mechanically gated,
  cannot declare completion, traces every eligibility/fallback reason, and charges
  both calls when it falls back. It has no live eligibility or agreement observations
  yet and is not part of the Q3 pilot result above.
- Prototype B now has a strict shadow-study trace schema, executable recall scorer,
  and a research-only fixed core protocol: three lanes receive a forced one- or
  two-step probe, all three continue autonomously for five equal steps, and an
  opaque-label review chooses the endpoint before the first repository
  synchronization. Probe rankings classify architectural/contract versus cosmetic
  distinctions. Twelve live studies passed every protocol invariant; seven retained
  the final winner and five did not. The production/default tournament remains
  unchanged.

## Q4 shadow-survivor futility result

The first 12 two-step studies completed the decision-relevant shadow protocol before
their surrounding task runs were stopped. All 12 had exhaustive three-lane ground
truth, equal five-step continuations, isolated branches, blinded endpoint review,
complete usage accounting, and no scorer violation. Probe top-2 retained the blinded
final winner in 7/12 studies (58.3%; 95% Wilson interval 31.95%--80.67%). Among the
five rank-3 late bloomers, three probe distinctions were architectural/contractual or
mixed and two were cosmetic; failures were not confined to cosmetic implementation
progress.

At 12 studies, eight slots remained to the planned minimum of 20. Even if every one
were a hit, the maximum attainable rate was 15/20 (75%), below the fixed 90% gate.
Enrollment was therefore stopped for mathematical futility. The surrounding task
runs were cancelled after their complete shadow records were copied, so this early
cohort does not provide public grader outcomes; it does provide the exhaustive
counterfactual ranking needed for the tournament decision. The machine-readable
record is `shadow_survivor_futility_analysis.json`.

## Inferences

- Removing only accumulated selected windows understates the opportunity after generic
  candidate-history compaction, because old history moves into the larger
  `selected_initial` component. A replacement must address both components.
- `latest-state` removed 58.3% of rendered bytes but increased uncached input per
  routing window by 26.9%, confirming the cache-rewrite risk.
- `checkpoint-plus-delta` had the best compact-mode cache economics: 57.3% fewer
  rendered bytes and only 0.6% more uncached input per window. That efficiency did
  not rescue its protected-quality regression.
- `decision-log-only` is a useful lossy ablation, not the leading safety candidate:
  free-form historical state summaries are not mechanically complete.
- The recent-exact-tail baseline outscored every structured mode in this small live
  sample, so the experiment does not establish that structured state adds value over
  ordinary truncation.

## Live pilot facts

The completed pilot contains 20 runs, 456 byte-validated ordinary routing records,
and 961 ordinary supervisor model calls. Outcomes were deliberately kept paired by
task and mode:

| Task | full | latest | checkpoint | decision log | recent tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| Returns | 1 | 0 | 0 | 0 | 0 |
| PSD Tools | 0 | 0 | 1 | 0 | 1 |
| Happy DOM | 0 | 1 | 1 | 1 | 1 |
| Drizzle | timeout | timeout | timeout | timeout | 1 |

Aggregate successes were 1/4 for `full`, 1/4 for `latest-state`, 2/4 for
`checkpoint-plus-delta`, 1/4 for `decision-log-only`, and 3/4 for
`recent-exact-tail`. These are single stochastic attempts, not success-rate estimates.
They show complementarity rather than monotone improvement: full uniquely retained
Returns, while compact modes gained PSD, Happy DOM, or Drizzle.

Measured ordinary-routing economics were:

| Mode | rendered bytes removed | estimated request tokens removed | raw input/window | uncached input/window | cache-read fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0% | 0% | 57,955 | 12,053 | 79.2% |
| latest state | 58.3% | 55.4% | 52,704 | 15,300 | 71.0% |
| checkpoint + delta | 57.3% | 55.8% | 42,768 | 12,123 | 71.7% |
| decision log | 35.9% | 33.1% | 56,815 | 12,670 | 77.7% |
| recent exact tail | 59.1% | 58.2% | 45,384 | 14,266 | 68.6% |

All compact modes reduced raw input per routing window, but all also increased
uncached input per window because their cache-read fractions fell. The increase was
smallest for checkpoint (+0.6%), followed by decision log (+5.1%), recent tail
(+18.4%), and latest state (+26.9%). Aggregate total uncached input is additionally
confounded by different route lengths and the four Drizzle timeouts, so the
window-normalized comparison is the safer cache-economics read.

## Speculation to test

- Three-window checkpoints may be too frequent for short tasks and too infrequent for
  unusually large diffs. A byte threshold may outperform a fixed interval.
- Single-lane serial repair windows may support a compact selector with no advertised
  audit tools, but the current 19-decision archive sample contains no one-lane cases.
- Shallow probes may rank architectural investigations well but kill implementation
  lanes whose advantage appears only after compilation or integration testing.

## Experiment limitations

The live pilot used one attempt per task/mode and cannot estimate variance. Each run
used three DeepSeek V4 Flash candidate lanes and a DeepSeek V4 Pro supervisor. Four
Drizzle modes timed out without a grader score; only recent-tail finished Drizzle.

The pilot binary used an aggressive compact-mode bootstrap: before the first
cumulative state existed, compact modes omitted the exact selected-initial message.
That omission was found during replay validation and is corrected in source. Pilot
quality results must therefore be labeled an aggressive-bootstrap ablation and
replicated with the corrected bootstrap before any production recommendation.

## Current recommendation

Promote no compact Q3 mode. Preserve the full control as default and retain the
capture/replay switches for research. The exploratory 40-call replay is now
complete and reinforces the no-go: the next informative Q3 run is a
corrected-bootstrap live replication of full, checkpoint, and recent-tail on the
protected Returns case plus the gained PSD/Happy cases, with multiple seeds.

Do not implement Prototype C or a production tournament: Prototype B failed its 90%
top-2 survivor-recall gate by mathematical futility. Prototype A remains a policy
experiment over existing execution capability, not evidence for survivor retention.
