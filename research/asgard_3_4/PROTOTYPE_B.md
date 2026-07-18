# Prototype B: shadow survivor-recall study

## Status

The fixed three-lane calibration protocol, trace schema, and scorer are executable.
Set both `ASGARD_WINDOW_POLICY_MODE=explicit-probe` and
`ASGARD_SHADOW_SURVIVOR_STUDY=1` with exactly three configured candidate models. The
research-only override turns the initial incomplete route into a three-lane,
one- or two-step probe with mechanically generated distinct probe contracts, so
calibration does not depend on the supervisor spontaneously proposing a probe.
Set `ASGARD_SHADOW_PROBE_STEPS=1` or `2`; the default is two. Invalid setup fails
closed before candidate work.

After the probe, the core ranks all lanes, autonomously continues all three isolated
lane repositories/transcripts for five steps, runs an opaque-label blinded end
review, and only then lets the existing synchronization path publish the final rank-1
endpoint. The flag-absent default path does not make any additional model call or
change its window policy.

## Pre-registered first experiment

Use three lanes, two probe steps, `K=2`, and five additional autonomous steps for
every lane. Three lanes make the sole nominally killed lane the shadow control and
provide complete ground truth without sampling. Do not tell any candidate whether it
survived the probe ranking. Continue each lane from its own probe messages and
repository, with no inter-window supervisor advice, so survivor status is not a
treatment confound.

The probe reviewer returns a strict permutation of all lane ids plus a structured
classification of whether the ranking distinction is architectural/contractual,
cosmetic, mixed, or unclear, with concrete evidence. The first two lanes are the
hypothetical survivors. All three lanes then continue independently for the same
step budget. A separate end reviewer sees randomized opaque endpoint labels, not lane
ids, model ids, probe ranks, survivor status, or the probe review's prose. It returns
a strict permutation of those labels. Only after that review may one endpoint be
published or the repositories synchronized.

The go/no-go metric asks whether the blinded final-best lane was funded by the
probe's top two:

`count(final winner in probe top 2) / complete two-step studies`

The scorer requires at least 20 complete two-step studies and observed recall of at
least 90% to pass. It also reports the 95% Wilson interval, final-winner recall at
top 1/2/3 separately for one- and two-step probes, whether the final best lane was a
killed late bloomer, and a stricter 95% top-1 funding diagnostic. The observed top-2
threshold is the decision rule; the interval is an uncertainty warning, not a second
hidden gate. A later, production-facing tournament should use a larger cohort before
changing defaults.

Studies with sampled rather than exhaustive killed branches remain useful diagnostics,
but are marked `complete_ground_truth=false` and cannot enter the gate. Likewise, a
two-lane `K=2` study is structurally incapable of killing a lane and is ineligible.

## Implemented core boundary

The core patch is an environment-gated alternative around the first forced probe
window, not a rewrite of repository synchronization:

1. Parse an opt-in `ASGARD_SHADOW_SURVIVOR_STUDY=1` configuration. Absent
   configuration selects the existing loop before any extra model call, clone,
   trace, or randomization. Invalid research setup fails closed.
2. Freeze an opaque `base_snapshot_id` for `common_patch`, `common_messages`, and the
   canonical plan at the start of the study window.
3. Run the existing candidate machinery for exactly one or two probe steps in the
   already-isolated candidate clones. Capture each probe outcome, cumulative patch,
   delta patch, ledger, continuation messages, and usage.
4. Use a research-only `rank_probe_trajectories` supervisor tool whose validated
   result is a permutation of `0..candidate_count`. Do not reuse
   `select_trajectory`: its wire shape records only one winner and cannot establish
   top-K recall.
5. Do **not** assign `common_patch`, replace `common_messages`, call
   `synchronize_candidate_repositories`, or append selected trajectory history after
   the probe review.
6. Invoke `tool_loop::run` again for every survivor and, for complete calibration,
   every killed lane. Pass that lane's own probe continuation messages and plan, the
   unchanged lane registry/repository, and the same fixed continuation-step limit.
   Add no supervisor advice and do not reveal disposition.
7. Render end dossiers from endpoint state. Randomly permute fresh opaque labels and
   build the end-review request through a typed function that accepts only sanitized
   endpoint dossiers. It must not accept the probe decision/history type. Remap audit
   tool label arguments to registries outside the prompt. A unit test should assert
   that model id, numeric lane id, probe rank, `survivor`, and `killed` are absent from
   the rendered request.
8. Validate a strict opaque-label permutation from a research-only
   `rank_shadow_endpoints` tool. Emit all four trace record types described by
   `survivor_recall.schema.json`.
9. Only now choose/publish an endpoint (normally the blinded final rank 1), update
   canonical history, and synchronize repositories once. Research runs must record
   this policy in their surrounding run manifest because it changes opt-in run
   behavior even though the default is unchanged.

The current `CandidateRepository` layout is sufficient: each lane is already a fresh
clone, and its registry is scoped to that clone. The key isolation error to avoid is
the existing synchronization at the end of every ordinary window. The continuation
must occur before that block. No new filesystem copy primitive is needed for the
three-lane exhaustive study. A focused repository test proves that all three states
remain distinct before the final synchronization and converge only afterward.

## Trace contract and accounting

`survivor_recall.schema.json` defines:

- `asgard_shadow_tournament_config`: fixed bounds and frozen base snapshot;
- `asgard_shadow_probe_ranking`: complete initial permutation, survivors/killed,
  per-lane probe candidate usage, and probe-review usage;
- `asgard_shadow_continuation`: hidden lane-to-label mapping, disposition, snapshot
  ids, isolation/publication invariants, fixed steps, and per-branch usage;
- `asgard_shadow_end_review`: blinded opaque-label permutation and review usage.

Candidate probe usage before ranking is recorded once in the probe record's
`candidate_usage` vector and is not duplicated in continuation records. The scorer
totals probe candidates, probe review, post-probe continuations, and end review.
Overall run cost remains independently authoritative in `result.json` and
`asgard_usage_by_model`; the two totals should reconcile after subtracting ordinary
run setup such as task-contract extraction.

For isolation, every continuation must name the same frozen `base_snapshot_id`, set
`isolated=true`, and set `published_to_canonical=false`. The scorer rejects missing
survivors, variable step budgets, duplicate labels, incomplete rankings, non-blinded
review, or no continued killed lane. Partial killed-lane sampling is reported but not
gate-eligible.

## Running the scorer

```bash
python3 research/asgard_3_4/analyze_survivor_recall.py \
  /path/to/archives-or-jsonl --output /tmp/asgard-survivor-recall.json
```

Use `--min-complete-studies` only when a different sample-size rule was registered
before examining outcomes. The default is 20.

Focused validation:

```bash
python3 -m unittest discover -s research/asgard_3_4 \
  -p 'test_analyze_survivor_recall.py'
jq empty research/asgard_3_4/survivor_recall.schema.json
```

## Required core tests before a live study

- Environment flag absent: byte-identical existing supervisor request and exactly
  the existing number of model calls, repository syncs, and trace records.
- Probe and end-rank validation rejects duplicates, omissions, out-of-range ids, and
  non-contiguous ranks.
- A killed lane is demonstrably continued from its own probe messages and patch.
- Every branch receives the same continuation step bound; cancellation cleans all
  clones without publishing an endpoint.
- Candidate repositories remain distinct through blinded review and synchronize
  exactly once afterward.
- Opaque labels are unique and their presentation order varies independently of lane
  order; the reviewer prompt contains no probe metadata.
- Per-model totals include probe candidates, probe review, continuations, and end
  review exactly once.
