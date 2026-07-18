# Asgard 3/4 trajectory-compression research

## Ordinary-routing prompt modes

The production `full` prompt remains the default and is byte-for-byte unchanged.
Research runs can select one of five ordinary-routing representations with
`ASGARD_SUPERVISOR_PROMPT_MODE`:

- `full`: append-only selected trajectory control.
- `latest-state`: latest explicitly cumulative state plus the mechanically derived
  canonical execution history; no exact older selected-window payloads.
- `checkpoint-plus-delta`: a frozen state and ledger checkpoint every N global
  windows plus exact selected-window deltas after it.
- `decision-log-only`: structured prior decisions and the canonical execution
  history, without historical candidate briefs.
- `recent-exact-tail`: a cumulative checkpoint plus the last N exact selected
  windows; this is the simple truncation baseline.

The checkpoint interval defaults to 3 and the recent tail to 2. Override them with
`ASGARD_SUPERVISOR_CHECKPOINT_INTERVAL` and
`ASGARD_SUPERVISOR_RECENT_EXACT_TAIL`. These switches affect ordinary routing
only; isolated completion review remains unchanged.

Every ordinary decision emits an `asgard_supervisor_prompt_mode` trace record with
the chosen and counterfactual full-control prompt bytes/token estimates. Set
`ASGARD_CAPTURE_SUPERVISOR_REPLAYS=1` only for research capture runs. It adds exact
`asgard_supervisor_replay_state`, `asgard_supervisor_replay_request`, and
`asgard_supervisor_replay_response` records, including messages, tools, parameters,
model output, reasoning, and the per-call usage vector. Captures can be very large
and may contain task or repository content.
`replay_capture.schema.json` describes the capture and window-policy trace shapes.

Example local build:

```bash
cargo build --release --target x86_64-unknown-linux-musl
```

The controlled DeepSWE pilot uses identical task/model settings and places the
mode variables in each task's `[environment.env]` table so the containerized Anvil
process inherits them. The common runner shape is:

```bash
uv run python bpr_agent.py --engine deepswe \
  --models q3=deepseek::deepseek-v4-flash \
  --tasksdir /path/to/mode-specific/tasks \
  --asgard-candidates 3 \
  --asgard-supervisor deepseek::deepseek-v4-pro \
  --anvil-bin /path/to/anvil/target/x86_64-unknown-linux-musl/release/anvil \
  --no-anvil-rebuild --headless
```

The checked-in `live_experiments.json` stages the corrected-bootstrap Q3 and Q4
paired studies under `/tmp/asgard-3-4-live-v2`:

```bash
python3 research/asgard_3_4/prepare_live_experiments.py
python3 research/asgard_3_4/run_live_detached.py \
  --batch q3-corrected-full --batch q3-corrected-checkpoint \
  --batch q3-corrected-recent-tail --attempt 1
python3 research/asgard_3_4/run_live_detached.py \
  --batch q3-corrected-full --batch q3-corrected-checkpoint \
  --batch q3-corrected-recent-tail --attempt 1 --status
```

The detached launcher gives each controller an explicit log and process session,
so it can continue across research-agent turns. Status classifies archives with
both `result.json` and `anvil-trace.jsonl` as captured and reports cancellation or
corrupt ZIPs separately; infrastructure marker JSON does not count as a completed
result. A controller PID may be unobservable from a later sandbox PID namespace,
so artifact counts and the controller log are authoritative.

Aggregate completed pilot archives with:

```bash
python3 research/asgard_3_4/analyze_live_pilot.py \
  /path/to/archive/directories --output /tmp/asgard-pilot-report.json
```

The report separates ordinary-routing and completion-review calls, sums uncached,
cached, output, and thought tokens, compares each compact prompt with its rendered
full-control counterfactual, and preserves paired task outcomes.
`live_pilot_summary.json` preserves the compact result of the completed 20-run
aggressive-bootstrap pilot; `RESULTS.md` gives the no-go decision and limitations.

Score `asgard_supervisor_replay_result` JSONL entirely offline with:

```bash
python3 research/asgard_3_4/analyze_supervisor_replays.py replays.jsonl \
  --pilot-report /tmp/asgard-pilot-report.json \
  --protected-corpus research/asgard_3_4/known_success_corpus.jsonl \
  --output /tmp/asgard-replay-analysis.json
```

The analyzer reports fallback rate; non-fallback and effective (full-fallback)
winner, completion, and funding agreement; signed candidate/step deltas; the full
usage vector and cache fractions; and exact disagreement rows/windows. Its Q3
gates fail closed without supplied prompt/raw-input reduction or protected-record
identity: at least 25% input reduction, 95% effective winner agreement, and 100%
effective protected endpoint agreement. Use `--require-automated-gates` for a
nonzero exit when any automated gate is not met. Even when they pass,
`recommendation_ready` remains false until a human reviews preservation of the
known-success evidence obligations; agreement alone cannot automate that review.

Plan the same-state replay batch before making any API calls:

```bash
python3 research/asgard_3_4/plan_supervisor_replay_batch.py \
  /path/to/full-control-archives \
  --protected-corpus research/asgard_3_4/known_success_corpus.jsonl \
  --sample-size 100 --seed asgard-q3-replay-v1 \
  --output /tmp/asgard-replay-plan.json
```

With no `--sample-size`, every completed ordinary decision is selected. Sampling
is deterministic and balanced across one/multi-lane, long-history (default: ten
prior windows), post-completion-review, and protected-endpoint strata. Every
captured protected endpoint is forced into the sample even if that exceeds the
requested size. Stage 1 runs those protected diagnostics first; Stage 2 contains
the remaining overall-agreement sample. The plan renders all four compact modes
locally and reports exact prompt bytes, conservative request-token estimates,
and a separate UTF-8 byte/token ceiling for checkpoint, latest-state,
decision-log-only, and recent-tail calls. It does not contain rendered task
content and performs no network operations.

The protected-stage stop result is preserved in `protected_endpoint_replay.json`.
Three of four modes disagreed with the full terminal decision, so the
pre-registered confirmatory replay stopped there. A later explicit approval ran
the remaining 36 calls as exploratory follow-up. The complete 10-state x 4-mode
analysis is preserved in `supervisor_replay_40_analysis.json`; the stop-gate
failure still governs the promotion decision.

## Conservative serial fast path

Set `ASGARD_SUPERVISOR_FAST_PATH=conservative` together with a non-`full` prompt
mode to test a one-response ordinary selector with no audit tools. It is eligible
only when the prior supervisor funded exactly one work lane with one non-empty
serial advice, the lane did not fail or get cancelled, the current window produced
a non-empty delta, and a successful shell command followed a recorded edit. The
selector must explicitly confirm both confidence and that the serial prerequisite
was addressed. It cannot nominate completion; uncertainty, invalid output, or any
failed criterion falls back to the ordinary full-tool supervisor, with both calls'
usage retained.

Every enabled window emits `asgard_fast_path` with eligibility reasons, whether the
path was used, fallback reason, and its full usage vector. Exact fast-path requests
and responses are also captured when `ASGARD_CAPTURE_SUPERVISOR_REPLAYS=1` is set.
This is an experiment switch; `disabled` remains the default.

## Explicit probe policy (Prototype A)

Set `ASGARD_WINDOW_POLICY_MODE=explicit-probe` to require every initial or
incomplete route to choose one of two validated window kinds:

- `probe`: 2-5 lanes, 1-2 steps, with a distinct hypothesis, concrete
  falsification action, and observable continue/stop signal for every lane;
- `work`: the existing 1-5 lane, 1-10 step behavior and ordinary advice schema.

The default `dynamic` mode retains the existing prompt and tool schemas. After a
probe, the structured lane contracts are included in the supervisor dossier with
an instruction to judge information gain rather than patch volume. Each opt-in
window emits `asgard_window_kind` with its actual kind, candidate count, steps,
and structured hypotheses. The policy is only a prompt/schema experiment over
the existing one-winner window mechanism; it does not preserve probe survivors.

Analyze paired dynamic and explicit-probe archives with:

```bash
python3 research/asgard_3_4/analyze_probe_policy.py \
  /path/to/dynamic /path/to/explicit --output /tmp/asgard-probe-policy.json
```

The analyzer uses only structured config, window-kind, candidate-count/depth,
and per-window usage traces. It reports the joint count x depth distribution,
the fraction of windows already eligible for a shallow tournament, candidate
cost per lane-step, total cost per run, paired task outcomes, and the
pre-specified 5% behavioral-sameness tolerance. Shadow-survivor runs are
rejected so they cannot contaminate this policy comparison.

## Archive corpus

`extract_archive_corpus.py` turns one or more Anvil result zip archives into a
machine-readable JSONL window corpus. It joins `anvil-stderr.txt` dossier byte
telemetry to ordinary `asgard_decision` supervisor records by ordinal and excludes
`completion_review` records. This is necessarily an ordinal join because current
decision trace records have no window field.

Run it directly on archives:

```bash
python3 research/asgard_3_4/extract_archive_corpus.py run-1.zip run-2.zip \
  --output /tmp/asgard-windows.jsonl
```

Or use a manifest such as the eight-case live-regression set:

```bash
python3 research/asgard_3_4/extract_archive_corpus.py \
  --manifest scripts/asgard_live_regressions.json \
  --output /tmp/asgard-live-regressions.jsonl
```

Each `asgard_routing_window` record contains archive, case, task, and window
identity; the three exact dossier component byte counts; derived history growth
and removal ceilings; the full ordinary decision and common decision fields;
candidate count (preferring an explicit field, otherwise `len(advices)`), next
steps, and one/multi-lane classification; plus `result.json`, `reward.json`, and
the result-level usage fields. The final `asgard_corpus_summary` record aggregates
Q3 byte ceilings and Q4 retrospective distributions. Use `--records-only` to omit
that final row.

Some older archives contain dossier telemetry but predate `asgard_decision` trace
records. Those windows are retained with `alignment.status="decision_missing"`
and unknown decision-derived fields; the summary reports this coverage gap. Use
`--require-decisions` when building a replay corpus that must reject such gaps.

The removal values are deliberately named **ceilings**. In particular,
`all_history_removal_ceiling` is the impossible perfect ceiling of removing both
the stable selected-initial component and accumulated selected windows, while
`older_selected_windows_removal_ceiling` measures the narrower accumulated-history
opportunity. Neither claims that a compact replacement is safe or free.
`full_dossier_measured` is only the sum of the three byte components emitted by
the telemetry; it is not an exact serialized request size and does not include
separately rendered task, checklist, policy, or supervisor-decision messages.

## Protected known-success corpus

`known_success_corpus.jsonl` is the strict quality corpus requested by the
research brief. Its first row is a manifest; the remaining 18 rows are every
successful trace in the exact 40-run `asgard6-claims` and
`asgard9-flash-flash` cohorts. Those traces cover 15 distinct task/run
identities, with three successes common to both cohorts. Task and run are one
identity (`task-id::rN`), so two runs of the same task are separate protected
regressions.

Regenerate it from the archived public result metadata and Anvil archives:

```bash
python3 research/asgard_3_4/build_known_success_corpus.py \
  --agentresults-root /home/jonathan/Projects/brokkbench/agentresults \
  --archive-root /home/jonathan/brokkbench-archive \
  --output research/asgard_3_4/known_success_corpus.jsonl
```

The defaults fail closed unless they find 40 results and nine successes in each
cohort, a 15-identity union, and three common successes. The corpus records:

- the full agent-side contract checklist and first two supervisor-selected
  architectural assessments;
- the final changed-file surface plus mechanically parsed added declarations and
  patch hunk contexts (not the old patch itself);
- every ordinary supervisor boundary state, unresolved-risk summary, next
  advice, selected lane, and funding decision;
- all test/build/check commands visible in candidate-side traces and stderr,
  explicitly scoped to all lanes when older telemetry cannot prove lane
  ownership;
- actual candidate-count/window-depth sequences where traced, with an explicit
  source or `null` and a reason where old v6 telemetry cannot recover a depth;
- completion-review decisions and an explicit `unknown` assessment of whether a
  winning lane looked weak after one or two steps.

Facts and research inferences are separate top-level fields.
`known_success_corpus.schema.json` describes both JSONL record types. The builder
selects successes using public agent-result metadata and reads only
`anvil-trace.jsonl`, `anvil-stderr.txt`, and `model.patch` from an archive. It
does **not** read or copy `verifier-output.txt` or `verifier.tar.gz`; hidden-test
details therefore cannot enter the corpus.

Coverage limitations are deliberately machine visible. All 328 successful
ordinary-routing windows have candidate counts, but 23 v6 window depths are
unknown (initial windows and some post-review continuations were not traced).
All 18 one/two-step weakness fields are `unknown`: these archive versions select
only after a complete funded window and emit no explicit early weak/strong score.
The 268 observed verification commands span all candidate lanes; tool completion
is not treated as proof that test assertions passed. Supervisor state summaries
are preserved as model assessments, not relabeled as ground truth.

Exact CLI help:

```text
usage: extract_archive_corpus.py [-h] [-m MANIFEST] [-o OUTPUT]
                                 [--records-only] [--require-decisions]
                                 [ARCHIVE ...]

Extract aligned Asgard ordinary-routing windows and Q3/Q4 summary metrics from
Anvil result zip archives.

positional arguments:
  ARCHIVE               Anvil result zip archive (repeatable)

options:
  -h, --help            show this help message and exit
  -m, --manifest MANIFEST
                        JSON list, or object with a cases list, whose entries
                        contain archive paths
  -o, --output OUTPUT   write JSONL here instead of stdout
  --records-only        omit the final asgard_corpus_summary JSONL record
  --require-decisions   fail if any dossier telemetry row lacks an ordinary
                        supervisor decision
```

## Shadow survivor-recall study (Prototype B)

Existing winner-only archives cannot measure whether a shallow probe would have
killed a later-best lane. `PROTOTYPE_B.md` specifies an opt-in three-lane calibration
that continues the probe top two and the killed lane independently, then ranks all
endpoints under opaque labels. `survivor_recall.schema.json` defines its trace
contract, and the scorer is ready for trace captures:

```bash
python3 research/asgard_3_4/analyze_survivor_recall.py \
  /path/to/archives-or-jsonl --output /tmp/asgard-survivor-recall.json
```

The scorer excludes partial, non-isolated, non-blinded, variable-budget, or otherwise
invalid studies from the 90% two-step top-2 recall gate. It separately reports
one-step and two-step top-1/top-2/top-3 recall, architecture/contract versus cosmetic
probe distinctions, late-bloomer kills, task outcomes, and attainable continuation
savings after measured review overhead. It stops for mathematical futility when even
perfect remaining studies could not reach the fixed threshold. Run the executable
fixed calibration with
`ASGARD_WINDOW_POLICY_MODE=explicit-probe`,
`ASGARD_SHADOW_SURVIVOR_STUDY=1`, `ASGARD_SHADOW_PROBE_STEPS=1` or `2`, and exactly
three candidate models. See the bounded protocol and required invariants in
`PROTOTYPE_B.md`.

Run the focused tests with:

```bash
python3 -m unittest discover -s research/asgard_3_4 -p 'test_*.py'
```
