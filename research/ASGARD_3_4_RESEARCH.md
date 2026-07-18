# Asgard research brief: stateful/delta supervision and probe tournaments

## Workspace and scope

- Worktree: `/mnt/optane/anvil-asgard-stateful-probe`
- Branch: `asgard/stateful-probe-tournament`
- Base: `bc8a690` (`Synchronize Asgard repositories incrementally`)
- This is a research branch. Code is cheap; information is the deliverable.
- It is acceptable to duplicate or rewrite production code, add experiment-only environment toggles,
  and leave prototypes unmerged. Do not optimize around merge conflicts with concurrent work on `master`.
- Do not change the benchmark or hidden grader. Never use hidden-test outcomes as runtime inputs.

The two interventions to investigate are:

3. A stateful/delta supervisor that avoids repeatedly paying to reconstruct canonical history.
4. A cheap probe tournament before long candidate funding.

The current implementation already has partial versions of both ideas. The first obligation is to
state precisely what new mechanism each experiment adds. A prompt rewrite that merely redescribes
existing behavior is a negative result, not a successful implementation.

## Current Asgard facts to preserve in the analysis

Read `src/agent.rs`, especially the Asgard section beginning near `AsgardCandidate`, before coding.
The important functions and structures are:

- `run_asgard_trajectory_loop`
- `AsgardSupervisorHistory` and `AsgardSupervisorDecision`
- `run_asgard_initial_advice`
- `run_asgard_supervisor`
- `run_asgard_completion_review`
- `run_asgard_supervisor_tool_steps`
- `asgard_initial_advice_messages`
- `asgard_supervisor_messages`
- `asgard_completion_review_messages`
- `asgard_advise_trajectories_tool` and `asgard_select_trajectory_tool`
- `AsgardExecutionLedger`

Current behavior:

- At task start and after every window, the supervisor chooses 1-5 candidates and 1-10 steps.
- More candidates are already requested under uncertainty; one lane is requested for a serial bug.
- Every lane in a window starts from the same single canonical state.
- After a window, exactly one winner becomes canonical and all other work is discarded.
- The ordinary supervisor can already select immediately in one LLM response; audit tools are optional.
- Prior selected windows and decisions are appended to the supervisor dossier. This is cache-friendly,
  but the logical history and non-candidate residual still grow.
- `state_summary` exists, but it is free-form and is not currently trusted as a sufficient replacement
  for the selected trajectory history.
- A short multi-lane window followed by selection is already possible. Therefore a "probe tournament"
  that only asks the supervisor to choose two steps instead of ten adds no execution mechanism.
- Terminal completion review is isolated and has different evidence needs. Do not silently apply a
  routing compression experiment to completion review.

Relevant v9 cohort observations (40 Flash/Flash runs versus v6):

- Total raw tokens: 942.6M -> 711.3M (-24.5%).
- Exact candidate tool-loop raw tokens: 794.5M -> 541.1M (-31.9%).
- Mixed non-candidate residual: 148.1M -> 170.2M (+14.9%).
- Binary success is flat at 9/40, with only 3 common successes and 6 flips each way.
- v9 ran 858 windows and 169 completion reviews.
- The residual is not pure supervisor usage; it also includes window summaries, completion reviews,
  contract extraction, initial advice, and compaction. Concurrent work on `master` is adding exact
  phase attribution, so use request bytes and local call counters until that can be cherry-picked.

## Primary quality objective: retain the historical success union

The most important quality target is not aggregate 9/40 parity. v6 and v9 solved only three of the same
runs, while their task/run union solved 15/40. Treat every task that any prior Asgard version solved as a
protected regression case. A new intervention that loses one protected solution mode and gains a different
task has not demonstrated monotone progress, even when its aggregate score is unchanged.

Build a machine-readable known-success corpus before recommending either intervention. For each protected
task include every available successful trace and identify, without hidden-test leakage:

- the architectural or contract reading that distinguished the successful direction;
- the decisive implementation surface and verification command;
- evidence or unresolved risk that had to survive a window boundary for the winning continuation;
- the candidate-count/window-depth sequence that preserved the solution mode;
- whether the successful lane looked weak after one or two steps.

Use the corpus in two ways:

1. Replay gate: a compressed supervisor dossier must preserve every decisive evidence obligation and should
   retain the full supervisor's winner on successful trajectories.
2. Live gate: by default every rerun of every historically solved task must succeed. Report per-task success
   rates; do not hide a regression behind aggregate replacement wins.

Direct task-specific answer memory is a separate, explicitly labeled experiment because feeding an old patch
or hidden failure back into the same benchmark changes what the benchmark measures. The default research here
is about retaining solution modes through better routing and representation, not memorizing answers.

## Research question 3: stateful/delta supervisor

### Hypothesis

An ordinary routing decision may not need the exact transcript of every prior selected window. A
compact, explicitly cumulative canonical state plus the current-window evidence may preserve routing
quality while reducing supervisor input and possibly supervisor turns.

This is not established. The current append-only dossier benefits from provider prompt caching, and a
rewritten state snapshot can trade raw cached input for more expensive uncached input. Measure the full
usage vector; do not report only total text bytes.

### Required prototypes

Build at least these three replayable prompt modes behind an experiment-only switch. Default production
behavior must remain available as the control.

1. `full` control: byte-for-byte current supervisor dossier behavior.
2. `latest-state`: original task, contract checklist, one explicitly cumulative latest canonical state,
   current verified-evidence ledger, and current candidates; omit exact older selected-window payloads.
3. `checkpoint-plus-delta`: a periodically frozen canonical checkpoint plus only events since that
   checkpoint and the current candidates. Explore checkpoint intervals or a size threshold.

If a fourth mode is useful, test `decision-log-only`: retain structured prior decisions and evidence IDs,
but omit candidate-authored historical briefs and raw historical trajectory messages.

Do not use a new LLM call merely to create the compact state unless its cost is included. Prefer first:

- making `state_summary` explicitly cumulative in the selection contract;
- carrying structured contract status/evidence references forward;
- deterministically rendering the latest execution ledger, patch manifest, and unresolved risks;
- retaining stable original-task and checklist messages as a cacheable prefix.

It is fine to prototype a structured `AsgardCanonicalState`, for example:

- selected architectural direction and files/symbols involved;
- established facts and their evidence IDs;
- unresolved contracts and adverse conditions;
- known failed approaches or concrete defects;
- latest production/test patch manifest;
- latest successful and failed verification commands;
- next serial dependency, if any.

Treat every field as lossy unless it is derived mechanically. Preserve references back to source ledger
entries so an omitted detail can be audited in replay.

### Fast-path experiment

Separately test whether an ordinary routing call can safely use a compact one-response selector with no
audit tools in conservative cases. The existing supervisor can already choose in one response, so this
experiment must change either the advertised tool catalog or the dossier size to be distinct.

Candidate eligibility rules to test, not assume:

- one active lane;
- no candidate execution failure;
- the current advice named one concrete serial prerequisite;
- the resulting diff and evidence directly address that prerequisite;
- no terminal completion review is being performed.

Keep an escape to the full supervisor when validation fails or the model reports uncertainty. Trace the
eligibility reason and fallback reason. A deterministic fast path must never itself declare terminal
completion; completion remains a nomination followed by isolated review.

### Evaluation for question 3

First build an offline replay corpus from existing v9 traces if available locally. If traces are not
available, add a capture format and run a small fresh set. A replay record should contain the exact full
control prompt, tools, model parameters, response, decision, usage vector, current patches/manifests,
and the eventual task outcome when known.

For each alternative mode report:

- prompt bytes and estimated/request tokens per routing turn;
- uncached input, cached input, output, and thought tokens when actually called;
- winner agreement with the full control;
- complete-nomination agreement;
- next candidate-count and step-count deltas;
- advice obligation coverage, especially contract IDs and serial prerequisites;
- fallback rate;
- evidence present in control but absent from compact mode;
- downstream result on a small paired live run.

Stratify disagreement review across:

- one-lane serial repair windows;
- multi-lane architecture disagreements;
- post-review repair windows;
- long histories / high window ordinals;
- v6-only and v9-only successes if the trace corpus permits.

The perfect ceiling is all ordinary-routing input, but that is not an attainable estimate. Produce a
measured ceiling from the replay corpus, then a realistic estimate after fallback and cache effects.

### Stop/go rules for question 3

Go to a live paired pilot only if an alternative:

- removes at least 25% of ordinary-routing raw input on replay;
- retains at least 95% winner agreement overall;
- retains 100% winner agreement on known success endpoints or explains every disagreement manually;
- has no systematic loss of unresolved-contract or adverse-condition obligations;
- does not convert cached input savings into a larger uncached-input bill.

Treat failure to beat the full control as a useful negative result. Do not promote a compact mode based
on shorter rendered text alone.

## Research question 4: cheap probe tournament

### First resolve whether the proposal is actually new

Document the semantic difference between each of these:

1. Current Asgard choosing N candidates and a 1-2 step window.
2. An explicitly labeled probe window that still selects one canonical winner afterward.
3. A true tournament that preserves two or more probe survivors and funds them further before selecting
   a single canonical winner.

Mode 2 is mostly policy/structure over capability Asgard already has. It may still improve model behavior,
but measure it as a prompt-policy experiment. Mode 3 is a genuinely new mechanism and violates the
current one-canonical-state-at-each-window invariant until the tournament closes.

### Prototype A: explicit probe policy (cheap, low structural novelty)

Add an experiment mode where initial advice and incomplete routing decisions explicitly choose a window
kind:

- `probe`: 2-5 lanes, 1-2 steps, each advice has a hypothesis, falsification action, and observable
  continue/stop signal;
- `work`: 1-5 lanes, 1-10 steps, current semantics.

The supervisor following a probe should receive the hypotheses and signals in structured form and judge
information gain, not patch volume. Validate schema combinations. Do not let a `probe` label alone count
as savings; compare actual candidate count, steps, and tokens with the unconstrained dynamic supervisor.

Prototype A answers whether more structure makes the existing short-window capability happen at useful
times and whether the eventual best lane can be identified from shallow evidence.

### Prototype B: shadow survivor-recall study

Before implementing a production survivor tournament, measure whether it could work:

- On a bounded sample, launch 2-5 probe lanes for 1-2 steps.
- Ask the supervisor to rank all lanes and choose K survivors, not merely one winner.
- In research mode only, continue both the chosen survivors and at least one lane that would have been
  killed, for a fixed additional budget.
- Judge which continuation is best at the later horizon using the same endpoint review machinery or a
  blinded supervisor comparison.
- Record whether the eventual best lane was in top-1, top-2, and top-3 after the probe.

This is deliberately wasteful shadow execution to estimate survivor recall. It must not be reported as
a token-saving rollout. It estimates how often a cheap probe can avoid expensive funding without killing
late-blooming lanes.

You may copy repositories or retain divergent repositories however is simplest in the research branch.
Correct measurement matters more than preserving the production synchronization architecture.

### Prototype C: true tournament only if B supports it

If top-K survivor recall is promising, prototype a real two-stage window:

1. probe stage: N lanes x 1-2 steps;
2. funding decision: retain K lanes, with K < N;
3. development stage: each survivor receives an independent additional step budget from its own probe
   state, not from a shared canonical winner;
4. final selection: choose one canonical state and synchronize all repositories only then.

This requires representing multiple live repository/history states within a window. Keep it experiment-
only. Do not weaken isolation, accidentally apply one survivor's patch to another, or count candidate
summarization/routing calls outside the cost.

### Evaluation for question 4

Report:

- top-1/top-2/top-3 survivor recall at 1 and 2 probe steps;
- raw and usage-vector cost of probe + funding + final selection;
- theoretical one-winner oracle ceiling versus measured attainable savings;
- how often probes distinguish architecture/contract readings versus cosmetic variations;
- late-bloomer kill rate;
- task-level outcomes, partial scores, timeouts, and agent failures;
- whether explicit probe policy merely recreates current short windows;
- the fraction of windows eligible for a tournament in practice.

Use paired tasks and preserve task-level identity. Include the v6/v9 wrong-direction flips when possible,
because a cost policy that preferentially kills complementary lanes could keep total tokens low while
silently erasing the union-oracle upside.

### Stop/go rules for question 4

- Do not build Prototype C unless top-2 recall after a two-step probe is at least 90% on the shadow set.
- Do not recommend top-1-only funding unless its measured recall is at least 95%; the current evidence
  already shows strong run complementarity.
- A rollout recommendation needs at least 10% candidate raw-token savings after including added supervisor,
  summary, and shadow/funding overhead, with no loss of known success cases in the pilot.
- If explicit probe policy yields the same candidate/step distribution as control, conclude that current
  dynamic windows already subsume it.

## Instrumentation and artifacts

Add experiment-specific trace records rather than scraping prose. Suggested records:

- `asgard_supervisor_prompt_mode`: mode, window, prompt bytes, history bytes, current-dossier bytes.
- `asgard_fast_path`: eligible, reasons, used, fallback reason.
- `asgard_window_kind`: probe/work, candidate count, steps, hypotheses.
- `asgard_probe_ranking`: lane ranks, survivor K, confidence/uncertainty, evidence references.
- `asgard_survivor_outcome`: later-horizon winner, whether top-K contained it, per-lane usage.

Keep the existing aggregate usage accounting intact. If you add direct LLM calls, include every call in
both aggregate usage and per-model usage. Do not use `next(iter(models.values()))`-style assumptions in
analysis; candidate and supervisor models may differ.

Produce these branch artifacts:

1. `research/asgard_3_4/README.md`: exact commands and experiment design.
2. Machine-readable replay/capture schema and scripts.
3. Unit tests for prompt modes, schema validation, repository survivor isolation, and trace accounting.
4. `research/asgard_3_4/RESULTS.md`: facts, inferences, speculation, negative results, and recommendation.
5. One or more commits with descriptive messages. Do not merge or push unless asked.

## Validation discipline

- Run `cargo fmt`.
- Run focused Asgard unit tests while iterating.
- Run the full `cargo test` outside the restricted sandbox because Wiremock binds localhost.
- Run `cargo clippy --all-targets -- -D warnings` before declaring the prototype ready.
- Do not add lint suppressions to evade a design issue.

## Expected final report

Lead with direct verdicts:

- Does stateful/delta supervision reduce actual raw cost after cache effects, and at what measured quality
  risk?
- Is explicit probe mode behaviorally different from today's dynamic windows?
- Can shallow probes identify later winners with enough top-K recall to justify a true tournament?
- What is the perfect-oracle ceiling, what is realistically attainable, and which measurements support
  that haircut?
- Which code, if any, is worth promoting to production?

Separate facts from inference and speculation. Do not sum overlapping savings ceilings. A well-supported
"no" is a successful research result.
