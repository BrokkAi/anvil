# mj sol/luna on hard20: three runs, from 100% infra death to noise-distance from vanilla sol

*2026-07-31. The mjolnir primary/subagent configuration (sol+high thor,
luna+xhigh eitri) measured three times on `deepswe-fable-hard20.tasks`
(20 tasks; vanilla sol@high expects 15.2/20 one-shot, per-task rates from
`deep-swe/published-results/deepswe-v1.1/per-task-by-model-effort.csv`).
Goal, per owner: **beat vanilla sol@high at sol@high prices** — sol@xhigh
(85% on this set) rejected on cost. Evidence: `/mnt/optane/hard20-solluna{,2,3}`.*

## Headline trajectory

| run | binaries | result | vs sol@high E=15.2 |
|---|---|---|---|
| 1 | mj 1.2.1 (roster), anvil chain-fix only | killed at 12 resolved: 4 S, 8 poisoned | — (invalid config) |
| 2 | mj master (pinned sol/luna), all fixes | **12/20**, zero timeouts | P(vanilla ≤12) = 0.006 |
| 3 | + subagent debrief (mj `136bbd3`) | **13/20** | P(vanilla ≤13) = **0.06** |

Cost, run 3 (trace tokens × blended rates fitted from the published CSV):
**~$12.9/task ≈ 3.7× vanilla sol@high's $3.47**. Two 2h attempts alone were
$84 of the $258 total. On the stated goal the config now ties on quality
within noise and clearly fails on price.

## What run 1 died of (all fixed, all field-verified in runs 2–3)

1. **Sticky chained-Responses failures** (anvil `eae9792`): Bedrock Mantle
   intermittently ends streams with `server_error` inside HTTP 200 (~6–9%/req);
   retries re-sent the same poisoned `previous_response_id`, killing 48/48
   attempts on 07-30 while fresh sessions succeeded in the same minutes. The
   exact failing request replayed clean 3/3 as full input. Fix: evict the whole
   cached chain lineage on stream failure. Field: 9 evictions, 9 recoveries in
   one smoke attempt.
2. **A 500 wearing a 400's code** (anvil `2d73a76`): `invalid_prompt:
   Internal server error` classified terminal; now the message earns the
   patient tier, genuine prompt rejections stay terminal.
3. **Roster contamination** (mj 1.2.1): per-call `create_subagent`
   agent/model selection let sol staff subagents with sonnet-4-6, sonnet-5,
   opus-4-8 (cattrs: 333 Claude calls, zero luna). Upstream removed it
   (`91395c1`); runs 2–3 verified pure sol/luna. Corollary fix: review model
   pinned via `[loki]` (brokkbench `mjolnir_config`) because mj's `auto`
   review deliberately prefers a *different* model than the primary.
4. **Silent tool-surface loss** (brokkbench `94e81106072`): the deepswe
   engine never staged the bifrost shim its own MCP config pointed at
   (ENOENT while anvil's bundled 0.8.6 sat unused), and
   `BPR_AGENT_ALLOWED_TOOLS` stripped `create_subagent`/`subagent_cancel` —
   a full sol/luna smoke solved its task with **zero delegations** because
   the tool never reached the catalog. Also renamed the phantom
   `*_by_reference` entries to `*_by_location`.
5. **Timeout laundering** (brokkbench): the wall-clock kill's own
   `TimeoutExpired` (despite `check=False`) reclassified solver timeouts as
   INFRA errors, granting free 2h do-overs nondeterministically.
6. **Call-count explosion**: run-1 timeouts made 10–18× vanilla's LLM calls
   (bandit: 20 subagents, 2,738 tool calls vs vanilla's 29 steps) — cold-start
   fan-out where each fresh subagent re-oriented from zero. mj master's
   retained sessions + `resume` (take-and-return id, empty = new) is the
   structural mitigation; runs 2–3 averaged 2–5 subagents/attempt.

## The debrief experiment (run 3 vs run 2, only delta = mj `136bbd3`)

After each successful subagent task turn, the runtime asks one canned exit
interview on the retained prefix-cached session (VERIFIED / UNVERIFIED /
COMMITMENTS / ANOMALIES / NEXT) and injects it as `<debrief>` in the report;
the primary is told to treat UNVERIFIED as its re-check list. Marginal cost:
cached reads + ~1k luna tokens per subagent.

Scorecard: flips up — sqlfmt 28/32→**32/32** (sol-0/4: first win in that
bucket all project), tengo 0/23→23/23, opa-template 2/5→5/5; partial climbs
— pebble 0/59→57/59, python-statemachine 65→69/72. Flips down — opa-rego
25/25→0/25 (see audit below), dynamodb and scriggo → 2h timeouts (run 3 ran
31% longer overall; the debrief's wall-clock tax took back two ~80-minute
run-2 wins). Net +1. Attribution is suggestive, not proven: n=1 per cell and
the catastrophic-miss class moved tasks rather than shrinking. On content
(ignoring the clock) run 3 solved 15/20.

## Loss taxonomy after three runs

- **Hidden-test near-miss** (bandit 86/88 three times; textual 19/20 twice):
  the missed f2p tests exist only in the grader's hidden patch; no runnable
  signal. Interpretation-refinement territory, not verification.
- **Incomplete enumerated-spec delivery** (opa-rego run 3, audited): the
  instruction *enumerates 17 EvalProfile methods by name*; run 3 shipped 4
  (`Stat, RulePaths, Diff, HasChanges`) — the hidden test package failed to
  compile, all 25 f2p "did not run", p2p 6/6 green. Run 2 shipped all 17 and
  won. Nothing was misread; enumerated surface was partially delivered as
  done. A finalize-time completeness pass against instruction-enumerated
  surfaces is mechanical and would have caught it cold. tengo run 2 (0/23,
  oracle-narrowing at close-out) and pebble run 2 (0/59) are the same
  outcome class with different proximate causes; roughly 1–2 attempts per
  run land here, on different tasks each time.
- **Clock losses** (dynamodb, scriggo run 3): content solved or nearly so at
  ~80–120m; the 2h cap plus debrief overhead decided the outcome. Also the
  cost tail: these attempts are 3–4× the median attempt cost.

## Open levers (owner's call; constraint = sol@high prices)

1. **Kill the cost/clock tail**: finalize by ~60–75 min instead of riding
   the cap. Would have made run 3 ~$9/task and preserved scriggo.
2. **luna@max on the cheap seat**: vanilla luna max = 67.2% full / 67.5%
   hard20 (+10 over xhigh) at $3.12/task vanilla — the only published
   config upgrade compatible with the cost constraint.
3. **Finalize-time completeness check** against instruction-enumerated
   surfaces (the opa-rego class). Protocol/prompt-level, not harness.
4. **runs≥2 per config** before believing any number: single-run variance
   on this set is ±2 tasks demonstrated.

Structural verdict so far, consistent with asgard held-out A from the other
side of the authority split: the duo runs at luna-level results for
3.7× sol prices; the coordination tax still exceeds the delegation dividend.
The one mechanism repeatedly earning its keep is independent fresh reads
(sol-swing and sol-never wins concentrate where subagents did recon), which
is also the cheapest part of the architecture.

## Addendum: run 4 (sol+high / luna+max, session affordance) — 2026-07-31 evening

Config delta from run 3: luna xhigh→max (anvil preset + brokkbench effort
gate had to learn `max` first — both silently/loudly rejected it), the
`<session>` resume affordance in reports, mj at 1.3.0-era master.

**Result: 13/20 — ties run 3. P(vanilla sol@high ≤13) = 0.06. Wall 4.75h
(slowest run; luna@max thinks long).** Tokens/task: sol in 9.6M (−26% vs
run 3), luna in 16.6M (+36%) — the load moved to the cheap seat; sol output
stays at vanilla parity (30k).

The luna@max signal is the strongest any lever has produced: it cleared
**both chronic single-test walls** — bandit 88/88 after three straight
86/88, textual 20/20 after two straight 19/20 — and held sqlfmt (sol-0/4)
and tengo. Nine tasks have now won all three post-fix runs.

The giveback is the two per-run stochastic taxes, unchanged in expectation:
(i) the wrong-reading/deference class took cliffy (0/37, both prior runs
won it) and opa-rego — the latter a novel failure: sol read OPA's
contributor guide and **ended its turn asking the nonexistent user for
maintainer sign-off and DCO attestation** (1-minute attempt, 0-byte patch);
(ii) the 2h cap took pebble and scriggo, scriggo doubly via the known,
still-unfixed mjolnir non-exit hang burning attempt 1 at the wall.

Resume uptake after the report-level affordance: **1 of 43 spawns** — up
from 0/63, and the one use was inside bandit's breakthrough win, but the
affordance alone does not change habits.

Standing after four runs: 12 → 13 → 13 against the 15.2 bar. The near-miss
tax is paid off; the plateau is now precisely the two stochastic taxes, and
the matching levers remain: spend/wrap-up discipline for the cap tax, a
finalize completeness check against instruction-enumerated surfaces for the
wrong-reading tax (cliffy 0/37 fits the opa-rego audit's pattern), and the
mj non-exit hang fix.

## Addendum 2: runs 4-5, the async redesign, and the five-run verdict — 2026-08-01

Run 4 (luna@max + report-level resume affordance): **13/20**, broke both
chronic single-test walls (bandit 88/88, textual 20/20). Trace forensics
after it exposed the real subagent lifecycle: reports deliver only between
primary turns, an implementing primary never ends its turn, and cancel —
the only pull — dropped the report and released the resumable session
(resume: 1/43; one finished analysis destroyed unread; one just-booted
subagent killed defensively after git status showed unexplained edits).

The async redesign (mj `f4aa67d`): ending the turn is the await; every
wake carries finished reports in full plus <subagent_progress> for
still-running subagents (activity watermark shared with reports, diffstat
since spawn); a parked primary is woken with progress alone after
subagents.progress_wake_minutes (default 20); subagent_cancel returns the
full report via a bus claim. Plus mj `1598697`/`3a6ccfc` (session note),
the headless autonomy directive (never block on unobtainable approvals —
added after sol twice obeyed OPA's AGENTS.md contribution gate and quit in
one minute asking a nonexistent user for DCO sign-off), and `max` effort
plumbed through anvil's preset list and brokkbench's effort gate (both
silently/loudly rejected it before).

Run 5 (async + luna@max + debrief): killed at 16 resolved per the new
kill-early policy, **9 W / 7 L, max-possible 13 < 15.2**. Mechanically the
design worked exactly as intended (corrected metrics, deepest-history
counting): 13 wake injections, 9 with progress blocks, exactly 2 heartbeat
wakes, 8 lossless cancel-returns, 14/14 injected reports debriefed, zero
timeouts, median attempt 35m (best ever), first-ever pebble win (59/59).
Behaviorally it halved sol's request volume — and exposed the residual:
four near-miss regressions on previously-won tasks (bandit 83/88, cattrs
68/69, fastapi 136/137, sqlfmt 28/32) at 2-4x speed.

**Forensic correction that reframes everything**: every failing f2p test
in those four lives in the grader's HIDDEN suite; p2p was green (sqlfmt's
4 regressions excepted); the agents authored and ran their own tests, all
green. "Finalize test discipline" was a wrong narrative — nothing runnable
was red. The loss class is **behavior-space coverage of the instruction**:
one edge behavior per task (nested-router override levels,
detailed_validation=False, CLI metric counts) that the divided reading
missed. The old marathon turns covered these incidentally through
redundant grinding; async removed the waste and the incidental coverage
together.

**Five-run verdict** (12, 13, 13, 9/16-killed vs E=15.2): with every
mechanical confound now stripped — no infra, no contamination, no starved
reports, no timeouts, no policy quits, best-ever cost/latency — the duo
still misses 1-few hidden edge behaviors on 3-5 tasks per run, which is
exactly the gap. Vanilla sol is one continuous mind holding the whole
instruction against the whole implementation; the duo divides that
reading and pays the tail behaviors. Coordination taxes interpretation
depth, and this benchmark prices interpretation depth. Remaining lever
rated above an increment: make the primary enumerate the instruction's
behavior surface and cover each item in its own tests before delivering.
If that fails, the honest conclusion is that this task class does not
reward a second mind at any price found across asgard v2 and five mj
configurations.

## Addendum 3: runs 6-7 — multi-edit, discrete review resurrected, and the first result above the bar (2026-08-01)

Two product fixes preceded these runs. **Multi-hunk edit** (anvil
`fca13eb7`-era, modeled on oh-my-pi's replace schema): `edit` takes
sequential `edits` entries, `write_file` owns heavy rewrites; live adoption
was immediate (all batch-shape, up to 5 hunks/call, 11 write_file rewrites
in one attempt vs ~2 historically). **Headless autonomy directive** (mj):
never block on unobtainable approvals — killed the opa-rego policy-quit
class on contact (22/25 with a real patch, then an outright win in run 7).

**Discrete review's silent death, fully diagnosed**: review last ran in
run 1 (single-prompt fallback). Since run 2: `detect_bifrost()` searches
MJ_BIFROST_PATH then PATH; neither reached /opt/work/bin in the container,
and the fatal-review change had removed the fallback — every review died
at birth, unlogged. One env var (MJ_BIFROST_PATH) fixed detection; one
more version skew (mj master's analyze_diff flags vs anvil's bundled
bifrost 0.8.6) needed a separately staged bifrost 0.8.18 for mj
(MJ_BIFROST_BIN → /opt/work/bin/bifrost-mj). The supervisor also gained
the owner-approved bounded completeness mandate: every explicitly stated
requirement must have demonstrated behavior; findings quote the verbatim
requirement span; absent speculative hardening is never a finding.

**Run 6** (multi-edit, no DR, sol+high/luna+max): killed at 15 resolved
per policy, 10 W / 5 TF, max 15 < 15.2. Cleanest failure profile to date
(every loss >= 4/5 partial, no timeouts).

**Run 7** (solo sol+high — no primary subagent tools — DR at sol+medium
with completeness mandate, lanes available on luna but unused):
**16/20 in 2.3h total, the first run above the vanilla expectation of
15.2.** Review supervisor ran in ~87% of attempts. The four losses:
cliffy 36/37, dynamodb 21/37 (0-for-7 across all configs; audit
candidate), and the two tasks vanilla sol also never solves (sqlfmt
28/32, python-statemachine 67/72). Paired: won every vanilla-sure task
except dynamodb, plus opa-template (vanilla 1/4) and opa-rego.

Honest statistics: P(vanilla sol@high >= 16) = 0.42 — above expectation
for the first time after six runs at or below it, not yet significant;
significance needs >= 18 or replication. Cost: ~$8.6/task at current
rates (sol 10.6M input/task, review included) ≈ 2.5x vanilla's $3.47 —
the quality bar moved first; the price bar has not.

The configuration that did it is notable for what it lacks: no worker
DAG, no supervisor-directed checkpoints, no delegation at all. One strong
implementer with good tools (multi-hunk edit, code intelligence), one
fresh independent reviewer holding the verbatim contract with a bounded
completeness mandate, and prompts that name the operating reality. Every
coordination architecture this project built underperformed it.

### Cost decomposition postscript (2026-08-01)

Run-7 spend split by session: primary 163 requests / 10.1M sol input per
task (95%), reviewer 13 requests / 0.5M (5%), subagents zero. Multi-edit
was fully adopted (run 6: 2.07 hunks per edit call, 49% multi-entry, 6
write_file rewrites/task) and request counts still ROSE across runs 5→7
(101 → 115 → 177 sol requests/task). Two supply-side cost narratives
died on measurement (finalize test discipline; edit granularity as the
dominant term — promoted from one 29-edit tail case where edits were
~30% of calls).

Standing conclusion: **step count is a behavioral equilibrium, not a
tooling artifact.** The model works until satisfied; cheaper and richer
affordances get reinvested in more reading and verification rather than
banked. mini-swe's ~38 steps are scarcity, not efficiency. Run 7's 10.1M
primary tokens and its 16/20 are one phenomenon: cost per solved task
$10.70 vs vanilla's $4.60. The one demand-side lever not yet built or
falsified is a model-visible spend budget that changes the satisfaction
criterion. Runs: 12, 13, 13, 9/16k, 10/15k, **16** vs E=15.2.

### Tool-integrity audit (2026-08-01)

Prompted by the cost equilibrium: if we pay 2.5x for more investigation
without more quality, are the tools working as intended, or is sol
fighting them? Three parallel Opus auditors read six run-7 traces
end-to-end (pebble+kysely, dynamodb+cliffy, skrub+python-statemachine).

**Verdict: the request volume survives audit; the tooling does not.**
Tool/env-attributable turns are 5-16% across all six traces (worst in Go,
least in Python), strictly redundant verification 1-4% — the model's
requests are overwhelmingly real work (pebble: 55% productive
first-attempt implementation). But wall clock and several specific
behaviors trace to concrete defects:

1. **rtk wrapper dishonesty (the headline).** At >10MiB the capture
   keeps the FIRST 10MiB and drops the tail — where test failures
   cluster — then summarizes the truncated buffer as if complete:
   "Go test: 930 passed in 66 packages ... Exit code: 1" with zero
   failures listed, twice. Panic stacks and test identities erased;
   ruff diagnostics collapsed to "Found 1 error." even inside a file
   the model had redirected to; the advertised tee log unreachable by
   the sandboxed read_file. Sol's countermeasures were rational and
   expensive: 11 RTK_DISABLED=1 bypasses (~17min re-running, pebble), a
   self-invented 5-part escape incantation used 14x (psm). Ownership
   settled: rtk is vendored INTO anvil (rtk_core; shell.rs rewrites
   every command through `anvil __rtk`), and `ANVIL_RTK_DISABLED=1` is
   an existing zero-code global kill switch — rip-vs-fix is a free A/B.
2. **run_shell_command timeout is milliseconds** (1s round-up floor).
   Skrub's reviewer passed `timeout: 120` meaning seconds, was killed at
   1s three times, and abandoned empirical verification — every finding
   filed source-reviewed-only. Same unit bug exists in the qwencode
   original (worse: no floor, 120 = 120ms). The schema says
   "milliseconds"; the model's unit prior beats the doc.
3. **Container hygiene**: git identity unset in 100% of sessions
   (~2 turns each; the engine only configures identity in the grading
   phase), GOPATH/bin off PATH (6 diagnosis turns + 10 prefixed
   commands), dash-not-bash, one pre-existing OOM package (266s to
   exonerate).
4. **edit batches apply non-atomically** — a mid-batch failure leaves
   the file moved under the model's stale anchors (4-turn recovery
   observed; base rate 0-5%). oh-my-pi ships the same stop-at-first
   semantics but with an explicit recovery script in the error text.
5. **Whole-project verification**: 18 full tsc runs = 10.6% of
   dynamodb's wall; two full pytest suites = 64% of skrub's tool time.
6. **Compaction restarts as request multiplier**: dynamodb restarted 3x,
   re-reading files and replaying one malformed edit verbatim; 64%
   post-compaction re-reads in psm. Plus ~35 of pebble's 104min never
   reached the agent at all (container contention) — wall-clock
   comparisons are contaminated.

**The quality finding that answers the original question**: psm's six
review rounds consumed 53% of the task's requests and moved the hidden
suite zero — rat-holed on ungraded deepcopy edge cases while the one
discoverable spec bug (child-shadows-parent precedence, verbatim in the
spec, inverted by a `reversed()`) sat untouched from minute 15. Pebble's
FIRST review round found a real durability bug that shipped. Round one
of review earns its cost; rounds two through six bought nothing.

**Decision round (owner, 2026-08-01)**: timeout moves to seconds with a
[10..3600] clamp (schema rename `timeout_seconds` proposed, pending);
rtk rip-vs-fix pulled out to anvil#327, pending the free ANVIL_RTK_DISABLED A/B (fix path if
kept: exit-code cross-check — never claim clean when exit != 0 — plus
truncation-aware summaries, tail-not-head capture, skip wrapping
redirected commands, RTK_TEE_DIR into the workspace); container fix
located (deepswe_agent_engine.py prep block + solver env prefix);
edit atomicity recommendation: in-memory apply, single write (we lack
omp's fuzzy fallback, so mid-batch failures are likelier for us);
whole-project verification dropped as not-ours (benchmark fitting);
compaction replay filed as anvil#326.

**Landed (2026-08-01 midday)**: timeout_seconds schema ([10..3600],
deployment cap env, default 60s->120s) + edit-batch recovery script,
anvil `48c7450`; deterministic whitespace ladder for edit matching
(brokk-EditBlock-style tiers, no fuzzy scoring per owner) `7abbe25`,
gates independently re-run green (1362+19). Solver git identity +
go/bin PATH landed in brokkbench `0e415fec` — the audit's "identity
unset" was a prep-vs-solver HOME split: prep's `git config --global`
wrote to /root while the solver reads /opt/work/home. Review churn
ticketed as mjolnir#535; rtk cluster as anvil#327 (rip-vs-fix pending
the ANVIL_RTK_DISABLED A/B). Trace provenance corrected: the audit set
was all run 7; run 6 verified DR-free (0 review sessions in 3 spot
checks). Ready next: fresh musl snapshot -> replication run, optionally
as the rtk A/B.

### Run 8 (replication, all fixes) — killed at max 15, 2026-08-01 evening

Config: run-7 invocation verbatim; anvil 0.24.2 musl `0b228ad4`
(compaction digests #326, rtk replaced by vendored oh-my-pi minimizer,
timeout_seconds, whitespace ladder, batch recovery script), engine
git-identity/PATH fix live. **Killed per policy at 9W/5L of 14 graded —
max possible 15 < 15.2**, the same stopping point as run 6. Run 7's 16
did not replicate; treat 16/20 as within single-run variance
(P(vanilla>=16)=0.42 stands).

Losses, all p2p-green hidden-suite near-misses except one: bandit 83/88
(worse than its chronic 86/88), fastapi 136/137, cliffy 0/37 (its
all-or-nothing signature), opa-template 4/5, and **dynamodb 37/37 f2p —
the first full hidden-suite sweep in eight attempts across every config
— lost to a single p2p regression (1266/1267)**. The freeze-invariant
wall finally fell; the loss class moved.

Field verification of the two watch items (mid-run trace reads):
compaction restart carried the new digest snapshot ("ALREADY DONE ...
do not re-issue tool calls") with 139 messages of productive
continuation after it; minimizer spill round-trip confirmed — model
read `.brokk/shell-output/<id>.txt` back via read_file and got real
targeted test output. The audit's tee-unreachability class is dead.
Harness bug noted: `costUsd` is 0.00 in every run-8 result row; token
counts intact, engine cost calc broken.

Score history: 12, 13, 13, 9/16k, 10/15k, **16**, 9/14k(max 15).

### Probe experiments E1/E2: the step count was scaffold-shaped all along (2026-08-02)

Prompted by the owner's differential challenge ("how is this different in
mini-swe?"): vanilla trajectories from the published artifacts show
mini-swe-agent runs with **zero limits** (step/cost/wall all 0) and no
budget language — same model, no counterweight, 25-36 steps on
bandit/cliffy. The "satisfaction criterion" story was wrong. Measured
head-to-head (bandit): identical avg prefix (~48k tok), identical
shell-batching density (3.5 vs 3.0 ops/cmd); the 4.6x step differential
= micro-action granularity (read/semantic/plan calls at one request
each) + loops vanilla never enters (22 git-diff self-reviews, 29
verification episodes). Vanilla passed without ever running the repo
suite; its prompt scripts a 6-phase linear workflow with an explicit
terminal command.

Two 3-task probes (bandit/cliffy/fd; solo sol+high; DR off), one seed:

| arm | steps (b/c/f) | $ (b/c/f) | f2p |
|---|---|---|---|
| run 8 (full catalog) | 116/160/64 | 6.11/10.09/3.07 | 83-88 F / 0-37 F / W |
| E1 script transplant (full catalog + mswe 6-phase + "commit and end; do not re-inspect") | 55/61/54 | 3.50/3.80/2.87 | **88/88 W** / 36-37 F / W |
| E2 catalog cut (shell+edit+write only) | 35/37/45 | 2.04/2.86/2.62 | 86-88 F / 0-37 F / W |
| vanilla (published) | 25-36 | 2.6-4.4 | pass 3/4 / pass 2/4 / — |

Both levers real: the completion script halves steps with the full
catalog; the catalog cut reaches vanilla step counts and vanilla-or-
below cost. E1's outcomes were the best of any sol arm on these tasks
(bandit 88/88 at $3.50 — vanilla's average price — and cliffy 36/37 vs
run 8's 0/37). n=1/cell; needs replication. Conclusion: **the tool
catalog defines action granularity and the prompt defines the episode's
terminal state; step count follows the scaffold, not a model-internal
criterion.** Next candidates: E1+E2 combined arm; full-20 run of the
best arm. Probe knobs: BPR_INSTRUCTION_SUFFIX_FILE,
BPR_AGENT_TOOL_ALLOWLIST (brokkbench, uncommitted at probe time).
Latent harness bug found: without MJ_EITRI_MODEL the engine takes a
legacy --thor CLI path that current mj rejects (exit 2, instant).

**E3 (script + minimal catalog combined, same 3 tasks)**: bandit 86/88 F
at 37 steps/$2.68; **cliffy 37/37 W — the first full cliffy pass of the
campaign** — at 29 steps/$2.71; fd W at 53/$2.64. Grid summary (avg
$/task, task-wins of 3): run8 $6.42, 1; E1 $3.39, 2; E2 $2.51, 1; E3
$2.68, 2. Outcome differences between probe arms are within n=1
hidden-suite variance (bandit 83-88/88 across arms; cliffy spans
0, 36/37, 37/37); the cost differences are large and consistent. Every
probe arm ran DR-off. Decision pending: full-20 of a probe arm.

### Prod generalization landed; internal A/B begins (2026-08-02)

Plan approved and executed: anvil gains a default-on # Completion
section (checklist-bounded episodes, no speculative re-verification),
cross-type batching guidance, and update_plan batching (anvil
`34895aa`); mj's review correction loop is bounded
(max_correction_rounds default 1, verification-only re-reviews,
mjolnir `3c528d3`, closes mj#535); the harness computes costUsd from
usageByModel tokens (fail-to-None, never silent zero) and the fatal
legacy --thor path is gone (brokkbench `32a53d3`). Snapshots: anvil
0.24.2 `12b3909a`, mj 1.3.0 `27fa97af`. Smoke (fd, no MJ env pins):
SUCCESS 43/43, 45 steps, costUsd $1.90 populated, 8m.

Benchmark reframed per owner: **vanilla-sol-on-anvil vs
sol+luna-on-anvil**, both arms DR-off, 2 runs/arm with run 2 gated on
run-1 sanity; the bar is the other arm, not the overfit mswe published
numbers. Arm A (solo) run 1 launched.

**Arm A run 1 (solo sol, new prompts, DR-off): 14/20, $3.75/task avg,
1.6h sweep.** First outright dynamodb WIN of the campaign (37/37 f2p +
p2p green, $9.98); bandit 88/88 and cliffy 37/37 replicate the probe
wins; pebble 59/59. All six losses are hidden-suite near-misses
(textual 19/20, cattrs 68/69, opa-template 4/5, opa-rego 22/25, psm
67/72, sqlfmt 28/32). Cost sits at vanilla's published price with the
full tool catalog. AAr2 launched at 20 threads; ABr1 mid-flight.
