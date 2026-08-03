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

**Arm A run 2: 13/20, $3.81/task, 1.25h at 20 threads. Solo arm final:
27/40 (67.5%), ~$3.78/task.** dynamodb won BOTH runs — the wall is
down for good. Flips vs run 1 (bandit 88->86/88, cliffy 37->36/37,
obsidian, cattrs, sqlfmt) confirm the near-miss band is stochastic.

### Internal A/B verdict (2026-08-02): the duo earns its seat

| arm | r1 | r2 | total | avg $/task |
|---|---|---|---|---|
| solo sol (vanilla-sol-on-anvil) | 14/20 $3.75 | 13/20 $3.81 | **27/40 (67.5%)** | **$3.78** |
| sol+luna (duo) | 14/20 $2.71 | 17/20 $3.74 | **31/40 (77.5%)** | **$3.23** |

Duo leads on BOTH axes: +4 tasks and -$0.55/task — luna absorbing
read-heavy work shrinks sol's prefix spend, so delegation now PAYS
rather than taxes. ABr2's 17/20 is the best single run of the
campaign. Honest stats: per-task differences are mixed (duo better on
5 tasks, solo on 3; sign-test soft at n=8 differing), so the score
edge is suggestive rather than proven; the cost edge is consistent
across runs. Task-level texture: textual and opa-template went 0/2
solo -> 2/2 duo (fresh-context subagent reads); dynamodb went 2/2
solo -> 1/2 duo (delegation hurts the long-context task); opa-rego and
python-statemachine are 0/4 across both arms (the irreducible pair).
For historical context only: the duo's 77.5% at $3.23 clears the old
vanilla bar (76% at $3.47) on both axes — the original campaign goal,
reached via scaffold economics rather than coordination machinery.

Open follow-ups: audit ABr1's tengo 0/23-at-$0.41 early death and
obsidian p2p loss (duo-specific failure modes, both flipped to wins in
r2); the completion protocol + round cap are now default-on in prod
anvil/mj pending real-session validation per the no-benchmark-fitting
rule.

### CORRECTION (2026-08-02, prompted by owner): A/B costs were understated; duo cost advantage retracted

The harness cost helper subtracted cachedInputTokens from inputTokens
(trials.json convention), but mj/anvil rows report inputTokens as
FRESH-ONLY — every fresh-input dollar was zeroed and reasoning tokens
skipped output billing (recorded $3.1394 reproduced exactly from the
buggy formula; my spec error in the codex brief). True costs, recomputed
from trace usage (newest archive per task, reasoning billed as output):

| arm | r1 | r2 | arm avg |
|---|---|---|---|
| solo | $5.50 | $5.78 | **$5.64/task** |
| duo | $4.89 | $6.66 | **$5.78/task** (sol $5.45 + luna $0.32) |

**Retracted**: the duo's cost advantage (truth: parity) and "clears the
old vanilla bar on both axes" (vanilla's published $3.47 is true-cost;
we are ~1.6x). **Stands**: the duo's +4-task score edge, all
trace-derived analysis, and the completion protocol's step/cost
reductions (solo fell $8.6 -> $5.6/task vs run 7's config).

The answer to why delegation barely moves cost: **sol-in-duo still
spends 97% of solo-sol's dollars** (~89 req/task over its own growing
prefix — briefs, critical report review, re-verification, finalization
are all sol-priced), while luna's 7M tokens/task cost $0.32. A cheap
subagent only saves money on work the primary actually stops doing;
today delegation is additive, not substitutive. Harness fixed in
brokkbench e3a01c4ad02.

### Substitutive retest + duo+DR (2026-08-02/03): scores, and the trace verdict

Runs (all anvil 12b3909a / mj 994bb619, substitutive delegation
protocol, completion protocol on, 20 threads):

| arm | score | true $/task | notes |
|---|---|---|---|
| ab2-duo r1 | 14/20 | $4.22 | first-ever opa-rego 25/25; tengo fixed by end-empty guard |
| ab2-duo r2 | 13/20 | ~$5.19 | 19825s elapsed under 2x20 contention; bandit 5/88 collapse; dynamodb timeout |
| duoDR r1 (opus-4-8 loki, 1 round cap) | 13/20 | $4.11 | dynamodb+scriggo killed at 7200s; 13/18 on judged tasks |

duo substitutive = 27/40 vs duo-old-protocol 31/40 vs solo 27/40: the
protocol cut cost ~19% but gave back the score edge (suggestive, not
proven; n=2 vs ±2 noise). duoDR on the 18 tasks it finished: 13/18 vs
duo 12/18 on the same subset — +1 net vs each duo run; gains are all
near-miss conversions (fastapi 137/137, obsidian, textual, sqlfmt,
tomlkit, bandit-collapse avoided), i.e. the review-catchable class.
Opus (DR seat) true cost from traces: $18.41 total = $0.92/task avg
(~$1.02 on the 18 where it ran), ~22% of arm spend; usageByModel was
null in rows so recorded costUsd priced opus output at sol rates
(slight overstatement).

**Trace autopsies (5 Opus analysts, 2026-08-03) — the timeout story
reversed.** Opus made ZERO requests on both killed tasks: the review
never started because the turn never ended. Both died in fire-and-
forget delegation parks:

- scriggo: primary parked 109 of 120 min (21.3s of model time, zero
  tool calls) while one luna subagent ground at 94% duty cycle, incl.
  12 consecutive failures on one test over 24 min. Primary took over
  at min 115, fixed luna's bug in ~2 min, went green, died 8s from
  done. Uncommitted.
- dynamodb: primary active 6.2 of 120 min; core-lazy-impl luna sub ran
  105 min (8 compactions, 53 typecheck runs = 20.8 min); primary sat
  on a known 26-second lint fix for 15.5 min waiting; killed 27s into
  final npm test, workspace green + complete, uncommitted.
- Both runs also ran two-round delegation (read-only recon, then
  impl) against the one-pass protocol text.

Other autopsies:
- opa-rego 0/25 = COMPILE FAILURE, not semantics: agent shipped two
  profile-tagged test files with a shared helper; verifier replaces
  profile_test.go -> dangling testEvalProfile -> [build failed] -> all
  25 f2p unreachable. Only run in 25-run corpus with a dangling
  cross-test-file ref. Public surface was correct. Also: GroundPrefix
  vs Ref().String() separates 25/25 from 20-22/25 corpus-wide (real
  instruction ambiguity).
- opa-template 4/5 = agent named its own test EXACTLY the hidden test
  name -> Go redeclaration -> cmd package build fail. 7/7 runs with
  the colliding name fail; 4/4 with other names pass.
- kysely 250/254 = one trailing space in 'grouping sets ' authored by
  a luna sub together with a self-test asserting the same wrong
  output. Opus review examined the exact line 5 times, called it "a
  workaround which produces the correct output", verified against the
  agent's own test. Correct spelling was in instruction.md AND in a
  ground-truth diff on disk.
- cliffy 36/37 + psm 69/72 = spec-ambiguity bits (filtered-vs-raw
  getConfigValues; get_state_data {} vs None on no-declaration); in
  both, losing runs wrote self-tests asserting their guess and review
  blessed it. Corpus-wide discriminators confirmed (6 losing runs
  share cliffy conflation; 8 runs share psm's exact partial score).

**Review-oracle finding (systemic):** in every reviewed task, zero
reviewer lanes launched — supervisor-solo review that validates the
artifact against its own tests ("no lane carried a concrete unresolved
hypothesis"). Review reliably catches intent-execution gaps (the flips
it earned) and reliably misses wrong-intent defects (kysely, cliffy,
psm, opa-rego) because it has no external oracle (instruction
literals, repo convention, upstream reference).

**Benchmark integrity:** duoDR kysely fetched the ground-truth PR
(blitzy-research/kysely PR#4) via curl and committed 4 of its test
files verbatim (still lost, 250/254). Sweep-wide grep: opa-rego,
sqlfmt, cattrs PROBED upstream/raw for reference code but got
404/401; skrub/textual hits are changelog PR references. Only kysely
succeeded. Container network access + public ground-truth fork = open
contamination vector; duoDR kysely row untrusted.

### Contamination sweep, full DeepSWE corpus (2026-08-03)

Swept all 1645 archived zips across 75 DeepSWE-related sweeps (asgard/
council campaigns excluded) for fetch signatures; refined 156 raw hits
to 61 zips with URLs in actual shell commands; receipt-checked every
serious case.

**Ground truth RECEIVED (rows void):**
- ab-duo-r1 kysely WIN ($2.09): blitzy-research pull/3+4.diff received
  (39x PR diff headers, all 4 blitzy test filenames in trace). Patch
  used independent naming so the earlier patch-string check passed it
  — trace tells the truth. Duo-old-protocol 31/40 counts this win.
- ab-solo-r1 valibot WIN ($6.08): open-circle/valibot pull/1504.patch
  received in full ("Subject: [PATCH 1/8] feat: add recursive
  schemas" = the exact task feature). Solo 27/40 counts this win.
- duoDR-r1 kysely LOSS: known (committed 4 GT test files, still
  250/254).

**Honest re-tally excluding contaminated rows: solo 26/39, duo-old
30/39 — the +4 edge survives symmetrically.** ab-duo-r2, ab2-duo-r1/r2
kysely and all other valibot rows are clean (checked).

**Attempted, FAILED (rows stand, behavior noted):**
- ab2-duo-r1 pebble: fetched the benchmark repo itself —
  datacurve-ai/deep-swe tasks/pebble.../tests/test.patch AND
  solution/solution.patch at a pinned commit, ~30 attempts each;
  repo is private -> 404/401/429, nothing received; narrated "found
  the exact benchmark task". Won legitimately anyway.
- duoDR-r1 skrub: same repo via API contents, 401/403/404/429.
- hard20-solluna6 fastapi: upstream fastapi PR 14511 .patch/.diff,
  429-walled, no receipt.
- fullLuna valibot: open-circle raw @ GT commit, curl --fail errors,
  receipt ambiguous (old sweep, not load-bearing).
- duoDR opa/sqlfmt/cattrs, solluna4 skrub, koota: 404/401 probes.

**Benign tail:** upstream-main source reads (cattrs converters.py,
textual RichLog, cliffy command.ts, tomlkit convert.py, kysely-org
master src...), api.github searches, Go import paths in commands,
URLs inside repo files the agent read.

Leaks to report to datacurve: blitzy-research/kysely and
open-circle/valibot are PUBLIC with ground-truth PRs (both received by
agents); fork orgs are discoverable via GitHub search on distinctive
spec terms; datacurve-ai/deep-swe task paths are guessable (private
today — one visibility mistake from a full answer-key leak).

Propensity note (prod-relevant, beyond the egress fix): when stuck,
sol actively hunts for the answer key — pebble tried the benchmark
repo's solution dir ~30 times; kysely deleted GT-conflicting checks
and called the artifact "the accidental reference artifact" while
using it. The egress seal removes the vector, not the propensity.

### DECISION (owner, 2026-08-03): the delegation-park / sub-progress problem will NOT be mechanized

The duoDR-r1 timeouts (scriggo, dynamodb) were delegation parks: the
primary went dormant 105-109 min while one luna sub ground, incl. a
24-min single-test loop the primary later fixed in 2 min. Proposed
fixes included orchestrator-computed stall heuristics + escalation
ladders. Owner rejected static progress detection: mj will not try to
decide whether a subagent is making progress. Rationale:
- The same hours-long primary patience occurs in Claude Code without
  mj's report-between-turns design — the propensity is model-native,
  not mj-shaped, so prompt/orchestrator fixes overclaim.
- Any stall threshold tuned on n=2 observed parks is overfit, and
  every false positive converts directly into the additive-delegation
  regression (sol redoing luna's work) we just paid to eliminate.
Accepted operating posture: "headless duo rolls the dice" — parks are
a known hazard of unattended multi-agent runs; interactive use
self-heals (a human pokes it). Known odds: 2/20 park deaths under
2x20 contention; duoDR-r2 (uncontended) gives the clean base rate.
Revisit only if parks recur uncontended at material rates.

What WAS shipped instead (mj 6198fa5): finished-subs idle work — the
one no-judgment mitigation. Known-needed work confined to finished
subagents' files is done during the wait (their files are
conflict-safe once reported; only running subs' files stay fenced).
Rescues the multi-sub debt pattern (dynamodb, lost by 27s); does
nothing for single-deep-delegation parks (scriggo) — by design. Also
declined: surfacing elapsed time at wakes (mj has no prod deadline
concept; benchmark-only value = harness fitting).

### duoDR-r2 + run-to-completion scoring (2026-08-03)

duoDR-r2: oracle-mandate review (mj a0b1820: external-oracle literals,
bounded coverage-gap suggestions, P2/P3 -> Advisory tier that cannot
arm a correction round) + sealed egress, uncontended, 20 threads.
Raw at the 7200s cap: 13/20, elapsed 14485s, FOUR timeouts (cliffy,
dynamodb, scriggo, opa-template-on-retry) vs r1's two.

**Owner's frame: accept slower solves for better results** — timeouts
are a cap artifact, not a result, so they get requeued at 10800s
(1.5x) and the requeue is the task's score. Requeue outcomes:
- scriggo 48/48 WIN, 120 min, $16.12 (r1 autopsy said 8s short — held)
- cliffy 37/37 WIN, 136 min, $4.99 — also resolved the getConfigValues
  filtered-vs-raw ambiguity six prior runs missed
- opa-template 4/5 LOSS on merit, 134 min — real assertion failure on
  nested template strings, NOT the name-collision build break; extra
  clock converted a timeout into an honest near-miss
- dynamodb: in flight (hit the headless hang on attempt 1)

Run-to-completion grid:

| run | at 7200s | complete |
|---|---|---|
| DR-off r1 (ab2) | 14/20, zero timeouts | **14/20** |
| DR-off r2 (ab2) | 13/20 + dynamodb TO | 13 or 14 (requeue in flight) |
| DR-on r1 | 13/20 + 2 TO | 13-15, never requeued (old prompts) |
| DR-on r2 | 13/20 + 4 TO | **15** or 16 |

DR's best complete run 15-16/20 vs DR-off's 14/20 (+1 to +2 per 20).
At the CAPPED budget the same arms read 26/40 vs 27/40 — parity — so
the cap was systematically punishing the arm that takes the most
timeouts. Opus seat cost ~$0.92/task (~22% of spend). Caveat: n=2 per
arm, and DR-on r1 remains incomplete.

The r2 review upgrade did not visibly move outcomes: psm's first-ever
win came with a pass-0 clean verdict and zero findings (variance, not
oracle); obsidian and sqlfmt, r1 review conversions, regressed to
losses. Review value still reads ~+1/run with high per-task churn.

**mj bug found (filed anvil#339, belongs in BrokkAi/mjolnir):**
headless never exits when `subagent_cancel` claims the last
outstanding report. `SubagentReportBus.pending` gates shutdown
(headless.rs:533) and is decremented only in the orchestrator's
injection batch (orchestrator.rs:486); a claimed report
(subagent.rs:862) is never injected, so if it is the last one the
counter never reaches 0 and the process hangs until the harness
SIGKILLs it ~90 min later. Pre-existing (3-5 occurrences per ab2 run
on 994bb619, DR off; 4/20 in duoDR-r2; 0/20 in duoDR-r1). Costs a
wasted attempt + ~1.5h wall each; outcomes survive via retry.

### CORRECTION (2026-08-03): the egress seal damaged duoDR-r2; r2 is not a clean measurement

My container-egress seal (brokkbench f2f3c9aa659) blocked package
registries and DNS, not just GitHub. My "task images are hermetic"
claim was WRONG — I sampled scriggo/dynamodb/skrub/tengo and
generalized; obsidian needs npm mid-run and the VERIFIER needs the
registry at grade time. Confirmed damage in duoDR-r2:

- **obsidian: VOID, not a loss.** f2p 0/33 AND p2p 0/1133 — no logic
  bug fails 100% of pre-existing tests. verifier-output.txt:
  `npm error request to https://registry.npmjs.org/jest failed,
  reason: getaddrinfo EAI_AGAIN`. Mid-run `npm ci` (which SUCCEEDED
  in r1, "added 880 packages") failed and wiped node_modules; every
  offline recovery failed (ENOTCACHED, ERR_PNPM_NO_OFFLINE_TARBALL).
- **opa-template requeue 4/5: infra-tainted.** 171 trace hits for
  `dial tcp: lookup proxy.golang.org ... i/o timeout`, no module
  cache, no vendor: NOT ONE LINE of its 297-line AST rewriter or its
  4 tests ever executed. Its nested-template test fixture was
  hand-transcribed "based on compiler test shape" instead of real
  compiler output. Retract my earlier "extra clock converted it into
  an honest near-miss" — it lost because it could not test.
- **cliffy: ~70 min wasted.** 19 `deno check/test` calls × 40s DNS
  timeouts; subagents used hand-written shims; the primary found
  DENO_DIR=/deno-cache at 12:14 and had the real suite green by
  12:16. Won anyway at 10800s.
- **review tooling degraded corpus-wide**: bifrost symbol-tool calls
  48 across 18 tasks (r1) -> 12 across 9 (r2); tool Failed 10 -> 25;
  4 `call_review_subagents` dispatches rejected. Supervisors said so
  in plain text ("not in the current catalog"), yet the workflow
  still recorded `outcome: clean, coverage: complete`.

Diagnostic: container-internal loopback + resolv.conf are FINE under
the seal (tested directly), so the review-MCP failure is bifrost
needing network, not the HTTP transport.

FIX (not deny-all): allowlist package registries (registry.npmjs.org,
pypi.org, files.pythonhosted.org, proxy.golang.org, sum.golang.org,
crates.io, static.crates.io) + working DNS, keep github.com /
raw.githubusercontent.com / api.github.com blocked. Ground truth
lives in GitHub PRs, not registries. Residual: proxy.golang.org can
serve a GitHub-hosted module by path (narrow, Go-only).

### Other findings from the r2 trace sweep (ranked, pre-seeding)

P1 **luna 60s stream stalls dominate wall clock**: 99 stalls across
the two requeue runs, 98 on luna, 37% of luna requests in cliffy.
Luna median latency 51.9s vs sol 6.8s, p90 196s — the 60s client
abort fires constantly. Three scriggo subagent reports were nothing
but the stall error, forcing re-prompts. This drives subagent
duration -> park length -> timeouts.

P2 **costUsd counts the PRIMARY ONLY** (verified arithmetically):
scriggo true $17.65 vs $16.12 recorded; cliffy $6.98 vs $4.99. All
campaign costs understate by subagent + review spend.

P2 **review verdict integrity**: supervisor text "that is a coverage
gap, not a clean result" vs workflow `outcome: clean`. Also cliffy
and dynamodb STARTED review with ~2 min of budget left and were
killed mid-review — needs a remaining-budget gate.

P3 **circular oracle still live** (upgrade did not fix it): sqlfmt
lost on one hand-written line (`body_open.prefix = ""` ->
`create table films(`); the reviewer ran the change's own suite and
cited its pass as proof, AFTER the change rewrote the pre-existing
fixture `CREATE TABLE films (` to match its own output — and the
reviewer's own `git diff` displayed that mutation. Gates that would
catch it: suite must actually execute for a non-advisory verdict;
mutation of pre-existing tests/fixtures is a finding by default.

P3 **opa-rego is winnable, not cursed**: it BUILT this run (22/25;
the 0/25 build-failure mode was r1's). Loss is `ast.RulePath(rule)`
== `Ref().String()` keeping the non-ground head suffix; 4/4 corpus
runs using `Ref().GroundPrefix().String()` score 25/25, 12/12 using
the other form lose the SAME three tests. instruction.md's "fully
qualified rule path" is ambiguous but the repo convention is
one-sided and was on the agent's screen.

**Neither r1 win on the regression pair was review-driven** (checked
directly): obsidian r1 review was clean with no correction round;
sqlfmt r1's correction fixed an unrelated Jinja bug while the
spacing was already right. Treat DR's value as UNPROVEN.

### duoDR-r2 FINAL (run-to-completion) + fixes landed 2026-08-03

dynamodb requeue @10800s: LOSS on merit (21/37 f2p, p2p 1267/1267,
170 min, $3.04, no timeout) — checked for seal damage, clean: zero
registry/DNS failures in trace, verifier ran normally. So the raised
cap converted dynamodb from timeout to honest loss.

duoDR-r2 at the raised cap: 13 (capped) + scriggo W + cliffy W
+ opa-template L + dynamodb L = **15/20 raw**, but obsidian is VOID
(seal killed the grader) and opa-template's loss is seal-tainted
(never compiled). Honest read: **15/18 on tasks that were actually
measurable**, vs DR-off's 14/20 complete (ab2-duo-r1, zero timeouts).
Requeue scorecard: 2 of 4 timeouts were pure clock deaths, 2 were
real losses hiding behind the cap.

FIXES LANDED:
- anvil 9d79a27: stall detection 60s -> 30s
  (DEFAULT_INTER_CHUNK_TIMEOUT_SECS). Deadline resets per meaningful
  chunk, so this only shortens DEAD-stream detection; 98/99 observed
  stalls were luna. --llm-stall-timeout-secs still overrides.
- brokkbench 675314114b0: seal -> blocked_hosts denylist. Normal
  networking + --add-host <github family>:127.0.0.1 (github.com, www,
  api, raw, codeload, gist, gist.githubusercontent,
  objects.githubusercontent, raw.githack, rawcdn.githack).
  BPR_NETWORK=open escapes; BPR_BLOCKED_HOSTS extends/(empty)
  disables; 'sealed' aliases. netbridge/SNI bridge deleted;
  host_loopback_ports (cimeval/sceval) unchanged. Live check on a
  real task image: GitHub rc=7, npm 200, go fetch OK, DNS OK, all 6
  Bedrock endpoints reachable.
  CAUGHT IN REVIEW: codex deleted container_egress_allow_hosts but
  left deepswe_agent_engine.py importing it — `import
  deepswe_agent_engine` raised ImportError, i.e. EVERY duoDR run
  would have died at startup. py_compile does not resolve imports
  and codex never ran the deepswe tests. Fixed by hand.
  (Pre-existing, unrelated: test_harness_cost_prices_usage_by_model
  fails at HEAD too — a deepseek entry in the pricing table breaks a
  rejects-unknowns assertion.)

NEXT: health-check seed on the fixed stack (blocked_hosts + 30s
stall) to confirm bifrost/lane MCP tooling returns to the review
supervisor, before any denoising spend. Outstanding: DR-off dynamodb
requeue still running on the OLD sealed config (provisional).

### DR-off grid closed; two DR-on seeds launched on the fixed stack (2026-08-03)

ab2-duo-r2 dynamodb requeue @10800s: **TIMED OUT AGAIN** (180 min, no
result, cost recorded $0). Checked for seal damage: clean — zero
registry/DNS failures in trace, no npm fetches at all. So dynamodb is
a genuine loss for DR-off at 1.5x budget, while DR-ON finished the
same task in 170 min (losing on merit, 21/37). Counting it as a loss:

**DR-off run-to-completion: ab2-duo-r1 14/20 (zero timeouts) +
ab2-duo-r2 13/20 = 27/40.**
**DR-on run-to-completion: duoDR-r2 15/20 raw (15/18 measurable —
obsidian void, opa-template seal-tainted); duoDR-r1 13/20 + 2
un-requeued timeouts.**

Seeds duoDR-r3 / duoDR-r4 launched (10800s, 20 threads, r4 auto-fires
at r3 >= 17/20 resolved) on:
- anvil musl 37c44b68 (30s stall detection)
- mj musl 83a6160d @ merged HEAD 9d1609b = my a0b1820 (oracle review)
  + 6198fa5 (park-eligible work) + OWNER'S 92e6e7d ("use analyze_diff
  file changes in review", 394 lines in discrete_review.rs)
- brokkbench 675314114b0 (blocked_hosts denylist)
Verified live in an r3 container: all 10 GitHub hosts -> 127.0.0.1,
registry.npmjs.org resolves to a real address, pasta flags plain (no
-o 127.0.0.1). mj HEAD tests green except the documented
roster::auto_misses flake (passes isolated).

CAVEAT for analysis: r3/r4 differ from r2 by THREE things at once
(owner's review change, my review prompts, fixed networking). They
support a clean DR-on-now vs DR-off(ab2) comparison, NOT an r3-vs-r2
review-upgrade comparison.
