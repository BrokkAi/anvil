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
