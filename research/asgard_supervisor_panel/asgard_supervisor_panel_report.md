# Asgard supervisor replay panel

## Method

The panel contains four archived Asgard endpoints that the supervisor incorrectly declared
complete and two nontrivial endpoints it correctly declared complete. Replays use the native
DeepSeek V4 Pro endpoint with reasoning enabled. They reconstruct the final three candidate
histories, withhold the grader result, and do not run candidates, tools, repositories, or graders.

The production-faithful dossier places the selected naturally-ended lane's test-file inventory
and cumulative non-test diff before its trajectory, matching Anvil. The first baseline,
`adversarial`, `contract_binding`, and `evidence_binding` experiments predated that ordering fix;
their qualitative failure modes remain useful, but their scores should not be treated as a clean
comparison with the later prompts.

## Cases

| Case | Expected | Why |
| --- | --- | --- |
| Drizzle r1 | incomplete | `preceding(0)` / `following(0)` were rejected, and primitive lag/lead defaults rendered incorrectly |
| Happy DOM r2 | incomplete | shutdown did not cancel an already-blocked stream read, causing integration timeouts |
| Ofetch r2 | incomplete | retry/failure state violated logical-request and cooldown/origin contracts |
| Wazero r2 | incomplete | errors did not contain the required contiguous `module closed` phrase |
| Happy DOM r1 | complete | passed 14/14 F2P and 165/165 P2P tests |
| Returns r1 | complete | passed 110/110 F2P and 172/172 P2P tests |

## Hand-reviewed results

The concise baseline reproduced all four false completions and preserved both correct
completions: **2/6**.

The `adversarial` prompt produced **4/6** boolean decisions and preserved both correct controls.
It cleanly caught Wazero. It also stopped Drizzle, but for a candidate-test API problem while
missing both production defects; that is not a useful supervisor win. It still declared Happy
DOM and Ofetch complete after describing concrete risks. The hand-reviewed useful score is
therefore **3/6**.

The `contract_binding` prompt cleanly caught Ofetch's premature retry failure accounting and
Wazero's exact-string violation, but still declared Drizzle and Happy DOM complete: **2/4** on
the known-bad cases. Controls were not rerun because this prompt was already not a winner.

`evidence_binding` still declared Drizzle complete. After correcting dossier order,
`endpoint_behavior` and `grounded_contract` also declared both Drizzle and Happy DOM complete.
Those runs were stopped after the two informative failures rather than spending calls on Ofetch
and Wazero, which the earlier contract prompt had already handled.

## What the failures show

- Wazero was a completion-threshold error: Pro found the exact string defect and initially
  rationalized it as a minor phrasing issue. Adversarial contract wording fixed it.
- Ofetch was also tractable by prompting: explicit binding from the discovered retry scenario to
  the task contract changed the decision to incomplete.
- Happy DOM is not an information-discovery failure. Across several prompts, Pro correctly stated
  that a forever-pending `reader.read()` is never unblocked because the abort handler only sets a
  flag, then called that scenario out of scope despite the task explicitly requiring shutdown to
  interrupt body consumption. Stronger one-turn wording did not resolve the contradiction.
- Drizzle is an evidence-grounding failure. Pro repeatedly trusted the candidate's checklist and
  did not reconcile the actual `value < 1` predicate or raw default rendering with the task, even
  when instructed to locate the enforcing expressions. A cumulative diff and opportunistic code
  excerpts are not a reliable substitute for repository inspection.

## Verdict and next experiment

Prompt-only changes improved some decisions but did not produce a prompt worth installing as a
general solution. The next controlled experiment should keep this same six-case panel and let Pro
make read-only, lane-aware inspection calls against the provisional winner before the terminal
`select_trajectory` call. Include `read_file`, `grep_search`, `list_directory`, and Bifrost's
`search_symbols`, `get_symbol_sources`, `get_symbol_locations`, and `get_summaries`; exclude Bash,
edits, web, and subagents.

That experiment directly addresses Drizzle's missing evidence. Happy DOM additionally suggests a
multi-step audit/decision flow: inspection and counterexample assessment should precede the final
selection call, because the one-turn model can understand a decisive counterexample and still
rationalize it away in the same response.

## Reproduction

```bash
python3 research/asgard_supervisor_panel/asgard_supervisor_panel.py \
  --api-key-file ~/.secrets/deepseek_bpr_key \
  --prompt research/asgard_supervisor_panel/asgard_supervisor_prompts/grounded_contract.txt \
  --output /tmp/asgard-supervisor-panel.jsonl
```

The archive paths and expected hidden verdicts are listed in
`research/asgard_supervisor_panel/asgard_supervisor_panel.json`.
