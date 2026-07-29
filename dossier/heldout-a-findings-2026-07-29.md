# Held-out A: the generalization result and the transfer problem

*2026-07-29. First measurement on tasks never used for iteration. 27 tasks
drawn from the 54 luna-solvable held-out set (stratified split; sibling set B
and the 24 luna-unsolvable tasks remain sealed). Binary anvil-e82c8ea:
simplification doctrine + time leases. Supervisor bedrock sol@high, workers
luna@xhigh, runs=1.*

## Result

| | rate |
|---|---|
| asgard (sol supervising luna) | 17/27 = 63.0% raw; 65.4% excluding one timeout |
| vanilla luna @xhigh (published) | 54.6% → **1.15x** |
| vanilla sol @high (published) | 71.3% → **0.88x** |

The expectation bar (beat the worker model) is cleared modestly. The actual
goal (beat the supervisor model working alone) is missed.

Note the probe-35 comparison is not a regression: that set was ~98%
vanilla-luna tasks where asgard scored 80% (0.82x ratio). Held-out A is a
harder set where asgard scores 61.5% (1.14x vs luna). In ratio terms the
design generalized; the headline fell because the tasks got harder.

## The diagnostic cut

Grouping tasks by how well vanilla sol does on them alone:

| vanilla sol | n | asgard | vanilla luna | vanilla sol |
|---|---|---|---|---|
| 4/4 | 13 | 69% | 58% | 100% |
| 3/4 | 4 | 100% | 69% | 75% |
| <=2/4 | 10 | 40% | 45% | 32% |

**Sol's competence does not transfer through supervision.** Where sol solves
a task every time alone, asgard solves it 69% of the time: +11 points
over luna-alone, -31 points against sol-alone. Where luna struggles, asgard
(40%) is at or below luna-alone (45%). The system behaves as *the worker plus
a small boost*, uniformly — not as the supervisor's judgment executing
through the worker's hands.

Paired view: 4 tasks lost that vanilla sol solves 4/4 (csstree-shorthand-
expansion, claude-code-by-agents-recursive-delegation,
koota-composite-trait-aspects, scriggo-method-declarations/timeout) against
2 won where vanilla sol scores <=1/4 (quill-shared-toolbar-focus,
sqlfmt-create-table-ddl-formatting). Supervision trades 4 reliable
supervisor solves for 2 wins in territory neither model owns.

## Falsifiable prediction for the reserved 24

The 24 luna-unsolvable tasks are, by construction, ones vanilla luna scores
0/4 on; vanilla sol scores 34% per-attempt across them. If asgard is
"worker + small boost", asgard should land well below 34% — i.e. the
beat-vanilla-sol test fails for structural reasons rather than fixable
harness or doctrine bugs.

This prediction is worth testing before investing further in supervisor
mechanisms, because it discriminates between two very different worlds:
(a) supervision is a fixable engineering problem with remaining bugs, or
(b) supervision cannot transfer supervisor competence to a weaker worker,
in which case the architecture's value proposition needs restating (e.g.
parallel breadth and cost, not capability amplification).

## Harness bug found and fixed during the sweep

brokkbench 10a7e737: instance_is_running treated a FAILED describe-instances
call as a spot reclaim. At 22-VM width EC2 throttles that call, so three
healthy VMs were killed mid-attempt and their tasks restarted; only one of
four reported deaths was real. Now retried, with only positive AWS answers
counting as death. Recovery additionally required clearing sticky attempts
counters in the orchestrator state file, which otherwise refuses retries with
"FAILED after 4 attempts: unknown".

This is the third bug of an identical shape in this project (failed
observation collapsed into a definite negative: quota 429s laundered as
normal completions; a region-mismatched query read as termination; a
throttled query read as termination). The general lesson is that "unknown"
needs to be a representable state that callers must handle, which is the
same principle the sl3 postmortem reached about supervisor evidence.
