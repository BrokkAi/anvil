#!/usr/bin/env python3
"""Compare dynamic Asgard windows with the explicit-probe policy.

Inputs are Anvil result archives produced by the Q4 policy experiment. The
analyzer uses structured policy, window, kind, and candidate-usage trace records;
it never infers a probe from prose. Candidate cost includes both each lane's tool
loop and its mandatory window-summary call, exactly as charged by Anvil.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


USAGE_KEYS = ("input", "output", "thought", "cachedRead", "cachedWrite")


def discover_inputs(paths: Iterable[Path]) -> list[Path]:
    archives: set[Path] = set()
    for path in paths:
        if path.is_dir():
            archives.update(path.rglob("*.zip"))
        elif path.suffix == ".zip":
            archives.add(path)
        else:
            raise ValueError(f"not a .zip or directory: {path}")
    return sorted(archives)


def _member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name))
    except KeyError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{archive.filename}:{name} is not an object")
    return value


def _trace_rows(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    try:
        raw = archive.read("anvil-trace.jsonl").decode("utf-8", "replace")
    except KeyError as error:
        raise ValueError(f"{archive.filename} has no anvil-trace.jsonl") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{archive.filename}:anvil-trace.jsonl:{line_number}: {error.msg}"
            ) from error
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _zero_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def _add_usage(total: dict[str, int], value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("usage must be an object")
    for key in USAGE_KEYS:
        amount = value.get(key)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError(f"invalid usage.{key}: {amount!r}")
        total[key] += amount


def _rollup(value: dict[str, int]) -> dict[str, int | float | None]:
    raw = value["input"] + value["cachedRead"] + value["cachedWrite"]
    cache_eligible = value["input"] + value["cachedRead"]
    return {
        **value,
        "rawInput": raw,
        "cachedReadFraction": (
            value["cachedRead"] / cache_eligible if cache_eligible else None
        ),
    }


def _indexed_rows(
    path: Path, rows: list[dict[str, Any]], record_type: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("type") != record_type:
            continue
        window = row.get("window")
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError(f"{path}: {record_type} has invalid window {window!r}")
        if window in indexed:
            raise ValueError(f"{path}: duplicate {record_type} for window {window}")
        indexed[window] = row
    return indexed


def _validated_candidate_usage(
    path: Path,
    window: int,
    count: int,
    steps: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    if row.get("candidate_count") != count or row.get("window_steps") != steps:
        raise ValueError(f"{path}: window {window} candidate usage count/steps disagree")
    lanes = row.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != count:
        raise ValueError(f"{path}: window {window} candidate usage has wrong lane count")
    lane_ids = [lane.get("lane") for lane in lanes if isinstance(lane, dict)]
    if sorted(lane_ids) != list(range(count)):
        raise ValueError(f"{path}: window {window} candidate usage has invalid lane ids")
    summed = _zero_usage()
    for lane in lanes:
        _add_usage(summed, lane.get("usage"))
    total = _zero_usage()
    _add_usage(total, row.get("usage"))
    if summed != total:
        raise ValueError(f"{path}: window {window} lane usage does not sum to total")
    return total


def extract_run(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        rows = _trace_rows(archive)
        result = _member(archive, "result.json")
        reward = _member(archive, "reward.json")
    configs = [row for row in rows if row.get("type") == "asgard_window_policy_config"]
    if len(configs) != 1:
        raise ValueError(f"{path}: expected one asgard_window_policy_config")
    config = configs[0]
    mode = config.get("mode")
    if mode not in {"dynamic", "explicit-probe"}:
        raise ValueError(f"{path}: invalid window policy {mode!r}")
    if config.get("shadow_survivor_study") is not False:
        raise ValueError(f"{path}: probe-policy analysis excludes shadow survivor runs")
    windows = _indexed_rows(path, rows, "asgard_window")
    kinds = _indexed_rows(path, rows, "asgard_window_kind")
    usage_rows = _indexed_rows(path, rows, "asgard_candidate_window_usage")
    if mode == "dynamic" and kinds:
        raise ValueError(f"{path}: dynamic run unexpectedly contains window-kind records")
    if mode == "explicit-probe" and set(kinds) != set(windows):
        raise ValueError(f"{path}: explicit-probe window-kind coverage is incomplete")
    if not set(usage_rows).issubset(windows):
        raise ValueError(f"{path}: candidate usage references an unknown window")
    candidate_usage = _zero_usage()
    joint = Counter()
    kind_counts = Counter()
    lane_steps = 0
    tournament_eligible = 0
    rendered_windows: list[dict[str, Any]] = []
    for window, row in sorted(windows.items()):
        count = row.get("candidate_count")
        steps = row.get("window_steps")
        if not isinstance(count, int) or not isinstance(steps, int):
            raise ValueError(f"{path}: window {window} lacks count/steps")
        kind = kinds[window].get("kind") if mode == "explicit-probe" else "dynamic"
        if mode == "explicit-probe" and kind not in {"probe", "work"}:
            raise ValueError(f"{path}: window {window} has invalid kind {kind!r}")
        eligible = count >= 2 and steps <= 2
        tournament_eligible += eligible
        joint[f"{count}x{steps}"] += 1
        kind_counts[str(kind)] += 1
        lane_steps += count * steps
        usage = usage_rows.get(window)
        if usage is not None:
            validated_usage = _validated_candidate_usage(
                path, window, count, steps, usage
            )
            _add_usage(candidate_usage, validated_usage)
        rendered_windows.append(
            {
                "window": window,
                "kind": kind,
                "candidate_count": count,
                "window_steps": steps,
                "lane_steps": count * steps,
                "tournament_eligible": eligible,
                "candidate_usage_captured": usage is not None,
            }
        )
    nested_reward = result.get("reward")
    nested_reward = nested_reward if isinstance(nested_reward, dict) else {}
    total_usage = {
        "input": int(result.get("inputTokens", 0) or 0),
        "output": int(result.get("outputTokens", 0) or 0),
        "thought": int(result.get("reasoningOutputTokens", 0) or 0),
        "cachedRead": int(result.get("cachedInputTokens", 0) or 0),
        "cachedWrite": int(result.get("cacheWriteTokens", 0) or 0),
    }
    return {
        "archive": str(path),
        "task": result.get("taskId"),
        "mode": mode,
        "reward": reward.get("reward", nested_reward.get("reward")),
        "partial": reward.get("partial", nested_reward.get("partial")),
        "stop_reason": result.get("stopReason"),
        "windows": rendered_windows,
        "window_count": len(windows),
        "candidate_usage_coverage": len(set(usage_rows) & set(windows)) / len(windows) if windows else None,
        "candidate_usage": _rollup(candidate_usage),
        "total_usage": _rollup(total_usage),
        "lane_steps": lane_steps,
        "joint_candidate_step_counts": dict(sorted(joint.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "tournament_eligible_windows": tournament_eligible,
    }


def _distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {key: value / total for key, value in sorted(counter.items())} if total else {}


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_usage = _zero_usage()
    total_usage = _zero_usage()
    joint = Counter()
    kinds = Counter()
    windows = lane_steps = eligible = covered = 0
    for run in runs:
        _add_usage(candidate_usage, run["candidate_usage"])
        _add_usage(total_usage, run["total_usage"])
        joint.update(run["joint_candidate_step_counts"])
        kinds.update(run["kind_counts"])
        windows += run["window_count"]
        lane_steps += run["lane_steps"]
        eligible += run["tournament_eligible_windows"]
        covered += sum(row["candidate_usage_captured"] for row in run["windows"])
    candidate = _rollup(candidate_usage)
    total = _rollup(total_usage)
    return {
        "runs": len(runs),
        "successes": sum(run["reward"] == 1 for run in runs),
        "timeouts": sum(run["stop_reason"] == "TIMEOUT" for run in runs),
        "stop_reasons": dict(sorted(Counter(run["stop_reason"] for run in runs).items())),
        "windows": windows,
        "lane_steps": lane_steps,
        "candidate_usage_coverage": covered / windows if windows else None,
        "candidate_usage": candidate,
        "candidate_raw_input_per_window": candidate["rawInput"] / windows if windows else None,
        "candidate_raw_input_per_lane_step": (
            candidate["rawInput"] / lane_steps if lane_steps else None
        ),
        "total_usage": total,
        "total_raw_input_per_run": total["rawInput"] / len(runs) if runs else None,
        "joint_candidate_step_distribution": _distribution(joint),
        "kind_counts": dict(sorted(kinds.items())),
        "tournament_eligible_windows": eligible,
        "tournament_eligible_fraction": eligible / windows if windows else None,
    }


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    modes = {
        mode: _aggregate([run for run in runs if run["mode"] == mode])
        for mode in ("dynamic", "explicit-probe")
        if any(run["mode"] == mode for run in runs)
    }
    comparison = None
    if set(modes) == {"dynamic", "explicit-probe"}:
        dynamic = modes["dynamic"]
        explicit = modes["explicit-probe"]
        tv = _total_variation(
            dynamic["joint_candidate_step_distribution"],
            explicit["joint_candidate_step_distribution"],
        )
        dynamic_lane_steps = dynamic["lane_steps"] / dynamic["windows"] if dynamic["windows"] else None
        explicit_lane_steps = explicit["lane_steps"] / explicit["windows"] if explicit["windows"] else None
        lane_step_delta = (
            explicit_lane_steps / dynamic_lane_steps - 1
            if dynamic_lane_steps and explicit_lane_steps is not None
            else None
        )
        comparison = {
            "joint_distribution_total_variation": tv,
            "mean_lane_steps_per_window_delta_fraction": lane_step_delta,
            "behaviorally_same_by_prespecified_tolerance": (
                tv <= 0.05
                and lane_step_delta is not None
                and abs(lane_step_delta) <= 0.05
            ),
            "candidate_raw_input_per_lane_step_reduction_fraction": (
                1
                - explicit["candidate_raw_input_per_lane_step"]
                / dynamic["candidate_raw_input_per_lane_step"]
                if dynamic["candidate_raw_input_per_lane_step"]
                and explicit["candidate_raw_input_per_lane_step"] is not None
                else None
            ),
            "total_raw_input_per_run_reduction_fraction": (
                1
                - explicit["total_raw_input_per_run"] / dynamic["total_raw_input_per_run"]
                if dynamic["total_raw_input_per_run"]
                and explicit["total_raw_input_per_run"] is not None
                else None
            ),
        }
    paired: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in sorted({run["task"] for run in runs if run["task"]}):
        paired[task] = {}
        for mode in ("dynamic", "explicit-probe"):
            attempts = [run for run in runs if run["task"] == task and run["mode"] == mode]
            if attempts:
                paired[task][mode] = [
                    {
                        "archive": run["archive"],
                        "reward": run["reward"],
                        "partial": run["partial"],
                        "stop_reason": run["stop_reason"],
                    }
                    for run in attempts
                ]
    return {"runs": runs, "modes": modes, "comparison": comparison, "paired_outcomes": paired}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="+", type=Path, help="archive or directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        inputs = discover_inputs(args.path)
        if not inputs:
            parser.error("no archives found")
        report = summarize([extract_run(path) for path in inputs])
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
