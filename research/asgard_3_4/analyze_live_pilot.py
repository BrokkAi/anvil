#!/usr/bin/env python3
"""Aggregate paired Asgard prompt-mode pilot archives."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


USAGE_KEYS = ("input", "output", "thought", "cachedRead", "cachedWrite")


def discover_archives(
    paths: Iterable[Path], *, skip_incomplete: bool = False
) -> list[Path]:
    archives: set[Path] = set()
    for path in paths:
        if path.is_dir():
            archives.update(path.rglob("*.zip"))
        elif path.suffix == ".zip":
            archives.add(path)
        else:
            raise ValueError(f"not an archive or directory: {path}")
    discovered = sorted(archives)
    if not skip_incomplete:
        return discovered
    captured: list[Path] = []
    for path in discovered:
        try:
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
        except zipfile.BadZipFile:
            continue
        if {"anvil-trace.jsonl", "result.json"}.issubset(members):
            captured.append(path)
    return captured


def _json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name))
    except KeyError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} in {archive.filename} is not an object")
    return value


def _trace_rows(archive: zipfile.ZipFile) -> Iterable[dict[str, Any]]:
    try:
        raw = archive.read("anvil-trace.jsonl").decode("utf-8", "replace")
    except KeyError as error:
        raise ValueError(f"{archive.filename} has no anvil-trace.jsonl") from error
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
            yield value


def _zero_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def _add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in USAGE_KEYS:
        value = usage.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] += value


def _usage_rollup(usage: dict[str, int]) -> dict[str, float | int | None]:
    uncached = usage["input"]
    cached = usage["cachedRead"]
    cache_write = usage["cachedWrite"]
    raw_input = uncached + cached + cache_write
    cache_eligible_input = uncached + cached
    return {
        **usage,
        "rawInput": raw_input,
        "uncachedFraction": (
            uncached / cache_eligible_input if cache_eligible_input else None
        ),
        "cachedReadFraction": (
            cached / cache_eligible_input if cache_eligible_input else None
        ),
    }


def _usage_per(usage: dict[str, int], count: int) -> dict[str, float | None]:
    rolled = _usage_rollup(usage)
    return {
        key: (float(value) / count if count else None)
        for key, value in rolled.items()
        if key in (*USAGE_KEYS, "rawInput")
    }


def extract_run(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        result = _json_member(archive, "result.json")
        reward = _json_member(archive, "reward.json")
        rows = list(_trace_rows(archive))

    prompt_records: list[dict[str, Any]] = []
    response_usage = {
        "supervisor": _zero_usage(),
        "completion_review": _zero_usage(),
    }
    response_calls = Counter()
    decisions = Counter()
    windows: list[dict[str, Any]] = []
    pending_request: dict[str, Any] | None = None
    for row in rows:
        record_type = row.get("type")
        if record_type == "asgard_supervisor_prompt_mode":
            prompt_records.append(row)
        elif record_type == "asgard_supervisor_replay_request":
            pending_request = row
        elif record_type == "asgard_supervisor_replay_response":
            if pending_request is None:
                continue
            if row.get("call_index") != pending_request.get("call_index"):
                raise ValueError(
                    f"{path}: replay response call_index {row.get('call_index')} "
                    f"does not match pending request {pending_request.get('call_index')}"
                )
            decision_call = pending_request.get("decision_call")
            if decision_call in response_usage:
                _add_usage(response_usage[decision_call], row.get("usage") or {})
                response_calls[decision_call] += 1
            pending_request = None
        elif record_type == "asgard_decision":
            decisions[str(row.get("call"))] += 1
        elif record_type == "asgard_window":
            windows.append(row)

    modes = {str(row.get("mode")) for row in prompt_records}
    if len(modes) != 1:
        raise ValueError(f"{path}: expected one captured mode, got {sorted(modes)}")
    mode = next(iter(modes))
    chosen_bytes = sum(int(row.get("prompt_bytes", 0)) for row in prompt_records)
    control_bytes = sum(
        int(row.get("full_control_prompt_bytes", 0)) for row in prompt_records
    )
    chosen_tokens = sum(
        int(row.get("estimated_request_tokens", 0)) for row in prompt_records
    )
    control_tokens = sum(
        int(row.get("full_control_estimated_request_tokens", 0))
        for row in prompt_records
    )
    return {
        "archive": str(path),
        "task": result.get("taskId"),
        "mode": mode,
        "reward": reward.get("reward", (result.get("reward") or {}).get("reward")),
        "partial": reward.get("partial", (result.get("reward") or {}).get("partial")),
        "stop_reason": result.get("stopReason"),
        "total_usage": {
            "input": result.get("inputTokens", 0),
            "cachedRead": result.get("cachedInputTokens", 0),
            "cachedWrite": result.get("cacheWriteTokens", 0),
            "output": result.get("outputTokens", 0),
            "thought": result.get("reasoningOutputTokens", 0),
        },
        "ordinary_routing": {
            "windows": len(prompt_records),
            "decisions": decisions.get("supervisor", 0),
            "model_calls": response_calls.get("supervisor", 0),
            "usage": _usage_rollup(response_usage["supervisor"]),
            "prompt_bytes": chosen_bytes,
            "full_control_prompt_bytes": control_bytes,
            "prompt_byte_reduction_fraction": (
                1 - chosen_bytes / control_bytes if control_bytes else None
            ),
            "estimated_request_tokens": chosen_tokens,
            "full_control_estimated_request_tokens": control_tokens,
            "estimated_token_reduction_fraction": (
                1 - chosen_tokens / control_tokens if control_tokens else None
            ),
        },
        "completion_review": {
            "decisions": decisions.get("completion_review", 0),
            "model_calls": response_calls.get("completion_review", 0),
            "usage": _usage_rollup(response_usage["completion_review"]),
        },
        "window_policy": {
            "candidate_counts": dict(
                Counter(str(row.get("candidate_count")) for row in windows)
            ),
            "step_counts": dict(
                Counter(str(row.get("window_steps")) for row in windows)
            ),
        },
    }


def _sum_usage(runs: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, int]:
    total = _zero_usage()
    for run in runs:
        value: Any = run
        for part in path:
            value = value[part]
        _add_usage(total, value)
    return total


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_mode[run["mode"]].append(run)
    mode_rows: dict[str, Any] = {}
    for mode, mode_runs in sorted(by_mode.items()):
        routing = [run["ordinary_routing"] for run in mode_runs]
        chosen_bytes = sum(row["prompt_bytes"] for row in routing)
        control_bytes = sum(row["full_control_prompt_bytes"] for row in routing)
        chosen_tokens = sum(row["estimated_request_tokens"] for row in routing)
        control_tokens = sum(
            row["full_control_estimated_request_tokens"] for row in routing
        )
        ordinary_usage = _sum_usage(mode_runs, ("ordinary_routing", "usage"))
        completion_usage = _sum_usage(mode_runs, ("completion_review", "usage"))
        total_usage = _sum_usage(mode_runs, ("total_usage",))
        ordinary_windows = sum(row["windows"] for row in routing)
        ordinary_model_calls = sum(row["model_calls"] for row in routing)
        mode_rows[mode] = {
            "runs": len(mode_runs),
            "successes": sum(run["reward"] == 1 for run in mode_runs),
            "mean_partial": (
                sum(float(run["partial"] or 0) for run in mode_runs) / len(mode_runs)
            ),
            "ordinary_windows": ordinary_windows,
            "ordinary_model_calls": ordinary_model_calls,
            "ordinary_usage": _usage_rollup(ordinary_usage),
            "ordinary_usage_per_window": _usage_per(
                ordinary_usage, ordinary_windows
            ),
            "ordinary_usage_per_model_call": _usage_per(
                ordinary_usage, ordinary_model_calls
            ),
            "completion_review_usage": _usage_rollup(completion_usage),
            "completion_review_calls": sum(
                run["completion_review"]["model_calls"] for run in mode_runs
            ),
            "total_usage": _usage_rollup(total_usage),
            "total_usage_per_run": _usage_per(total_usage, len(mode_runs)),
            "prompt_bytes": chosen_bytes,
            "full_control_prompt_bytes": control_bytes,
            "prompt_byte_reduction_fraction": (
                1 - chosen_bytes / control_bytes if control_bytes else None
            ),
            "estimated_request_tokens": chosen_tokens,
            "full_control_estimated_request_tokens": control_tokens,
            "estimated_token_reduction_fraction": (
                1 - chosen_tokens / control_tokens if control_tokens else None
            ),
            "candidate_counts": dict(
                sum(
                    (
                        Counter(run["window_policy"]["candidate_counts"])
                        for run in mode_runs
                    ),
                    Counter(),
                )
            ),
            "step_counts": dict(
                sum(
                    (
                        Counter(run["window_policy"]["step_counts"])
                        for run in mode_runs
                    ),
                    Counter(),
                )
            ),
        }
    tasks = sorted({str(run["task"]) for run in runs})
    paired: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in tasks:
        paired[task] = {}
        for mode in sorted(by_mode):
            attempts = [
                run for run in runs if run["task"] == task and run["mode"] == mode
            ]
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
    return {"runs": runs, "modes": mode_rows, "paired_task_outcomes": paired}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="+", type=Path, help="archive or archive directory")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="ignore cancellation/corrupt ZIPs without a captured trace and result",
    )
    args = parser.parse_args()
    try:
        archives = discover_archives(args.path, skip_incomplete=args.skip_incomplete)
        if not archives:
            parser.error("no zip archives found")
        report = summarize([extract_run(path) for path in archives])
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
