#!/usr/bin/env python3
"""Gate an Asgard experiment on retention of every historically solved task.

Inputs are normalized JSON/JSONL records. Historical inputs form a union: a task
enters the protected set if any historical record solved it. Candidate records
are then evaluated task-by-task, not by aggregate score. By default every
candidate rerun of every protected task must succeed.

This is an evaluation gate, not runtime task memory. It deliberately does not
feed historical patches, hidden failures, or answers into Asgard prompts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK_KEYS = ("task", "task_id", "instance_id", "name")
SUCCESS_STATUSES = {"success", "succeeded", "passed", "pass", "solved"}


def records_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [record for record in value if isinstance(record, dict)]
    if isinstance(value, dict):
        for key in ("runs", "records", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [record for record in nested if isinstance(record, dict)]
        return [value]
    return []


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        return records_from_value(json.loads(text))
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.extend(records_from_value(json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        return records


def field(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def task_id(record: dict[str, Any]) -> str:
    value = field(record, TASK_KEYS)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record has no task identifier ({TASK_KEYS}): {record!r}")
    return value.strip()


def succeeded(record: dict[str, Any]) -> bool:
    for key in ("success", "solved", "passed"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
    status = field(record, ("status", "outcome", "result"))
    return isinstance(status, str) and status.strip().lower() in SUCCESS_STATUSES


def preserved_obligations(record: dict[str, Any]) -> set[str]:
    value = record.get("preserved_obligations", [])
    if not isinstance(value, list):
        raise ValueError(f"preserved_obligations must be an array: {record!r}")
    return {item for item in value if isinstance(item, str)}


def load_obligations(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("tasks"), dict):
        value = value["tasks"]
    if not isinstance(value, dict):
        raise ValueError("obligations file must be an object or contain a tasks object")
    obligations: dict[str, set[str]] = {}
    for task, items in value.items():
        if not isinstance(task, str) or not isinstance(items, list):
            raise ValueError("each obligations entry must map a task string to an array")
        obligations[task] = {item for item in items if isinstance(item, str)}
    return obligations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="append",
        type=Path,
        required=True,
        help="Historical JSON/JSONL result file; repeat to form the success union.",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--minimum-success-rate",
        type=float,
        default=1.0,
        help="Required candidate success fraction for each protected task (default: 1.0).",
    )
    parser.add_argument(
        "--obligations",
        type=Path,
        help="Optional task -> decisive evidence obligation manifest for handoff replay.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        parser.error("--minimum-success-rate must be between 0 and 1")

    historical_records = [
        record for path in args.history for record in load_records(path)
    ]
    protected = {task_id(record) for record in historical_records if succeeded(record)}
    if not protected:
        raise ValueError("historical inputs contain no successful tasks")

    candidate_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_records(args.candidate):
        candidate_by_task[task_id(record)].append(record)
    obligations = load_obligations(args.obligations)

    missing: list[str] = []
    regressed: dict[str, dict[str, Any]] = {}
    obligation_failures: dict[str, list[str]] = {}
    for task in sorted(protected):
        records = candidate_by_task.get(task, [])
        if not records:
            missing.append(task)
            continue
        successes = sum(succeeded(record) for record in records)
        success_rate = successes / len(records)
        if success_rate < args.minimum_success_rate:
            regressed[task] = {
                "successes": successes,
                "runs": len(records),
                "success_rate": success_rate,
            }
        required = obligations.get(task, set())
        if required:
            preserved = set().union(*(preserved_obligations(record) for record in records))
            omitted = sorted(required - preserved)
            if omitted:
                obligation_failures[task] = omitted

    report = {
        "protected_tasks": len(protected),
        "protected_task_ids": sorted(protected),
        "candidate_tasks": len(candidate_by_task),
        "minimum_success_rate": args.minimum_success_rate,
        "missing_tasks": missing,
        "regressed_tasks": regressed,
        "obligation_failures": obligation_failures,
        "passed": not missing and not regressed and not obligation_failures,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
