#!/usr/bin/env python3
"""Plan same-state compact-supervisor replay batches without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import render_replay_prompts as prompts


TARGET_MODES = (
    "checkpoint-plus-delta",
    "latest-state",
    "decision-log-only",
    "recent-exact-tail",
)
RUN_RE = re.compile(r"-r([1-9][0-9]*)-")
LANE_RE = re.compile(r'<lane_trajectory\s+index="(\d+)"')


def discover_archives(paths: Iterable[Path]) -> list[Path]:
    archives: set[Path] = set()
    for path in paths:
        if path.is_dir():
            archives.update(item for item in path.rglob("*.zip") if item.is_file())
        elif path.is_file() and path.suffix == ".zip":
            archives.add(path)
        else:
            raise ValueError(f"not an archive or directory: {path}")
    return sorted(archives)


def load_protected_identities(path: Path | None) -> set[str]:
    if path is None:
        return set()
    identities: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: row is not an object")
        listed = row.get("protected_identities")
        if isinstance(listed, list):
            identities.update(item for item in listed if isinstance(item, str))
        identity = row.get("protected_identity")
        if isinstance(identity, str):
            identities.add(identity)
    return identities


def _read_archive(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with zipfile.ZipFile(path) as archive:
            result = json.loads(archive.read("result.json"))
            raw = archive.read("anvil-trace.jsonl").decode("utf-8", "replace")
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{path}: result.json is not an object")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:anvil-trace.jsonl:{number}: {error.msg}") from error
        if isinstance(row, dict):
            rows.append(row)
    return result, rows


def _run(path: Path) -> int:
    match = RUN_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer run from archive name: {path.name}")
    return int(match.group(1))


def _post_review_windows(rows: list[dict[str, Any]]) -> set[int]:
    post_review: set[int] = set()
    review_seen = False
    for row in rows:
        if row.get("type") == "asgard_decision" and row.get("call") == "completion_review":
            review_seen = True
        elif row.get("type") == "asgard_supervisor_prompt_mode":
            window = row.get("window")
            if review_seen and isinstance(window, int):
                post_review.add(window)
            review_seen = False
    return post_review


def _candidate_count(record: dict[str, Any], windows: dict[int, dict[str, Any]]) -> tuple[int | None, str]:
    window = record["prompt"].get("window")
    traced = windows.get(window)
    if traced and isinstance(traced.get("candidate_count"), int):
        return traced["candidate_count"], "trace.asgard_window"
    dossier = record["state"].get("candidate_trajectories")
    if isinstance(dossier, str):
        lanes = {int(value) for value in LANE_RE.findall(dossier)}
        if lanes:
            return len(lanes), "captured candidate_trajectories lane tags"
    return None, "unknown"


def _history_count(record: dict[str, Any]) -> int:
    history = record["state"].get("supervisor_history") or {}
    entries = list(history.get("checkpointed") or []) + list(history.get("selected_windows") or [])
    windows = [entry.get("window") for entry in entries if isinstance(entry, dict)]
    return len({value for value in windows if isinstance(value, int)})


def _message_bytes(messages: list[dict[str, Any]]) -> int:
    return len(prompts.render_dossier_messages(messages).encode("utf-8"))


def _tool_bytes(record: dict[str, Any]) -> int:
    request = record["request"]
    payload = {
        "tools": request.get("tools") or [],
        "parameters": request.get("parameters") or {},
    }
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _estimate_mode(record: dict[str, Any], mode: str) -> dict[str, Any]:
    messages = prompts.render_mode_messages(record, mode)
    prompt_bytes = _message_bytes(messages)
    full_bytes = _message_bytes(prompts.full_control_messages(record))
    captured_full_tokens = record["prompt"].get("full_control_estimated_request_tokens")
    ratio_estimate = 0
    if isinstance(captured_full_tokens, int) and captured_full_tokens >= 0 and full_bytes:
        ratio_estimate = math.ceil(prompt_bytes * captured_full_tokens / full_bytes)
    # One token per three UTF-8 bytes is deliberately more conservative than
    # the common bytes/4 planning heuristic. The captured full-control ratio is
    # used whenever it predicts a larger request.
    prompt_tokens = max(math.ceil(prompt_bytes / 3), ratio_estimate)
    tools_bytes = _tool_bytes(record)
    tool_tokens = math.ceil(tools_bytes / 3)
    return {
        "prompt_bytes": prompt_bytes,
        "full_control_prompt_bytes": full_bytes,
        "prompt_byte_reduction_fraction": 1 - prompt_bytes / full_bytes if full_bytes else None,
        "estimated_prompt_tokens_conservative": prompt_tokens,
        "estimated_tool_and_parameter_tokens_conservative": tool_tokens,
        "estimated_request_input_tokens_conservative": prompt_tokens + tool_tokens,
        "utf8_byte_token_ceiling": prompt_bytes + tools_bytes,
        "estimation_method": "max(captured full-control token/byte ratio, UTF-8 bytes/3), plus tools bytes/3",
    }


def extract_archive_records(
    path: Path,
    protected_identities: set[str],
    long_history_windows: int,
    modes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result, rows = _read_archive(path)
    task = result.get("taskId")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{path}: result.json has no taskId")
    run = _run(path)
    identity = f"{task}::r{run}"
    captures = [
        record
        for record in prompts.captured_routing_states(rows)
        if isinstance(record.get("control_decision"), dict)
    ]
    source_modes = {str(record["prompt"].get("mode")) for record in captures}
    if not captures:
        return [], {"archive": str(path), "status": "skipped_no_completed_capture"}
    if source_modes != {"full"}:
        return [], {
            "archive": str(path),
            "status": "skipped_non_full_capture",
            "source_modes": sorted(source_modes),
        }
    for record in captures:
        prompts.validate_captured_mode(record)
    window_rows = {
        row["window"]: row
        for row in rows
        if row.get("type") == "asgard_window" and isinstance(row.get("window"), int)
    }
    post_review = _post_review_windows(rows)
    endpoint_window = max(int(record["prompt"]["window"]) for record in captures)
    is_protected = identity in protected_identities
    extracted: list[dict[str, Any]] = []
    for record in captures:
        window = int(record["prompt"]["window"])
        candidate_count, candidate_source = _candidate_count(record, window_rows)
        history_count = _history_count(record)
        protected_endpoint = is_protected and window == endpoint_window
        extracted.append(
            {
                "record_id": f"{path.stem}::w{window}",
                "source_archive": str(path),
                "task_id": task,
                "run": run,
                "task_run_identity": identity,
                "protected_identity": identity if is_protected else None,
                "protected_endpoint": protected_endpoint,
                "window": window,
                "control_decision": {
                    field: record["control_decision"].get(field)
                    for field in (
                        "winner",
                        "complete",
                        "next_candidate_count",
                        "next_window_steps",
                    )
                },
                "strata": {
                    "candidate_count": candidate_count,
                    "candidate_count_source": candidate_source,
                    "lane_mode": (
                        "unknown" if candidate_count is None else ("one" if candidate_count == 1 else "multi")
                    ),
                    "history_windows": history_count,
                    "long_history": history_count >= long_history_windows,
                    "post_completion_review": window in post_review,
                    "protected_endpoint": protected_endpoint,
                },
                "target_estimates": {
                    mode: _estimate_mode(record, mode) for mode in modes
                },
            }
        )
    return extracted, {
        "archive": str(path),
        "status": "included_full_capture",
        "task_id": task,
        "run": run,
        "task_run_identity": identity,
        "protected": is_protected,
        "completed_ordinary_records": len(extracted),
    }


def _stratum_key(record: dict[str, Any]) -> tuple[str, bool, bool, bool]:
    strata = record["strata"]
    return (
        strata["lane_mode"],
        strata["long_history"],
        strata["post_completion_review"],
        strata["protected_endpoint"],
    )


def select_records(
    records: list[dict[str, Any]], sample_size: int | None, seed: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forced = [record for record in records if record["protected_endpoint"]]
    forced_ids = {record["record_id"] for record in forced}
    if sample_size is None or sample_size >= len(records):
        selected = list(records)
        method = "all"
    else:
        selected = list(forced)
        remaining_slots = max(0, sample_size - len(selected))
        groups: dict[tuple[str, bool, bool, bool], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["record_id"] not in forced_ids:
                groups[_stratum_key(record)].append(record)
        for group in groups.values():
            group.sort(
                key=lambda record: hashlib.sha256(
                    f"{seed}\0{record['record_id']}".encode()
                ).hexdigest()
            )
        keys = sorted(groups, key=lambda key: tuple(str(value) for value in key))
        while remaining_slots and any(groups.values()):
            for key in keys:
                if remaining_slots == 0:
                    break
                if groups[key]:
                    selected.append(groups[key].pop(0))
                    remaining_slots -= 1
        method = "forced protected endpoints plus seeded round-robin strata"
    selected.sort(key=lambda record: (record["source_archive"], record["window"]))
    missing_forced = forced_ids - {record["record_id"] for record in selected}
    if missing_forced:
        raise AssertionError(f"protected endpoints omitted: {sorted(missing_forced)}")
    return selected, {
        "method": method,
        "requested_sample_size": sample_size,
        "seed": seed,
        "available_records": len(records),
        "selected_records": len(selected),
        "forced_protected_endpoints": len(forced),
        "sample_size_exceeded_for_protected_endpoints": bool(
            sample_size is not None and len(forced) > sample_size
        ),
    }


def _strata_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lane_mode": dict(sorted(Counter(record["strata"]["lane_mode"] for record in records).items())),
        "long_history": dict(sorted(Counter(str(record["strata"]["long_history"]).lower() for record in records).items())),
        "post_completion_review": dict(
            sorted(Counter(str(record["strata"]["post_completion_review"]).lower() for record in records).items())
        ),
        "protected_endpoint": dict(
            sorted(Counter(str(record["strata"]["protected_endpoint"]).lower() for record in records).items())
        ),
        "joint": dict(
            sorted(
                Counter("|".join(str(value).lower() for value in _stratum_key(record)) for record in records).items()
            )
        ),
    }


def _mode_totals(records: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for mode in modes:
        estimates = [record["target_estimates"][mode] for record in records]
        prompt_bytes = sum(item["prompt_bytes"] for item in estimates)
        full_bytes = sum(item["full_control_prompt_bytes"] for item in estimates)
        totals[mode] = {
            "calls": len(estimates),
            "prompt_bytes": prompt_bytes,
            "full_control_prompt_bytes": full_bytes,
            "prompt_byte_reduction_fraction": 1 - prompt_bytes / full_bytes if full_bytes else None,
            "estimated_prompt_tokens_conservative": sum(
                item["estimated_prompt_tokens_conservative"] for item in estimates
            ),
            "estimated_request_input_tokens_conservative": sum(
                item["estimated_request_input_tokens_conservative"] for item in estimates
            ),
            "utf8_byte_token_ceiling": sum(item["utf8_byte_token_ceiling"] for item in estimates),
        }
    return totals


def build_plan(
    archives: list[Path],
    protected_identities: set[str],
    modes: tuple[str, ...] = TARGET_MODES,
    sample_size: int | None = None,
    seed: str = "asgard-q3-replay-v1",
    long_history_windows: int = 10,
) -> dict[str, Any]:
    if long_history_windows < 1:
        raise ValueError("long_history_windows must be positive")
    records: list[dict[str, Any]] = []
    archive_reports: list[dict[str, Any]] = []
    for archive in archives:
        extracted, report = extract_archive_records(
            archive, protected_identities, long_history_windows, modes
        )
        records.extend(extracted)
        archive_reports.append(report)
    if not records:
        raise ValueError("no completed ordinary decisions in full-control captures")
    record_ids = [record["record_id"] for record in records]
    duplicates = [item for item, count in Counter(record_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate replay record ids: {duplicates}")
    selected, selection = select_records(records, sample_size, seed)
    protected_available = {
        record["task_run_identity"] for record in records if record["protected_identity"]
    }
    protected_endpoints = [record for record in selected if record["protected_endpoint"]]
    endpoint_ids = {record["task_run_identity"] for record in protected_endpoints}
    if endpoint_ids != protected_available:
        raise AssertionError("not every captured protected endpoint was selected")
    diagnostic_ids = {record["record_id"] for record in protected_endpoints}
    overall = [record for record in selected if record["record_id"] not in diagnostic_ids]
    stages = [
        {
            "stage": 1,
            "name": "protected-endpoint-diagnostic",
            "purpose": "fail fast on historically successful task/run endpoints",
            "records": [record["record_id"] for record in protected_endpoints],
            "base_records": len(protected_endpoints),
            "calls": len(protected_endpoints) * len(modes),
            "mode_totals": _mode_totals(protected_endpoints, modes),
            "stop_condition": "stop before overall replay on any protected endpoint disagreement or unresolved evidence-obligation review",
        },
        {
            "stage": 2,
            "name": "overall-agreement",
            "purpose": "estimate overall and stratified agreement after protected diagnostics",
            "records": [record["record_id"] for record in overall],
            "base_records": len(overall),
            "calls": len(overall) * len(modes),
            "mode_totals": _mode_totals(overall, modes),
            "start_condition": "stage 1 protected diagnostics pass",
        },
    ]
    return {
        "type": "asgard_supervisor_replay_batch_plan",
        "offline_only": True,
        "target_modes": list(modes),
        "archives": archive_reports,
        "selection": selection,
        "available_strata": _strata_counts(records),
        "selected_strata": _strata_counts(selected),
        "protected_coverage": {
            "known_protected_identities": len(protected_identities),
            "captured_protected_identities": sorted(protected_available),
            "captured_protected_endpoints": len(protected_available),
            "selected_protected_endpoints": len(protected_endpoints),
            "missing_known_protected_identities": sorted(protected_identities - protected_available),
            "all_captured_protected_endpoints_selected": endpoint_ids == protected_available,
        },
        "totals": {
            "base_records": len(selected),
            "calls": len(selected) * len(modes),
            "by_mode": _mode_totals(selected, modes),
        },
        "stages": stages,
        "records": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="+", type=Path, help="full-capture archive or directory")
    parser.add_argument(
        "--protected-corpus",
        type=Path,
        default=Path(__file__).with_name("known_success_corpus.jsonl"),
    )
    parser.add_argument("--mode", action="append", choices=TARGET_MODES)
    parser.add_argument("--sample-size", type=int, help="base records; protected endpoints may exceed it")
    parser.add_argument("--seed", default="asgard-q3-replay-v1")
    parser.add_argument("--long-history-windows", type=int, default=10)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args(argv)
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be positive")
    try:
        archives = discover_archives(args.path)
        if not archives:
            raise ValueError("no zip archives found")
        protected = load_protected_identities(args.protected_corpus)
        modes = tuple(dict.fromkeys(args.mode or TARGET_MODES))
        plan = build_plan(
            archives,
            protected,
            modes=modes,
            sample_size=args.sample_size,
            seed=args.seed,
            long_history_windows=args.long_history_windows,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
