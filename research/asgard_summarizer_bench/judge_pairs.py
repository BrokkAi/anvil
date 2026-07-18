#!/usr/bin/env python3
"""Expose deterministic blinded A/B packets for human/agent judging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def datasets() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in (ROOT / "v9_read_only_100.jsonl", ROOT / "v9_edit_producing_100.jsonl"):
        result.update(load_rows(path))
    return result


def summaries() -> tuple[dict[str, dict], dict[str, dict]]:
    qwen = load_rows(ROOT / "qwen35-4b-stratified-results.jsonl")
    qwen.update(load_rows(ROOT / "qwen35-4b-read-only-guard-results.jsonl"))
    gemma = load_rows(ROOT / "gemma4-e4b-stratified-results.jsonl")
    gemma.update(load_rows(ROOT / "gemma4-e4b-read-only-guard-results.jsonl"))
    return qwen, gemma


def a_is_qwen(identity: str) -> bool:
    return int(hashlib.sha256(("blind:" + identity).encode()).hexdigest(), 16) % 2 == 0


def ordered_ids(rows: dict[str, dict]) -> list[str]:
    return sorted(rows, key=lambda identity: hashlib.sha256(("order:" + identity).encode()).hexdigest())


def packet(identity: str, rows: dict[str, dict], qwen: dict[str, dict], gemma: dict[str, dict]) -> dict:
    row = rows[identity]
    q_summary = qwen[identity].get("summary")
    g_summary = gemma[identity].get("summary")
    if q_summary is None or g_summary is None:
        raise ValueError(f"missing valid paired summary for {identity}")
    first, second = (q_summary, g_summary) if a_is_qwen(identity) else (g_summary, q_summary)
    return {
        "id": identity,
        "activity_class": row["activity_class"],
        "original_task": row["task_text"],
        "window_messages": row["window_messages"],
        "supervisor_pseudo_reference": row.get("selected_lane_reference"),
        "reference_warning": (
            "The pseudo-reference is cumulative and may describe state absent from this window. "
            "Use the captured window as ground truth."
        ),
        "summary_a": first,
        "summary_b": second,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=(0, 1, 2))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--id")
    parser.add_argument("--unblind", action="store_true")
    args = parser.parse_args()
    rows = datasets()
    qwen, gemma = summaries()

    if args.unblind:
        for identity in ordered_ids(rows):
            print(
                json.dumps(
                    {
                        "id": identity,
                        "summary_a_model": "qwen35-4b" if a_is_qwen(identity) else "gemma4-e4b",
                        "summary_b_model": "gemma4-e4b" if a_is_qwen(identity) else "qwen35-4b",
                    },
                    separators=(",", ":"),
                )
            )
        return

    if args.id:
        print(json.dumps(packet(args.id, rows, qwen, gemma), indent=2))
        return

    if args.shard is None or not args.list:
        parser.error("use --id ID, --unblind, or --shard N --list")
    for index, identity in enumerate(ordered_ids(rows)):
        if index % 3 == args.shard:
            print(f"{identity}\t{rows[identity]['activity_class']}\t{rows[identity]['window_bytes']}")


if __name__ == "__main__":
    main()
