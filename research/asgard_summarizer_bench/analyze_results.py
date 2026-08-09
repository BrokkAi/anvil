#!/usr/bin/env python3
"""Compute paired structural and weak-reference metrics for the summarizer probe."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path


LITERAL_RE = re.compile(
    r"`([^`]{3,120})`|\b([A-Za-z_][A-Za-z0-9_.:/-]{3,80}(?:\.[A-Za-z0-9]+|_[A-Za-z0-9_]+|/[A-Za-z0-9_.-]+))\b"
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def salient_literals(text: str) -> set[str]:
    values = set()
    for match in LITERAL_RE.finditer(text):
        value = (match.group(1) or match.group(2)).strip("`'\".,:;()[]{}")
        if len(value) >= 4 and not value.isdigit():
            values.add(value.lower())
    return values


def reference_recall(reference: str | None, output: str) -> float | None:
    if not reference:
        return None
    literals = salient_literals(reference)
    if not literals:
        return None
    lowered = output.lower()
    return sum(value in lowered for value in literals) / len(literals)


def location_grounding(summary: dict, source: str) -> tuple[int, int]:
    grounded = 0
    total = 0
    lowered = source.lower()
    for edit in summary.get("edits", []):
        location = edit.get("location", "")
        candidates = {
            token.strip("`'\".,:;()[]{}")
            for token in re.split(r"[\s,]+", location.lower())
            if len(token.strip("`'\".,:;()[]{}")) >= 4
        }
        if not candidates:
            continue
        total += 1
        grounded += any(candidate in lowered for candidate in candidates)
    return grounded, total


def model_metrics(rows: list[dict], dataset: dict[str, dict]) -> dict:
    valid = [row for row in rows if not row.get("error") and row.get("summary")]
    latencies = [row["latency_seconds"] for row in valid]
    prompt_tokens = [row.get("usage", {}).get("prompt_tokens", 0) for row in valid]
    completion_tokens = [row.get("usage", {}).get("completion_tokens", 0) for row in valid]
    compression = []
    recalls = []
    grounded = total_locations = 0
    counts = {"edits": [], "evidence": [], "risks": []}
    evidence_statuses: collections.Counter[str] = collections.Counter()
    completion_like_directions = 0
    per_id_recall: dict[str, float] = {}
    for row in valid:
        source = dataset[row["id"]]
        output = json.dumps(row["summary"], sort_keys=True)
        compression.append(len(output.encode()) / max(source["window_bytes"], 1))
        recall = reference_recall(source.get("selected_lane_reference"), output)
        if recall is not None:
            recalls.append(recall)
            per_id_recall[row["id"]] = recall
        good, count = location_grounding(row["summary"], source["window_text"])
        grounded += good
        total_locations += count
        counts["edits"].append(len(row["summary"].get("edits", [])))
        counts["evidence"].append(len(row["summary"].get("evidence", [])))
        counts["risks"].append(len(row["summary"].get("unresolved_risks", [])))
        evidence_statuses.update(
            item.get("status", "missing") for item in row["summary"].get("evidence", [])
        )
        completion_like_directions += bool(
            re.search(r"complet|done|finish", row["summary"].get("direction", ""), re.IGNORECASE)
        )
    return {
        "rows": len(rows),
        "valid": len(valid),
        "errors": sum(bool(row.get("error")) for row in rows),
        "schema_invalid": sum(bool(row.get("validation_error")) for row in rows),
        "latency_seconds": {
            "median": statistics.median(latencies) if latencies else None,
            "p90": percentile(latencies, 0.9),
            "max": max(latencies, default=None),
        },
        "tokens": {
            "prompt_total": sum(prompt_tokens),
            "completion_total": sum(completion_tokens),
            "completion_median": statistics.median(completion_tokens) if completion_tokens else None,
            "completion_p90": percentile(completion_tokens, 0.9),
            "completion_max": max(completion_tokens, default=None),
        },
        "finish_reasons": dict(sorted(collections.Counter(row.get("finish_reason") for row in valid).items())),
        "compression_ratio": {
            "median": statistics.median(compression) if compression else None,
            "p90": percentile(compression, 0.9),
        },
        "selected_lane_reference_literal_recall": {
            "n": len(recalls),
            "mean": statistics.mean(recalls) if recalls else None,
            "median": statistics.median(recalls) if recalls else None,
        },
        "edit_location_grounding": {
            "grounded": grounded,
            "total": total_locations,
            "rate": grounded / total_locations if total_locations else None,
        },
        "summary_items_median": {
            key: statistics.median(values) if values else None for key, values in counts.items()
        },
        "edit_presence": {
            "summaries_with_edits": sum(value > 0 for value in counts["edits"]),
            "rate": sum(value > 0 for value in counts["edits"]) / len(valid) if valid else None,
            "total_items": sum(counts["edits"]),
        },
        "evidence_statuses": dict(sorted(evidence_statuses.items())),
        "completion_like_directions": completion_like_directions,
        "_per_id_recall": per_id_recall,
    }


def stratified_metrics(rows: list[dict], dataset: dict[str, dict]) -> dict:
    result = {"all": model_metrics(rows, dataset)}
    classes = sorted({dataset[row["id"]].get("activity_class", "unclassified") for row in rows})
    for activity_class in classes:
        subset = [
            row
            for row in rows
            if dataset[row["id"]].get("activity_class", "unclassified") == activity_class
        ]
        result[activity_class] = model_metrics(subset, dataset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = {
        row["id"]: row
        for path in args.dataset
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }
    report: dict[str, dict] = {}
    for path in args.result:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Retry runs append a newer record for the same id.
        rows = list({row["id"]: row for row in rows}.values())
        rows = [row for row in rows if row["id"] in dataset]
        model = rows[0]["model"] if rows else path.stem
        report[model] = stratified_metrics(rows, dataset)

    if len(report) == 2:
        names = list(report)
        left = report[names[0]]["all"]["_per_id_recall"]
        right = report[names[1]]["all"]["_per_id_recall"]
        common = sorted(set(left) & set(right))
        report["paired_reference_recall"] = {
            "n": len(common),
            f"{names[0]}_higher": sum(left[key] > right[key] for key in common),
            f"{names[1]}_higher": sum(right[key] > left[key] for key in common),
            "ties": sum(math.isclose(left[key], right[key]) for key in common),
        }
    for metrics in report.values():
        if isinstance(metrics, dict):
            metrics.pop("_per_id_recall", None)
            for stratum in metrics.values():
                if isinstance(stratum, dict):
                    stratum.pop("_per_id_recall", None)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
