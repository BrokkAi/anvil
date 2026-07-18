#!/usr/bin/env python3
"""Extract ordinary Asgard routing windows from Anvil result archives."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


SCHEMA_VERSION = 1
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
DOSSIER_MESSAGE = "assembled Asgard supervisor dossier"
DOSSIER_FIELDS = (
    "window",
    "selected_initial_bytes",
    "selected_windows_bytes",
    "candidate_trajectories_bytes",
)
USAGE_FIELDS = (
    "inputTokens",
    "cachedInputTokens",
    "cacheWriteTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "costUsd",
    "usageByModel",
    "llmMillis",
)


@dataclass(frozen=True)
class ArchiveInput:
    path: Path
    case_metadata: dict[str, Any]


def parse_dossier_telemetry(stderr_text: str) -> tuple[list[dict[str, int]], list[str]]:
    """Return dossier byte telemetry rows and non-fatal parse warnings."""
    rows: list[dict[str, int]] = []
    warnings: list[str] = []
    for line_number, raw_line in enumerate(stderr_text.splitlines(), 1):
        if DOSSIER_MESSAGE not in raw_line:
            continue
        line = ANSI_ESCAPE_RE.sub("", raw_line)
        values = {
            key: int(value)
            for key, value in re.findall(
                r"\b(" + "|".join(DOSSIER_FIELDS) + r")\s*=\s*(\d+)", line
            )
        }
        missing = [field for field in DOSSIER_FIELDS if field not in values]
        if missing:
            warnings.append(
                f"anvil-stderr.txt:{line_number}: dossier telemetry missing "
                + ", ".join(missing)
            )
            continue
        rows.append(values)
    return rows, warnings


def parse_trace(trace_text: str) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """Parse ordinary supervisor decisions, explicitly excluding completion review."""
    ordinary: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    warnings: list[str] = []
    for line_number, raw_line in enumerate(trace_text.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            warnings.append(f"anvil-trace.jsonl:{line_number}: {error.msg}")
            continue
        if row.get("type") != "asgard_decision":
            continue
        call = row.get("call")
        counts[str(call)] += 1
        if call == "supervisor":
            ordinary.append(row)
    return ordinary, dict(counts), warnings


def infer_candidate_count(decision: dict[str, Any] | None) -> tuple[int | None, str | None]:
    if decision is None:
        return None, None
    for field in ("next_candidate_count", "candidate_count", "next_candidates"):
        value = decision.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value, f"decision.{field}"
    advices = decision.get("advices")
    if isinstance(advices, list):
        return len(advices), "len(decision.advices)"
    return None, None


def lane_mode(candidate_count: int | None) -> str:
    if candidate_count is None:
        return "unknown"
    return "one" if candidate_count == 1 else "multi"


def _read_json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any] | None:
    try:
        value = json.loads(archive.read(name))
    except KeyError:
        return None
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _read_text_member(archive: zipfile.ZipFile, name: str) -> str:
    try:
        return archive.read(name).decode("utf-8", "replace")
    except KeyError as error:
        raise ValueError(f"archive is missing required member {name}") from error


def _result_usage(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {field: result[field] for field in USAGE_FIELDS if field in result}


def _decision_payload(trace_row: dict[str, Any] | None) -> dict[str, Any]:
    decision = trace_row.get("decision") if trace_row else None
    if not isinstance(decision, dict):
        decision = None
    candidate_count, source = infer_candidate_count(decision)
    return {
        "present": decision is not None,
        "winner": decision.get("winner") if decision else None,
        "complete": decision.get("complete") if decision else None,
        "advices": decision.get("advices") if decision else None,
        "next_window_steps": decision.get("next_window_steps") if decision else None,
        "state_summary": decision.get("state_summary") if decision else None,
        "contracts": decision.get("contracts") if decision else None,
        "candidate_count": candidate_count,
        "candidate_count_source": source,
        "lane_mode": lane_mode(candidate_count),
        "challenge": trace_row.get("challenge") if trace_row else None,
        "trace_timestamp": trace_row.get("timestamp") if trace_row else None,
        "raw": decision,
    }


def extract_archive(source: ArchiveInput) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract aligned window records plus per-archive accounting."""
    try:
        archive = zipfile.ZipFile(source.path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot open {source.path}: {error}") from error

    with archive:
        result = _read_json_member(archive, "result.json")
        reward = _read_json_member(archive, "reward.json")
        telemetry, stderr_warnings = parse_dossier_telemetry(
            _read_text_member(archive, "anvil-stderr.txt")
        )
        decisions, trace_counts, trace_warnings = parse_trace(
            _read_text_member(archive, "anvil-trace.jsonl")
        )

    archive_id = source.path.stem
    task_id = result.get("taskId") if result else None
    case_id = source.case_metadata.get("id")
    warnings = stderr_warnings + trace_warnings
    if len(telemetry) != len(decisions):
        warnings.append(
            "ordinary decision/telemetry count mismatch; aligned available decisions "
            f"by ordinal ({len(decisions)} decisions, {len(telemetry)} telemetry rows)"
        )

    records: list[dict[str, Any]] = []
    previous_history_bytes: int | None = None
    first_history_bytes: int | None = None
    for ordinal, dossier in enumerate(telemetry, 1):
        trace_row = decisions[ordinal - 1] if ordinal <= len(decisions) else None
        selected_initial = dossier["selected_initial_bytes"]
        selected_windows = dossier["selected_windows_bytes"]
        candidates = dossier["candidate_trajectories_bytes"]
        full_history = selected_initial + selected_windows
        full_dossier = full_history + candidates
        if first_history_bytes is None:
            first_history_bytes = full_history
        record = {
            "record_type": "asgard_routing_window",
            "schema_version": SCHEMA_VERSION,
            "archive_id": archive_id,
            "archive_path": str(source.path),
            "task_id": task_id,
            "case_id": case_id,
            "window": dossier["window"],
            "telemetry_ordinal": ordinal,
            "alignment": {
                "method": "ordinary_decision_ordinal_to_telemetry_ordinal",
                "status": "aligned" if trace_row else "decision_missing",
                "ordinary_decision_ordinal": ordinal if trace_row else None,
            },
            "dossier_bytes": {
                "selected_initial": selected_initial,
                "selected_windows": selected_windows,
                "candidate_trajectories": candidates,
                "full_history": full_history,
                "full_dossier_measured": full_dossier,
                "history_growth_from_previous": (
                    None
                    if previous_history_bytes is None
                    else full_history - previous_history_bytes
                ),
                "history_growth_from_first": full_history - first_history_bytes,
                # These are ceilings, not estimates of a safe compact prompt.
                "older_selected_windows_removal_ceiling": selected_windows,
                "all_history_removal_ceiling": full_history,
                "older_selected_windows_removal_ceiling_fraction": (
                    selected_windows / full_dossier if full_dossier else None
                ),
                "all_history_removal_ceiling_fraction": (
                    full_history / full_dossier if full_dossier else None
                ),
            },
            "decision": _decision_payload(trace_row),
            "result": result,
            "usage": _result_usage(result),
            "reward": reward if reward is not None else (result or {}).get("reward"),
            "case_metadata": source.case_metadata or None,
        }
        records.append(record)
        previous_history_bytes = full_history

    stats = {
        "archive_id": archive_id,
        "archive_path": str(source.path),
        "task_id": task_id,
        "case_id": case_id,
        "windows": len(telemetry),
        "ordinary_decisions": len(decisions),
        "aligned_windows": min(len(telemetry), len(decisions)),
        "windows_missing_decisions": max(0, len(telemetry) - len(decisions)),
        "unused_ordinary_decisions": max(0, len(decisions) - len(telemetry)),
        "decision_trace_counts": trace_counts,
        "warnings": warnings,
    }
    return records, stats


def summarize(records: list[dict[str, Any]], archives: list[dict[str, Any]]) -> dict[str, Any]:
    dossier_rows = [record["dossier_bytes"] for record in records]
    candidate_counts = Counter(
        str(record["decision"]["candidate_count"])
        if record["decision"]["candidate_count"] is not None
        else "unknown"
        for record in records
    )
    next_steps = Counter(
        str(record["decision"]["next_window_steps"])
        if record["decision"]["next_window_steps"] is not None
        else "unknown"
        for record in records
    )
    modes = Counter(record["decision"]["lane_mode"] for record in records)
    full_dossier = sum(row["full_dossier_measured"] for row in dossier_rows)
    selected_windows = sum(row["selected_windows"] for row in dossier_rows)
    all_history = sum(row["full_history"] for row in dossier_rows)
    candidates = sum(row["candidate_trajectories"] for row in dossier_rows)

    history_by_archive: list[dict[str, Any]] = []
    for archive in archives:
        matching = [
            record for record in records if record["archive_id"] == archive["archive_id"]
        ]
        if matching:
            first = matching[0]["dossier_bytes"]["full_history"]
            last = matching[-1]["dossier_bytes"]["full_history"]
            history_by_archive.append(
                {
                    "archive_id": archive["archive_id"],
                    "windows": len(matching),
                    "first_full_history_bytes": first,
                    "last_full_history_bytes": last,
                    "growth_bytes": last - first,
                }
            )

    return {
        "record_type": "asgard_corpus_summary",
        "schema_version": SCHEMA_VERSION,
        "archive_count": len(archives),
        "window_count": len(records),
        "alignment": {
            "aligned_windows": sum(row["aligned_windows"] for row in archives),
            "windows_missing_decisions": sum(
                row["windows_missing_decisions"] for row in archives
            ),
            "unused_ordinary_decisions": sum(
                row["unused_ordinary_decisions"] for row in archives
            ),
        },
        "q3_byte_ceiling": {
            "full_dossier_measured_bytes": full_dossier,
            "candidate_current_bytes": candidates,
            "full_history_bytes": all_history,
            "older_selected_windows_removal_ceiling_bytes": selected_windows,
            "older_selected_windows_removal_ceiling_fraction": (
                selected_windows / full_dossier if full_dossier else None
            ),
            "all_history_removal_ceiling_bytes": all_history,
            "all_history_removal_ceiling_fraction": (
                all_history / full_dossier if full_dossier else None
            ),
            "history_growth_by_archive": history_by_archive,
        },
        "q4_retrospective": {
            "candidate_count_distribution": dict(sorted(candidate_counts.items())),
            "next_window_steps_distribution": dict(sorted(next_steps.items())),
            "lane_mode_distribution": dict(sorted(modes.items())),
        },
        "archives": archives,
    }


def load_manifest(path: Path) -> list[ArchiveInput]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {path}: {error}") from error
    cases = value.get("cases") if isinstance(value, dict) else value
    if not isinstance(cases, list):
        raise ValueError("manifest must be a JSON list or an object with a 'cases' list")
    sources: list[ArchiveInput] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("archive"), str):
            raise ValueError(f"manifest case {index} must have a string 'archive' field")
        sources.append(ArchiveInput(Path(case["archive"]), dict(case)))
    return sources


def collect_sources(manifest: Path | None, archives: Iterable[str]) -> list[ArchiveInput]:
    sources = load_manifest(manifest) if manifest else []
    sources.extend(ArchiveInput(Path(path), {}) for path in archives)
    if not sources:
        raise ValueError("provide at least one archive or --manifest")
    seen: set[Path] = set()
    unique: list[ArchiveInput] = []
    for source in sources:
        key = source.path.expanduser().resolve()
        if key not in seen:
            seen.add(key)
            unique.append(ArchiveInput(key, source.case_metadata))
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract aligned Asgard ordinary-routing windows and Q3/Q4 summary "
            "metrics from Anvil result zip archives."
        )
    )
    parser.add_argument(
        "archives",
        nargs="*",
        metavar="ARCHIVE",
        help="Anvil result zip archive (repeatable)",
    )
    parser.add_argument(
        "-m",
        "--manifest",
        type=Path,
        help="JSON list, or object with a cases list, whose entries contain archive paths",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write JSONL here instead of stdout",
    )
    parser.add_argument(
        "--records-only",
        action="store_true",
        help="omit the final asgard_corpus_summary JSONL record",
    )
    parser.add_argument(
        "--require-decisions",
        action="store_true",
        help="fail if any dossier telemetry row lacks an ordinary supervisor decision",
    )
    return parser


def write_jsonl(rows: Iterable[dict[str, Any]], output: TextIO) -> None:
    for row in rows:
        output.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
        output.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        sources = collect_sources(args.manifest, args.archives)
        all_records: list[dict[str, Any]] = []
        archive_stats: list[dict[str, Any]] = []
        for source in sources:
            records, stats = extract_archive(source)
            all_records.extend(records)
            archive_stats.append(stats)
        summary = summarize(all_records, archive_stats)
        if args.require_decisions and summary["alignment"]["windows_missing_decisions"]:
            raise ValueError(
                f"{summary['alignment']['windows_missing_decisions']} telemetry rows lack "
                "ordinary supervisor decisions"
            )
        rows = all_records if args.records_only else [*all_records, summary]
        if args.output:
            with args.output.open("w", encoding="utf-8") as output:
                write_jsonl(rows, output)
        else:
            write_jsonl(rows, sys.stdout)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
