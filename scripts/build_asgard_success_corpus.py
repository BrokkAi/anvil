#!/usr/bin/env python3
"""Build the checked-in Asgard historical-success corpus from cohort archives.

The corpus is evaluation memory, not prompt memory: it records every successful
task-run and its successful trajectories without exposing historical solutions
to a new Asgard run.  The companion obligations manifest is suitable for the
known-success gate and deterministic handoff/replay audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable


CORPUS_SCHEMA = "asgard-historical-success-corpus/v1"
OBLIGATIONS_SCHEMA = "asgard-known-success-obligations/v1"


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def load_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if isinstance(value, dict):
                yield value


def successful_runs(path: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for event in load_json_lines(path):
        if event.get("kind") != "completed":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or data.get("outcome") != "SUCCESS":
            continue
        archive = data.get("archive")
        task = event.get("project")
        run_number = event.get("runNumber")
        revision = event.get("revision")
        if not isinstance(archive, str) or not isinstance(task, str):
            raise ValueError(f"malformed successful completion in {path}: {event!r}")
        if not isinstance(run_number, int) or not isinstance(revision, str):
            raise ValueError(f"malformed task identity in {path}: {event!r}")
        runs.append(
            {
                "task_id": task,
                "run_number": run_number,
                "revision": revision,
                "archive": Path(archive),
            }
        )
    return runs


def zip_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        with archive.open(name) as stream:
            value = json.load(stream)
    except KeyError as error:
        raise ValueError(f"{archive.filename}: missing {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{archive.filename}:{name}: expected JSON object")
    return value


def zip_member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    try:
        with archive.open(name) as stream:
            return sha256_stream(stream)
    except KeyError as error:
        raise ValueError(f"{archive.filename}: missing {name}") from error


def trace_evidence(
    archive: zipfile.ZipFile,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, int]]:
    checklist: list[dict[str, Any]] = []
    final_supervisor: dict[str, Any] | None = None
    final_review: dict[str, Any] | None = None
    counts: Counter[str] = Counter()
    try:
        trace = archive.open("anvil-trace.jsonl")
    except KeyError as error:
        raise ValueError(f"{archive.filename}: missing anvil-trace.jsonl") from error
    with trace:
        for line_number, raw_line in enumerate(trace, 1):
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{archive.filename}:anvil-trace.jsonl:{line_number}: {error}"
                ) from error
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if isinstance(event_type, str):
                counts[event_type] += 1
            if event_type == "asgard_checklist" and isinstance(event.get("contracts"), list):
                checklist = [
                    row for row in event["contracts"] if isinstance(row, dict)
                ]
            if event_type != "asgard_decision":
                continue
            decision = event.get("decision")
            if not isinstance(decision, dict):
                continue
            compact = {
                "complete": decision.get("complete"),
                "winner": decision.get("winner"),
                "state_summary": decision.get("state_summary"),
                "contracts": decision.get("contracts"),
            }
            if event.get("call") == "supervisor":
                final_supervisor = compact
            elif event.get("call") == "completion_review":
                final_review = compact
    return checklist, final_supervisor, final_review, dict(sorted(counts.items()))


def enriched_contracts(
    checklist: list[dict[str, Any]], review: dict[str, Any] | None
) -> list[dict[str, Any]]:
    by_id = {
        row.get("id"): row
        for row in checklist
        if isinstance(row.get("id"), str)
    }
    review_rows = review.get("contracts") if review is not None else None
    if not isinstance(review_rows, list):
        return checklist
    enriched: list[dict[str, Any]] = []
    for row in review_rows:
        if not isinstance(row, dict):
            continue
        merged = dict(by_id.get(row.get("id"), {}))
        merged.update(row)
        enriched.append(merged)
    return enriched or checklist


def exemplar(
    *, cohort: str, source_log: Path, run: dict[str, Any]
) -> dict[str, Any]:
    archive_path = run["archive"].expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        result = zip_json(archive, "result.json")
        checklist, final_supervisor, final_review, trace_counts = trace_evidence(archive)
        contracts = enriched_contracts(checklist, final_review)
        return {
            "task_id": run["task_id"],
            "run_number": run["run_number"],
            "outcome": "SUCCESS",
            "cohort": cohort,
            "source_log": str(source_log.resolve()),
            "revision": run["revision"],
            "archive": str(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_path(archive_path),
            "anvil_sha256": result.get("anvilSha256"),
            "mjolnir_sha256": result.get("mjolnirSha256"),
            "guidance_fingerprint": result.get("guidanceFingerprint"),
            "model_wire_id": result.get("modelWireId"),
            "model_patch_sha256": zip_member_sha256(archive, "model.patch"),
            "instruction_sha256": zip_member_sha256(archive, "instruction.md"),
            "trace_event_counts": trace_counts,
            "checklist_contracts": checklist,
            "terminal_supervisor": final_supervisor,
            "terminal_completion_review": (
                None
                if final_review is None
                else {
                    **final_review,
                    "contracts": contracts,
                }
            ),
        }


def stable_obligations(exemplars: list[dict[str, Any]]) -> dict[str, list[str]]:
    obligations: dict[str, set[str]] = defaultdict(set)
    for item in exemplars:
        task = item["task_id"]
        review = item.get("terminal_completion_review")
        rows = review.get("contracts") if isinstance(review, dict) else None
        if not isinstance(rows, list):
            rows = item.get("checklist_contracts", [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                obligations[task].add(text.strip())
    return {task: sorted(values) for task, values in sorted(obligations.items())}


def parse_cohort(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("cohort must be LABEL=JSONL_PATH")
    return label.strip(), Path(path).expanduser()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", type=parse_cohort, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--obligations", type=Path, required=True)
    args = parser.parse_args()

    cohort_metadata: list[dict[str, Any]] = []
    exemplars: list[dict[str, Any]] = []
    for label, source_log in args.cohort:
        source_log = source_log.resolve()
        runs = successful_runs(source_log)
        cohort_metadata.append(
            {
                "label": label,
                "source_log": str(source_log),
                "source_log_sha256": sha256_path(source_log),
                "successful_task_runs": len(runs),
            }
        )
        for run in runs:
            exemplars.append(exemplar(cohort=label, source_log=source_log, run=run))

    exemplars.sort(key=lambda row: (row["task_id"], row["run_number"], row["cohort"]))
    by_task_run: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, item in enumerate(exemplars):
        by_task_run[(item["task_id"], item["run_number"])].append(index)
    protected_task_runs = [
        {
            "task_id": task,
            "run_number": run_number,
            "exemplar_indices": indices,
        }
        for (task, run_number), indices in sorted(by_task_run.items())
    ]
    protected_tasks = sorted({task for task, _run_number in by_task_run})
    corpus = {
        "schema_version": CORPUS_SCHEMA,
        "description": (
            "Union of every task-run solved by Asgard v6 or v9. Historical patches "
            "and review evidence are evaluation-only and must not enter rollout prompts."
        ),
        "cohorts": cohort_metadata,
        "protected_task_count": len(protected_tasks),
        "protected_task_run_count": len(protected_task_runs),
        "successful_exemplar_count": len(exemplars),
        "protected_tasks": protected_tasks,
        "protected_task_runs": protected_task_runs,
        "records": exemplars,
    }
    obligations = {
        "schema_version": OBLIGATIONS_SCHEMA,
        "description": (
            "Deterministically recovered task contracts from historically successful "
            "trajectories; used for handoff/replay audits, never as solver prompt input."
        ),
        "tasks": stable_obligations(exemplars),
    }
    if len(protected_tasks) != 11 or len(protected_task_runs) != 15 or len(exemplars) != 18:
        raise ValueError(
            "unexpected historical union: "
            f"{len(protected_tasks)} tasks, {len(protected_task_runs)} task-runs, "
            f"{len(exemplars)} exemplars"
        )
    write_json(args.corpus, corpus)
    write_json(args.obligations, obligations)
    print(
        f"wrote {len(exemplars)} exemplars covering "
        f"{len(protected_task_runs)} task-runs / {len(protected_tasks)} tasks"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
