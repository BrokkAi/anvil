#!/usr/bin/env python3
"""Build the protected v6/v9 Asgard known-success trajectory corpus.

The builder reads public result metadata and agent-side traces only. It never reads
``verifier-output.txt`` or ``verifier.tar.gz`` and does not copy hidden-test output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_COHORTS = ("asgard6-claims", "asgard9-flash-flash")
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
DOSSIER_RE = re.compile(r"assembled Asgard supervisor dossier.*?\bwindow\s*=\s*(\d+)")
SUMMARY_LANE_RE = re.compile(r"summarized Asgard candidate trajectory.*?\blane\s*=\s*(\d+)")
SHELL_ARGS_RE = re.compile(
    r"executing tool run_shell_command with args:\s*(\{.*\})\s+\(sandbox="
)
ARCHIVE_RUN_RE = re.compile(r"-r(\d+)-\d+-\d+\.zip$")
WORKTREE_RE = re.compile(r"/opt/work/run/tmp/anvil-asgard-clones/[^/\s]+")
VERIFY_RE = re.compile(
    r"(?:\b(?:pytest|tox|mypy|pyright|ruff|eslint|tsc)\b|"
    r"\bpython(?:3)?\s+-m\s+(?:pytest|unittest|mypy)\b|"
    r"\b(?:cargo|go)\s+(?:test|check|clippy|build|vet)\b|"
    r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:test|check|lint|build)\b|"
    r"\b(?:mvn|gradle|make)\b[^\n]{0,80}\b(?:test|check|build)\b)",
    re.IGNORECASE,
)
DECL_RE = re.compile(
    r"^\+\s*(?:(?:export|public|private|protected|internal|abstract|final|"
    r"static|async)\s+)*(class|interface|type|enum|function|def|func|struct|"
    r"trait)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)


@dataclass(frozen=True)
class ResultInput:
    cohort: str
    result_path: Path
    result: dict[str, Any]
    task_id: str
    run: int
    archive_path: Path

    @property
    def identity(self) -> str:
        return f"{self.task_id}::r{self.run}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read result {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return value


def _run_from_result(path: Path, task_id: str, archive_path: Path) -> int:
    directory = path.parent.name
    suffix = directory.removeprefix(task_id)
    if suffix.isdigit() and int(suffix) > 0:
        return int(suffix)
    match = ARCHIVE_RUN_RE.search(archive_path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot infer task run from {path} or {archive_path.name}")


def discover_results(
    agentresults_root: Path,
    archive_root: Path,
    cohorts: Iterable[str] = DEFAULT_COHORTS,
) -> tuple[list[ResultInput], dict[str, int]]:
    """Discover exact cohort results and return successful inputs plus run counts."""
    successes: list[ResultInput] = []
    run_counts: dict[str, int] = {}
    for cohort in cohorts:
        paths = sorted(agentresults_root.glob(f"*/{cohort}-*.json"))
        run_counts[cohort] = len(paths)
        seen: set[str] = set()
        for path in paths:
            result = _read_json(path)
            task_id = result.get("taskId")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"result has no taskId: {path}")
            recorded_archive = result.get("archivePath")
            if not isinstance(recorded_archive, str) or not recorded_archive:
                raise ValueError(f"result has no archivePath: {path}")
            archive_path = Path(recorded_archive)
            if not archive_path.is_file():
                archive_path = archive_root / task_id / Path(recorded_archive).name
            run = _run_from_result(path, task_id, archive_path)
            identity = f"{task_id}::r{run}"
            if identity in seen:
                raise ValueError(f"duplicate {cohort} identity {identity}")
            seen.add(identity)
            if result.get("outcome") == "SUCCESS":
                if not archive_path.is_file():
                    raise ValueError(f"successful result archive is missing: {archive_path}")
                successes.append(
                    ResultInput(cohort, path, result, task_id, run, archive_path)
                )
    return successes, run_counts


def _iter_trace(archive: zipfile.ZipFile) -> Iterable[dict[str, Any]]:
    try:
        member = archive.open("anvil-trace.jsonl")
    except KeyError as error:
        raise ValueError("archive is missing anvil-trace.jsonl") from error
    with member:
        for line_number, raw in enumerate(member, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"anvil-trace.jsonl:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if isinstance(row, dict):
                yield row


def _read_member(archive: zipfile.ZipFile, name: str) -> str:
    try:
        return archive.read(name).decode("utf-8", "replace")
    except KeyError as error:
        raise ValueError(f"archive is missing {name}") from error


def _stderr_windows(stderr: str) -> dict[int, int]:
    """Infer candidates per window from candidate summaries before each dossier."""
    lanes: set[int] = set()
    windows: dict[int, int] = {}
    for raw in stderr.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw)
        lane_match = SUMMARY_LANE_RE.search(line)
        if lane_match:
            lanes.add(int(lane_match.group(1)))
        dossier_match = DOSSIER_RE.search(line)
        if dossier_match:
            windows[int(dossier_match.group(1))] = len(lanes)
            lanes.clear()
    return windows


def _stderr_verification_commands(stderr: str) -> list[str]:
    """Recover full commands because older tool_timing rows truncate at 120 bytes."""
    commands: set[str] = set()
    for raw in stderr.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw)
        match = SHELL_ARGS_RE.search(line)
        if not match:
            continue
        try:
            arguments = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if isinstance(command, str) and VERIFY_RE.search(command):
            commands.add(_normalise_command(command))
    return sorted(commands)


def _normalise_command(command: str) -> str:
    return WORKTREE_RE.sub("<candidate-worktree>", " ".join(command.split()))


def _patch_surface(patch: str, changed_files: list[str]) -> dict[str, Any]:
    current_file: str | None = None
    declarations: set[tuple[str, str, str]] = set()
    hunk_contexts: set[tuple[str, str]] = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if current_file is None:
            continue
        if line.startswith("@@"):
            parts = line.split("@@", 2)
            context = parts[2].strip() if len(parts) == 3 else ""
            if context:
                hunk_contexts.add((current_file, context))
            continue
        match = DECL_RE.match(line)
        if match:
            declarations.add((current_file, match.group(1), match.group(2)))
    return {
        "changed_files": sorted(changed_files),
        "added_declarations": [
            {"file": file, "kind": kind, "name": name}
            for file, kind, name in sorted(declarations)
        ],
        "patch_hunk_contexts": [
            {"file": file, "context": context}
            for file, context in sorted(hunk_contexts)
        ],
        "derivation": (
            "changed_files comes from public result metadata; declarations and hunk "
            "contexts are mechanical model.patch parsing, not semantic proof"
        ),
    }


def _decision_view(window: int, row: dict[str, Any]) -> dict[str, Any]:
    decision = row.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    advices = decision.get("advices")
    return {
        "window": window,
        "winner": decision.get("winner"),
        "complete": decision.get("complete"),
        "state_summary": decision.get("state_summary"),
        "contracts": decision.get("contracts"),
        "next_advices": advices if isinstance(advices, list) else [],
        "next_candidate_count": decision.get("next_candidate_count"),
        "next_window_steps": decision.get("next_window_steps"),
        "challenge": row.get("challenge"),
    }


def extract_success_trace(source: ResultInput, agentresults_root: Path) -> dict[str, Any]:
    """Extract one successful trajectory without reading any verifier artifact."""
    try:
        archive = zipfile.ZipFile(source.archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot open {source.archive_path}: {error}") from error

    checklist: list[dict[str, Any]] = []
    supervisor_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    explicit_windows: list[dict[str, Any]] = []
    verification: dict[str, bool | None] = {}
    with archive:
        for row in _iter_trace(archive):
            row_type = row.get("type")
            if row_type == "asgard_checklist":
                contracts = row.get("contracts")
                if isinstance(contracts, list):
                    checklist = [item for item in contracts if isinstance(item, dict)]
            elif row_type == "asgard_window":
                explicit_windows.append(row)
            elif row_type == "asgard_decision":
                if row.get("call") == "supervisor":
                    supervisor_rows.append(row)
                elif row.get("call") == "completion_review":
                    completion_rows.append(row)
            elif row_type == "tool_timing" and row.get("tool") == "run_shell_command":
                command = row.get("command")
                if isinstance(command, str) and VERIFY_RE.search(command):
                    normalised = _normalise_command(command)
                    verification[normalised] = bool(row.get("success"))
        stderr = _read_member(archive, "anvil-stderr.txt")
        stderr_counts = _stderr_windows(stderr)
        for command in _stderr_verification_commands(stderr):
            verification.setdefault(command, None)
        patch = _read_member(archive, "model.patch")

    decisions = [
        _decision_view(index, row) for index, row in enumerate(supervisor_rows, 1)
    ]
    explicit_by_window = {
        row.get("window"): row
        for row in explicit_windows
        if isinstance(row.get("window"), int)
    }
    sequence: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions, 1):
        explicit = explicit_by_window.get(index)
        if explicit:
            candidate_count = explicit.get("candidate_count")
            candidate_source = "trace.asgard_window.candidate_count"
            window_steps = explicit.get("window_steps")
            steps_source = "trace.asgard_window.window_steps"
        else:
            candidate_count = stderr_counts.get(index)
            candidate_source = (
                "stderr candidate-summary lane count" if candidate_count else "unknown"
            )
            previous = decisions[index - 2] if index > 1 else None
            window_steps = previous.get("next_window_steps") if previous else None
            steps_source = (
                "previous supervisor decision.next_window_steps"
                if window_steps is not None
                else "unknown: initial depth was not traced in this archive version"
            )
        sequence.append(
            {
                "window": index,
                "candidate_count": candidate_count,
                "candidate_count_source": candidate_source,
                "window_steps": window_steps,
                "window_steps_source": steps_source,
                "selected_winner": decision.get("winner"),
                "complete_nomination": decision.get("complete"),
            }
        )

    early_assessments = [
        item["state_summary"]
        for item in decisions[:2]
        if isinstance(item.get("state_summary"), str)
    ]
    changed_files = [
        item for item in source.result.get("changedFiles", []) if isinstance(item, str)
    ]
    surface_names = {Path(item).name for item in changed_files}
    first_surface_assessment = next(
        (
            item["state_summary"]
            for item in decisions
            if isinstance(item.get("state_summary"), str)
            and any(name in item["state_summary"] for name in surface_names)
        ),
        None,
    )
    terminal_assessment = next(
        (
            item["state_summary"]
            for item in reversed(decisions)
            if isinstance(item.get("state_summary"), str)
        ),
        None,
    )
    result_rel = source.result_path.relative_to(agentresults_root).as_posix()
    result_digest = hashlib.sha256(source.result_path.read_bytes()).hexdigest()
    reward = source.result.get("reward")
    public_reward = {
        key: reward.get(key)
        for key in ("reward", "partial", "f2p", "p2p")
        if isinstance(reward, dict) and key in reward
    }
    return {
        "type": "asgard_known_success_trace",
        "schema_version": SCHEMA_VERSION,
        "protected_identity": source.identity,
        "task_id": source.task_id,
        "run": source.run,
        "cohort": source.cohort,
        "provenance": {
            "agent_result": result_rel,
            "agent_result_sha256": result_digest,
            "archive": f"{source.task_id}/{source.archive_path.name}",
            "base_commit": source.result.get("baseCommit"),
            "anvil_sha256": source.result.get("anvilSha256"),
            "selection_rule": "public result outcome == SUCCESS",
            "excluded_members": ["verifier-output.txt", "verifier.tar.gz"],
        },
        "facts": {
            "public_endpoint": {
                "outcome": source.result.get("outcome"),
                "stop_reason": source.result.get("stopReason"),
                "reward": public_reward,
            },
            "contract_reading": {
                "checklist": checklist,
                "source": "agent-side asgard_checklist trace record",
            },
            "architectural_direction": {
                "early_supervisor_assessments": early_assessments,
                "first_assessment_mentioning_final_surface": first_surface_assessment,
                "terminal_supervisor_assessment": terminal_assessment,
                "source": (
                    "supervisor-authored selected-lane state summaries. The surface "
                    "assessment is the first summary mechanically mentioning a final "
                    "changed-file basename; these are model assessments, not "
                    "independent ground truth"
                ),
            },
            "implementation_surface": _patch_surface(
                patch,
                changed_files,
            ),
            "cross_window_evidence_and_risk": {
                "selected_boundary_records": decisions,
                "source": (
                    "ordinary supervisor decisions; state_summary is selected-lane "
                    "evidence/risk and next_advices are obligations funded next"
                ),
            },
            "verification": {
                "commands_observed": [
                    {"command": command, "tool_reported_success": success}
                    for command, success in sorted(verification.items())
                ],
                "scope": (
                    "all candidate lanes; older trace records do not reliably attach "
                    "tool_timing rows to the selected lane. null success means the "
                    "full command was recovered only from stderr because the timing "
                    "row was truncated; tool success is not equivalent to assertions "
                    "passing"
                ),
                "supervisor_reported_evidence": [
                    item["state_summary"]
                    for item in decisions
                    if isinstance(item.get("state_summary"), str)
                    and VERIFY_RE.search(item["state_summary"])
                ],
            },
            "funding_sequence": sequence,
            "completion_review": [
                _decision_view(index, row)
                for index, row in enumerate(completion_rows, 1)
            ],
            "weak_after_one_or_two_steps": {
                "status": "unknown",
                "reason": (
                    "the archive has no explicit weak/strong score for the eventually "
                    "selected lane at exactly candidate step 1 or 2; selection occurs "
                    "only after a complete funded window"
                ),
            },
        },
        "inferences": {
            "replay_obligations": (
                "A compressed replay should preserve the complete checklist, every "
                "selected boundary state_summary and next advice, the patch surface, "
                "and verification references. This is a research gate, not a claim "
                "that every recorded sentence caused the success."
            )
        },
    }


def build_corpus(
    agentresults_root: Path,
    archive_root: Path,
    cohorts: tuple[str, ...] = DEFAULT_COHORTS,
    expected_runs: int | None = 40,
    expected_successes: int | None = 9,
    expected_union: int | None = 15,
    expected_common: int | None = 3,
) -> list[dict[str, Any]]:
    successes, run_counts = discover_results(agentresults_root, archive_root, cohorts)
    success_counts = Counter(item.cohort for item in successes)
    identities_by_cohort = {
        cohort: {item.identity for item in successes if item.cohort == cohort}
        for cohort in cohorts
    }
    union = set().union(*identities_by_cohort.values())
    common = set.intersection(*identities_by_cohort.values())
    if expected_runs is not None:
        for cohort in cohorts:
            if run_counts[cohort] != expected_runs:
                raise ValueError(
                    f"{cohort}: expected {expected_runs} results, found {run_counts[cohort]}"
                )
    if expected_successes is not None:
        for cohort in cohorts:
            if success_counts[cohort] != expected_successes:
                raise ValueError(
                    f"{cohort}: expected {expected_successes} successes, "
                    f"found {success_counts[cohort]}"
                )
    if expected_union is not None and len(union) != expected_union:
        raise ValueError(f"expected success union {expected_union}, found {len(union)}")
    if expected_common is not None and len(common) != expected_common:
        raise ValueError(f"expected common successes {expected_common}, found {len(common)}")

    manifest = {
        "type": "asgard_known_success_manifest",
        "schema_version": SCHEMA_VERSION,
        "cohorts": list(cohorts),
        "result_counts": dict(sorted(run_counts.items())),
        "success_trace_counts": {
            cohort: success_counts[cohort] for cohort in cohorts
        },
        "protected_union_count": len(union),
        "protected_identities": sorted(union),
        "common_success_count": len(common),
        "common_success_identities": sorted(common),
        "success_trace_count": len(successes),
        "hidden_test_leakage_policy": (
            "Only public result status/reward and agent-side trace, stderr, and patch "
            "members are read; verifier output and verifier archives are excluded."
        ),
    }
    records = [
        extract_success_trace(item, agentresults_root)
        for item in sorted(successes, key=lambda item: (item.identity, item.cohort))
    ]
    common_set = set(common)
    for record in records:
        record["common_success"] = record["protected_identity"] in common_set
    return [manifest, *records]


def _write_jsonl(records: list[dict[str, Any]], output: Path | None) -> None:
    text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the strict v6/v9 Asgard known-success JSONL corpus."
    )
    parser.add_argument(
        "--agentresults-root",
        type=Path,
        default=Path("/home/jonathan/Projects/brokkbench/agentresults"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/jonathan/brokkbench-archive"),
    )
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--expected-runs", type=int, default=40)
    parser.add_argument("--expected-successes", type=int, default=9)
    parser.add_argument("--expected-union", type=int, default=15)
    parser.add_argument("--expected-common", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = build_corpus(
            args.agentresults_root,
            args.archive_root,
            expected_runs=args.expected_runs,
            expected_successes=args.expected_successes,
            expected_union=args.expected_union,
            expected_common=args.expected_common,
        )
        _write_jsonl(records, args.output)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
