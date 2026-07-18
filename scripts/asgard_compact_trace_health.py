#!/usr/bin/env python3
"""Audit Asgard compact-trajectory runs (v12+) for supervisor dumbassery.

v12 replaced LLM candidate-window summarization with compact deterministic
rendering plus an unbudgeted `view_tool_call` retrieval tool. That trade only
pays off if the supervisor actually retrieves. The failure mode v11 removed was
a summarizer inventing work; the failure mode it introduces is a supervisor
confidently adjudicating from one-line summaries it never expanded.

This script reads `anvil-trace.jsonl` files (loose, in directories, or inside
run archives) and reports the signals that distinguish those cases.

Usage:
    asgard_compact_trace_health.py PATH [PATH ...] [--json OUT] [--verbose]

PATH may be an anvil-trace.jsonl, a .zip run archive, or a directory searched
recursively for either.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TRACE_NAME = "anvil-trace.jsonl"
HANDLE_RE = re.compile(r"w(\d+)l(\d+)m(\d+)$")

# A decision reached with no retrieval is not automatically wrong — a window
# whose compact lines genuinely settle the question needs no expansion. It is
# only alarming in aggregate, or when the decision claims execution evidence.
EXECUTION_EVIDENCE_MARKERS = (
    "exit 0",
    "passed",
    "green",
    "test result",
    "ledger",
)


@dataclass
class RunHealth:
    source: str
    handoffs: list[dict] = field(default_factory=list)
    retrievals: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)
    legacy_modes: Counter = field(default_factory=Counter)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    @property
    def retrieved_handle_count(self) -> int:
        return sum(len(record.get("handles") or []) for record in self.retrievals)

    @property
    def unresolved_handle_count(self) -> int:
        return sum(len(record.get("unresolved") or []) for record in self.retrievals)

    def compression_ratios(self) -> list[float]:
        ratios = []
        for handoff in self.handoffs:
            raw = handoff.get("raw_bytes") or 0
            packed = handoff.get("packed_bytes") or 0
            if raw and packed:
                ratios.append(raw / packed)
        return ratios

    def windows_with_retrieval(self) -> set:
        return {record.get("window") for record in self.retrievals}

    def windows_seen(self) -> set:
        return {record.get("window") for record in self.handoffs}


def iter_trace_sources(path: Path):
    """Yields (label, lines) for every anvil-trace.jsonl reachable from path."""
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and (child.name == TRACE_NAME or child.suffix == ".zip"):
                yield from iter_trace_sources(child)
        return
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as bundle:
                for name in bundle.namelist():
                    if name.endswith(TRACE_NAME):
                        text = bundle.read(name).decode("utf-8", errors="replace")
                        yield f"{path}::{name}", text.splitlines()
        except zipfile.BadZipFile:
            print(f"warning: {path} is not a readable archive", file=sys.stderr)
        return
    if path.name == TRACE_NAME:
        yield str(path), path.read_text(errors="replace").splitlines()


def analyze(label: str, lines: list[str]) -> RunHealth:
    health = RunHealth(source=label)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = record.get("type")
        if kind == "asgard_candidate_handoff":
            health.handoffs.append(record)
            health.legacy_modes[record.get("mode", "unknown")] += 1
        elif kind == "asgard_supervisor_retrieval":
            health.retrievals.append(record)
        elif kind == "asgard_decision":
            health.decisions.append(record)
        elif kind == "asgard_window":
            health.windows.append(record)
    return health


def decision_claims_execution_evidence(decision: dict) -> bool:
    payload = decision.get("decision") or {}
    blob = json.dumps(payload.get("contracts") or []) + " " + str(
        payload.get("state_summary", "")
    )
    lowered = blob.lower()
    return any(marker in lowered for marker in EXECUTION_EVIDENCE_MARKERS)


def classify_unresolved(runs: list[RunHealth]) -> tuple[int, int, int]:
    """Splits unresolved handles into prior-window, current-window, malformed.

    Prior-window misses are expected and benign: the dossier shows trajectories
    from earlier windows whose handles look valid but are not retrievable, and
    the supervisor gets an explicit error rather than wrong data. A
    current-window miss is the one that means something is broken.
    """
    prior = current = malformed = 0
    for run in runs:
        for record in run.retrievals:
            window = record.get("window")
            for handle in record.get("unresolved") or []:
                parsed = HANDLE_RE.match(handle)
                if not parsed:
                    malformed += 1
                elif int(parsed.group(1)) != window:
                    prior += 1
                else:
                    current += 1
    return prior, current, malformed


def report(runs: list[RunHealth], verbose: bool) -> dict:
    total_handoffs = sum(len(run.handoffs) for run in runs)
    total_retrievals = sum(len(run.retrievals) for run in runs)
    total_handles = sum(run.retrieved_handle_count for run in runs)
    total_unresolved = sum(run.unresolved_handle_count for run in runs)
    all_ratios = [ratio for run in runs for ratio in run.compression_ratios()]
    legacy = Counter()
    for run in runs:
        legacy.update(run.legacy_modes)

    windows_total = 0
    windows_without_retrieval = 0
    for run in runs:
        seen = run.windows_seen()
        retrieved = run.windows_with_retrieval()
        windows_total += len(seen)
        windows_without_retrieval += len(seen - retrieved)

    silent_execution_claims = []
    for run in runs:
        retrieved_windows = run.windows_with_retrieval()
        for decision in run.decisions:
            if not decision_claims_execution_evidence(decision):
                continue
            # asgard_decision carries no window field; treat a run that never
            # retrieved at all as the alarming case.
            if not retrieved_windows:
                silent_execution_claims.append(run.source)
                break

    summary = {
        "runs": len(runs),
        "candidate_handoffs": total_handoffs,
        "handoff_modes": dict(legacy),
        "retrieval_rounds": total_retrievals,
        "handles_retrieved": total_handles,
        "handles_unresolved": total_unresolved,
        "windows_total": windows_total,
        "windows_without_retrieval": windows_without_retrieval,
        "compression_ratio_median": (
            round(statistics.median(all_ratios), 1) if all_ratios else None
        ),
        "compression_ratio_p90": (
            round(sorted(all_ratios)[int(len(all_ratios) * 0.9)], 1)
            if len(all_ratios) >= 10
            else None
        ),
        "runs_claiming_execution_evidence_without_retrieving": sorted(
            set(silent_execution_claims)
        ),
    }

    print("=" * 72)
    print("Asgard compact-trajectory health")
    print("=" * 72)
    print(f"runs analyzed                : {summary['runs']}")
    print(f"candidate handoffs           : {total_handoffs}")
    if all_ratios:
        print(
            f"compression (raw/compact)    : median {summary['compression_ratio_median']}x"
            + (
                f", p90 {summary['compression_ratio_p90']}x"
                if summary["compression_ratio_p90"]
                else ""
            )
        )
    print()
    print("-- retrieval behavior " + "-" * 50)
    print(f"retrieval rounds             : {total_retrievals}")
    print(f"handles retrieved            : {total_handles}")
    if windows_total:
        pct = 100.0 * windows_without_retrieval / windows_total
        print(
            f"windows with no retrieval    : {windows_without_retrieval}/{windows_total} ({pct:.0f}%)"
        )
    print()

    problems = []

    stale = {mode: count for mode, count in legacy.items() if mode != "compact_deterministic"}
    if stale:
        problems.append(
            f"DEPLOY: {sum(stale.values())} handoffs used pre-v11 modes {stale} — "
            "some lane is running an old binary."
        )
    if total_handoffs and total_retrievals == 0:
        problems.append(
            "BLIND: the supervisor never called view_tool_call. Either the tool is not "
            "reaching it, or the prompt is not persuading it to expand — check that a "
            "view_tool_call definition appears in the supervisor's tool list."
        )
    prior_miss, current_miss, malformed_miss = classify_unresolved(runs)
    summary["unresolved_prior_window"] = prior_miss
    summary["unresolved_current_window"] = current_miss
    summary["unresolved_malformed"] = malformed_miss
    if prior_miss:
        requested = total_handles + total_unresolved
        pct = 100.0 * prior_miss / requested if requested else 0.0
        print(
            f"prior-window handles asked : {prior_miss} ({pct:.0f}% of requests) "
            "- not retrievable by design"
        )
    if current_miss:
        problems.append(
            f"HANDLES: {current_miss} handle(s) naming the CURRENT window failed to resolve. "
            "Either the supervisor invented an id or handle minting and resolution disagree."
        )
    if malformed_miss:
        problems.append(
            f"HANDLES: {malformed_miss} malformed handle(s) - the supervisor is not copying "
            "ids verbatim from the compact lines."
        )
    if windows_total and windows_without_retrieval / windows_total > 0.7:
        problems.append(
            f"SHALLOW: {windows_without_retrieval}/{windows_total} windows decided with no "
            "retrieval at all. The supervisor may be adjudicating from one-line summaries."
        )
    if summary["runs_claiming_execution_evidence_without_retrieving"]:
        problems.append(
            "UNGROUNDED: "
            f"{len(summary['runs_claiming_execution_evidence_without_retrieving'])} run(s) cited "
            "execution evidence in a decision without ever retrieving a full tool result. "
            "A compact line reports a command's exit code, not what it exercised."
        )
    if all_ratios and statistics.median(all_ratios) < 2.0:
        problems.append(
            f"WEAK: median compression is only {statistics.median(all_ratios):.1f}x — "
            "compaction may not be firing on the dominant entry types."
        )

    if problems:
        print("-- findings " + "-" * 60)
        for problem in problems:
            print(f"  * {problem}")
    else:
        print("no anomalies detected")
    print()

    if verbose:
        print("-- per run " + "-" * 61)
        for run in runs:
            ratios = run.compression_ratios()
            median = f"{statistics.median(ratios):.1f}x" if ratios else "n/a"
            print(
                f"  {run.source}\n"
                f"    handoffs={len(run.handoffs)} retrievals={len(run.retrievals)} "
                f"handles={run.retrieved_handle_count} unresolved={run.unresolved_handle_count} "
                f"decisions={run.decision_count} compression={median}"
            )

    summary["findings"] = problems
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="write the summary as JSON")
    parser.add_argument("--verbose", action="store_true", help="per-run breakdown")
    args = parser.parse_args()

    runs = []
    for path in args.paths:
        if not path.exists():
            print(f"warning: {path} does not exist", file=sys.stderr)
            continue
        for label, lines in iter_trace_sources(path):
            runs.append(analyze(label, lines))

    if not runs:
        print("no anvil-trace.jsonl found in the given paths", file=sys.stderr)
        return 1

    summary = report(runs, args.verbose)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")
    return 2 if summary["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
