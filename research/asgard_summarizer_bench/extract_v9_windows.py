#!/usr/bin/env python3
"""Extract a deterministic 100-example candidate-window sample from Asgard v9 archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path


LANE_RE = re.compile(r"/asgard-(\d+)-")
ADVICE_MARKER = "<asgard_next_window_advice"
DIRECT_WRITE_TOOLS = {"edit", "write_file"}
SHELL_SOURCE_EDIT_RE = re.compile(r"\b(?:sed|ruby)\s+-[^\n]*i\b", re.IGNORECASE)


def text_parts(message: dict) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def lane_id(messages: list[dict]) -> int | None:
    for message in messages:
        if message.get("role") != "system":
            continue
        match = LANE_RE.search(text_parts(message))
        if match:
            return int(match.group(1))
    return None


def current_window_messages(messages: list[dict]) -> list[dict] | None:
    starts = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and ADVICE_MARKER in text_parts(message)
    ]
    if not starts:
        return None
    return messages[starts[-1] :]


def render_messages(messages: list[dict]) -> str:
    rendered: list[str] = []
    for index, message in enumerate(messages):
        rendered.append(f'<message index="{index}" role="{message.get("role", "unknown")}">')
        text = text_parts(message)
        if text:
            rendered.extend(("<content>", text, "</content>"))
        reasoning = message.get("reasoning_content")
        if reasoning:
            rendered.extend(("<reasoning>", str(reasoning), "</reasoning>"))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            rendered.append(
                f'<tool_call name="{function.get("name", "unknown")}">'
            )
            rendered.append(str(function.get("arguments", "")))
            rendered.append("</tool_call>")
        rendered.append("</message>")
    return "\n".join(rendered)


def classify_activity(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate observation/verification windows from windows that change worktree state.

    A shell is general-purpose, so a shell-only window is kept in a third, excluded stratum
    unless the command contains an unmistakable in-place source editor. This makes the two
    benchmark cohorts positive-evidence sets rather than guessing whether a build/test or
    opaque shell command changed the worktree.
    """
    evidence: list[dict] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name", "")
            arguments = function.get("arguments", "")
            if name in DIRECT_WRITE_TOOLS:
                evidence.append({"tool": name})
            elif name == "run_shell_command":
                try:
                    command = json.loads(arguments).get("command", "")
                except (json.JSONDecodeError, AttributeError):
                    command = arguments
                evidence.append({"tool": name, "command": command})
    if any(item["tool"] in DIRECT_WRITE_TOOLS for item in evidence):
        return "edit-producing", evidence
    shell_commands = [item["command"] for item in evidence if item["tool"] == "run_shell_command"]
    if any(SHELL_SOURCE_EDIT_RE.search(command) for command in shell_commands):
        return "edit-producing", evidence
    if shell_commands:
        return "ambiguous-shell", evidence
    return "read-only", []


def load_candidates(analysis_path: Path) -> list[dict]:
    analysis = json.loads(analysis_path.read_text())
    candidates: list[dict] = []
    for run in analysis["runs"]:
        if run.get("version") != "v9":
            continue
        archive = Path(run["path"])
        with zipfile.ZipFile(archive) as bundle:
            task_text = bundle.read("instruction.md").decode("utf-8", errors="replace")
            events = [
                json.loads(line)
                for line in bundle.read("anvil-trace.jsonl").splitlines()
                if line.strip()
            ]

        boundaries = sorted(
            (event for event in events if event.get("type") == "asgard_window"),
            key=lambda event: event["timestamp"],
        )
        requests = sorted(
            (event for event in events if event.get("type") == "llm_request"),
            key=lambda event: event["timestamp"],
        )
        decisions = sorted(
            (
                event
                for event in events
                if event.get("type") == "asgard_decision" and event.get("call") == "supervisor"
            ),
            key=lambda event: event["timestamp"],
        )
        lower = ""
        for boundary in boundaries:
            upper = boundary["timestamp"]
            window_requests = [
                request
                for request in requests
                if lower < request["timestamp"] < upper
            ]
            window_decisions = [
                event for event in decisions if lower < event["timestamp"] <= upper
            ]
            reference_decision = window_decisions[-1]["decision"] if window_decisions else None
            lower = upper
            latest_by_lane: dict[int, dict] = {}
            for request in window_requests:
                lane = lane_id(request["messages"])
                if lane is None:
                    continue
                latest_by_lane[lane] = request

            for lane, request in sorted(latest_by_lane.items()):
                messages = current_window_messages(request["messages"])
                if not messages:
                    continue
                rendered = render_messages(messages)
                activity_class, write_evidence = classify_activity(messages)
                identity = f'{run["task"]}:r{run["run"]}:w{boundary["window"]}:l{lane}'
                candidates.append(
                    {
                        "id": identity,
                        "task": run["task"],
                        "run": run["run"],
                        "window": boundary["window"],
                        "lane": lane,
                        "candidate_count": boundary["candidate_count"],
                        "window_steps": boundary["window_steps"],
                        "source_archive": str(archive),
                        "source_request_timestamp": request["timestamp"],
                        "source_model": request["model"],
                        "source_turn": request["turn"],
                        "terminal_response_included": False,
                        "selected_lane": (
                            reference_decision.get("winner") if reference_decision else None
                        ),
                        "selected_lane_reference": (
                            reference_decision.get("state_summary")
                            if reference_decision and reference_decision.get("winner") == lane
                            else None
                        ),
                        "task_text": task_text,
                        "window_messages": messages,
                        "window_text": rendered,
                        "window_bytes": len(rendered.encode()),
                        "activity_class": activity_class,
                        "write_evidence": write_evidence,
                    }
                )
    return candidates


def stable_pick(rows: list[dict], fraction: float, salt: str) -> dict:
    ordered = sorted(rows, key=lambda row: (row["window"], row["lane"], row["id"]))
    target = round(fraction * (len(ordered) - 1))
    ordinal = int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16) % len(ordered)
    return ordered[(target + ordinal % 3 - 1) % len(ordered)]


def sample_100(candidates: list[dict]) -> list[dict]:
    by_run: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in candidates:
        by_run[(row["task"], row["run"])].append(row)

    selected: dict[str, dict] = {}
    for key, rows in sorted(by_run.items()):
        for fraction in (0.25, 0.75):
            row = stable_pick(rows, fraction, f"{key}:{fraction}")
            selected[row["id"]] = row

    remaining = [row for row in candidates if row["id"] not in selected]
    remaining.sort(key=lambda row: (row["window_bytes"], row["id"]))
    needed = 100 - len(selected)
    if needed < 0:
        raise ValueError(f"two-per-run base sample already has {len(selected)} rows")
    for index in range(needed):
        position = round((index + 0.5) * (len(remaining) - 1) / max(needed, 1))
        row = remaining[position]
        selected[row["id"]] = row
    if len(selected) != 100:
        raise ValueError(f"expected 100 unique rows, got {len(selected)}")
    return sorted(selected.values(), key=lambda row: (row["task"], row["run"], row["window"], row["lane"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path)
    parser.add_argument("--read-only-output", type=Path)
    parser.add_argument("--edit-producing-output", type=Path)
    args = parser.parse_args()

    candidates = load_candidates(args.analysis)
    sample = sample_100(candidates)
    read_only_sample = sample_100(
        [row for row in candidates if row["activity_class"] == "read-only"]
    )
    edit_producing_sample = sample_100(
        [row for row in candidates if row["activity_class"] == "edit-producing"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in sample:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    if args.all_output:
        with args.all_output.open("w") as stream:
            for row in candidates:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    for path, rows in (
        (args.read_only_output, read_only_sample),
        (args.edit_producing_output, edit_producing_sample),
    ):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "candidate_windows": len(candidates),
                "sample_windows": len(sample),
                "tasks": len({row["task"] for row in sample}),
                "runs": len({(row["task"], row["run"]) for row in sample}),
                "activity_classes": dict(
                    sorted(
                        (kind, sum(row["activity_class"] == kind for row in sample))
                        for kind in {row["activity_class"] for row in sample}
                    )
                ),
                "stratified_samples": {
                    "read-only": len(read_only_sample),
                    "edit-producing": len(edit_producing_sample),
                },
                "window_bytes": {
                    "min": min(row["window_bytes"] for row in sample),
                    "median": sorted(row["window_bytes"] for row in sample)[len(sample) // 2],
                    "max": max(row["window_bytes"] for row in sample),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
