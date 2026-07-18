#!/usr/bin/env python3
"""Launch or inspect staged live-experiment controllers across goal turns."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LaunchSpec:
    batch: str
    attempt: int
    command: list[str]
    working_directory: Path
    root: Path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def discover_specs(
    root: Path, batches: set[str], attempts: set[int]
) -> list[LaunchSpec]:
    specs: list[LaunchSpec] = []
    for run_path in sorted(root.glob("*/run.json")):
        value = json.loads(run_path.read_text(encoding="utf-8"))
        batch = value.get("batch", {}).get("name")
        commands = value.get("commands")
        cwd = value.get("working_directory")
        if not isinstance(batch, str) or not isinstance(commands, list) or not isinstance(cwd, str):
            raise ValueError(f"invalid staged run descriptor: {run_path}")
        if batches and batch not in batches:
            continue
        for attempt, command in enumerate(commands, 1):
            if attempts and attempt not in attempts:
                continue
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ValueError(f"invalid command in {run_path}")
            specs.append(LaunchSpec(batch, attempt, command, Path(cwd), run_path.parent))
    missing = batches - {spec.batch for spec in specs}
    if missing:
        raise ValueError(f"unknown or unselected batches: {', '.join(sorted(missing))}")
    return specs


def controller_state(spec: LaunchSpec) -> dict[str, Any]:
    attempt_root = spec.root / f"attempt-{spec.attempt}"
    state_path = attempt_root / "controller.json"
    state: dict[str, Any] = {}
    if state_path.is_file():
        value = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            state = value
    pid = state.get("pid")
    captured_archives = 0
    incomplete_archives = 0
    for path in (attempt_root / "archives").rglob("*.zip"):
        try:
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
        except zipfile.BadZipFile:
            incomplete_archives += 1
            continue
        if {"anvil-trace.jsonl", "result.json"}.issubset(members):
            captured_archives += 1
        else:
            incomplete_archives += 1
    result_paths = [
        path
        for path in (attempt_root / "results").rglob("*.json")
        if ".running" not in path.parts
    ]
    completed_results = sum(
        not path.name.startswith("INFRA_ERROR") for path in result_paths
    )
    infra_errors = sum(path.name.startswith("INFRA_ERROR") for path in result_paths)
    running_markers = len(
        list((attempt_root / "results" / ".running").glob("*/*.json"))
    )
    return {
        "batch": spec.batch,
        "attempt": spec.attempt,
        "pid": pid,
        "alive": isinstance(pid, int) and pid_alive(pid),
        "captured_archives": captured_archives,
        "incomplete_archives": incomplete_archives,
        "completed_results": completed_results,
        "infra_errors": infra_errors,
        "running_markers": running_markers,
        "log": str(attempt_root / "controller.log"),
    }


def launch(spec: LaunchSpec) -> dict[str, Any]:
    attempt_root = spec.root / f"attempt-{spec.attempt}"
    attempt_root.mkdir(parents=True, exist_ok=True)
    current = controller_state(spec)
    if current["alive"]:
        raise ValueError(
            f"{spec.batch} attempt {spec.attempt} controller {current['pid']} is alive"
        )
    log_path = attempt_root / "controller.log"
    with log_path.open("ab", buffering=0) as output:
        process = subprocess.Popen(
            spec.command,
            cwd=spec.working_directory,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = {
        "batch": spec.batch,
        "attempt": spec.attempt,
        "pid": process.pid,
        "command": spec.command,
        "working_directory": str(spec.working_directory),
        "log": str(log_path),
    }
    (attempt_root / "controller.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return controller_state(spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/asgard-3-4-live-v2"))
    parser.add_argument("--batch", action="append", default=[])
    parser.add_argument("--attempt", action="append", type=int, default=[])
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        specs = discover_specs(args.root, set(args.batch), set(args.attempt))
        if not specs:
            parser.error("no staged commands selected")
        rows = [controller_state(spec) if args.status else launch(spec) for spec in specs]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    json.dump({"controllers": rows}, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
