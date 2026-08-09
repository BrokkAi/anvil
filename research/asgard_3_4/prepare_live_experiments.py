#!/usr/bin/env python3
"""Stage reproducible task directories and runner commands for live studies."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("manifest must be a schema_version 1 object")
    return value


def rewrite_task_environment(
    task_toml: Path, managed_keys: set[str], environment: dict[str, str]
) -> None:
    text = task_toml.read_text(encoding="utf-8")
    lines = text.splitlines()
    header = "[environment.env]"
    try:
        start = lines.index(header)
    except ValueError:
        insertion = next(
            (index for index, line in enumerate(lines) if line.startswith("[solution")),
            len(lines),
        )
        lines[insertion:insertion] = [header]
        start = insertion
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("[")),
        len(lines),
    )
    kept = []
    for line in lines[start + 1 : end]:
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key not in managed_keys:
            kept.append(line)
    injected = [f'{key} = {json.dumps(value)}' for key, value in sorted(environment.items())]
    lines[start + 1 : end] = injected + kept
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def runner_command(
    manifest: dict[str, Any], batch: dict[str, Any], group: int, runs: int
) -> list[str]:
    root = Path(manifest["output_root"])
    batch_root = root / batch["name"]
    attempt_root = batch_root / f"attempt-{group}"
    return [
        "uv",
        "run",
        "python",
        "bpr_agent.py",
        "--engine",
        "deepswe",
        "--models",
        f'{batch["label"]}g{group}={manifest["candidate_model"]}',
        "--tasksdir",
        str(batch_root / "tasks"),
        "--results-dir",
        str(attempt_root / "results"),
        "--archive-dir",
        str(attempt_root / "archives"),
        "--threads",
        str(manifest["threads"]),
        "--launch-threads",
        str(manifest["threads"]),
        "--runs",
        str(runs),
        "--asgard-candidates",
        str(manifest["candidate_count"]),
        "--asgard-supervisor",
        manifest["supervisor_model"],
        "--anvil-bin",
        manifest["anvil_bin"],
        "--no-anvil-rebuild",
        "--headless",
    ]


def stage(manifest: dict[str, Any], selected: set[str]) -> list[list[str]]:
    source_root = Path(manifest["source_tasks_root"])
    output_root = Path(manifest["output_root"])
    managed = set(manifest["managed_environment_keys"])
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    batches = [
        batch
        for batch in manifest["batches"]
        if not selected or batch["name"] in selected
    ]
    missing = selected - {batch["name"] for batch in batches}
    if missing:
        raise ValueError(f"unknown batches: {', '.join(sorted(missing))}")
    commands = []
    for batch in batches:
        batch_root = output_root / batch["name"]
        tasks_root = batch_root / "tasks"
        for task in batch["tasks"]:
            source = source_root / task
            if not (source / "task.toml").is_file():
                raise ValueError(f"missing source task: {source}")
            destination = tasks_root / task
            shutil.copytree(source, destination)
            rewrite_task_environment(
                destination / "task.toml", managed, batch["environment"]
            )
        batch_commands = [
            runner_command(manifest, batch, group, runs)
            for group, runs in enumerate(batch["run_groups"], 1)
        ]
        if any(runs not in {1, 2} for runs in batch["run_groups"]):
            raise ValueError(f"{batch['name']}: every run group must be 1 or 2")
        commands.extend(batch_commands)
        (batch_root / "run.json").write_text(
            json.dumps(
                {
                    "batch": batch,
                    "commands": batch_commands,
                    "working_directory": manifest["runner_root"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (output_root / "commands.sh").write_text(
        "\n".join(
            f"(cd {shlex.quote(manifest['runner_root'])} && {shlex.join(command)})"
            for command in commands
        )
        + "\n",
        encoding="utf-8",
    )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("live_experiments.json"),
    )
    parser.add_argument("--batch", action="append", default=[])
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        commands = stage(manifest, set(args.batch))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    for command in commands:
        sys.stdout.write(shlex.join(command) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
