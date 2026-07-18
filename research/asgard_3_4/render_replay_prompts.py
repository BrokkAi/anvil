#!/usr/bin/env python3
"""Render same-state compact Asgard prompts from replay-capture records.

This module deliberately performs no model calls.  It turns a captured ordinary
`full` routing state into deterministic alternative requests and can validate its
renderer byte-for-byte against requests captured from compact live runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable


MODES = (
    "full",
    "latest-state",
    "checkpoint-plus-delta",
    "decision-log-only",
    "recent-exact-tail",
)

STATE_UPDATE_CONTRACT = (
    "<state_update_contract>The state_summary in your next select_trajectory call must be a "
    "self-contained cumulative replacement for this state: retain the selected architectural "
    "direction, files and symbols involved, established facts with window-qualified evidence "
    "IDs, unresolved task contracts and adverse conditions, known failed approaches or defects, "
    "latest verification results, and the next serial dependency. Do not assume omitted prior "
    "prose will be available on the next routing turn.</state_update_contract>"
)


def trace_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("anvil-trace.jsonl").decode("utf-8", "replace")
    else:
        raw = path.read_text(encoding="utf-8")
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: trace row is not an object")
        yield value


def captured_routing_states(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prompt: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    active: dict[str, Any] | None = None
    for row in rows:
        kind = row.get("type")
        if kind == "asgard_supervisor_prompt_mode":
            prompt = row
            state = None
        elif kind == "asgard_supervisor_replay_state" and prompt is not None:
            state = row
        elif (
            kind == "asgard_supervisor_replay_request"
            and row.get("decision_call") == "supervisor"
            and row.get("call_index") == 1
            and prompt is not None
            and state is not None
        ):
            active = {"prompt": prompt, "state": state, "request": row}
            records.append(active)
            prompt = None
            state = None
        elif (
            kind == "asgard_supervisor_replay_response"
            and active is not None
            and active.get("first_response") is None
            and row.get("call_index") == 1
        ):
            active["first_response"] = row
        elif (
            kind == "asgard_decision"
            and row.get("call") == "supervisor"
            and active is not None
        ):
            active["control_decision"] = row.get("decision")
            active = None
    return records


def _message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _rust_debug_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _rust_debug_option(value: Any) -> str:
    if value is None:
        return "None"
    return f"Some({_rust_debug_string(str(value))})"


def render_dossier_messages(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for index, message in enumerate(messages):
        rendered.append(
            f"<message index={index} role={_rust_debug_string(message['role'])} "
            f"name={_rust_debug_option(message.get('name'))} "
            f"tool_call_id={_rust_debug_option(message.get('tool_call_id'))}>\n"
        )
        for part in message.get("content", []):
            if part.get("type") == "text":
                rendered.append(f"<content>\n{part.get('text', '')}\n</content>\n")
            elif part.get("type") == "image":
                rendered.append(
                    f"<image_url>\n{part.get('image_url', '')}\n</image_url>\n"
                )
        if message.get("reasoning_content") is not None:
            rendered.append(
                f"<reasoning>\n{message['reasoning_content']}\n</reasoning>\n"
            )
        for call in message.get("tool_calls") or []:
            function = call["function"]
            rendered.append(
                f"<tool_call id={_rust_debug_string(call['id'])} "
                f"name={_rust_debug_string(function['name'])}>\n"
                f"{function['arguments']}\n</tool_call>\n"
            )
        rendered.append("</message>\n")
    return "".join(rendered)


def render_history_entry(entry: dict[str, Any]) -> str:
    return (
        f"<supervisor_decision window=\"{entry['window']}\" "
        f"selected_lane=\"{entry['winner']}\">\n{entry['state_summary']}\n"
        "</supervisor_decision>"
    )


def render_ledger(entries: list[list[Any]], through: int | None = None) -> str:
    rendered: list[str] = []
    for window, ledger in entries:
        if through is not None and window > through:
            break
        pretty = json.dumps(ledger, indent=2, ensure_ascii=False, separators=(",", ": "))
        rendered.append(
            f"<canonical_execution_ledger window=\"{window}\">\n{pretty}\n"
            "</canonical_execution_ledger>"
        )
    return "\n".join(rendered)


def _canonical_state(
    mode: str, state: dict[str, Any] | None, ledger: str
) -> dict[str, Any]:
    source_window = state["window"] if state else 0
    summary = state["state_summary"] if state else "No prior selected window."
    text = (
        f"<canonical_state mode=\"{mode}\" cumulative=\"true\" "
        f"source_window=\"{source_window}\">\n"
        f"<latest_state_summary lossy=\"true\">\n{summary}\n</latest_state_summary>\n"
        "<canonical_execution_history mechanically_derived=\"true\" "
        f"evidence_ids=\"window plus ledger id\">\n{ledger}\n"
        "</canonical_execution_history>\n"
        f"{STATE_UPDATE_CONTRACT}\n</canonical_state>"
    )
    return _message("assistant", text)


def _append_tail(
    messages: list[dict[str, Any]],
    windows: list[list[dict[str, Any]]],
    history: list[dict[str, Any]],
    start: int,
) -> None:
    if len(windows) != len(history):
        raise ValueError("selected window messages/history lengths differ")
    for index in range(start, len(windows)):
        messages.append(
            _message("user", f'<selected_trajectory_window_boundary index="{index}" />')
        )
        decision = render_history_entry(history[index])
        dossier = render_dossier_messages(windows[index])
        messages.append(
            _message(
                "assistant",
                f'<selected_trajectory_window index="{index}">\n{dossier}\n'
                f"</selected_trajectory_window>\n{decision}",
            )
        )


def full_control_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    request_messages = record["request"]["messages"]
    if len(request_messages) < 4:
        raise ValueError("captured request lacks stable prefix/current dossier")
    state = record["state"]
    history_object = state["supervisor_history"]
    messages = copy.deepcopy(request_messages[:3])
    selected_initial = render_dossier_messages(state["selected_trajectory_initial"])
    messages.append(
        _message(
            "assistant",
            f"<selected_trajectory_initial>\n{selected_initial}\n"
            "</selected_trajectory_initial>",
        )
    )
    checkpointed = history_object["checkpointed"]
    if checkpointed:
        rendered = "\n".join(render_history_entry(row) for row in checkpointed)
        messages.append(
            _message(
                "assistant",
                '<supervisor_decision_history checkpointed="true">\n'
                f"{rendered}\n</supervisor_decision_history>",
            )
        )
    _append_tail(
        messages,
        state["selected_trajectory_windows"],
        history_object["selected_windows"],
        0,
    )
    messages.append(copy.deepcopy(request_messages[-1]))
    return messages


def render_mode_messages(
    record: dict[str, Any],
    mode: str,
    checkpoint_interval: int = 3,
    recent_exact_tail: int = 2,
    preserve_bootstrap_initial: bool = True,
) -> list[dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    request_messages = record["request"]["messages"]
    full_messages = full_control_messages(record)
    if mode == "full":
        return full_messages
    if len(request_messages) < 4:
        raise ValueError("captured request lacks stable prefix/current dossier")
    state = record["state"]
    windows = state["selected_trajectory_windows"]
    history_object = state["supervisor_history"]
    checkpointed = history_object["checkpointed"]
    selected = history_object["selected_windows"]
    ledgers = state["canonical_ledger"]
    latest = selected[-1] if selected else (checkpointed[-1] if checkpointed else None)
    messages = copy.deepcopy(full_messages[:3])
    if preserve_bootstrap_initial and latest is None:
        messages.append(copy.deepcopy(full_messages[3]))

    if mode == "latest-state":
        messages.append(_canonical_state(mode, latest, render_ledger(ledgers)))
    elif mode == "decision-log-only":
        decisions = "\n".join(render_history_entry(row) for row in checkpointed + selected)
        ledger = render_ledger(ledgers)
        messages.append(
            _message(
                "assistant",
                '<canonical_state mode="decision-log-only" cumulative="true">\n'
                f"<prior_decisions>\n{decisions}\n</prior_decisions>\n"
                "<canonical_execution_history mechanically_derived=\"true\">\n"
                f"{ledger}\n</canonical_execution_history>\n"
                f"{STATE_UPDATE_CONTRACT}\n</canonical_state>",
            )
        )
    elif mode == "checkpoint-plus-delta":
        latest_window = latest["window"] if latest else 0
        interval_checkpoint = latest_window // checkpoint_interval * checkpoint_interval
        history_checkpoint = checkpointed[-1]["window"] if checkpointed else 0
        checkpoint_window = max(interval_checkpoint, history_checkpoint)
        checkpoint = next(
            (
                row
                for row in checkpointed + selected
                if row["window"] == checkpoint_window
            ),
            None,
        )
        messages.append(
            _canonical_state(
                mode, checkpoint, render_ledger(ledgers, through=checkpoint_window)
            )
        )
        delta_start = next(
            (index for index, row in enumerate(selected) if row["window"] > checkpoint_window),
            len(selected),
        )
        _append_tail(messages, windows, selected, delta_start)
    elif mode == "recent-exact-tail":
        tail_start = max(0, len(selected) - recent_exact_tail)
        checkpoint = selected[tail_start - 1] if tail_start else (checkpointed[-1] if checkpointed else None)
        messages.append(_canonical_state(mode, checkpoint, render_ledger(ledgers)))
        _append_tail(messages, windows, selected, tail_start)

    messages.append(copy.deepcopy(full_messages[-1]))
    return messages


def validate_captured_mode(record: dict[str, Any]) -> None:
    prompt = record["prompt"]
    mode = prompt["mode"]
    measured_full_bytes = len(
        render_dossier_messages(full_control_messages(record)).encode("utf-8")
    )
    expected_full_bytes = prompt.get("full_control_prompt_bytes")
    if expected_full_bytes is not None and measured_full_bytes != expected_full_bytes:
        raise ValueError(
            f"captured {mode} window {prompt.get('window')} full-control bytes "
            f"{measured_full_bytes} != telemetry {expected_full_bytes}"
        )
    measured_chosen_bytes = len(
        render_dossier_messages(record["request"]["messages"]).encode("utf-8")
    )
    expected_chosen_bytes = prompt.get("prompt_bytes")
    if expected_chosen_bytes is not None and measured_chosen_bytes != expected_chosen_bytes:
        raise ValueError(
            f"captured {mode} window {prompt.get('window')} chosen bytes "
            f"{measured_chosen_bytes} != telemetry {expected_chosen_bytes}"
        )
    rendered = render_mode_messages(
        record,
        mode,
        checkpoint_interval=int(prompt.get("checkpoint_interval", 3)),
        recent_exact_tail=int(prompt.get("recent_exact_tail", 2)),
    )
    if rendered == record["request"]["messages"]:
        return
    # Captures made before the bootstrap-safety correction omitted the exact
    # selected-initial message when no cumulative state existed. Accept that
    # historical ablation for reproducibility, while new replays default to the
    # corrected representation.
    legacy = render_mode_messages(
        record,
        mode,
        checkpoint_interval=int(prompt.get("checkpoint_interval", 3)),
        recent_exact_tail=int(prompt.get("recent_exact_tail", 2)),
        preserve_bootstrap_initial=False,
    )
    if legacy != record["request"]["messages"]:
        raise ValueError(
            f"captured {mode} window {prompt.get('window')} does not match renderer"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="archive zip or anvil-trace.jsonl")
    parser.add_argument("--validate", action="store_true", help="validate captured mode")
    parser.add_argument("--mode", choices=MODES, help="render this mode as JSONL")
    args = parser.parse_args()
    records = captured_routing_states(trace_rows(args.trace))
    if not records:
        parser.error("no complete ordinary routing capture records found")
    try:
        if args.validate:
            for record in records:
                validate_captured_mode(record)
        if args.mode:
            for record in records:
                output = {
                    "window": record["prompt"].get("window"),
                    "source_mode": record["prompt"].get("mode"),
                    "target_mode": args.mode,
                    "messages": render_mode_messages(record, args.mode),
                    "tools": record["request"]["tools"],
                    "model": record["request"]["model"],
                    "parameters": record["request"]["parameters"],
                }
                print(json.dumps(output, ensure_ascii=False))
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.validate:
        print(f"validated {len(records)} routing records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
