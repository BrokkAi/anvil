#!/usr/bin/env python3
"""Call one-turn same-state Asgard supervisor replays and score agreement.

Audit-tool responses are recorded as fallbacks rather than executed: archived
candidate repositories no longer exist at that exact routing state.  The replay
therefore measures a conservative compact-selector fast path plus fallback rate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import render_replay_prompts as prompts


def source_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source_archive": str(path)}
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                result = json.loads(archive.read("result.json"))
            if isinstance(result, dict) and isinstance(result.get("taskId"), str):
                metadata["task_id"] = result["taskId"]
        except (KeyError, OSError, ValueError, zipfile.BadZipFile):
            pass
    match = re.search(r"-r(?P<run>[1-9][0-9]*)-", path.name)
    if match:
        metadata["run"] = int(match.group("run"))
    return metadata


def openai_body(record: dict[str, Any], mode: str) -> dict[str, Any]:
    request = record["request"]
    parameters = request.get("parameters") or {}
    body: dict[str, Any] = {
        "model": str(request["model"]).split("::", 1)[-1],
        "messages": prompts.render_mode_messages(record, mode),
        "tools": request["tools"],
        "stream": False,
    }
    for name in ("temperature", "service_tier"):
        if parameters.get(name) is not None:
            body[name] = parameters[name]
    return body


def parse_selection(response: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None, "malformed_response"
    calls = message.get("tool_calls") or []
    select_calls = [
        call
        for call in calls
        if (call.get("function") or {}).get("name") == "select_trajectory"
    ]
    if len(select_calls) != 1:
        names = [str((call.get("function") or {}).get("name")) for call in calls]
        return None, "audit_or_no_selection:" + ",".join(names)
    try:
        arguments = json.loads(select_calls[0]["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, "invalid_selection_arguments"
    if not isinstance(arguments, dict):
        return None, "invalid_selection_arguments"
    return arguments, "selected"


def usage_vector(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    cached = usage.get("prompt_cache_hit_tokens", usage.get("cached_tokens", 0))
    uncached = usage.get("prompt_cache_miss_tokens")
    if uncached is None:
        uncached = max(0, int(usage.get("prompt_tokens", 0)) - int(cached or 0))
    return {
        "input": int(uncached or 0),
        "cachedRead": int(cached or 0),
        "cachedWrite": int(usage.get("prompt_cache_write_tokens", 0) or 0),
        "output": int(usage.get("completion_tokens", 0) or 0),
        "thought": int(
            usage.get("reasoning_tokens", usage.get("reasoning_output_tokens", 0)) or 0
        ),
    }


def score_record(
    record: dict[str, Any],
    mode: str,
    response: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection, status = parse_selection(response)
    control = record.get("control_decision") or {}
    compared = selection is not None and bool(control)
    target_messages = prompts.render_mode_messages(record, mode)
    prompt_bytes = len(prompts.render_dossier_messages(target_messages).encode("utf-8"))
    full_control_prompt_bytes = len(
        prompts.render_dossier_messages(record["request"]["messages"]).encode("utf-8")
    )
    usage = usage_vector(response)
    control_usage = (record.get("first_response") or {}).get("usage") or {}
    raw_input_tokens = usage["input"] + usage["cachedRead"] + usage["cachedWrite"]
    full_control_raw_input_tokens = sum(
        int(control_usage.get(key, 0) or 0) for key in ("input", "cachedRead", "cachedWrite")
    )
    result = {
        "type": "asgard_supervisor_replay_result",
        "window": record["prompt"].get("window"),
        "source_mode": record["prompt"].get("mode"),
        "target_mode": mode,
        "status": status,
        "fallback_required": selection is None,
        "selection": selection,
        "control_decision": control or None,
        "winner_agreement": (
            selection.get("winner") == control.get("winner") if compared else None
        ),
        "complete_agreement": (
            selection.get("complete") == control.get("complete") if compared else None
        ),
        "candidate_count_agreement": (
            selection.get("next_candidate_count") == control.get("next_candidate_count")
            if compared
            else None
        ),
        "step_count_agreement": (
            selection.get("next_window_steps") == control.get("next_window_steps")
            if compared
            else None
        ),
        "usage": usage,
        "prompt_bytes": prompt_bytes,
        "full_control_prompt_bytes": full_control_prompt_bytes,
        "prompt_byte_reduction_fraction": (
            1 - prompt_bytes / full_control_prompt_bytes
            if full_control_prompt_bytes
            else None
        ),
        "raw_input_tokens": raw_input_tokens,
        "full_control_raw_input_tokens": full_control_raw_input_tokens or None,
    }
    result.update(metadata or {})
    if isinstance(result.get("task_id"), str) and isinstance(result.get("run"), int):
        result["task_run_identity"] = f"{result['task_id']}::r{result['run']}"
    result["record_id"] = (
        f"{result.get('task_run_identity', result.get('source_archive', 'trace'))}::"
        f"w{result['window']}::{mode}"
    )
    return result


def post_json(
    endpoint: str, api_key: str, body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("provider response is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="full-control archive or trace JSONL")
    parser.add_argument("--mode", action="append", choices=prompts.MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--window",
        action="append",
        type=int,
        help="replay only this ordinary window (repeatable)",
    )
    parser.add_argument("--endpoint", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = prompts.captured_routing_states(prompts.trace_rows(args.trace))
    records = [record for record in records if record.get("control_decision")]
    if args.window:
        selected_windows = set(args.window)
        records = [
            record
            for record in records
            if record["prompt"].get("window") in selected_windows
        ]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error("no completed ordinary routing records found")
    api_key = os.environ.get(args.api_key_env, "")
    metadata = source_metadata(args.trace)
    if not args.dry_run and not api_key:
        parser.error(f"{args.api_key_env} is not set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for record in records:
            for mode in args.mode:
                body = openai_body(record, mode)
                if args.dry_run:
                    result = {
                        "type": "asgard_supervisor_replay_dry_run",
                        "window": record["prompt"].get("window"),
                        "source_mode": record["prompt"].get("mode"),
                        "target_mode": mode,
                        "message_bytes": len(
                            json.dumps(body["messages"], ensure_ascii=False).encode("utf-8")
                        ),
                    }
                else:
                    last_error: Exception | None = None
                    for attempt in range(args.retries):
                        try:
                            response = post_json(args.endpoint, api_key, body, args.timeout)
                            result = score_record(record, mode, response, metadata)
                            break
                        except (OSError, ValueError, urllib.error.HTTPError) as error:
                            last_error = error
                            if attempt + 1 < args.retries:
                                time.sleep(min(8, 2**attempt))
                    else:
                        raise RuntimeError(
                            f"replay failed after {args.retries} attempts: {last_error}"
                        ) from last_error
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
