#!/usr/bin/env python3
"""Run the same summarization prompts against an OpenAI-compatible batch server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = """You are writing a compact, loss-aware handoff of one completed Asgard candidate work window to a trajectory supervisor. Report what the window actually established, not what it hoped to accomplish. Preserve adverse evidence, failed checks, incomplete work, and consequential uncertainty. Summarize source edits semantically by file or symbol; do not reproduce patches or routine narration. Return only one JSON object with exactly these keys: direction (string), progress (string), edits (array of objects with location and change strings), evidence (array of objects with check, status, and details strings; status is passed, failed, or inconclusive), unresolved_risks (array of strings), and next_step (string)."""
READ_ONLY_GUARD = """This captured window is positively classified read-only: it contains only known non-writing tool calls. Set edits to an empty array. Reads and verification may expose changes made before this window, but do not attribute those changes to this window."""

SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["direction", "progress", "edits", "evidence", "unresolved_risks", "next_step"],
    "properties": {
        "direction": {"type": "string"},
        "progress": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["location", "change"],
                "properties": {"location": {"type": "string"}, "change": {"type": "string"}},
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["check", "status", "details"],
                "properties": {
                    "check": {"type": "string"},
                    "status": {"type": "string", "enum": ["passed", "failed", "inconclusive"]},
                    "details": {"type": "string"},
                },
            },
        },
        "unresolved_risks": {"type": "array", "items": {"type": "string"}},
        "next_step": {"type": "string"},
    },
}


def prompt(row: dict, read_only_guard: bool) -> list[dict]:
    system = SYSTEM_PROMPT
    if read_only_guard and row.get("activity_class") == "read-only":
        system += "\n\n" + READ_ONLY_GUARD
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "ORIGINAL TASK (complete):\n" + row["task_text"]},
        {
            "role": "user",
            "content": "<candidate_window>\n" + row["window_text"] + "\n</candidate_window>",
        },
    ]


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(url: str, model: str, *, prompt_text: str | None = None, messages: list[dict] | None = None, timeout: float) -> dict:
    payload: dict = {"model": model}
    if messages is not None:
        payload["messages"] = messages
    else:
        payload["prompt"] = prompt_text
    request = urllib.request.Request(
        url.rstrip("/") + "/tokenize",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_summary(text: str) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return None, str(error)
    required = {"direction", "progress", "edits", "evidence", "unresolved_risks", "next_step"}
    if set(value) != required:
        return None, f"keys were {sorted(value)}"
    return value, None


async def main_async(args: argparse.Namespace) -> None:
    rows = [
        json.loads(line)
        for path in args.dataset
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    completed: set[str] = set()
    if args.output.exists():
        prior = [
            json.loads(line) for line in args.output.read_text().splitlines() if line.strip()
        ]
        completed = {
            row["id"]
            for row in prior
            if not args.retry_invalid
            or (not row.get("error") and not row.get("validation_error"))
        }
    semaphore = asyncio.Semaphore(args.concurrency)
    output_lock = asyncio.Lock()

    async def run_one(row: dict) -> None:
        if row["id"] in completed:
            return
        async with semaphore:
            started = time.monotonic()
            messages = prompt(row, args.read_only_guard)
            payload: dict = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "candidate_window_summary", "strict": True, "schema": SUMMARY_SCHEMA},
                },
            }
            source_tokens = prompt_tokens = max_tokens = None
            if args.max_tokens is not None:
                max_tokens = args.max_tokens
            elif args.max_output_input_ratio is not None:
                source_info, prompt_info = await asyncio.gather(
                    asyncio.to_thread(
                        tokenize,
                        args.url,
                        args.model,
                        prompt_text=row["window_text"],
                        timeout=args.timeout,
                    ),
                    asyncio.to_thread(
                        tokenize,
                        args.url,
                        args.model,
                        messages=messages,
                        timeout=args.timeout,
                    ),
                )
                source_tokens = source_info["count"]
                prompt_tokens = prompt_info["count"]
                remaining_context = prompt_info["max_model_len"] - prompt_tokens
                max_tokens = max(
                    1,
                    min(
                        math.ceil(prompt_tokens * args.max_output_input_ratio),
                        remaining_context,
                    ),
                )
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            try:
                response = await asyncio.to_thread(post, args.url, payload, args.timeout)
                content = response["choices"][0]["message"]["content"]
                summary, validation_error = validate_summary(content)
                result = {
                    "id": row["id"],
                    "model": args.model,
                    "latency_seconds": time.monotonic() - started,
                    "usage": response.get("usage"),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "budget": {
                        "source_tokens": source_tokens,
                        "prompt_tokens": prompt_tokens,
                        "max_tokens": max_tokens,
                    },
                    "summary": summary,
                    "raw_output": content,
                    "validation_error": validation_error,
                    "error": None,
                }
            except Exception as error:  # preserve every failed row for diagnosis/resume
                result = {
                    "id": row["id"],
                    "model": args.model,
                    "latency_seconds": time.monotonic() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            async with output_lock:
                with args.output.open("a") as stream:
                    stream.write(json.dumps(result, separators=(",", ":")) + "\n")
                print(json.dumps({"id": row["id"], "error": result.get("error"), "seconds": round(result["latency_seconds"], 2)}), flush=True)

    await asyncio.gather(*(run_one(row) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="fixed completion ceiling overriding source-sized budgeting",
    )
    parser.add_argument(
        "--max-output-input-ratio",
        type=float,
        default=1.0,
        help="completion ceiling as a multiple of the exact tokenized summarizer input (default: 1.0)",
    )
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--limit", type=int, help="run only the first N rows (useful for smoke tests)")
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="skip successful rows but retry prior request/schema failures",
    )
    parser.add_argument(
        "--read-only-guard",
        action="store_true",
        help="tell the model to emit no edits for positively classified read-only windows",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
