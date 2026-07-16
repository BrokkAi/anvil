#!/usr/bin/env python3
"""Replay a small panel of archived Asgard supervisor decisions.

This does not run candidates, tools, repositories, or graders. It reconstructs
the final candidate conversations from Anvil's archived LLM trace, supplies the
selected endpoint's production diff and test-file inventory in production
Asgard order, and asks the supervisor model to judge the same terminal decision
under alternate prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


LANE_RE = re.compile(r"anvil-asgard-(?:clones|worktrees)/asgard-(\d+)-")
CANDIDATE_PATH_RE = re.compile(
    r"anvil-asgard-(?:clones|worktrees)/asgard-\d+-[^\\\s\"']+"
)
SELECTED_RE = re.compile(r"\[Asgard selected lane (\d+):")
EXIT_CODE_RE = re.compile(
    r"Exit code: (-?\d+)|Command completed with exit code (-?\d+)"
)
SANDBOX_WARNING = "[WARNING] OS sandbox unavailable"

CONTRACT_EXTRACTION_PROMPT = """You are extracting the explicit contract checklist from a software task description, before any implementation exists. List every externally checkable requirement the task states: function and method signatures including argument order and curried versus non-curried forms, exact strings, separators, suffixes, and output formats, boundary values and their required handling, lifecycle and cancellation obligations (what must unblock, interrupt, close, or clean up what, including operations already in flight), concurrency and atomicity requirements, error types and exact error text, compatibility constraints, and explicit prohibitions. Quote the task's own words wherever possible. One requirement per entry; split compound sentences into separate entries. Do not invent requirements the task does not state and do not add generic quality goals. Tag each entry's kind: "inspection" when a reviewer can verify it by reading the final code (signatures, argument order, exact strings present in source, type shapes); "execution" when verification requires running a scenario (blocking and unblocking, timing, cancellation of in-flight operations, end-to-end produced output, formatting of emitted streams); "delivery" for repository-delivery obligations that do not affect runtime behavior (working on a named branch, committing the work, repository cleanliness). For each entry, if the requirement only holds meaning under a specific adverse condition — an operation already blocked or in flight when the triggering event fires, an exhausted window or resource, an externally stalled dependency, a boundary or zero value — record that condition verbatim in adverse_condition; otherwise set adverse_condition to null. When the task names classification values, categories, or enumerated labels (for example conflict types, error kinds, source values, status codes), add a separate contract that each reported classification carries the correct value for its specific scenario — distinct from, and in addition to, the contract that the item is detected or reported at all; verifying a count or presence does not verify a label. When a requirement combines, maps, constructs from, or dispatches over multiple positional inputs (combining N containers, building a record from several fields, constructor argument lists), add a separate contract that each input's value reaches its correct output position, with adverse_condition stating that the inputs must be pairwise distinguishable and the operation non-commutative — evidence from same-valued or commutative examples cannot detect positional swaps. Success paths get adverse conditions too: when a requirement's happy path can be partially wrong (right for the first element, the common case, or homogeneous inputs), record the discriminating input shape as its adverse_condition rather than leaving it null. When the task requires emitting a well-known textual format or stream whose name implies standard structural conventions (for example a manifest stream with document separators and source annotations, a unified diff, a standard header block), also add one contract per structural convention the named format prescribes — including how the stream begins, how documents or sections are delimited, and how identifying annotations are formed and normalized — marked kind "execution" and quoting the format name from the task. Only do this for formats whose conventions you are certain of; do not invent conventions. Call extract_task_contracts exactly once. Do not answer in prose."""


def archive_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in value
        )
    return "" if value is None else str(value)


def lane_from_request(event: dict[str, Any]) -> int | None:
    match = LANE_RE.search(json.dumps(event.get("messages", []), ensure_ascii=False))
    return int(match.group(1)) if match else None


def response_message(response: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.get("text", ""),
    }
    calls = response.get("tool_calls") or []
    if calls:
        message["tool_calls"] = calls
    return message


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(message))
    raw = json.dumps(normalized, ensure_ascii=False)
    raw = CANDIDATE_PATH_RE.sub("anvil-asgard-clones/asgard-LANE", raw)
    return json.loads(raw)


def common_prefix(histories: list[list[dict[str, Any]]]) -> int:
    width = min(map(len, histories))
    for index in range(width):
        first = normalize_message(histories[0][index])
        if any(normalize_message(history[index]) != first for history in histories[1:]):
            return index
    return width


def render_messages(messages: list[dict[str, Any]]) -> str:
    rendered = []
    for message in messages:
        role = message.get("role", "unknown")
        body: dict[str, Any] = {"content": message.get("content", "")}
        if message.get("tool_calls"):
            body["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            body["tool_call_id"] = message["tool_call_id"]
        rendered.append(
            f'<message role="{role}">\n{json.dumps(body, ensure_ascii=False)}\n</message>'
        )
    return "\n".join(rendered)


def is_test_path(path: str) -> bool:
    lower = path.lower()
    file_name = lower.rsplit("/", 1)[-1]
    return (
        any(
            segment in {"test", "tests", "__tests__", "integrationtest"}
            for segment in lower.split("/")
        )
        or file_name.endswith("_test.go")
        or file_name.endswith("test.java")
        or file_name.endswith("tests.cs")
        or file_name.endswith("test.cs")
        or ".test." in file_name
        or ".spec." in file_name
    )


def patch_surfaces(patch: str) -> tuple[str, str, dict[str, list[str]]]:
    sections = re.split(r"(?=^diff --git a/)", patch, flags=re.MULTILINE)
    production: list[str] = []
    test_sections: list[str] = []
    created_tests: list[str] = []
    modified_tests: list[str] = []
    for section in sections:
        match = re.match(r"diff --git a/(.+?) b/(.+?)\n", section)
        if not match:
            continue
        old_path, new_path = match.groups()
        if is_test_path(old_path) or is_test_path(new_path):
            target = created_tests if "\nnew file mode " in section else modified_tests
            target.append(old_path)
            test_sections.append(section)
        else:
            production.append(section)
    return (
        "".join(production),
        "".join(test_sections),
        {
            "candidate_created_test_files": created_tests,
            "candidate_modified_test_files": modified_tests,
        },
    )


def parse_tool_arguments(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw or "")
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_execution_ledger(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Mechanically walk one lane's full reconstructed history and log every
    run_shell_command / edit / write_file tool call the lane actually issued.
    """
    entries: list[dict[str, Any]] = []
    edit_steps: list[dict[str, Any]] = []
    total_shell_commands = 0
    for step, message in enumerate(history):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            name = function.get("name")
            if name not in ("run_shell_command", "edit", "write_file"):
                continue
            arguments = parse_tool_arguments(function.get("arguments", ""))
            if arguments is None:
                continue
            if name == "run_shell_command":
                total_shell_commands += 1
                command = str(arguments.get("command", ""))
                if len(command) > 500:
                    command = command[:500] + "…"
                call_id = call.get("id")
                result_text = None
                for later in history[step + 1 :]:
                    if later.get("role") == "tool" and later.get("tool_call_id") == call_id:
                        result_text = message_text(later.get("content"))
                        break
                if result_text is None:
                    exit_code: int | None = None
                    output_tail = ""
                else:
                    matches = EXIT_CODE_RE.findall(result_text)
                    if matches:
                        group1, group2 = matches[-1]
                        exit_code = int(group1 or group2)
                    else:
                        exit_code = 0
                    kept_lines = [
                        line
                        for line in result_text.splitlines()
                        if SANDBOX_WARNING not in line
                    ]
                    output_tail = "\n".join(kept_lines)[-400:]
                entries.append(
                    {
                        "id": f"L{total_shell_commands}",
                        "step": step,
                        "command": command,
                        "exit_code": exit_code,
                        "output_tail": output_tail,
                    }
                )
            else:
                edit_steps.append({"step": step, "file": arguments.get("file_path")})

    last_shell_step = entries[-1]["step"] if entries else -1
    files_edited_after_last_command = [
        edit_step["file"] for edit_step in edit_steps if edit_step["step"] > last_shell_step
    ]
    last_edit_step = edit_steps[-1]["step"] if edit_steps else None

    entries_truncated = len(entries) > 120
    display_entries = entries[:20] + entries[-100:] if entries_truncated else entries

    ledger: dict[str, Any] = {
        "entries": display_entries,
        "edit_steps": edit_steps[-150:] if len(edit_steps) > 150 else edit_steps,
        "last_edit_step": last_edit_step,
        "files_edited_after_last_command": files_edited_after_last_command,
        "total_shell_commands": total_shell_commands,
        "entries_truncated": entries_truncated,
        "history_horizon": (
            "reconstructed history ends at the lane's final model turn; results of "
            "tool calls issued in that final turn, and any actions after it, are not "
            "captured in this ledger"
        ),
    }
    if len(edit_steps) > 150:
        ledger["edit_steps_truncated"] = True
    return ledger


def contract_extraction_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "extract_task_contracts",
            "description": (
                "Extract the explicit, externally checkable contract checklist from a "
                "software task description, before any implementation exists."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["contracts"],
                "properties": {
                    "contracts": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 75,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "kind", "text", "adverse_condition"],
                            "properties": {
                                "id": {"type": "string", "pattern": "^C[0-9]+$"},
                                "kind": {
                                    "type": "string",
                                    "enum": ["inspection", "execution", "delivery"],
                                },
                                "text": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 600,
                                },
                                "adverse_condition": {
                                    "type": ["string", "null"],
                                    "maxLength": 400,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def parse_contract_extraction(
    response: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    message = response["choices"][0]["message"]
    for call in message.get("tool_calls") or []:
        if call.get("function", {}).get("name") == "extract_task_contracts":
            return parse_tool_call_json(call["function"]["arguments"]), message
    return None, message


def extract_task_contracts(api_key: str, model: str, instruction: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CONTRACT_EXTRACTION_PROMPT},
        {"role": "user", "content": f"<original_task>\n{instruction}\n</original_task>"},
    ]
    tools = [contract_extraction_tool()]
    response = api_call(api_key, model, messages, tools)
    contracts, assistant = parse_contract_extraction(response)
    if contracts is None:
        append_model_turn(
            messages,
            assistant,
            "You must call extract_task_contracts with valid JSON arguments now. Do not answer in prose.",
        )
        response = api_call(api_key, model, messages, tools)
        contracts, _ = parse_contract_extraction(response)
    if contracts is None:
        raise RuntimeError("contract extraction produced no extract_task_contracts tool call")
    contracts["_usage"] = response.get("usage", {})
    return contracts


def load_task_contracts(
    case_id: str,
    instruction: str,
    *,
    cache_dir: Path,
    refresh: bool,
    api_key: str,
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{case_id}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    if dry_run:
        print(
            f"warning: no contract cache for case {case_id!r}; "
            "inserting placeholder checklist (drop --dry-run to extract)",
            file=sys.stderr,
        )
        return {
            "contracts": [
                {
                    "id": "C0",
                    "kind": "inspection",
                    "text": "(placeholder: run without --dry-run to extract)",
                }
            ]
        }
    contracts = extract_task_contracts(api_key, model, instruction)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(contracts, ensure_ascii=False, indent=2))
    return contracts


def reconstruct_case(
    case: dict[str, Any],
    *,
    selected_only: bool = False,
    evidence_protocol: bool = False,
    contracts_cache: Path | None = None,
    refresh_contracts: bool = False,
    api_key: str = "",
    model: str = "",
    dry_run: bool = False,
) -> tuple[str, dict[str, Any], list[str]]:
    archive = Path(case["archive"])
    with zipfile.ZipFile(archive) as zf:
        instruction = archive_text(zf, "instruction.md")
        patch = archive_text(zf, "model.patch")
        events = [
            json.loads(line)
            for line in archive_text(zf, "anvil-trace.jsonl").splitlines()
            if line.strip()
        ]
        acp_events = [
            json.loads(line)
            for line in archive_text(zf, "mjolnir-events.jsonl").splitlines()
            if line.strip()
        ]

    thought_stream = "".join(
        event.get("text", "")
        for event in acp_events
        if event.get("type") == "agent_thought"
    )
    selected_matches = SELECTED_RE.findall(thought_stream)
    if not selected_matches:
        raise ValueError(f"no selected lane marker in {archive}")
    selected_lane = int(selected_matches[-1]) - 1

    latest: dict[int, list[dict[str, Any]]] = {}
    for index, event in enumerate(events[:-1]):
        if event.get("type") != "llm_request":
            continue
        lane = lane_from_request(event)
        if lane is None or events[index + 1].get("type") != "llm_response":
            continue
        history = list(event.get("messages", []))
        history.append(response_message(events[index + 1].get("response", {})))
        latest[lane] = history
    if len(latest) != 3:
        raise ValueError(f"expected 3 final lane histories in {archive}, got {sorted(latest)}")

    histories = [latest[index] for index in range(3)]
    prefix_len = common_prefix(histories)
    production_patch, test_patch, test_inventory = patch_surfaces(patch)

    checklist_entries: list[dict[str, Any]] = []
    dossier = ["<original_task>", instruction, "</original_task>"]
    if evidence_protocol:
        assert contracts_cache is not None
        task_contracts = load_task_contracts(
            case["id"],
            instruction,
            cache_dir=contracts_cache,
            refresh=refresh_contracts,
            api_key=api_key,
            model=model,
            dry_run=dry_run,
        )
        checklist_entries = task_contracts.get("contracts", [])
        checklist_public = {"contracts": task_contracts.get("contracts", [])}
        dossier.extend(
            [
                '<task_contract_checklist derived_from_task_text_only="true" independent_of_candidates="true">',
                json.dumps(checklist_public, ensure_ascii=False, indent=2),
                "</task_contract_checklist>",
            ]
        )
    dossier.extend(
        [
            "<shared_selected_history>",
            render_messages(histories[0][:prefix_len]),
            "</shared_selected_history>",
        ]
    )
    included_lanes = [selected_lane] if selected_only else list(range(len(histories)))
    for lane in included_lanes:
        history = histories[lane]
        dossier.append(f'<candidate_trajectory lane="{lane}">')
        if lane == selected_lane:
            dossier.extend(
                [
                    '<candidate_test_file_edits derived_from_full_patch="true">',
                    json.dumps(test_inventory, ensure_ascii=False, indent=2),
                    "</candidate_test_file_edits>",
                    '<terminal_non_test_patch cumulative_from_task_baseline="true">',
                    production_patch or "(no non-test changes)",
                    "</terminal_non_test_patch>",
                ]
            )
            if evidence_protocol:
                capped_test_patch = test_patch
                if len(capped_test_patch) > 200_000:
                    capped_test_patch = (
                        capped_test_patch[:200_000]
                        + "\n... test patch truncated (mechanical cap)"
                    )
                dossier.extend(
                    [
                        '<terminal_test_patch cumulative_from_task_baseline="true">',
                        capped_test_patch or "(no test changes)",
                        "</terminal_test_patch>",
                    ]
                )
                ledger = build_execution_ledger(history)
                dossier.extend(
                    [
                        '<execution_ledger mechanically_derived="true" '
                        'source="lane tool calls, not candidate claims">',
                        json.dumps(ledger, ensure_ascii=False, indent=2),
                        "</execution_ledger>",
                    ]
                )
        dossier.extend(
            [
                render_messages(history[prefix_len:]),
                "</candidate_trajectory>",
            ]
        )
    dossier.append(
        "Judge the candidate lanes and the selected endpoint now. The grader verdict is intentionally withheld."
    )
    metadata = {
        "selected_lane_original": selected_lane,
        "selected_only": selected_only,
        "shared_messages": prefix_len,
        "lane_messages": [len(history) - prefix_len for history in histories],
        "dossier_chars": sum(len(part) for part in dossier),
    }
    return "\n".join(dossier), metadata, checklist_entries


def decision_tool(candidate_count: int = 3, *, evidence_protocol: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "winner": {
            "type": "integer",
            "minimum": 0,
            "maximum": candidate_count - 1,
        },
        "complete": {"type": "boolean"},
        "next_window_steps": {"type": "integer", "minimum": 3, "maximum": 12},
        "state_summary": {"type": "string", "minLength": 1},
        "advices": {
            "type": "array",
            "minItems": 0,
            "maxItems": candidate_count,
            "uniqueItems": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["strategy", "scope_basis"],
                "properties": {
                    "strategy": {"type": "string", "minLength": 1},
                    "scope_basis": {"type": "string", "minLength": 1},
                },
            },
        },
    }
    if evidence_protocol:
        properties["contracts"] = {
            "type": "array",
            "maxItems": 90,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "status", "evidence"],
                "properties": {
                    "id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["verified", "violated", "unverified"],
                    },
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 1500},
                    "adverse_condition_evidence": {
                        "type": "string",
                        "maxLength": 1200,
                        "description": (
                            "Required when the checklist entry carries an adverse_condition: "
                            "quote the test or code lines showing that exact condition is "
                            "constructed, and list every action the verifying test performs "
                            "after the triggering event."
                        ),
                    },
                },
            },
        }
    return {
        "type": "function",
        "function": {
            "name": "select_trajectory",
            "description": "Select the canonical Asgard trajectory and decide whether the original task is complete.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["winner", "complete", "state_summary", "advices"],
                "additionalProperties": False,
            },
        },
    }


def api_call(
    api_key: str, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "stream": False,
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {error.code}: {detail}") from error


def parse_tool_call_json(raw: str) -> dict[str, Any] | None:
    """Parse tool-call arguments, tolerating trailing garbage after the object."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value, _ = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def parse_decision(response: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    message = response["choices"][0]["message"]
    for call in message.get("tool_calls") or []:
        if call.get("function", {}).get("name") == "select_trajectory":
            return parse_tool_call_json(call["function"]["arguments"]), message
    return None, message


def append_model_turn(
    messages: list[dict[str, Any]], assistant: dict[str, Any], feedback: str
) -> None:
    """Append the assistant turn plus feedback while keeping the conversation
    valid for the API: an assistant message carrying tool_calls must be answered
    by tool messages (one per tool_call_id) before any other message.
    """
    messages.append(assistant)
    calls = assistant.get("tool_calls") or []
    if calls:
        for call in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": feedback,
                }
            )
    else:
        messages.append({"role": "user", "content": feedback})


def call_decision_tool(
    api_key: str, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Call the API once, retrying a single time if no tool call comes back.

    Mutates `messages` in place with the retry turns so the caller's
    conversation stays consistent across further rounds.
    """
    response = api_call(api_key, model, messages, tools)
    decision, assistant = parse_decision(response)
    for _ in range(2):
        if decision is not None:
            break
        append_model_turn(
            messages,
            assistant,
            "You must call select_trajectory with valid JSON arguments now. Do not answer in prose.",
        )
        response = api_call(api_key, model, messages, tools)
        decision, assistant = parse_decision(response)
    if decision is None:
        raise RuntimeError("supervisor produced no select_trajectory tool call")
    return decision, assistant, response.get("usage", {})


def validate_contract_rows(
    decision: dict[str, Any], checklist: list[dict[str, Any]]
) -> list[str]:
    rows = decision.get("contracts") or []
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict):
            by_id.setdefault(row.get("id"), []).append(row)

    violations: list[str] = []
    for entry in checklist:
        contract_id = entry["id"]
        matches = by_id.get(contract_id, [])
        if not matches:
            violations.append(f"contract {contract_id} is missing")
        elif len(matches) > 1:
            violations.append(f"contract {contract_id} appears more than once")
        else:
            row = matches[0]
            status = row.get("status")
            if entry.get("kind") == "delivery":
                # Delivery-mechanics obligations rank below functional contracts
                # and may sit beyond the ledger's history horizon: a filled,
                # non-violated row is sufficient for complete=true.
                if status == "violated":
                    violations.append(f"contract {contract_id} is violated")
                continue
            if status != "verified":
                violations.append(f"contract {contract_id} is {status}")
            elif not str(row.get("evidence") or "").strip():
                violations.append(f"contract {contract_id} has empty evidence")
            elif entry.get("kind") == "execution" and not re.search(
                r"\bL\d+\b",
                f"{row.get('evidence') or ''} {row.get('adverse_condition_evidence') or ''}",
            ):
                violations.append(
                    f"contract {contract_id} is kind=execution but its row cites no "
                    "execution_ledger entry (L<n>); execution contracts cannot be "
                    "verified by inspection alone — cite the ledger entry that "
                    "exercised this behavior, or mark the row unverified"
                )
            elif entry.get("adverse_condition") and not str(
                row.get("adverse_condition_evidence") or ""
            ).strip():
                violations.append(
                    f"contract {contract_id} carries adverse_condition "
                    f"({entry['adverse_condition'][:120]!r}) but its row has no "
                    "adverse_condition_evidence showing that condition constructed"
                )
    return violations


def violation_message(violations: list[str]) -> str:
    return (
        "complete=true was not accepted: "
        + "; ".join(violations)
        + ". Every checklist contract must have a verified row citing concrete evidence "
        "(execution_ledger entry ids, or quoted code from the terminal patches or trajectory). "
        "If you cannot cite such evidence, return complete=false with advices that direct the "
        "next candidate windows to produce exactly that evidence."
    )


CHALLENGE_MESSAGE = (
    "Before your verdict is accepted, re-adjudicate only the rows that carry "
    "adverse_condition_evidence, using your own stated facts. For each: "
    "(1) The contract's triggering event is X and the pending operation is P. List what "
    "your evidence says the test does after X. If any of those actions — releasing, "
    "closing, erroring, enqueuing, sending, receiving, advancing timers, or completing "
    "the awaited resource — could cause P to resume regardless of X, then the test does "
    "not demonstrate that X unblocks P: mark the row unverified. The order in which "
    "flags were set does not matter; what matters is what made the blocked call return. "
    "(2) If your evidence says P notices a flag after a read or wait returns, or on the "
    "next loop iteration, identify what makes the blocked call return in the cited test. "
    "If the answer is a test action rather than X itself, mark the row unverified. "
    "(3) If the row's evidence is inspection-based, confirm the quoted code shows X "
    "proactively completing, erroring, or waking the specific pending P — a state check "
    "on a later code path does not qualify. "
    "(4) For every row that cites a passing test (not only adverse-condition rows), "
    "state the weakest implementation that would still pass the cited assertions. If "
    "that implementation would violate the contract — a wrong classification that still "
    "produces a nonzero count, a stream missing its first separator that still contains "
    "the substring, a result that is right in the common case only — the evidence is "
    "not discriminating: mark the row unverified. "
    "(5) For every row whose contract names enumerated values, classifications, or "
    "exact labels, quote from the cited test or from ledger output the line showing the "
    "exact required value in each scenario the contract names. An assertion on a "
    "different scenario's value does not transfer. If you cannot produce the quote, "
    "mark the row unverified. "
    "Then call select_trajectory again: either the same verdict with rows you "
    "re-confirmed, or complete=false with corrected rows and advices targeting the "
    "evidence gaps. Do not soften a rule violation because other checks pass."
)


def run_decision(
    api_key: str,
    model: str,
    prompt: str,
    dossier: str,
    *,
    evidence_protocol: bool = False,
    checklist_entries: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int, bool]:
    tools = [decision_tool(evidence_protocol=evidence_protocol)]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": dossier},
    ]
    decision, assistant, usage = call_decision_tool(api_key, model, messages, tools)

    validation_rounds = 0
    challenge_used = False
    checklist_entries = checklist_entries or []
    if evidence_protocol and checklist_entries:
        while True:
            if decision.get("complete") is True:
                violations = validate_contract_rows(decision, checklist_entries)
                if violations:
                    if validation_rounds >= 2:
                        # Mirror live Asgard: a complete=true that cannot produce
                        # valid evidence rows fails closed as incomplete.
                        decision["_failed_validation"] = violations
                        break
                    validation_rounds += 1
                    append_model_turn(messages, assistant, violation_message(violations))
                    decision, assistant, usage = call_decision_tool(
                        api_key, model, messages, tools
                    )
                    continue
                if not challenge_used:
                    challenge_used = True
                    append_model_turn(messages, assistant, CHALLENGE_MESSAGE)
                    decision, assistant, usage = call_decision_tool(
                        api_key, model, messages, tools
                    )
                    continue
            break

    return decision, usage, validation_rounds, challenge_used


def print_summary(summary: dict[tuple[str, str], dict[str, Any]]) -> None:
    if not summary:
        return
    print("case | prompt | expected_complete | votes complete=true/total | agree_fraction", file=sys.stderr)
    for (case_id, prompt_name), stats in summary.items():
        total = stats["total"]
        votes_true = stats["votes_true"]
        agree = stats["agree"]
        agree_fraction = f"{agree}/{total}" if total else "n/a"
        print(
            f"{case_id} | {prompt_name} | {stats['expected']} | {votes_true}/{total} | {agree_fraction}",
            file=sys.stderr,
        )


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "asgard_supervisor_panel.json")
    parser.add_argument("--prompt", action="append", type=Path)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--evidence-protocol", action="store_true")
    parser.add_argument(
        "--contracts-cache", type=Path, default=root / "asgard_contract_cache"
    )
    parser.add_argument("--refresh-contracts", action="store_true")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--dump-dossier", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = [
        case for case in manifest["cases"]
        if not args.case_ids or case["id"] in args.case_ids
    ]
    prompts = args.prompt or [
        root / "asgard_supervisor_prompts" / "baseline.txt",
        root / "asgard_supervisor_prompts" / "adversarial.txt",
    ]
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if args.api_key_file:
        api_key = args.api_key_file.read_text().strip()
    if not args.dry_run and not api_key:
        parser.error("set DEEPSEEK_API_KEY or pass --api-key-file")

    if args.dump_dossier:
        args.dump_dossier.mkdir(parents=True, exist_ok=True)

    summary: dict[tuple[str, str], dict[str, Any]] = {}
    output = args.output.open("a") if args.output else sys.stdout
    try:
        for case in cases:
            dossier, metadata, checklist_entries = reconstruct_case(
                case,
                selected_only=args.selected_only,
                evidence_protocol=args.evidence_protocol,
                contracts_cache=args.contracts_cache,
                refresh_contracts=args.refresh_contracts,
                api_key=api_key,
                model=args.model,
                dry_run=args.dry_run,
            )
            if args.dump_dossier:
                (args.dump_dossier / f"{case['id']}.txt").write_text(dossier)
            for prompt_path in prompts:
                samples = args.samples if args.evidence_protocol else 1
                for sample_index in range(1, samples + 1):
                    result: dict[str, Any] = {
                        "case": case["id"],
                        "prompt": prompt_path.stem,
                        "expected_complete": case["expected_complete"],
                        "review_note": case["review_note"],
                        **metadata,
                    }
                    if args.evidence_protocol:
                        result["sample"] = sample_index
                    if not args.dry_run:
                        try:
                            decision, usage, validation_rounds, challenged = run_decision(
                                api_key,
                                args.model,
                                prompt_path.read_text(),
                                dossier,
                                evidence_protocol=args.evidence_protocol,
                                checklist_entries=checklist_entries,
                            )
                        except (RuntimeError, ValueError, KeyError, urllib.error.URLError, TimeoutError, OSError) as error:
                            result["error"] = str(error)
                            print(json.dumps(result, ensure_ascii=False), file=output, flush=True)
                            print(
                                f"[error] {case['id']} / {prompt_path.stem} sample {sample_index}: {error}",
                                file=sys.stderr,
                            )
                            continue
                        # A complete=true that never produced valid evidence rows
                        # fails closed as incomplete for scoring purposes.
                        effective_complete = decision.get("complete") is True and not decision.get(
                            "_failed_validation"
                        )
                        result["decision"] = decision
                        result["effective_complete"] = effective_complete
                        result["correct_complete"] = (
                            effective_complete == case["expected_complete"]
                        )
                        result["usage"] = usage
                        if args.evidence_protocol:
                            result["validation_rounds"] = validation_rounds
                            result["challenged"] = challenged

                        key = (case["id"], prompt_path.stem)
                        stats = summary.setdefault(
                            key,
                            {"expected": case["expected_complete"], "votes_true": 0, "total": 0, "agree": 0},
                        )
                        stats["total"] += 1
                        if effective_complete:
                            stats["votes_true"] += 1
                        if result["correct_complete"]:
                            stats["agree"] += 1
                    print(json.dumps(result, ensure_ascii=False), file=output, flush=True)
    finally:
        if output is not sys.stdout:
            output.close()
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
