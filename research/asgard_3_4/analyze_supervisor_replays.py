#!/usr/bin/env python3
"""Strict offline analysis for ``asgard_supervisor_replay_result`` JSONL.

Fallbacks are scored as effective agreement because the replay protocol sends
them back to the captured full-control supervisor. They are never included in
non-fallback agreement. Obligation preservation is intentionally not inferred.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RECORD_TYPE = "asgard_supervisor_replay_result"
USAGE_KEYS = ("input", "cachedRead", "cachedWrite", "output", "thought")
DECISION_FIELDS = {
    "winner": "winner",
    "completion": "complete",
    "next_candidate_count": "next_candidate_count",
    "next_step_count": "next_window_steps",
}
REDUCTION_THRESHOLD = 0.25
WINNER_THRESHOLD = 0.95
PROTECTED_ENDPOINT_THRESHOLD = 1.0


def discover_jsonl(paths: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(item for item in path.rglob("*.jsonl") if item.is_file())
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError(f"path does not exist: {path}")
    return sorted(files)


def read_replay_results(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in discover_jsonl(paths):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            if row.get("type") != RECORD_TYPE:
                continue
            copied = dict(row)
            copied["_source"] = str(path)
            copied["_line"] = line_number
            validate_record(copied)
            records.append(copied)
    if not records:
        raise ValueError("no asgard_supervisor_replay_result records found")
    return records


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decision_value(decision: dict[str, Any], field: str) -> Any:
    return decision.get(field)


def validate_record(row: dict[str, Any]) -> None:
    location = f"{row.get('_source', '<memory>')}:{row.get('_line', '?')}"
    mode = row.get("target_mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"{location}: target_mode must be a non-empty string")
    fallback = row.get("fallback_required")
    if not isinstance(fallback, bool):
        raise ValueError(f"{location}: fallback_required must be boolean")
    control = row.get("control_decision")
    if not isinstance(control, dict):
        raise ValueError(f"{location}: control_decision must be an object")
    if (
        not _is_int(control.get("winner"))
        or control["winner"] < 0
        or not isinstance(control.get("complete"), bool)
    ):
        raise ValueError(f"{location}: control winner/complete must be typed")
    for field in ("next_candidate_count", "next_window_steps"):
        value = control.get(field)
        if value is not None and (not _is_int(value) or value < 1):
            raise ValueError(f"{location}: control {field} must be positive or null")
    status = row.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"{location}: status must be a non-empty string")
    selection = row.get("selection")
    if fallback:
        if selection is not None:
            raise ValueError(f"{location}: fallback record must have null selection")
        if status == "selected":
            raise ValueError(f"{location}: selected status cannot require fallback")
    else:
        if status != "selected":
            raise ValueError(f"{location}: non-fallback record must have selected status")
        if not isinstance(selection, dict):
            raise ValueError(f"{location}: non-fallback selection must be an object")
        if (
            not _is_int(selection.get("winner"))
            or selection["winner"] < 0
            or not isinstance(selection.get("complete"), bool)
        ):
            raise ValueError(f"{location}: selection winner/complete must be typed")
        for field in ("next_candidate_count", "next_window_steps"):
            value = selection.get(field)
            if value is not None and (not _is_int(value) or value < 1):
                raise ValueError(f"{location}: selection {field} must be positive or null")
        expected = {
            "winner_agreement": selection.get("winner") == control.get("winner"),
            "complete_agreement": selection.get("complete") == control.get("complete"),
            "candidate_count_agreement": selection.get("next_candidate_count")
            == control.get("next_candidate_count"),
            "step_count_agreement": selection.get("next_window_steps")
            == control.get("next_window_steps"),
        }
        for field, derived in expected.items():
            if field in row and row[field] is not None and row[field] is not derived:
                raise ValueError(f"{location}: {field} contradicts decisions")
    usage = row.get("usage")
    if not isinstance(usage, dict):
        raise ValueError(f"{location}: usage must be an object")
    for key in USAGE_KEYS:
        value = usage.get(key, 0)
        if not _is_int(value) or value < 0:
            raise ValueError(f"{location}: usage.{key} must be a non-negative integer")


def _zero_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def _usage_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = _zero_usage()
    for row in rows:
        usage = row.get("usage") or {}
        for key in USAGE_KEYS:
            total[key] += int(usage.get(key, 0))
    eligible = total["input"] + total["cachedRead"]
    raw = eligible + total["cachedWrite"]
    count = len(rows)
    return {
        **total,
        "rawInput": raw,
        "cacheEligibleInput": eligible,
        "cachedReadFraction": total["cachedRead"] / eligible if eligible else None,
        "uncachedFraction": total["input"] / eligible if eligible else None,
        "cachedWriteFractionOfRaw": total["cachedWrite"] / raw if raw else None,
        "perRecord": {
            key: total[key] / count if count else None for key in (*USAGE_KEYS,)
        },
    }


def _agreement(matches: int, compared: int) -> dict[str, Any]:
    return {
        "matches": matches,
        "compared": compared,
        "rate": matches / compared if compared else None,
    }


def _delta_summary(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
        "histogram": dict(sorted(Counter(str(value) for value in values).items())),
        "sign_convention": "replay selection minus full-control decision",
    }


def _identity(row: dict[str, Any]) -> str | None:
    for key in ("protected_identity", "task_run_identity", "identity"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    endpoint = row.get("protected_endpoint")
    if isinstance(endpoint, str) and endpoint:
        return endpoint
    if isinstance(endpoint, dict):
        for key in ("protected_identity", "identity", "task_run_identity"):
            value = endpoint.get(key)
            if isinstance(value, str) and value:
                return value
    task = row.get("task_id", row.get("task"))
    run = row.get("run")
    if isinstance(task, str) and task and _is_int(run) and run > 0:
        return f"{task}::r{run}"
    return None


def load_protected_identities(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values: list[dict[str, Any]] = []
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error.msg}") from error
            if isinstance(item, dict):
                values.append(item)
    else:
        if isinstance(parsed, dict):
            values = [parsed]
        elif isinstance(parsed, list):
            values = [item for item in parsed if isinstance(item, dict)]
        else:
            raise ValueError(f"{path}: protected corpus is not an object/list/JSONL")
    identities: set[str] = set()
    for value in values:
        listed = value.get("protected_identities")
        if isinstance(listed, list):
            identities.update(item for item in listed if isinstance(item, str))
        identity = value.get("protected_identity")
        if isinstance(identity, str):
            identities.add(identity)
    return identities


def _is_protected(row: dict[str, Any], identities: set[str]) -> bool:
    for key in ("protected_endpoint", "is_protected_endpoint", "protected"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return bool(value)
        if isinstance(value, dict):
            enabled = value.get("protected", value.get("is_protected"))
            if isinstance(enabled, bool):
                return enabled
            if _identity(row):
                return True
    identity = _identity(row)
    return bool(identity and (identity in identities or row.get("protected_identity")))


def _record_reference(row: dict[str, Any]) -> dict[str, Any]:
    selection = row.get("selection") or {}
    control = row["control_decision"]
    return {
        "record_id": row.get("record_id")
        or f"{row.get('_source', '<memory>')}:{row.get('_line', '?')}",
        "source": row.get("_source"),
        "line": row.get("_line"),
        "identity": _identity(row),
        "window": row.get("window"),
        "status": row.get("status"),
        "control": {
            field: control.get(field)
            for field in ("winner", "complete", "next_candidate_count", "next_window_steps")
        },
        "selection": {
            field: selection.get(field)
            for field in ("winner", "complete", "next_candidate_count", "next_window_steps")
        },
    }


def _lookup(mapping: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value: Any = mapping
        for part in path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _paired_reduction(
    rows: list[dict[str, Any]],
    chosen_paths: tuple[tuple[str, ...], ...],
    control_paths: tuple[tuple[str, ...], ...],
    metric: str,
) -> dict[str, Any] | None:
    chosen = 0.0
    control = 0.0
    covered = 0
    for row in rows:
        left = _lookup(row, *chosen_paths)
        right = _lookup(row, *control_paths)
        if left is None or right is None:
            continue
        chosen += left
        control += right
        covered += 1
    if not covered or control <= 0:
        return None
    return {
        "metric": metric,
        "fraction": 1 - chosen / control,
        "source": "replay result paired fields",
        "records_covered": covered,
        "chosen": chosen,
        "full_control": control,
    }


def _direct_fraction(rows: list[dict[str, Any]], names: tuple[str, ...], metric: str) -> dict[str, Any] | None:
    values: list[float] = []
    for row in rows:
        for name in names:
            value = _lookup(
                row,
                (name,),
                ("prompt", name),
                ("prompt_metrics", name),
            )
            if value is not None:
                values.append(value)
                break
    if not values:
        return None
    return {
        "metric": metric,
        "fraction": sum(values) / len(values),
        "source": "mean replay-result fraction fields",
        "records_covered": len(values),
    }


def _pilot_reductions(pilot: dict[str, Any] | None, mode: str) -> list[dict[str, Any]]:
    if not pilot:
        return []
    modes = pilot.get("modes")
    mode_row = modes.get(mode) if isinstance(modes, dict) else None
    if not isinstance(mode_row, dict):
        return []
    evidence: list[dict[str, Any]] = []
    for name, metric in (
        ("raw_input_reduction_fraction", "raw_input_tokens"),
        ("prompt_byte_reduction_fraction", "prompt_bytes"),
        ("estimated_token_reduction_fraction", "estimated_request_tokens"),
    ):
        value = _lookup(mode_row, (name,), ("ordinary_routing", name))
        if value is not None:
            evidence.append(
                {
                    "metric": metric,
                    "fraction": value,
                    "source": "pilot report mode aggregate",
                }
            )
    full_row = modes.get("full") if isinstance(modes, dict) else None
    if isinstance(full_row, dict):
        chosen_raw = _lookup(mode_row, ("ordinary_usage", "rawInput"))
        full_raw = _lookup(full_row, ("ordinary_usage", "rawInput"))
        chosen_windows = _lookup(mode_row, ("ordinary_windows",))
        full_windows = _lookup(full_row, ("ordinary_windows",))
        if (
            chosen_raw is not None
            and full_raw is not None
            and full_raw > 0
            and chosen_windows == full_windows
        ):
            evidence.append(
                {
                    "metric": "raw_input_tokens",
                    "fraction": 1 - chosen_raw / full_raw,
                    "source": "pilot mode totals with equal ordinary-window counts",
                }
            )
    return evidence


def reduction_evidence(
    rows: list[dict[str, Any]], pilot: dict[str, Any] | None, mode: str
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for item in (
        _paired_reduction(
            rows,
            (("raw_input_tokens",), ("rawInput",), ("usage", "rawInput")),
            (("full_control_raw_input_tokens",), ("full_control_rawInput",)),
            "raw_input_tokens",
        ),
        _paired_reduction(
            rows,
            (("prompt_bytes",), ("message_bytes",), ("prompt", "prompt_bytes")),
            (("full_control_prompt_bytes",), ("prompt", "full_control_prompt_bytes")),
            "prompt_bytes",
        ),
        _paired_reduction(
            rows,
            (("estimated_request_tokens",),),
            (("full_control_estimated_request_tokens",),),
            "estimated_request_tokens",
        ),
        _direct_fraction(rows, ("raw_input_reduction_fraction",), "raw_input_tokens"),
        _direct_fraction(
            rows,
            ("prompt_byte_reduction_fraction", "prompt_reduction_fraction"),
            "prompt_bytes",
        ),
        _direct_fraction(
            rows, ("estimated_token_reduction_fraction",), "estimated_request_tokens"
        ),
    ):
        if item is not None:
            evidence.append(item)
    evidence.extend(_pilot_reductions(pilot, mode))
    valid = [
        item
        for item in evidence
        if math.isfinite(item["fraction"]) and item["fraction"] <= 1
    ]
    conservative = min(valid, key=lambda item: item["fraction"]) if valid else None
    return {
        "evidence": evidence,
        "gate_fraction": conservative["fraction"] if conservative else None,
        "gate_metric": conservative["metric"] if conservative else None,
        "gate_source": conservative["source"] if conservative else None,
        "selection_rule": "minimum supplied reduction fraction (fail closed)",
    }


def _gate(name: str, value: float | None, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "operator": ">=",
        "status": "unknown" if value is None else ("pass" if value >= threshold else "fail"),
    }


def summarize_mode(
    mode: str,
    rows: list[dict[str, Any]],
    protected_identities: set[str],
    pilot: dict[str, Any] | None,
) -> dict[str, Any]:
    fallback_rows = [row for row in rows if row["fallback_required"]]
    selected_rows = [row for row in rows if not row["fallback_required"]]
    nonfallback_matches: dict[str, int] = defaultdict(int)
    effective_matches: dict[str, int] = defaultdict(int)
    candidate_deltas: list[int] = []
    step_deltas: list[int] = []
    disagreement_rows: list[dict[str, Any]] = []
    protected_rows = [row for row in rows if _is_protected(row, protected_identities)]
    protected_effective_matches = 0

    for row in rows:
        control = row["control_decision"]
        selection = row.get("selection")
        if row["fallback_required"]:
            for label in DECISION_FIELDS:
                effective_matches[label] += 1
            effective_matches["endpoint"] += 1
            if row in protected_rows:
                protected_effective_matches += 1
            continue
        assert isinstance(selection, dict)
        mismatches: list[str] = []
        for label, field in DECISION_FIELDS.items():
            matches = _decision_value(selection, field) == _decision_value(control, field)
            nonfallback_matches[label] += int(matches)
            effective_matches[label] += int(matches)
            if not matches:
                mismatches.append(label)
        endpoint_match = (
            selection.get("winner") == control.get("winner")
            and selection.get("complete") == control.get("complete")
        )
        nonfallback_matches["endpoint"] += int(endpoint_match)
        effective_matches["endpoint"] += int(endpoint_match)
        if row in protected_rows:
            protected_effective_matches += int(endpoint_match)
        for field, target in (
            ("next_candidate_count", candidate_deltas),
            ("next_window_steps", step_deltas),
        ):
            left, right = selection.get(field), control.get(field)
            if _is_int(left) and _is_int(right):
                target.append(left - right)
        if mismatches:
            reference = _record_reference(row)
            reference["disagreement_fields"] = mismatches
            disagreement_rows.append(reference)

    reduction = reduction_evidence(rows, pilot, mode)
    winner_effective = _agreement(effective_matches["winner"], len(rows))
    protected_agreement = _agreement(
        protected_effective_matches, len(protected_rows)
    )
    gates = {
        "input_reduction": {
            **_gate(
                "prompt/raw input reduction",
                reduction["gate_fraction"],
                REDUCTION_THRESHOLD,
            ),
            "metric": reduction["gate_metric"],
            "source": reduction["gate_source"],
        },
        "overall_winner_agreement": _gate(
            "effective winner agreement", winner_effective["rate"], WINNER_THRESHOLD
        ),
        "protected_endpoint_agreement": {
            **_gate(
                "effective protected endpoint agreement",
                protected_agreement["rate"],
                PROTECTED_ENDPOINT_THRESHOLD,
            ),
            "protected_records": len(protected_rows),
        },
        "obligation_preservation_review": {
            "status": "manual_required",
            "automatically_inferred": False,
            "reason": (
                "winner/completion agreement cannot establish that every decisive "
                "evidence obligation survived compression"
            ),
        },
    }
    automated = [
        gates["input_reduction"]["status"],
        gates["overall_winner_agreement"]["status"],
        gates["protected_endpoint_agreement"]["status"],
    ]
    return {
        "mode": mode,
        "records": len(rows),
        "fallback": {
            "count": len(fallback_rows),
            "rate": len(fallback_rows) / len(rows),
            "status_counts": dict(sorted(Counter(str(row.get("status")) for row in fallback_rows).items())),
            "rows": [_record_reference(row) for row in fallback_rows],
        },
        "nonfallback": {
            "records": len(selected_rows),
            "winner_agreement": _agreement(nonfallback_matches["winner"], len(selected_rows)),
            "completion_agreement": _agreement(
                nonfallback_matches["completion"], len(selected_rows)
            ),
            "endpoint_agreement": _agreement(nonfallback_matches["endpoint"], len(selected_rows)),
            "next_candidate_count_agreement": _agreement(
                nonfallback_matches["next_candidate_count"], len(selected_rows)
            ),
            "next_step_count_agreement": _agreement(
                nonfallback_matches["next_step_count"], len(selected_rows)
            ),
            "next_candidate_count_signed_delta": _delta_summary(candidate_deltas),
            "next_step_count_signed_delta": _delta_summary(step_deltas),
        },
        "effective_with_full_fallback": {
            "assumption": "fallback executes the captured full-control decision",
            "winner_agreement": winner_effective,
            "completion_agreement": _agreement(effective_matches["completion"], len(rows)),
            "endpoint_agreement": _agreement(effective_matches["endpoint"], len(rows)),
            "next_candidate_count_agreement": _agreement(
                effective_matches["next_candidate_count"], len(rows)
            ),
            "next_step_count_agreement": _agreement(
                effective_matches["next_step_count"], len(rows)
            ),
        },
        "protected": {
            "records": len(protected_rows),
            "effective_endpoint_agreement": protected_agreement,
            "identities": sorted(
                {identity for row in protected_rows if (identity := _identity(row))}
            ),
        },
        "usage": {
            "all": _usage_rollup(rows),
            "nonfallback": _usage_rollup(selected_rows),
            "fallback": _usage_rollup(fallback_rows),
        },
        "prompt_or_raw_input_reduction": reduction,
        "disagreements": {
            "count": len(disagreement_rows),
            "rows": disagreement_rows,
            "windows": [
                {"identity": row.get("identity"), "window": row.get("window")}
                for row in disagreement_rows
            ],
        },
        "gates": gates,
        "automated_gates_pass": all(status == "pass" for status in automated),
        "recommendation_ready": False,
        "recommendation_status": (
            "manual-obligation-review-required"
            if all(status == "pass" for status in automated)
            else "automated-gates-not-satisfied"
        ),
    }


def summarize(
    records: list[dict[str, Any]],
    protected_identities: set[str] | None = None,
    pilot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        validate_record(record)
        by_mode[record["target_mode"]].append(record)
    identities = protected_identities or set()
    modes = {
        mode: summarize_mode(mode, rows, identities, pilot)
        for mode, rows in sorted(by_mode.items())
    }
    return {
        "type": "asgard_supervisor_replay_analysis",
        "records": len(records),
        "modes": modes,
        "gate_policy": {
            "input_reduction_minimum": REDUCTION_THRESHOLD,
            "effective_winner_agreement_minimum": WINNER_THRESHOLD,
            "protected_endpoint_agreement_minimum": PROTECTED_ENDPOINT_THRESHOLD,
            "fallback_semantics": "effective agreement delegates to full control",
            "obligation_preservation": "manual review required",
        },
        "all_automated_gates_pass": all(
            row["automated_gates_pass"] for row in modes.values()
        ),
        "recommendation_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="+", type=Path, help="replay JSONL file or directory")
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--protected-corpus", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--require-automated-gates",
        action="store_true",
        help="exit 3 unless every mode passes all three automated Q3 gates",
    )
    args = parser.parse_args(argv)
    try:
        records = read_replay_results(args.path)
        pilot = json.loads(args.pilot_report.read_text()) if args.pilot_report else None
        if pilot is not None and not isinstance(pilot, dict):
            raise ValueError("pilot report must contain a JSON object")
        protected = load_protected_identities(args.protected_corpus)
        report = summarize(records, protected, pilot)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    if args.require_automated_gates and not report["all_automated_gates_pass"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
