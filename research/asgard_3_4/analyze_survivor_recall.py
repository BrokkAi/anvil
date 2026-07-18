#!/usr/bin/env python3
"""Score blinded Asgard shallow-probe survivor-recall studies.

The input is either an Anvil result archive containing ``anvil-trace.jsonl`` or
an uncompressed JSONL trace.  Only complete, isolated, blinded studies count
toward the default top-2 recall gate.  Partial studies remain visible in the
report so that a sampled killed lane cannot accidentally be presented as full
ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


USAGE_KEYS = ("input", "output", "thought", "cachedRead", "cachedWrite")
RECORD_TYPES = {
    "asgard_shadow_tournament_config",
    "asgard_shadow_probe_ranking",
    "asgard_shadow_continuation",
    "asgard_shadow_end_review",
}


def discover_inputs(paths: Iterable[Path]) -> list[Path]:
    inputs: set[Path] = set()
    for path in paths:
        if path.is_dir():
            inputs.update(path.rglob("*.zip"))
            inputs.update(path.rglob("*.jsonl"))
        elif path.suffix in {".zip", ".jsonl"}:
            inputs.add(path)
        else:
            raise ValueError(f"not a .zip, .jsonl, or directory: {path}")
    return sorted(inputs)


def _parse_rows(raw: str, source: str) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: {error.msg}") from error
        if isinstance(value, dict) and value.get("type") in RECORD_TYPES:
            yield value


def trace_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                raw = archive.read("anvil-trace.jsonl").decode("utf-8", "replace")
        except KeyError as error:
            raise ValueError(f"{path} has no anvil-trace.jsonl") from error
        return list(_parse_rows(raw, f"{path}:anvil-trace.jsonl"))
    return list(_parse_rows(path.read_text(encoding="utf-8"), str(path)))


def public_outcome(path: Path) -> dict[str, Any]:
    if path.suffix != ".zip":
        return {"reward": None, "partial": None, "stop_reason": None}
    with zipfile.ZipFile(path) as archive:
        def member(name: str) -> dict[str, Any]:
            try:
                value = json.loads(archive.read(name))
            except KeyError:
                return {}
            return value if isinstance(value, dict) else {}

        result = member("result.json")
        reward = member("reward.json")
    nested_reward = result.get("reward")
    nested_reward = nested_reward if isinstance(nested_reward, dict) else {}
    return {
        "reward": reward.get("reward", nested_reward.get("reward")),
        "partial": reward.get("partial", nested_reward.get("partial")),
        "stop_reason": result.get("stopReason"),
    }


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("usage must be an object")
    result: dict[str, int] = {}
    for key in USAGE_KEYS:
        if key not in value:
            raise ValueError(f"usage is missing {key}")
        amount = value.get(key, 0)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError(f"invalid usage.{key}: {amount!r}")
        result[key] = amount
    return result


def _add_usage(total: dict[str, int], value: Any) -> None:
    for key, amount in _usage(value).items():
        total[key] += amount


def _zero_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def _raw_input(value: dict[str, int]) -> int:
    return value["input"] + value["cachedRead"] + value["cachedWrite"]


def _sum_usage(values: Iterable[dict[str, int]]) -> dict[str, int]:
    total = _zero_usage()
    for value in values:
        _add_usage(total, value)
    return total


def _ranked_ids(rows: Any, key: str) -> list[Any]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("ranking must be a non-empty list")
    normalized: list[tuple[int, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ranking rows must be objects")
        rank = row.get("rank")
        value = row.get(key)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError(f"invalid rank: {rank!r}")
        normalized.append((rank, value))
    normalized.sort()
    expected = list(range(1, len(normalized) + 1))
    if [rank for rank, _ in normalized] != expected:
        raise ValueError("ranking must contain contiguous ranks starting at 1")
    values = [value for _, value in normalized]
    if len(set(values)) != len(values):
        raise ValueError(f"ranking contains duplicate {key} values")
    return values


def _one(records: list[dict[str, Any]], record_type: str) -> dict[str, Any]:
    matches = [row for row in records if row.get("type") == record_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {record_type}, found {len(matches)}")
    return matches[0]


def score_study(
    source: str, study_id: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    config = _one(records, "asgard_shadow_tournament_config")
    probe = _one(records, "asgard_shadow_probe_ranking")
    review = _one(records, "asgard_shadow_end_review")
    continuations = [
        row for row in records if row.get("type") == "asgard_shadow_continuation"
    ]
    violations: list[str] = []

    candidate_count = config.get("candidate_count")
    probe_steps = config.get("probe_steps")
    continuation_steps = config.get("continuation_steps")
    top_k = config.get("top_k")
    valid_candidate_count = (
        isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count in range(2, 6)
    )
    lanes = list(range(candidate_count)) if valid_candidate_count else []
    if not valid_candidate_count:
        violations.append("candidate_count must be between 2 and 5")
    valid_probe_steps = (
        isinstance(probe_steps, int)
        and not isinstance(probe_steps, bool)
        and probe_steps in {1, 2}
    )
    if not valid_probe_steps:
        violations.append("probe_steps must be 1 or 2")
    if (
        not isinstance(continuation_steps, int)
        or isinstance(continuation_steps, bool)
        or continuation_steps < 1
    ):
        violations.append("continuation_steps must be positive")
    valid_top_k = (
        isinstance(top_k, int)
        and not isinstance(top_k, bool)
        and 0 < top_k < len(lanes)
    )
    if not valid_top_k:
        violations.append("top_k must leave at least one killed lane")

    try:
        probe_ranking = _ranked_ids(probe.get("ranking"), "lane")
    except ValueError as error:
        violations.append(f"probe ranking: {error}")
        probe_ranking = []
    if any(not isinstance(lane, int) or isinstance(lane, bool) for lane in probe_ranking):
        violations.append("probe ranking lane ids must be integers")
    if set(probe_ranking) != set(lanes):
        violations.append("probe ranking must cover every configured lane")
    survivors = probe.get("survivors")
    killed = probe.get("killed")
    expected_survivors = probe_ranking[:top_k] if valid_top_k else []
    if not isinstance(survivors, list) or survivors != expected_survivors:
        violations.append("survivors must be the first top_k probe-ranked lanes")
        survivors = []
    if not isinstance(killed, list) or set(killed) != set(lanes) - set(survivors):
        violations.append("killed must be the complement of survivors")
        killed = []
    distinction_kind = probe.get("distinction_kind")
    if distinction_kind not in {
        "architecture-contract",
        "cosmetic",
        "mixed",
        "unclear",
    }:
        violations.append("probe ranking requires a valid distinction_kind")
    distinction_evidence = probe.get("distinction_evidence")
    if (
        not isinstance(distinction_evidence, list)
        or not distinction_evidence
        or any(not isinstance(row, str) or not row.strip() for row in distinction_evidence)
    ):
        violations.append("probe ranking requires non-empty distinction_evidence")
        distinction_evidence = []

    probe_candidate_usage = _zero_usage()
    probe_usage_lanes: set[int] = set()
    candidate_usage_rows = probe.get("candidate_usage")
    if not isinstance(candidate_usage_rows, list):
        violations.append("probe candidate_usage must account for every lane")
    else:
        for row in candidate_usage_rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("lane"), int)
                or isinstance(row.get("lane"), bool)
                or row.get("lane") not in lanes
            ):
                violations.append("probe candidate_usage has an invalid lane")
                continue
            lane = row["lane"]
            if lane in probe_usage_lanes:
                violations.append(f"probe candidate_usage duplicates lane {lane}")
                continue
            probe_usage_lanes.add(lane)
            _add_usage(probe_candidate_usage, row.get("usage"))
        if probe_usage_lanes != set(lanes):
            violations.append("probe candidate_usage must cover every lane")

    continuation_by_lane: dict[int, dict[str, Any]] = {}
    continuation_usage_by_lane: dict[int, dict[str, int]] = {}
    label_to_lane: dict[str, int] = {}
    continuation_usage = _zero_usage()
    for row in continuations:
        lane = row.get("lane")
        label = row.get("review_label")
        if lane in continuation_by_lane:
            violations.append(f"lane {lane!r} has duplicate continuations")
            continue
        if not isinstance(lane, int) or isinstance(lane, bool) or lane not in lanes:
            violations.append(f"continuation has unknown lane {lane!r}")
            continue
        if not isinstance(label, str) or not label:
            violations.append(f"lane {lane} has no opaque review_label")
            continue
        if label in label_to_lane:
            violations.append(f"duplicate review_label {label!r}")
            continue
        continuation_by_lane[lane] = row
        label_to_lane[label] = lane
        lane_usage = _usage(row.get("usage"))
        continuation_usage_by_lane[lane] = lane_usage
        _add_usage(continuation_usage, lane_usage)
        if row.get("base_snapshot_id") != config.get("base_snapshot_id"):
            violations.append(f"lane {lane} did not start from the frozen base snapshot")
        if row.get("continuation_steps") != continuation_steps:
            violations.append(f"lane {lane} did not receive the fixed continuation budget")
        if row.get("isolated") is not True:
            violations.append(f"lane {lane} was not recorded as isolated")
        if row.get("published_to_canonical") is not False:
            violations.append(f"lane {lane} was published before end review")
        expected = "survivor" if lane in survivors else "killed-shadow"
        if row.get("disposition") != expected:
            violations.append(f"lane {lane} has incorrect disposition")

    missing_survivors = set(survivors) - set(continuation_by_lane)
    if missing_survivors:
        violations.append(f"survivors lack continuations: {sorted(missing_survivors)}")
    continued_killed = set(killed) & set(continuation_by_lane)
    if not continued_killed:
        violations.append("no killed lane received an independent continuation")
    complete_ground_truth = set(continuation_by_lane) == set(lanes)

    try:
        final_labels = _ranked_ids(review.get("ranking"), "review_label")
    except ValueError as error:
        violations.append(f"end review ranking: {error}")
        final_labels = []
    if set(final_labels) != set(label_to_lane):
        violations.append("end review ranking must cover every continued branch")
    final_ranking = [
        label_to_lane[label] for label in final_labels if label in label_to_lane
    ]
    if review.get("blinded") is not True:
        violations.append("end review was not blinded")
    if review.get("probe_metadata_excluded") is not True:
        violations.append("end review did not mechanically exclude probe metadata")

    final_winner = final_ranking[0] if final_ranking else None
    final_winner_probe_rank = (
        probe_ranking.index(final_winner) + 1
        if final_winner is not None and final_winner in probe_ranking
        else None
    )
    top_hits = {
        f"top{depth}": (
            final_winner_probe_rank <= depth
            if final_winner_probe_rank is not None
            else None
        )
        for depth in range(1, min(3, len(lanes)) + 1)
    }
    final_winner_survived = top_hits.get(f"top{top_k}") if valid_top_k else None
    eligible_for_gate = (
        complete_ground_truth
        and valid_top_k
        and top_k == 2
        and candidate_count >= 3
        and valid_probe_steps
        and not violations
    )

    usage = {
        "probe_candidates": probe_candidate_usage,
        "probe_review": _usage(probe.get("usage")),
        "continuations": continuation_usage,
        "end_review": _usage(review.get("usage")),
    }
    total_usage = _zero_usage()
    for value in usage.values():
        _add_usage(total_usage, value)
    usage["total"] = total_usage
    survivor_continuation_usage = _sum_usage(
        continuation_usage_by_lane[lane]
        for lane in survivors
        if lane in continuation_usage_by_lane
    )
    oracle_continuation_usage = _sum_usage(
        [continuation_usage_by_lane[final_winner]]
        if final_winner in continuation_usage_by_lane
        else []
    )
    continuation_raw = _raw_input(continuation_usage)
    total_raw = _raw_input(total_usage)
    survivor_raw = _raw_input(survivor_continuation_usage)
    oracle_raw = _raw_input(oracle_continuation_usage)
    savings = {
        "all_lane_continuation_raw_input": continuation_raw,
        "top_k_continuation_raw_input": survivor_raw,
        "one_winner_oracle_continuation_raw_input": oracle_raw,
        "top_k_continuation_savings_fraction": (
            1 - survivor_raw / continuation_raw if continuation_raw else None
        ),
        "one_winner_oracle_continuation_savings_fraction": (
            1 - oracle_raw / continuation_raw if continuation_raw else None
        ),
        "top_k_total_savings_after_measured_overhead_fraction": (
            (continuation_raw - survivor_raw) / total_raw if total_raw else None
        ),
        "one_winner_oracle_total_savings_after_measured_overhead_fraction": (
            (continuation_raw - oracle_raw) / total_raw if total_raw else None
        ),
    }
    return {
        "source": source,
        "study_id": study_id,
        "task": config.get("task"),
        "window": config.get("window"),
        "candidate_count": candidate_count,
        "probe_steps": probe_steps,
        "continuation_steps": continuation_steps,
        "top_k": top_k,
        "probe_ranking": probe_ranking,
        "survivors": survivors,
        "killed": killed,
        "distinction_kind": distinction_kind,
        "distinction_evidence": distinction_evidence,
        "continued_lanes": sorted(continuation_by_lane),
        "continued_killed": sorted(continued_killed),
        "complete_ground_truth": complete_ground_truth,
        "final_ranking": final_ranking,
        "final_winner": final_winner,
        "final_winner_probe_rank": final_winner_probe_rank,
        "probe_recall": top_hits,
        "final_winner_survived": final_winner_survived,
        "late_bloomer_killed": final_winner_survived is False,
        "eligible_for_gate": eligible_for_gate,
        "protocol_violations": violations,
        "usage": usage,
        "continuation_usage_by_lane": {
            str(lane): value for lane, value in sorted(continuation_usage_by_lane.items())
        },
        "savings": savings,
    }


def extract_studies(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows(path):
        study_id = row.get("study_id")
        if not isinstance(study_id, str) or not study_id:
            raise ValueError(f"{path}: shadow record has no study_id")
        grouped[study_id].append(row)
    outcome = public_outcome(path)
    studies = [
        score_study(str(path), study_id, records)
        for study_id, records in sorted(grouped.items())
    ]
    for study in studies:
        study["public_outcome"] = outcome
    return studies


def _wilson(
    successes: int, trials: int, z: float = 1.959963984540054
) -> dict[str, float] | None:
    if trials == 0:
        return None
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    variance = rate * (1 - rate) / trials + z * z / (4 * trials * trials)
    margin = z * math.sqrt(variance) / denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def summarize(
    studies: list[dict[str, Any]], min_complete_studies: int = 20
) -> dict[str, Any]:
    eligible = [study for study in studies if study["eligible_for_gate"]]
    by_probe_steps: dict[str, dict[str, Any]] = {}
    for probe_steps in (1, 2):
        group = [study for study in eligible if study["probe_steps"] == probe_steps]
        recall: dict[str, Any] = {}
        for depth in (1, 2, 3):
            key = f"top{depth}"
            opportunities = sum(study["probe_recall"].get(key) is not None for study in group)
            hits = sum(study["probe_recall"].get(key) is True for study in group)
            recall[key] = {
                "hits": hits,
                "opportunities": opportunities,
                "rate": hits / opportunities if opportunities else None,
                "wilson_95": _wilson(hits, opportunities),
            }
        shadow_raw = sum(_raw_input(study["usage"]["total"]) for study in group)
        top2_saved_raw = sum(
            study["savings"]["all_lane_continuation_raw_input"]
            - study["savings"]["top_k_continuation_raw_input"]
            for study in group
        )
        oracle_saved_raw = sum(
            study["savings"]["all_lane_continuation_raw_input"]
            - study["savings"]["one_winner_oracle_continuation_raw_input"]
            for study in group
        )
        by_probe_steps[str(probe_steps)] = {
            "eligible_studies": len(group),
            "recall": recall,
            "late_bloomer_kill_rate": (
                1 - recall["top2"]["rate"]
                if recall["top2"]["rate"] is not None
                else None
            ),
            "shadow_total_raw_input": shadow_raw,
            "top2_total_savings_after_measured_overhead_fraction": (
                top2_saved_raw / shadow_raw if shadow_raw else None
            ),
            "one_winner_oracle_total_savings_after_measured_overhead_fraction": (
                oracle_saved_raw / shadow_raw if shadow_raw else None
            ),
            "probe_distinction_counts": {
                kind: sum(study["distinction_kind"] == kind for study in group)
                for kind in (
                    "architecture-contract",
                    "cosmetic",
                    "mixed",
                    "unclear",
                )
            },
            "architecture_or_contract_distinction_fraction": (
                sum(
                    study["distinction_kind"] in {"architecture-contract", "mixed"}
                    for study in group
                )
                / len(group)
                if group
                else None
            ),
        }
    two_step = by_probe_steps["2"]
    two_step_top2 = two_step["recall"]["top2"]
    observed = two_step_top2["rate"]
    remaining_to_minimum = max(
        0, min_complete_studies - two_step_top2["opportunities"]
    )
    maximum_at_minimum = (
        (two_step_top2["hits"] + remaining_to_minimum) / min_complete_studies
    )
    futility_stop = (
        two_step_top2["opportunities"] < min_complete_studies
        and maximum_at_minimum < 0.9
    )
    gate_status = "insufficient-data"
    if futility_stop:
        gate_status = "fail-futility"
    elif two_step["eligible_studies"] >= min_complete_studies:
        gate_status = "pass" if observed is not None and observed >= 0.9 else "fail"
    top1_status = "insufficient-data"
    if two_step["eligible_studies"] >= min_complete_studies:
        top1_rate = two_step["recall"]["top1"]["rate"]
        top1_status = "pass" if top1_rate is not None and top1_rate >= 0.95 else "fail"
    return {
        "studies": studies,
        "summary": {
            "total_studies": len(studies),
            "complete_ground_truth_studies": sum(
                study["complete_ground_truth"] for study in studies
            ),
            "eligible_top2_studies": len(eligible),
            "protocol_invalid_studies": sum(
                bool(study["protocol_violations"]) for study in studies
            ),
            "by_probe_steps": by_probe_steps,
            "two_step_top2_hits": two_step_top2["hits"],
            "two_step_top2_opportunities": two_step_top2["opportunities"],
            "top2_recall": observed,
            "top2_recall_wilson_95": two_step_top2["wilson_95"],
            "minimum_complete_studies": min_complete_studies,
            "remaining_studies_to_minimum": remaining_to_minimum,
            "maximum_attainable_top2_recall_at_minimum": maximum_at_minimum,
            "futility_stop": futility_stop,
            "gate_threshold": 0.9,
            "gate_status": gate_status,
            "top1_funding_threshold": 0.95,
            "top1_funding_status": top1_status,
            "task_outcomes": [
                {
                    "task": study["task"],
                    "probe_steps": study["probe_steps"],
                    **study.get("public_outcome", {}),
                }
                for study in studies
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="+", type=Path, help="archive, JSONL, or directory"
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--min-complete-studies", type=int, default=20)
    args = parser.parse_args()
    if args.min_complete_studies < 1:
        parser.error("--min-complete-studies must be positive")
    try:
        inputs = discover_inputs(args.path)
        if not inputs:
            parser.error("no .zip or .jsonl inputs found")
        studies = [study for path in inputs for study in extract_studies(path)]
        if not studies:
            parser.error("no shadow survivor-recall records found")
        report = summarize(studies, args.min_complete_studies)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
