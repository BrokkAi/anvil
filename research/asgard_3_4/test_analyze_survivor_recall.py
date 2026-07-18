import json
import tempfile
import unittest
from pathlib import Path

import analyze_survivor_recall as recall


def usage(input_tokens: int = 0) -> dict[str, int]:
    return {
        "input": input_tokens,
        "output": 0,
        "thought": 0,
        "cachedRead": 0,
        "cachedWrite": 0,
    }


def complete_rows(study_id: str = "s-1") -> list[dict]:
    rows = [
        {
            "type": "asgard_shadow_tournament_config",
            "study_id": study_id,
            "task": "task-1",
            "window": 1,
            "candidate_count": 3,
            "probe_steps": 2,
            "continuation_steps": 5,
            "top_k": 2,
            "base_snapshot_id": "base-a",
        },
        {
            "type": "asgard_shadow_probe_ranking",
            "study_id": study_id,
            "ranking": [
                {"lane": 0, "rank": 1},
                {"lane": 1, "rank": 2},
                {"lane": 2, "rank": 3},
            ],
            "survivors": [0, 1],
            "killed": [2],
            "candidate_usage": [
                {"lane": lane, "usage": usage(5 + lane)} for lane in range(3)
            ],
            "usage": usage(10),
        },
    ]
    for lane, label, disposition in [
        (0, "opaque-c", "survivor"),
        (1, "opaque-a", "survivor"),
        (2, "opaque-b", "killed-shadow"),
    ]:
        rows.append(
            {
                "type": "asgard_shadow_continuation",
                "study_id": study_id,
                "lane": lane,
                "review_label": label,
                "disposition": disposition,
                "base_snapshot_id": "base-a",
                "endpoint_snapshot_id": f"end-{lane}",
                "continuation_steps": 5,
                "isolated": True,
                "published_to_canonical": False,
                "usage": usage(20 + lane),
            }
        )
    rows.append(
        {
            "type": "asgard_shadow_end_review",
            "study_id": study_id,
            "blinded": True,
            "probe_metadata_excluded": True,
            "ranking": [
                {"review_label": "opaque-b", "rank": 1},
                {"review_label": "opaque-c", "rank": 2},
                {"review_label": "opaque-a", "rank": 3},
            ],
            "usage": usage(30),
        }
    )
    return rows


class AnalyzeSurvivorRecallTest(unittest.TestCase):
    def test_scores_complete_blinded_top2_study(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in complete_rows()) + "\n",
                encoding="utf-8",
            )
            study = recall.extract_studies(path)[0]
        self.assertTrue(study["complete_ground_truth"])
        self.assertTrue(study["eligible_for_gate"])
        self.assertEqual(study["final_ranking"], [2, 0, 1])
        self.assertEqual(study["top_k_recalled"], 1)
        self.assertEqual(study["top_k_recall"], 0.5)
        self.assertFalse(study["final_winner_survived"])
        self.assertEqual(study["usage"]["total"]["input"], 121)

    def test_partial_killed_sampling_is_visible_but_gate_ineligible(self) -> None:
        rows = complete_rows()
        rows = [
            row
            for row in rows
            if not (
                row["type"] == "asgard_shadow_continuation" and row.get("lane") == 1
            )
        ]
        review = next(row for row in rows if row["type"] == "asgard_shadow_end_review")
        review["ranking"] = [
            {"review_label": "opaque-b", "rank": 1},
            {"review_label": "opaque-c", "rank": 2},
        ]
        study = recall.score_study("memory", "s-1", rows)
        self.assertFalse(study["complete_ground_truth"])
        self.assertFalse(study["eligible_for_gate"])
        self.assertEqual(study["protocol_violations"], ["survivors lack continuations: [1]"])

    def test_gate_requires_minimum_complete_studies(self) -> None:
        study = recall.score_study("memory", "s-1", complete_rows())
        report = recall.summarize([study], min_complete_studies=2)
        self.assertEqual(report["summary"]["gate_status"], "insufficient-data")
        report = recall.summarize([study, study], min_complete_studies=2)
        self.assertEqual(report["summary"]["gate_status"], "fail")

    def test_boolean_probe_step_is_not_accepted_as_one(self) -> None:
        rows = complete_rows()
        config = next(
            row
            for row in rows
            if row["type"] == "asgard_shadow_tournament_config"
        )
        config["probe_steps"] = True
        study = recall.score_study("memory", "s-1", rows)
        self.assertFalse(study["eligible_for_gate"])
        self.assertIn("probe_steps must be 1 or 2", study["protocol_violations"])


if __name__ == "__main__":
    unittest.main()
