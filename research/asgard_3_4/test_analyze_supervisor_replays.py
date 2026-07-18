import json
import tempfile
import unittest
from pathlib import Path

import analyze_supervisor_replays as analysis


def replay_row(index: int, *, fallback: bool = False, winner: int = 0) -> dict:
    control = {
        "winner": 0,
        "complete": False,
        "next_candidate_count": 2,
        "next_window_steps": 4,
    }
    selection = None
    if not fallback:
        selection = {
            "winner": winner,
            "complete": False,
            "next_candidate_count": 3 if index == 0 else 2,
            "next_window_steps": 2 if index == 0 else 4,
        }
    return {
        "type": analysis.RECORD_TYPE,
        "record_id": f"row-{index}",
        "target_mode": "checkpoint-plus-delta",
        "window": index + 1,
        "status": "audit_or_no_selection:grep_search" if fallback else "selected",
        "fallback_required": fallback,
        "selection": selection,
        "control_decision": control,
        "usage": {
            "input": 10,
            "cachedRead": 30,
            "cachedWrite": 0,
            "output": 2,
            "thought": 1,
        },
        "prompt_bytes": 50,
        "full_control_prompt_bytes": 100,
    }


class AnalyzeSupervisorReplaysTest(unittest.TestCase):
    def test_reports_nonfallback_effective_deltas_usage_and_gates(self) -> None:
        rows = [replay_row(index) for index in range(19)]
        rows[1]["selection"]["winner"] = 1
        rows.append(replay_row(19, fallback=True))
        rows[2]["protected_identity"] = "task-a::r1"

        report = analysis.summarize(rows, {"task-a::r1"})
        mode = report["modes"]["checkpoint-plus-delta"]

        self.assertEqual(mode["records"], 20)
        self.assertEqual(mode["fallback"]["count"], 1)
        self.assertAlmostEqual(mode["fallback"]["rate"], 0.05)
        self.assertEqual(mode["nonfallback"]["winner_agreement"]["matches"], 18)
        self.assertEqual(mode["nonfallback"]["winner_agreement"]["compared"], 19)
        self.assertAlmostEqual(
            mode["effective_with_full_fallback"]["winner_agreement"]["rate"],
            0.95,
        )
        self.assertEqual(
            mode["nonfallback"]["next_candidate_count_signed_delta"]["histogram"],
            {"0": 18, "1": 1},
        )
        self.assertEqual(
            mode["nonfallback"]["next_step_count_signed_delta"]["min"], -2
        )
        self.assertEqual(mode["disagreements"]["count"], 2)
        self.assertEqual(
            mode["disagreements"]["rows"][0]["disagreement_fields"],
            ["next_candidate_count", "next_step_count"],
        )
        self.assertEqual(mode["usage"]["all"]["rawInput"], 800)
        self.assertAlmostEqual(mode["usage"]["all"]["cachedReadFraction"], 0.75)
        self.assertEqual(
            mode["prompt_or_raw_input_reduction"]["gate_fraction"], 0.5
        )
        self.assertEqual(mode["protected"]["records"], 1)
        self.assertEqual(
            mode["protected"]["effective_endpoint_agreement"]["rate"], 1.0
        )
        self.assertTrue(mode["automated_gates_pass"])
        self.assertFalse(mode["recommendation_ready"])
        self.assertEqual(
            mode["gates"]["obligation_preservation_review"]["status"],
            "manual_required",
        )

    def test_missing_reduction_and_protected_coverage_fail_closed(self) -> None:
        row = replay_row(0)
        row.pop("prompt_bytes")
        row.pop("full_control_prompt_bytes")
        mode = analysis.summarize([row])["modes"]["checkpoint-plus-delta"]
        self.assertEqual(mode["gates"]["input_reduction"]["status"], "unknown")
        self.assertEqual(
            mode["gates"]["protected_endpoint_agreement"]["status"], "unknown"
        )
        self.assertFalse(mode["automated_gates_pass"])

    def test_accepts_pilot_reduction_when_rows_lack_prompt_fields(self) -> None:
        row = replay_row(0)
        row.pop("prompt_bytes")
        row.pop("full_control_prompt_bytes")
        row["protected_endpoint"] = True
        pilot = {
            "modes": {
                "checkpoint-plus-delta": {
                    "prompt_byte_reduction_fraction": 0.3
                }
            }
        }
        mode = analysis.summarize([row], pilot=pilot)["modes"][
            "checkpoint-plus-delta"
        ]
        self.assertEqual(mode["gates"]["input_reduction"]["status"], "pass")
        self.assertEqual(mode["gates"]["input_reduction"]["source"], "pilot report mode aggregate")

    def test_rejects_inconsistent_precomputed_agreement(self) -> None:
        row = replay_row(0)
        row["winner_agreement"] = False
        with self.assertRaisesRegex(ValueError, "contradicts decisions"):
            analysis.summarize([row])

    def test_reads_only_result_rows_and_tracks_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replays.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "unrelated"}),
                        json.dumps(replay_row(0)),
                    ]
                )
                + "\n"
            )
            records = analysis.read_replay_results([path])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["_line"], 2)


if __name__ == "__main__":
    unittest.main()
