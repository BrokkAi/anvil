import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import analyze_live_pilot as pilot


class AnalyzeLivePilotTest(unittest.TestCase):
    def test_extracts_phase_usage_and_counterfactual_prompt_savings(self) -> None:
        rows = [
            {
                "type": "asgard_supervisor_prompt_mode",
                "window": 2,
                "mode": "latest-state",
                "prompt_bytes": 60,
                "full_control_prompt_bytes": 100,
                "estimated_request_tokens": 15,
                "full_control_estimated_request_tokens": 25,
            },
            {
                "type": "asgard_supervisor_replay_request",
                "decision_call": "supervisor",
                "call_index": 1,
            },
            {
                "type": "asgard_supervisor_replay_response",
                "call_index": 1,
                "response": {},
                "usage": {
                    "input": 10,
                    "output": 2,
                    "thought": 3,
                    "cachedRead": 5,
                    "cachedWrite": 0,
                },
            },
            {
                "type": "asgard_decision",
                "call": "supervisor",
                "decision": {"winner": 0},
            },
            {
                "type": "asgard_window",
                "window": 2,
                "candidate_count": 1,
                "window_steps": 3,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "anvil-trace.jsonl",
                    "\n".join(json.dumps(row) for row in rows),
                )
                archive.writestr(
                    "result.json",
                    json.dumps(
                        {
                            "taskId": "task-1",
                            "inputTokens": 20,
                            "cachedInputTokens": 5,
                            "outputTokens": 4,
                            "reasoningOutputTokens": 3,
                        }
                    ),
                )
                archive.writestr(
                    "reward.json", json.dumps({"reward": 1, "partial": 1.0})
                )
            run = pilot.extract_run(path)
            summary = pilot.summarize([run])

        self.assertEqual(run["mode"], "latest-state")
        self.assertEqual(run["ordinary_routing"]["usage"]["input"], 10)
        self.assertEqual(run["ordinary_routing"]["usage"]["cachedRead"], 5)
        self.assertEqual(run["ordinary_routing"]["usage"]["rawInput"], 15)
        self.assertAlmostEqual(
            run["ordinary_routing"]["usage"]["cachedReadFraction"], 1 / 3
        )
        self.assertAlmostEqual(
            run["ordinary_routing"]["prompt_byte_reduction_fraction"], 0.4
        )
        self.assertEqual(run["window_policy"]["candidate_counts"], {"1": 1})
        self.assertEqual(
            summary["modes"]["latest-state"]["ordinary_usage_per_window"][
                "rawInput"
            ],
            15,
        )

    def test_rejects_mismatched_replay_call_index(self) -> None:
        rows = [
            {
                "type": "asgard_supervisor_prompt_mode",
                "mode": "full",
                "prompt_bytes": 1,
                "full_control_prompt_bytes": 1,
            },
            {
                "type": "asgard_supervisor_replay_request",
                "decision_call": "supervisor",
                "call_index": 1,
            },
            {
                "type": "asgard_supervisor_replay_response",
                "call_index": 2,
                "usage": {},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "anvil-trace.jsonl",
                    "\n".join(json.dumps(row) for row in rows),
                )
                archive.writestr("result.json", "{}")
                archive.writestr("reward.json", "{}")
            with self.assertRaisesRegex(ValueError, "does not match pending request"):
                pilot.extract_run(path)


if __name__ == "__main__":
    unittest.main()
