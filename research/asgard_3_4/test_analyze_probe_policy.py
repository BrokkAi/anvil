import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import analyze_probe_policy as policy


def usage(amount: int) -> dict[str, int]:
    return {
        "input": amount,
        "output": 0,
        "thought": 0,
        "cachedRead": 0,
        "cachedWrite": 0,
    }


class ProbePolicyAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def archive(
        self,
        name: str,
        mode: str,
        *,
        count: int = 3,
        steps: int = 2,
        task: str = "task-a",
        candidate_input: int = 60,
        total_input: int = 100,
        reported_candidate_input: int | None = None,
        shadow_survivor_study: bool = False,
        extra_rows: list[dict] | None = None,
    ) -> Path:
        rows = [
            {
                "type": "asgard_window_policy_config",
                "mode": mode,
                "shadow_survivor_study": shadow_survivor_study,
                "shadow_probe_steps": 2,
            }
        ]
        if mode == "explicit-probe":
            rows.append(
                {
                    "type": "asgard_window_kind",
                    "window": 1,
                    "kind": "probe",
                    "candidate_count": count,
                    "window_steps": steps,
                    "hypotheses": [],
                }
            )
        rows.append(
            {
                "type": "asgard_window",
                "window": 1,
                "candidate_count": count,
                "window_steps": steps,
            }
        )
        lane_amounts = [candidate_input // count] * count
        lane_amounts[-1] += candidate_input - sum(lane_amounts)
        rows.append(
            {
                "type": "asgard_candidate_window_usage",
                "window": 1,
                "candidate_count": count,
                "window_steps": steps,
                "lanes": [
                    {"lane": lane, "model": "deepseek", "usage": usage(amount)}
                    for lane, amount in enumerate(lane_amounts)
                ],
                "usage": usage(
                    candidate_input
                    if reported_candidate_input is None
                    else reported_candidate_input
                ),
            }
        )
        rows.extend(extra_rows or [])
        result = {
            "taskId": task,
            "stopReason": "SUCCESS",
            "inputTokens": total_input,
            "outputTokens": 0,
            "reasoningOutputTokens": 0,
            "cachedInputTokens": 0,
            "cacheWriteTokens": 0,
            "reward": {"reward": 1, "partial": 1},
        }
        path = self.root / f"{name}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("anvil-trace.jsonl", "\n".join(map(json.dumps, rows)))
            archive.writestr("result.json", json.dumps(result))
            archive.writestr("reward.json", json.dumps({"reward": 1, "partial": 1}))
        return path

    def test_compares_policy_distribution_cost_and_paired_attempts(self) -> None:
        dynamic = self.archive(
            "dynamic", "dynamic", steps=3, candidate_input=90, total_input=100
        )
        explicit = self.archive(
            "explicit", "explicit-probe", steps=2, candidate_input=60, total_input=80
        )
        explicit_second = self.archive(
            "explicit-second",
            "explicit-probe",
            steps=2,
            task="task-a",
            candidate_input=60,
            total_input=80,
        )
        report = policy.summarize(
            [policy.extract_run(path) for path in (dynamic, explicit, explicit_second)]
        )
        self.assertEqual(report["modes"]["dynamic"]["candidate_usage_coverage"], 1)
        self.assertEqual(
            report["modes"]["explicit-probe"]["tournament_eligible_fraction"], 1
        )
        self.assertEqual(report["comparison"]["joint_distribution_total_variation"], 1)
        self.assertAlmostEqual(
            report["comparison"]["candidate_raw_input_per_lane_step_reduction_fraction"],
            0,
        )
        self.assertAlmostEqual(
            report["comparison"]["total_raw_input_per_run_reduction_fraction"], 0.2
        )
        self.assertFalse(report["comparison"]["behaviorally_same_by_prespecified_tolerance"])
        self.assertEqual(len(report["paired_outcomes"]["task-a"]["explicit-probe"]), 2)

    def test_rejects_candidate_usage_that_does_not_reconcile(self) -> None:
        path = self.archive("bad-usage", "dynamic", reported_candidate_input=61)
        with self.assertRaisesRegex(ValueError, "lane usage does not sum"):
            policy.extract_run(path)

    def test_rejects_duplicate_window_records(self) -> None:
        duplicate = {
            "type": "asgard_window",
            "window": 1,
            "candidate_count": 3,
            "window_steps": 2,
        }
        path = self.archive("duplicate", "dynamic", extra_rows=[duplicate])
        with self.assertRaisesRegex(ValueError, "duplicate asgard_window"):
            policy.extract_run(path)

    def test_rejects_shadow_survivor_runs(self) -> None:
        path = self.archive("shadow", "dynamic", shadow_survivor_study=True)
        with self.assertRaisesRegex(ValueError, "excludes shadow survivor"):
            policy.extract_run(path)


if __name__ == "__main__":
    unittest.main()
