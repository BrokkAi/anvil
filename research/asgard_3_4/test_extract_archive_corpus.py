import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import extract_archive_corpus as corpus


class ExtractArchiveCorpusTest(unittest.TestCase):
    def test_parses_ansi_dossier_telemetry(self) -> None:
        text = (
            "\x1b[32m INFO\x1b[0m assembled Asgard supervisor dossier "
            "\x1b[3mwindow\x1b[0m=2 "
            "\x1b[3mselected_initial_bytes\x1b[0m=100 "
            "\x1b[3mselected_windows_bytes\x1b[0m=50 "
            "\x1b[3mcandidate_trajectories_bytes\x1b[0m=25\n"
        )
        rows, warnings = corpus.parse_dossier_telemetry(text)
        self.assertEqual(warnings, [])
        self.assertEqual(
            rows,
            [
                {
                    "window": 2,
                    "selected_initial_bytes": 100,
                    "selected_windows_bytes": 50,
                    "candidate_trajectories_bytes": 25,
                }
            ],
        )

    def test_trace_excludes_completion_review(self) -> None:
        trace = "\n".join(
            json.dumps(row)
            for row in (
                {"type": "asgard_decision", "call": "completion_review", "decision": {}},
                {
                    "type": "asgard_decision",
                    "call": "supervisor",
                    "decision": {"advices": ["a", "b"]},
                },
            )
        )
        decisions, counts, warnings = corpus.parse_trace(trace)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(counts, {"completion_review": 1, "supervisor": 1})
        self.assertEqual(warnings, [])

    def test_candidate_count_prefers_explicit_field_then_advice_length(self) -> None:
        self.assertEqual(
            corpus.infer_candidate_count(
                {"next_candidate_count": 1, "advices": ["a", "b"]}
            ),
            (1, "decision.next_candidate_count"),
        )
        self.assertEqual(
            corpus.infer_candidate_count({"advices": ["a", "b"]}),
            (2, "len(decision.advices)"),
        )

    def test_end_to_end_alignment_metrics_and_missing_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "run.zip"
            telemetry = "\n".join(
                (
                    "assembled Asgard supervisor dossier window=1 "
                    "selected_initial_bytes=100 selected_windows_bytes=0 "
                    "candidate_trajectories_bytes=50",
                    "assembled Asgard supervisor dossier window=2 "
                    "selected_initial_bytes=100 selected_windows_bytes=40 "
                    "candidate_trajectories_bytes=60",
                )
            )
            trace = "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "type": "asgard_decision",
                        "call": "supervisor",
                        "decision": {
                            "winner": 0,
                            "complete": False,
                            "advices": ["continue"],
                            "next_window_steps": 2,
                        },
                    },
                    {
                        "type": "asgard_decision",
                        "call": "completion_review",
                        "decision": {"winner": 0, "complete": False},
                    },
                )
            )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("anvil-stderr.txt", telemetry)
                archive.writestr("anvil-trace.jsonl", trace)
                archive.writestr(
                    "result.json",
                    json.dumps({"taskId": "task-1", "inputTokens": 123}),
                )
                archive.writestr("reward.json", json.dumps({"reward": 1}))

            records, stats = corpus.extract_archive(
                corpus.ArchiveInput(archive_path, {"id": "case-1"})
            )
            summary = corpus.summarize(records, [stats])

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["decision"]["lane_mode"], "one")
        self.assertEqual(records[1]["alignment"]["status"], "decision_missing")
        self.assertEqual(records[1]["dossier_bytes"]["history_growth_from_previous"], 40)
        self.assertEqual(
            records[1]["dossier_bytes"]["older_selected_windows_removal_ceiling"], 40
        )
        self.assertEqual(summary["alignment"]["windows_missing_decisions"], 1)
        self.assertEqual(summary["q3_byte_ceiling"]["full_dossier_measured_bytes"], 350)
        self.assertEqual(summary["q4_retrospective"]["lane_mode_distribution"], {"one": 1, "unknown": 1})


if __name__ == "__main__":
    unittest.main()
