import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import build_known_success_corpus as corpus


class BuildKnownSuccessCorpusTest(unittest.TestCase):
    def _write_archive(self, root: Path, cohort: str, explicit_window: bool) -> Path:
        task_id = "demo-task"
        archive_dir = root / task_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{cohort}-{task_id}-r1-100-200.zip"
        rows = [
            {
                "type": "asgard_checklist",
                "contracts": [
                    {
                        "id": "C1",
                        "kind": "execution",
                        "text": "preserve ordering",
                        "adverse_condition": "two values",
                    }
                ],
            }
        ]
        if explicit_window:
            rows.append(
                {
                    "type": "asgard_window",
                    "window": 1,
                    "candidate_count": 2,
                    "window_steps": 4,
                }
            )
        rows.extend(
            [
                {
                    "type": "tool_timing",
                    "tool": "run_shell_command",
                    "command": "python -m pytest tests/test_demo.py",
                    "success": True,
                },
                {
                    "type": "asgard_decision",
                    "call": "supervisor",
                    "decision": {
                        "winner": 1,
                        "complete": True,
                        "advices": ["run the focused test"],
                        "next_candidate_count": 1,
                        "next_window_steps": 3,
                        "state_summary": "The adapter is correct; pytest passed.",
                    },
                },
                {
                    "type": "asgard_decision",
                    "call": "completion_review",
                    "decision": {"winner": 1, "complete": True},
                },
            ]
        )
        stderr = "\n".join(
            [
                "summarized Asgard candidate trajectory lane=1",
                "summarized Asgard candidate trajectory lane=2",
                "assembled Asgard supervisor dossier window=1",
                'executing tool run_shell_command with args: {"command":"cargo test demo","timeout":1000} (sandbox=WorkspaceWrite)',
            ]
        )
        patch = """diff --git a/demo.py b/demo.py
--- a/demo.py
+++ b/demo.py
@@ -1 +1,2 @@ module
+class Adapter:
+    pass
"""
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "anvil-trace.jsonl",
                "\n".join(json.dumps(row) for row in rows) + "\n",
            )
            archive.writestr("anvil-stderr.txt", stderr)
            archive.writestr("model.patch", patch)
            # The builder must neither parse nor copy these hidden artifacts.
            archive.writestr("verifier-output.txt", "SECRET HIDDEN FAILURE DETAIL")
            archive.writestr("verifier.tar.gz", b"not actually a tarball")
        return archive_path

    def _write_result(
        self, agentresults: Path, cohort: str, archive_path: Path
    ) -> None:
        result_dir = agentresults / "demo-task1"
        result_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "taskId": "demo-task",
            "outcome": "SUCCESS",
            "stopReason": "SUCCESS",
            "archivePath": str(archive_path),
            "baseCommit": "abc",
            "anvilSha256": "def",
            "changedFiles": ["demo.py"],
            "reward": {"reward": 1, "partial": 1.0},
        }
        (result_dir / f"{cohort}-abc.json").write_text(json.dumps(result))

    def test_builds_union_and_separates_facts_from_inferences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agentresults = root / "agentresults"
            archives = root / "archives"
            for cohort, explicit in zip(corpus.DEFAULT_COHORTS, (False, True)):
                archive = self._write_archive(archives, cohort, explicit)
                self._write_result(agentresults, cohort, archive)

            records = corpus.build_corpus(
                agentresults,
                archives,
                expected_runs=1,
                expected_successes=1,
                expected_union=1,
                expected_common=1,
            )

        manifest, v6, v9 = records
        self.assertEqual(manifest["protected_union_count"], 1)
        self.assertEqual(manifest["success_trace_count"], 2)
        self.assertEqual(v6["protected_identity"], "demo-task::r1")
        self.assertIn("facts", v6)
        self.assertIn("inferences", v6)
        self.assertEqual(
            v6["facts"]["funding_sequence"][0]["candidate_count"], 2
        )
        self.assertIsNone(v6["facts"]["funding_sequence"][0]["window_steps"])
        self.assertEqual(
            v9["facts"]["funding_sequence"][0]["window_steps"], 4
        )
        declarations = v6["facts"]["implementation_surface"]["added_declarations"]
        self.assertEqual(declarations[0]["name"], "Adapter")
        commands = v6["facts"]["verification"]["commands_observed"]
        self.assertEqual(
            {item["command"] for item in commands},
            {"cargo test demo", "python -m pytest tests/test_demo.py"},
        )
        self.assertEqual(
            v6["facts"]["weak_after_one_or_two_steps"]["status"], "unknown"
        )
        self.assertNotIn("SECRET HIDDEN FAILURE DETAIL", json.dumps(records))

    def test_strict_expected_counts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected 40 results"):
                corpus.build_corpus(
                    Path(directory),
                    Path(directory),
                    expected_runs=40,
                    expected_successes=9,
                    expected_union=15,
                    expected_common=3,
                )


if __name__ == "__main__":
    unittest.main()
