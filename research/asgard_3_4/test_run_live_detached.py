import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import run_live_detached as detached


class RunLiveDetachedTests(unittest.TestCase):
    def test_discovers_selected_attempts_and_reports_current_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch-a"
            batch.mkdir()
            (batch / "run.json").write_text(
                json.dumps(
                    {
                        "batch": {"name": "batch-a"},
                        "working_directory": temporary,
                        "commands": [["true"], ["false"]],
                    }
                ),
                encoding="utf-8",
            )
            specs = detached.discover_specs(root, {"batch-a"}, {2})
            self.assertEqual([(spec.batch, spec.attempt) for spec in specs], [("batch-a", 2)])
            attempt = batch / "attempt-2"
            attempt.mkdir()
            (attempt / "controller.json").write_text(
                json.dumps({"pid": os.getpid()}), encoding="utf-8"
            )
            state = detached.controller_state(specs[0])
            self.assertTrue(state["alive"])
            self.assertEqual(state["pid"], os.getpid())

    def test_status_distinguishes_captured_and_incomplete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch-a"
            batch.mkdir()
            (batch / "run.json").write_text(
                json.dumps(
                    {
                        "batch": {"name": "batch-a"},
                        "working_directory": temporary,
                        "commands": [["true"]],
                    }
                ),
                encoding="utf-8",
            )
            attempt = batch / "attempt-1"
            archives = attempt / "archives"
            results = attempt / "results"
            running = results / ".running" / "active"
            archives.mkdir(parents=True)
            running.mkdir(parents=True)
            with zipfile.ZipFile(archives / "captured.zip", "w") as archive:
                archive.writestr("anvil-trace.jsonl", "{}\n")
                archive.writestr("result.json", "{}")
            with zipfile.ZipFile(archives / "cancelled.zip", "w") as archive:
                archive.writestr("cancellation-error.json", "{}")
            (results / "INFRA_ERROR_LATEST.json").write_text("{}", encoding="utf-8")
            (results / "completed.json").write_text("{}", encoding="utf-8")
            (running / "meta.json").write_text("{}", encoding="utf-8")

            spec = detached.discover_specs(root, {"batch-a"}, {1})[0]
            state = detached.controller_state(spec)
            self.assertEqual(state["captured_archives"], 1)
            self.assertEqual(state["incomplete_archives"], 1)
            self.assertEqual(state["completed_results"], 1)
            self.assertEqual(state["infra_errors"], 1)
            self.assertEqual(state["running_markers"], 1)

    def test_unknown_batch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unknown or unselected"):
                detached.discover_specs(Path(temporary), {"missing"}, set())


if __name__ == "__main__":
    unittest.main()
