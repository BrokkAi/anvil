import json
import tempfile
import unittest
from pathlib import Path

import prepare_live_experiments as prepare


class PrepareLiveExperimentsTests(unittest.TestCase):
    def test_stages_tasks_replaces_only_managed_environment_and_keeps_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "task-a"
            source.mkdir(parents=True)
            (source / "task.toml").write_text(
                "[environment.env]\n"
                'ASGARD_CAPTURE_SUPERVISOR_REPLAYS = "1"\n'
                'KEEP_ME = "yes"\n'
                "[solution.env]\n",
                encoding="utf-8",
            )
            manifest = {
                "schema_version": 1,
                "source_tasks_root": str(root / "source"),
                "output_root": str(root / "output"),
                "runner_root": str(root),
                "anvil_bin": "/anvil",
                "candidate_model": "deepseek::flash",
                "supervisor_model": "deepseek::pro",
                "candidate_count": 3,
                "threads": 2,
                "managed_environment_keys": [
                    "ASGARD_CAPTURE_SUPERVISOR_REPLAYS",
                    "ASGARD_WINDOW_POLICY_MODE",
                ],
                "batches": [
                    {
                        "name": "policy",
                        "label": "policy",
                        "purpose": "test",
                        "tasks": ["task-a"],
                        "run_groups": [2, 1],
                        "environment": {"ASGARD_WINDOW_POLICY_MODE": "explicit-probe"},
                    }
                ],
            }
            commands = prepare.stage(manifest, set())
            staged = (root / "output" / "policy" / "tasks" / "task-a" / "task.toml").read_text()
            self.assertNotIn("ASGARD_CAPTURE_SUPERVISOR_REPLAYS", staged)
            self.assertIn('ASGARD_WINDOW_POLICY_MODE = "explicit-probe"', staged)
            self.assertIn('KEEP_ME = "yes"', staged)
            self.assertEqual(len(commands), 2)
            self.assertIn("--runs", commands[0])
            self.assertEqual(commands[0][commands[0].index("--runs") + 1], "2")
            self.assertEqual(commands[1][commands[1].index("--runs") + 1], "1")
            run = json.loads((root / "output" / "policy" / "run.json").read_text())
            self.assertEqual(run["batch"]["name"], "policy")
            self.assertEqual(len(run["commands"]), 2)

    def test_refuses_to_overwrite_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            manifest = {
                "output_root": str(output),
                "source_tasks_root": str(root),
                "managed_environment_keys": [],
                "batches": [],
            }
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare.stage(manifest, set())


if __name__ == "__main__":
    unittest.main()
