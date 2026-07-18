import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import plan_supervisor_replay_batch as planner
import render_replay_prompts as prompts


def message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def capture_rows(
    window: int,
    candidate_count: int,
    history: list[dict],
    selected_windows: list[list[dict]],
    source_mode: str = "full",
) -> list[dict]:
    state = {
        "window": window,
        "mode": source_mode,
        "selected_trajectory_initial": [message("assistant", "initial work")],
        "selected_trajectory_windows": selected_windows,
        "supervisor_history": {"checkpointed": [], "selected_windows": history},
        "canonical_ledger": [
            [item["window"], {"entries": [{"id": f"L{item['window']}"}]}]
            for item in history
        ],
        "candidate_trajectories": "\n".join(
            f'<lane_trajectory index="{lane}">x</lane_trajectory>'
            for lane in range(candidate_count)
        ),
    }
    shell = {
        "request": {
            "messages": [
                message("system", "system"),
                message("user", "task"),
                message("assistant", "checklist"),
                message("user", f"current candidates {window}"),
            ]
        },
        "state": state,
    }
    full_messages = prompts.full_control_messages(shell)
    prompt_bytes = len(prompts.render_dossier_messages(full_messages).encode())
    decision = {
        "winner": 0,
        "complete": window == 2,
        "next_candidate_count": 1,
        "next_window_steps": 3,
        "state_summary": f"state {window}",
    }
    return [
        {
            "type": "asgard_window",
            "window": window,
            "candidate_count": candidate_count,
            "window_steps": 3,
        },
        {
            "type": "asgard_supervisor_prompt_mode",
            "window": window,
            "mode": source_mode,
            "prompt_bytes": prompt_bytes,
            "full_control_prompt_bytes": prompt_bytes,
            "estimated_request_tokens": 100,
            "full_control_estimated_request_tokens": 100,
        },
        {"type": "asgard_supervisor_replay_state", **state},
        {
            "type": "asgard_supervisor_replay_request",
            "decision_call": "supervisor",
            "call_index": 1,
            "phase": "selection",
            "model": "deepseek::pro",
            "messages": full_messages,
            "tools": [{"type": "function", "function": {"name": "select"}}],
            "parameters": {},
        },
        {
            "type": "asgard_supervisor_replay_response",
            "call_index": 1,
            "response": {},
            "usage": {
                "input": 10,
                "cachedRead": 20,
                "cachedWrite": 0,
                "output": 1,
                "thought": 1,
            },
        },
        {"type": "asgard_decision", "call": "supervisor", "decision": decision},
    ]


def write_archive(
    root: Path, task: str, run: int, two_windows: bool, source_mode: str = "full"
) -> Path:
    rows = capture_rows(1, 1, [], [], source_mode)
    if two_windows:
        rows.append(
            {
                "type": "asgard_decision",
                "call": "completion_review",
                "decision": {"winner": 0, "complete": False},
            }
        )
        history = [{"window": 1, "winner": 0, "state_summary": "state 1"}]
        rows.extend(
            capture_rows(2, 3, history, [[message("assistant", "window one")]], source_mode)
        )
    path = root / f"full-{task}-r{run}-100-200.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.json", json.dumps({"taskId": task}))
        archive.writestr(
            "anvil-trace.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    return path


class PlanSupervisorReplayBatchTest(unittest.TestCase):
    def test_forces_protected_endpoint_and_builds_staged_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected_archive = write_archive(root, "task-a", 1, True)
            other_archive = write_archive(root, "task-b", 1, False)
            plan = planner.build_plan(
                [protected_archive, other_archive],
                {"task-a::r1"},
                sample_size=1,
                seed="fixed",
                long_history_windows=1,
            )

        self.assertEqual(plan["selection"]["requested_sample_size"], 1)
        self.assertEqual(plan["selection"]["selected_records"], 1)
        self.assertTrue(
            plan["protected_coverage"]["all_captured_protected_endpoints_selected"]
        )
        record = plan["records"][0]
        self.assertEqual(record["task_run_identity"], "task-a::r1")
        self.assertEqual(record["window"], 2)
        self.assertEqual(record["strata"]["lane_mode"], "multi")
        self.assertEqual(record["strata"]["history_windows"], 1)
        self.assertTrue(record["strata"]["long_history"])
        self.assertTrue(record["strata"]["post_completion_review"])
        self.assertTrue(record["protected_endpoint"])
        self.assertEqual(plan["stages"][0]["base_records"], 1)
        self.assertEqual(plan["stages"][0]["calls"], 4)
        self.assertEqual(plan["stages"][1]["base_records"], 0)
        self.assertEqual(plan["totals"]["calls"], 4)
        for mode in planner.TARGET_MODES:
            estimate = record["target_estimates"][mode]
            self.assertGreater(estimate["prompt_bytes"], 0)
            self.assertGreaterEqual(
                estimate["estimated_request_input_tokens_conservative"],
                estimate["estimated_prompt_tokens_conservative"],
            )
            self.assertGreaterEqual(
                estimate["utf8_byte_token_ceiling"],
                estimate["estimated_request_input_tokens_conservative"],
            )

    def test_all_records_and_seeded_sampling_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = [
                write_archive(root, "task-a", 1, True),
                write_archive(root, "task-b", 1, False),
            ]
            all_plan = planner.build_plan(archives, {"task-a::r1"})
            first = planner.build_plan(
                archives, {"task-a::r1"}, sample_size=2, seed="same"
            )
            second = planner.build_plan(
                archives, {"task-a::r1"}, sample_size=2, seed="same"
            )

        self.assertEqual(all_plan["totals"]["base_records"], 3)
        self.assertEqual(all_plan["totals"]["calls"], 12)
        self.assertEqual(
            [row["record_id"] for row in first["records"]],
            [row["record_id"] for row in second["records"]],
        )
        self.assertIn("one", all_plan["available_strata"]["lane_mode"])
        self.assertIn("multi", all_plan["available_strata"]["lane_mode"])

    def test_non_full_capture_is_discovered_but_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = write_archive(
                Path(directory), "task-c", 1, False, source_mode="latest-state"
            )
            records, report = planner.extract_archive_records(
                archive,
                set(),
                long_history_windows=10,
                modes=planner.TARGET_MODES,
            )
        self.assertEqual(records, [])
        self.assertEqual(report["status"], "skipped_non_full_capture")


if __name__ == "__main__":
    unittest.main()
