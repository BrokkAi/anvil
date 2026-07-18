import copy
import unittest

import render_replay_prompts as replay


def text_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


class RenderReplayPromptsTest(unittest.TestCase):
    def setUp(self) -> None:
        selected_initial_messages = [text_message("assistant", "initial evidence")]
        selected_initial = text_message(
            "assistant",
            "<selected_trajectory_initial>\n"
            f"{replay.render_dossier_messages(selected_initial_messages)}\n"
            "</selected_trajectory_initial>",
        )
        self.record = {
            "prompt": {
                "mode": "full",
                "window": 3,
                "checkpoint_interval": 3,
                "recent_exact_tail": 1,
            },
            "request": {
                "messages": [
                    text_message("system", "system"),
                    text_message("user", "task"),
                    text_message("assistant", "checklist"),
                    selected_initial,
                    text_message("user", "current dossier"),
                ],
                "tools": [],
                "model": "model",
                "parameters": {},
            },
            "state": {
                "selected_trajectory_initial": selected_initial_messages,
                "selected_trajectory_windows": [
                    [text_message("assistant", "window one")],
                    [text_message("assistant", "window two")],
                ],
                "supervisor_history": {
                    "checkpointed": [],
                    "selected_windows": [
                        {"window": 1, "winner": 0, "state_summary": "state one"},
                        {"window": 2, "winner": 1, "state_summary": "state two"},
                    ],
                },
                "canonical_ledger": [
                    [1, {"entries": [{"id": "L1"}]}],
                    [2, {"entries": [{"id": "L2"}]}],
                ],
            },
        }

    def test_recent_tail_uses_cumulative_checkpoint_and_exact_last_window(self) -> None:
        messages = replay.render_mode_messages(
            self.record, "recent-exact-tail", recent_exact_tail=1
        )
        middle = "\n".join(
            part["text"] for message in messages[3:-1] for part in message["content"]
        )
        self.assertIn('source_window="1"', middle)
        self.assertIn("state one", middle)
        self.assertIn("window two", middle)
        self.assertNotIn("window one", middle)
        self.assertEqual(messages[:3], self.record["request"]["messages"][:3])
        self.assertEqual(messages[-1], self.record["request"]["messages"][-1])

    def test_checkpoint_at_interval_freezes_state_and_ledger(self) -> None:
        self.record["state"]["supervisor_history"]["selected_windows"].append(
            {"window": 3, "winner": 0, "state_summary": "state three"}
        )
        self.record["state"]["selected_trajectory_windows"].append(
            [text_message("assistant", "window three")]
        )
        self.record["state"]["canonical_ledger"].append(
            [3, {"entries": [{"id": "L3"}]}]
        )
        messages = replay.render_mode_messages(
            self.record, "checkpoint-plus-delta", checkpoint_interval=3
        )
        self.assertEqual(len(messages), 5)
        checkpoint = messages[3]["content"][0]["text"]
        self.assertIn('source_window="3"', checkpoint)
        self.assertIn('"id": "L3"', checkpoint)
        self.assertNotIn("window three", checkpoint)

    def test_full_is_an_unmodified_copy(self) -> None:
        self.record["state"]["selected_trajectory_windows"] = []
        self.record["state"]["supervisor_history"]["selected_windows"] = []
        expected = copy.deepcopy(self.record["request"]["messages"])
        actual = replay.render_mode_messages(self.record, "full")
        self.assertEqual(actual, expected)
        self.assertIsNot(actual, self.record["request"]["messages"])

    def test_bootstrap_preserves_initial_until_cumulative_state_exists(self) -> None:
        self.record["state"]["selected_trajectory_windows"] = []
        self.record["state"]["supervisor_history"]["selected_windows"] = []
        self.record["state"]["canonical_ledger"] = []
        messages = replay.render_mode_messages(self.record, "latest-state")
        self.assertEqual(messages[3], self.record["request"]["messages"][3])
        self.assertIn("No prior selected window", messages[4]["content"][0]["text"])

        legacy = replay.render_mode_messages(
            self.record, "latest-state", preserve_bootstrap_initial=False
        )
        self.assertNotIn(
            "initial evidence",
            "\n".join(
                part["text"] for message in legacy for part in message["content"]
            ),
        )

    def test_groups_only_first_ordinary_request(self) -> None:
        rows = [
            {"type": "asgard_supervisor_prompt_mode", **self.record["prompt"]},
            {"type": "asgard_supervisor_replay_state", **self.record["state"]},
            {
                "type": "asgard_supervisor_replay_request",
                "decision_call": "supervisor",
                "call_index": 1,
                **self.record["request"],
            },
            {
                "type": "asgard_supervisor_replay_request",
                "decision_call": "supervisor",
                "call_index": 2,
            },
            {
                "type": "asgard_supervisor_replay_response",
                "call_index": 2,
                "response": {"kind": "tool_calls"},
            },
            {
                "type": "asgard_decision",
                "call": "supervisor",
                "decision": {"winner": 1, "complete": False},
            },
        ]
        records = replay.captured_routing_states(rows)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["prompt"]["window"], 3)
        self.assertEqual(records[0]["control_decision"]["winner"], 1)


if __name__ == "__main__":
    unittest.main()
