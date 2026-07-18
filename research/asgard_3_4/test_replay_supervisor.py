import json
import unittest

import replay_supervisor as replay


class ReplaySupervisorTest(unittest.TestCase):
    def test_scores_selection_and_deepseek_cache_usage(self) -> None:
        record = {
            "prompt": {"window": 4, "mode": "full"},
            "request": {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "s"}]},
                    {"role": "user", "content": [{"type": "text", "text": "task"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "checklist"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "<selected_trajectory_initial>\n\n</selected_trajectory_initial>",
                            }
                        ],
                    },
                    {"role": "user", "content": [{"type": "text", "text": "current"}]},
                ]
            },
            "state": {
                "selected_trajectory_initial": [],
                "selected_trajectory_windows": [],
                "supervisor_history": {"checkpointed": [], "selected_windows": []},
                "canonical_ledger": [],
            },
            "first_response": {
                "usage": {"input": 30, "cachedRead": 70, "cachedWrite": 0}
            },
            "control_decision": {
                "winner": 1,
                "complete": False,
                "next_candidate_count": 2,
                "next_window_steps": 3,
            },
        }
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "select_trajectory",
                                    "arguments": json.dumps(
                                        {
                                            "winner": 1,
                                            "complete": False,
                                            "next_candidate_count": 2,
                                            "next_window_steps": 5,
                                        }
                                    ),
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 60,
                "prompt_cache_miss_tokens": 40,
                "completion_tokens": 12,
            },
        }
        result = replay.score_record(record, "full", response)
        self.assertTrue(result["winner_agreement"])
        self.assertTrue(result["complete_agreement"])
        self.assertFalse(result["step_count_agreement"])
        self.assertEqual(result["usage"]["input"], 40)
        self.assertEqual(result["usage"]["cachedRead"], 60)
        self.assertEqual(result["raw_input_tokens"], 100)
        self.assertEqual(result["full_control_raw_input_tokens"], 100)
        self.assertEqual(result["prompt_byte_reduction_fraction"], 0)

    def test_audit_call_requires_full_fallback(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "grep_search",
                                    "arguments": "{}",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        selection, status = replay.parse_selection(response)
        self.assertIsNone(selection)
        self.assertEqual(status, "audit_or_no_selection:grep_search")


if __name__ == "__main__":
    unittest.main()
