import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parent


class ReplayCaptureSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(
            (ROOT / "replay_capture.schema.json").read_text(encoding="utf-8")
        )
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

    def test_fast_path_records_validate(self):
        usage = {
            "input": 10,
            "output": 2,
            "thought": 1,
            "cachedRead": 8,
            "cachedWrite": 0,
        }
        records = [
            {
                "type": "asgard_fast_path",
                "window": 7,
                "eligible": True,
                "reasons": [],
                "used": False,
                "fallback_reason": "fast-path selector reported uncertainty",
                "usage": usage,
            },
            {
                "type": "asgard_supervisor_fast_path_request",
                "model": "deepseek::deepseek-v4-pro",
                "messages": [],
                "tools": [{"type": "function"}],
            },
            {
                "type": "asgard_supervisor_fast_path_response",
                "response": {
                    "type": "asgard_supervisor_replay_response",
                    "call_index": 1,
                    "response": {"kind": "text", "text": "FULL_SUPERVISOR_REQUIRED"},
                    "usage": usage,
                },
            },
        ]
        for record in records:
            with self.subTest(record=record["type"]):
                self.validator.validate(record)

    def test_fast_path_usage_is_required(self):
        record = {
            "type": "asgard_fast_path",
            "window": 1,
            "eligible": False,
            "reasons": ["not_single_lane"],
            "used": False,
            "fallback_reason": "ineligible",
        }
        with self.assertRaises(jsonschema.ValidationError):
            self.validator.validate(record)

    def test_policy_and_candidate_usage_records_validate(self):
        usage = {
            "input": 10,
            "output": 2,
            "thought": 1,
            "cachedRead": 8,
            "cachedWrite": 0,
        }
        records = [
            {
                "type": "asgard_window_policy_config",
                "mode": "explicit-probe",
                "shadow_survivor_study": False,
                "shadow_probe_steps": 2,
            },
            {
                "type": "asgard_candidate_window_usage",
                "window": 1,
                "candidate_count": 1,
                "window_steps": 2,
                "lanes": [{"lane": 0, "model": "deepseek::flash", "usage": usage}],
                "usage": usage,
            },
        ]
        for record in records:
            with self.subTest(record=record["type"]):
                self.validator.validate(record)


if __name__ == "__main__":
    unittest.main()
