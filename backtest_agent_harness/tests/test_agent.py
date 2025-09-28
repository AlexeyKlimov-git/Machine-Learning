"""Контракт между ответом модели и разрешёнными действиями."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent import explain, propose, validate_plan
from src.runner import run


class AgentTests(unittest.TestCase):
    def test_model_plan_runs_demo_and_receives_real_metrics(self):
        with patch("src.agent.chat", side_effect=['{"steps":[{"tool":"demo","args":{"window":10}}]}', "ok"]) as model:
            plan = propose("Сравни стратегии с окном 10 на демонстрационных ценах", "qwen3.5:9b")
            with tempfile.TemporaryDirectory() as directory:
                result = run(plan, Path(directory) / "run")
                metrics = json.loads((Path(directory) / "run/0/metrics.json").read_text())
                report = explain("Сравни стратегии", plan, result, [metrics], "qwen3.5:9b")
        self.assertTrue(result["ok"])
        self.assertTrue(metrics["synthetic"])
        self.assertEqual(metrics["window"], 10)
        self.assertIn("moving_average_10", metrics)
        self.assertEqual(report, "ok")
        self.assertEqual(model.call_count, 2)
        self.assertIn("buy_and_hold", model.call_args.args[0][1]["content"])

    def test_model_cannot_inject_shell_command(self):
        with self.assertRaises(ValueError):
            validate_plan({"steps": [{"tool": "demo", "command": "echo unsafe"}]})

    def test_model_cannot_invent_extra_backtest_args(self):
        with self.assertRaises(ValueError):
            validate_plan(
                {
                    "steps": [
                        {
                            "tool": "backtest",
                            "args": {
                                "config": "c.json",
                                "from": "2024-01-01T00:00:00Z",
                                "to": "2024-02-01T00:00:00Z",
                                "snapshot": "s",
                                "cache": "cache",
                                "exec": "anything",
                            },
                        }
                    ]
                }
            )
