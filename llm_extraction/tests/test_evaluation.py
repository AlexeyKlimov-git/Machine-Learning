import unittest

from pydantic import ValidationError
from src.metrics import score
from src.prompts import messages
from src.schema import parse


class EvaluationTests(unittest.TestCase):
    def test_exact_and_wrong_value(self):
        gold = {"product": "card", "amount": 100.0, "category": "problem"}
        result = score([{"expected": gold, "prediction": {**gold, "amount": 200.0}}])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 1))
        self.assertAlmostEqual(result["field_micro_f1"], 2 / 3)

    def test_invalid_not_dropped(self):
        result = score(
            [
                {
                    "expected": {"product": "card", "amount": None, "category": None},
                    "prediction": None,
                }
            ]
        )
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["valid_response_rate"], 0)

    def test_schema_rejects_extra_missing_and_negative(self):
        for text in (
            "{}",
            '{"product":null,"amount":-1,"category":null}',
            '{"product":null,"amount":null,"category":null,"extra":1}',
            "```json\n{}\n```",
        ):
            with self.assertRaises(ValidationError):
                parse(text)

    def test_no_reward_for_empty_pairs(self):
        empty = {"product": None, "amount": None, "category": None}
        result = score([{"expected": empty, "prediction": empty}])
        self.assertEqual(result["field_micro_f1"], 0)
        self.assertEqual(result["record_exact_match"], 1)

    def test_examples_stay_separate(self):
        context = messages(
            "test ticket", [{"text": "dev ticket", "expected": {"amount": None}}]
        )
        self.assertEqual(
            [r["role"] for r in context], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(context[-1]["content"], "test ticket")
