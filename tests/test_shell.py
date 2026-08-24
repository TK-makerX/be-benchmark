from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from be_benchmark_shell.contracts import PredictionRequest
from be_benchmark_shell.service import evaluate_request


class ShellContractTests(unittest.TestCase):
    def test_valid_request(self):
        request = PredictionRequest.build("ACGT" * 6, "example", "example_operation")
        self.assertEqual(request.public_summary()["sequence_length"], 24)

    def test_invalid_sequence(self):
        with self.assertRaises(ValueError):
            PredictionRequest.build("X" * 24, "example", "example_operation")

    def test_empty_registry_never_predicts(self):
        payload = evaluate_request("ACGT" * 6, "example", "example_operation")
        self.assertEqual(payload["status"], "backend_not_configured")
        self.assertNotIn("prediction", payload)


if __name__ == "__main__":
    unittest.main()
