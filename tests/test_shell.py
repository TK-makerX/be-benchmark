from __future__ import annotations

import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from be_benchmark_shell.contracts import PredictionRequest
from be_benchmark_shell.registry import RegistryError
from be_benchmark_shell.service import evaluate_request


class ShellContractTests(unittest.TestCase):
    def test_valid_request(self):
        request = PredictionRequest.build("ACGT" * 6, "example", "example_operation")
        self.assertEqual(request.public_summary()["sequence_length"], 24)

    def test_invalid_sequence(self):
        with self.assertRaises(ValueError):
            PredictionRequest.build("X" * 24, "example", "example_operation")

    def test_empty_registry_never_predicts(self):
        with patch.dict(os.environ, {}, clear=True):
            payload = evaluate_request("ACGT" * 6, "example", "example_operation")
        self.assertEqual(payload["status"], "backend_not_configured")
        self.assertNotIn("prediction", payload)

    def test_missing_explicit_registry_is_not_silently_ignored(self):
        missing = ROOT / "missing-private-registry.json"
        with patch.dict(os.environ, {"BE_SHELL_REGISTRY": str(missing)}, clear=True):
            with self.assertRaises(RegistryError) as context:
                evaluate_request("ACGT" * 6, "example", "example_operation")
        self.assertEqual(str(context.exception), "configured registry is unavailable")
        self.assertNotIn(str(missing), str(context.exception))

    def test_invalid_explicit_registry_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.json"
            registry.write_text('{"schema_version":"0","backends":[]}', encoding="utf-8")
            with self.assertRaises(RegistryError) as context:
                evaluate_request("ACGT" * 6, "example", "example_operation", registry)
        self.assertEqual(str(context.exception), "configured registry is invalid")


if __name__ == "__main__":
    unittest.main()
