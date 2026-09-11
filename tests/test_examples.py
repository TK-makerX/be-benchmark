from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocumentedExampleTests(unittest.TestCase):
    def run_example(self, entry_point: str, *arguments: str, registry: str | None = None):
        env = os.environ.copy()
        env.pop("BE_SHELL_REGISTRY", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if registry is not None:
            env["BE_SHELL_REGISTRY"] = registry
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / entry_point), *arguments],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return completed.returncode, json.loads(completed.stdout)

    def test_demo_ignores_private_registry_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_registry = str(Path(directory) / "not-present.json")
            code, payload = self.run_example("examples/run_demo.py", registry=missing_registry)
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "backend_not_configured")
        self.assertEqual(payload["request"]["sequence_length"], 24)
        self.assertIsNone(payload["backend_id"])
        self.assertNotIn("prediction", payload)

    def test_documented_empty_cli(self):
        code, payload = self.run_example(
            "app/run_cli.py",
            "--sequence", "ACGT" * 6,
            "--editor-family", "example",
            "--task", "example_operation",
            registry=str(ROOT / "configs" / "backend_registry.template.json"),
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "backend_not_configured")
        self.assertNotIn("prediction", payload)

    def test_documented_invalid_cli(self):
        code, payload = self.run_example(
            "app/run_cli.py",
            "--sequence", "ACGT",
            "--editor-family", "example",
            "--task", "example_operation",
        )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
