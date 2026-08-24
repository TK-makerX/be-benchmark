from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WebContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        env = os.environ.copy()
        env.pop("BE_SHELL_REGISTRY", None)
        env["BE_WEB_HOST"] = "127.0.0.1"
        env["BE_WEB_PORT"] = str(cls.port)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cls.process = subprocess.Popen(
            [sys.executable, str(ROOT / "app" / "run_web_app.py")],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        scheme = "http"
        cls.base_url = f"{scheme}://127.0.0.1:{cls.port}"
        deadline = time.time() + 10
        while time.time() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError("web process exited during startup")
            try:
                with urllib.request.urlopen(cls.base_url + "/health", timeout=1) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("web process did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)

    def post(self, fields: dict[str, str]) -> tuple[int, str]:
        body = urllib.parse.urlencode(fields).encode()
        request = urllib.request.Request(self.base_url + "/", data=body, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def test_health(self):
        with urllib.request.urlopen(self.base_url + "/health", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read().decode()), {"status": "ok"})

    def test_empty_backend_post(self):
        status, body = self.post(
            {"sequence": "ACGT" * 6, "editor_family": "example", "task": "example_operation"}
        )
        self.assertEqual(status, 503)
        self.assertIn("backend_not_configured", body)
        self.assertNotIn("&quot;prediction&quot;", body)

    def test_invalid_post(self):
        status, body = self.post(
            {"sequence": "X" * 24, "editor_family": "example", "task": "example_operation"}
        )
        self.assertEqual(status, 400)
        self.assertIn("invalid_request", body)


if __name__ == "__main__":
    unittest.main()
