"""Run the public interface with generated input and an explicitly empty registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from be_benchmark_shell.service import evaluate_request


def main() -> int:
    payload = evaluate_request(
        "ACGT" * 6,
        "example",
        "example_operation",
        registry=ROOT / "configs" / "backend_registry.template.json",
    )
    if (
        payload.get("status") != "backend_not_configured"
        or payload.get("backend_id") is not None
        or "prediction" in payload
    ):
        raise RuntimeError("This demonstration requires the bundled empty registry")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
