"""Model registry loader for the benchmark-informed router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = SYSTEM_ROOT / "model_registry.json"


def load_registry(path: Path | None = None) -> Dict[str, Any]:
    registry_path = Path(path) if path else REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_models(path: Path | None = None) -> List[Dict[str, Any]]:
    return load_registry(path)["models"]

