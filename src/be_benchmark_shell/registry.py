from __future__ import annotations

import json
import os
from pathlib import Path

from .backends import BackendSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "configs" / "backend_registry.template.json"


def registry_path() -> Path:
    return Path(os.environ.get("BE_SHELL_REGISTRY", str(DEFAULT_REGISTRY))).expanduser().resolve()


def load_backend_specs(path: Path | None = None) -> list[BackendSpec]:
    selected = path or registry_path()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1":
        raise ValueError("unsupported backend registry schema")
    records = payload.get("backends")
    if not isinstance(records, list):
        raise ValueError("backends must be a list")
    specs = []
    for record in records:
        specs.append(
            BackendSpec(
                backend_id=str(record["backend_id"]),
                enabled=bool(record.get("enabled", False)),
                editor_families=tuple(str(value).lower() for value in record.get("editor_families", [])),
                tasks=tuple(str(value).lower() for value in record.get("tasks", [])),
                connector=str(record.get("connector", "")),
            )
        )
    return specs
