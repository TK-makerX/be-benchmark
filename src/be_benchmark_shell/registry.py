from __future__ import annotations

import json
import os
from pathlib import Path

from .backends import BackendSpec


class RegistryError(RuntimeError):
    """Raised when an explicitly configured registry cannot be used."""


def registry_path() -> Path | None:
    configured = os.environ.get("BE_SHELL_REGISTRY")
    if not configured:
        return None
    return Path(configured).expanduser().resolve()


def load_backend_specs(path: Path | None = None) -> list[BackendSpec]:
    selected = path if path is not None else registry_path()
    if selected is None:
        return []
    try:
        raw = selected.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError("configured registry is unavailable") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError("configured registry is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise RegistryError("configured registry is invalid")
    records = payload.get("backends")
    if not isinstance(records, list):
        raise RegistryError("configured registry is invalid")
    specs = []
    for record in records:
        try:
            specs.append(
                BackendSpec(
                    backend_id=str(record["backend_id"]),
                    enabled=bool(record.get("enabled", False)),
                    editor_families=tuple(str(value).lower() for value in record.get("editor_families", [])),
                    tasks=tuple(str(value).lower() for value in record.get("tasks", [])),
                    connector=str(record.get("connector", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError("configured registry is invalid") from exc
    return specs
