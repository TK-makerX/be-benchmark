"""Portable path resolution for the public release."""
from __future__ import annotations
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()

DATA_DIR = env_path("BE_DATA_DIR", REPOSITORY_ROOT / "data" / "full")
RESULTS_DIR = env_path("BE_RESULTS_DIR", REPOSITORY_ROOT / "results")
MODELS_DIR = env_path("BE_MODELS_DIR", REPOSITORY_ROOT / "models")
TOOLS_DIR = env_path("BE_TOOLS_DIR", REPOSITORY_ROOT / "tools")
