from __future__ import annotations
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

def _root(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()

def model_path(variable: str, *relative: str) -> Path:
    models_root = _root("BE_MODELS_DIR", REPOSITORY_ROOT / "models")
    return _root(variable, models_root.joinpath(*relative))

def data_path(variable: str, *relative: str) -> Path:
    data_root = _root("BE_DATA_DIR", REPOSITORY_ROOT / "data" / "full")
    return _root(variable, data_root.joinpath(*relative))

def tools_path() -> Path:
    return _root("BE_TOOLS_DIR", REPOSITORY_ROOT / "tools" / "bin")
