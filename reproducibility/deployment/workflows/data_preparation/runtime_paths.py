"""Portable path configuration for the public data-preparation utilities."""
from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


PROJECT_ROOT = env_path("BE_PROJECT_ROOT", REPOSITORY_ROOT)
CORE_ROOT = env_path("BE_BENCHMARK_CORE", REPOSITORY_ROOT)
DATA_ROOT = env_path("BE_DATA_DIR", CORE_ROOT / "data")
REPORT_ROOT = env_path("BE_REPORTS_DIR", CORE_ROOT / "reports")
MODELS_ROOT = env_path("BE_MODELS_DIR", REPOSITORY_ROOT / "models")
SCHEMA_ROOT = env_path("BE_SCHEMA_DIR", REPOSITORY_ROOT / "src" / "be_benchmark" / "schemas")
BEDICT_SUPPLEMENT = env_path(
    "BE_BEDICT_SUPPLEMENT",
    MODELS_ROOT / "BEDICT-V2" / "resources" / "paper_supplements" / "13059_2025_3586_MOESM4_ESM.xlsx",
)
CRISPRON_DEV_DATA = env_path(
    "BE_CRISPRON_DEV_DATA",
    MODELS_ROOT / "CRISPRon-BE" / "repo" / "data_for_model_development_V1.0",
)
