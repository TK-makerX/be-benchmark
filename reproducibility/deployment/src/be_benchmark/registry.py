"""Load the public model registry."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load_models():
    with (ROOT / "configs" / "model_registry_public.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
