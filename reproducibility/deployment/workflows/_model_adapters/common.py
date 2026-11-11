from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path


PREDICTION_FIELDNAMES = [
    "prediction_id",
    "benchmark_id",
    "model_id",
    "run_id",
    "access_mode",
    "task_type",
    "predicted_efficiency",
    "predicted_position",
    "predicted_outcome",
    "predicted_outcome_frequency",
    "predicted_outcome_proportion",
    "predicted_score",
    "model_editor",
    "model_cell_context",
    "raw_output_path",
    "model_version",
    "run_timestamp",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = PREDICTION_FIELDNAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_float(value) -> str:
    if value is None or value == "":
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    return f"{v:.12g}"


def prediction_row(**kwargs) -> dict[str, str]:
    row = {name: "" for name in PREDICTION_FIELDNAMES}
    row.update({k: "" if v is None else str(v) for k, v in kwargs.items()})
    return row


def dna_only(seq: str) -> bool:
    return bool(seq) and set(seq.upper()) <= set("ACGTN")


def normalize_editor_for_forecast(editor_class: str) -> str | None:
    editor_class = (editor_class or "").upper()
    if editor_class == "ABE":
        return "ABE"
    if editor_class == "CBE":
        return "CBE"
    return None

