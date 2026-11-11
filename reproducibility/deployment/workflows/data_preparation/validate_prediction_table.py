from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from runtime_paths import REPORT_ROOT, SCHEMA_ROOT


ROOT = REPORT_ROOT.parent
DEFAULT_PREDICTIONS = REPORT_ROOT / "pilot_v0_predictions.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_or_empty(value: str) -> bool:
    if value == "" or value is None:
        return True
    try:
        v = float(value)
    except ValueError:
        return False
    return not (math.isnan(v) or math.isinf(v))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    schema = load_json(SCHEMA_ROOT / "prediction_schema.json")
    rows = read_csv(args.predictions)
    required = schema["required_columns"]
    errors: list[str] = []
    warnings: list[str] = []

    if not rows:
        errors.append("prediction table has no rows")
    else:
        columns = set(rows[0].keys())
        for col in required:
            if col not in columns:
                errors.append(f"missing required column: {col}")

    ids = set()
    numeric_cols = [
        "predicted_efficiency",
        "predicted_position",
        "predicted_outcome_frequency",
        "predicted_outcome_proportion",
        "predicted_score",
    ]
    for i, row in enumerate(rows, start=2):
        pid = row.get("prediction_id", "")
        if not pid:
            errors.append(f"row {i}: empty prediction_id")
        elif pid in ids:
            errors.append(f"row {i}: duplicate prediction_id {pid}")
        ids.add(pid)

        for col in required:
            if not row.get(col, ""):
                errors.append(f"row {i}: empty required column {col}")
        for col in numeric_cols:
            if not numeric_or_empty(row.get(col, "")):
                errors.append(f"row {i}: non-numeric {col}={row.get(col)!r}")

        if row.get("task_type") == "efficiency" and not row.get("predicted_efficiency"):
            warnings.append(f"row {i}: efficiency task without predicted_efficiency")
        if row.get("task_type") == "outcome_distribution" and not (row.get("predicted_outcome_frequency") or row.get("predicted_outcome_proportion") or row.get("predicted_score")):
            warnings.append(f"row {i}: outcome_distribution without frequency/proportion/score")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_table": str(args.predictions),
        "row_count": len(rows),
        "counts_by_model": dict(Counter(row.get("model_id", "") for row in rows)),
        "counts_by_task": dict(Counter(row.get("task_type", "") for row in rows)),
        "errors": errors,
        "warnings": warnings[:100],
        "warning_count": len(warnings),
        "ok": not errors,
    }
    out = args.report or REPORT_ROOT / "pilot_v0_prediction_validation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
