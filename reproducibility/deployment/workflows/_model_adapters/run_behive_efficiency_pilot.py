from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from runtime_paths import model_path

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


def as_clean_number(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v) or math.isinf(v):
        return ""
    return clean_float(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_BEHIVE_EFFICIENCY_REPO", "BE-Hive-efficiency", "repo"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import predict as be_efficiency_model  # noqa: PLC0415

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v1_behive_efficiency"

    supported = {
        (row["Public base editor"], row["Celltype"])
        for _, row in be_efficiency_model.models_design.iterrows()
    }

    predictions: list[dict[str, str]] = []
    raw_rows: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    pred_idx = 0
    initialized: tuple[str, str] | None = None

    for row in rows:
        benchmark_id = row["benchmark_id"]
        seq = (row.get("sequence_context") or "").upper()
        base_editor = row.get("editor", "")
        celltype = row.get("cell_context", "")
        if len(seq) != 50 or not dna_only(seq) or "N" in seq:
            skips.append({"benchmark_id": benchmark_id, "reason": "BE-Hive efficiency requires concrete 50nt sequence"})
            continue
        if (base_editor, celltype) not in supported:
            skips.append({"benchmark_id": benchmark_id, "reason": f"unsupported BE-Hive efficiency editor/celltype: {base_editor}/{celltype}"})
            continue

        if initialized != (base_editor, celltype):
            be_efficiency_model.init_model(base_editor=base_editor, celltype=celltype)
            initialized = (base_editor, celltype)

        pred = be_efficiency_model.predict(seq)
        logit_score = as_clean_number(pred.get("Predicted logit score"))
        fraction = as_clean_number(pred.get("Predicted fraction of sequenced reads with base editing activity"))
        raw_record = {
            "benchmark_id": benchmark_id,
            "base_editor": base_editor,
            "celltype": celltype,
            "sequence_context": seq,
            "predicted_logit_score": logit_score,
            "predicted_fraction": fraction,
        }
        raw_rows.append(raw_record)

        pred_idx += 1
        predictions.append(
            prediction_row(
                prediction_id=f"pilot_behive_efficiency_{pred_idx:05d}",
                benchmark_id=benchmark_id,
                model_id="behive_efficiency",
                run_id=run_id,
                access_mode="local",
                task_type="efficiency",
                predicted_efficiency=fraction,
                predicted_score=logit_score,
                model_editor=base_editor,
                model_cell_context=celltype,
                raw_output_path=str(out_dir / "raw_behive_efficiency_pilot.csv"),
                model_version="local pretrained sklearn model; adapter pilot_v1",
                run_timestamp=run_ts,
                notes=(
                    "BE-Hive efficiency native 50nt input. "
                    "Predicted logit score is recorded as predicted_score; calibrated fraction is blank unless model calibration parameters are supplied."
                ),
            )
        )

    write_csv(
        out_dir / "raw_behive_efficiency_pilot.csv",
        raw_rows,
        ["benchmark_id", "base_editor", "celltype", "sequence_context", "predicted_logit_score", "predicted_fraction"],
    )
    write_csv(out_dir / "predictions_behive_efficiency_pilot.csv", predictions)
    write_csv(out_dir / "skipped_behive_efficiency_pilot.csv", skips, ["benchmark_id", "reason"])
    print(
        json.dumps(
            {
                "model_id": "behive_efficiency",
                "predictions": len(predictions),
                "skipped": len(skips),
                "output": str(out_dir / "predictions_behive_efficiency_pilot.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
