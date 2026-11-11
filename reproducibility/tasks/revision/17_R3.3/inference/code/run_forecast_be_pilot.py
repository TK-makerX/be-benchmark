from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import clean_float, dna_only, normalize_editor_for_forecast, prediction_row, read_csv, utc_now, write_csv


def choose_20nt_sequence(row: dict[str, str]) -> str:
    protospacer = (row.get("protospacer_20nt") or "").upper()
    sequence = (row.get("sequence_context") or "").upper()
    if len(protospacer) == 20 and dna_only(protospacer):
        return protospacer
    if len(sequence) == 20 and dna_only(sequence):
        return sequence
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    import forecast_be  # noqa: PLC0415

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v0_forecast_be"
    predictions: list[dict[str, str]] = []
    raw_records: list[dict] = []
    skips: list[dict[str, str]] = []

    pred_idx = 0
    for row in rows:
        benchmark_id = row["benchmark_id"]
        target_seq = choose_20nt_sequence(row)
        editor = normalize_editor_for_forecast(row.get("editor_class", ""))
        if not target_seq:
            skips.append({"benchmark_id": benchmark_id, "reason": "no 20nt protospacer or sequence_context"})
            continue
        if editor is None:
            skips.append({"benchmark_id": benchmark_id, "reason": f"unsupported editor_class {row.get('editor_class', '')}"})
            continue

        total = forecast_be.predict_total(target_seq, editor=editor, mean=None, std=None)
        positions = forecast_be.predict(target_seq, editor=editor, mean=None, std=None)
        raw_records.append({"benchmark_id": benchmark_id, "target_seq": target_seq, "editor": editor, "predict_total": float(total), "predict_positions": positions})

        pred_idx += 1
        predictions.append(
            prediction_row(
                prediction_id=f"pilot_forecast_be_total_{pred_idx:05d}",
                benchmark_id=benchmark_id,
                model_id="forecast_be",
                run_id=run_id,
                access_mode="local",
                task_type="efficiency",
                predicted_efficiency=clean_float(total),
                predicted_score=clean_float(total),
                model_editor=editor,
                model_cell_context=row.get("cell_context", ""),
                raw_output_path=str(out_dir / "raw_forecast_be_pilot.json"),
                model_version="local adapter pilot_v0",
                run_timestamp=run_ts,
                notes="FORECasT-BE total efficiency prediction on 20nt target.",
            )
        )

        for pos, score in positions:
            pred_idx += 1
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_forecast_be_pos_{pred_idx:05d}",
                    benchmark_id=benchmark_id,
                    model_id="forecast_be",
                    run_id=run_id,
                    access_mode="local",
                    task_type="position_rate",
                    predicted_position=pos,
                    predicted_score=clean_float(score),
                    model_editor=editor,
                    model_cell_context=row.get("cell_context", ""),
                    raw_output_path=str(out_dir / "raw_forecast_be_pilot.json"),
                    model_version="local adapter pilot_v0",
                    run_timestamp=run_ts,
                    notes="FORECasT-BE position-specific score. Empty score means target base absent at that position.",
                )
            )

    (out_dir / "raw_forecast_be_pilot.json").write_text(json.dumps(raw_records, indent=2), encoding="utf-8")
    write_csv(out_dir / "predictions_forecast_be_pilot.csv", predictions)
    write_csv(out_dir / "skipped_forecast_be_pilot.csv", skips, ["benchmark_id", "reason"])
    print(json.dumps({"model_id": "forecast_be", "predictions": len(predictions), "skipped": len(skips), "output": str(out_dir / "predictions_forecast_be_pilot.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

