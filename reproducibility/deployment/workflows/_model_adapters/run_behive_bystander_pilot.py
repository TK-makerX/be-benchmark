from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import model_path

import pandas as pd

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


def json_safe(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_BEHIVE_BYSTANDER_REPO", "BE-Hive-bystander", "repo"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import predict as bystander_model  # noqa: PLC0415

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v1_behive_bystander"

    supported = {
        (row["Public base editor"], row["Celltype"])
        for _, row in bystander_model.models_design.iterrows()
    }

    predictions: list[dict[str, str]] = []
    raw_frames: list[pd.DataFrame] = []
    stats_records: list[dict] = []
    skips: list[dict[str, str]] = []
    pred_idx = 0
    initialized: tuple[str, str] | None = None

    for row in rows:
        benchmark_id = row["benchmark_id"]
        seq = (row.get("sequence_context") or "").upper()
        base_editor = row.get("editor", "")
        celltype = row.get("cell_context", "")
        if len(seq) != 50 or not dna_only(seq) or "N" in seq:
            skips.append({"benchmark_id": benchmark_id, "reason": "BE-Hive bystander requires concrete 50nt sequence"})
            continue
        if (base_editor, celltype) not in supported:
            skips.append({"benchmark_id": benchmark_id, "reason": f"unsupported BE-Hive bystander editor/celltype: {base_editor}/{celltype}"})
            continue

        if initialized != (base_editor, celltype):
            bystander_model.init_model(base_editor=base_editor, celltype=celltype)
            initialized = (base_editor, celltype)

        pred_df, stats = bystander_model.predict(seq)
        pred_df = bystander_model.add_genotype_column(pred_df, stats)
        pred_df.insert(0, "benchmark_id", benchmark_id)
        pred_df.insert(1, "base_editor", base_editor)
        pred_df.insert(2, "celltype", celltype)
        raw_frames.append(pred_df)
        stats_record = {k: json_safe(v) for k, v in stats.items()}
        stats_record["benchmark_id"] = benchmark_id
        stats_records.append(stats_record)

        for _, pred in pred_df.iterrows():
            pred_idx += 1
            freq = clean_float(pred.get("Predicted frequency"))
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_behive_bystander_{pred_idx:05d}",
                    benchmark_id=benchmark_id,
                    model_id="behive_bystander",
                    run_id=run_id,
                    access_mode="local",
                    task_type="outcome_distribution",
                    predicted_outcome=pred.get("Genotype", ""),
                    predicted_outcome_frequency=freq,
                    predicted_score=freq,
                    model_editor=base_editor,
                    model_cell_context=celltype,
                    raw_output_path=str(out_dir / "raw_behive_bystander_pilot.csv"),
                    model_version="local pretrained PyTorch model; adapter pilot_v1",
                    run_timestamp=run_ts,
                    notes=(
                        "BE-Hive bystander native 50nt input. Predicted frequency is probability mass over enumerated edited genotypes; "
                        "the no-edit genotype is not included."
                    ),
                )
            )

    if raw_frames:
        pd.concat(raw_frames, ignore_index=True, sort=False).to_csv(out_dir / "raw_behive_bystander_pilot.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "raw_behive_bystander_pilot.csv", index=False)

    (out_dir / "raw_behive_bystander_stats.json").write_text(
        json.dumps(stats_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "predictions_behive_bystander_pilot.csv", predictions)
    write_csv(out_dir / "skipped_behive_bystander_pilot.csv", skips, ["benchmark_id", "reason"])
    print(
        json.dumps(
            {
                "model_id": "behive_bystander",
                "predictions": len(predictions),
                "skipped": len(skips),
                "output": str(out_dir / "predictions_behive_bystander_pilot.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
