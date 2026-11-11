from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import model_path

import numpy as np

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_EPCNNBE_REPO", "EPCNNBE", "repo"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    import EPCNNBE_web as epcnnbe  # noqa: PLC0415

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v0_epcnnbe"
    predictions: list[dict[str, str]] = []
    raw_rows: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    pred_idx = 0

    for editor_class, be_name in [("ABE", "abe"), ("CBE", "cbe")]:
        selected = []
        for row in rows:
            target = (row.get("sequence_context") or "").upper()
            outcome = (row.get("reference_outcome") or "").upper()
            if row.get("editor_class") != editor_class:
                continue
            if len(target) != 30 or len(outcome) != 30 or not dna_only(target) or not dna_only(outcome):
                skips.append({"benchmark_id": row["benchmark_id"], "reason": f"EPCNNBE pilot requires 30nt target and 30nt outcome for {editor_class}"})
                continue
            selected.append(row)

        if not selected:
            continue

        seqs_all = np.array([[row["sequence_context"].upper(), row["reference_outcome"].upper()] for row in selected])
        target_oh, outcome_oh = epcnnbe.onehot_encoding(seqs_all, 0, 30)
        test = {"tar_oh": target_oh, "oc_oh": outcome_oh, "seqs": seqs_all}
        raw = epcnnbe.ensemble_test_request(be_name, 1, 0, test, str(args.repo / "be_weights") + "\\")

        for row, result in zip(selected, raw):
            pred_idx += 1
            raw_record = {
                "benchmark_id": row["benchmark_id"],
                "extend target": str(result[0]),
                "outcome": str(result[1]),
                "edit efficiency": clean_float(result[2]),
                "outcome proportion": clean_float(result[3]),
                "outcome frequecy": clean_float(result[4]),
                "picking score": clean_float(result[5]),
                "editor_class": editor_class,
            }
            raw_rows.append(raw_record)
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_epcnnbe_{pred_idx:05d}",
                    benchmark_id=row["benchmark_id"],
                    model_id="epcnnbe",
                    run_id=run_id,
                    access_mode="local",
                    task_type="outcome_distribution",
                    predicted_efficiency=raw_record["edit efficiency"],
                    predicted_outcome=raw_record["outcome"],
                    predicted_outcome_frequency=raw_record["outcome frequecy"],
                    predicted_outcome_proportion=raw_record["outcome proportion"],
                    predicted_score=raw_record["picking score"],
                    model_editor=editor_class,
                    model_cell_context=row.get("cell_context", ""),
                    raw_output_path=str(out_dir / "raw_epcnnbe_pilot.csv"),
                    model_version="local adapter pilot_v0",
                    run_timestamp=run_ts,
                    notes="Pilot uses provided target/outcome pairs, not full EPCNNBE-generated outcome universe; efficiency is for adapter smoke testing only.",
                )
            )

    write_csv(out_dir / "raw_epcnnbe_pilot.csv", raw_rows, ["benchmark_id", "extend target", "outcome", "edit efficiency", "outcome proportion", "outcome frequecy", "picking score", "editor_class"])
    write_csv(out_dir / "predictions_epcnnbe_pilot.csv", predictions)
    write_csv(out_dir / "skipped_epcnnbe_pilot.csv", skips, ["benchmark_id", "reason"])
    print(json.dumps({"model_id": "epcnnbe", "predictions": len(predictions), "skipped": len(skips), "output": str(out_dir / "predictions_epcnnbe_pilot.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

