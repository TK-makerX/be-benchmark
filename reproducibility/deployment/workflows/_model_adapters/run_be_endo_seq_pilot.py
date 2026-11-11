from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import model_path

import pandas as pd
import tensorflow as tf

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


def editor_for_be_endo(row: dict[str, str]) -> str | None:
    editor = (row.get("editor") or row.get("editor_class") or "").upper()
    if "ABE" in editor:
        return "ABE"
    if "CBE" in editor or "BE4" in editor or "C>" in (row.get("intended_edit") or "").upper():
        return "CBE"
    return None


def is_valid_be_endo_sequence(seq: str) -> bool:
    seq = seq.upper()
    return len(seq) == 40 and dna_only(seq) and "N" not in seq and seq[30:33] in {"AGG", "CGG", "TGG", "GGG"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_BE_ENDO_REPO", "BE_Endo", "repo"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from BE_endo_Pack.Effiency import Eff as Effiency  # noqa: PLC0415
    from BE_endo_Pack.Preprocess import Preprocessing as Prepro  # noqa: PLC0415

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v2_be_endo_seq"

    grouped: dict[str, list[dict[str, str]]] = {"ABE": [], "CBE": []}
    skips: list[dict[str, str]] = []
    for row in rows:
        benchmark_id = row["benchmark_id"]
        seq = (row.get("sequence_context") or "").upper()
        editor = editor_for_be_endo(row)
        if editor is None:
            skips.append({"benchmark_id": benchmark_id, "reason": "no BE_Endo editor mapping"})
            continue
        if not is_valid_be_endo_sequence(seq):
            skips.append({"benchmark_id": benchmark_id, "reason": "BE_Endo BE_Seq requires 40bp sequence with NGG PAM at positions 31-33"})
            continue
        grouped[editor].append(dict(row))

    raw_frames: list[pd.DataFrame] = []
    predictions: list[dict[str, str]] = []
    pred_idx = 0

    for editor, editor_rows in grouped.items():
        if not editor_rows:
            continue
        input_path = out_dir / f"raw_be_endo_seq_input_{editor}.tsv"
        with input_path.open("w", encoding="utf-8", newline="") as handle:
            for idx, row in enumerate(editor_rows):
                pos = f"chrPilot:{1000 + idx * 100}-{1020 + idx * 100}"
                handle.write(f"{row['sequence_context'].upper()}\t{pos}\n")

        processor = Prepro.input_process(str(input_path), editor)
        seq_40s, targets, labels, _positions = processor.process()
        model_path = args.repo / "model_data_seq" / f"{editor}_Seq" / f"{editor}_Seq.h5"
        eff_prediction = Effiency.pip_seq(str(model_path), seq_40s, labels, editor)

        columns = [f"base{i}" for i in range(1, 21)]
        eff_table = pd.DataFrame(eff_prediction, columns=columns)
        eff_table.insert(0, "target", targets)
        eff_table.insert(0, "seq", seq_40s)
        eff_table.insert(0, "benchmark_id", [row["benchmark_id"] for row in editor_rows])
        eff_table.insert(1, "editor", editor)
        raw_frames.append(eff_table)

        for row, target, values in zip(editor_rows, targets, eff_prediction):
            for pos, score in enumerate(values, start=1):
                pred_idx += 1
                predictions.append(
                    prediction_row(
                        prediction_id=f"pilot_be_endo_seq_{pred_idx:05d}",
                        benchmark_id=row["benchmark_id"],
                        model_id="be_endo",
                        run_id=run_id,
                        access_mode="local",
                        task_type="position_rate",
                        predicted_position=pos,
                        predicted_score=clean_float(score),
                        model_editor=f"{editor}_BE_Seq",
                        model_cell_context=row.get("cell_context", ""),
                        raw_output_path=str(out_dir / "raw_be_endo_seq_pilot.tsv"),
                        model_version="local sequence-only BE_Seq model; adapter pilot_v2",
                        run_timestamp=run_ts,
                        notes=(
                            "BE_Endo sequence-only BE_Seq adapter. "
                            "Full endogenous-factor BE_Endo mode is not included because external endo_factors.bed/bedtools are not configured."
                        ),
                    )
                )
        tf.keras.backend.clear_session()

    if raw_frames:
        pd.concat(raw_frames, ignore_index=True, sort=False).to_csv(out_dir / "raw_be_endo_seq_pilot.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "raw_be_endo_seq_pilot.tsv", sep="\t", index=False)
    write_csv(out_dir / "predictions_be_endo_seq_pilot.csv", predictions)
    write_csv(out_dir / "skipped_be_endo_seq_pilot.csv", skips, ["benchmark_id", "reason"])
    print(
        json.dumps(
            {
                "model_id": "be_endo",
                "predictions": len(predictions),
                "skipped": len(skips),
                "output": str(out_dir / "predictions_be_endo_seq_pilot.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
