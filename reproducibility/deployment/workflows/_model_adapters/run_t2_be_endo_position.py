from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import model_path

import pandas as pd
import tensorflow as tf


ADAPTER_DIR = Path(__file__).resolve().parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from t2_common import clean_float, dna, write_csv  # noqa: E402


def editor_for_be_endo(row: dict[str, str]) -> str | None:
    editor = (row.get("editor") or row.get("editor_class") or "").upper()
    if "ABE" in editor:
        return "ABE"
    if "CBE" in editor or "BE4" in editor or "C>" in (row.get("intended_edit") or "").upper():
        return "CBE"
    return None


def valid_seq40(seq: str) -> bool:
    seq = str(seq).upper()
    return len(seq) == 40 and dna(seq) and seq[30:33] in {"AGG", "CGG", "TGG", "GGG"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_BE_ENDO_REPO", "BE_Endo", "repo"))
    parser.add_argument("--run-id", default="T2_expand_v0_be_endo")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from BE_endo_Pack.Effiency import Eff as Effiency  # noqa: PLC0415
    from BE_endo_Pack.Preprocess import Preprocessing as Prepro  # noqa: PLC0415

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, str]]] = {"ABE": [], "CBE": []}
    skips: list[dict[str, str]] = []
    for _, row in panel.iterrows():
        record = row.to_dict()
        target_id = record["target_profile_id"]
        seq40 = record.get("sequence_context_40nt", "").upper()
        editor = editor_for_be_endo(record)
        if editor is None:
            skips.append({"target_profile_id": target_id, "reason": "no BE_Endo editor mapping"})
            continue
        if not valid_seq40(seq40):
            skips.append({"target_profile_id": target_id, "reason": "BE_Endo BE_Seq requires 40bp sequence with NGG PAM at positions 31-33"})
            continue
        record["sequence_context_40nt"] = seq40
        grouped[editor].append(record)

    raw_frames: list[pd.DataFrame] = []
    position_rows: list[dict] = []
    for editor, rows in grouped.items():
        if not rows:
            continue
        input_path = args.output_dir / f"raw_t2_be_endo_seq_input_{editor}.tsv"
        with input_path.open("w", encoding="utf-8", newline="") as handle:
            for idx, row in enumerate(rows):
                pos = f"chrT2:{1000 + idx * 100}-{1020 + idx * 100}"
                handle.write(f"{row['sequence_context_40nt']}\t{pos}\n")

        processor = Prepro.input_process(str(input_path), editor)
        seq_40s, targets, labels, _positions = processor.process()
        model_path = args.repo / "model_data_seq" / f"{editor}_Seq" / f"{editor}_Seq.h5"
        eff_prediction = Effiency.pip_seq(str(model_path), seq_40s, labels, editor)

        columns = [f"base{i}" for i in range(1, 21)]
        eff_table = pd.DataFrame(eff_prediction, columns=columns)
        eff_table.insert(0, "target", targets)
        eff_table.insert(0, "seq", seq_40s)
        eff_table.insert(0, "target_profile_id", [row["target_profile_id"] for row in rows])
        eff_table.insert(1, "editor", editor)
        raw_frames.append(eff_table)

        for row, values in zip(rows, eff_prediction):
            protospacer = row.get("protospacer_20nt", "").upper()
            relation = row.get("source_relation_to_model_for_t2", "external_candidate")
            for pos, score in enumerate(values, start=1):
                ref_base = protospacer[pos - 1] if len(protospacer) == 20 else ""
                position_rows.append(
                    {
                        "prediction_id": f"be_endo_t2_position_{len(position_rows) + 1:08d}",
                        "target_profile_id": row["target_profile_id"],
                        "model_id": "be_endo",
                        "run_id": args.run_id,
                        "model_editor": f"{editor}_BE_Seq",
                        "model_cell_context": "",
                        "protospacer_position": pos,
                        "reference_base": ref_base,
                        "predicted_position_rate": "",
                        "predicted_position_score": clean_float(score),
                        "prediction_scale": "uncalibrated_position_score",
                        "source_relation_to_model": relation,
                        "notes": "BE_Endo sequence-only BE_Seq output treated as uncalibrated position score in T2.",
                    }
                )
        tf.keras.backend.clear_session()

    if raw_frames:
        pd.concat(raw_frames, ignore_index=True, sort=False).to_csv(args.output_dir / "raw_t2_be_endo_seq.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "raw_t2_be_endo_seq.tsv", sep="\t", index=False)
    write_csv(args.output_dir / "predictions_t2_be_endo_positions.csv", position_rows)
    write_csv(args.output_dir / "skipped_t2_be_endo.csv", skips)
    print(json.dumps({"model_id": "be_endo", "targets": len(panel), "position_predictions": len(position_rows), "skipped": len(skips)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
