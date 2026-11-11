from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import model_path

import pandas as pd
import torch

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


def infer_pam4(sequence_context: str, protospacer: str, pam: str) -> str:
    sequence_context = (sequence_context or "").upper()
    protospacer = (protospacer or "").upper()
    pam = (pam or "").upper()
    if len(pam) == 4 and dna_only(pam):
        return pam
    idx = sequence_context.find(protospacer)
    if idx >= 0:
        candidate = sequence_context[idx + len(protospacer): idx + len(protospacer) + 4]
        if len(candidate) == 4 and dna_only(candidate):
            return candidate
    if len(pam) == 3 and dna_only(pam):
        return pam + "N"
    return ""


def editor_to_bedict_model(row: dict[str, str]) -> str | None:
    editor = (row.get("editor") or "").upper()
    cas = (row.get("cas_variant") or "").upper()
    if "ABE8E" in editor and "SPRY" in cas:
        return "ABE8e-SpRY"
    if "ABE8E" in editor and "NG" in cas:
        return "ABE8e-NG"
    if "ABEMAX" in editor and "SPRY" in cas:
        return "ABEmax-SpRY"
    if "ABEMAX" in editor and "NG" in cas:
        return "ABEmax-NG"
    if "SPRY-ABE8E" in editor or "ABE8E-SPRY" in editor:
        return "ABE8e-SpRY"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_BEDICT_REPO", "BEDICT-V2", "repo"))
    parser.add_argument("--max-inputs-per-editor", type=int, default=20)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from utils.sequence_process import HaplotypeSeqProcessor, SeqProcessConfig  # noqa: PLC0415
    from proportion_model.src.predict_model import BEDICT_EncEnc_HaplotypeModel  # noqa: PLC0415

    target_conv_nucl = {
        "ABEmax-NG": ("A", "G"),
        "ABE8e-NG": ("A", "G"),
        "ABE8e-SpCas9": ("A", "G"),
        "ABE8e-SpRY": ("A", "G"),
        "ABEmax-SpCas9": ("A", "G"),
        "ABEmax-SpRY": ("A", "G"),
    }

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v0_bedict_v2"
    predictions: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    raw_tables: list[Path] = []
    pred_idx = 0

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        model_editor = editor_to_bedict_model(row)
        protospacer = (row.get("protospacer_20nt") or "").upper()
        pam4 = infer_pam4(row.get("sequence_context", ""), protospacer, row.get("pam", ""))
        if model_editor is None:
            skips.append({"benchmark_id": row["benchmark_id"], "reason": "no supported BEDICT-V2 ABE editor mapping"})
            continue
        if len(protospacer) != 20 or len(pam4) != 4 or not dna_only(protospacer) or not dna_only(pam4):
            skips.append({"benchmark_id": row["benchmark_id"], "reason": "BEDICT-V2 requires 20nt protospacer plus 4nt PAM"})
            continue
        row = dict(row)
        row["_pam4"] = pam4
        grouped.setdefault(model_editor, []).append(row)

    seqconfig = SeqProcessConfig(24, (1, 24), (1, 20), 1)
    device = torch.device("cpu")

    for model_editor, editor_rows in grouped.items():
        editor_rows = editor_rows[: args.max_inputs_per_editor]
        input_df = pd.DataFrame(
            {
                "ID": [r["benchmark_id"] for r in editor_rows],
                "Reference": [r["protospacer_20nt"].upper() for r in editor_rows],
                "PAM": [r["_pam4"].upper() for r in editor_rows],
            }
        )
        input_df["protospacer_PAM"] = input_df["Reference"] + input_df["PAM"]
        raw_input_path = out_dir / f"raw_bedict_v2_input_{model_editor}.csv"
        input_df.to_csv(raw_input_path, index=False)

        seq_processor = HaplotypeSeqProcessor(model_editor, target_conv_nucl[model_editor], seqconfig)
        bedict = BEDICT_EncEnc_HaplotypeModel(seq_processor, seqconfig, device)
        model_path = args.repo / "proportion_model" / "output" / "experiment_run_proportions_encenc_two_model" / f"{model_editor}_proportions_encenc_two_model" / "protospacer_PAM" / "exp_version_0" / "train_val"

        combined = pd.DataFrame(columns=["seq_id", "Inp_seq", "Outp_seq"])
        for run in range(3):
            model_dir = model_path / f"run_{run}"
            pred_df = bedict.predict_from_dataframe(input_df, ["ID", "protospacer_PAM"], str(model_dir), outpseq_col=None, outcome_col=None, renormalize=True, batch_size=5)
            combined["seq_id"] = pred_df["seq_id"]
            combined["Inp_seq"] = pred_df["Inp_seq"]
            combined["Outp_seq"] = pred_df["Outp_seq"]
            combined[f"score_run_{run}"] = pred_df["pred_score"]

        score_cols = [f"score_run_{run}" for run in range(3)]
        combined["average_predicted_proportion"] = combined[score_cols].mean(axis=1)
        combined["std_predicted_proportion"] = combined[score_cols].std(axis=1)
        raw_output_path = out_dir / f"raw_bedict_v2_predictions_{model_editor}.csv"
        combined.to_csv(raw_output_path, index=False)
        raw_tables.append(raw_output_path)

        for _, pred in combined.iterrows():
            pred_idx += 1
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_bedict_v2_{pred_idx:05d}",
                    benchmark_id=pred["seq_id"],
                    model_id="bedict_v2",
                    run_id=run_id,
                    access_mode="local",
                    task_type="outcome_distribution",
                    predicted_outcome=pred["Outp_seq"],
                    predicted_outcome_proportion=clean_float(pred["average_predicted_proportion"]),
                    predicted_score=clean_float(pred["average_predicted_proportion"]),
                    model_editor=model_editor,
                    raw_output_path=str(raw_output_path),
                    model_version="local adapter pilot_v0 proportion model only",
                    run_timestamp=run_ts,
                    notes="BEDICT-V2 proportion-model adapter. Absolute efficiency stage is not included in this pilot wrapper.",
                )
            )

    write_csv(out_dir / "predictions_bedict_v2_pilot.csv", predictions)
    write_csv(out_dir / "skipped_bedict_v2_pilot.csv", skips, ["benchmark_id", "reason"])
    print(json.dumps({"model_id": "bedict_v2", "predictions": len(predictions), "skipped": len(skips), "raw_tables": [str(p) for p in raw_tables], "output": str(out_dir / "predictions_bedict_v2_pilot.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

