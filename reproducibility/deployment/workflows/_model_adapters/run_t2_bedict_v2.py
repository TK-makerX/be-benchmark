from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import model_path

import pandas as pd
import torch

from t2_common import clean_float, dna, events_from_sequences, normalize_outcome_scores, outcome_rows_to_position_rows, write_csv


TARGET_CONV_NUCL = {
    "ABEmax-NG": ("A", "G"),
    "ABE8e-NG": ("A", "G"),
    "ABE8e-SpCas9": ("A", "G"),
    "ABE8e-SpRY": ("A", "G"),
    "ABEmax-SpCas9": ("A", "G"),
    "ABEmax-SpRY": ("A", "G"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_BEDICT_REPO", "BEDICT-V2", "repo"))
    parser.add_argument("--run-id", default="T2_expand_v0_bedict_v2")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from utils.sequence_process import HaplotypeSeqProcessor, SeqProcessConfig  # noqa: PLC0415
    from proportion_model.src.predict_model import BEDICT_EncEnc_HaplotypeModel  # noqa: PLC0415

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seqconfig = SeqProcessConfig(24, (1, 24), (1, 20), 1)
    device = torch.device("cpu")
    skips: list[dict] = []
    raw_paths: list[str] = []
    outcome_rows: list[dict] = []

    grouped: dict[str, pd.DataFrame] = {}
    for _, row in panel.iterrows():
        target_id = row["target_profile_id"]
        protospacer = row.get("protospacer_20nt", "").upper()
        pam4 = row.get("pam4", "").upper()
        model_editor = row.get("t2_model_editor", "") or "ABEmax-SpCas9"
        if model_editor not in TARGET_CONV_NUCL:
            skips.append({"target_profile_id": target_id, "reason": f"unsupported BEDICT-V2 model editor {model_editor}"})
            continue
        if len(protospacer) != 20 or len(pam4) != 4 or not dna(protospacer) or not dna(pam4):
            skips.append({"target_profile_id": target_id, "reason": "BEDICT-V2 requires 20nt protospacer plus 4nt PAM"})
            continue
        grouped.setdefault(model_editor, []).append(row.to_dict())

    pred_idx = 0
    for model_editor, row_dicts in grouped.items():
        input_df = pd.DataFrame(
            {
                "ID": [r["target_profile_id"] for r in row_dicts],
                "Reference": [r["protospacer_20nt"].upper() for r in row_dicts],
                "PAM": [r["pam4"].upper() for r in row_dicts],
            }
        )
        input_df["protospacer_PAM"] = input_df["Reference"] + input_df["PAM"]
        raw_input = args.output_dir / f"raw_t2_bedict_v2_input_{model_editor}.csv"
        input_df.to_csv(raw_input, index=False)

        seq_processor = HaplotypeSeqProcessor(model_editor, TARGET_CONV_NUCL[model_editor], seqconfig)
        bedict = BEDICT_EncEnc_HaplotypeModel(seq_processor, seqconfig, device)
        model_path = (
            args.repo
            / "proportion_model"
            / "output"
            / "experiment_run_proportions_encenc_two_model"
            / f"{model_editor}_proportions_encenc_two_model"
            / "protospacer_PAM"
            / "exp_version_0"
            / "train_val"
        )

        combined = pd.DataFrame(columns=["seq_id", "Inp_seq", "Outp_seq"])
        for run in range(3):
            pred_df = bedict.predict_from_dataframe(
                input_df,
                ["ID", "protospacer_PAM"],
                str(model_path / f"run_{run}"),
                outpseq_col=None,
                outcome_col=None,
                renormalize=True,
                batch_size=args.batch_size,
            )
            combined["seq_id"] = pred_df["seq_id"]
            combined["Inp_seq"] = pred_df["Inp_seq"]
            combined["Outp_seq"] = pred_df["Outp_seq"]
            combined[f"score_run_{run}"] = pred_df["pred_score"]

        score_cols = [f"score_run_{run}" for run in range(3)]
        combined["average_predicted_proportion"] = combined[score_cols].mean(axis=1)
        combined["std_predicted_proportion"] = combined[score_cols].std(axis=1)
        raw_output = args.output_dir / f"raw_t2_bedict_v2_predictions_{model_editor}.csv"
        combined.to_csv(raw_output, index=False)
        raw_paths.append(str(raw_output))

        panel_by_id = {r["target_profile_id"]: r for r in row_dicts}
        for _, pred in combined.iterrows():
            target_id = str(pred["seq_id"])
            events = events_from_sequences(str(pred["Inp_seq"]), str(pred["Outp_seq"]), 0, 1, 20)
            if events == "no_tracked_substrate_change":
                continue
            panel_row = panel_by_id.get(target_id, {})
            pred_idx += 1
            outcome_rows.append(
                {
                    "prediction_id": f"bedict_t2_outcome_{pred_idx:08d}",
                    "target_profile_id": target_id,
                    "model_id": "bedict_v2",
                    "run_id": args.run_id,
                    "model_editor": model_editor,
                    "model_cell_context": "",
                    "predicted_outcome_events": events,
                    "predicted_outcome_score": clean_float(pred["average_predicted_proportion"]),
                    "predicted_outcome_frequency": clean_float(pred["average_predicted_proportion"]),
                    "prediction_scale": "conditional_outcome_frequency_given_edited_reads",
                    "source_relation_to_model": panel_row.get("source_relation_to_model_for_t2", "proxy_external_candidate"),
                    "notes": (
                        "BEDICT-V2 proportion model on 20nt protospacer+4nt PAM. "
                        "T2 uses generic ABEmax-SpCas9 proxy mapping for BE-Hive ABE labels; no-edit haplotypes are removed and edited products are renormalized."
                    ),
                }
            )

    outcome_rows = normalize_outcome_scores(outcome_rows, score_key="predicted_outcome_score", output_key="predicted_outcome_frequency")
    position_rows = outcome_rows_to_position_rows(outcome_rows, id_prefix="bedict_t2_position")

    write_csv(args.output_dir / "predictions_t2_bedict_v2_outcomes.csv", outcome_rows)
    write_csv(args.output_dir / "predictions_t2_bedict_v2_positions.csv", position_rows)
    write_csv(args.output_dir / "skipped_t2_bedict_v2.csv", skips)
    (args.output_dir / "raw_t2_bedict_v2_manifest.json").write_text(json.dumps(raw_paths, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_id": "bedict_v2",
                "targets": len(panel),
                "outcome_predictions": len(outcome_rows),
                "position_predictions": len(position_rows),
                "skipped": len(skips),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
