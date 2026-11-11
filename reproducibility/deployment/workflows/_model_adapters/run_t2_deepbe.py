from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf

from t2_common import clean_float, dna, events_from_sequences, normalize_outcome_scores, outcome_rows_to_position_rows, write_csv


ADAPTER_DIR = Path(__file__).resolve().parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from run_deepbe_pilot import DEFAULT_WEIGHTS, choose_pam_variant, run_deepbe_8e  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--run-id", default="T2_expand_v0_deepbe")
    args = parser.parse_args()

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    for _, row in panel.iterrows():
        target_id = row["target_profile_id"]
        seq30 = row.get("sequence_context_30nt", "").upper()
        if row.get("editor_class", "").upper() != "ABE":
            skips.append({"target_profile_id": target_id, "reason": "DeepBE T2 adapter currently uses ABE8e(V106W) only"})
            continue
        if len(seq30) != 30 or not dna(seq30):
            skips.append({"target_profile_id": target_id, "reason": "DeepBE requires concrete 30nt 4+20+PAM+3 window"})
            continue
        if "A" not in seq30[6:14]:
            skips.append({"target_profile_id": target_id, "reason": "DeepBE ABE editable window has no A bases"})
            continue
        variant, feature, variant_source = choose_pam_variant(row.to_dict())
        record = row.to_dict()
        record["benchmark_id"] = target_id
        record["_deepbe_target"] = seq30
        record["_target_source"] = "sequence_context_30nt"
        record["_pam_variant"] = variant
        record["_pam_feature"] = str(feature)
        record["_pam_variant_source"] = variant_source
        selected.append(record)

    raw_rows, _raw_pam = run_deepbe_8e(selected, args.weights, args.output_dir)
    panel_by_id = {row["target_profile_id"]: row for _, row in panel.iterrows()}

    outcome_rows: list[dict] = []
    for raw in raw_rows:
        target_id = raw["benchmark_id"]
        events = events_from_sequences(raw["deepbe_target_30nt"], raw["edited_output_30nt"], 4, -3, 20)
        if events == "no_tracked_substrate_change":
            continue
        panel_row = panel_by_id.get(target_id, {})
        outcome_rows.append(
            {
                "prediction_id": f"deepbe_t2_outcome_{len(outcome_rows) + 1:08d}",
                "target_profile_id": target_id,
                "model_id": "deepbe",
                "run_id": args.run_id,
                "model_editor": f"DeepBE_8e|{raw.get('pam_variant', '')}",
                "model_cell_context": "",
                "predicted_outcome_events": events,
                "predicted_outcome_score": clean_float(raw["prediction_score"]),
                "predicted_outcome_frequency": clean_float(raw["prediction_score"]),
                "prediction_scale": "conditional_outcome_frequency_given_edited_reads",
                "source_relation_to_model": panel_row.get("source_relation_to_model_for_t2", "proxy_external_candidate"),
                "notes": (
                    "DeepBE ABE8e(V106W) outcome scores on 30nt input. "
                    "Scores are renormalized across nonzero edited outcomes for T2 conditional profile comparison."
                ),
            }
        )

    outcome_rows = normalize_outcome_scores(outcome_rows, score_key="predicted_outcome_score", output_key="predicted_outcome_frequency")
    position_rows = outcome_rows_to_position_rows(outcome_rows, id_prefix="deepbe_t2_position")

    write_csv(args.output_dir / "predictions_t2_deepbe_outcomes.csv", outcome_rows)
    write_csv(args.output_dir / "predictions_t2_deepbe_positions.csv", position_rows)
    write_csv(args.output_dir / "skipped_t2_deepbe.csv", skips)
    tf.keras.backend.clear_session()
    print(
        json.dumps(
            {
                "model_id": "deepbe",
                "targets": len(panel),
                "selected_inputs": len(selected),
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
