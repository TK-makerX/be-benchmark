from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

from runtime_paths import model_path

import numpy as np
import pandas as pd

from t2_common import clean_float, dna, events_from_sequences, normalize_outcome_scores, outcome_rows_to_position_rows, write_csv


def edited_outcomes(seq30: str, editor_class: str) -> list[str]:
    editor_class = editor_class.upper()
    if editor_class == "ABE":
        ref_base, alt_base = "A", "G"
    elif editor_class == "CBE":
        ref_base, alt_base = "C", "T"
    else:
        return []
    editable_indexes = [idx for idx in range(6, 14) if seq30[idx] == ref_base]
    if not editable_indexes:
        return []
    outcomes = [seq30]
    for size in range(1, len(editable_indexes) + 1):
        for combo in combinations(editable_indexes, size):
            chars = list(seq30)
            for idx in combo:
                chars[idx] = alt_base
            outcomes.append("".join(chars))
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_EPCNNBE_REPO", "EPCNNBE", "repo"))
    parser.add_argument("--run-id", default="T3_exploratory_v0_epcnnbe")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    import EPCNNBE_web as epcnnbe  # noqa: PLC0415

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    outcome_rows: list[dict] = []
    skips: list[dict] = []

    for editor_class, be_name in [("ABE", "abe"), ("CBE", "cbe")]:
        selected_pairs: list[tuple[dict, str]] = []
        for _, row in panel[panel["editor_class"].eq(editor_class)].iterrows():
            seq30 = row.get("sequence_context_30nt", "").upper()
            target_id = row["target_profile_id"]
            if len(seq30) != 30 or not dna(seq30):
                skips.append({"target_profile_id": target_id, "reason": "EPCNNBE T3 requires concrete 30nt target"})
                continue
            outcomes = edited_outcomes(seq30, editor_class)
            if len(outcomes) <= 1:
                skips.append({"target_profile_id": target_id, "reason": "EPCNNBE editable window has no substrate base"})
                continue
            row_dict = row.to_dict()
            for outcome in outcomes:
                selected_pairs.append((row_dict, outcome))

        if not selected_pairs:
            continue

        seqs_all = np.array([[row["sequence_context_30nt"].upper(), outcome] for row, outcome in selected_pairs])
        target_oh, outcome_oh = epcnnbe.onehot_encoding(seqs_all, 0, 30)
        test = {"tar_oh": target_oh, "oc_oh": outcome_oh, "seqs": seqs_all}
        raw = epcnnbe.ensemble_test_request(be_name, 1, 0, test, str(args.repo / "be_weights") + "\\")

        for row, result in zip([pair[0] for pair in selected_pairs], raw):
            target_id = row["target_profile_id"]
            target = str(result[0]).upper()
            outcome = str(result[1]).upper()
            out_prop = clean_float(result[3])
            raw_record = {
                "target_profile_id": target_id,
                "target_sequence_30nt": target,
                "predicted_outcome_30nt": outcome,
                "edit_efficiency": clean_float(result[2]),
                "outcome_proportion": out_prop,
                "outcome_frequency": clean_float(result[4]),
                "picking_score": clean_float(result[5]),
                "editor_class": editor_class,
            }
            raw_rows.append(raw_record)
            events = events_from_sequences(target, outcome, 4, -3, 20)
            if events == "no_tracked_substrate_change":
                continue
            try:
                prop_value = float(out_prop)
            except Exception:
                continue
            if prop_value <= 0:
                continue
            outcome_rows.append(
                {
                    "prediction_id": f"epcnnbe_t3_outcome_{len(outcome_rows) + 1:08d}",
                    "target_profile_id": target_id,
                    "model_id": "epcnnbe",
                    "run_id": args.run_id,
                    "model_editor": editor_class,
                    "model_cell_context": "",
                    "predicted_outcome_events": events,
                    "predicted_outcome_score": out_prop,
                    "predicted_outcome_frequency": out_prop,
                    "prediction_scale": "conditional_outcome_frequency_given_edited_reads",
                    "source_relation_to_model": row.get("source_relation_to_model_for_t3", "proxy_external_candidate"),
                    "notes": "EPCNNBE exploratory T3 profile: conditional outcome proportions marginalized to editable-position rates.",
                }
            )

    outcome_rows = normalize_outcome_scores(outcome_rows, score_key="predicted_outcome_score", output_key="predicted_outcome_frequency")
    position_rows = outcome_rows_to_position_rows(outcome_rows, id_prefix="epcnnbe_t3_position")
    for row in position_rows:
        row["notes"] = "EPCNNBE exploratory T3 position profile derived from generated outcome proportions."

    pd.DataFrame(raw_rows).to_csv(args.output_dir / "raw_t3_epcnnbe_outcomes.csv", index=False)
    write_csv(args.output_dir / "predictions_t3_epcnnbe_outcomes.csv", outcome_rows)
    write_csv(args.output_dir / "predictions_t3_epcnnbe_positions.csv", position_rows)
    write_csv(args.output_dir / "skipped_t3_epcnnbe.csv", skips)
    print(
        json.dumps(
            {
                "model_id": "epcnnbe",
                "targets": panel["target_profile_id"].nunique(),
                "raw_outcome_rows": len(raw_rows),
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
