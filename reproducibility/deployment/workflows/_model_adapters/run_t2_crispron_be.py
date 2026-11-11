from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from runtime_paths import model_path, tools_path

import pandas as pd


ADAPTER_DIR = Path(__file__).resolve().parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from t2_common import clean_float, dna, events_from_sequences, normalize_outcome_scores, outcome_rows_to_position_rows, write_csv  # noqa: E402
from run_crispron_be_pilot import VALID_EDITORS, run_editor  # noqa: E402


def editor_for_crispron(row: dict[str, str]) -> str | None:
    editor = (row.get("editor") or row.get("editor_class") or "").upper()
    if "ABE" in editor:
        return "ABE"
    if "CBE" in editor or "BE4" in editor or "C>" in (row.get("intended_edit") or "").upper():
        return "CBE"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_CRISPRON_REPO", "CRISPRon-BE", "repo"))
    parser.add_argument("--tools-bin", type=Path, default=tools_path())
    parser.add_argument("--run-id", default="T2_expand_v0_crispron_be")
    args = parser.parse_args()

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, str]]] = {"ABE": [], "CBE": []}
    panel_by_id: dict[str, dict[str, str]] = {}
    skips: list[dict[str, str]] = []
    for _, row in panel.iterrows():
        record = row.to_dict()
        target_id = record["target_profile_id"]
        seq30 = record.get("sequence_context_30nt", "").upper()
        editor = editor_for_crispron(record)
        if editor is None or editor not in VALID_EDITORS:
            skips.append({"target_profile_id": target_id, "reason": "no CRISPRon-BE editor mapping"})
            continue
        if len(seq30) != 30 or not dna(seq30) or seq30[25:27] != "GG":
            skips.append({"target_profile_id": target_id, "reason": "CRISPRon-BE requires 30nt DNA target with NGG at positions 25-27"})
            continue
        record["benchmark_id"] = target_id
        record["sequence_context"] = seq30
        grouped[editor].append(record)
        panel_by_id[target_id] = record

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(args.tools_bin), env.get("PATH", "")])
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    old_predictions: list[dict[str, str]] = []
    raw_manifest: list[dict[str, str]] = []
    for editor, rows in grouped.items():
        if not rows:
            continue
        preds, manifest, editor_skips = run_editor(
            editor=editor,
            rows=rows,
            output_dir=args.output_dir,
            repo=args.repo,
            env=env,
            python_exe=Path(sys.executable),
            run_ts="",
        )
        old_predictions.extend(preds)
        raw_manifest.extend(manifest)
        for skip in editor_skips:
            skips.append({"target_profile_id": skip.get("benchmark_id", ""), "reason": skip.get("reason", "")})

    outcome_rows: list[dict] = []
    for old in old_predictions:
        if old.get("task_type") != "outcome_distribution":
            continue
        target_id = old.get("benchmark_id", "")
        panel_row = panel_by_id.get(target_id, {})
        seq30 = panel_row.get("sequence_context", "")
        outcome = old.get("predicted_outcome", "")
        events = events_from_sequences(seq30, outcome, 4, -3, 20)
        if not events or events == "no_tracked_substrate_change":
            continue
        score = old.get("predicted_outcome_proportion") or old.get("predicted_outcome_frequency")
        outcome_rows.append(
            {
                "prediction_id": f"crispron_t2_outcome_{len(outcome_rows) + 1:08d}",
                "target_profile_id": target_id,
                "model_id": "crispron_be",
                "run_id": args.run_id,
                "model_editor": old.get("model_editor", ""),
                "model_cell_context": "",
                "predicted_outcome_events": events,
                "predicted_outcome_score": clean_float(score),
                "predicted_outcome_frequency": clean_float(score),
                "prediction_scale": "conditional_outcome_frequency_given_edited_reads",
                "source_relation_to_model": panel_row.get("source_relation_to_model_for_t2", "external_candidate"),
                "notes": "CRISPRon-BE outcome rows converted from pred_freq/pred_eff to T2 conditional edited-product scale.",
            }
        )

    outcome_rows = normalize_outcome_scores(outcome_rows, score_key="predicted_outcome_score", output_key="predicted_outcome_frequency")
    position_rows = outcome_rows_to_position_rows(outcome_rows, id_prefix="crispron_t2_position")

    write_csv(args.output_dir / "predictions_t2_crispron_be_outcomes.csv", outcome_rows)
    write_csv(args.output_dir / "predictions_t2_crispron_be_positions.csv", position_rows)
    write_csv(args.output_dir / "skipped_t2_crispron_be.csv", skips)
    pd.DataFrame(raw_manifest).to_csv(args.output_dir / "raw_t2_crispron_be_manifest.csv", index=False)
    print(json.dumps({"model_id": "crispron_be", "targets": len(panel), "outcome_predictions": len(outcome_rows), "position_predictions": len(position_rows), "skipped": len(skips)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
