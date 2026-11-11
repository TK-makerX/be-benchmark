from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from runtime_paths import model_path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


POSITION_COL_RE = re.compile(r"^([ACGT])(-?\d+)$")


def position_columns(df: pd.DataFrame) -> List[str]:
    return [str(col) for col in df.columns if POSITION_COL_RE.match(str(col))]


def outcome_events(row: pd.Series, pos_cols: Sequence[str]) -> Tuple[str, int]:
    events: List[str] = []
    for col in pos_cols:
        match = POSITION_COL_RE.match(col)
        if not match:
            continue
        ref_base = match.group(1)
        pos = int(match.group(2))
        value = str(row.get(col, "")).upper()
        if value and value != ref_base:
            events.append(f"{ref_base}{pos}>{value}")
    return ";".join(events) or "no_tracked_substrate_change", len(events)


def clean_float(value) -> str:
    try:
        out = float(value)
    except Exception:
        return ""
    if pd.isna(out):
        return ""
    return f"{out:.12g}"


def public_editor_lookup(models_design: pd.DataFrame) -> Dict[tuple[str, str], str]:
    lookup: Dict[tuple[str, str], str] = {}
    for _, row in models_design.iterrows():
        cell = str(row["Celltype"])
        public = str(row["Public base editor"])
        internal = str(row["Internal base editor"])
        lookup[(public, cell)] = public
        lookup[(internal, cell)] = public
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_BEHIVE_BYSTANDER_REPO", "BE-Hive-bystander", "repo"))
    parser.add_argument("--run-id", default="T2_smoke_v0_behive_bystander")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import predict as bystander_model  # noqa: PLC0415

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    lookup = public_editor_lookup(bystander_model.models_design)
    outcome_rows: List[dict] = []
    position_rows: List[dict] = []
    stats_rows: List[dict] = []
    skipped: List[dict] = []
    initialized: tuple[str, str] | None = None

    for _, target in panel.iterrows():
        target_id = target["target_profile_id"]
        seq = target.get("sequence_50nt", "").upper()
        cell = target.get("cell_context", "")
        requested_editor = target.get("editor", "")
        public_editor = lookup.get((requested_editor, cell), "")
        if not public_editor:
            skipped.append({"target_profile_id": target_id, "reason": f"unsupported editor/cell_context: {requested_editor}/{cell}"})
            continue
        if len(seq) != 50 or set(seq) - set("ACGT"):
            skipped.append({"target_profile_id": target_id, "reason": "missing concrete BE-Hive-native 50nt sequence"})
            continue

        if initialized != (public_editor, cell):
            bystander_model.init_model(base_editor=public_editor, celltype=cell)
            initialized = (public_editor, cell)

        pred_df, stats = bystander_model.predict(seq)
        pos_cols = position_columns(pred_df)
        stats_row = {k: str(v) for k, v in stats.items()}
        stats_row.update({"target_profile_id": target_id, "requested_editor": requested_editor, "public_editor": public_editor})
        stats_rows.append(stats_row)

        pred_df = pred_df.sort_values("Predicted frequency", ascending=False).reset_index(drop=True)
        for rank, (_, pred) in enumerate(pred_df.iterrows(), start=1):
            freq = pred.get("Predicted frequency")
            events, edit_count = outcome_events(pred, pos_cols)
            outcome_rows.append(
                {
                    "prediction_id": f"behive_t2_outcome_{len(outcome_rows) + 1:08d}",
                    "target_profile_id": target_id,
                    "model_id": "behive_bystander",
                    "run_id": run_id,
                    "model_editor": public_editor,
                    "model_cell_context": cell,
                    "outcome_rank": rank,
                    "predicted_outcome_events": events,
                    "predicted_outcome_edit_count": edit_count,
                    "predicted_outcome_frequency": clean_float(freq),
                    "prediction_scale": "conditional_outcome_frequency_given_edited_reads",
                    "source_relation_to_model": "source_associated",
                    "notes": "BE-Hive bystander product distribution on BE-Hive-compatible native 50nt input.",
                }
            )

        for col in pos_cols:
            match = POSITION_COL_RE.match(col)
            if not match:
                continue
            ref_base = match.group(1)
            pos = int(match.group(2))
            edited = pred_df[col].astype(str).str.upper() != ref_base
            rate = pred_df.loc[edited, "Predicted frequency"].astype(float).sum()
            position_rows.append(
                {
                    "prediction_id": f"behive_t2_position_{len(position_rows) + 1:08d}",
                    "target_profile_id": target_id,
                    "model_id": "behive_bystander",
                    "run_id": run_id,
                    "model_editor": public_editor,
                    "model_cell_context": cell,
                    "protospacer_position": pos,
                    "reference_base": ref_base,
                    "predicted_position_rate": clean_float(rate),
                    "predicted_position_score": clean_float(rate),
                    "prediction_scale": "conditional_position_rate_given_edited_reads",
                    "source_relation_to_model": "source_associated",
                    "notes": "Position rate derived by marginalizing predicted BE-Hive product distribution.",
                }
            )

    pd.DataFrame(outcome_rows).to_csv(args.output_dir / "predictions_t2_behive_bystander_outcomes.csv", index=False)
    pd.DataFrame(position_rows).to_csv(args.output_dir / "predictions_t2_behive_bystander_positions.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped_t2_behive_bystander.csv", index=False)
    (args.output_dir / "raw_t2_behive_bystander_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")
    summary = {
        "model_id": "behive_bystander",
        "targets": len(panel),
        "outcome_predictions": len(outcome_rows),
        "position_predictions": len(position_rows),
        "skipped": len(skipped),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
