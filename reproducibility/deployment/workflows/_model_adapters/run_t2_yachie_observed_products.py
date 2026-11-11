from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from runtime_paths import model_path, data_path

import pandas as pd


ADAPTER_DIR = Path(__file__).resolve().parent
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from t2_common import clean_float, outcome_rows_to_position_rows, parse_events, write_csv  # noqa: E402


def load_yachie_module(repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    script = repo / "base-editing-prediction.py"
    spec = importlib.util.spec_from_file_location("yachie_base_editing_prediction", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Yachie script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_model_file(row: dict[str, str]) -> str | None:
    editor = (row.get("editor") or "").upper()
    editor_class = (row.get("editor_class") or "").upper()
    if "ABEMAX" in editor:
        return "ABEmax.csv"
    if editor_class == "ABE" or "ABE" in editor:
        return "ABE.csv"
    if "BE4" in editor:
        return "BE4max.csv"
    if "AID" in editor:
        return "TargetAID.csv"
    if editor_class == "CBE":
        return "BE4max.csv"
    return None


def apply_events(seq30: str, events: str, protospacer_start: int = 4) -> str:
    chars = list(seq30.upper())
    for ref_base, pos, alt_base in parse_events(events):
        idx = protospacer_start + pos - 1
        if idx < 0 or idx >= len(chars):
            raise ValueError(f"event outside 30nt window: {ref_base}{pos}>{alt_base}")
        if chars[idx] != ref_base:
            raise ValueError(f"reference mismatch at position {pos}: expected {chars[idx]}, event has {ref_base}")
        chars[idx] = alt_base
    return "".join(chars)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--outcome-profile", type=Path, default=data_path("BE_T2_OUTCOME_PROFILE", "processed", "T2_bystander_matrix", "T2_observed_outcome_profiles.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_YACHIE_REPO", "Yachie-base-editing-prediction", "repo"))
    parser.add_argument("--run-id", default="T2_expand_v0_yachie")
    parser.add_argument("--start", type=int, default=-24)
    args = parser.parse_args()

    yachie = load_yachie_module(args.repo)
    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    observed = pd.read_csv(args.outcome_profile, dtype=str, low_memory=False).fillna("")
    observed = observed[observed["target_profile_id"].isin(set(panel["target_profile_id"]))].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_by_id = {row["target_profile_id"]: row.to_dict() for _, row in panel.iterrows()}
    loaded_models: dict[str, tuple] = {}
    raw_rows: list[dict] = []
    outcome_rows: list[dict] = []
    skips: list[dict] = []

    for _, obs in observed.iterrows():
        target_id = obs["target_profile_id"]
        target = panel_by_id.get(target_id, {})
        seq30 = target.get("sequence_context_30nt", "").upper()
        events = obs.get("outcome_events", "")
        if len(seq30) != 30 or set(seq30) - set("ACGT"):
            skips.append({"target_profile_id": target_id, "outcome_profile_id": obs.get("outcome_profile_id", ""), "reason": "missing concrete 30nt target"})
            continue
        try:
            edited = apply_events(seq30, events, protospacer_start=4)
        except Exception as exc:
            skips.append({"target_profile_id": target_id, "outcome_profile_id": obs.get("outcome_profile_id", ""), "reason": f"cannot materialize observed outcome: {exc}"})
            continue
        if edited == seq30:
            skips.append({"target_profile_id": target_id, "outcome_profile_id": obs.get("outcome_profile_id", ""), "reason": "no-edit outcome skipped"})
            continue
        model_file = choose_model_file(target)
        if model_file is None:
            skips.append({"target_profile_id": target_id, "outcome_profile_id": obs.get("outcome_profile_id", ""), "reason": "no Yachie sample model mapping"})
            continue
        if model_file not in loaded_models:
            loaded_models[model_file] = yachie.read_csv(str(args.repo / "sample_models" / model_file))
        transition_prob_dict, conditional_transition_prob_dict, model_name = loaded_models[model_file]
        try:
            score = yachie.simulation(
                seq30,
                edited,
                transition_prob_dict,
                conditional_transition_prob_dict,
                start=args.start,
                end=args.start + len(seq30) - 1,
                model=model_name,
            )
        except Exception as exc:
            skips.append({"target_profile_id": target_id, "outcome_profile_id": obs.get("outcome_profile_id", ""), "reason": f"Yachie simulation failed: {type(exc).__name__}: {str(exc)[:120]}"})
            continue

        raw_rows.append(
            {
                "target_profile_id": target_id,
                "outcome_profile_id": obs.get("outcome_profile_id", ""),
                "target_sequence_30nt": seq30,
                "edited_output_30nt": edited,
                "model_file": model_file,
                "model_name": model_name,
                "score": clean_float(score),
            }
        )
        outcome_rows.append(
            {
                "prediction_id": f"yachie_t2_outcome_{len(outcome_rows) + 1:08d}",
                "target_profile_id": target_id,
                "model_id": "yachie_base_editing_prediction",
                "run_id": args.run_id,
                "model_editor": model_name,
                "model_cell_context": "",
                "predicted_outcome_events": events,
                "predicted_outcome_score": clean_float(score),
                "predicted_outcome_frequency": "",
                "prediction_scale": "observed_outcome_candidate_score",
                "source_relation_to_model": target.get("source_relation_to_model_for_t2", "proxy_external_candidate"),
                "notes": "Yachie historical model scores only observed edited product candidates; not a complete product distribution.",
            }
        )

    position_rows = outcome_rows_to_position_rows(outcome_rows, id_prefix="yachie_t2_position", score_col="predicted_outcome_score")
    for row in position_rows:
        row["predicted_position_rate"] = ""
        row["prediction_scale"] = "observed_position_candidate_score"

    pd.DataFrame(raw_rows).to_csv(args.output_dir / "raw_t2_yachie_observed_products.csv", index=False)
    write_csv(args.output_dir / "predictions_t2_yachie_outcomes.csv", outcome_rows)
    write_csv(args.output_dir / "predictions_t2_yachie_positions.csv", position_rows)
    write_csv(args.output_dir / "skipped_t2_yachie.csv", skips)
    print(json.dumps({"model_id": "yachie_base_editing_prediction", "targets": panel["target_profile_id"].nunique(), "outcome_predictions": len(outcome_rows), "position_predictions": len(position_rows), "skipped": len(skips)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
