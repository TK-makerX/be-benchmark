from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from runtime_paths import model_path

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


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
    if editor_class == "CBE":
        return "BE4max.csv"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_YACHIE_REPO", "Yachie-base-editing-prediction", "repo"))
    parser.add_argument("--start", type=int, default=-24, help="Relative start position for 30nt 4+20+PAM+3 pilot windows.")
    args = parser.parse_args()

    yachie = load_yachie_module(args.repo)
    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v0_yachie"

    loaded_models: dict[str, tuple] = {}
    predictions: list[dict[str, str]] = []
    raw_rows: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    pred_idx = 0

    for row in rows:
        benchmark_id = row["benchmark_id"]
        target = (row.get("sequence_context") or "").upper()
        outcome = (row.get("reference_outcome") or "").upper()
        if len(target) != 30 or len(outcome) != 30 or not dna_only(target) or not dna_only(outcome) or "N" in target or "N" in outcome:
            skips.append({"benchmark_id": benchmark_id, "reason": "Yachie pilot requires concrete 30nt target and 30nt outcome"})
            continue
        if target == outcome:
            skips.append({"benchmark_id": benchmark_id, "reason": "Yachie predicts edited patterns; no-edit outcome skipped"})
            continue

        model_file = choose_model_file(row)
        if model_file is None:
            skips.append({"benchmark_id": benchmark_id, "reason": "no Yachie sample model mapping"})
            continue
        if model_file not in loaded_models:
            loaded_models[model_file] = yachie.read_csv(str(args.repo / "sample_models" / model_file))
        transition_prob_dict, conditional_transition_prob_dict, model_name = loaded_models[model_file]

        start = args.start
        end = start + len(target) - 1
        try:
            score = yachie.simulation(
                target,
                outcome,
                transition_prob_dict,
                conditional_transition_prob_dict,
                start=start,
                end=end,
                model=model_name,
            )
        except Exception as exc:  # noqa: BLE001 - record model incompatibility rather than crashing the full benchmark.
            skips.append({"benchmark_id": benchmark_id, "reason": f"Yachie simulation failed: {type(exc).__name__}: {str(exc)[:120]}"})
            continue

        raw_record = {
            "benchmark_id": benchmark_id,
            "target_sequence": target,
            "editing_outcome": outcome,
            "model_file": model_file,
            "model_name": model_name,
            "start": str(start),
            "end": str(end),
            "editing_frequency": clean_float(score),
        }
        raw_rows.append(raw_record)

        pred_idx += 1
        predictions.append(
            prediction_row(
                prediction_id=f"pilot_yachie_{pred_idx:05d}",
                benchmark_id=benchmark_id,
                model_id="yachie_base_editing_prediction",
                run_id=run_id,
                access_mode="local",
                task_type="outcome_distribution",
                predicted_outcome=outcome,
                predicted_outcome_frequency=clean_float(score),
                predicted_score=clean_float(score),
                model_editor=model_name,
                model_cell_context=row.get("cell_context", ""),
                raw_output_path=str(out_dir / "raw_yachie_pilot.csv"),
                model_version="local sample model; adapter pilot_v0",
                run_timestamp=run_ts,
                notes=(
                    "Yachie/Sakata 2020 historical probabilistic model. "
                    "Pilot assumes 30nt window coordinates start=-24,end=5 for 4bp upstream + 20nt protospacer + PAM + 3bp downstream."
                ),
            )
        )

    write_csv(
        out_dir / "raw_yachie_pilot.csv",
        raw_rows,
        ["benchmark_id", "target_sequence", "editing_outcome", "model_file", "model_name", "start", "end", "editing_frequency"],
    )
    write_csv(out_dir / "predictions_yachie_pilot.csv", predictions)
    write_csv(out_dir / "skipped_yachie_pilot.csv", skips, ["benchmark_id", "reason"])
    print(
        json.dumps(
            {
                "model_id": "yachie_base_editing_prediction",
                "predictions": len(predictions),
                "skipped": len(skips),
                "output": str(out_dir / "predictions_yachie_pilot.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
