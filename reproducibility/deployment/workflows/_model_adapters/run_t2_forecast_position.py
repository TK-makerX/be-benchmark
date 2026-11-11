from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import model_path
from typing import List

import pandas as pd


def clean_float(value) -> str:
    try:
        out = float(value)
    except Exception:
        return ""
    if pd.isna(out):
        return ""
    return f"{out:.12g}"


def forecast_editor(editor_class: str) -> str:
    editor_class = (editor_class or "").upper()
    if editor_class == "ABE":
        return "ABE"
    if editor_class == "CBE":
        return "CBE"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_FORECAST_REPO", "FORECasT-BE", "repo"))
    parser.add_argument("--run-id", default="T2_smoke_v0_forecast_be")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import forecast_be  # noqa: PLC0415

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id
    rows: List[dict] = []
    skipped: List[dict] = []

    for _, target in panel.iterrows():
        target_id = target["target_profile_id"]
        protospacer = target.get("protospacer_20nt", "").upper()
        model_editor = forecast_editor(target.get("editor_class", ""))
        if len(protospacer) != 20 or set(protospacer) - set("ACGT"):
            skipped.append({"target_profile_id": target_id, "reason": "missing concrete 20nt protospacer"})
            continue
        if not model_editor:
            skipped.append({"target_profile_id": target_id, "reason": f"unsupported editor_class {target.get('editor_class', '')}"})
            continue

        try:
            predictions = forecast_be.predict(protospacer, editor=model_editor, mean=None, std=None)
        except Exception as exc:
            skipped.append({"target_profile_id": target_id, "reason": repr(exc)})
            continue

        for pos, score in predictions:
            if score is None:
                continue
            ref_base = protospacer[int(pos) - 1] if 1 <= int(pos) <= len(protospacer) else ""
            rows.append(
                {
                    "prediction_id": f"forecast_t2_position_{len(rows) + 1:08d}",
                    "target_profile_id": target_id,
                    "model_id": "forecast_be",
                    "run_id": run_id,
                    "model_editor": model_editor,
                    "model_cell_context": "",
                    "protospacer_position": int(pos),
                    "reference_base": ref_base,
                    "predicted_position_rate": "",
                    "predicted_position_score": clean_float(score),
                    "prediction_scale": "uncalibrated_position_score",
                    "source_relation_to_model": "external_candidate",
                    "notes": "FORECasT-BE position score. Use rank/correlation metrics; do not use RMSE/MAE as calibrated probability.",
                }
            )

    pd.DataFrame(rows).to_csv(args.output_dir / "predictions_t2_forecast_position.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped_t2_forecast_position.csv", index=False)
    summary = {
        "model_id": "forecast_be",
        "targets": len(panel),
        "position_predictions": len(rows),
        "skipped": len(skipped),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
