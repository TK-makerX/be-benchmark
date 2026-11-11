from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import warnings
from pathlib import Path


REPO = Path(os.environ.get("BE_FORECAST_REPO", Path(os.environ.get("BE_MODELS_DIR", "models")) / "FORECasT-BE" / "repo")).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description="FORECasT-BE worker for BE prediction system")
    parser.add_argument("--sequence", required=True, help="20nt protospacer")
    parser.add_argument("--editor", required=True, choices=["ABE", "CBE"])
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    warnings.filterwarnings("ignore", category=UserWarning)
    with contextlib.redirect_stdout(sys.stderr):
        import forecast_be  # noqa: PLC0415

        total = forecast_be.predict_total(args.sequence, editor=args.editor, mean=None, std=None)
        positions = forecast_be.predict(args.sequence, editor=args.editor, mean=None, std=None)

    payload = {
        "model_id": "forecast_be",
        "display_name": "FORECasT-BE",
        "status": "completed",
        "editor": args.editor,
        "protospacer_20nt": args.sequence,
        "prediction_type": "native_total_score_and_position_scores",
        "predicted_efficiency_score": float(total) if total is not None else None,
        "score_scale": "FORECasT-BE native score; safest as rank/relative score unless calibration parameters are supplied.",
        "position_scores": [
            {"position": int(position), "score": None if score is None else float(score)}
            for position, score in positions
        ],
        "benchmark_interpretation": "Use with T1/T4 scalar and T3/T7 rank-style caveats from the benchmark.",
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

