"""Run the FORECasT-BE README example and export JSON predictions."""

from __future__ import annotations

import os
import json
import sys
import warnings
from pathlib import Path


def main() -> None:
    repo = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "FORECasT-BE/repo"
    out_dir = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "FORECasT-BE/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    warnings.filterwarnings("ignore")
    import forecast_be  # noqa: PLC0415

    seq = "TCTGCTCAGCTCATGCCGAT"
    editor = "CBE"
    result = {
        "target_seq": seq,
        "editor": editor,
        "predict_total": float(forecast_be.predict_total(seq, editor=editor, mean=None, std=None)),
        "predict_positions": [
            {"position": int(pos), "score": None if score is None else float(score)}
            for pos, score in forecast_be.predict(seq, editor=editor, mean=None, std=None)
        ],
    }
    out = out_dir / "CBE_example_prediction.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
