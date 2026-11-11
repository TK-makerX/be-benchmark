"""Run the BE-Hive efficiency README example and export prediction JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(r"E:\AIselect\BEs\BE-Hive-efficiency\repo"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"E:\AIselect\BEs\BE-Hive-efficiency\outputs"))
    parser.add_argument("--base-editor", default="BE4")
    parser.add_argument("--celltype", default="mES")
    parser.add_argument("--seq", default="TATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTACCGTGGACGTGAG")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import predict as be_efficiency_model  # noqa: PLC0415

    be_efficiency_model.init_model(base_editor=args.base_editor, celltype=args.celltype)
    pred = be_efficiency_model.predict(args.seq)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.base_editor}_{args.celltype}_example_prediction.json"
    out.write_text(json.dumps(pred, indent=2), encoding="utf-8")

    print(f"Prediction JSON: {out}")
    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
