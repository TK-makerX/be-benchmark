"""Run the BE-Hive bystander example and export predictions.

This script uses the local compatibility-patched copy of
maxwshen/be_predict_bystander in ../repo.
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("BE_MODELS_DIR", os.environ.get("BE_REPLAY_PROJECT", "."))) / "BE-Hive-bystander/repo",
        help="Path to the be_predict_bystander source folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("BE_MODELS_DIR", os.environ.get("BE_REPLAY_PROJECT", "."))) / "BE-Hive-bystander/outputs",
        help="Output directory for prediction CSV and stats JSON.",
    )
    parser.add_argument("--base-editor", default="BE4")
    parser.add_argument("--celltype", default="mES")
    parser.add_argument(
        "--seq",
        default="TATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTACCGTGGACGTGAG",
        help="50-nt input sequence; spacer positions 1-20 and NGG PAM positions 21-23.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    import predict as bystander_model  # noqa: PLC0415

    bystander_model.init_model(base_editor=args.base_editor, celltype=args.celltype)
    pred_df, stats = bystander_model.predict(args.seq)

    stem = f"{args.base_editor}_{args.celltype}".replace("/", "-")
    pred_csv = output_dir / f"{stem}_example_predictions.csv"
    stats_json = output_dir / f"{stem}_example_stats.json"

    pred_df.to_csv(pred_csv, index=False)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Prediction CSV: {pred_csv}")
    print(f"Stats JSON: {stats_json}")
    print(f"Prediction shape: {pred_df.shape}")
    print(f"Total predicted probability: {stats['Total predicted probability']}")


if __name__ == "__main__":
    main()
