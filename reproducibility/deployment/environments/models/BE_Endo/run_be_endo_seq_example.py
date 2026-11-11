"""Run BE_Endo sequence-only example without bedtools/endogenous BED input.

The upstream default pipeline predicts both BE_Seq and BE_Endo and requires
bedtools plus an external endo_factors.bed. This minimal reproducibility check
loads the bundled sequence-only Keras model and exports efficiency predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(r"E:\AIselect\BEs\BE_Endo\repo"))
    parser.add_argument("--input", type=Path, default=Path(r"E:\AIselect\BEs\BE_Endo\reproduction\test_input.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"E:\AIselect\BEs\BE_Endo\outputs"))
    parser.add_argument("--editor", default="CBE", choices=["ABE", "CBE"])
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from BE_endo_Pack.Effiency import Eff as Effiency  # noqa: PLC0415
    from BE_endo_Pack.Preprocess import Preprocessing as Prepro  # noqa: PLC0415

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = Prepro.input_process(str(args.input), args.editor)
    seq_40s, targets, labels, positions = processor.process()

    model_path = args.repo / "model_data_seq" / f"{args.editor}_Seq" / f"{args.editor}_Seq.h5"
    eff_prediction = Effiency.pip_seq(str(model_path), seq_40s, labels, args.editor)

    columns = [f"base{i}" for i in range(1, 21)]
    eff_table = pd.DataFrame(eff_prediction, columns=columns)
    eff_table.insert(0, "target", targets)
    eff_table.insert(0, "seq", seq_40s)
    out = args.output_dir / f"{args.editor}_BE_Seq_example_eff.tsv"
    eff_table.to_csv(out, sep="\t", index=False)

    print(f"Output: {out}")
    print(eff_table.to_string(index=False))


if __name__ == "__main__":
    main()
