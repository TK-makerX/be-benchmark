from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
import json

import pandas as pd

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


def write_deepbaseeditor_input(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Target number\t30 bp target sequence (4 bp + 20 bp protospacer + PAM + 3 bp)\n")
        for idx, row in enumerate(rows, start=1):
            handle.write(f"{idx}\t{row['_target_30nt']}\n")


def run_legacy_abe_efficiency(repo: Path, input_path: Path) -> Path:
    """Execute upstream Python2/TF1-style script through TF2 compat + lib2to3."""
    import tensorflow.compat.v1 as tf_v1  # noqa: PLC0415

    tf_v1.disable_v2_behavior()
    sys.modules["tensorflow"] = tf_v1

    source_path = repo / "TEST_ABE_Efficiency.py"
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        "parameters           = {'0': 'ABE_Efficiency_sample.txt'}",
        "parameters           = {'0': '" + input_path.as_posix() + "'}",
    )
    source = source.replace("path                 = './'", "path                 = ''")
    # Upstream ABE_Efficiency has a local preprocessing length mismatch.
    # The already reproduced example used this exact compatibility patch.
    source = source.replace("length  = 24", "length  = 25")

    from lib2to3.refactor import RefactoringTool, get_fixers_from_package  # noqa: PLC0415

    output_xlsx = repo / "outputs" / "TEST_OUTPUT_fortest.xlsx"
    if output_xlsx.exists():
        output_xlsx.unlink()

    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
        converted = str(tool.refactor_string(source, str(source_path)))
        globals_dict = {"__name__": "__main__", "__file__": str(source_path)}
        exec(compile(converted, str(source_path), "exec"), globals_dict)
    finally:
        os.chdir(old_cwd)

    if not output_xlsx.exists():
        raise RuntimeError(f"DeepBaseEditor output was not created: {output_xlsx}")
    return output_xlsx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v0_deepbaseeditor_abe_efficiency"

    selected: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    for row in rows:
        benchmark_id = row["benchmark_id"]
        target = (row.get("sequence_context") or "").upper()
        if (row.get("editor_class") or "").upper() != "ABE":
            skips.append({"benchmark_id": benchmark_id, "reason": "DeepBaseEditor ABE_Efficiency pilot only maps ABE rows"})
            continue
        if len(target) != 30 or not dna_only(target) or "N" in target:
            skips.append({"benchmark_id": benchmark_id, "reason": "DeepBaseEditor ABE_Efficiency requires concrete 30nt target"})
            continue
        row = dict(row)
        row["_target_30nt"] = target
        selected.append(row)

    predictions: list[dict[str, str]] = []
    raw_rows: list[dict[str, str]] = []

    if selected:
        input_path = out_dir / "raw_deepbaseeditor_abe_efficiency_input.txt"
        write_deepbaseeditor_input(selected, input_path)
        raw_xlsx = run_legacy_abe_efficiency(args.repo, input_path)
        copied_xlsx = out_dir / "raw_deepbaseeditor_abe_efficiency_pilot.xlsx"
        shutil.copy2(raw_xlsx, copied_xlsx)

        sheet = pd.read_excel(copied_xlsx, sheet_name=0, header=None)
        for idx, row in enumerate(selected):
            if idx >= len(sheet):
                skips.append({"benchmark_id": row["benchmark_id"], "reason": "DeepBaseEditor output row missing"})
                continue
            target = str(sheet.iloc[idx, 0]).upper()
            raw_score = sheet.iloc[idx, 2]
            score = clean_float(raw_score)
            efficiency = ""
            try:
                efficiency = clean_float(float(raw_score) / 100.0)
            except (TypeError, ValueError):
                efficiency = ""

            raw_record = {
                "benchmark_id": row["benchmark_id"],
                "target_sequence": target,
                "raw_activity_score_percent": score,
                "converted_efficiency_0_1": efficiency,
            }
            raw_rows.append(raw_record)
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_deepbaseeditor_abe_eff_{idx + 1:05d}",
                    benchmark_id=row["benchmark_id"],
                    model_id="deepbaseeditor",
                    run_id=run_id,
                    access_mode="local",
                    task_type="efficiency",
                    predicted_efficiency=efficiency,
                    predicted_score=score,
                    model_editor="ABE_Efficiency",
                    model_cell_context=row.get("cell_context", ""),
                    raw_output_path=str(copied_xlsx),
                    model_version="local legacy TF1 weights; adapter pilot_v0",
                    run_timestamp=run_ts,
                    notes=(
                        "DeepBaseEditor ABE_Efficiency historical baseline. "
                        "Raw script reports 100*model_output; predicted_efficiency stores raw_score/100."
                    ),
                )
            )

    write_csv(
        out_dir / "raw_deepbaseeditor_abe_efficiency_pilot.csv",
        raw_rows,
        ["benchmark_id", "target_sequence", "raw_activity_score_percent", "converted_efficiency_0_1"],
    )
    write_csv(out_dir / "predictions_deepbaseeditor_abe_efficiency_pilot.csv", predictions)
    write_csv(out_dir / "skipped_deepbaseeditor_abe_efficiency_pilot.csv", skips, ["benchmark_id", "reason"])
    print(
        json.dumps(
            {
                "model_id": "deepbaseeditor",
                "selected_inputs": len(selected),
                "predictions": len(predictions),
                "skipped": len(skips),
                "output": str(out_dir / "predictions_deepbaseeditor_abe_efficiency_pilot.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
