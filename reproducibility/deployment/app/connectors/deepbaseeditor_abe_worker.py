from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO = Path(os.environ.get("BE_DEEPBASEEDITOR_REPO", Path(os.environ.get("BE_MODELS_DIR", "models")) / "DeepBaseEditor-Paired-Library" / "repo" / "DeepBaseEditor" / "ABE_Efficiency")).expanduser()


def _write_input(sequence: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Target number\t30 bp target sequence (4 bp + 20 bp protospacer + PAM + 3 bp)\n"
        f"1\t{sequence}\n",
        encoding="utf-8",
    )


def _run_legacy_script(input_path: Path) -> Path:
    import tensorflow.compat.v1 as tf_v1  # noqa: PLC0415

    tf_v1.disable_v2_behavior()
    sys.modules["tensorflow"] = tf_v1

    source_path = REPO / "TEST_ABE_Efficiency.py"
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        "parameters           = {'0': 'ABE_Efficiency_sample.txt'}",
        "parameters           = {'0': '" + input_path.as_posix() + "'}",
    )
    source = source.replace("path                 = './'", "path                 = ''")
    # This is the same compatibility patch used by the reproduced local example.
    source = source.replace("length  = 24", "length  = 25")

    from lib2to3.refactor import RefactoringTool, get_fixers_from_package  # noqa: PLC0415

    output_xlsx = REPO / "outputs" / "TEST_OUTPUT_fortest.xlsx"
    if output_xlsx.exists():
        output_xlsx.unlink()

    old_cwd = Path.cwd()
    try:
        os.chdir(REPO)
        tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
        converted = str(tool.refactor_string(source, str(source_path)))
        globals_dict = {"__name__": "__main__", "__file__": str(source_path)}
        with contextlib.redirect_stdout(sys.stderr):
            exec(compile(converted, str(source_path), "exec"), globals_dict)
    finally:
        os.chdir(old_cwd)

    if not output_xlsx.exists():
        raise RuntimeError(f"DeepBaseEditor output was not created: {output_xlsx}")
    return output_xlsx


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepBaseEditor ABE_Efficiency worker")
    parser.add_argument("--sequence", required=True, help="30nt sequence: 4bp + 20bp protospacer + PAM + 3bp")
    args = parser.parse_args()

    sequence = args.sequence.upper().strip()
    if len(sequence) != 30 or set(sequence) - set("ACGT"):
        print(
            json.dumps(
                {
                    "model_id": "deepbaseeditor",
                    "status": "skipped",
                    "reason": "DeepBaseEditor ABE_Efficiency requires a concrete 30nt A/C/G/T input.",
                    "sequence": sequence,
                },
                ensure_ascii=False,
            )
        )
        return 0

    with tempfile.TemporaryDirectory(prefix="be_prediction_system_deepbaseeditor_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "deepbaseeditor_input.txt"
        copied_xlsx = tmp_dir / "deepbaseeditor_raw_output.xlsx"
        _write_input(sequence, input_path)
        raw_xlsx = _run_legacy_script(input_path)
        shutil.copy2(raw_xlsx, copied_xlsx)

        sheet = pd.read_excel(copied_xlsx, sheet_name=0, header=None)
        raw_target = str(sheet.iloc[0, 0]).upper()
        raw_score = float(sheet.iloc[0, 2])
        payload = {
            "model_id": "deepbaseeditor",
            "display_name": "DeepBaseEditor ABE_Efficiency",
            "status": "completed",
            "editor": "ABE",
            "target_30nt": raw_target,
            "prediction_type": "ABE_scalar_efficiency",
            "raw_activity_score_percent": raw_score,
            "predicted_efficiency_score": raw_score / 100.0,
            "score_scale": "DeepBaseEditor historical ABE_Efficiency output; raw script reports 100*model_output, converted score is raw/100.",
            "benchmark_interpretation": "Use as ABE scalar-efficiency backend with T1/T4/T6 caveats; legacy compatibility runner.",
        }
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

