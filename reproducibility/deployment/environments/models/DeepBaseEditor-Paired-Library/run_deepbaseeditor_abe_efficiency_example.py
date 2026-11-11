"""Run DeepBaseEditor ABE_Efficiency example through TensorFlow v1 compat."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    repo = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "DeepBaseEditor-Paired-Library/repo/DeepBaseEditor/ABE_Efficiency"
    out_dir = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "DeepBaseEditor-Paired-Library/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow.compat.v1 as tf_v1  # noqa: PLC0415

    tf_v1.disable_v2_behavior()
    sys.modules["tensorflow"] = tf_v1

    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        source_path = repo / "TEST_ABE_Efficiency.py"
        source = source_path.read_text(encoding="utf-8")
        source = source.replace(
            "parameters           = {'0': 'ABE_Efficiency_sample.txt'}",
            "parameters           = {'0': 'DeepBaseEditor-Paired-Library/reproduction/ABE_Efficiency_sample_30nt.txt'}",
        )
        source = source.replace("path                 = './'", "path                 = ''")
        source = source.replace("length  = 24", "length  = 25")
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package  # noqa: PLC0415

        tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
        converted = str(tool.refactor_string(source, str(source_path)))
        globals_dict = {"__name__": "__main__", "__file__": str(source_path)}
        exec(compile(converted, str(source_path), "exec"), globals_dict)
    finally:
        os.chdir(old_cwd)

    src = repo / "outputs" / "TEST_OUTPUT_fortest.xlsx"
    dst = out_dir / "ABE_Efficiency_TEST_OUTPUT_fortest.xlsx"
    shutil.copy2(src, dst)
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
