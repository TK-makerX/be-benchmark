"""Execute the BEDICT-V2 author example and export its combined predictions."""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "BEDICT-V2/repo",
        help="BEDICT-V2 repository/worktree path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "BEDICT-V2/outputs",
        help="Directory for the executed notebook and prediction table.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    source_notebook = repo / "notebooks" / "Inference_user_defined_sequence.ipynb"
    executed_notebook = output_dir / "Inference_user_defined_sequence_executed.ipynb"
    prediction_csv = output_dir / "ABEmax-SpRY_example_predictions.csv"

    if not source_notebook.exists():
        raise FileNotFoundError(source_notebook)
    output_dir.mkdir(parents=True, exist_ok=True)

    with source_notebook.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    export_code = (
        "final_df.to_csv(" + json.dumps(str(prediction_csv)) + ", index=False)\n"
        "print('Exported combined predictions:', "
        + json.dumps(str(prediction_csv))
        + ")\n"
        "print('Prediction table shape:', final_df.shape)"
    )
    notebook.cells.append(nbformat.v4.new_code_cell(export_code))

    executor = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(source_notebook.parent)}})

    with executed_notebook.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    print(f"Executed notebook: {executed_notebook}")
    print(f"Combined predictions: {prediction_csv}")


if __name__ == "__main__":
    main()
