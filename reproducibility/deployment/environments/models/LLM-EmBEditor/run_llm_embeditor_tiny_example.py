"""Run a tiny end-to-end LLM-EmBEditor example.

This validates the repository pipeline on bundled example JSONL data using a
small HuggingFace test model. It is not a substitute for the authors' full
Qwen/Llama-scale experiment.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(r"E:\AIselect\BEs\LLM-EmBEditor")
    repo = root / "repo"
    data_dir = root / "reproduction" / "example_data"
    out_dir = root / "outputs" / "tiny_example"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        shutil.copy2(
            repo / "data" / "example" / f"ABE8e_{split}_example.jsonl",
            data_dir / f"ABE8e_{split}.jsonl",
        )

    model_name = "hf-internal-testing/tiny-random-bert"
    subprocess.run(
        [
            sys.executable,
            str(repo / "generate_embeddings.py"),
            "--model_name",
            model_name,
            "--dataset",
            "ABE8e",
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(out_dir),
            "--pooling_strategy",
            "mean_only",
            "--max_seq_length",
            "128",
            "--batch_size",
            "8",
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
    )

    embedding_path = out_dir / "precomputed_embeddings" / "tiny-random-bert" / "ABE8e" / "pooling_mean_only"
    subprocess.run(
        [
            sys.executable,
            str(repo / "regression_head.py"),
            "--embedding_path",
            str(embedding_path),
            "--hidden_dims",
            "16",
            "--epochs",
            "2",
            "--batch_size",
            "8",
            "--output_dir",
            str(out_dir / "regression_head"),
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
    )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
