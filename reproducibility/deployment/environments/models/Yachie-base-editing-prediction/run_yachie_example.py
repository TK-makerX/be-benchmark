"""Run the Yachie base-editing-prediction README example."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "Yachie-base-editing-prediction/repo"
    output_dir = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "Yachie-base-editing-prediction/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo / "base-editing-prediction.py"),
        "-i",
        "ACACACACACACTCTGATCATACGA",
        "-o",
        "ACACACATGTGCTCTGATCATACGA",
        "-e",
        "-1",
        "-m",
        str(repo / "sample_models" / "TargetACEmax.csv"),
    ]
    result = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, check=True)
    out = output_dir / "TargetACEmax_example_stdout.txt"
    out.write_text(result.stdout, encoding="utf-8")
    print(result.stdout)
    print(f"Saved stdout: {out}")


if __name__ == "__main__":
    main()
