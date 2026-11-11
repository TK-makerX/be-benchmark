"""Run the EPCNNBE README command-line example."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(r"E:\AIselect\BEs\EPCNNBE\repo")
    out_dir = Path(r"E:\AIselect\BEs\EPCNNBE\outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(repo / "EPCNNBE_web.py"), "abe", "test_seq_SOD1.fa"]
    subprocess.run(cmd, cwd=repo, check=True)

    src = repo / "test_seq_SOD1_res.csv"
    dst = out_dir / "test_seq_SOD1_abe_res.csv"
    shutil.copy2(src, dst)
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
