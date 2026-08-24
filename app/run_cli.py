from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from be_benchmark_shell.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
