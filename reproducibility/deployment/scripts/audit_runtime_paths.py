from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\/]|/(?:home|Users|mnt)/)")
SUFFIXES = {".py", ".ps1", ".sh", ".cmd"}
targets = [
    ROOT / "src",
    ROOT / "app",
    ROOT / "workflows" / "_model_adapters",
    ROOT / "workflows" / "data_preparation",
    ROOT / "tools" / "bin",
]
failures = []
for target in targets:
    paths = [target] if target.is_file() else sorted(target.rglob("*")) if target.exists() else []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if PATTERN.search(line):
                failures.append({"file": path.relative_to(ROOT).as_posix(), "line": number})
print(json.dumps({"status": "PASS" if not failures else "FAIL", "runtime_absolute_paths": failures}, indent=2))
raise SystemExit(0 if not failures else 1)
