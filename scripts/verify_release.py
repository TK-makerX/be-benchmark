from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checksums" / "release_sha256.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    failures = []
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"]: row for row in rows}
    actual = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] == ".git":
            continue
        actual[relative.as_posix()] = path
    if "LICENSE" not in actual:
        failures.append("MIT LICENSE is missing")
    for forbidden_name in ("LICENSE_PENDING.md", "CITATION.cff", "CITATION.cff.template"):
        if forbidden_name in actual:
            failures.append(f"obsolete or intentionally omitted release file present: {forbidden_name}")
    license_path = actual.get("LICENSE")
    if license_path is not None:
        license_text = license_path.read_text(encoding="utf-8", errors="replace")
        license_markers = (
            "MIT License\n",
            "Copyright (c) 2026 BE-Benchmark contributors",
            "Permission is hereby granted, free of charge",
            'THE SOFTWARE IS PROVIDED "AS IS"',
        )
        if not license_text.startswith(license_markers[0]) or any(marker not in license_text for marker in license_markers[1:]):
            failures.append("LICENSE does not contain the standard MIT grant")
    readme_text = actual.get("README.md").read_text(encoding="utf-8", errors="replace") if actual.get("README.md") else ""
    pyproject_text = actual.get("pyproject.toml").read_text(encoding="utf-8", errors="replace") if actual.get("pyproject.toml") else ""
    if "MIT License" not in readme_text:
        failures.append("README does not identify the MIT License")
    if 'license = {file = "LICENSE"}' not in pyproject_text:
        failures.append("pyproject.toml does not reference LICENSE")
    for name in sorted(set(expected) ^ set(actual)):
        failures.append(f"manifest mismatch: {name}")
    for name in sorted(set(expected) & set(actual)):
        path = actual[name]
        if int(expected[name]["size_bytes"]) != path.stat().st_size:
            failures.append(f"size mismatch: {name}")
        if expected[name]["sha256"] != digest(path):
            failures.append(f"hash mismatch: {name}")

    forbidden_roots = {"data", "results", "figures", "workflows", "environments", "third_party", "models", "weights"}
    for name, path in actual.items():
        if Path(name).parts and Path(name).parts[0] in forbidden_roots:
            failures.append(f"forbidden content directory: {name}")
        if path.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pkl", ".pickle", ".pt", ".pth", ".h5", ".ckpt", ".pb", ".onnx", ".zip", ".tar", ".gz"}:
            failures.append(f"forbidden data/model/archive file: {name}")
        if path.stat().st_size >= 5 * 1024 * 1024:
            failures.append(f"unexpected large file: {name}")
        if path.suffix.lower() in {".pyc", ".pyo"} or "__pycache__" in path.parts:
            failures.append(f"Python cache present: {name}")

    leak_patterns = {
        "numbered figure": r"(?i)\bfig(?:ure)?\.?\s*[0-9]+(?:[a-z])?\b",
        "numbered supplementary table": r"(?i)\btable\s*s[0-9]+\b",
        "external URL": r"(?i)\b(?:https?|ftp)://",
        "machine path": r"(?i)(?:[A-Za-z]:\\|/(?:home|Users|mnt)/)",
        "long DNA literal": r"(?i)\b[ACGTN]{20,}\b",
        "obsolete citation placeholder": (
            r"(?i)(?:REPLACE_WITH_(?:SOFTWARE_TITLE|AUTHOR|REPOSITORY_URL)|"
            + "pending author " + "approval|complete the " + "citation template)"
        ),
    }
    for name, path in actual.items():
        if path == Path(__file__).resolve() or path.suffix.lower() not in {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".cff", ".example"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in leak_patterns.items():
            if re.search(pattern, text):
                failures.append(f"{label} found in {name}")

    for path in actual.values():
        if path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                failures.append(f"syntax error: {path.relative_to(ROOT)}:{exc.lineno}")

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    smoke = subprocess.run(
        [
            sys.executable,
            str(ROOT / "app" / "run_cli.py"),
            "--sequence", "ACGT" * 6,
            "--editor-family", "example",
            "--task", "example_operation",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    try:
        payload = json.loads(smoke.stdout)
    except json.JSONDecodeError:
        payload = {}
    if smoke.returncode != 0 or "prediction" in payload:
        failures.append("shell CLI smoke test failed or emitted a numerical-result field")

    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1
    print("PASS: shell-only scope, exact manifest, no manuscript-data signatures, no numerical prediction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
