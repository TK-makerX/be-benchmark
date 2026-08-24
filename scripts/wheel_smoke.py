from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed: " + " ".join(command) + "\n" + completed.stdout + "\n" + completed.stderr
        )
    return completed


def environment_python(directory: Path) -> Path:
    if os.name == "nt":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def console_script(directory: Path) -> Path:
    if os.name == "nt":
        return directory / "Scripts" / "be-benchmark-shell.exe"
    return directory / "bin" / "be-benchmark-shell"


def audit_wheel(path: Path) -> None:
    forbidden_suffixes = {
        ".csv", ".tsv", ".xlsx", ".parquet", ".pkl", ".pt", ".pth",
        ".h5", ".ckpt", ".onnx", ".pdf", ".png", ".svg", ".tif",
    }
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            top = name.split("/", 1)[0]
            if not (top == "be_benchmark_shell" or top.startswith("be_benchmark_shell-")):
                raise RuntimeError(f"unexpected wheel member: {name}")
            if Path(name).suffix.lower() in forbidden_suffixes:
                raise RuntimeError(f"forbidden wheel member: {name}")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        license_names = [name for name in names if name.endswith(".dist-info/licenses/LICENSE")]
        if len(metadata_names) != 1 or len(license_names) != 1:
            raise RuntimeError("wheel license metadata is incomplete")
        metadata = archive.read(metadata_names[0]).decode("utf-8", errors="replace")
        for marker in ("License-Expression: MIT", "License-File: LICENSE"):
            if marker not in metadata:
                raise RuntimeError(f"wheel metadata marker is missing: {marker}")
        for marker in (
            "Author-email:",
            "Maintainer-email:",
            "Home-page:",
            "Project-URL:",
            "http" + "://",
            "https" + "://",
        ):
            if marker in metadata:
                raise RuntimeError(f"unexpected wheel metadata field: {marker}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="be_shell_wheel_") as temporary:
        work = Path(temporary)
        source = work / "source"
        shutil.copytree(
            ROOT,
            source,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "build", "dist", "*.egg-info", "__pycache__", "*.pyc"
            ),
        )
        wheelhouse = work / "wheelhouse"
        wheelhouse.mkdir()
        build_command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
        ]
        if os.environ.get("BE_SHELL_OFFLINE_BUILD") == "1":
            build_command.append("--no-build-isolation")
        build_command.append(str(source))
        run(build_command, work)
        wheels = list(wheelhouse.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("wheel build did not produce exactly one artifact")
        audit_wheel(wheels[0])

        environment = work / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment_python(environment)
        run(
            [str(python), "-m", "pip", "install", "--no-deps", "--disable-pip-version-check", str(wheels[0])],
            work,
        )
        run_directory = work / "run"
        run_directory.mkdir()
        runtime_env = os.environ.copy()
        runtime_env.pop("BE_SHELL_REGISTRY", None)
        runtime_env.pop("PYTHONPATH", None)
        completed = run(
            [
                str(console_script(environment)),
                "--sequence",
                "ACGT" * 6,
                "--editor-family",
                "example",
                "--task",
                "example_operation",
            ],
            run_directory,
            runtime_env,
        )
        payload = json.loads(completed.stdout)
        if payload.get("status") != "backend_not_configured":
            raise RuntimeError("installed CLI did not return the empty-backend status")
        if payload.get("backend_id") is not None or "prediction" in payload:
            raise RuntimeError("installed CLI emitted an unexpected result field")
    print("PASS: isolated wheel build, metadata and installed CLI contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
