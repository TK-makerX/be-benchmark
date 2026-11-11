from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def version(executable: Path) -> str:
    result = subprocess.run([str(executable), "--version"], capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def main():
    args = parse_args()
    root = args.project_root
    raw = args.output / "predictions" / "raw"
    logs = args.output / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    tf_python = Path(r"E:\AIselect\envs\be_tf210\Scripts\python.exe")
    forecast_python = Path(r"E:\AIselect\envs\forecastbe\Scripts\python.exe")
    specs = [
        {
            "model": "DeepBaseEditor", "id": "deepbaseeditor", "python": tf_python,
            "adapter": root / "benchmark_core/adapters/run_deepbaseeditor_abe_efficiency_pilot.py",
            "extra": ["--repo", str(root / "DeepBaseEditor-Paired-Library/repo/DeepBaseEditor/ABE_Efficiency")],
            "weights": str(root / "DeepBaseEditor-Paired-Library/repo/DeepBaseEditor/ABE_Efficiency/ABE_Efficiency_Weight"),
        },
        {
            "model": "FORECasT-BE", "id": "forecast_be", "python": forecast_python,
            "adapter": root / "benchmark_core/adapters/run_forecast_be_pilot.py",
            "extra": ["--repo", str(root / "FORECasT-BE/repo")],
            "weights": str(root / "FORECasT-BE/repo/saved_models/ABE"),
        },
        {
            "model": "DeepBE", "id": "deepbe", "python": tf_python,
            "adapter": root / "benchmark_core/adapters/run_deepbe_pilot.py",
            "extra": [],
            "weights": str(root / "DeepBE/resources/extracted_models/DeepBE/DeepBE_8e_model"),
        },
        {
            "model": "CRISPRon-BE", "id": "crispron_be", "python": tf_python,
            "adapter": root / "benchmark_core/adapters/run_crispron_be_pilot.py",
            "extra": ["--repo", str(root / "CRISPRon-BE/repo"), "--tools-bin", str(root / "tools/bin")],
            "weights": str(root / "CRISPRon-BE/repo/models"),
        },
    ]

    records = []
    for spec in specs:
        model_dir = raw / spec["id"]
        model_dir.mkdir(parents=True, exist_ok=True)
        command = [str(spec["python"]), str(spec["adapter"]), "--panel", str(args.panel), "--output-dir", str(model_dir), *spec["extra"]]
        stdout_path = logs / f"{spec['id']}.stdout.log"
        stderr_path = logs / f"{spec['id']}.stderr.log"
        started = datetime.now(timezone.utc)
        print(json.dumps({"event": "model_start", "model": spec["model"], "started_utc": started.isoformat(), "command": command}), flush=True)
        tic = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=False)
        seconds = time.perf_counter() - tic
        finished = datetime.now(timezone.utc)
        record = {
            "model": spec["model"], "model_id": spec["id"], "returncode": result.returncode,
            "started_utc": started.isoformat(), "finished_utc": finished.isoformat(), "runtime_seconds": seconds,
            "command": command, "python_executable": str(spec["python"]), "python_version": version(spec["python"]),
            "adapter": str(spec["adapter"]), "weight_path": spec["weights"], "output_dir": str(model_dir),
            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
        }
        records.append(record)
        print(json.dumps({"event": "model_finish", **record}), flush=True)

    report = {"panel": str(args.panel), "models": records, "all_returncode_zero": all(row["returncode"] == 0 for row in records)}
    (logs / "MODEL_INFERENCE_EXECUTION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_returncode_zero"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
