#!/usr/bin/env python
"""Run a T6 reproducibility/applicability timing pass.

This run is not an accuracy benchmark. It calls existing adapters on their
small validated panels and records whether they run, how many rows they emit,
how many inputs are skipped, and wall-clock runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
DEFAULT_OUTPUT = ROOT / "formal_results" / "T6_reproducibility_v0"


@dataclass(frozen=True)
class T6Spec:
    model_id: str
    display_name: str
    python_exe: Path
    adapter_script: Path
    panel: Path
    prediction_output_name: str
    skipped_output_name: str
    include_tools: bool = False


BEDICT_PYTHON = Path(r"E:\AIselect\envs\bedict_v2\python.exe")
BEHIVE_EFF_PYTHON = Path(r"E:\AIselect\envs\behive_efficiency\python.exe")
FORECAST_PYTHON = Path(r"E:\AIselect\envs\forecastbe\Scripts\python.exe")
TF210_PYTHON = Path(r"E:\AIselect\envs\be_tf210\Scripts\python.exe")


SPECS = [
    T6Spec(
        "bedict_v2",
        "BEDICT-V2",
        BEDICT_PYTHON,
        ADAPTERS / "run_bedict_v2_pilot.py",
        ROOT / "data" / "pilot_panel_v0.csv",
        "predictions_bedict_v2_pilot.csv",
        "skipped_bedict_v2_pilot.csv",
    ),
    T6Spec(
        "behive_bystander",
        "BE-Hive bystander",
        BEDICT_PYTHON,
        ADAPTERS / "run_behive_bystander_pilot.py",
        ROOT / "data" / "pilot_panel_v1_behive.csv",
        "predictions_behive_bystander_pilot.csv",
        "skipped_behive_bystander_pilot.csv",
    ),
    T6Spec(
        "behive_efficiency",
        "BE-Hive efficiency",
        BEHIVE_EFF_PYTHON,
        ADAPTERS / "run_behive_efficiency_pilot.py",
        ROOT / "data" / "pilot_panel_v1_behive.csv",
        "predictions_behive_efficiency_pilot.csv",
        "skipped_behive_efficiency_pilot.csv",
    ),
    T6Spec(
        "forecast_be",
        "FORECasT-BE",
        FORECAST_PYTHON,
        ADAPTERS / "run_forecast_be_pilot.py",
        ROOT / "data" / "pilot_panel_v0.csv",
        "predictions_forecast_be_pilot.csv",
        "skipped_forecast_be_pilot.csv",
    ),
    T6Spec(
        "yachie_base_editing_prediction",
        "Yachie base-editing-prediction",
        BEDICT_PYTHON,
        ADAPTERS / "run_yachie_pilot.py",
        ROOT / "data" / "pilot_panel_v0.csv",
        "predictions_yachie_pilot.csv",
        "skipped_yachie_pilot.csv",
    ),
    T6Spec(
        "be_endo",
        "BE_Endo sequence-only",
        TF210_PYTHON,
        ADAPTERS / "run_be_endo_seq_pilot.py",
        ROOT / "data" / "pilot_panel_v2_be_endo.csv",
        "predictions_be_endo_seq_pilot.csv",
        "skipped_be_endo_seq_pilot.csv",
    ),
    T6Spec(
        "deepbe",
        "DeepBE",
        TF210_PYTHON,
        ADAPTERS / "run_deepbe_pilot.py",
        ROOT / "data" / "pilot_panel_v0.csv",
        "predictions_deepbe_pilot.csv",
        "skipped_deepbe_pilot.csv",
    ),
    T6Spec(
        "epcnnbe",
        "EPCNNBE",
        BEDICT_PYTHON,
        ADAPTERS / "run_epcnnbe_pilot.py",
        ROOT / "data" / "pilot_panel_v0.csv",
        "predictions_epcnnbe_pilot.csv",
        "skipped_epcnnbe_pilot.csv",
    ),
    T6Spec(
        "deepbaseeditor",
        "DeepBaseEditor",
        TF210_PYTHON,
        ADAPTERS / "run_deepbaseeditor_abe_efficiency_pilot.py",
        ROOT / "data" / "pilot_panel_v0.csv",
        "predictions_deepbaseeditor_abe_efficiency_pilot.csv",
        "skipped_deepbaseeditor_abe_efficiency_pilot.csv",
    ),
    T6Spec(
        "crispron_be",
        "CRISPRon-BE",
        TF210_PYTHON,
        ADAPTERS / "run_crispron_be_pilot.py",
        ROOT / "data" / "pilot_panel_v3_crispron_be.csv",
        "predictions_crispron_be_pilot.csv",
        "skipped_crispron_be_pilot.csv",
        include_tools=True,
    ),
]


def env_for_python(python_exe: Path, include_tools: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env_root = python_exe.parent
    if env_root.name.lower() == "scripts":
        env_root = env_root.parent
    parts = [
        str(env_root),
        str(env_root / "Library" / "bin"),
        str(env_root / "DLLs"),
        str(env_root / "Scripts"),
    ]
    if include_tools:
        parts.insert(0, str(Path(r"E:\AIselect\BEs\tools\bin")))
    parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join([p for p in parts if p])
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def read_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def run_one(spec: T6Spec, output_dir: Path) -> dict[str, Any]:
    model_dir = output_dir / "runs" / spec.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / f"{spec.model_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    panel_rows = read_row_count(spec.panel)
    cmd = [
        str(spec.python_exe),
        str(spec.adapter_script),
        "--panel",
        str(spec.panel),
        "--output-dir",
        str(model_dir),
    ]

    started_at = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    if not spec.python_exe.exists():
        returncode = None
        with log_path.open("w", encoding="utf-8") as log:
            print(f"Python executable not found: {spec.python_exe}", file=log)
    elif not spec.panel.exists():
        returncode = None
        with log_path.open("w", encoding="utf-8") as log:
            print(f"Panel not found: {spec.panel}", file=log)
    else:
        with log_path.open("w", encoding="utf-8") as log:
            print("$ " + " ".join(cmd), file=log)
            proc = subprocess.run(
                cmd,
                cwd=str(ADAPTERS),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_for_python(spec.python_exe, spec.include_tools),
            )
            print(f"# returncode={proc.returncode}", file=log)
            returncode = proc.returncode
    elapsed = perf_counter() - start
    ended_at = datetime.now(timezone.utc).isoformat()

    pred_path = model_dir / spec.prediction_output_name
    skipped_path = model_dir / spec.skipped_output_name
    copied_pred = copy_if_exists(pred_path, output_dir / "predictions" / "raw" / f"{spec.model_id}.csv")
    copied_skipped = copy_if_exists(skipped_path, output_dir / "skipped" / f"{spec.model_id}.csv")
    prediction_rows = read_row_count(pred_path)
    skipped_rows = read_row_count(skipped_path)
    seconds_per_1000 = (elapsed / panel_rows * 1000) if panel_rows else None

    return {
        "model_id": spec.model_id,
        "display_name": spec.display_name,
        "status": "completed" if returncode == 0 else "failed_or_not_run",
        "returncode": "" if returncode is None else returncode,
        "panel": str(spec.panel),
        "panel_rows": panel_rows,
        "prediction_rows": prediction_rows,
        "skipped_rows": skipped_rows,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_1000_panel_rows": "" if seconds_per_1000 is None else round(seconds_per_1000, 3),
        "python_exe": str(spec.python_exe),
        "adapter_script": str(spec.adapter_script),
        "log": str(log_path),
        "prediction_output": str(pred_path) if pred_path.exists() else "",
        "copied_prediction_output": copied_pred,
        "skipped_output": str(skipped_path) if skipped_path.exists() else "",
        "copied_skipped_output": copied_skipped,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", type=str, default=",".join(spec.model_id for spec in SPECS))
    args = parser.parse_args()

    requested = {x.strip() for x in args.models.split(",") if x.strip()}
    specs = [spec for spec in SPECS if spec.model_id in requested]
    if not specs:
        raise ValueError(f"No matching models for --models={args.models}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [run_one(spec, args.output_dir) for spec in specs]
    write_csv(args.output_dir / "T6_REPRODUCIBILITY_TIMING_STATUS.csv", run_rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "T6 reproducibility/applicability timing pass",
        "accuracy_leaderboard": False,
        "output_dir": str(args.output_dir),
        "models_requested": [spec.model_id for spec in specs],
        "completed_models": [row["model_id"] for row in run_rows if row["status"] == "completed"],
        "failed_or_not_run_models": [row["model_id"] for row in run_rows if row["status"] != "completed"],
        "status_csv": str(args.output_dir / "T6_REPRODUCIBILITY_TIMING_STATUS.csv"),
        "rows": run_rows,
    }
    (args.output_dir / "T6_REPRODUCIBILITY_TIMING_MANIFEST.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["failed_or_not_run_models"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
