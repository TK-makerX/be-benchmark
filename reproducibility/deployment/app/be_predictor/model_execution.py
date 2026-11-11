"""Optional real model execution layer.

The router is the safe default. This module is used only when the user asks the
MVP to invoke real local model backends.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
FORECAST_WORKER = SYSTEM_ROOT / "connectors" / "forecast_be_worker.py"
DEEPBASEEDITOR_WORKER = SYSTEM_ROOT / "connectors" / "deepbaseeditor_abe_worker.py"
def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()

RELEASE_ROOT = SYSTEM_ROOT.parent
MODELS_ROOT = _env_path("BE_MODELS_DIR", RELEASE_ROOT / "models")
BENCHMARK_CORE = _env_path("BE_BENCHMARK_CORE", RELEASE_ROOT / "workflows")
ADAPTERS_DIR = _env_path("BE_ADAPTERS_DIR", BENCHMARK_CORE / "_model_adapters")
FORECAST_PYTHON = _env_path("BE_FORECAST_PYTHON", Path(sys.executable))
DEEPBASEEDITOR_PYTHON = _env_path("BE_TF210_PYTHON", Path(sys.executable))
BEDICT_PYTHON = _env_path("BE_BEDICT_PYTHON", Path(sys.executable))
BEHIVE_EFFICIENCY_PYTHON = _env_path("BE_BEHIVE_PYTHON", Path(sys.executable))
TF210_PYTHON = _env_path("BE_TF210_PYTHON", Path(sys.executable))
TOOLS_BIN = _env_path("BE_TOOLS_DIR", RELEASE_ROOT / "tools" / "bin")


ADAPTER_BACKENDS: Dict[str, Dict[str, Any]] = {
    "bedict_v2": {
        "python": BEDICT_PYTHON,
        "script": ADAPTERS_DIR / "run_bedict_v2_pilot.py",
        "predictions": "predictions_bedict_v2_pilot.csv",
        "skipped": "skipped_bedict_v2_pilot.csv",
        "timeout": 180,
    },
    "behive_bystander": {
        "python": BEDICT_PYTHON,
        "script": ADAPTERS_DIR / "run_behive_bystander_pilot.py",
        "predictions": "predictions_behive_bystander_pilot.csv",
        "skipped": "skipped_behive_bystander_pilot.csv",
        "timeout": 180,
    },
    "behive_efficiency": {
        "python": BEHIVE_EFFICIENCY_PYTHON,
        "script": ADAPTERS_DIR / "run_behive_efficiency_pilot.py",
        "predictions": "predictions_behive_efficiency_pilot.csv",
        "skipped": "skipped_behive_efficiency_pilot.csv",
        "timeout": 180,
        "env_kind": "behive_efficiency",
    },
    "deepbe": {
        "python": TF210_PYTHON,
        "script": ADAPTERS_DIR / "run_deepbe_pilot.py",
        "predictions": "predictions_deepbe_pilot.csv",
        "skipped": "skipped_deepbe_pilot.csv",
        "timeout": 240,
    },
    "yachie_base_editing_prediction": {
        "python": BEDICT_PYTHON,
        "script": ADAPTERS_DIR / "run_yachie_pilot.py",
        "predictions": "predictions_yachie_pilot.csv",
        "skipped": "skipped_yachie_pilot.csv",
        "timeout": 120,
    },
    "be_endo": {
        "python": TF210_PYTHON,
        "script": ADAPTERS_DIR / "run_be_endo_seq_pilot.py",
        "predictions": "predictions_be_endo_seq_pilot.csv",
        "skipped": "skipped_be_endo_seq_pilot.csv",
        "timeout": 240,
    },
    "epcnnbe": {
        "python": BEDICT_PYTHON,
        "script": ADAPTERS_DIR / "run_epcnnbe_pilot.py",
        "predictions": "predictions_epcnnbe_pilot.csv",
        "skipped": "skipped_epcnnbe_pilot.csv",
        "timeout": 180,
    },
    "crispron_be": {
        "python": TF210_PYTHON,
        "script": ADAPTERS_DIR / "run_crispron_be_pilot.py",
        "predictions": "predictions_crispron_be_pilot.csv",
        "skipped": "skipped_crispron_be_pilot.csv",
        "timeout": 300,
        "env_kind": "crispron_be",
    },
}


DEFAULT_BACKENDS = [
    "forecast_be",
    "deepbaseeditor",
    "deepbe",
    "bedict_v2",
    "behive_bystander",
    "behive_efficiency",
    "yachie_base_editing_prediction",
    "be_endo",
    "epcnnbe",
    "crispron_be",
]


def _protospacer_20nt(sequence: str) -> str:
    seq = (sequence or "").upper()
    if len(seq) == 30:
        return seq[4:24]
    if len(seq) == 40:
        return seq[10:30]
    if len(seq) >= 23 and seq[20:23]:
        return seq[:20]
    if len(seq) >= 20:
        return seq[:20]
    return seq


def _pam3(sequence: str) -> str:
    seq = (sequence or "").upper()
    if len(seq) == 23:
        return seq[20:23]
    if len(seq) == 30:
        return seq[24:27]
    if len(seq) == 40:
        return seq[30:33]
    if len(seq) >= 3:
        return seq[-3:]
    return ""


def _pam4(sequence: str) -> str:
    seq = (sequence or "").upper()
    if len(seq) == 30:
        return seq[24:28]
    pam = _pam3(seq)
    if len(pam) == 3:
        return pam + "N"
    return pam[:4]


def _infer_adapter_cas(sequence: str, editor: str, cas_variant: str = "") -> str:
    text = " ".join([editor or "", cas_variant or ""]).upper()
    if "SPRY" in text:
        return "SpRY"
    if "NG" in text and "NGG" not in text:
        return "NG"
    pam = _pam3(sequence)
    if len(pam) == 3 and pam[1:] == "GG":
        return "SpCas9"
    return cas_variant or ""


def _adapter_editor_name(editor: str, editor_family: str, backend_id: str) -> str:
    text = (editor or "").strip()
    upper = text.upper()
    if backend_id == "bedict_v2":
        if "ABEMAX" in upper:
            return "ABEmax"
        if "ABE8E" in upper:
            return "ABE8e"
        if editor_family == "ABE":
            # BEDICT-V2 supports named ABEmax/ABE8e models; use ABE8e only
            # when the user did not provide a more specific compatible name.
            return "ABE8e"
    if backend_id in {"behive_bystander", "behive_efficiency"}:
        return text or ("BE4" if editor_family == "CBE" else "ABE")
    return text or editor_family


def _write_single_panel(
    *,
    path: Path,
    sequence: str,
    editor: str,
    editor_family: str,
    cell_context: str = "",
    cas_variant: str = "",
    reference_outcome: str = "",
    backend_id: str = "",
    duplicate_for_single_sample_model: bool = False,
) -> None:
    sequence = (sequence or "").upper()
    fields = [
        "benchmark_id",
        "sequence_context",
        "protospacer_20nt",
        "pam",
        "editor",
        "editor_class",
        "cas_variant",
        "cell_context",
        "reference_outcome",
        "intended_edit",
        "notes",
    ]
    row = {
        "benchmark_id": "user_input_0001",
        "sequence_context": sequence,
        "protospacer_20nt": _protospacer_20nt(sequence),
        "pam": _pam4(sequence) if backend_id == "bedict_v2" else _pam3(sequence),
        "editor": _adapter_editor_name(editor, editor_family, backend_id),
        "editor_class": editor_family,
        "cas_variant": _infer_adapter_cas(sequence, editor, cas_variant),
        "cell_context": cell_context,
        "reference_outcome": (reference_outcome or "").upper(),
        "intended_edit": "A>G" if editor_family == "ABE" else ("C>T" if editor_family == "CBE" else ""),
        "notes": "single-sample prediction-system connector panel",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        if duplicate_for_single_sample_model:
            padding = dict(row)
            padding["benchmark_id"] = "user_input_padding"
            writer.writerow(padding)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_prediction_rows(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    efficiency = []
    positions = []
    outcomes = []
    for row in rows:
        task_type = row.get("task_type", "")
        if task_type == "efficiency" or (not task_type and row.get("predicted_efficiency")):
            eff = _as_float(row.get("predicted_efficiency"))
            score = _as_float(row.get("predicted_score"))
            if eff is not None or score is not None:
                efficiency.append(
                    {
                        "predicted_efficiency": eff,
                        "predicted_score": score,
                        "model_editor": row.get("model_editor", ""),
                        "notes": row.get("notes", ""),
                    }
                )
        if row.get("predicted_position"):
            positions.append(
                {
                    "position": int(float(row["predicted_position"])),
                    "score": _as_float(row.get("predicted_score")),
                    "model_editor": row.get("model_editor", ""),
                }
            )
        if row.get("predicted_outcome"):
            outcomes.append(
                {
                    "outcome": row.get("predicted_outcome", ""),
                    "frequency": _as_float(row.get("predicted_outcome_frequency")),
                    "proportion": _as_float(row.get("predicted_outcome_proportion")),
                    "score": _as_float(row.get("predicted_score")),
                    "model_editor": row.get("model_editor", ""),
                }
            )
    outcomes.sort(
        key=lambda item: (
            item.get("proportion") is not None,
            item.get("frequency") is not None,
            item.get("score") is not None,
            item.get("proportion") or item.get("frequency") or item.get("score") or float("-inf"),
        ),
        reverse=True,
    )
    return {
        "efficiency": efficiency[:5],
        "position_scores": positions[:30],
        "top_outcomes": outcomes[:10],
    }


def _env_for_backend(spec: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    kind = spec.get("env_kind", "")
    python_path = Path(spec["python"])
    if kind == "behive_efficiency":
        env_root = python_path.parent
        env["PATH"] = os.pathsep.join(
            [
                str(env_root),
                str(env_root / "Scripts"),
                str(env_root / "Library" / "bin"),
                env.get("PATH", ""),
            ]
        )
    if kind == "crispron_be":
        env["PATH"] = os.pathsep.join([str(TOOLS_BIN), env.get("PATH", "")])
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def run_benchmark_adapter_backend(
    backend_id: str,
    sequence: str,
    editor: str,
    editor_family: str,
    *,
    cell_context: str = "",
    cas_variant: str = "",
    reference_outcome: str = "",
) -> Dict[str, Any]:
    spec = ADAPTER_BACKENDS[backend_id]
    python_exe = Path(spec["python"])
    script = Path(spec["script"])
    if not python_exe.exists():
        return {"model_id": backend_id, "status": "failed", "reason": f"Python not found: {python_exe}"}
    if not script.exists():
        return {"model_id": backend_id, "status": "failed", "reason": f"Adapter script not found: {script}"}

    with tempfile.TemporaryDirectory(prefix=f"be_prediction_system_{backend_id}_") as tmp:
        tmp_dir = Path(tmp)
        panel = tmp_dir / "panel.csv"
        out_dir = tmp_dir / "adapter_output"
        _write_single_panel(
            path=panel,
            sequence=sequence,
            editor=editor,
            editor_family=editor_family,
            cell_context=cell_context,
            cas_variant=cas_variant,
            reference_outcome=reference_outcome,
            backend_id=backend_id,
            duplicate_for_single_sample_model=(backend_id == "epcnnbe"),
        )
        cmd = [str(python_exe), str(script), "--panel", str(panel), "--output-dir", str(out_dir)]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ADAPTERS_DIR),
                text=True,
                capture_output=True,
                timeout=int(spec.get("timeout", 180)),
                check=False,
                env=_env_for_backend(spec),
            )
        except subprocess.TimeoutExpired:
            return {
                "model_id": backend_id,
                "status": "failed",
                "reason": f"{backend_id} timed out after {spec.get('timeout', 180)} seconds.",
            }
        if completed.returncode != 0:
            return {
                "model_id": backend_id,
                "status": "failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout[-3000:],
                "stderr": completed.stderr[-5000:],
            }
        pred_path = out_dir / spec["predictions"]
        skip_path = out_dir / spec["skipped"]
        prediction_rows = [
            row for row in _read_csv_rows(pred_path)
            if row.get("benchmark_id") in {"", "user_input_0001"}
        ]
        skipped_rows = [
            row for row in _read_csv_rows(skip_path)
            if row.get("benchmark_id") in {"", "user_input_0001"}
        ]
        if not prediction_rows:
            reason = "; ".join(sorted({row.get("reason", "") for row in skipped_rows if row.get("reason")}))
            return {
                "model_id": backend_id,
                "status": "skipped",
                "reason": reason or "Adapter completed but returned no prediction rows.",
                "skipped_rows": skipped_rows[:5],
                "stdout": completed.stdout[-2000:],
            }
        summary = _summarize_prediction_rows(prediction_rows)
        return {
            "model_id": backend_id,
            "status": "completed",
            "prediction_count": len(prediction_rows),
            "skipped_count": len(skipped_rows),
            "prediction_summary": summary,
            "stdout": completed.stdout[-2000:],
            "stderr_note": completed.stderr[-2000:] if completed.stderr else "",
        }


def run_forecast_be(sequence: str, editor_family: str, timeout_seconds: int = 60) -> Dict[str, Any]:
    if editor_family not in {"ABE", "CBE"}:
        return {
            "model_id": "forecast_be",
            "status": "skipped",
            "reason": f"FORECasT-BE requires ABE or CBE editor family, got {editor_family}.",
        }
    if not FORECAST_PYTHON.exists():
        return {
            "model_id": "forecast_be",
            "status": "failed",
            "reason": f"FORECasT-BE Python environment not found: {FORECAST_PYTHON}",
        }
    protospacer = _protospacer_20nt(sequence)
    if len(protospacer) != 20 or set(protospacer) - set("ACGT"):
        return {
            "model_id": "forecast_be",
            "status": "skipped",
            "reason": "FORECasT-BE connector needs a concrete 20nt A/C/G/T protospacer.",
            "protospacer_20nt": protospacer,
        }
    cmd = [
        str(FORECAST_PYTHON),
        str(FORECAST_WORKER),
        "--sequence",
        protospacer,
        "--editor",
        editor_family,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(SYSTEM_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "model_id": "forecast_be",
            "status": "failed",
            "reason": f"FORECasT-BE timed out after {timeout_seconds} seconds.",
            "protospacer_20nt": protospacer,
        }
    if completed.returncode != 0:
        return {
            "model_id": "forecast_be",
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
            "protospacer_20nt": protospacer,
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "model_id": "forecast_be",
            "status": "failed",
            "reason": f"Could not parse FORECasT-BE JSON: {exc}",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
            "protospacer_20nt": protospacer,
        }
    payload["stderr_note"] = completed.stderr[-2000:] if completed.stderr else ""
    return payload


def _target_30nt(sequence: str) -> str:
    seq = (sequence or "").upper()
    if len(seq) == 30:
        return seq
    return ""


def run_deepbaseeditor_abe(sequence: str, editor_family: str, timeout_seconds: int = 180) -> Dict[str, Any]:
    if editor_family != "ABE":
        return {
            "model_id": "deepbaseeditor",
            "status": "skipped",
            "reason": f"DeepBaseEditor ABE_Efficiency connector requires ABE editor family, got {editor_family}.",
        }
    if not DEEPBASEEDITOR_PYTHON.exists():
        return {
            "model_id": "deepbaseeditor",
            "status": "failed",
            "reason": f"DeepBaseEditor Python environment not found: {DEEPBASEEDITOR_PYTHON}",
        }
    target = _target_30nt(sequence)
    if len(target) != 30 or set(target) - set("ACGT"):
        return {
            "model_id": "deepbaseeditor",
            "status": "skipped",
            "reason": "DeepBaseEditor ABE_Efficiency needs a concrete 30nt target window: 4bp + 20bp protospacer + PAM + 3bp.",
            "input_length": len(sequence or ""),
        }
    cmd = [
        str(DEEPBASEEDITOR_PYTHON),
        str(DEEPBASEEDITOR_WORKER),
        "--sequence",
        target,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(SYSTEM_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "model_id": "deepbaseeditor",
            "status": "failed",
            "reason": f"DeepBaseEditor timed out after {timeout_seconds} seconds.",
            "target_30nt": target,
        }
    if completed.returncode != 0:
        return {
            "model_id": "deepbaseeditor",
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
            "target_30nt": target,
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "model_id": "deepbaseeditor",
            "status": "failed",
            "reason": f"Could not parse DeepBaseEditor JSON: {exc}",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
            "target_30nt": target,
        }
    payload["stderr_note"] = completed.stderr[-2000:] if completed.stderr else ""
    return payload


def run_selected_backends(
    sequence: str,
    editor_family: str,
    editor: str = "",
    backend_ids: Optional[Iterable[str]] = None,
    cell_context: str = "",
    cas_variant: str = "",
    reference_outcome: str = "",
) -> List[Dict[str, Any]]:
    requested = list(backend_ids or DEFAULT_BACKENDS)
    outputs: List[Dict[str, Any]] = []
    for backend_id in requested:
        if backend_id == "forecast_be":
            outputs.append(run_forecast_be(sequence, editor_family))
        elif backend_id == "deepbaseeditor":
            outputs.append(run_deepbaseeditor_abe(sequence, editor_family))
        elif backend_id in ADAPTER_BACKENDS:
            outputs.append(
                run_benchmark_adapter_backend(
                    backend_id,
                    sequence,
                    editor,
                    editor_family,
                    cell_context=cell_context,
                    cas_variant=cas_variant,
                    reference_outcome=reference_outcome,
                )
            )
        else:
            outputs.append(
                {
                    "model_id": backend_id,
                    "status": "not_implemented",
                    "reason": "This backend connector has not been implemented in the MVP yet.",
                }
            )
    return outputs
