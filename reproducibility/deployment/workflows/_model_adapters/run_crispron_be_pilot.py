from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from runtime_paths import model_path, tools_path

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


MODEL_ID = "crispron_be"
RUN_ID = "pilot_v3_crispron_be"
VALID_EDITORS = {"ABE": "0.5-0.5-0-0-0", "CBE": "0.5-0.5-0"}


def editor_for_crispron(row: dict[str, str]) -> str | None:
    editor = (row.get("editor") or row.get("editor_class") or "").upper()
    if "ABE" in editor:
        return "ABE"
    if "CBE" in editor or "BE4" in editor or "C>" in (row.get("intended_edit") or "").upper():
        return "CBE"
    return None


def is_valid_crispron_30mer(seq: str) -> bool:
    seq = seq.upper()
    return len(seq) == 30 and dna_only(seq) and "N" not in seq and seq[25:27] == "GG"


def write_native_fastas(rows: list[dict[str, str]], out_dir: Path) -> tuple[Path, Path]:
    thirty_path = out_dir / "30mers.fa"
    guides_path = out_dir / "23mers.fa"
    with thirty_path.open("w", encoding="utf-8", newline="\n") as thirty, guides_path.open("w", encoding="utf-8", newline="\n") as guides:
        for row in rows:
            seq = row["sequence_context"].upper()
            bid = row["benchmark_id"]
            thirty.write(f">{bid}\n{seq}\n")
            guides.write(f">{bid}\n{seq[4:27]}\n")
    return thirty_path, guides_path


def run_logged(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        print("$ " + " ".join(cmd), file=log)
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        print(f"# returncode={proc.returncode}", file=log)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with return code {proc.returncode}; see {log_path}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def percent_to_fraction(value: str) -> str:
    try:
        return clean_float(float(value) / 100.0)
    except (TypeError, ValueError):
        return ""


def raw_percent_to_proportion(freq_pct: str, eff_pct: str) -> str:
    try:
        freq = float(freq_pct)
        eff = float(eff_pct)
    except (TypeError, ValueError):
        return ""
    if eff <= 0:
        return ""
    return clean_float(freq / eff)


def model_dirs(repo: Path) -> list[str]:
    best = repo / "data" / "CRISPRon_models_V0" / "best"
    return [str(p) for p in sorted(best.iterdir(), key=lambda x: x.name) if p.is_dir()]


def run_editor(
    *,
    editor: str,
    rows: list[dict[str, str]],
    output_dir: Path,
    repo: Path,
    env: dict[str, str],
    python_exe: Path,
    run_ts: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    editor_dir = output_dir / editor.lower()
    editor_dir.mkdir(parents=True, exist_ok=True)
    log_path = editor_dir / "crispron_be.log"
    log_path.write_text("", encoding="utf-8")

    thirty_path, guides_path = write_native_fastas(rows, editor_dir)
    row_by_id = {row["benchmark_id"]: row for row in rows}

    run_logged(
        [
            str(python_exe),
            str(repo / "bin" / "CRISPRspec_CRISPRoff_pipeline.py"),
            "--guides",
            str(guides_path),
            "--specificity_report",
            str(editor_dir / "CRISPRspec.tsv"),
            "--guide_params_out",
            str(editor_dir / "CRISPRparams.tsv"),
            "--duplex_energy_params",
            str(repo / "data" / "model" / "energy_dics.pkl"),
            "--rnafold_x",
            "RNAfold",
            "--no_azimuth",
        ],
        cwd=repo,
        env=env,
        log_path=log_path,
    )

    run_logged(
        [
            str(python_exe),
            str(repo / "bin" / "DeepCRISPRon_eval.py"),
            str(editor_dir),
            str(thirty_path),
            str(editor_dir / "CRISPRparams.tsv"),
            *model_dirs(repo),
        ],
        cwd=repo,
        env=env,
        log_path=log_path,
    )

    run_logged(
        [
            str(python_exe),
            str(repo / "bin" / "DeepCRISPRonBE_eval.py"),
            editor,
            str(thirty_path),
            str(editor_dir / "CRISPRparams.tsv"),
            str(editor_dir / "crispron.csv"),
            str(repo / "data" / "CRISPRonBE_models"),
            str(editor_dir),
            VALID_EDITORS[editor],
        ],
        cwd=repo,
        env=env,
        log_path=log_path,
    )

    raw_path = editor_dir / f"crispron{editor}_prediction.tsv"
    raw_rows = read_tsv(raw_path)
    predictions: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    efficiency_seen: set[str] = set()
    pred_idx = 0

    for raw in raw_rows:
        benchmark_id = raw.get("seq_id", "")
        panel_row = row_by_id.get(benchmark_id, {})
        eff_pct = raw.get("pred_eff", "")
        freq_pct = raw.get("pred_freq", "")

        if benchmark_id and benchmark_id not in efficiency_seen:
            efficiency_seen.add(benchmark_id)
            pred_idx += 1
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_crispron_be_{editor.lower()}_eff_{pred_idx:05d}",
                    benchmark_id=benchmark_id,
                    model_id=MODEL_ID,
                    run_id=RUN_ID,
                    access_mode="local_hybrid",
                    task_type="efficiency",
                    predicted_efficiency=percent_to_fraction(eff_pct),
                    predicted_score=clean_float(eff_pct),
                    model_editor=f"CRISPRon-{editor}",
                    model_cell_context=panel_row.get("cell_context", ""),
                    raw_output_path=str(raw_path),
                    model_version="local official CRISPRon-BE models; adapter pilot_v3",
                    run_timestamp=run_ts,
                    notes=(
                        "CRISPRon-BE hybrid reproduction: Windows TensorFlow inference plus project-local WSL RNAfold helper. "
                        "Raw output reports percentages; predicted_efficiency is divided by 100."
                    ),
                )
            )

        pred_idx += 1
        predictions.append(
            prediction_row(
                prediction_id=f"pilot_crispron_be_{editor.lower()}_outcome_{pred_idx:05d}",
                benchmark_id=benchmark_id,
                model_id=MODEL_ID,
                run_id=RUN_ID,
                access_mode="local_hybrid",
                task_type="outcome_distribution",
                predicted_efficiency=percent_to_fraction(eff_pct),
                predicted_outcome=raw.get("outcome", ""),
                predicted_outcome_frequency=percent_to_fraction(freq_pct),
                predicted_outcome_proportion=raw_percent_to_proportion(freq_pct, eff_pct),
                predicted_score=clean_float(freq_pct),
                model_editor=f"CRISPRon-{editor}",
                model_cell_context=panel_row.get("cell_context", ""),
                raw_output_path=str(raw_path),
                model_version="local official CRISPRon-BE models; adapter pilot_v3",
                run_timestamp=run_ts,
                notes=(
                    "CRISPRon-BE outcome row. Raw pred_freq is a percentage of total reads; "
                    "predicted_outcome_frequency is divided by 100 and predicted_outcome_proportion is pred_freq/pred_eff."
                ),
            )
        )

    raw_manifest = [
        {"name": "30mers.fa", "path": str(thirty_path)},
        {"name": "23mers.fa", "path": str(guides_path)},
        {"name": "CRISPRparams.tsv", "path": str(editor_dir / "CRISPRparams.tsv")},
        {"name": "crispron.csv", "path": str(editor_dir / "crispron.csv")},
        {"name": raw_path.name, "path": str(raw_path)},
        {"name": "log", "path": str(log_path)},
    ]
    for row in rows:
        if row["benchmark_id"] not in efficiency_seen:
            skips.append(
                {
                    "benchmark_id": row["benchmark_id"],
                    "reason": f"CRISPRon-BE produced no {editor} prediction; usually no target base in the editing window",
                }
            )
    return predictions, raw_manifest, skips


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=model_path("BE_CRISPRON_REPO", "CRISPRon-BE", "repo"))
    parser.add_argument("--tools-bin", type=Path, default=tools_path())
    args = parser.parse_args()

    python_exe = Path(sys.executable)
    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()

    grouped: dict[str, list[dict[str, str]]] = {"ABE": [], "CBE": []}
    skips: list[dict[str, str]] = []
    for row in rows:
        benchmark_id = row["benchmark_id"]
        seq = (row.get("sequence_context") or "").upper()
        editor = editor_for_crispron(row)
        if editor is None or editor not in VALID_EDITORS:
            skips.append({"benchmark_id": benchmark_id, "reason": "no CRISPRon-BE editor mapping"})
            continue
        if not is_valid_crispron_30mer(seq):
            skips.append({"benchmark_id": benchmark_id, "reason": "CRISPRon-BE pilot requires 30nt DNA target with NGG PAM at positions 25-27"})
            continue
        grouped[editor].append(dict(row, sequence_context=seq))

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(args.tools_bin), env.get("PATH", "")])
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    all_predictions: list[dict[str, str]] = []
    raw_manifest: list[dict[str, str]] = []
    for editor, editor_rows in grouped.items():
        if not editor_rows:
            continue
        predictions, manifest, editor_skips = run_editor(
            editor=editor,
            rows=editor_rows,
            output_dir=out_dir,
            repo=args.repo,
            env=env,
            python_exe=python_exe,
            run_ts=run_ts,
        )
        all_predictions.extend(predictions)
        raw_manifest.extend(manifest)
        skips.extend(editor_skips)

    write_csv(out_dir / "predictions_crispron_be_pilot.csv", all_predictions)
    write_csv(out_dir / "skipped_crispron_be_pilot.csv", skips, ["benchmark_id", "reason"])
    write_csv(out_dir / "raw_crispron_be_manifest.csv", raw_manifest, ["name", "path"])

    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "predictions": len(all_predictions),
                "skipped": len(skips),
                "output": str(out_dir / "predictions_crispron_be_pilot.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
