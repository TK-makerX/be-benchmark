from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from common import clean_float, dna_only, prediction_row, read_csv, utc_now, write_csv


DEEPBE_ROOT = None
DEFAULT_WEIGHTS = None

PAM_VARIANTS = {
    "SpCas9": {
        "feature": 0,
        "model_file": "PAM_variant_SpCas9_model.h5",
        "aliases": ("SPCAS9", "CAS9", "NGG"),
    },
    "VRQR": {
        "feature": 1,
        "model_file": "PAM_variant_VRQR_model.h5",
        "aliases": ("VRQR",),
    },
    "SpCas9-NG": {
        "feature": 2,
        "model_file": "PAM_variant_NG_model.h5",
        "aliases": ("SPCAS9-NG", "SPCAS9NG", "SPRY-NG", "NG"),
    },
    "NRRH": {
        "feature": 3,
        "model_file": "PAM_variant_NRRH_model.h5",
        "aliases": ("NRRH",),
    },
    "NRTH": {
        "feature": 4,
        "model_file": "PAM_variant_NRTH_model.h5",
        "aliases": ("NRTH",),
    },
    "NRCH": {
        "feature": 5,
        "model_file": "PAM_variant_NRCH_model.h5",
        "aliases": ("NRCH",),
    },
    "SpG": {
        "feature": 6,
        "model_file": "PAM_variant_SpG_model.h5",
        "aliases": ("SPG",),
    },
    "SpRY": {
        "feature": 7,
        "model_file": "PAM_variant_SpRY_model.h5",
        "aliases": ("SPRY",),
    },
    "Sc++": {
        "feature": 8,
        "model_file": "PAM_variant_Sc++_model.h5",
        "aliases": ("SC++", "SCPLUSPLUS"),
    },
}


def one_hot_window(seq: str, start: int, length: int) -> np.ndarray:
    arr = np.zeros((length, 4), dtype=int)
    seq = seq.upper()
    for i in range(length):
        nt = seq[i + start]
        if nt == "A":
            arr[i, 0] = 1
        elif nt == "C":
            arr[i, 1] = 1
        elif nt == "G":
            arr[i, 2] = 1
        elif nt == "T":
            arr[i, 3] = 1
        else:
            raise ValueError(f"DeepBE requires A/C/G/T only; got {nt!r} in {seq!r}")
    return arr


def choose_deepbe_target(row: dict[str, str]) -> tuple[str, str]:
    """Return DeepBE's 30nt format: 4bp + 20bp protospacer + PAM + 3bp."""
    sequence = (row.get("sequence_context") or "").upper()
    protospacer = (row.get("protospacer_20nt") or "").upper()
    pam = (row.get("pam") or "").upper()

    if len(sequence) == 30 and dna_only(sequence) and "N" not in sequence:
        return sequence, "sequence_context_30nt"

    if len(protospacer) == 20 and dna_only(protospacer) and "N" not in protospacer:
        idx = sequence.find(protospacer) if sequence else -1
        if idx >= 4 and idx + 26 <= len(sequence):
            candidate = sequence[idx - 4 : idx + 26]
            if len(candidate) == 30 and dna_only(candidate) and "N" not in candidate:
                return candidate, "extracted_from_sequence_context"
        if len(pam) >= 3 and dna_only(pam[:3]) and "N" not in pam[:3]:
            # Last-resort construction: DeepBE needs flanks, so this is not used for
            # production scoring. It is intentionally rejected below by containing N.
            return "NNNN" + protospacer + pam[:3] + "NNN", "insufficient_flanks"
    return "", "no_30nt_deepbe_window"


def choose_pam_variant(row: dict[str, str]) -> tuple[str, int, str]:
    text = " ".join(
        [
            row.get("cas_variant", ""),
            row.get("editor", ""),
            row.get("notes", ""),
        ]
    ).upper()
    for variant, spec in PAM_VARIANTS.items():
        for alias in spec["aliases"]:
            if alias in text:
                return variant, int(spec["feature"]), "explicit"
    return "SpCas9", 0, "defaulted_unknown_to_SpCas9"


def outcome_index_sets(short_length: int = 8) -> list[list[int]]:
    alist = []
    for x in range(short_length):
        b = []
        for i in range(1, 2**x + 1):
            for k in range(1, 2 ** (7 - x) + 1):
                b.append(2 ** (8 - x) * i - k - 1)
        alist.append(b)
    return alist


def allowed_abe8e_indices(target: str, short_start: int = 6, short_length: int = 8) -> list[int]:
    motif = target[short_start : short_start + short_length]
    non = [2 ** (7 - k) for k, nt in enumerate(motif) if nt == "A"]
    on = []
    for j in range(len(non)):
        for combo in itertools.combinations(non, j + 1):
            on.append(sum(combo))
    return [idx - 1 for idx in on]


def edited_sequence_for_index(target: str, idx: int, alist: list[list[int]], short_start: int = 6) -> str:
    edited = target
    for k, indices in enumerate(alist):
        if idx in indices:
            edited = edited[: k + short_start] + "G" + edited[k + 1 + short_start :]
    return edited


def run_deepbe_8e(
    selected: list[dict[str, str]],
    weights: Path,
    out_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not selected:
        return [], []

    final_model = tf.keras.models.load_model(str(weights / "DeepBE" / "DeepBE_8e_model"), compile=False)
    x_seq = np.stack([one_hot_window(row["_deepbe_target"], start=5, length=10) for row in selected])
    x_pam = np.stack([one_hot_window(row["_deepbe_target"], start=0, length=30) for row in selected])
    x_prop = np.stack([one_hot_window(row["_deepbe_target"], start=4, length=20) for row in selected])

    pam_sum = np.zeros((len(selected),), dtype=float)
    raw_pam_rows: list[dict[str, str]] = []
    for variant in sorted(set(row["_pam_variant"] for row in selected)):
        spec = PAM_VARIANTS[variant]
        pam_model_path = weights / "PAM" / spec["model_file"]
        pam_model = tf.keras.models.load_model(str(pam_model_path), compile=False)
        indices = [i for i, row in enumerate(selected) if row["_pam_variant"] == variant]
        scores = pam_model.predict(x_pam[indices], batch_size=128, verbose=0).reshape(-1)
        for local_idx, score in zip(indices, scores):
            pam_features = np.zeros((9,), dtype=float)
            pam_features[int(spec["feature"])] = float(score)
            pam_sum[local_idx] = pam_features.sum()
            raw_pam_rows.append(
                {
                    "benchmark_id": selected[local_idx]["benchmark_id"],
                    "deepbe_target_30nt": selected[local_idx]["_deepbe_target"],
                    "pam_variant": variant,
                    "pam_feature": str(spec["feature"]),
                    "pam_score": clean_float(score),
                }
            )
        del pam_model
        tf.keras.backend.clear_session()

    raw_scores = pd.DataFrame(final_model.predict([[x_seq, pam_sum], x_prop], batch_size=128, verbose=0))
    alist = outcome_index_sets(short_length=8)

    raw_rows: list[dict[str, str]] = []
    for row_i, row in enumerate(selected):
        target = row["_deepbe_target"]
        values = raw_scores.iloc[row_i].astype(float).copy()
        eff = float(values.sum())
        allowed = set(allowed_abe8e_indices(target))
        if not allowed:
            continue
        disallowed = [idx for idx in range(len(values)) if idx not in allowed]
        values.iloc[disallowed] = 0.0
        pattern_sum = float(values.sum())
        if pattern_sum > 0:
            values = (values / pattern_sum) * eff

        for outcome_idx, score in enumerate(values.tolist()):
            if float(score) == 0.0:
                continue
            raw_rows.append(
                {
                    "benchmark_id": row["benchmark_id"],
                    "deepbe_target_30nt": target,
                    "edited_output_30nt": edited_sequence_for_index(target, outcome_idx, alist),
                    "pam_variant": row["_pam_variant"],
                    "pam_feature": row["_pam_feature"],
                    "outcome_index": str(outcome_idx),
                    "prediction_score": clean_float(score),
                }
            )

    raw_output = pd.DataFrame(raw_rows)
    raw_output.to_csv(out_dir / "raw_deepbe_pilot.csv", index=False)
    pd.DataFrame(raw_pam_rows).to_csv(out_dir / "raw_deepbe_pam_scores.csv", index=False)
    return raw_rows, raw_pam_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.panel)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    run_id = "pilot_v0_deepbe"

    selected: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    for row in rows:
        benchmark_id = row["benchmark_id"]
        if (row.get("editor_class") or "").upper() != "ABE":
            skips.append({"benchmark_id": benchmark_id, "reason": "DeepBE pilot wrapper currently maps only ABE rows to DeepBE_8e"})
            continue
        target, target_source = choose_deepbe_target(row)
        if len(target) != 30 or not dna_only(target) or "N" in target:
            skips.append({"benchmark_id": benchmark_id, "reason": f"DeepBE requires concrete 30nt 4+20+PAM+3 window; got {target_source}"})
            continue
        if "A" not in target[6:14]:
            skips.append({"benchmark_id": benchmark_id, "reason": "DeepBE_8e editable window has no A bases"})
            continue
        variant, feature, variant_source = choose_pam_variant(row)
        row = dict(row)
        row["_deepbe_target"] = target
        row["_target_source"] = target_source
        row["_pam_variant"] = variant
        row["_pam_feature"] = str(feature)
        row["_pam_variant_source"] = variant_source
        selected.append(row)

    raw_rows, _ = run_deepbe_8e(selected, args.weights, out_dir)
    raw_by_benchmark = {}
    for raw in raw_rows:
        raw_by_benchmark.setdefault(raw["benchmark_id"], []).append(raw)

    predictions: list[dict[str, str]] = []
    pred_idx = 0
    for row in selected:
        note = (
            "DeepBE ABE8e(V106W) local adapter using 30nt 4bp+20bp+PAM+3bp input. "
            f"target_source={row['_target_source']}; pam_variant_source={row['_pam_variant_source']}."
        )
        raw_for_benchmark = raw_by_benchmark.get(row["benchmark_id"], [])
        total_like_efficiency = 0.0
        for raw in raw_for_benchmark:
            try:
                total_like_efficiency += float(raw["prediction_score"])
            except (TypeError, ValueError):
                pass
        if raw_for_benchmark:
            pred_idx += 1
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_deepbe_total_{pred_idx:05d}",
                    benchmark_id=row["benchmark_id"],
                    model_id="deepbe",
                    run_id=run_id,
                    access_mode="local",
                    task_type="efficiency",
                    predicted_efficiency=clean_float(total_like_efficiency),
                    predicted_score=clean_float(total_like_efficiency),
                    model_editor=f"DeepBE_8e|{row['_pam_variant']}",
                    model_cell_context=row.get("cell_context", ""),
                    raw_output_path=str(out_dir / "raw_deepbe_pilot.csv"),
                    model_version="local release weights; adapter pilot_v0",
                    run_timestamp=run_ts,
                    notes=(
                        note
                        + " T1-derived total-like efficiency is the sum of DeepBE predicted edited-outcome scores; "
                        "use as supplementary, not primary scalar-efficiency evidence."
                    ),
                )
            )
        for raw in raw_for_benchmark:
            pred_idx += 1
            predictions.append(
                prediction_row(
                    prediction_id=f"pilot_deepbe_{pred_idx:05d}",
                    benchmark_id=row["benchmark_id"],
                    model_id="deepbe",
                    run_id=run_id,
                    access_mode="local",
                    task_type="outcome_distribution",
                    predicted_outcome=raw["edited_output_30nt"],
                    predicted_outcome_frequency=raw["prediction_score"],
                    predicted_outcome_proportion=raw["prediction_score"],
                    predicted_score=raw["prediction_score"],
                    model_editor=f"DeepBE_8e|{row['_pam_variant']}",
                    model_cell_context=row.get("cell_context", ""),
                    raw_output_path=str(out_dir / "raw_deepbe_pilot.csv"),
                    model_version="local release weights; adapter pilot_v0",
                    run_timestamp=run_ts,
                    notes=note,
                )
            )

    write_csv(out_dir / "predictions_deepbe_pilot.csv", predictions)
    write_csv(out_dir / "skipped_deepbe_pilot.csv", skips, ["benchmark_id", "reason"])
    print(
        json.dumps(
            {
                "model_id": "deepbe",
                "selected_inputs": len(selected),
                "predictions": len(predictions),
                "skipped": len(skips),
                "output": str(out_dir / "predictions_deepbe_pilot.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
