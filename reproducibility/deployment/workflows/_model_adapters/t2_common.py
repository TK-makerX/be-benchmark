from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


EVENT_RE = re.compile(r"^([ACGT])(-?\d+)>([ACGT])$")


def clean_float(value) -> str:
    if value is None or value == "":
        return ""
    try:
        out = float(value)
    except Exception:
        return ""
    if math.isnan(out) or math.isinf(out):
        return ""
    return f"{out:.12g}"


def dna(seq: str) -> bool:
    return bool(seq) and set(str(seq).upper()) <= set("ACGT")


def events_from_sequences(reference: str, outcome: str, protospacer_start: int, min_pos: int, max_pos: int) -> str:
    reference = str(reference).upper()
    outcome = str(outcome).upper()
    if len(reference) != len(outcome):
        return ""
    events = []
    for idx, (ref_base, out_base) in enumerate(zip(reference, outcome)):
        if ref_base == out_base:
            continue
        if ref_base not in "ACGT" or out_base not in "ACGT":
            continue
        pos = idx - protospacer_start + 1
        if min_pos <= pos <= max_pos:
            events.append(f"{ref_base}{pos}>{out_base}")
    return ";".join(events) if events else "no_tracked_substrate_change"


def parse_events(events: str) -> list[tuple[str, int, str]]:
    parsed = []
    for part in str(events).split(";"):
        match = EVENT_RE.match(part)
        if match:
            parsed.append((match.group(1), int(match.group(2)), match.group(3)))
    return parsed


def outcome_rows_to_position_rows(
    outcome_rows: Iterable[dict],
    *,
    id_prefix: str,
    value_col: str = "predicted_outcome_frequency",
    score_col: str = "predicted_outcome_score",
) -> list[dict]:
    df = pd.DataFrame(list(outcome_rows))
    if df.empty:
        return []
    position_rows = []
    for (target_id, model_id, run_id, model_editor, cell, scale, relation), sub in df.groupby(
        [
            "target_profile_id",
            "model_id",
            "run_id",
            "model_editor",
            "model_cell_context",
            "prediction_scale",
            "source_relation_to_model",
        ],
        dropna=False,
        sort=False,
    ):
        pos_values: dict[tuple[int, str], float] = {}
        for _, row in sub.iterrows():
            events = parse_events(row.get("predicted_outcome_events", ""))
            try:
                value = float(row.get(value_col, ""))
            except Exception:
                try:
                    value = float(row.get(score_col, ""))
                except Exception:
                    continue
            for ref_base, pos, _alt_base in events:
                key = (pos, ref_base)
                pos_values[key] = pos_values.get(key, 0.0) + value
        if scale == "conditional_outcome_frequency_given_edited_reads":
            position_scale = "conditional_position_rate_given_edited_reads"
        else:
            position_scale = str(scale).replace("outcome", "position")
        for (pos, ref_base), value in sorted(pos_values.items()):
            position_rows.append(
                {
                    "prediction_id": f"{id_prefix}_{len(position_rows) + 1:08d}",
                    "target_profile_id": target_id,
                    "model_id": model_id,
                    "run_id": run_id,
                    "model_editor": model_editor,
                    "model_cell_context": cell,
                    "protospacer_position": pos,
                    "reference_base": ref_base,
                    "predicted_position_rate": clean_float(value) if "conditional" in str(scale) else "",
                    "predicted_position_score": clean_float(value),
                    "prediction_scale": position_scale,
                    "source_relation_to_model": relation,
                    "notes": "Position profile derived by marginalizing model outcome/event rows.",
                }
            )
    return position_rows


def normalize_outcome_scores(rows: list[dict], *, score_key: str, output_key: str) -> list[dict]:
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    score = pd.to_numeric(df[score_key], errors="coerce").fillna(0.0)
    df["_score_num"] = score
    sums = df.groupby("target_profile_id")["_score_num"].transform("sum")
    df[output_key] = [
        clean_float(v / s) if s > 0 else ""
        for v, s in zip(df["_score_num"].tolist(), sums.tolist())
    ]
    df = df.drop(columns=["_score_num"])
    return df.to_dict(orient="records")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
