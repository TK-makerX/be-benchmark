#!/usr/bin/env python
"""
Prepare model-specific T2 expanded panels.

The panels stay on the unified T2 target matrix, but each model gets the
sequence window and product subset it can actually consume.
"""

from __future__ import annotations

import os
import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "benchmark_core"
T2 = ROOT / "data" / "processed" / "T2_bystander_matrix"
OUT = T2 / "panels"
BEHIVE_MODELS = Path(os.environ.get("BE_MODELS_DIR", os.environ.get("BE_REPLAY_PROJECT", "."))) / "BE-Hive-bystander/repo/models.csv"

EVENT_RE = re.compile(r"^([ACGT])(-?\d+)>([ACGT])$")


def dna(seq: str) -> bool:
    return bool(seq) and set(str(seq).upper()) <= set("ACGT")


def sample_by_group(df: pd.DataFrame, group_cols: list[str], n: int, max_rows: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    parts = []
    for _, sub in df.groupby(group_cols, dropna=False, sort=True):
        sub = sub.copy()
        sub["_outcomes_sort"] = pd.to_numeric(sub.get("outcome_row_count"), errors="coerce")
        sub["_eff_sort"] = pd.to_numeric(sub.get("measured_total_efficiency"), errors="coerce")
        sub = sub.sort_values(["_outcomes_sort", "_eff_sort", "target_profile_id"], ascending=[True, False, True])
        parts.append(sub.head(n))
    out = pd.concat(parts, ignore_index=True).drop(columns=["_outcomes_sort", "_eff_sort"], errors="ignore")
    if max_rows and len(out) > max_rows:
        out = out.sort_values(group_cols + ["target_profile_id"]).head(max_rows)
    return out.reset_index(drop=True)


def behive_supported_pairs() -> set[tuple[str, str]]:
    models = pd.read_csv(BEHIVE_MODELS, dtype=str).fillna("")
    pairs: set[tuple[str, str]] = set()
    for _, row in models.iterrows():
        cell = row["Celltype"]
        pairs.add((row["Public base editor"], cell))
        pairs.add((row["Internal base editor"], cell))
    return pairs


def find_protospacer_start(row: pd.Series) -> int:
    seq = str(row.get("sequence_50nt", "")).upper()
    protospacer = str(row.get("protospacer_20nt", "")).upper()
    if not dna(seq) or len(seq) != 50 or not dna(protospacer) or len(protospacer) != 20:
        return -1
    return seq.find(protospacer)


def add_windows(target: pd.DataFrame) -> pd.DataFrame:
    target = target.copy()
    starts = []
    seq30 = []
    seq40 = []
    pam4 = []
    for _, row in target.iterrows():
        seq = str(row.get("sequence_50nt", "")).upper()
        protospacer = str(row.get("protospacer_20nt", "")).upper()
        start = find_protospacer_start(row)
        starts.append(start)
        if start >= 4 and start + 26 <= len(seq):
            seq30.append(seq[start - 4 : start + 26])
        else:
            seq30.append("")
        if start >= 10 and start + 30 <= len(seq):
            seq40.append(seq[start - 10 : start + 30])
        else:
            seq40.append("")
        if start >= 0 and start + 24 <= len(seq):
            pam4.append(seq[start + 20 : start + 24])
        elif len(str(row.get("pam", ""))) == 3:
            pam4.append(str(row.get("pam", "")).upper() + "N")
        else:
            pam4.append("")
    target["protospacer_start_in_50nt"] = starts
    target["sequence_context_30nt"] = seq30
    target["sequence_context_40nt"] = seq40
    target["pam4"] = pam4
    target["benchmark_id"] = target["target_profile_id"]
    return target


def parse_event(event: str) -> tuple[str, int, str] | None:
    match = EVENT_RE.match(str(event))
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def events_ok(events: str, min_pos: int, max_pos: int, ref: str | None = None, alt: str | None = None) -> bool:
    events = str(events)
    if not events or events == "no_tracked_substrate_change":
        return False
    for part in events.split(";"):
        parsed = parse_event(part)
        if parsed is None:
            return False
        rbase, pos, abase = parsed
        if pos < min_pos or pos > max_pos:
            return False
        if ref is not None and rbase != ref:
            return False
        if alt is not None and abase != alt:
            return False
    return True


def target_event_flags(outcome: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_id, sub in outcome.groupby("target_profile_id", sort=False):
        events = sub["outcome_events"].astype(str).tolist()
        rows.append(
            {
                "target_profile_id": target_id,
                "observed_all_1_20_ag": all(events_ok(e, 1, 20, "A", "G") for e in events),
                "observed_all_3_10_ag": all(events_ok(e, 3, 10, "A", "G") for e in events),
                "observed_all_minus3_20_any": all(events_ok(e, -3, 20) for e in events),
                "observed_all_1_20_any": all(events_ok(e, 1, 20) for e in events),
            }
        )
    return pd.DataFrame(rows)


def write_panel(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=T2)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--behive-per-group", type=int, default=3)
    parser.add_argument("--forecast-per-group", type=int, default=4)
    parser.add_argument("--max-behive", type=int, default=100)
    parser.add_argument("--max-forecast", type=int, default=160)
    parser.add_argument("--max-bedict", type=int, default=40)
    parser.add_argument("--max-deepbe", type=int, default=40)
    parser.add_argument("--max-crispron", type=int, default=40)
    parser.add_argument("--max-be-endo", type=int, default=40)
    parser.add_argument("--max-yachie", type=int, default=40)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(args.matrix_dir / "T2_observed_target_profiles.csv", dtype=str, low_memory=False).fillna("")
    outcome = pd.read_csv(args.matrix_dir / "T2_observed_outcome_profiles.csv", dtype=str, low_memory=False).fillna("")
    target = add_windows(target)
    flags = target_event_flags(outcome)
    target = target.merge(flags, on="target_profile_id", how="left")

    target["outcome_row_count_num"] = pd.to_numeric(target["outcome_row_count"], errors="coerce")
    complete = target["outcome_row_count_num"].le(50)
    qc_b = target["qc_level"].eq("B")
    concrete30 = target["sequence_context_30nt"].map(lambda s: len(str(s)) == 30 and dna(str(s)))
    concrete40 = target["sequence_context_40nt"].map(lambda s: len(str(s)) == 40 and dna(str(s)))
    prot20 = target["protospacer_20nt"].map(lambda s: len(str(s)) == 20 and dna(str(s)))

    supported_behive = behive_supported_pairs()
    behive = target[
        qc_b
        & target["behive_bystander_input_eligible"].eq("True")
        & complete
        & target["cell_context"].isin(["mES", "HEK293T"])
    ].copy()
    behive = behive[behive.apply(lambda row: (row["editor"], row["cell_context"]) in supported_behive, axis=1)]
    behive = sample_by_group(behive, ["cell_context", "editor_class", "editor"], args.behive_per_group, args.max_behive)

    forecast = target[
        qc_b
        & target["forecast_position_input_eligible"].eq("True")
        & target["editor_class"].isin(["ABE", "CBE"])
        & prot20
    ].copy()
    forecast = sample_by_group(forecast, ["cell_context", "editor_class", "source_dataset"], args.forecast_per_group, args.max_forecast)

    bedict = target[
        qc_b
        & complete
        & prot20
        & target["pam4"].map(lambda s: len(str(s)) == 4 and dna(str(s)))
        & target["editor_class"].eq("ABE")
        & target["observed_all_1_20_ag"].eq(True)
    ].copy()
    bedict["t2_model_editor"] = "ABEmax-SpCas9"
    bedict["source_relation_to_model_for_t2"] = "proxy_external_candidate"
    bedict = sample_by_group(bedict, ["editor", "cell_context"], 20, args.max_bedict)

    deepbe = target[
        qc_b
        & complete
        & concrete30
        & target["editor_class"].eq("ABE")
        & target["observed_all_3_10_ag"].eq(True)
    ].copy()
    deepbe = deepbe[deepbe["sequence_context_30nt"].str.slice(6, 14).str.contains("A", regex=False)]
    deepbe["source_relation_to_model_for_t2"] = "proxy_external_candidate"
    deepbe = sample_by_group(deepbe, ["editor", "cell_context"], 20, args.max_deepbe)

    crispron = target[
        qc_b
        & complete
        & concrete30
        & target["sequence_context_30nt"].str.slice(25, 27).eq("GG")
        & target["editor_class"].isin(["ABE", "CBE"])
        & target["observed_all_minus3_20_any"].eq(True)
    ].copy()
    crispron["source_relation_to_model_for_t2"] = "external_candidate"
    crispron = sample_by_group(crispron, ["editor_class", "editor", "cell_context"], 8, args.max_crispron)

    be_endo = target[
        qc_b
        & concrete40
        & target["sequence_context_40nt"].str.slice(30, 33).isin(["AGG", "CGG", "TGG", "GGG"])
        & target["editor_class"].isin(["ABE", "CBE"])
        & target["observed_all_1_20_any"].eq(True)
    ].copy()
    be_endo["source_relation_to_model_for_t2"] = "external_candidate"
    be_endo = sample_by_group(be_endo, ["editor_class", "editor", "cell_context"], 8, args.max_be_endo)

    yachie = target[
        qc_b
        & complete
        & concrete30
        & target["editor_class"].isin(["ABE", "CBE"])
        & target["observed_all_minus3_20_any"].eq(True)
    ].copy()
    yachie["source_relation_to_model_for_t2"] = "proxy_external_candidate"
    yachie = sample_by_group(yachie, ["editor_class", "editor", "cell_context"], 8, args.max_yachie)

    panels = {
        "behive_bystander": ("T2_expand_behive_native50.csv", behive),
        "forecast_be": ("T2_expand_forecast_position.csv", forecast),
        "bedict_v2": ("T2_expand_bedict_v2.csv", bedict),
        "deepbe": ("T2_expand_deepbe.csv", deepbe),
        "crispron_be": ("T2_expand_crispron_be.csv", crispron),
        "be_endo": ("T2_expand_be_endo.csv", be_endo),
        "yachie_base_editing_prediction": ("T2_expand_yachie_targets.csv", yachie),
    }
    manifest: dict[str, object] = {"matrix_dir": str(args.matrix_dir), "output_dir": str(args.output_dir), "panels": {}}
    union_parts = []
    for model_id, (filename, df) in panels.items():
        path = args.output_dir / filename
        write_panel(df, path)
        union_parts.append(df)
        manifest["panels"][model_id] = {"path": str(path), "rows": len(df)}

    union = pd.concat(union_parts, ignore_index=True, sort=False)
    union = union.drop_duplicates(subset=["target_profile_id"]).reset_index(drop=True)
    union_path = args.output_dir / "T2_expand_targets_union.csv"
    write_panel(union, union_path)
    manifest["union_panel"] = {"path": str(union_path), "rows": len(union)}

    manifest_path = args.output_dir / "T2_expand_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
