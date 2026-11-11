#!/usr/bin/env python
"""Prepare expanded T3 v1 model panels from the existing local data pool."""

from __future__ import annotations

import os
import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
T2_MATRIX = ROOT / "data" / "processed" / "T2_bystander_matrix"
T3_MATRIX = ROOT / "data" / "processed" / "T3_position_profile_matrix"
OUT = T3_MATRIX / "panels_v1"
BEHIVE_MODELS = Path(os.environ.get("BE_MODELS_DIR", os.environ.get("BE_REPLAY_PROJECT", "."))) / "BE-Hive-bystander/repo/models.csv"

EVENT_RE = re.compile(r"^([ACGT])(-?\d+)>([ACGT])$")
DNA = set("ACGT")

PANEL_FILES = {
    "behive_bystander": "T3_v1_behive_bystander.csv",
    "forecast_be": "T3_v1_forecast_be.csv",
    "bedict_v2": "T3_v1_bedict_v2.csv",
    "deepbe": "T3_v1_deepbe.csv",
    "crispron_be": "T3_v1_crispron_be.csv",
    "be_endo": "T3_v1_be_endo.csv",
    "yachie_base_editing_prediction": "T3_v1_yachie.csv",
    "epcnnbe": "T3_v1_epcnnbe.csv",
    "deepbaseeditor": "T3_v1_deepbaseeditor.csv",
}


def dna(seq: object) -> bool:
    text = str(seq or "").upper()
    return bool(text) and set(text) <= DNA


def find_protospacer_start(row: pd.Series) -> int:
    seq = str(row.get("sequence_50nt", "")).upper()
    protospacer = str(row.get("protospacer_20nt", "")).upper()
    if len(seq) != 50 or len(protospacer) != 20 or not dna(seq) or not dna(protospacer):
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
        start = find_protospacer_start(row)
        starts.append(start)
        seq30.append(seq[start - 4 : start + 26] if start >= 4 and start + 26 <= len(seq) else "")
        seq40.append(seq[start - 10 : start + 30] if start >= 10 and start + 30 <= len(seq) else "")
        pam = str(row.get("pam", "")).upper()
        if start >= 0 and start + 24 <= len(seq):
            pam4.append(seq[start + 20 : start + 24])
        elif len(pam) == 3:
            pam4.append(pam + "N")
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
    if not events or events == "no_tracked_substrate_change":
        return False
    for part in str(events).split(";"):
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


def behive_supported_pairs() -> set[tuple[str, str]]:
    models = pd.read_csv(BEHIVE_MODELS, dtype=str).fillna("")
    pairs: set[tuple[str, str]] = set()
    for _, row in models.iterrows():
        pairs.add((row["Public base editor"], row["Celltype"]))
        pairs.add((row["Internal base editor"], row["Celltype"]))
    return pairs


def balanced_sample(df: pd.DataFrame, group_cols: list[str], max_targets: int) -> pd.DataFrame:
    if df.empty or len(df) <= max_targets:
        return df.copy().reset_index(drop=True)
    groups = []
    for key, sub in df.groupby(group_cols, dropna=False, sort=True):
        sub = sub.copy()
        sub["_outcomes_sort"] = pd.to_numeric(sub["outcome_row_count"], errors="coerce")
        sub["_eff_sort"] = pd.to_numeric(sub["measured_total_efficiency"], errors="coerce")
        sub = sub.sort_values(["_outcomes_sort", "_eff_sort", "target_profile_id"], ascending=[True, False, True])
        groups.append((key, sub.reset_index(drop=True)))
    selected = []
    seen_targets: set[str] = set()
    seen_seq30: set[str] = set()
    cursor = 0
    while len(selected) < max_targets:
        added = False
        for _, sub in groups:
            if cursor >= len(sub):
                continue
            row = sub.iloc[cursor].to_dict()
            target_id = row["target_profile_id"]
            seq30 = row.get("sequence_context_30nt", "")
            if target_id not in seen_targets and (not seq30 or seq30 not in seen_seq30):
                selected.append(row)
                seen_targets.add(target_id)
                if seq30:
                    seen_seq30.add(seq30)
                added = True
                if len(selected) >= max_targets:
                    break
        cursor += 1
        if cursor > max(len(sub) for _, sub in groups) and not added:
            break
    if len(selected) < max_targets:
        for _, sub in groups:
            for _, row_series in sub.iterrows():
                row = row_series.to_dict()
                if row["target_profile_id"] in seen_targets:
                    continue
                selected.append(row)
                seen_targets.add(row["target_profile_id"])
                if len(selected) >= max_targets:
                    break
            if len(selected) >= max_targets:
                break
    out = pd.DataFrame(selected).drop(columns=["_outcomes_sort", "_eff_sort"], errors="ignore")
    return out.reset_index(drop=True)


def write_panel(df: pd.DataFrame, path: Path, *, relation: str, model_id: str) -> None:
    df = df.copy()
    df["source_relation_to_model_for_t2"] = relation
    df["source_relation_to_model_for_t3"] = relation
    df["t3_panel_role"] = "expanded_v1"
    df["t3_candidate_model"] = model_id
    df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=T2_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-main-targets", type=int, default=300)
    parser.add_argument("--max-exploratory-targets", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(args.matrix_dir / "T2_observed_target_profiles.csv", dtype=str, low_memory=False).fillna("")
    outcome = pd.read_csv(args.matrix_dir / "T2_observed_outcome_profiles.csv", dtype=str, low_memory=False).fillna("")
    position = pd.read_csv(args.matrix_dir / "T2_observed_position_profiles.csv", dtype=str, low_memory=False).fillna("")
    target = add_windows(target)
    target = target.merge(target_event_flags(outcome), on="target_profile_id", how="left")
    target["outcome_row_count_num"] = pd.to_numeric(target["outcome_row_count"], errors="coerce")

    qc = target["qc_level"].eq("B")
    complete = target["outcome_row_count_num"].le(50)
    prot20 = target["protospacer_20nt"].map(lambda s: len(str(s)) == 20 and dna(s))
    concrete30 = target["sequence_context_30nt"].map(lambda s: len(str(s)) == 30 and dna(s))
    concrete40 = target["sequence_context_40nt"].map(lambda s: len(str(s)) == 40 and dna(s))
    editable_epcnn = target.apply(
        lambda row: ("A" in row["sequence_context_30nt"][6:14]) if row["editor_class"] == "ABE" else ("C" in row["sequence_context_30nt"][6:14]),
        axis=1,
    )

    supported_behive = behive_supported_pairs()
    panels: dict[str, tuple[pd.DataFrame, str, int, list[str]]] = {}

    behive = target[
        qc
        & target["behive_bystander_input_eligible"].eq("True")
        & complete
        & target["cell_context"].isin(["mES", "HEK293T"])
    ].copy()
    behive = behive[behive.apply(lambda row: (row["editor"], row["cell_context"]) in supported_behive, axis=1)]
    panels["behive_bystander"] = (behive, "source_associated", args.max_main_targets, ["cell_context", "editor_class", "editor"])

    forecast = target[qc & target["forecast_position_input_eligible"].eq("True") & target["editor_class"].isin(["ABE", "CBE"]) & prot20].copy()
    panels["forecast_be"] = (forecast, "external_candidate", args.max_main_targets, ["cell_context", "editor_class", "source_dataset"])

    bedict = target[
        qc
        & complete
        & prot20
        & target["pam4"].map(lambda s: len(str(s)) == 4 and dna(s))
        & target["editor_class"].eq("ABE")
        & target["observed_all_1_20_ag"].eq(True)
    ].copy()
    bedict["t2_model_editor"] = "ABEmax-SpCas9"
    panels["bedict_v2"] = (bedict, "proxy_external_candidate", args.max_main_targets, ["cell_context", "editor"])

    deepbe = target[
        qc
        & complete
        & concrete30
        & target["editor_class"].eq("ABE")
        & target["observed_all_3_10_ag"].eq(True)
        & target["sequence_context_30nt"].str.slice(6, 14).str.contains("A", regex=False)
    ].copy()
    panels["deepbe"] = (deepbe, "proxy_external_candidate", args.max_main_targets, ["cell_context", "editor"])

    crispron = target[
        qc
        & complete
        & concrete30
        & target["sequence_context_30nt"].str.slice(25, 27).eq("GG")
        & target["editor_class"].isin(["ABE", "CBE"])
        & target["observed_all_minus3_20_any"].eq(True)
    ].copy()
    panels["crispron_be"] = (crispron, "external_candidate", args.max_main_targets, ["cell_context", "editor_class", "editor"])

    be_endo = target[
        qc
        & concrete40
        & target["sequence_context_40nt"].str.slice(30, 33).isin(["AGG", "CGG", "TGG", "GGG"])
        & target["editor_class"].isin(["ABE", "CBE"])
        & target["observed_all_1_20_any"].eq(True)
    ].copy()
    panels["be_endo"] = (be_endo, "external_candidate", args.max_main_targets, ["cell_context", "editor_class", "editor"])

    yachie = target[
        qc
        & complete
        & concrete30
        & target["editor_class"].isin(["ABE", "CBE"])
        & target["observed_all_minus3_20_any"].eq(True)
    ].copy()
    panels["yachie_base_editing_prediction"] = (yachie, "proxy_external_candidate", args.max_main_targets, ["cell_context", "editor_class", "editor"])

    epcnnbe = target[qc & concrete30 & target["editor_class"].isin(["ABE", "CBE"]) & editable_epcnn].copy()
    panels["epcnnbe"] = (epcnnbe, "proxy_external_candidate", args.max_exploratory_targets, ["cell_context", "editor_class", "editor"])
    panels["deepbaseeditor"] = (epcnnbe.copy(), "proxy_external_candidate", args.max_exploratory_targets, ["cell_context", "editor_class", "editor"])

    position_counts = position.groupby("target_profile_id").size()
    manifest: dict[str, object] = {"output_dir": str(args.output_dir), "panels": {}}
    panel_cov_rows = []
    audit = pd.read_csv(T3_MATRIX / "T3_model_inclusion_audit.csv", dtype=str, low_memory=False).fillna("")
    prior_panel_cov = pd.read_csv(T3_MATRIX / "T3_model_panel_coverage.csv", dtype=str, low_memory=False).fillna("")
    for model_id, (df, relation, max_targets, group_cols) in panels.items():
        sampled = balanced_sample(df, group_cols, max_targets)
        path = args.output_dir / PANEL_FILES[model_id]
        write_panel(sampled, path, relation=relation, model_id=model_id)
        position_rows = int(position_counts.reindex(sampled["target_profile_id"]).fillna(0).sum())
        manifest["panels"][model_id] = {
            "path": str(path),
            "eligible_targets": len(df),
            "selected_targets": len(sampled),
            "selected_position_rows": position_rows,
            "relation": relation,
        }
        panel_cov_rows.append(
            {
                "model_id": model_id,
                "input_type": prior_panel_cov.loc[prior_panel_cov["model_id"].eq(model_id), "input_type"].iloc[0] if model_id in set(prior_panel_cov["model_id"]) else "",
                "panel_path": str(path),
                "panel_targets": len(sampled),
                "panel_observed_position_rows": position_rows,
                "panel_editor_classes": ";".join(sorted(set(sampled["editor_class"].astype(str)))) if not sampled.empty else "",
                "panel_editors": ";".join(sorted(set(sampled["editor"].astype(str)))) if not sampled.empty else "",
                "panel_cell_contexts": ";".join(sorted(set(sampled["cell_context"].astype(str)))) if not sampled.empty else "",
            }
        )

    audit.to_csv(args.output_dir / "T3_v1_model_inclusion_seed.csv", index=False)
    pd.DataFrame(panel_cov_rows).to_csv(args.output_dir / "T3_v1_model_panel_coverage_seed.csv", index=False)
    (args.output_dir / "T3_v1_panel_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
