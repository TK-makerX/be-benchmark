#!/usr/bin/env python
"""Build T7 from raw repeated-design/editor groups.

This is the corrected T7 entry point. It defines recommendation ground truth
from original matrices first, then attaches existing model predictions only as
downstream scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
UNIFIED = ROOT / "data" / "unified_matrix_v5"
T7_RAW_VIEW = UNIFIED / "task_views" / "T7_editor_recommendation_view_v5.csv"
T1_RAW_VIEW = UNIFIED / "task_views" / "T1_total_efficiency_view_v5.csv"
T2_MATRIX = ROOT / "data" / "processed" / "T2_bystander_matrix"
T1_PREDICTIONS = (
    ROOT
    / "formal_results"
    / "T1_v6_common_core_abe_5model_full30000"
    / "predictions"
    / "combined"
    / "T1_V6_FIGURE_EVALUATION_TABLE.csv"
)
T2_RUNS = ROOT / "runs" / "T2_expand_v0"
BEDICT_V2 = PROJECT_ROOT / "BEDICT-V2"
OUT_DEFAULT = ROOT / "data" / "processed" / "T7_editor_design_recommendation_raw_v1"
REPORT_DEFAULT = ROOT / "reports" / "T7_RECOMMENDATION_RAW_AUDIT_V1.md"

EDITOR_GROUP_COLS = [
    "source_family",
    "design_id",
    "cell_context",
    "editor_class",
    "cas_variant",
    "pam_scope",
    "delivery",
    "timepoint",
]

DESIGN_GROUP_COLS = [
    "source_family",
    "editor_context_id",
    "editor",
    "cell_context",
    "editor_class",
    "cas_variant",
    "pam_scope",
    "delivery",
    "timepoint",
]

T2_GROUP_COLS = ["source_family", "source_row_key", "cell_context", "editor_class"]

RAW_COLS = [
    "observation_id",
    "design_id",
    "editor_context_id",
    "source_family",
    "source_dataset",
    "source_file",
    "source_row_key",
    "source_associated_models",
    "measured_total_efficiency",
    "measurement_scale",
    "qc_level",
    "sequence_50nt",
    "protospacer_20nt",
    "pam",
    "target_base",
    "target_position",
    "intended_edit",
    "design_hash",
    "editor",
    "editor_class",
    "editor_family",
    "cas_variant",
    "pam_scope",
    "cell_context",
    "delivery",
    "timepoint",
    "replicate_strategy",
    "editor_context_hash",
]


def numeric(series: pd.Series | str) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bool_series(series: pd.Series | str) -> pd.Series:
    if isinstance(series, str):
        return pd.Series(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def stable_id(prefix: str, parts: Iterable[object]) -> str:
    text = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def join_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].fillna("").astype(str).agg("|".join, axis=1)


def fmt(value: object) -> str:
    try:
        out = float(value)
    except Exception:
        return ""
    if math.isnan(out) or math.isinf(out):
        return ""
    return f"{out:.12g}"


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}g}")
        else:
            out[col] = out[col].fillna("").astype(str)
    lines = ["| " + " | ".join(out.columns) + " |", "| " + " | ".join(["---"] * len(out.columns)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in out.columns) + " |")
    return "\n".join(lines)


def read_raw_efficiency(path: Path) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in RAW_COLS if c in available]
    df = pd.read_csv(path, dtype=str, low_memory=False, usecols=usecols).fillna("")
    df["measured_efficiency"] = numeric(df["measured_total_efficiency"])
    return df


def add_truth_labels(candidate: pd.DataFrame, high_fraction: float, top_k: int) -> pd.DataFrame:
    candidate = candidate.copy()
    group = candidate.groupby("recommendation_group_id", dropna=False)["measured_efficiency"]
    sizes = candidate.groupby("recommendation_group_id", dropna=False)["observation_id"].transform("size")
    candidate["observed_rank_in_group"] = group.rank(method="min", ascending=False)
    candidate["high_label_cutoff_rank"] = np.maximum(1, np.ceil(sizes.astype(float) * high_fraction).astype(int))
    candidate["high_efficiency_label"] = candidate["observed_rank_in_group"].le(candidate["high_label_cutoff_rank"])
    candidate["recommendation_top_k"] = np.minimum(top_k, sizes.astype(int))
    candidate["oracle_efficiency_in_group"] = group.transform("max")
    candidate["candidate_regret_from_oracle"] = candidate["oracle_efficiency_in_group"] - candidate["measured_efficiency"]
    return candidate.reset_index(drop=True)


def build_efficiency_scenario(
    df: pd.DataFrame,
    *,
    scenario: str,
    source_view: str,
    group_cols: list[str],
    prefix: str,
    min_candidates: int,
    min_distinct_candidate_col: str,
    min_distinct_candidates: int,
    min_measured_unique: int,
    high_fraction: float,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["_group_key"] = join_key(data, group_cols)
    group_ids = {key: stable_id(prefix, [scenario, key]) for key in sorted(data["_group_key"].unique())}
    data["recommendation_group_id"] = data["_group_key"].map(group_ids)

    groups = (
        data.groupby(["recommendation_group_id", "_group_key"], dropna=False)
        .agg(
            source_family=("source_family", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
            n_candidates=("observation_id", "nunique"),
            n_designs=("design_id", "nunique"),
            n_editor_contexts=("editor_context_id", "nunique"),
            n_editors=("editor", "nunique"),
            measured_count=("measured_efficiency", "count"),
            measured_unique=("measured_efficiency", "nunique"),
            min_measured_efficiency=("measured_efficiency", "min"),
            median_measured_efficiency=("measured_efficiency", "median"),
            max_measured_efficiency=("measured_efficiency", "max"),
            candidate_editors=("editor", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
            cell_context=("cell_context", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
            editor_class=("editor_class", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
        )
        .reset_index()
    )
    groups["scenario"] = scenario
    groups["source_truth_view"] = source_view
    groups["objective"] = "high_efficiency"
    groups["candidate_kind"] = "editor_context" if scenario.startswith("editor") else "design"
    groups["eligible_group"] = (
        groups["n_candidates"].ge(min_candidates)
        & groups[min_distinct_candidate_col].ge(min_distinct_candidates)
        & groups["measured_count"].ge(min_candidates)
        & groups["measured_unique"].ge(min_measured_unique)
    )
    eligible_ids = set(groups.loc[groups["eligible_group"], "recommendation_group_id"])
    candidate = data[data["recommendation_group_id"].isin(eligible_ids)].copy()
    candidate["scenario"] = scenario
    candidate["source_truth_view"] = source_view
    candidate["objective"] = "high_efficiency"
    candidate["candidate_kind"] = groups["candidate_kind"].iloc[0] if not groups.empty else ""
    candidate = add_truth_labels(candidate, high_fraction=high_fraction, top_k=top_k)
    keep = [
        "scenario",
        "source_truth_view",
        "objective",
        "candidate_kind",
        "recommendation_group_id",
        "observation_id",
        "design_id",
        "editor_context_id",
        "source_family",
        "source_dataset",
        "source_file",
        "source_row_key",
        "source_associated_models",
        "editor",
        "editor_class",
        "editor_family",
        "cas_variant",
        "pam_scope",
        "cell_context",
        "delivery",
        "timepoint",
        "replicate_strategy",
        "sequence_50nt",
        "protospacer_20nt",
        "pam",
        "target_base",
        "target_position",
        "intended_edit",
        "measurement_scale",
        "qc_level",
        "measured_efficiency",
        "observed_rank_in_group",
        "high_efficiency_label",
        "high_label_cutoff_rank",
        "recommendation_top_k",
        "oracle_efficiency_in_group",
        "candidate_regret_from_oracle",
    ]
    keep = [c for c in keep if c in candidate.columns]
    return candidate[keep].copy(), groups.drop(columns=["_group_key"], errors="ignore")


def build_t2_multiobjective(matrix_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.read_csv(matrix_dir / "target_profile_t2_draft.csv", dtype=str, low_memory=False).fillna("")
    target["measured_efficiency"] = numeric(target["measured_total_efficiency"])
    target["observed_product_purity"] = numeric(target["observed_max_product_frequency"])
    target["observed_bystander_burden"] = numeric(target["observed_any_bystander_frequency"])
    target["_group_key"] = join_key(target, T2_GROUP_COLS)
    group_ids = {key: stable_id("T7T2", ["t2_multiobjective", key]) for key in sorted(target["_group_key"].unique())}
    target["recommendation_group_id"] = target["_group_key"].map(group_ids)
    groups = (
        target.groupby(["recommendation_group_id", "_group_key"], dropna=False)
        .agg(
            source_family=("source_family", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
            n_candidates=("target_profile_id", "nunique"),
            n_editors=("editor", "nunique"),
            efficiency_count=("measured_efficiency", "count"),
            efficiency_unique=("measured_efficiency", "nunique"),
            purity_count=("observed_product_purity", "count"),
            purity_unique=("observed_product_purity", "nunique"),
            bystander_count=("observed_bystander_burden", "count"),
            bystander_unique=("observed_bystander_burden", "nunique"),
            candidate_editors=("editor", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
            cell_context=("cell_context", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
            editor_class=("editor_class", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
        )
        .reset_index()
    )
    base = groups["n_candidates"].ge(2) & groups["n_editors"].ge(2)
    groups["eligible_high_efficiency_group"] = base & groups["efficiency_count"].ge(2) & groups["efficiency_unique"].ge(2)
    groups["eligible_product_purity_group"] = base & groups["purity_count"].ge(2) & groups["purity_unique"].ge(2)
    groups["eligible_low_bystander_group"] = base & groups["bystander_count"].ge(2) & groups["bystander_unique"].ge(2)
    groups["eligible_multiobjective_group"] = (
        groups["eligible_high_efficiency_group"]
        & groups["eligible_product_purity_group"]
        & groups["eligible_low_bystander_group"]
    )
    eligible_ids = set(
        groups.loc[
            groups[
                [
                    "eligible_high_efficiency_group",
                    "eligible_product_purity_group",
                    "eligible_low_bystander_group",
                    "eligible_multiobjective_group",
                ]
            ].any(axis=1),
            "recommendation_group_id",
        ]
    )
    cand = target[target["recommendation_group_id"].isin(eligible_ids)].copy()
    cand["efficiency_rank_desc"] = cand.groupby("recommendation_group_id")["measured_efficiency"].rank(
        method="min", ascending=False
    )
    cand["product_purity_rank_desc"] = cand.groupby("recommendation_group_id")["observed_product_purity"].rank(
        method="min", ascending=False
    )
    cand["bystander_burden_rank_asc"] = cand.groupby("recommendation_group_id")["observed_bystander_burden"].rank(
        method="min", ascending=True
    )
    cand["pareto_efficient_eff_purity_bystander"] = False
    for group_id, sub in cand.groupby("recommendation_group_id", sort=False):
        values = sub[["measured_efficiency", "observed_product_purity", "observed_bystander_burden"]].to_numpy(float)
        ok = ~np.isnan(values).any(axis=1)
        pareto = np.zeros(len(sub), dtype=bool)
        for i, vi in enumerate(values):
            if not ok[i]:
                continue
            dominated = False
            for j, vj in enumerate(values):
                if i == j or not ok[j]:
                    continue
                no_worse = (vj[0] >= vi[0]) and (vj[1] >= vi[1]) and (vj[2] <= vi[2])
                better = (vj[0] > vi[0]) or (vj[1] > vi[1]) or (vj[2] < vi[2])
                if no_worse and better:
                    dominated = True
                    break
            pareto[i] = not dominated
        cand.loc[sub.index, "pareto_efficient_eff_purity_bystander"] = pareto
    keep = [
        "recommendation_group_id",
        "target_profile_id",
        "source_family",
        "source_dataset",
        "source_file",
        "source_row_key",
        "editor",
        "editor_class",
        "editor_family",
        "cell_context",
        "sequence_50nt",
        "protospacer_20nt",
        "pam",
        "target_position",
        "intended_edit",
        "target_base",
        "measured_efficiency",
        "observed_product_purity",
        "observed_bystander_burden",
        "observed_desired_product_frequency",
        "outcome_row_count",
        "editable_position_count",
        "efficiency_rank_desc",
        "product_purity_rank_desc",
        "bystander_burden_rank_asc",
        "pareto_efficient_eff_purity_bystander",
    ]
    keep = [c for c in keep if c in cand.columns]
    return cand[keep].copy(), groups.drop(columns=["_group_key"], errors="ignore")


def dcg(values: Iterable[float]) -> float:
    total = 0.0
    for i, value in enumerate(values, start=1):
        if pd.isna(value):
            value = 0.0
        total += float(value) / math.log2(i + 1)
    return total


def score_group(sub: pd.DataFrame, top_k: int) -> dict[str, object]:
    ranked = sub.sort_values(["predicted_recommendation_score", "observation_id"], ascending=[False, True]).copy()
    k = min(top_k, len(ranked))
    top = ranked.head(k)
    best_eff = float(ranked["measured_efficiency"].max())
    true_best = ranked["measured_efficiency"].eq(best_eff).tolist()
    true_best_ranks = [i + 1 for i, flag in enumerate(true_best) if flag]
    high = ranked["high_efficiency_label"].astype(bool)
    top_high = int(top["high_efficiency_label"].astype(bool).sum())
    high_count = int(high.sum())
    baseline = high_count / len(ranked) if len(ranked) else np.nan
    precision = top_high / k if k else np.nan
    ideal = ranked.sort_values(["measured_efficiency", "observation_id"], ascending=[False, True]).head(k)
    ideal_dcg = dcg(ideal["measured_efficiency"])
    ndcg = dcg(top["measured_efficiency"]) / ideal_dcg if ideal_dcg > 0 else np.nan
    return {
        "recommendation_group_id": ranked.iloc[0]["recommendation_group_id"],
        "n_candidates": len(ranked),
        "top_k_used": k,
        "top1_best_hit": bool(true_best[0]),
        "topk_best_hit": bool(top["measured_efficiency"].eq(best_eff).any()),
        "mrr": 1.0 / min(true_best_ranks) if true_best_ranks else np.nan,
        "ndcg_at_k": ndcg,
        "precision_at_k": precision,
        "recall_at_k": top_high / high_count if high_count else np.nan,
        "enrichment_at_k": precision / baseline if baseline else np.nan,
        "regret_top1": best_eff - float(ranked.iloc[0]["measured_efficiency"]),
        "predicted_top_efficiency": float(ranked.iloc[0]["measured_efficiency"]),
        "oracle_efficiency": best_eff,
    }


def load_t1_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    df["valid_recommendation_score"] = bool_series(df.get("valid", df.get("is_valid_t1_prediction", "")))
    df["predicted_recommendation_score"] = numeric(
        df.get("predicted_for_correlation", df.get("prediction_value_for_correlation", ""))
    )
    if "display_name" not in df.columns:
        df["display_name"] = df["model_id"]
    return df[
        [
            "observation_id",
            "model_id",
            "display_name",
            "valid_recommendation_score",
            "predicted_recommendation_score",
            "prediction_value_kind",
        ]
    ].drop_duplicates(["observation_id", "model_id"])


def score_existing_t1_predictions(candidates: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = (
        candidates.groupby(["scenario", "recommendation_group_id"], dropna=False)
        .agg(raw_candidate_count=("observation_id", "nunique"), source_family=("source_family", "first"))
        .reset_index()
    )
    top_k_by_scenario = (
        candidates.groupby("scenario")["recommendation_top_k"].max().astype(int).to_dict()
        if "recommendation_top_k" in candidates.columns
        else {}
    )
    models = predictions[["model_id", "display_name"]].drop_duplicates().sort_values("model_id")
    coverage_rows = []
    metric_rows = []

    keep_cols = [
        "scenario",
        "recommendation_group_id",
        "observation_id",
        "measured_efficiency",
        "high_efficiency_label",
        "recommendation_top_k",
    ]
    intersect = candidates[keep_cols].merge(predictions, on="observation_id", how="inner")
    intersect["_valid_score"] = intersect["predicted_recommendation_score"].where(
        intersect["valid_recommendation_score"], np.nan
    )
    detail = (
        intersect.groupby(["scenario", "model_id", "display_name", "recommendation_group_id"], dropna=False)
        .agg(
            predicted_candidate_count=("observation_id", "nunique"),
            valid_prediction_count=("valid_recommendation_score", "sum"),
            unique_prediction_scores=("_valid_score", "nunique"),
        )
        .reset_index()
        .merge(info[["scenario", "recommendation_group_id", "raw_candidate_count"]], on=["scenario", "recommendation_group_id"], how="left")
    )
    detail["complete_raw_candidate_set"] = detail["predicted_candidate_count"].eq(detail["raw_candidate_count"]) & detail[
        "valid_prediction_count"
    ].eq(detail["raw_candidate_count"])
    detail["scoreable_rank_group"] = detail["complete_raw_candidate_set"] & detail["unique_prediction_scores"].ge(2)

    scoreable_keys = detail.loc[
        detail["scoreable_rank_group"], ["scenario", "model_id", "recommendation_group_id"]
    ].copy()
    scored_lookup = pd.DataFrame()
    if not scoreable_keys.empty:
        scored_input = intersect.merge(scoreable_keys, on=["scenario", "model_id", "recommendation_group_id"], how="inner")
        scored_input = scored_input[scored_input["valid_recommendation_score"] & scored_input["predicted_recommendation_score"].notna()]
        scored_rows = []
        for (scenario, model_id, group_id), sub in scored_input.groupby(
            ["scenario", "model_id", "recommendation_group_id"], sort=False
        ):
            top_k = int(top_k_by_scenario.get(scenario, 1))
            row = score_group(sub, top_k=top_k)
            label = str(sub["display_name"].iloc[0])
            row.update({"scenario": scenario, "model_id": model_id, "display_name": label})
            scored_rows.append(row)
        scored_lookup = pd.DataFrame(scored_rows)
        metric_rows = scored_rows

    for scenario in sorted(info["scenario"].unique()):
        scenario_info = info[info["scenario"].eq(scenario)]
        eligible_groups = int(scenario_info["recommendation_group_id"].nunique())
        eligible_candidates = int(scenario_info["raw_candidate_count"].sum())
        for _, model in models.iterrows():
            mid = model["model_id"]
            label = model["display_name"]
            detail_sub = detail[detail["scenario"].eq(scenario) & detail["model_id"].eq(mid)].copy()
            score_df = (
                scored_lookup[scored_lookup["scenario"].eq(scenario) & scored_lookup["model_id"].eq(mid)].copy()
                if not scored_lookup.empty
                else pd.DataFrame()
            )
            if not detail_sub.empty:
                complete_groups = int(detail_sub["complete_raw_candidate_set"].sum())
                scoreable_groups = int(detail_sub["scoreable_rank_group"].sum())
                predicted_groups = int(detail_sub["recommendation_group_id"].nunique())
                predicted_candidates = int(detail_sub["predicted_candidate_count"].sum())
            else:
                complete_groups = scoreable_groups = predicted_groups = predicted_candidates = 0
            row = {
                "scenario": scenario,
                "model_id": mid,
                "display_name": label,
                "eligible_raw_groups": eligible_groups,
                "eligible_raw_candidates": eligible_candidates,
                "groups_with_any_prediction": predicted_groups,
                "predicted_candidates_in_raw_groups": predicted_candidates,
                "complete_raw_candidate_groups": complete_groups,
                "scoreable_rank_groups": scoreable_groups,
            }
            if score_df.empty:
                row.update(
                    {
                        "scored_groups": 0,
                        "mean_top1_best_hit": np.nan,
                        "mean_topk_best_hit": np.nan,
                        "mean_mrr": np.nan,
                        "mean_ndcg_at_k": np.nan,
                        "mean_precision_at_k": np.nan,
                        "mean_enrichment_at_k": np.nan,
                        "mean_regret_top1": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        "scored_groups": int(score_df["recommendation_group_id"].nunique()),
                        "mean_top1_best_hit": float(score_df["top1_best_hit"].mean()),
                        "mean_topk_best_hit": float(score_df["topk_best_hit"].mean()),
                        "mean_mrr": float(score_df["mrr"].mean()),
                        "mean_ndcg_at_k": float(score_df["ndcg_at_k"].mean()),
                        "mean_precision_at_k": float(score_df["precision_at_k"].mean()),
                        "mean_enrichment_at_k": float(score_df["enrichment_at_k"].mean()),
                        "mean_regret_top1": float(score_df["regret_top1"].mean()),
                    }
                )
            coverage_rows.append(row)
    return pd.DataFrame(coverage_rows), pd.DataFrame(metric_rows), detail


def read_t2_outcome_predictions(t2_runs: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(t2_runs.glob("*/predictions_t2_*outcomes*.csv")):
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    pred = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    pred["predicted_outcome_frequency_num"] = numeric(pred.get("predicted_outcome_frequency", ""))
    return (
        pred.groupby(["model_id", "target_profile_id"], dropna=False)
        .agg(
            predicted_product_rows=("target_profile_id", "size"),
            predicted_max_product_frequency=("predicted_outcome_frequency_num", "max"),
            unique_predicted_product_frequencies=("predicted_outcome_frequency_num", "nunique"),
            prediction_scale=("prediction_scale", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
            source_relation_to_model=("source_relation_to_model", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
        )
        .reset_index()
    )


def t2_prediction_coverage(t2_candidates: pd.DataFrame, t2_groups: pd.DataFrame, t2_runs: Path) -> pd.DataFrame:
    pred = read_t2_outcome_predictions(t2_runs)
    if pred.empty or t2_candidates.empty:
        return pd.DataFrame()
    group_counts = t2_candidates.groupby("recommendation_group_id")["target_profile_id"].nunique().rename("raw_n")
    models = pred[["model_id"]].drop_duplicates().sort_values("model_id")
    grid = t2_candidates[["recommendation_group_id", "target_profile_id"]].merge(models, how="cross")
    joined = grid.merge(pred, on=["model_id", "target_profile_id"], how="left")
    rows = []
    for (model_id, group_id), sub in joined.groupby(["model_id", "recommendation_group_id"], sort=False):
        raw_n = int(group_counts.loc[group_id])
        pred_sub = sub[sub["predicted_max_product_frequency"].notna()]
        unique_scores = int(pred_sub["predicted_max_product_frequency"].nunique(dropna=True))
        rows.append(
            {
                "model_id": model_id,
                "recommendation_group_id": group_id,
                "raw_candidate_count": raw_n,
                "predicted_candidate_count": int(pred_sub["target_profile_id"].nunique()),
                "complete_raw_candidate_set": bool(pred_sub["target_profile_id"].nunique() == raw_n),
                "unique_predicted_product_purity_scores": unique_scores,
                "scoreable_product_purity_group": bool(pred_sub["target_profile_id"].nunique() == raw_n and unique_scores >= 2),
            }
        )
    return pd.DataFrame(rows)


def bedict_v2_source_inventory(path: Path) -> pd.DataFrame:
    rows = []
    supplement_dir = path / "resources" / "paper_supplements"
    for file in sorted(supplement_dir.glob("*")):
        row = {"file": str(file), "name": file.name, "bytes": file.stat().st_size, "suffix": file.suffix.lower()}
        if file.suffix.lower() == ".xlsx":
            try:
                with zipfile.ZipFile(file) as zf:
                    workbook = zf.read("xl/workbook.xml").decode("utf-8", errors="ignore")
                # Lightweight sheet-name extraction; enough for source inventory.
                names = []
                for part in workbook.split("<sheet "):
                    if 'name="' in part:
                        names.append(part.split('name="', 1)[1].split('"', 1)[0])
                row["sheet_count"] = len(names)
                row["sheet_names"] = ";".join(names[:80])
                if any("SpRY" in name or "SpCas9" in name or "ABE" in name for name in names):
                    row["t7_relevance"] = "possible_raw_editor_design_measurements"
                else:
                    row["t7_relevance"] = "metadata_or_feature_tables"
            except Exception as exc:
                row["sheet_count"] = ""
                row["sheet_names"] = ""
                row["t7_relevance"] = f"xlsx_inventory_failed:{type(exc).__name__}"
        else:
            row["sheet_count"] = ""
            row["sheet_names"] = ""
            row["t7_relevance"] = "non_xlsx_supplement"
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_raw_groups(groups: pd.DataFrame) -> pd.DataFrame:
    eligible = groups[groups["eligible_group"]].copy()
    rows = []
    for (scenario, source), sub in eligible.groupby(["scenario", "source_family"], dropna=False):
        rows.append(
            {
                "scenario": scenario,
                "source_family": source,
                "eligible_raw_groups": int(sub["recommendation_group_id"].nunique()),
                "eligible_raw_candidates": int(sub["n_candidates"].sum()),
                "median_candidates_per_group": float(sub["n_candidates"].median()),
                "max_candidates_per_group": int(sub["n_candidates"].max()),
                "unique_editors_in_groups": int(
                    len(set(";".join(sub["candidate_editors"].fillna("").astype(str)).split(";")) - {""})
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_t2_groups(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flag, objective in [
        ("eligible_high_efficiency_group", "high_efficiency"),
        ("eligible_product_purity_group", "high_product_purity"),
        ("eligible_low_bystander_group", "low_bystander_burden"),
        ("eligible_multiobjective_group", "efficiency_purity_low_bystander_pareto"),
    ]:
        sub = groups[groups[flag]].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "objective": objective,
                "eligible_raw_groups": int(sub["recommendation_group_id"].nunique()),
                "eligible_raw_candidates": int(sub["n_candidates"].sum()),
                "median_candidates_per_group": float(sub["n_candidates"].median()),
                "max_candidates_per_group": int(sub["n_candidates"].max()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    report: Path,
    raw_summary: pd.DataFrame,
    t2_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    metrics: pd.DataFrame,
    bedict_inventory: pd.DataFrame,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    metric_show = coverage[
        [
            "scenario",
            "model_id",
            "eligible_raw_groups",
            "groups_with_any_prediction",
            "complete_raw_candidate_groups",
            "scoreable_rank_groups",
            "scored_groups",
            "mean_top1_best_hit",
            "mean_ndcg_at_k",
            "mean_regret_top1",
        ]
    ].copy()
    bedict_show = bedict_inventory[["name", "bytes", "sheet_count", "t7_relevance"]].copy()
    parts = [
        "# T7 raw recommendation-eligibility audit v1",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Correction",
        "",
        "This run defines T7 ground truth from original repeated-design/editor groups before attaching model predictions. It does not start from T1/T2/T3 leaderboards or model rankings.",
        "",
        "## Raw high-efficiency recommendation groups",
        "",
        md_table(raw_summary, digits=4),
        "",
        "## T2 product-purity and bystander recommendation audit",
        "",
        md_table(t2_summary, digits=4),
        "",
        "## Existing T1 prediction coverage after raw groups are fixed",
        "",
        "Complete means all candidates in the raw recommendation group have valid predictions for the model. Scoreable additionally requires non-tied prediction scores.",
        "",
        md_table(metric_show, digits=4),
        "",
        "## BEDICT-V2 supplementary source inventory",
        "",
        "BEDICT2 measured rows are already represented in `unified_matrix_v5`; this inventory records extra BEDICT-V2 supplement files that may support future source-specific expansion.",
        "",
        md_table(bedict_show, digits=4),
        "",
        "## Guardrails",
        "",
        "- Raw recommendation groups define ground truth before model predictions are merged.",
        "- Design-selection scoring is not reported from the old 30k T1 subset as complete raw-group performance because no model covers full raw design groups in this run.",
        "- T2 product-purity/bystander objectives remain audit-first until prediction coverage is sufficient for formal multi-objective ranking.",
        "",
    ]
    report.write_text("\n".join(parts), encoding="utf-8")


def write_schema(out_dir: Path, report: Path) -> None:
    schema = {
        "task": "T7",
        "version": "raw_v1",
        "source_of_truth_rule": "ground truth starts from original repeated-design/editor groups",
        "inputs": {
            "editor_selection_raw": str(T7_RAW_VIEW),
            "design_selection_raw": str(T1_RAW_VIEW),
            "t2_multiobjective_raw": str(T2_MATRIX),
            "bedict_v2": str(BEDICT_V2),
            "model_predictions_merged_after_audit": str(T1_PREDICTIONS),
        },
        "outputs": {
            "raw_ground_truth_candidates": str(out_dir / "T7_raw_ground_truth_candidates.csv"),
            "raw_group_eligibility": str(out_dir / "T7_raw_group_eligibility.csv"),
            "raw_group_summary": str(out_dir / "T7_raw_group_summary.csv"),
            "t2_multiobjective_candidates": str(out_dir / "T7_t2_multiobjective_candidates.csv"),
            "t2_multiobjective_group_eligibility": str(out_dir / "T7_t2_multiobjective_group_eligibility.csv"),
            "model_prediction_coverage_summary": str(out_dir / "T7_model_prediction_coverage_summary.csv"),
            "model_prediction_coverage_detail": str(out_dir / "T7_model_prediction_coverage_detail.csv"),
            "recommendation_metrics_complete_raw_groups": str(out_dir / "T7_recommendation_metrics_complete_raw_groups.csv"),
            "bedict_v2_source_inventory": str(out_dir / "T7_bedict_v2_source_inventory.csv"),
            "report": str(report),
        },
    }
    (out_dir / "T7_SCHEMA_RAW_V1.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw T7 editor-selection view...", flush=True)
    editor_raw = read_raw_efficiency(T7_RAW_VIEW)
    print(f"Loaded editor-selection raw rows: {len(editor_raw)}", flush=True)
    editor_candidates, editor_groups = build_efficiency_scenario(
        editor_raw,
        scenario="editor_selection_high_efficiency",
        source_view=str(T7_RAW_VIEW),
        group_cols=EDITOR_GROUP_COLS,
        prefix="T7ERAW",
        min_candidates=2,
        min_distinct_candidate_col="n_editor_contexts",
        min_distinct_candidates=2,
        min_measured_unique=2,
        high_fraction=0.10,
        top_k=1,
    )
    del editor_raw
    print(f"Built editor-selection raw candidates: {len(editor_candidates)}", flush=True)

    print("Loading raw T1 design-selection view...", flush=True)
    design_raw = read_raw_efficiency(T1_RAW_VIEW)
    print(f"Loaded design-selection raw rows: {len(design_raw)}", flush=True)
    design_candidates, design_groups = build_efficiency_scenario(
        design_raw,
        scenario="design_selection_high_efficiency",
        source_view=str(T1_RAW_VIEW),
        group_cols=DESIGN_GROUP_COLS,
        prefix="T7DRAW",
        min_candidates=20,
        min_distinct_candidate_col="n_designs",
        min_distinct_candidates=20,
        min_measured_unique=5,
        high_fraction=0.10,
        top_k=10,
    )
    del design_raw
    print(f"Built design-selection raw candidates: {len(design_candidates)}", flush=True)

    raw_candidates = pd.concat([editor_candidates, design_candidates], ignore_index=True, sort=False)
    raw_groups = pd.concat([editor_groups, design_groups], ignore_index=True, sort=False)
    raw_summary = aggregate_raw_groups(raw_groups)
    raw_groups.to_csv(args.output_dir / "T7_raw_group_eligibility.csv", index=False)
    raw_summary.to_csv(args.output_dir / "T7_raw_group_summary.csv", index=False)
    raw_candidates.to_csv(args.output_dir / "T7_raw_ground_truth_candidates.csv", index=False)
    print("Wrote raw group eligibility summaries and ground-truth candidates.", flush=True)

    print("Building T2 multi-objective raw audit...", flush=True)
    t2_candidates, t2_groups = build_t2_multiobjective(T2_MATRIX)
    t2_summary = aggregate_t2_groups(t2_groups)
    t2_groups.to_csv(args.output_dir / "T7_t2_multiobjective_group_eligibility.csv", index=False)
    t2_summary.to_csv(args.output_dir / "T7_t2_multiobjective_group_summary.csv", index=False)
    print("Wrote T2 multi-objective summaries.", flush=True)

    print("Merging existing T1 predictions after raw groups are fixed...", flush=True)
    predictions = load_t1_predictions(T1_PREDICTIONS)
    coverage, metrics, detail = score_existing_t1_predictions(raw_candidates, predictions)
    coverage.to_csv(args.output_dir / "T7_model_prediction_coverage_summary.csv", index=False)
    metrics.to_csv(args.output_dir / "T7_recommendation_metrics_complete_raw_groups.csv", index=False)
    detail.to_csv(args.output_dir / "T7_model_prediction_coverage_detail.csv", index=False)
    print("Wrote model coverage and complete raw-group metrics.", flush=True)

    print("Auditing existing T2 outcome-prediction coverage...", flush=True)
    t2_cov = t2_prediction_coverage(t2_candidates, t2_groups, T2_RUNS)
    t2_cov.to_csv(args.output_dir / "T7_t2_prediction_coverage_after_raw_audit.csv", index=False)
    print("Wrote T2 prediction coverage.", flush=True)

    print("Inventorying BEDICT-V2 supplementary files...", flush=True)
    bedict_inventory = bedict_v2_source_inventory(BEDICT_V2)
    bedict_inventory.to_csv(args.output_dir / "T7_bedict_v2_source_inventory.csv", index=False)

    print("Writing T2 ground-truth candidate table...", flush=True)
    t2_candidates.to_csv(args.output_dir / "T7_t2_multiobjective_candidates.csv", index=False)
    write_schema(args.output_dir, args.report)
    write_report(args.report, raw_summary, t2_summary, coverage, metrics, bedict_inventory)

    print(f"Wrote corrected T7 raw audit outputs to {args.output_dir}")
    print(f"Wrote corrected T7 raw audit report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
