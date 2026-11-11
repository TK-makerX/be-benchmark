#!/usr/bin/env python
"""
Build the T3 position-profile audit.

T3 asks whether models predict which editable positions inside the editing
window are edited and at what rate.  This script promotes the T2 draft
position matrix into a T3 audit surface, but does not run a formal T3
benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
T2_MATRIX = ROOT / "data" / "processed" / "T2_bystander_matrix"
T2_PANELS = T2_MATRIX / "panels"
T2_RUNS = ROOT / "runs" / "T2_expand_v0"
T2_RESULTS = ROOT / "formal_results" / "T2_expand_v0"
OUT_DEFAULT = ROOT / "data" / "processed" / "T3_position_profile_matrix"
T3_PANELS = OUT_DEFAULT / "panels"
T3_EXPLORATORY_RUNS = ROOT / "runs" / "T3_exploratory_v0"
REPORT_DEFAULT = ROOT / "reports" / "T3_POSITION_PROFILE_AUDIT.md"

EXPECTED_MODELS = [
    "behive_bystander",
    "forecast_be",
    "bedict_v2",
    "deepbe",
    "crispron_be",
    "be_endo",
    "yachie_base_editing_prediction",
    "epcnnbe",
    "deepbaseeditor",
]

MODEL_META = {
    "behive_bystander": {
        "t3_role": "main_source_associated_conditional_derived",
        "output_unit": "editable_position",
        "output_origin": "derived_from_predicted_product_distribution",
        "input_type": "native_50nt_supported_behive_editor_cell",
        "default_prediction_scale": "conditional_position_rate_given_edited_reads",
        "score_family": "conditional_rate",
        "include_in_t3": "yes_source_associated",
        "recommended_metrics": "pooled_spearman; pooled_pearson; per-target_spearman; MAE; RMSE; top-k recovery",
        "disallowed_metrics": "",
        "notes": "Conditional position rates are derived by marginalizing BE-Hive predicted edited-product distributions; source-associated rows cannot support external-generalization claims.",
    },
    "forecast_be": {
        "t3_role": "main_external_rank_only_direct",
        "output_unit": "editable_position",
        "output_origin": "direct_position_score",
        "input_type": "protospacer_20nt_forecast_position",
        "default_prediction_scale": "uncalibrated_position_score",
        "score_family": "rank_only",
        "include_in_t3": "yes_rank_only",
        "recommended_metrics": "pooled_spearman; pooled_pearson_with_caution; per-target_spearman; top-1 accuracy; top-k recovery; active-window recall",
        "disallowed_metrics": "MAE; RMSE; calibration_slope",
        "notes": "FORECasT-BE emits uncalibrated position scores in the local adapter; do not treat scores as probabilities.",
    },
    "bedict_v2": {
        "t3_role": "supplementary_proxy_conditional_derived",
        "output_unit": "editable_position",
        "output_origin": "derived_from_predicted_product_distribution",
        "input_type": "ABE_proxy_protospacer20_pam4_all_AG",
        "default_prediction_scale": "conditional_position_rate_given_edited_reads",
        "score_family": "conditional_rate_proxy",
        "include_in_t3": "yes_supplementary_proxy",
        "recommended_metrics": "pooled_spearman; pooled_pearson; per-target_spearman; MAE; RMSE; top-k recovery",
        "disallowed_metrics": "",
        "notes": "BEDICT-V2 is evaluated through a model-compatible ABEmax-SpCas9 proxy panel; label as proxy external.",
    },
    "deepbe": {
        "t3_role": "supplementary_proxy_conditional_derived",
        "output_unit": "editable_position",
        "output_origin": "derived_from_predicted_product_distribution",
        "input_type": "ABE_proxy_30nt_window_deepbe",
        "default_prediction_scale": "conditional_position_rate_given_edited_reads",
        "score_family": "conditional_rate_proxy",
        "include_in_t3": "yes_supplementary_proxy",
        "recommended_metrics": "pooled_spearman; pooled_pearson; per-target_spearman; MAE; RMSE; top-k recovery",
        "disallowed_metrics": "",
        "notes": "DeepBE is evaluated through a model-compatible proxy panel where local outputs support product/profile marginalization.",
    },
    "crispron_be": {
        "t3_role": "main_external_conditional_derived",
        "output_unit": "editable_position",
        "output_origin": "derived_from_predicted_product_distribution",
        "input_type": "ABE_30nt_NGG_crispron",
        "default_prediction_scale": "conditional_position_rate_given_edited_reads",
        "score_family": "conditional_rate",
        "include_in_t3": "yes_external",
        "recommended_metrics": "pooled_spearman; pooled_pearson; per-target_spearman; MAE; RMSE; top-k recovery",
        "disallowed_metrics": "",
        "notes": "CRISPRon-BE conditional position rates are derived by marginalizing conditional outcome/profile outputs.",
    },
    "be_endo": {
        "t3_role": "main_external_rank_only_direct",
        "output_unit": "editable_position",
        "output_origin": "direct_position_score",
        "input_type": "40nt_NGG_be_endo_sequence_only",
        "default_prediction_scale": "uncalibrated_position_score",
        "score_family": "rank_only",
        "include_in_t3": "yes_rank_only",
        "recommended_metrics": "pooled_spearman; pooled_pearson_with_caution; per-target_spearman; top-1 accuracy; top-k recovery; active-window recall",
        "disallowed_metrics": "MAE; RMSE; calibration_slope",
        "notes": "BE_Endo sequence-only BE_Seq output is treated as an uncalibrated position score.",
    },
    "yachie_base_editing_prediction": {
        "t3_role": "supplementary_proxy_observed_candidate_score",
        "output_unit": "editable_position",
        "output_origin": "observed_candidate_or_proxy_score",
        "input_type": "ABE_proxy_30nt_yachie",
        "default_prediction_scale": "observed_position_candidate_score",
        "score_family": "rank_only_proxy",
        "include_in_t3": "supplementary_only",
        "recommended_metrics": "pooled_spearman; per-target_spearman; top-1 accuracy; top-k recovery",
        "disallowed_metrics": "MAE; RMSE; calibration_slope; calibrated_probability_claims",
        "notes": "Yachie local candidate scores should remain clearly labelled as observed/proxy candidate scores, not calibrated model probabilities.",
    },
    "epcnnbe": {
        "t3_role": "exploratory_proxy_conditional_derived",
        "output_unit": "editable_position",
        "output_origin": "derived_from_generated_outcome_proportions",
        "input_type": "30nt_ABE_CBE_epcnnbe_generated_outcomes",
        "default_prediction_scale": "conditional_position_rate_given_edited_reads",
        "score_family": "conditional_rate_exploratory_proxy",
        "include_in_t3": "exploratory_supplementary_only",
        "recommended_metrics": "pooled_spearman; pooled_pearson; per-target_spearman; MAE; RMSE; top-k recovery",
        "disallowed_metrics": "main_claim_without_exploratory_label",
        "notes": "EPCNNBE can be added as an exploratory T3-derived model by enumerating generated ABE/CBE outcome combinations and marginalizing predicted outcome proportions to editable positions.",
    },
    "deepbaseeditor": {
        "t3_role": "exploratory_historical_proxy_conditional_derived",
        "output_unit": "editable_position",
        "output_origin": "derived_from_historical_proportion_modules",
        "input_type": "DeepBaseEditor_ABE_CBE_fixed_window_proportion_modules",
        "default_prediction_scale": "conditional_position_rate_given_edited_reads",
        "score_family": "conditional_rate_exploratory_historical_proxy",
        "include_in_t3": "exploratory_supplementary_only",
        "recommended_metrics": "pooled_spearman; pooled_pearson; per-target_spearman; MAE; RMSE; top-k recovery",
        "disallowed_metrics": "main_claim_without_exploratory_label",
        "notes": "DeepBaseEditor proportion modules can be added as historical exploratory T3-derived profiles; ABE uses a 26nt module input and CBE uses a 25nt module input with fixed 8bp editing windows.",
    },
}

PANEL_FILES = {
    "behive_bystander": ("t2", "T2_expand_behive_native50.csv"),
    "forecast_be": ("t2", "T2_expand_forecast_position.csv"),
    "bedict_v2": ("t2", "T2_expand_bedict_v2.csv"),
    "deepbe": ("t2", "T2_expand_deepbe.csv"),
    "crispron_be": ("t2", "T2_expand_crispron_be.csv"),
    "be_endo": ("t2", "T2_expand_be_endo.csv"),
    "yachie_base_editing_prediction": ("t2", "T2_expand_yachie_targets.csv"),
    "epcnnbe": ("t3", "T3_exploratory_epcnnbe.csv"),
    "deepbaseeditor": ("t3", "T3_exploratory_deepbaseeditor.csv"),
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        out = float(value)
    except Exception:
        return ""
    if math.isnan(out) or math.isinf(out):
        return ""
    return f"{out:.12g}"


def bool_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def dna(value: object) -> bool:
    text = str(value or "").upper()
    return bool(text) and set(text) <= set("ACGT")


def find_protospacer_start(row: pd.Series) -> int:
    seq = str(row.get("sequence_50nt", "")).upper()
    protospacer = str(row.get("protospacer_20nt", "")).upper()
    if not dna(seq) or len(seq) != 50 or not dna(protospacer) or len(protospacer) != 20:
        return -1
    return seq.find(protospacer)


def add_sequence_windows(target: pd.DataFrame) -> pd.DataFrame:
    target = target.copy()
    starts: list[int] = []
    seq30: list[str] = []
    seq40: list[str] = []
    pam4: list[str] = []
    for _, row in target.iterrows():
        seq = str(row.get("sequence_50nt", "")).upper()
        start = find_protospacer_start(row)
        starts.append(start)
        seq30.append(seq[start - 4 : start + 26] if start >= 4 and start + 26 <= len(seq) else "")
        seq40.append(seq[start - 10 : start + 30] if start >= 10 and start + 30 <= len(seq) else "")
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
    return target


def add_input_flags(target: pd.DataFrame) -> pd.DataFrame:
    target = add_sequence_windows(target)
    outcome_count = numeric(target["outcome_row_count"])
    target["complete_outcome_profile_le_50"] = outcome_count.le(50).fillna(False).astype(str)
    target["native_50nt_available_flag"] = bool_text(target["native_50nt_available"]).astype(str)
    target["protospacer_20nt_available_flag"] = bool_text(target["protospacer_20nt_available"]).astype(str)
    target["behive_bystander_input_eligible_flag"] = bool_text(target["behive_bystander_input_eligible"]).astype(str)
    target["forecast_position_input_eligible_flag"] = bool_text(target["forecast_position_input_eligible"]).astype(str)
    target["concrete_30nt_available"] = target["sequence_context_30nt"].map(lambda s: len(str(s)) == 30 and dna(s)).astype(str)
    target["concrete_40nt_available"] = target["sequence_context_40nt"].map(lambda s: len(str(s)) == 40 and dna(s)).astype(str)
    target["ngg_pam"] = target["pam_type"].eq("NGG").astype(str)
    target["abe_editor_class"] = target["editor_class"].eq("ABE").astype(str)
    target["cbe_editor_class"] = target["editor_class"].eq("CBE").astype(str)
    return target


def position_key(df: pd.DataFrame) -> pd.Series:
    return numeric(df["protospacer_position"]).astype("Int64").astype(str)


def read_position_predictions(runs_dirs: Iterable[Path]) -> pd.DataFrame:
    paths: list[Path] = []
    for runs_dir in runs_dirs:
        paths.extend(sorted(set(runs_dir.glob("*/predictions_t2_*position*.csv")) | set(runs_dir.glob("*/predictions_t2_*positions*.csv"))))
        paths.extend(sorted(set(runs_dir.glob("*/predictions_t3_*position*.csv")) | set(runs_dir.glob("*/predictions_t3_*positions*.csv"))))
    paths = sorted(set(paths))
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            df = read_csv(path)
            df["_prediction_file"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    pred = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    pred["_pos"] = position_key(pred)
    pred["_pred"] = numeric(pred.get("predicted_position_rate", ""))
    score = numeric(pred.get("predicted_position_score", ""))
    pred.loc[pred["_pred"].isna(), "_pred"] = score
    return pred


def build_observed_position_profile(target: pd.DataFrame, position: pd.DataFrame) -> pd.DataFrame:
    position = position.copy()
    position = position.rename(columns={"measurement_scale": "observed_position_rate_scale"})
    position["observed_position_rate"] = position["observed_conditional_position_rate"]
    position["_pos"] = position_key(position)
    target_cols = [
        "target_profile_id",
        "source_family",
        "source_dataset",
        "source_relation_to_model",
        "source_associated_models",
        "editor_family",
        "replicate_strategy",
        "sequence_context_original",
        "sequence_50nt",
        "protospacer_20nt",
        "native_50nt_available",
        "protospacer_20nt_available",
        "behive_bystander_input_eligible",
        "forecast_position_input_eligible",
        "pam",
        "pam_type",
        "target_position",
        "intended_edit",
        "target_base",
        "measured_total_efficiency",
        "total_read_count",
        "observed_outcome_frequency_sum",
        "observed_max_product_frequency",
        "observed_mean_edited_positions_per_product",
        "observed_any_bystander_frequency",
        "observed_desired_product_frequency",
        "measurement_scale",
        "qc_sequence_pass",
        "qc_label_pass",
        "qc_level",
        "exclusion_reason",
        "notes",
        "protospacer_start_in_50nt",
        "sequence_context_30nt",
        "sequence_context_40nt",
        "pam4",
        "complete_outcome_profile_le_50",
        "native_50nt_available_flag",
        "protospacer_20nt_available_flag",
        "behive_bystander_input_eligible_flag",
        "forecast_position_input_eligible_flag",
        "concrete_30nt_available",
        "concrete_40nt_available",
        "ngg_pam",
        "abe_editor_class",
        "cbe_editor_class",
    ]
    merged = position.merge(target[target_cols], on="target_profile_id", how="left", suffixes=("", "_target"))
    merged = merged.rename(columns={"measurement_scale": "target_profile_measurement_scale", "notes": "target_notes"})
    merged.insert(0, "t3_position_profile_id", merged["position_profile_id"])
    columns = [
        "t3_position_profile_id",
        "position_profile_id",
        "target_profile_id",
        "source_profile_id",
        "source_family",
        "source_dataset",
        "source_relation_to_model",
        "source_associated_models",
        "source_row_key",
        "editor",
        "editor_class",
        "editor_family",
        "cell_context",
        "replicate_strategy",
        "sequence_50nt",
        "sequence_context_30nt",
        "sequence_context_40nt",
        "protospacer_20nt",
        "pam",
        "pam_type",
        "pam4",
        "protospacer_position",
        "reference_base",
        "target_base",
        "intended_edit",
        "target_position",
        "intended_position",
        "is_intended_position",
        "observed_position_rate",
        "observed_conditional_position_rate",
        "observed_absolute_position_rate_estimate",
        "observed_position_rate_scale",
        "target_profile_measurement_scale",
        "measured_total_efficiency",
        "total_read_count",
        "edited_read_count",
        "outcome_row_count",
        "editable_position_count",
        "observed_outcome_frequency_sum",
        "observed_max_product_frequency",
        "observed_mean_edited_positions_per_product",
        "observed_any_bystander_frequency",
        "observed_desired_product_frequency",
        "native_50nt_available",
        "protospacer_20nt_available",
        "behive_bystander_input_eligible",
        "forecast_position_input_eligible",
        "complete_outcome_profile_le_50",
        "concrete_30nt_available",
        "concrete_40nt_available",
        "ngg_pam",
        "qc_sequence_pass",
        "qc_label_pass",
        "qc_level",
        "exclusion_reason",
        "source_file",
        "target_notes",
        "_pos",
    ]
    return merged[[col for col in columns if col in merged.columns]]


def group_observed_coverage(observed: pd.DataFrame) -> pd.DataFrame:
    obs = observed.copy()
    obs["_obs_rate_num"] = numeric(obs["observed_position_rate"])
    obs["_abs_rate_num"] = numeric(obs["observed_absolute_position_rate_estimate"])
    group_cols = [
        "source_family",
        "source_dataset",
        "source_relation_to_model",
        "editor_class",
        "editor",
        "cell_context",
        "observed_position_rate_scale",
    ]
    rows = []
    for key, sub in obs.groupby(group_cols, dropna=False, sort=True):
        target_count = sub["target_profile_id"].nunique()
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "target_rows": target_count,
                "editable_position_rows": len(sub),
                "mean_editable_positions_per_target": fmt(len(sub) / target_count if target_count else None),
                "median_observed_position_rate": fmt(sub["_obs_rate_num"].median()),
                "mean_observed_position_rate": fmt(sub["_obs_rate_num"].mean()),
                "rows_with_absolute_position_estimate": int(sub["_abs_rate_num"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def input_type_coverage(target: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    flag_specs = [
        ("native_50nt_available", "native_50nt_available_flag", "Target has native 50 nt sequence context."),
        ("protospacer_20nt_available", "protospacer_20nt_available_flag", "Target has 20 nt protospacer."),
        ("behive_bystander_input_eligible", "behive_bystander_input_eligible_flag", "Target passes BE-Hive bystander input eligibility."),
        ("forecast_position_input_eligible", "forecast_position_input_eligible_flag", "Target passes FORECasT-BE position input eligibility."),
        ("complete_outcome_profile_le_50", "complete_outcome_profile_le_50", "Observed product profile has at most 50 rows."),
        ("concrete_30nt_available", "concrete_30nt_available", "Can derive a concrete 30 nt model window."),
        ("concrete_40nt_available", "concrete_40nt_available", "Can derive a concrete 40 nt model window."),
        ("ngg_pam", "ngg_pam", "Target has NGG PAM type."),
        ("abe_editor_class", "abe_editor_class", "ABE-class target."),
        ("cbe_editor_class", "cbe_editor_class", "CBE-class target."),
    ]
    target_groups = ["source_family", "source_dataset", "editor_class", "editor", "cell_context"]
    rows = []
    target_lookup = target.set_index("target_profile_id")
    for input_type, flag_col, description in flag_specs:
        eligible_targets = set(target.loc[bool_text(target[flag_col]), "target_profile_id"])
        eligible_obs = observed[observed["target_profile_id"].isin(eligible_targets)]
        for key, sub in eligible_obs.groupby(target_groups, dropna=False, sort=True):
            target_ids = set(sub["target_profile_id"])
            target_sub = target_lookup.loc[list(target_ids)] if target_ids else pd.DataFrame()
            rows.append(
                {
                    "input_type": input_type,
                    "input_description": description,
                    **dict(zip(target_groups, key)),
                    "eligible_targets": len(target_ids),
                    "eligible_position_rows": len(sub),
                    "median_editable_positions_per_target": fmt(numeric(target_sub["editable_position_count"]).median() if not target_sub.empty else None),
                }
            )
    return pd.DataFrame(rows)


def panel_coverage(
    target: pd.DataFrame,
    observed: pd.DataFrame,
    pred: pd.DataFrame,
    t2_panels_dir: Path,
    t3_panels_dir: Path,
) -> pd.DataFrame:
    pred_counts = pd.DataFrame()
    matched_counts = pd.DataFrame()
    if not pred.empty:
        pred_counts = pred.groupby("model_id", dropna=False).agg(
            prediction_rows=("prediction_id", "count"),
            prediction_targets=("target_profile_id", "nunique"),
            prediction_scales=("prediction_scale", lambda x: ";".join(sorted(set(map(str, x))))),
            source_relations=("source_relation_to_model", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        obs_keys = observed[["target_profile_id", "_pos", "reference_base", "observed_position_rate"]].copy()
        joined = pred.merge(obs_keys, on=["target_profile_id", "_pos", "reference_base"], how="inner")
        matched_counts = joined.groupby("model_id", dropna=False).agg(
            matched_position_rows=("prediction_id", "count"),
            matched_targets=("target_profile_id", "nunique"),
        )

    rows = []
    for model_id in EXPECTED_MODELS:
        panel_group, panel_filename = PANEL_FILES[model_id]
        panel_path = (t3_panels_dir if panel_group == "t3" else t2_panels_dir) / panel_filename
        panel_targets: set[str] = set()
        panel_df = pd.DataFrame()
        if panel_path.exists():
            panel_df = read_csv(panel_path)
            panel_targets = set(panel_df["target_profile_id"])
        panel_obs = observed[observed["target_profile_id"].isin(panel_targets)]
        pred_row = pred_counts.loc[model_id].to_dict() if model_id in pred_counts.index else {}
        match_row = matched_counts.loc[model_id].to_dict() if model_id in matched_counts.index else {}
        meta = MODEL_META[model_id]
        rows.append(
            {
                "model_id": model_id,
                "input_type": meta["input_type"],
                "panel_path": str(panel_path) if panel_path.exists() else "",
                "panel_targets": len(panel_targets),
                "panel_observed_position_rows": len(panel_obs),
                "panel_editor_classes": ";".join(sorted(set(panel_df.get("editor_class", pd.Series(dtype=str)).astype(str)))) if not panel_df.empty else "",
                "panel_editors": ";".join(sorted(set(panel_df.get("editor", pd.Series(dtype=str)).astype(str)))) if not panel_df.empty else "",
                "panel_cell_contexts": ";".join(sorted(set(panel_df.get("cell_context", pd.Series(dtype=str)).astype(str)))) if not panel_df.empty else "",
                "prediction_rows": int(pred_row.get("prediction_rows", 0) or 0),
                "prediction_targets": int(pred_row.get("prediction_targets", 0) or 0),
                "prediction_scales": pred_row.get("prediction_scales", meta["default_prediction_scale"]),
                "source_relations": pred_row.get("source_relations", ""),
                "matched_position_rows": int(match_row.get("matched_position_rows", 0) or 0),
                "matched_targets": int(match_row.get("matched_targets", 0) or 0),
                "unmatched_prediction_rows": int((pred_row.get("prediction_rows", 0) or 0) - (match_row.get("matched_position_rows", 0) or 0)),
            }
        )
    return pd.DataFrame(rows)


def model_output_audit(panel_cov: pd.DataFrame, metrics_path: Path) -> pd.DataFrame:
    metrics = read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    metric_lookup = {}
    if not metrics.empty:
        for _, row in metrics.iterrows():
            metric_lookup[str(row["model_id"])] = row.to_dict()

    rows = []
    panel_lookup = panel_cov.set_index("model_id")
    for model_id in EXPECTED_MODELS:
        meta = MODEL_META[model_id]
        panel_row = panel_lookup.loc[model_id].to_dict() if model_id in panel_lookup.index else {}
        metric_row = metric_lookup.get(model_id, {})
        rows.append(
            {
                "model_id": model_id,
                "proposed_t3_role": meta["t3_role"],
                "include_in_t3": meta["include_in_t3"],
                "output_unit": meta["output_unit"],
                "output_origin": meta["output_origin"],
                "prediction_scale": panel_row.get("prediction_scales", meta["default_prediction_scale"]),
                "score_family": meta["score_family"],
                "source_relation_to_model": panel_row.get("source_relations", metric_row.get("source_relation_to_model", "")),
                "panel_targets": panel_row.get("panel_targets", 0),
                "panel_observed_position_rows": panel_row.get("panel_observed_position_rows", 0),
                "prediction_rows": panel_row.get("prediction_rows", 0),
                "matched_position_rows": panel_row.get("matched_position_rows", 0),
                "matched_targets": panel_row.get("matched_targets", 0),
                "recommended_metrics": meta["recommended_metrics"],
                "disallowed_metrics": meta["disallowed_metrics"],
                "previous_t2_pooled_spearman": metric_row.get("pooled_spearman", ""),
                "previous_t2_top1_position_accuracy": metric_row.get("top1_position_accuracy", ""),
                "notes": meta["notes"],
            }
        )
    return pd.DataFrame(rows)


def metric_plan(model_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in model_audit.iterrows():
        calibrated = "conditional_position_rate_given_edited_reads" in str(row["prediction_scale"])
        rows.append(
            {
                "model_id": row["model_id"],
                "prediction_scale": row["prediction_scale"],
                "target_observation_scale": "conditional_position_rate_given_edited_reads",
                "primary_position_metrics": row["recommended_metrics"],
                "rate_error_metrics_allowed": str(bool(calibrated and "observed_candidate" not in str(row["prediction_scale"]))),
                "rank_metrics_allowed": "True",
                "explicitly_disallowed_metrics": row["disallowed_metrics"],
                "guardrail": "Do not use MAE/RMSE for uncalibrated or observed-candidate scores." if not calibrated else "MAE/RMSE are allowed only on the conditional edited-read scale.",
            }
        )
    return pd.DataFrame(rows)


def schema_manifest(output_dir: Path) -> dict[str, object]:
    return {
        "task_id": "T3",
        "task_name": "position_level_editing_profile_prediction",
        "unit_of_analysis": "editable position inside the editing window",
        "observed_table": {
            "path": str(output_dir / "T3_observed_position_profile.csv"),
            "primary_key": "t3_position_profile_id",
            "rate_column": "observed_position_rate",
            "rate_scale_column": "observed_position_rate_scale",
            "current_primary_scale": "conditional_position_rate_given_edited_reads",
        },
        "prediction_table_contract": {
            "required_columns": [
                "prediction_id",
                "target_profile_id",
                "model_id",
                "run_id",
                "model_editor",
                "model_cell_context",
                "protospacer_position",
                "reference_base",
                "predicted_position_rate",
                "predicted_position_score",
                "prediction_scale",
                "source_relation_to_model",
                "notes",
            ],
            "join_keys": ["target_profile_id", "protospacer_position", "reference_base"],
        },
        "scale_rules": [
            "Conditional position rates are evaluated against observed_conditional_position_rate.",
            "Absolute position-rate estimates are retained but not mixed into conditional-rate metrics.",
            "Uncalibrated position scores use rank/top-k metrics, not MAE/RMSE.",
            "Source-associated rows are not broad external-generalization evidence.",
        ],
    }


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_existing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def collapsed_input_summary(input_cov: pd.DataFrame) -> pd.DataFrame:
    if input_cov.empty:
        return pd.DataFrame()
    return (
        input_cov.groupby(["input_type", "input_description"], dropna=False, sort=True)
        .agg(
            eligible_targets=("eligible_targets", "sum"),
            eligible_position_rows=("eligible_position_rows", "sum"),
            source_editor_context_groups=("input_type", "count"),
        )
        .reset_index()
        .sort_values(["input_type"])
    )


def write_report(
    report_path: Path,
    output_dir: Path,
    observed: pd.DataFrame,
    source_cov: pd.DataFrame,
    input_cov: pd.DataFrame,
    panel_cov: pd.DataFrame,
    model_audit: pd.DataFrame,
    metric_plan_df: pd.DataFrame,
) -> None:
    input_summary = collapsed_input_summary(input_cov)
    source_summary = (
        source_cov.groupby(["source_family", "source_dataset", "editor_class", "cell_context"], dropna=False, sort=True)
        .agg(target_rows=("target_rows", "sum"), editable_position_rows=("editable_position_rows", "sum"))
        .reset_index()
        .sort_values(["source_family", "source_dataset", "editor_class", "cell_context"])
    )
    model_cols = [
        "model_id",
        "proposed_t3_role",
        "prediction_scale",
        "source_relation_to_model",
        "panel_targets",
        "matched_position_rows",
        "matched_targets",
        "include_in_t3",
    ]
    metric_cols = ["model_id", "rate_error_metrics_allowed", "rank_metrics_allowed", "explicitly_disallowed_metrics"]
    lines = [
        "# T3 position-profile audit",
        "",
        f"Generated: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "## Scope",
        "",
        "T3 is position-level editing profile prediction: predicting which editable positions within the editing window are edited and at what rate. This audit restores the locked T3 scope after T1 scalar efficiency and T2 outcome/bystander distribution were completed. It does not run a formal T3 benchmark.",
        "",
        "## Input anchors",
        "",
        f"- Observed position matrix: `{T2_MATRIX / 'position_profile_t2_draft.csv'}`",
        f"- Target metadata: `{T2_MATRIX / 'target_profile_t2_draft.csv'}`",
        f"- T2 expanded predictions: `{T2_RESULTS}`",
        f"- T3 exploratory expansion run: `{T3_EXPLORATORY_RUNS}`",
        f"- T3 exploratory panels: `{T3_PANELS}`",
        "",
        "## Observed T3 surface",
        "",
        f"- Observed editable-position rows: {len(observed)}",
        f"- Target profiles: {observed['target_profile_id'].nunique()}",
        f"- Source families: {'; '.join(sorted(set(observed['source_family'].astype(str))))}",
        f"- Editors: {observed['editor'].nunique()}",
        f"- Cell contexts: {'; '.join(sorted(set(observed['cell_context'].astype(str))))}",
        f"- Primary observed rate scale: {'; '.join(sorted(set(observed['observed_position_rate_scale'].astype(str))))}",
        "",
        "## Coverage by source/editor/context",
        "",
        "Full table: `T3_position_coverage_by_source_editor_context.csv`",
        "",
    ]
    lines.extend(["```text", source_summary.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Model-compatible input coverage",
            "",
            "Full table: `T3_position_coverage_by_input_type.csv`",
            "",
        ]
    )
    lines.extend(["```text", input_summary.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Expanded-panel coverage",
            "",
            "Full table: `T3_model_panel_coverage.csv`",
            "",
        ]
    )
    lines.extend(["```text", panel_cov.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Proposed T3 model inclusion",
            "",
            "Full table: `T3_model_inclusion_audit.csv`",
            "",
        ]
    )
    lines.extend(["```text", model_audit[model_cols].to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Metric guardrails",
            "",
            "Full table: `T3_metric_plan.csv`",
            "",
        ]
    )
    lines.extend(["```text", metric_plan_df[metric_cols].to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## T3 schema",
            "",
            "The proposed observed table is `T3_observed_position_profile.csv`. Its unit is one editable position for one target/editor/context observation. The primary response column is `observed_position_rate`, and the scale is stored in `observed_position_rate_scale`.",
            "",
            "Prediction rows should join on `target_profile_id`, `protospacer_position`, and `reference_base`, and must label one of `predicted_position_rate` or `predicted_position_score` plus `prediction_scale`.",
            "",
            "## Guardrails",
            "",
            "- T3 must not use T1 scalar total efficiency as its response variable.",
            "- T3 must not duplicate T2 outcome/product distribution metrics; product outputs only matter when marginalized to editable-position rates.",
            "- T3 must not become T4 high/low efficiency classification.",
            "- T3 may stratify by source/editor/context, but broad stress generalization remains T5.",
            "- Conditional position rates and absolute position-rate estimates must remain explicitly labelled and not pooled blindly.",
            "- Uncalibrated position scores and observed-candidate scores use rank/top-k metrics, not MAE/RMSE.",
            "- Source-associated and proxy rows must remain labelled in all formal T3 figures and tables.",
            "",
            "## Review gates before formal T3",
            "",
            "1. Confirm the proposed model-inclusion roles, especially proxy/supplementary labels for BEDICT-V2, DeepBE, and Yachie.",
            "2. Confirm that EPCNNBE and DeepBaseEditor remain exploratory/supplementary unless a larger T3-specific panel is run.",
            "3. Decide whether formal T3 should use only the existing 168-target T2 expanded union panel or generate a larger position-specific panel.",
            "4. Add T3-specific top-k and active-window recall scoring before formal benchmarking.",
            "5. Add representative target plots only after the T3 scoring table is frozen.",
            "",
            "## Output files",
            "",
        ]
    )
    for path in sorted(output_dir.glob("T3_*")):
        lines.append(f"- `{path}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=T2_MATRIX)
    parser.add_argument("--panels-dir", type=Path, default=T2_PANELS)
    parser.add_argument("--t3-panels-dir", type=Path, default=T3_PANELS)
    parser.add_argument("--runs-dir", type=Path, default=T2_RUNS)
    parser.add_argument("--t3-runs-dir", type=Path, default=T3_EXPLORATORY_RUNS)
    parser.add_argument("--t2-results-dir", type=Path, default=T2_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--report-path", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = read_csv(args.matrix_dir / "target_profile_t2_draft.csv")
    position = read_csv(args.matrix_dir / "position_profile_t2_draft.csv")
    target = add_input_flags(target)

    observed = build_observed_position_profile(target, position)
    source_cov = group_observed_coverage(observed)
    input_cov = input_type_coverage(target, observed)
    pred = read_position_predictions([args.runs_dir, args.t3_runs_dir])
    panel_cov = panel_coverage(target, observed, pred, args.panels_dir, args.t3_panels_dir)
    model_audit = model_output_audit(panel_cov, args.t2_results_dir / "T2_expanded_position_metrics.csv")
    metric_plan_df = metric_plan(model_audit)
    schema = schema_manifest(args.output_dir)

    write_table(args.output_dir / "T3_observed_position_profile.csv", observed.drop(columns=["_pos"], errors="ignore"))
    write_table(args.output_dir / "T3_position_coverage_by_source_editor_context.csv", source_cov)
    write_table(args.output_dir / "T3_position_coverage_by_input_type.csv", input_cov)
    write_table(args.output_dir / "T3_model_panel_coverage.csv", panel_cov)
    write_table(args.output_dir / "T3_model_inclusion_audit.csv", model_audit)
    write_table(args.output_dir / "T3_metric_plan.csv", metric_plan_df)
    (args.output_dir / "T3_SCHEMA.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "output_dir": str(args.output_dir),
        "report_path": str(args.report_path),
        "source_matrix_dir": str(args.matrix_dir),
        "source_panels_dir": str(args.panels_dir),
        "source_runs_dir": str(args.runs_dir),
        "t3_panels_dir": str(args.t3_panels_dir),
        "t3_runs_dir": str(args.t3_runs_dir),
        "observed_position_rows": len(observed),
        "observed_targets": int(observed["target_profile_id"].nunique()),
        "model_audit_rows": len(model_audit),
        "files": {
            "observed": str(args.output_dir / "T3_observed_position_profile.csv"),
            "source_coverage": str(args.output_dir / "T3_position_coverage_by_source_editor_context.csv"),
            "input_coverage": str(args.output_dir / "T3_position_coverage_by_input_type.csv"),
            "panel_coverage": str(args.output_dir / "T3_model_panel_coverage.csv"),
            "model_inclusion": str(args.output_dir / "T3_model_inclusion_audit.csv"),
            "metric_plan": str(args.output_dir / "T3_metric_plan.csv"),
            "schema": str(args.output_dir / "T3_SCHEMA.json"),
        },
    }
    (args.output_dir / "T3_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.report_path, args.output_dir, observed, source_cov, input_cov, panel_cov, model_audit, metric_plan_df)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
