#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "benchmark_core"
MATRIX = ROOT / "data" / "processed" / "T2_bystander_matrix"
RUNS = ROOT / "runs" / "T2_expand_v0"
OUT = ROOT / "formal_results" / "T2_expand_v0"
REGISTRY = ROOT / "config" / "model_registry.csv"
EXPECTED_MODELS = [
    "behive_bystander",
    "forecast_be",
    "bedict_v2",
    "deepbe",
    "crispron_be",
    "be_endo",
    "yachie_base_editing_prediction",
]
T2_INCLUDED_REASON = {
    "behive_bystander": "Native bystander/product distribution model; emits T2 outcome and derived position profiles on compatible BE-Hive inputs.",
    "forecast_be": "Native position-specific editing-rate model; included for T2 position-profile metrics.",
    "bedict_v2": "Outcome/profile model with adapter to normalized edited-product and position profiles on compatible ABE targets.",
    "deepbe": "Outcome/profile model with adapter to normalized edited-product and position profiles on compatible ABE targets.",
    "crispron_be": "Efficiency-and-outcome model with adapter-emitted conditional product and position profiles.",
    "be_endo": "Per-base efficiency model; included for T2 position-profile metrics where sequence/PAM constraints are satisfied.",
    "yachie_base_editing_prediction": "Historical pattern-frequency model; included as a proxy-compatible outcome/profile comparator.",
}
T2_EXCLUDED_REASON = {
    "behive_efficiency": "T1 scalar-efficiency model; current local output is not a bystander/product distribution or position profile.",
    "deepbaseeditor": "T1 scalar-efficiency model; no validated T2 adapter for normalized product or position-profile outputs.",
    "epcnnbe": "Not included in T2_expand_v0 because no validated adapter currently emits normalized edited-product distributions or position profiles on the unified T2 matrix.",
    "llm_embeditor": "Method-only retrainable pipeline, not an author pretrained BE predictor; excluded from the controlled BE-model denominator for T2_expand_v0.",
}
METHOD_ONLY_MODELS = {"llm_embeditor"}


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


def corr(x: pd.Series, y: pd.Series, rank: bool = False) -> Optional[float]:
    xy = pd.DataFrame({"x": numeric(x), "y": numeric(y)}).dropna()
    if len(xy) < 2 or xy["x"].nunique() < 2 or xy["y"].nunique() < 2:
        return None
    if rank:
        return float(xy["x"].rank(method="average").corr(xy["y"].rank(method="average")))
    return float(xy["x"].corr(xy["y"]))


def mae(x: pd.Series, y: pd.Series) -> Optional[float]:
    xy = pd.DataFrame({"x": numeric(x), "y": numeric(y)}).dropna()
    if xy.empty:
        return None
    return float(np.mean(np.abs(xy["x"] - xy["y"])))


def rmse(x: pd.Series, y: pd.Series) -> Optional[float]:
    xy = pd.DataFrame({"x": numeric(x), "y": numeric(y)}).dropna()
    if xy.empty:
        return None
    return float(np.sqrt(np.mean((xy["x"] - xy["y"]) ** 2)))


def jsd(obs: Dict[str, float], pred: Dict[str, float]) -> Optional[float]:
    keys = sorted(set(obs) | set(pred))
    if not keys:
        return None
    p = np.array([obs.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([pred.get(k, 0.0) for k in keys], dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    val = 0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p]))
    val += 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q]))
    return float(val)


def read_existing(paths: Iterable[Path]) -> list[pd.DataFrame]:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
            if not df.empty:
                df["_prediction_file"] = str(path)
                frames.append(df)
    return frames


def build_run_status(runs_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for model_id in EXPECTED_MODELS:
        model_dir = runs_dir / model_id
        position_files = sorted(set(model_dir.glob("predictions_t2_*position*.csv")) | set(model_dir.glob("predictions_t2_*positions*.csv"))) if model_dir.exists() else []
        outcome_files = sorted(model_dir.glob("predictions_t2_*outcomes*.csv")) if model_dir.exists() else []
        skipped_files = sorted(model_dir.glob("skipped*.csv")) if model_dir.exists() else []
        prediction_rows = 0
        skipped_rows = 0
        for path in position_files + outcome_files:
            try:
                prediction_rows += len(pd.read_csv(path, dtype=str, low_memory=False))
            except Exception:
                pass
        for path in skipped_files:
            try:
                skipped_rows += len(pd.read_csv(path, dtype=str, low_memory=False))
            except Exception:
                pass
        status = "completed" if prediction_rows > 0 else "no_predictions"
        note = ""
        if model_id == "crispron_be" and prediction_rows == 0:
            log = model_dir / "abe" / "crispron_be.log"
            if log.exists():
                text = log.read_text(encoding="utf-8", errors="replace")
                if "WSL_E_DISTRO_NOT_FOUND" in text or "could not convert string to float" in text:
                    status = "blocked_environment"
                    note = "RNAfold/WSL feature step failed in current sandbox; see CRISPRon-BE log."
        rows.append(
            {
                "model_id": model_id,
                "status": status,
                "prediction_rows": prediction_rows,
                "skipped_rows": skipped_rows,
                "position_files": len(position_files),
                "outcome_files": len(outcome_files),
                "notes": note,
            }
        )
    status_df = pd.DataFrame(rows)
    status_df.to_csv(output_dir / "T2_expanded_run_status.csv", index=False)
    return status_df


def build_model_inclusion_audit(output_dir: Path, registry_path: Path, run_status: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    local = registry[registry["access_mode"].eq("local")].copy()
    status_map = dict(zip(run_status["model_id"], run_status["status"]))
    local["t2_included"] = local["model_id"].isin(EXPECTED_MODELS)
    local["controlled_be_model_denominator"] = ~local["model_id"].isin(METHOD_ONLY_MODELS)
    local["t2_run_status"] = local["model_id"].map(status_map).fillna("not_run")
    local["t2_inclusion_note"] = local.apply(
        lambda row: T2_INCLUDED_REASON.get(row["model_id"], T2_EXCLUDED_REASON.get(row["model_id"], "Not selected for T2_expand_v0.")),
        axis=1,
    )
    columns = [
        "model_id",
        "display_name",
        "benchmark_tier",
        "current_status",
        "primary_task",
        "output_kind",
        "controlled_be_model_denominator",
        "t2_included",
        "t2_run_status",
        "t2_inclusion_note",
    ]
    audit = local[columns].sort_values(["t2_included", "model_id"], ascending=[False, True]).reset_index(drop=True)
    audit.to_csv(output_dir / "T2_model_inclusion_audit.csv", index=False)
    summary = {
        "local_registry_roles": int(len(local)),
        "controlled_be_model_roles": int(local["controlled_be_model_denominator"].sum()),
        "t2_included_roles": int(local["t2_included"].sum()),
        "t2_included_controlled_be_roles": int((local["t2_included"] & local["controlled_be_model_denominator"]).sum()),
    }
    return audit, summary


def score_positions(matrix_dir: Path, runs_dir: Path, output_dir: Path) -> pd.DataFrame:
    observed = pd.read_csv(matrix_dir / "position_profile_t2_draft.csv", dtype=str, low_memory=False).fillna("")
    observed["_pos"] = numeric(observed["protospacer_position"]).astype("Int64").astype(str)
    observed["_obs"] = numeric(observed["observed_conditional_position_rate"])

    position_paths = sorted(set(runs_dir.glob("*/predictions_t2_*position*.csv")) | set(runs_dir.glob("*/predictions_t2_*positions*.csv")))
    pred_frames = read_existing(position_paths)
    if not pred_frames:
        return pd.DataFrame()
    pred = pd.concat(pred_frames, ignore_index=True, sort=False).fillna("")
    pred["_pos"] = numeric(pred["protospacer_position"]).astype("Int64").astype(str)
    pred["_pred"] = numeric(pred.get("predicted_position_rate", ""))
    pred_score = numeric(pred.get("predicted_position_score", ""))
    pred.loc[pred["_pred"].isna(), "_pred"] = pred_score

    joined = pred.merge(
        observed[["target_profile_id", "_pos", "reference_base", "observed_conditional_position_rate", "_obs"]],
        on=["target_profile_id", "_pos", "reference_base"],
        how="inner",
    )
    joined.to_csv(output_dir / "T2_expanded_position_joined_predictions.csv", index=False)

    metric_rows: list[dict] = []
    per_target_rows: list[dict] = []
    for (model_id, scale, relation), sub in joined.groupby(["model_id", "prediction_scale", "source_relation_to_model"], dropna=False):
        target_spearman = []
        top_matches = []
        for target_id, target_sub in sub.groupby("target_profile_id"):
            sp = corr(target_sub["_pred"], target_sub["_obs"], rank=True)
            if sp is not None:
                target_spearman.append(sp)
                per_target_rows.append({"model_id": model_id, "prediction_scale": scale, "target_profile_id": target_id, "matched_positions": len(target_sub), "spearman": fmt(sp)})
            if target_sub["_obs"].notna().any() and target_sub["_pred"].notna().any():
                obs_top = target_sub.sort_values(["_obs", "_pos"], ascending=[False, True]).iloc[0]["_pos"]
                pred_top = target_sub.sort_values(["_pred", "_pos"], ascending=[False, True]).iloc[0]["_pos"]
                top_matches.append(obs_top == pred_top)
        calibrated = scale == "conditional_position_rate_given_edited_reads"
        metric_rows.append(
            {
                "model_id": model_id,
                "prediction_scale": scale,
                "source_relation_to_model": relation,
                "matched_position_rows": len(sub),
                "matched_targets": sub["target_profile_id"].nunique(),
                "pooled_spearman": fmt(corr(sub["_pred"], sub["_obs"], rank=True)),
                "pooled_pearson": fmt(corr(sub["_pred"], sub["_obs"], rank=False)),
                "mae": fmt(mae(sub["_pred"], sub["_obs"])) if calibrated else "",
                "rmse": fmt(rmse(sub["_pred"], sub["_obs"])) if calibrated else "",
                "mean_per_target_spearman": fmt(np.mean(target_spearman) if target_spearman else None),
                "top1_position_accuracy": fmt(np.mean(top_matches) if top_matches else None),
            }
        )

    pd.DataFrame(per_target_rows).to_csv(output_dir / "T2_expanded_position_per_target_metrics.csv", index=False)
    metrics = pd.DataFrame(metric_rows).sort_values(["model_id", "prediction_scale"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "T2_expanded_position_metrics.csv", index=False)
    return metrics


def score_outcomes(matrix_dir: Path, runs_dir: Path, output_dir: Path) -> pd.DataFrame:
    observed = pd.read_csv(matrix_dir / "outcome_profile_t2_draft.csv", dtype=str, low_memory=False).fillna("")
    target = pd.read_csv(matrix_dir / "target_profile_t2_draft.csv", dtype=str, low_memory=False).fillna("")
    complete_targets = set(target.loc[pd.to_numeric(target["outcome_row_count"], errors="coerce") <= 50, "target_profile_id"])
    observed = observed[observed["target_profile_id"].isin(complete_targets)].copy()

    pred_frames = read_existing(runs_dir.glob("*/predictions_t2_*outcomes*.csv"))
    if not pred_frames:
        return pd.DataFrame()
    pred = pd.concat(pred_frames, ignore_index=True, sort=False).fillna("")
    pred = pred[pred["target_profile_id"].isin(complete_targets)].copy()
    if pred.empty:
        return pd.DataFrame()

    per_target_rows: list[dict] = []
    for (model_id, scale, relation, target_id), pred_sub in pred.groupby(["model_id", "prediction_scale", "source_relation_to_model", "target_profile_id"], dropna=False):
        obs_sub = observed[observed["target_profile_id"] == target_id]
        if obs_sub.empty:
            continue
        obs_map = {str(row["outcome_events"]): float(row["observed_outcome_frequency"]) for _, row in obs_sub.iterrows() if str(row["observed_outcome_frequency"]) != ""}
        if not obs_map:
            continue
        score_col = "predicted_outcome_frequency" if scale == "conditional_outcome_frequency_given_edited_reads" else "predicted_outcome_score"
        pred_map = {}
        for _, row in pred_sub.iterrows():
            key = str(row.get("predicted_outcome_events", ""))
            try:
                value = float(row.get(score_col, ""))
            except Exception:
                continue
            if key:
                pred_map[key] = pred_map.get(key, 0.0) + value
        if not pred_map:
            continue
        obs_top = max(obs_map, key=obs_map.get)
        pred_top = max(pred_map, key=pred_map.get)
        common_obs = sorted(obs_map)
        obs_vals = pd.Series([obs_map[k] for k in common_obs])
        pred_vals = pd.Series([pred_map.get(k, 0.0) for k in common_obs])
        is_distribution = scale == "conditional_outcome_frequency_given_edited_reads"
        per_target_rows.append(
            {
                "model_id": model_id,
                "prediction_scale": scale,
                "source_relation_to_model": relation,
                "target_profile_id": target_id,
                "observed_products": len(obs_map),
                "predicted_products": len(pred_map),
                "jsd": fmt(jsd(obs_map, pred_map)) if is_distribution else "",
                "observed_product_spearman": fmt(corr(pred_vals, obs_vals, rank=True)),
                "observed_product_pearson": fmt(corr(pred_vals, obs_vals, rank=False)),
                "top1_product_match": str(obs_top == pred_top),
                "predicted_mass_on_observed_products": fmt(sum(pred_map.get(k, 0.0) for k in obs_map)) if is_distribution else "",
            }
        )

    per_target = pd.DataFrame(per_target_rows)
    per_target.to_csv(output_dir / "T2_expanded_outcome_per_target_metrics.csv", index=False)
    if per_target.empty:
        return pd.DataFrame()

    metric_rows: list[dict] = []
    for (model_id, scale, relation), sub in per_target.groupby(["model_id", "prediction_scale", "source_relation_to_model"], dropna=False):
        metric_rows.append(
            {
                "model_id": model_id,
                "prediction_scale": scale,
                "source_relation_to_model": relation,
                "matched_targets": len(sub),
                "mean_jsd": fmt(numeric(sub["jsd"]).mean()) if "jsd" in sub else "",
                "median_jsd": fmt(numeric(sub["jsd"]).median()) if "jsd" in sub else "",
                "mean_observed_product_spearman": fmt(numeric(sub["observed_product_spearman"]).mean()),
                "top1_product_accuracy": fmt((sub["top1_product_match"] == "True").mean()),
                "mean_predicted_mass_on_observed_products": fmt(numeric(sub["predicted_mass_on_observed_products"]).mean()) if "predicted_mass_on_observed_products" in sub else "",
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["model_id", "prediction_scale"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "T2_expanded_outcome_metrics.csv", index=False)
    return metrics


def write_report(
    output_dir: Path,
    position_metrics: pd.DataFrame,
    outcome_metrics: pd.DataFrame,
    run_status: pd.DataFrame,
    inclusion_audit: pd.DataFrame,
    inclusion_summary: dict[str, int],
) -> None:
    lines = [
        "# T2 expanded benchmark report",
        "",
        "This run expands T2 from the two-model smoke test to the locally runnable T2-eligible subset of the project model pool. It keeps conditional profile metrics separate from uncalibrated or observed-candidate scores.",
        "",
        "The current local registry contains 11 model roles. Excluding the method-only LLM-EmBEditor training pipeline gives 10 controlled BE model roles; T2_expand_v0 includes 7 roles with validated bystander, product-profile, or position-profile outputs.",
        "",
        "## Model-pool inclusion audit",
        "",
        f"- Local registry roles: {inclusion_summary['local_registry_roles']}",
        f"- Controlled BE model roles: {inclusion_summary['controlled_be_model_roles']}",
        f"- T2 included roles: {inclusion_summary['t2_included_roles']}",
        f"- T2 included controlled BE roles: {inclusion_summary['t2_included_controlled_be_roles']}",
        "",
        "```text",
        inclusion_audit[
            [
                "model_id",
                "controlled_be_model_denominator",
                "t2_included",
                "t2_run_status",
                "primary_task",
                "t2_inclusion_note",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Position-profile metrics",
        "",
    ]
    if position_metrics.empty:
        lines.append("No position predictions were available.")
    else:
        lines.extend(["```text", position_metrics.to_string(index=False), "```"])
    lines.extend(["", "## Outcome/product metrics", ""])
    if outcome_metrics.empty:
        lines.append("No outcome predictions were available.")
    else:
        lines.extend(["```text", outcome_metrics.to_string(index=False), "```"])
    lines.extend(["", "## Run status", ""])
    if run_status.empty:
        lines.append("No run status table was generated.")
    else:
        lines.extend(["```text", run_status.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- `source_associated` rows are not broad external-generalization evidence.",
            "- `proxy_external_candidate` rows use model-compatible editor proxies where exact editor labels do not exist locally.",
            "- `uncalibrated_position_score` and observed-candidate scales use rank/top1 metrics; MAE/RMSE are intentionally blank.",
            "- JSD is reported only for models emitting a normalized conditional edited-product distribution.",
        ]
    )
    (output_dir / "T2_EXPANDED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_status = build_run_status(args.runs_dir, args.output_dir)
    inclusion_audit, inclusion_summary = build_model_inclusion_audit(args.output_dir, REGISTRY, run_status)
    position_metrics = score_positions(args.matrix_dir, args.runs_dir, args.output_dir)
    outcome_metrics = score_outcomes(args.matrix_dir, args.runs_dir, args.output_dir)
    write_report(args.output_dir, position_metrics, outcome_metrics, run_status, inclusion_audit, inclusion_summary)
    manifest = {
        "output_dir": str(args.output_dir),
        "position_metric_rows": len(position_metrics),
        "outcome_metric_rows": len(outcome_metrics),
        "run_status_rows": len(run_status),
        "model_inclusion_audit_rows": len(inclusion_audit),
        **inclusion_summary,
    }
    (args.output_dir / "T2_EXPANDED_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
