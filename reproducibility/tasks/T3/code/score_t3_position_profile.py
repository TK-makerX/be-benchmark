#!/usr/bin/env python
"""Score T3 position-level editing profile predictions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
T3_MATRIX = ROOT / "data" / "processed" / "T3_position_profile_matrix"
T2_RUNS = ROOT / "runs" / "T2_expand_v0"
T3_RUNS = ROOT / "runs" / "T3_exploratory_v0"
OUT = ROOT / "formal_results" / "T3_position_profile_v0"

RATE_SCALES = {"conditional_position_rate_given_edited_reads"}
RANK_ONLY_SCALES = {"uncalibrated_position_score", "observed_position_candidate_score"}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def numeric(series: pd.Series | str) -> pd.Series:
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


def corr(x: pd.Series, y: pd.Series, *, rank: bool = False) -> Optional[float]:
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


def calibration_line(pred: pd.Series, obs: pd.Series) -> tuple[Optional[float], Optional[float]]:
    xy = pd.DataFrame({"pred": numeric(pred), "obs": numeric(obs)}).dropna()
    if len(xy) < 2 or xy["pred"].nunique() < 2:
        return None, None
    var = float(np.var(xy["pred"], ddof=0))
    if var <= 0:
        return None, None
    cov = float(np.mean((xy["pred"] - xy["pred"].mean()) * (xy["obs"] - xy["obs"].mean())))
    slope = cov / var
    intercept = float(xy["obs"].mean() - slope * xy["pred"].mean())
    return slope, intercept


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
            if not df.empty:
                df["_prediction_file"] = str(path)
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    pred = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    pred["_pos"] = position_key(pred)
    pred["_pred_rate"] = numeric(pred.get("predicted_position_rate", ""))
    pred["_pred_score"] = numeric(pred.get("predicted_position_score", ""))
    pred["_pred"] = pred["_pred_rate"]
    pred.loc[pred["_pred"].isna(), "_pred"] = pred.loc[pred["_pred"].isna(), "_pred_score"]
    pred["_has_prediction"] = pred["_pred"].notna()
    return pred


def aggregate_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred
    rows = []
    group_cols = ["model_id", "target_profile_id", "_pos", "reference_base"]
    for key, sub in pred.groupby(group_cols, dropna=False, sort=False):
        scale = ";".join(sorted(set(str(x) for x in sub["prediction_scale"] if str(x))))
        relation = ";".join(sorted(set(str(x) for x in sub["source_relation_to_model"] if str(x))))
        model_editor = ";".join(sorted(set(str(x) for x in sub["model_editor"] if str(x))))
        model_cell_context = ";".join(sorted(set(str(x) for x in sub["model_cell_context"] if str(x))))
        values = numeric(sub["_pred"]).dropna()
        if not values.empty and any(s in scale for s in RATE_SCALES):
            value = float(values.sum())
        elif not values.empty:
            value = float(values.max())
        else:
            value = np.nan
        rows.append(
            {
                "model_id": key[0],
                "target_profile_id": key[1],
                "_pos": key[2],
                "reference_base": key[3],
                "model_editor": model_editor,
                "model_cell_context": model_cell_context,
                "prediction_scale": scale,
                "source_relation_to_model": relation,
                "predicted_value_raw": fmt(value),
                "_pred": value,
                "prediction_rows_collapsed": len(sub),
                "prediction_file": ";".join(sorted(set(str(x) for x in sub["_prediction_file"] if str(x)))),
            }
        )
    return pd.DataFrame(rows)


def sorted_position_ids(sub: pd.DataFrame, value_col: str, *, descending: bool = True) -> list[str]:
    sort_cols = [value_col, "protospacer_position_num", "reference_base"]
    ascending = [not descending, True, True]
    return (
        sub.sort_values(sort_cols, ascending=ascending)
        .assign(_position_id=lambda d: d["_pos"].astype(str) + ":" + d["reference_base"].astype(str))["_position_id"]
        .tolist()
    )


def target_metrics(target_sub: pd.DataFrame, *, allow_rate_error: bool) -> dict[str, object]:
    obs = numeric(target_sub["_obs"])
    pred = numeric(target_sub["_pred_filled"])
    active = obs > 0
    pred_positive = pred > 0
    target_sub = target_sub.copy()
    target_sub["protospacer_position_num"] = numeric(target_sub["protospacer_position"])

    out: dict[str, object] = {
        "matched_universe_positions": len(target_sub),
        "predicted_nonzero_positions": int(pred_positive.sum()),
        "observed_active_positions": int(active.sum()),
        "prediction_coverage_fraction": fmt(float(target_sub["has_model_prediction"].mean()) if len(target_sub) else None),
        "spearman": fmt(corr(pred, obs, rank=True)),
        "pearson": fmt(corr(pred, obs, rank=False)),
        "mae": fmt(mae(pred, obs)) if allow_rate_error else "",
        "rmse": fmt(rmse(pred, obs)) if allow_rate_error else "",
        "active_position_recall": fmt(float((active & pred_positive).sum() / active.sum()) if active.sum() else None),
        "predicted_positive_precision": fmt(float((active & pred_positive).sum() / pred_positive.sum()) if pred_positive.sum() else None),
    }

    if len(target_sub) and obs.notna().any() and pred.notna().any():
        obs_top = sorted_position_ids(target_sub, "_obs", descending=True)[0]
        pred_top = sorted_position_ids(target_sub, "_pred_filled", descending=True)[0]
        out["observed_top_position"] = obs_top
        out["predicted_top_position"] = pred_top
        out["top1_position_match"] = str(obs_top == pred_top)
        active_count = int(active.sum())
        k3 = min(3, len(target_sub))
        obs_top3 = set(sorted_position_ids(target_sub, "_obs", descending=True)[:k3])
        pred_top3 = set(sorted_position_ids(target_sub, "_pred_filled", descending=True)[:k3])
        out["top3_position_recovery"] = fmt(len(obs_top3 & pred_top3) / len(obs_top3) if obs_top3 else None)
        k_active = min(active_count, len(target_sub))
        if k_active > 0:
            obs_active_top = set(sorted_position_ids(target_sub, "_obs", descending=True)[:k_active])
            pred_active_top = set(sorted_position_ids(target_sub, "_pred_filled", descending=True)[:k_active])
            out["top_active_position_recovery"] = fmt(len(obs_active_top & pred_active_top) / len(obs_active_top))
        else:
            out["top_active_position_recovery"] = ""
    return out


def build_evaluation_table(observed: pd.DataFrame, model_audit: pd.DataFrame, panel_coverage: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_agg = aggregate_predictions(pred)
    observed = observed.copy()
    observed["_pos"] = position_key(observed)
    observed["_obs"] = numeric(observed["observed_position_rate"])

    eval_parts = []
    status_rows = []
    model_table = model_audit.merge(
        panel_coverage[["model_id", "panel_path"]],
        on="model_id",
        how="left",
    )
    for _, model in model_table.iterrows():
        model_id = model["model_id"]
        panel_path = Path(model.get("panel_path", ""))
        panel_targets: set[str] = set()
        if panel_path.exists():
            panel = read_csv(panel_path)
            panel_targets = set(panel["target_profile_id"])
        universe = observed[observed["target_profile_id"].isin(panel_targets)].copy()
        model_pred = pred_agg[pred_agg["model_id"].eq(model_id)].copy() if not pred_agg.empty else pd.DataFrame()
        joined = universe.merge(
            model_pred[
                [
                    "target_profile_id",
                    "_pos",
                    "reference_base",
                    "model_editor",
                    "model_cell_context",
                    "prediction_scale",
                    "source_relation_to_model",
                    "predicted_value_raw",
                    "_pred",
                    "prediction_rows_collapsed",
                    "prediction_file",
                ]
            ]
            if not model_pred.empty
            else pd.DataFrame(columns=["target_profile_id", "_pos", "reference_base"]),
            on=["target_profile_id", "_pos", "reference_base"],
            how="left",
        )
        if "prediction_scale" not in joined.columns:
            joined["prediction_scale"] = ""
        joined["prediction_scale"] = joined["prediction_scale"].replace("", model.get("prediction_scale", ""))
        joined["source_relation_to_model_prediction"] = joined.get("source_relation_to_model", "").replace("", model.get("source_relation_to_model", ""))
        joined["predicted_value_raw"] = numeric(joined.get("predicted_value_raw", ""))
        joined["has_model_prediction"] = joined["predicted_value_raw"].notna()
        joined["_pred_filled"] = joined["predicted_value_raw"].fillna(0.0)
        joined["predicted_position_value"] = joined["_pred_filled"].map(fmt)
        joined["missing_prediction_filled_with_zero"] = (~joined["has_model_prediction"]).astype(str)
        joined["model_id"] = model_id
        joined["include_in_t3"] = model.get("include_in_t3", "")
        joined["proposed_t3_role"] = model.get("proposed_t3_role", "")
        joined["score_family"] = model.get("score_family", "")
        eval_parts.append(joined)
        unmatched = 0
        if not model_pred.empty:
            matched_keys = set(zip(joined["target_profile_id"], joined["_pos"], joined["reference_base"]))
            unmatched = int(
                sum(
                    key not in matched_keys
                    for key in zip(model_pred["target_profile_id"], model_pred["_pos"], model_pred["reference_base"])
                )
            )
        status_rows.append(
            {
                "model_id": model_id,
                "include_in_t3": model.get("include_in_t3", ""),
                "panel_targets": len(panel_targets),
                "evaluation_universe_rows": len(universe),
                "prediction_rows_read": len(pred[pred["model_id"].eq(model_id)]) if not pred.empty else 0,
                "prediction_positions_after_collapse": len(model_pred),
                "matched_prediction_positions": int(joined["has_model_prediction"].sum()) if len(joined) else 0,
                "missing_universe_positions_filled_zero": int((~joined["has_model_prediction"]).sum()) if len(joined) else 0,
                "unmatched_prediction_positions_outside_universe": unmatched,
            }
        )
    evaluation = pd.concat(eval_parts, ignore_index=True, sort=False) if eval_parts else pd.DataFrame()
    return evaluation, pd.DataFrame(status_rows)


def score_models(evaluation: pd.DataFrame, model_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_lookup = model_audit.set_index("model_id").to_dict(orient="index")
    metric_rows: list[dict] = []
    per_target_rows: list[dict] = []

    for model_id, sub in evaluation.groupby("model_id", sort=False):
        audit = audit_lookup.get(model_id, {})
        scale = audit.get("prediction_scale", "")
        include = audit.get("include_in_t3", "")
        allow_rate_error = scale in RATE_SCALES and scale not in RANK_ONLY_SCALES
        target_spearman = []
        target_pearson = []
        top1 = []
        top3 = []
        top_active = []
        active_recall = []
        predicted_precision = []

        for target_id, target_sub in sub.groupby("target_profile_id", sort=False):
            tm = target_metrics(target_sub, allow_rate_error=allow_rate_error)
            row = {
                "model_id": model_id,
                "include_in_t3": include,
                "prediction_scale": scale,
                "target_profile_id": target_id,
                **tm,
            }
            per_target_rows.append(row)
            if row.get("spearman") != "":
                target_spearman.append(float(row["spearman"]))
            if row.get("pearson") != "":
                target_pearson.append(float(row["pearson"]))
            if row.get("top1_position_match") in ["True", "False"]:
                top1.append(row["top1_position_match"] == "True")
            if row.get("top3_position_recovery") != "":
                top3.append(float(row["top3_position_recovery"]))
            if row.get("top_active_position_recovery") != "":
                top_active.append(float(row["top_active_position_recovery"]))
            if row.get("active_position_recall") != "":
                active_recall.append(float(row["active_position_recall"]))
            if row.get("predicted_positive_precision") != "":
                predicted_precision.append(float(row["predicted_positive_precision"]))

        obs = numeric(sub["_obs"])
        pred = numeric(sub["_pred_filled"])
        slope, intercept = calibration_line(pred, obs) if allow_rate_error else (None, None)
        metric_rows.append(
            {
                "model_id": model_id,
                "include_in_t3": include,
                "proposed_t3_role": audit.get("proposed_t3_role", ""),
                "prediction_scale": scale,
                "source_relation_to_model": audit.get("source_relation_to_model", ""),
                "evaluation_universe_rows": len(sub),
                "evaluation_targets": sub["target_profile_id"].nunique(),
                "matched_prediction_positions": int(sub["has_model_prediction"].sum()),
                "prediction_coverage_fraction": fmt(float(sub["has_model_prediction"].mean()) if len(sub) else None),
                "pooled_spearman": fmt(corr(pred, obs, rank=True)),
                "pooled_pearson": fmt(corr(pred, obs, rank=False)),
                "mean_per_target_spearman": fmt(np.mean(target_spearman) if target_spearman else None),
                "median_per_target_spearman": fmt(np.median(target_spearman) if target_spearman else None),
                "mean_per_target_pearson": fmt(np.mean(target_pearson) if target_pearson else None),
                "mae": fmt(mae(pred, obs)) if allow_rate_error else "",
                "rmse": fmt(rmse(pred, obs)) if allow_rate_error else "",
                "calibration_slope_obs_on_pred": fmt(slope) if allow_rate_error else "",
                "calibration_intercept_obs_on_pred": fmt(intercept) if allow_rate_error else "",
                "mean_top1_position_accuracy": fmt(np.mean(top1) if top1 else None),
                "mean_top3_position_recovery": fmt(np.mean(top3) if top3 else None),
                "mean_top_active_position_recovery": fmt(np.mean(top_active) if top_active else None),
                "mean_active_position_recall": fmt(np.mean(active_recall) if active_recall else None),
                "mean_predicted_positive_precision": fmt(np.mean(predicted_precision) if predicted_precision else None),
                "rate_error_metrics_allowed": str(allow_rate_error),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    order = {model_id: idx for idx, model_id in enumerate(model_audit["model_id"].tolist())}
    metrics["_order"] = metrics["model_id"].map(order)
    metrics = metrics.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    per_target = pd.DataFrame(per_target_rows)
    return metrics, per_target


def write_report(output_dir: Path, metrics: pd.DataFrame, run_status: pd.DataFrame) -> None:
    main_mask = ~metrics["include_in_t3"].astype(str).str.contains("exploratory", case=False, na=False)
    lines = [
        "# T3 position-profile scoring report",
        "",
        f"Generated: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "## Scope",
        "",
        "T3 evaluates position-level editing profile prediction. The scoring universe is all observed editable positions in each model-specific panel target. Missing model predictions inside that universe are filled with zero. This is intentionally stricter than the older T2 inner-join position summary.",
        "",
        "## Main and supplementary metrics",
        "",
    ]
    display_cols = [
        "model_id",
        "include_in_t3",
        "prediction_scale",
        "evaluation_universe_rows",
        "evaluation_targets",
        "matched_prediction_positions",
        "prediction_coverage_fraction",
        "pooled_spearman",
        "mean_per_target_spearman",
        "mae",
        "rmse",
        "mean_top1_position_accuracy",
        "mean_top3_position_recovery",
        "mean_active_position_recall",
    ]
    lines.extend(["```text", metrics.loc[main_mask, display_cols].to_string(index=False), "```"])
    lines.extend(["", "## Exploratory metrics", ""])
    exploratory = metrics.loc[~main_mask, display_cols]
    if exploratory.empty:
        lines.append("No exploratory candidates were scored.")
    else:
        lines.extend(["```text", exploratory.to_string(index=False), "```"])
    lines.extend(["", "## Run status", "", "```text", run_status.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- T3's response is observed editable-position rate, not total editing efficiency.",
            "- Product/outcome models enter T3 only after marginalizing predicted edited-product profiles to editable positions.",
            "- MAE/RMSE/calibration are blank for uncalibrated position scores and observed-candidate scores.",
            "- EPCNNBE and DeepBaseEditor are exploratory supplementary candidates in this run, not main-claim models.",
            "- Source-associated, external, proxy, and exploratory labels must remain visible in figures and manuscript tables.",
        ]
    )
    (output_dir / "T3_POSITION_PROFILE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=T3_MATRIX)
    parser.add_argument("--t2-runs-dir", type=Path, default=T2_RUNS)
    parser.add_argument("--t3-runs-dir", type=Path, default=T3_RUNS)
    parser.add_argument("--model-audit", type=Path, default=None)
    parser.add_argument("--panel-coverage", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed = read_csv(args.matrix_dir / "T3_observed_position_profile.csv")
    model_audit = read_csv(args.model_audit or (args.matrix_dir / "T3_model_inclusion_audit.csv"))
    panel_coverage = read_csv(args.panel_coverage or (args.matrix_dir / "T3_model_panel_coverage.csv"))
    pred = read_position_predictions([args.t2_runs_dir, args.t3_runs_dir])

    evaluation, run_status = build_evaluation_table(observed, model_audit, panel_coverage, pred)
    metrics, per_target = score_models(evaluation, model_audit)

    output_eval = evaluation.copy()
    for col in ["_obs", "_pred", "_pred_filled"]:
        if col in output_eval.columns:
            output_eval[col] = output_eval[col].map(fmt)
    output_eval.to_csv(args.output_dir / "T3_position_joined_universe_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "T3_position_metrics.csv", index=False)
    per_target.to_csv(args.output_dir / "T3_position_per_target_metrics.csv", index=False)
    run_status.to_csv(args.output_dir / "T3_run_status.csv", index=False)
    write_report(args.output_dir, metrics, run_status)

    manifest = {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "output_dir": str(args.output_dir),
        "observed_rows": len(observed),
        "models_scored": len(metrics),
        "evaluation_rows": len(evaluation),
        "files": {
            "joined_universe": str(args.output_dir / "T3_position_joined_universe_predictions.csv"),
            "metrics": str(args.output_dir / "T3_position_metrics.csv"),
            "per_target": str(args.output_dir / "T3_position_per_target_metrics.csv"),
            "run_status": str(args.output_dir / "T3_run_status.csv"),
            "report": str(args.output_dir / "T3_POSITION_PROFILE_REPORT.md"),
        },
    }
    (args.output_dir / "T3_POSITION_PROFILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
