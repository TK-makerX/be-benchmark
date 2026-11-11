#!/usr/bin/env python
"""Score completed T7 prediction runs on recommendation groups."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMBERSHIP = (
    ROOT
    / "data"
    / "processed"
    / "T7_prediction_completion_panels_v1"
    / "T7_priority_pair_crispron_forecast_complete_group_pilot_membership.csv"
)
DEFAULT_EVAL = (
    ROOT
    / "formal_results"
    / "T7_prediction_completion_priority_pair_complete_group_pilot_v1"
    / "predictions"
    / "combined"
    / "T1_V5_SMOKE_EVALUATION_TABLE.csv"
)
DEFAULT_OUT = ROOT / "formal_results" / "T7_prediction_completion_priority_pair_complete_group_pilot_v1" / "t7_scores"


def numeric(series: pd.Series | str) -> pd.Series:
    if isinstance(series, str):
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def bool_series(series: pd.Series | str) -> pd.Series:
    if isinstance(series, str):
        return pd.Series(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


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
        "scenario": ranked.iloc[0]["scenario"],
        "recommendation_group_id": ranked.iloc[0]["recommendation_group_id"],
        "model_id": ranked.iloc[0]["model_id"],
        "display_name": ranked.iloc[0]["display_name"],
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


def load_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    score = pred.get("prediction_value_for_correlation", pred.get("predicted_efficiency", ""))
    pred["predicted_recommendation_score"] = numeric(score)
    pred["valid_recommendation_score"] = bool_series(
        pred.get("is_valid_t1_prediction", pred["predicted_recommendation_score"].notna())
    )
    if "display_name" not in pred.columns:
        pred["display_name"] = pred["model_id"]
    keep = [
        "observation_id",
        "model_id",
        "display_name",
        "predicted_recommendation_score",
        "valid_recommendation_score",
        "prediction_value_kind",
    ]
    return pred[keep].drop_duplicates(["observation_id", "model_id"])


def score(membership: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    member = membership.copy()
    member["measured_efficiency"] = numeric(member["measured_efficiency"])
    member["recommendation_top_k"] = numeric(member["recommendation_top_k"]).fillna(1).astype(int)
    member["high_efficiency_label"] = bool_series(member["high_efficiency_label"])
    group_info = (
        member.groupby(["scenario", "recommendation_group_id"], dropna=False)
        .agg(raw_candidate_count=("observation_id", "nunique"))
        .reset_index()
    )
    merged = member.merge(predictions, on="observation_id", how="inner")
    merged["_valid_score"] = merged["predicted_recommendation_score"].where(
        merged["valid_recommendation_score"], np.nan
    )
    detail = (
        merged.groupby(["scenario", "model_id", "display_name", "recommendation_group_id"], dropna=False)
        .agg(
            predicted_candidate_count=("observation_id", "nunique"),
            valid_prediction_count=("valid_recommendation_score", "sum"),
            unique_prediction_scores=("_valid_score", "nunique"),
        )
        .reset_index()
        .merge(group_info, on=["scenario", "recommendation_group_id"], how="left")
    )
    detail["complete_raw_candidate_set"] = detail["predicted_candidate_count"].eq(detail["raw_candidate_count"]) & detail[
        "valid_prediction_count"
    ].eq(detail["raw_candidate_count"])
    detail["scoreable_rank_group"] = detail["complete_raw_candidate_set"] & detail["unique_prediction_scores"].ge(2)
    score_keys = detail.loc[
        detail["scoreable_rank_group"], ["scenario", "model_id", "recommendation_group_id"]
    ]
    score_rows = []
    if not score_keys.empty:
        scored_input = merged.merge(score_keys, on=["scenario", "model_id", "recommendation_group_id"], how="inner")
        scored_input = scored_input[
            scored_input["valid_recommendation_score"] & scored_input["predicted_recommendation_score"].notna()
        ]
        for (_, _, _), sub in scored_input.groupby(["scenario", "model_id", "recommendation_group_id"], sort=False):
            top_k = int(sub["recommendation_top_k"].max())
            score_rows.append(score_group(sub, top_k=top_k))
    group_scores = pd.DataFrame(score_rows)
    summary_rows = []
    models = predictions[["model_id", "display_name"]].drop_duplicates().sort_values("model_id")
    for scenario, sinfo in group_info.groupby("scenario", dropna=False):
        for _, model in models.iterrows():
            mid = model["model_id"]
            sub = detail[detail["scenario"].eq(scenario) & detail["model_id"].eq(mid)]
            scores = group_scores[group_scores["scenario"].eq(scenario) & group_scores["model_id"].eq(mid)] if not group_scores.empty else pd.DataFrame()
            row = {
                "scenario": scenario,
                "model_id": mid,
                "display_name": model["display_name"],
                "eligible_raw_groups": int(sinfo["recommendation_group_id"].nunique()),
                "eligible_raw_candidates": int(sinfo["raw_candidate_count"].sum()),
                "groups_with_any_prediction": int(sub["recommendation_group_id"].nunique()) if not sub.empty else 0,
                "complete_raw_candidate_groups": int(sub["complete_raw_candidate_set"].sum()) if not sub.empty else 0,
                "scoreable_rank_groups": int(sub["scoreable_rank_group"].sum()) if not sub.empty else 0,
                "scored_groups": int(scores["recommendation_group_id"].nunique()) if not scores.empty else 0,
            }
            for col in [
                "top1_best_hit",
                "topk_best_hit",
                "mrr",
                "ndcg_at_k",
                "precision_at_k",
                "enrichment_at_k",
                "regret_top1",
            ]:
                row[f"mean_{col}"] = float(scores[col].mean()) if not scores.empty else np.nan
            summary_rows.append(row)
    return pd.DataFrame(summary_rows), group_scores, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--evaluation-table", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(args.membership, dtype=str, low_memory=False).fillna("")
    predictions = load_predictions(args.evaluation_table)
    summary, group_scores, detail = score(membership, predictions)
    summary.to_csv(args.output_dir / "T7_completed_prediction_metric_summary.csv", index=False)
    group_scores.to_csv(args.output_dir / "T7_completed_prediction_group_scores.csv", index=False)
    detail.to_csv(args.output_dir / "T7_completed_prediction_group_coverage_detail.csv", index=False)
    report = [
        "# T7 completed prediction scoring",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Membership: `{args.membership}`",
        f"Evaluation table: `{args.evaluation_table}`",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Guardrail",
        "",
        "Scores are recommendation metrics within completed raw T7 groups, not scalar-regression leaderboard results.",
        "",
    ]
    (args.output_dir / "T7_COMPLETED_PREDICTION_SCORE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote T7 scores to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
