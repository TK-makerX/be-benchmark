"""Recompute the T7 unique-design group metrics from the original frozen scores."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO = "design_selection_high_efficiency"


MODELS = ["crispron_be", "forecast_be"]


DISPLAY = {
    "crispron_be": "CRISPRon-BE",
    "forecast_be": "FORECasT-BE",
    "random": "Random",
    "oracle": "Oracle",
}


def dcg(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sum(values / np.log2(np.arange(2, len(values) + 2))))


def score_ranked_group(
    sub: pd.DataFrame,
    score_col: str,
    model_id: str,
) -> dict[str, object]:
    ranked = sub.sort_values([score_col, "design_id"], ascending=[False, True], kind="mergesort")
    n = len(ranked)
    k = min(10, n)
    top = ranked.head(k)
    best_eff = float(ranked["measured_efficiency"].max())
    best_mask = ranked["measured_efficiency"].eq(best_eff).to_numpy()
    first_best_rank = int(np.flatnonzero(best_mask)[0] + 1)
    high = ranked["high_efficiency_label"].astype(bool)
    prevalence = float(high.mean())
    precision = float(top["high_efficiency_label"].astype(bool).mean())
    ideal = ranked.sort_values(["measured_efficiency", "design_id"], ascending=[False, True]).head(k)
    ideal_dcg = dcg(ideal["measured_efficiency"].to_numpy())
    top_dcg = dcg(top["measured_efficiency"].to_numpy())
    return {
        "scenario": SCENARIO,
        "recommendation_group_id": ranked["recommendation_group_id"].iloc[0],
        "source_family": ranked["source_family"].iloc[0],
        "model_id": model_id,
        "display_name": DISPLAY[model_id],
        "n_candidates": n,
        "top_k_used": k,
        "top1_best_hit": bool(best_mask[0]),
        "topk_best_hit": bool(top["measured_efficiency"].eq(best_eff).any()),
        "mrr": 1.0 / first_best_rank,
        "ndcg_at_k": top_dcg / ideal_dcg if ideal_dcg > 0 else np.nan,
        "precision_at_k": precision,
        "recall_at_k": float(top["high_efficiency_label"].astype(bool).sum() / high.sum()),
        "enrichment_at_k": precision / prevalence if prevalence > 0 else np.nan,
        "regret_top1": best_eff - float(top["measured_efficiency"].iloc[0]),
        "predicted_top_efficiency": float(top["measured_efficiency"].iloc[0]),
        "oracle_efficiency": best_eff,
    }


def score_models(score_table: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    complete = coverage[coverage["scoreable_rank_group"]].copy()
    rows: list[dict[str, object]] = []
    for model_id in MODELS:
        group_ids = set(complete.loc[complete["model_id"].eq(model_id), "recommendation_group_id"])
        model = score_table[
            score_table["model_id"].eq(model_id)
            & score_table["recommendation_group_id"].isin(group_ids)
        ].copy()
        for _, sub in model.groupby("recommendation_group_id", sort=True):
            rows.append(score_ranked_group(sub, "predicted_recommendation_score", model_id))
    return pd.DataFrame(rows)


METRIC_COLS = [
    "ndcg_at_k",
    "precision_at_k",
    "enrichment_at_k",
    "mrr",
    "topk_best_hit",
    "top1_best_hit",
    "regret_top1",
]


def random_and_oracle_baselines(
    unique: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    groups: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for group_id, sub in unique.groupby("recommendation_group_id", sort=True):
        ordered = sub.sort_values("design_id", kind="mergesort").reset_index(drop=True)
        eff = ordered["measured_efficiency"].to_numpy(float)
        high = ordered["high_efficiency_label"].astype(bool).to_numpy()
        n = len(ordered)
        k = min(10, n)
        best_eff = float(np.max(eff))
        best = eff == best_eff
        ideal_order = np.lexsort((ordered["design_id"].to_numpy(), -eff))[:k]
        ideal_dcg = dcg(eff[ideal_order])
        groups.append(
            {
                "group_id": group_id,
                "source_family": ordered["source_family"].iloc[0],
                "eff": eff,
                "high": high,
                "best": best,
                "n": n,
                "k": k,
                "best_eff": best_eff,
                "ideal_dcg": ideal_dcg,
            }
        )
        oracle = ordered.iloc[ideal_order].copy()
        oracle["oracle_score"] = oracle["measured_efficiency"]
        oracle_rows.append(score_ranked_group(ordered.assign(oracle_score=eff), "oracle_score", "oracle"))

    random_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for replicate in range(1, replicates + 1):
        this_rep: list[dict[str, object]] = []
        for group in groups:
            n = int(group["n"])
            k = int(group["k"])
            eff = group["eff"]
            high = group["high"]
            best = group["best"]
            top_idx = rng.choice(n, size=k, replace=False)
            best_positions = np.flatnonzero(best[top_idx])
            if len(best_positions):
                first_best_rank = int(best_positions[0] + 1)
            else:
                best_count = int(best.sum())
                if k == n:
                    raise AssertionError("A random full ranking cannot omit all measured-best candidates.")
                remaining_positions = rng.choice(n - k, size=best_count, replace=False) + k + 1
                first_best_rank = int(np.min(remaining_positions))
            prevalence = float(np.mean(high))
            precision = float(np.mean(high[top_idx]))
            row = {
                "replicate": replicate,
                "scenario": SCENARIO,
                "recommendation_group_id": group["group_id"],
                "source_family": group["source_family"],
                "model_id": "random",
                "display_name": "Random",
                "n_candidates": n,
                "top_k_used": k,
                "top1_best_hit": bool(best[top_idx[0]]),
                "topk_best_hit": bool(np.any(best[top_idx])),
                "mrr": 1.0 / first_best_rank,
                "ndcg_at_k": dcg(eff[top_idx]) / float(group["ideal_dcg"]),
                "precision_at_k": precision,
                "recall_at_k": float(np.sum(high[top_idx]) / np.sum(high)),
                "enrichment_at_k": precision / prevalence if prevalence > 0 else np.nan,
                "regret_top1": float(group["best_eff"] - eff[top_idx[0]]),
                "predicted_top_efficiency": float(eff[top_idx[0]]),
                "oracle_efficiency": float(group["best_eff"]),
            }
            random_rows.append(row)
            this_rep.append(row)
        rep_df = pd.DataFrame(this_rep)
        rep_summary = {"replicate": replicate, "seed": seed}
        for metric in METRIC_COLS:
            rep_summary[f"mean_{metric}"] = float(rep_df[metric].astype(float).mean())
        replicate_rows.append(rep_summary)

    random_all = pd.DataFrame(random_rows)
    random_replicates = pd.DataFrame(replicate_rows)
    random_group = (
        random_all.groupby(["scenario", "recommendation_group_id", "source_family"], sort=True)
        .agg(
            n_candidates=("n_candidates", "first"),
            top_k_used=("top_k_used", "first"),
            top1_best_hit=("top1_best_hit", "mean"),
            topk_best_hit=("topk_best_hit", "mean"),
            mrr=("mrr", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            enrichment_at_k=("enrichment_at_k", "mean"),
            regret_top1=("regret_top1", "mean"),
            predicted_top_efficiency=("predicted_top_efficiency", "mean"),
            oracle_efficiency=("oracle_efficiency", "first"),
        )
        .reset_index()
    )
    random_group["model_id"] = "random"
    random_group["display_name"] = "Random"
    oracle_group = pd.DataFrame(oracle_rows)
    return random_replicates, random_group, oracle_group, random_all


def summarize_metrics(
    model_group: pd.DataFrame,
    random_replicates: pd.DataFrame,
    random_group: pd.DataFrame,
    oracle_group: pd.DataFrame,
    candidate_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in MODELS:
        sub = model_group[model_group["model_id"].eq(model_id)]
        row: dict[str, object] = {
            "scenario": SCENARIO,
            "model_id": model_id,
            "display_name": DISPLAY[model_id],
            "baseline_type": "model",
            "scored_groups": sub["recommendation_group_id"].nunique(),
            "unique_design_group_instances": candidate_count,
            "random_replicates": np.nan,
            "random_seed": np.nan,
        }
        for metric in METRIC_COLS:
            row[f"mean_{metric}"] = float(sub[metric].astype(float).mean())
            row[f"ci95_low_{metric}"] = np.nan
            row[f"ci95_high_{metric}"] = np.nan
        rows.append(row)

    random_row: dict[str, object] = {
        "scenario": SCENARIO,
        "model_id": "random",
        "display_name": "Random",
        "baseline_type": "baseline",
        "scored_groups": random_group["recommendation_group_id"].nunique(),
        "unique_design_group_instances": candidate_count,
        "random_replicates": len(random_replicates),
        "random_seed": int(random_replicates["seed"].iloc[0]),
    }
    for metric in METRIC_COLS:
        values = random_replicates[f"mean_{metric}"].astype(float)
        random_row[f"mean_{metric}"] = float(values.mean())
        random_row[f"ci95_low_{metric}"] = float(values.quantile(0.025))
        random_row[f"ci95_high_{metric}"] = float(values.quantile(0.975))
    rows.append(random_row)

    oracle_row: dict[str, object] = {
        "scenario": SCENARIO,
        "model_id": "oracle",
        "display_name": "Oracle",
        "baseline_type": "baseline",
        "scored_groups": oracle_group["recommendation_group_id"].nunique(),
        "unique_design_group_instances": candidate_count,
        "random_replicates": np.nan,
        "random_seed": np.nan,
    }
    for metric in METRIC_COLS:
        oracle_row[f"mean_{metric}"] = float(oracle_group[metric].astype(float).mean())
        oracle_row[f"ci95_low_{metric}"] = np.nan
        oracle_row[f"ci95_high_{metric}"] = np.nan
    rows.append(oracle_row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-replicates", type=int, default=1000)
    args = parser.parse_args()
    fields = ["recommendation_group_id", "design_id", "source_family", "measured_efficiency", "high_efficiency_label"]
    scores = pd.read_csv(args.task / "inputs/T7_unique_design_model_scores.csv", low_memory=False,
                         usecols=fields + ["model_id", "predicted_recommendation_score"])
    coverage = pd.read_csv(args.task / "expected/T7_unique_design_model_group_coverage.csv", low_memory=False)
    membership = pd.read_csv(args.task / "inputs/T7_unique_design_membership.csv", usecols=fields)
    if len(membership) != 438768 or membership.recommendation_group_id.nunique() != 39:
        raise ValueError("Unexpected T7 candidate denominator")
    group_scores = score_models(scores, coverage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_scores.to_csv(args.output_dir / "T7_unique_design_model_group_scores.csv", index=False)
    if args.random_replicates:
        random_distribution, random_group, oracle, _ = random_and_oracle_baselines(
            membership, args.random_replicates, 20260703
        )
        random_group.to_csv(args.output_dir / "T7_unique_design_random_group_scores.csv", index=False)
        random_distribution.to_csv(args.output_dir / "T7_unique_design_random_metric_distribution.csv", index=False)
        oracle.to_csv(args.output_dir / "T7_unique_design_oracle_group_scores.csv", index=False)


if __name__ == "__main__":
    main()
