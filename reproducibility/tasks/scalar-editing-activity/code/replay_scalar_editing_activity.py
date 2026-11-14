"""Recompute the T1 scalar associations from the frozen V8 common panel."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


MODELS = ["CRISPRon-BE", "DeepBE", "FORECasT-BE", "DeepBaseEditor"]


MIN_STRATUM_N = 500


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    return float(spearmanr(x, y).statistic), float(pearsonr(x, y).statistic), len(x)


def primary_metrics(common_long):
    rows = []
    for model in MODELS:
        part = common_long[common_long["display_name"].eq(model)]
        sp, pe, n = correlation(part["measured_total_efficiency"], part["prediction_for_association"])
        rows.append({"model": model, "analysis_level": "observation", "n": n, "spearman": sp, "pearson": pe})
        design = part.groupby("design_id", as_index=False).agg(
            measured_total_efficiency=("measured_total_efficiency", "mean"),
            prediction_for_association=("prediction_for_association", "mean"),
        )
        sp, pe, n = correlation(design["measured_total_efficiency"], design["prediction_for_association"])
        rows.append({"model": model, "analysis_level": "design_mean", "n": n, "spearman": sp, "pearson": pe})
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(common_labels, common_long, reps, seed):
    pivot = common_long.pivot(index="observation_id", columns="display_name", values="prediction_for_association")
    base = common_labels.set_index("observation_id")[["design_id", "measured_total_efficiency"]].join(pivot, how="inner").dropna().reset_index()
    codes, designs = pd.factorize(base["design_id"], sort=False)
    groups = [np.flatnonzero(codes == index) for index in range(len(designs))]
    y = base["measured_total_efficiency"].to_numpy(float)
    predictions = {model: base[model].to_numpy(float) for model in MODELS}
    point = {model: float(spearmanr(y, predictions[model]).statistic) for model in MODELS}
    rng = np.random.default_rng(seed)
    estimates = np.empty((reps, len(MODELS)), float)
    for rep in range(reps):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        for col, model in enumerate(MODELS):
            estimates[rep, col] = spearmanr(y[indices], predictions[model][indices]).statistic
    ci_rows = []
    for col, model in enumerate(MODELS):
        ci_rows.append({
            "model": model, "spearman": point[model], "ci_low": np.percentile(estimates[:, col], 2.5),
            "ci_high": np.percentile(estimates[:, col], 97.5), "n_observations": len(base),
            "n_designs": len(designs), "bootstrap_replicates": reps, "seed": seed,
        })
    pair_rows = []
    for first, second in combinations(MODELS, 2):
        i, j = MODELS.index(first), MODELS.index(second)
        diffs = estimates[:, i] - estimates[:, j]
        pair_rows.append({
            "model_1": first, "model_2": second, "spearman_difference_model1_minus_model2": point[first]-point[second],
            "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
            "fraction_bootstrap_difference_gt_0": float(np.mean(diffs > 0)), "bootstrap_replicates": reps,
            "n_observations": len(base), "n_designs": len(designs),
        })
    rep_frame = pd.DataFrame(estimates, columns=MODELS)
    rep_frame.insert(0, "bootstrap_replicate", np.arange(1, reps+1))
    return pd.DataFrame(ci_rows), pd.DataFrame(pair_rows), rep_frame


def metadata_strata(common_long, common_labels):
    eligible = []
    for field in ["editor", "cell_context"]:
        counts = common_labels[field].fillna("NA").astype(str).value_counts()
        for level, count in counts.items():
            if count >= MIN_STRATUM_N:
                eligible.append((field, level, int(count)))
    rows = []
    for field, level, count in eligible:
        for model in MODELS:
            part = common_long[(common_long[field].fillna("NA").astype(str).eq(level)) & common_long["display_name"].eq(model)]
            sp, pe, n = correlation(part["measured_total_efficiency"], part["prediction_for_association"])
            field_label = {"editor": "Editor", "cell_context": "Cell context"}[field]
            rows.append({"metadata_field": field, "level": level, "stratum_label": f"{field_label}: {level}", "model": model, "n": n, "spearman": sp, "pearson": pe, "minimum_n_predeclared": MIN_STRATUM_N})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    labels = pd.read_csv(args.task / "inputs/T1_V8_COMMON_VALID_LABELS.csv.gz", low_memory=False)
    scores = pd.read_csv(args.task / "inputs/T1_V8_COMMON_VALID_MODEL_OBSERVATIONS.csv.gz", low_memory=False)
    if len(labels) != 10000 or len(scores) != 40000:
        raise ValueError("Unexpected frozen T1 denominator")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_metrics(scores).to_csv(args.output_dir / "PRIMARY_ASSOCIATION_METRICS.csv", index=False)
    metadata_strata(scores, labels).to_csv(args.output_dir / "METADATA_STRATIFIED_ASSOCIATION.csv", index=False)
    boot, pairs, reps = paired_cluster_bootstrap(labels, scores, 2000, 20260713)
    boot.to_csv(args.output_dir / "PRIMARY_DESIGN_CLUSTER_BOOTSTRAP_95CI.csv", index=False)
    pairs.to_csv(args.output_dir / "PAIRED_SPEARMAN_DIFFERENCE_BOOTSTRAP.csv", index=False)
    reps.to_csv(args.output_dir / "PRIMARY_PAIRED_BOOTSTRAP_REPLICATES.csv.gz", index=False)


if __name__ == "__main__":
    main()
