"""Recompute the T1 V8.1 predicted-efficiency calibration and strata."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


MIN_STRATUM_N = 500


def load_analysis(v8: Path) -> pd.DataFrame:
    labels = pd.read_csv(v8 / "data" / "T1_V8_LABEL_BLIND_ANALYSIS_PANEL.csv.gz", low_memory=False)
    pred = pd.read_csv(v8 / "predictions" / "harmonized" / "crispron_be_harmonized.csv", low_memory=False)
    cols = ["observation_id", "design_id", "measured_total_efficiency", "editor", "cell_context"]
    analysis = labels[cols].merge(
        pred[["observation_id", "predicted_efficiency", "prediction_for_association"]],
        on="observation_id", how="inner", validate="one_to_one",
    )
    analysis = analysis.dropna(subset=["measured_total_efficiency", "predicted_efficiency", "design_id"]).copy()
    if len(analysis) != 10000 or analysis["observation_id"].nunique() != 10000:
        raise RuntimeError(f"Expected 10,000 unique CRISPRon-BE rows, found {len(analysis):,}")
    return analysis


def absolute_calibration(analysis: pd.DataFrame, v8: Path, out: Path, bootstrap_replicates: int, seed: int):
    observed = analysis["measured_total_efficiency"].to_numpy(float)
    predicted = analysis["predicted_efficiency"].to_numpy(float)
    error = predicted - observed
    slope, intercept = np.polyfit(predicted, observed, 1)
    summary = pd.DataFrame([{
        "model": "CRISPRon-BE",
        "n": len(analysis),
        "spearman_rank_association": spearmanr(observed, predicted).statistic,
        "pearson_linear_association": pearsonr(observed, predicted).statistic,
        "rmse": np.sqrt(np.mean(error ** 2)),
        "mae": np.mean(np.abs(error)),
        "r2_absolute_scale": 1 - np.sum(error ** 2) / np.sum((observed - observed.mean()) ** 2),
        "calibration_slope_observed_on_predicted": slope,
        "calibration_intercept_observed_on_predicted": intercept,
        "calibration_regression_definition": "observed_efficiency = intercept + slope * predicted_efficiency",
    }])

    qcat, edges = pd.qcut(
        analysis["predicted_efficiency"], q=10, retbins=True, duplicates="drop"
    )
    analysis = analysis.copy()
    analysis["predicted_efficiency_decile"] = qcat.cat.codes + 1
    analysis["predicted_efficiency_interval"] = qcat.astype(str)
    actual_bins = int(analysis["predicted_efficiency_decile"].nunique())
    if actual_bins < 2:
        raise RuntimeError("Predicted-efficiency quantile binning produced fewer than two bins")

    grouped = analysis.groupby("predicted_efficiency_decile", sort=True, observed=True).agg(
        n=("observation_id", "size"),
        n_designs=("design_id", "nunique"),
        mean_predicted_efficiency=("predicted_efficiency", "mean"),
        mean_observed_efficiency=("measured_total_efficiency", "mean"),
        min_predicted_efficiency=("predicted_efficiency", "min"),
        max_predicted_efficiency=("predicted_efficiency", "max"),
        interval=("predicted_efficiency_interval", "first"),
    ).reset_index()

    design_ids = pd.Index(analysis["design_id"].drop_duplicates())
    design_index = {design: i for i, design in enumerate(design_ids)}
    sum_matrix = np.zeros((len(design_ids), actual_bins), dtype=float)
    count_matrix = np.zeros((len(design_ids), actual_bins), dtype=np.int32)
    for row in analysis[["design_id", "predicted_efficiency_decile", "measured_total_efficiency"]].itertuples(index=False):
        i = design_index[row.design_id]
        j = int(row.predicted_efficiency_decile) - 1
        sum_matrix[i, j] += float(row.measured_total_efficiency)
        count_matrix[i, j] += 1

    rng = np.random.default_rng(seed)
    boot = np.full((bootstrap_replicates, actual_bins), np.nan, dtype=float)
    for replicate in range(bootstrap_replicates):
        sampled = rng.integers(0, len(design_ids), size=len(design_ids))
        sums = sum_matrix[sampled].sum(axis=0)
        counts = count_matrix[sampled].sum(axis=0)
        boot[replicate] = np.divide(sums, counts, out=np.full(actual_bins, np.nan), where=counts > 0)
    grouped["observed_efficiency_ci_low"] = np.nanquantile(boot, 0.025, axis=0)
    grouped["observed_efficiency_ci_high"] = np.nanquantile(boot, 0.975, axis=0)
    grouped["ci_method"] = "design-cluster bootstrap of mean observed efficiency within fixed predicted-efficiency bins"
    grouped["bootstrap_replicates"] = bootstrap_replicates
    grouped["bootstrap_seed"] = seed
    grouped["requested_quantile_bins"] = 10
    grouped["actual_quantile_bins"] = actual_bins

    old = pd.read_csv(v8 / "metrics" / "CRISPRON_ABSOLUTE_CALIBRATION_SUMMARY.csv").iloc[0]
    checks = [
        ("calibration_slope_observed_on_predicted", summary.iloc[0]["calibration_slope_observed_on_predicted"], old["calibration_slope_observed_on_predicted"]),
        ("calibration_intercept_observed_on_predicted", summary.iloc[0]["calibration_intercept_observed_on_predicted"], old["calibration_intercept_observed_on_predicted"]),
        ("rmse", summary.iloc[0]["rmse"], old["rmse"]),
        ("mae", summary.iloc[0]["mae"], old["mae"]),
        ("r2_absolute_scale", summary.iloc[0]["r2_absolute_scale"], old["r2"]),
    ]
    validation = pd.DataFrame([
        {"metric": name, "v8_1_recomputed": new, "v8_reported": prior,
         "absolute_difference": abs(new - prior), "pass_tolerance_1e-12": abs(new - prior) <= 1e-12}
        for name, new, prior in checks
    ])
    if not validation["pass_tolerance_1e-12"].all():
        raise RuntimeError("Independent calibration validation did not reproduce the V8 summary")

    summary.to_csv(out / "metrics" / "CRISPRON_ABSOLUTE_CALIBRATION_SUMMARY.csv", index=False)
    grouped.to_csv(out / "metrics" / "CRISPRON_PREDICTED_EFFICIENCY_DECILE_CALIBRATION.csv", index=False)
    validation.to_csv(out / "metrics" / "CRISPRON_CALIBRATION_NUMERIC_VALIDATION.csv", index=False)
    analysis.to_csv(out / "data" / "T1_V8_1_CRISPRON_CALIBRATION_ANALYSIS.csv.gz", index=False, compression="gzip")
    return summary, grouped, analysis, validation


def strata_audit(analysis: pd.DataFrame, v8: Path, out: Path):
    strata = pd.read_csv(v8 / "metrics" / "METADATA_STRATIFIED_ASSOCIATION.csv")
    eligible_rows, excluded_rows = [], []
    for field, group_label in [("cell_context", "Cell context"), ("editor", "Editor")]:
        counts = analysis[field].fillna("NA").astype(str).value_counts()
        for level, n in counts.items():
            row = {"metadata_field": field, "group_label": group_label, "level": level, "n": int(n), "minimum_n_predeclared": MIN_STRATUM_N}
            (eligible_rows if n >= MIN_STRATUM_N else excluded_rows).append(row)
    eligible = pd.DataFrame(eligible_rows)
    excluded = pd.DataFrame(excluded_rows, columns=eligible.columns)
    expected = set(zip(eligible["metadata_field"], eligible["level"]))
    represented = set(zip(strata["metadata_field"], strata["level"]))
    if expected != represented:
        raise RuntimeError(f"V8 strata mismatch: eligible={expected}, represented={represented}")

    cell_order = [x for x in ["HEK293T", "U2OS", "mES"] if ("cell_context", x) in expected]
    editor_order = [x for x in ["ABE", "ABE-CP1040"] if ("editor", x) in expected]
    order = [("cell_context", x) for x in cell_order] + [("editor", x) for x in editor_order]
    order_map = {key: i for i, key in enumerate(order)}
    n_map = {(r.metadata_field, r.level): int(r.n) for r in eligible.itertuples()}
    strata = strata.copy()
    strata["display_order"] = [order_map[(a, b)] for a, b in zip(strata.metadata_field, strata.level)]
    strata["group_label"] = strata["metadata_field"].map({"cell_context": "Cell context", "editor": "Editor"})
    strata["display_label"] = [f"{level}\n(n={n_map[(field, level)]:,})" for field, level in zip(strata.metadata_field, strata.level)]
    strata = strata.sort_values(["display_order", "model"]).reset_index(drop=True)
    eligible["display_order"] = [order_map[(a, b)] for a, b in zip(eligible.metadata_field, eligible.level)]
    eligible = eligible.sort_values("display_order").reset_index(drop=True)
    eligible.to_csv(out / "metrics" / "T1_eligible_metadata_strata.csv", index=False)
    excluded.to_csv(out / "metrics" / "T1_excluded_metadata_strata.csv", index=False)
    return strata, eligible, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for folder in ("metrics", "data"):
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    analysis = load_analysis(args.v8_root)
    absolute_calibration(analysis, args.v8_root, args.output_dir, 2000, 20260713)
    strata_audit(analysis, args.v8_root, args.output_dir)


if __name__ == "__main__":
    main()
