from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


MODELS = ["CRISPRon-BE", "DeepBE", "FORECasT-BE", "DeepBaseEditor"]
PRIMARY_SEED = 20260730
TOPK_SEED = 20260731
REPS = 2000
FRACTIONS = (0.01, 0.05, 0.10)
TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"E:\AIselect\BEs\benchmark_core\formal_results"
            r"\T4_high_low_classification_v2_1_context_unit_ci_revision"
        ),
    )
    return parser.parse_args()


def interval(values: pd.Series | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def add_exact_check(
    checks: list[dict[str, object]],
    check_id: str,
    category: str,
    expected: object,
    observed: object,
    note: str = "",
) -> None:
    status = "PASS" if expected == observed else "FAIL"
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "expected": expected,
            "observed": observed,
            "absolute_difference": (
                abs(float(observed) - float(expected))
                if isinstance(expected, (int, float, np.integer, np.floating))
                and isinstance(observed, (int, float, np.integer, np.floating))
                else np.nan
            ),
            "tolerance": 0,
            "status": status,
            "note": note,
        }
    )


def add_numeric_check(
    checks: list[dict[str, object]],
    check_id: str,
    category: str,
    expected: float,
    observed: float,
    tolerance: float = TOLERANCE,
    note: str = "",
) -> None:
    difference = abs(float(observed) - float(expected))
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "expected": float(expected),
            "observed": float(observed),
            "absolute_difference": difference,
            "tolerance": tolerance,
            "status": "PASS" if difference <= tolerance else "FAIL",
            "note": note,
        }
    )


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "units": root / "source_data" / "T4_V2_1_CONTEXT_UNITS_JOINED.csv.gz",
        "observations": (
            root
            / "source_data"
            / "T4_V2_1_OBSERVATION_LEVEL_JOINED_SENSITIVITY.csv.gz"
        ),
        "reference": (
            root / "source_data" / "T4_V2_1_REFERENCE_CONTEXT_UNITS.csv.gz"
        ),
        "thresholds": (
            root / "source_data" / "T4_V2_1_FROZEN_LABEL_THRESHOLDS.csv"
        ),
        "split": (
            root / "source_data" / "T4_V2_1_FROZEN_SCORE_SPLIT_ASSIGNMENT.csv"
        ),
        "primary": root / "tables" / "T4_V2_1_PRIMARY_CONTEXT_UNIT_METRICS.csv",
        "paired": root / "tables" / "T4_V2_1_PAIRED_BOOTSTRAP_DIFFERENCES.csv",
        "topk": root / "tables" / "T4_V2_1_TOPK_CONTEXT_UNIT_METRICS.csv",
        "construction": (
            root / "tables" / "T4_V2_1_CONTEXT_UNIT_CONSTRUCTION_SUMMARY.csv"
        ),
        "coverage": root / "tables" / "T4_V2_1_PREDICTION_COVERAGE.csv",
        "main_primary_reps": (
            root
            / "source_data"
            / "T4_V2_1_PRIMARY_CLUSTER_BOOTSTRAP_REPLICATES.csv.gz"
        ),
        "main_topk_reps": (
            root
            / "source_data"
            / "T4_V2_1_TOPK_CLUSTER_BOOTSTRAP_REPLICATES.csv.gz"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Verifier input missing: {missing}")
    return {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}


def verify_denominator_and_labels(
    tables: dict[str, pd.DataFrame],
    checks: list[dict[str, object]],
) -> pd.DataFrame:
    units = tables["units"].copy()
    observations = tables["observations"].copy()
    reference = tables["reference"].copy()
    thresholds = tables["thresholds"].copy()
    split = tables["split"].copy()

    add_exact_check(checks, "denominator.observations", "denominator", 10000, len(observations))
    add_exact_check(checks, "denominator.context_units", "denominator", 9963, len(units))
    add_exact_check(
        checks,
        "denominator.designs",
        "denominator",
        7987,
        units["design_id"].nunique(),
    )
    add_exact_check(
        checks,
        "denominator.observation_sum",
        "denominator",
        10000,
        int(units["observation_count"].sum()),
    )
    add_exact_check(
        checks,
        "denominator.repeated_units",
        "denominator",
        37,
        int(units["observation_count"].gt(1).sum()),
    )
    add_exact_check(
        checks,
        "denominator.excess_observations",
        "denominator",
        37,
        int((units["observation_count"] - 1).sum()),
    )
    add_exact_check(
        checks,
        "unit_id.uniqueness",
        "denominator",
        0,
        int(units["unit_id"].duplicated().sum()),
    )

    for model in MODELS:
        add_exact_check(
            checks,
            f"coverage.observation.{model}",
            "coverage",
            10000,
            int(observations[model].notna().sum()),
        )
        add_exact_check(
            checks,
            f"coverage.context_unit.{model}",
            "coverage",
            9963,
            int(units[model].notna().sum()),
        )

    reference_overlap = set(reference["design_id"].astype(str)) & set(
        units["design_id"].astype(str)
    )
    add_exact_check(
        checks,
        "leakage.reference_test_design_overlap",
        "leakage",
        0,
        len(reference_overlap),
    )
    calibration = set(
        split.loc[
            split["score_threshold_split"].eq("calibration_20pct_designs"),
            "design_id",
        ].astype(str)
    )
    heldout = set(
        split.loc[
            split["score_threshold_split"].eq("heldout_80pct_designs"),
            "design_id",
        ].astype(str)
    )
    add_exact_check(
        checks,
        "leakage.calibration_heldout_design_overlap",
        "leakage",
        0,
        len(calibration & heldout),
    )

    recomputed_thresholds = (
        reference.groupby(["editor", "cell_context"], as_index=False)
        .agg(
            recomputed_q25=(
                "measured_efficiency_reference_mean",
                lambda values: values.quantile(0.25),
            ),
            recomputed_q75=(
                "measured_efficiency_reference_mean",
                lambda values: values.quantile(0.75),
            ),
        )
        .sort_values(["editor", "cell_context"])
    )
    frozen = thresholds.merge(
        recomputed_thresholds,
        on=["editor", "cell_context"],
        how="outer",
        validate="one_to_one",
    )
    for _, row in frozen.iterrows():
        key = f"{row['editor']}|{row['cell_context']}"
        add_numeric_check(
            checks,
            f"threshold.q25.{key}",
            "label_threshold",
            row["reference_q25"],
            row["recomputed_q25"],
        )
        add_numeric_check(
            checks,
            f"threshold.q75.{key}",
            "label_threshold",
            row["reference_q75"],
            row["recomputed_q75"],
        )

    relabel = units.drop(
        columns=[
            column
            for column in [
                "reference_q25",
                "reference_median",
                "reference_q75",
                "reference_stratum_eligible",
                "v2_stratum_eligible",
            ]
            if column in units
        ]
    ).merge(
        thresholds[
            [
                "editor",
                "cell_context",
                "reference_q25",
                "reference_q75",
                "reference_stratum_eligible",
                "v2_stratum_eligible",
            ]
        ],
        on=["editor", "cell_context"],
        how="left",
        validate="many_to_one",
    )
    eligible = (
        relabel["reference_stratum_eligible"].fillna(False)
        & relabel["v2_stratum_eligible"].fillna(False)
    )
    relabel["independent_class"] = np.select(
        [
            eligible
            & relabel["measured_efficiency_mean"].le(relabel["reference_q25"]),
            eligible
            & relabel["measured_efficiency_mean"].ge(relabel["reference_q75"]),
            eligible,
        ],
        ["low", "high", "middle"],
        default="excluded",
    )
    mismatch = int(
        (
            relabel["independent_class"].astype(str)
            != units["v2_1_class"].astype(str)
        ).sum()
    )
    add_exact_check(
        checks,
        "labels.rowwise_class_mismatch",
        "labels",
        0,
        mismatch,
    )
    counts = relabel["independent_class"].value_counts()
    for label, expected in [("low", 2791), ("middle", 4855), ("high", 2317)]:
        add_exact_check(
            checks,
            f"labels.count.{label}",
            "labels",
            expected,
            int(counts.get(label, 0)),
        )
    clean = relabel[relabel["independent_class"].isin(["low", "high"])].copy()
    clean["independent_clean_label"] = clean["independent_class"].eq("high").astype(int)
    add_exact_check(checks, "labels.clean_units", "labels", 5108, len(clean))
    add_exact_check(
        checks,
        "labels.clean_designs",
        "labels",
        4480,
        clean["design_id"].nunique(),
    )
    return clean


def explicit_primary_bootstrap(
    clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = clean.sort_values("unit_id").reset_index(drop=True)
    designs = np.array(sorted(clean["design_id"].astype(str).unique()))
    groups = [
        clean.index[clean["design_id"].astype(str).eq(design)].to_numpy()
        for design in designs
    ]
    rng = np.random.default_rng(PRIMARY_SEED)
    replicate_rows = []
    for replicate in range(1, REPS + 1):
        sampled_group_ids = rng.integers(0, len(groups), size=len(groups))
        sampled_rows = np.concatenate([groups[index] for index in sampled_group_ids])
        sampled = clean.iloc[sampled_rows]
        y = sampled["independent_clean_label"].astype(int).to_numpy()
        prevalence = float(y.mean())
        for model in MODELS:
            score = sampled[model].to_numpy(float)
            replicate_rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "model": model,
                    "n_sampled_context_units": len(sampled),
                    "baseline_prevalence": prevalence,
                    "auroc": float(roc_auc_score(y, score)),
                    "auprc": float(average_precision_score(y, score)),
                }
            )
    replicates = pd.DataFrame(replicate_rows)
    y_point = clean["independent_clean_label"].astype(int).to_numpy()
    baseline_values = replicates.loc[
        replicates["model"].eq(MODELS[0]), "baseline_prevalence"
    ]
    baseline_low, baseline_high = interval(baseline_values)
    point_rows = []
    for model in MODELS:
        subset = replicates[replicates["model"].eq(model)]
        auroc_low, auroc_high = interval(subset["auroc"])
        auprc_low, auprc_high = interval(subset["auprc"])
        point_rows.append(
            {
                "model": model,
                "baseline_prevalence": float(y_point.mean()),
                "baseline_prevalence_ci_low": baseline_low,
                "baseline_prevalence_ci_high": baseline_high,
                "auroc": float(roc_auc_score(y_point, clean[model].to_numpy(float))),
                "auroc_ci_low": auroc_low,
                "auroc_ci_high": auroc_high,
                "auprc": float(
                    average_precision_score(y_point, clean[model].to_numpy(float))
                ),
                "auprc_ci_low": auprc_low,
                "auprc_ci_high": auprc_high,
            }
        )
    return pd.DataFrame(point_rows), replicates


def verify_primary_and_pairs(
    tables: dict[str, pd.DataFrame],
    clean: pd.DataFrame,
    checks: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    independent, replicates = explicit_primary_bootstrap(clean)
    reported = tables["primary"].set_index("model")
    for _, row in independent.iterrows():
        model = row["model"]
        for metric in [
            "baseline_prevalence",
            "baseline_prevalence_ci_low",
            "baseline_prevalence_ci_high",
            "auroc",
            "auroc_ci_low",
            "auroc_ci_high",
            "auprc",
            "auprc_ci_low",
            "auprc_ci_high",
        ]:
            add_numeric_check(
                checks,
                f"primary.{metric}.{model}",
                "primary_metric",
                reported.loc[model, metric],
                row[metric],
            )

    add_exact_check(
        checks,
        "bootstrap.primary.independent_replicates",
        "bootstrap",
        REPS,
        replicates["bootstrap_replicate"].nunique(),
    )
    add_exact_check(
        checks,
        "bootstrap.primary.reported_replicates",
        "bootstrap",
        REPS,
        tables["main_primary_reps"]["bootstrap_replicate"].nunique(),
    )

    reported_pairs = tables["paired"].set_index(["metric", "model_1", "model_2"])
    point = independent.set_index("model")
    pair_rows = []
    for metric in ["auroc", "auprc"]:
        pivot = replicates.pivot(
            index="bootstrap_replicate", columns="model", values=metric
        )
        for model_1, model_2 in itertools.combinations(MODELS, 2):
            values = pivot[model_1] - pivot[model_2]
            low, high = interval(values)
            estimate = float(point.loc[model_1, metric] - point.loc[model_2, metric])
            pair_rows.append(
                {
                    "metric": metric,
                    "model_1": model_1,
                    "model_2": model_2,
                    "difference_model1_minus_model2": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                }
            )
            key = (metric, model_1, model_2)
            for field, observed in [
                ("difference_model1_minus_model2", estimate),
                ("ci_low", low),
                ("ci_high", high),
            ]:
                add_numeric_check(
                    checks,
                    f"paired.{metric}.{model_1}.{model_2}.{field}",
                    "paired_difference",
                    reported_pairs.loc[key, field],
                    observed,
                )
            add_exact_check(
                checks,
                f"paired.{metric}.{model_1}.{model_2}.ci_excludes_zero",
                "paired_difference",
                bool(reported_pairs.loc[key, "ci_excludes_zero"]),
                bool(low > 0 or high < 0),
            )
    return replicates, pd.DataFrame(pair_rows)


def independent_topk_point(
    units: pd.DataFrame,
    model: str,
    fraction: float,
) -> dict[str, float | int | str]:
    ranked = units.sort_values(
        [model, "unit_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    k = int(math.ceil(fraction * len(units)))
    selected = ranked.head(k)
    total_high = int(ranked["v2_1_high_vs_rest_label"].sum())
    selected_high = int(selected["v2_1_high_vs_rest_label"].sum())
    baseline = total_high / len(ranked)
    boundary = float(selected[model].iloc[-1])
    above = int(ranked[model].gt(boundary).sum())
    tie_count = int(ranked[model].eq(boundary).sum())
    return {
        "model": model,
        "top_fraction": fraction,
        "k": k,
        "baseline_prevalence": baseline,
        "precision": selected_high / k,
        "recall": selected_high / total_high,
        "lift": (selected_high / k) / baseline,
        "boundary_score": boundary,
        "boundary_tie_count": tie_count,
        "boundary_tie_selected": k - above,
        "boundary_tie_omitted": tie_count - (k - above),
    }


def explicit_topk_bootstrap(
    units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = units.sort_values("unit_id").reset_index(drop=True)
    designs = np.array(sorted(units["design_id"].astype(str).unique()))
    groups = [
        units.index[units["design_id"].astype(str).eq(design)].to_numpy()
        for design in designs
    ]
    k_by_fraction = {
        fraction: int(math.ceil(fraction * len(units))) for fraction in FRACTIONS
    }
    rng = np.random.default_rng(TOPK_SEED)
    rows = []
    for replicate in range(1, REPS + 1):
        sampled_group_ids = rng.integers(0, len(groups), size=len(groups))
        sampled_rows = np.concatenate([groups[index] for index in sampled_group_ids])
        sampled = units.iloc[sampled_rows].copy()
        baseline = float(sampled["v2_1_high_vs_rest_label"].mean())
        total_high = int(sampled["v2_1_high_vs_rest_label"].sum())
        for model in MODELS:
            ranked = sampled.sort_values(
                [model, "unit_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            for fraction, k in k_by_fraction.items():
                selected = ranked.head(k)
                selected_high = int(selected["v2_1_high_vs_rest_label"].sum())
                precision = selected_high / k
                recall = selected_high / total_high
                rows.append(
                    {
                        "bootstrap_replicate": replicate,
                        "model": model,
                        "top_fraction": fraction,
                        "n_sampled_context_units": len(sampled),
                        "baseline_prevalence": baseline,
                        "precision": precision,
                        "recall": recall,
                        "lift": precision / baseline,
                    }
                )
    replicates = pd.DataFrame(rows)
    points = pd.DataFrame(
        [
            independent_topk_point(units, model, fraction)
            for model in MODELS
            for fraction in FRACTIONS
        ]
    )
    baseline_values = replicates[
        replicates["model"].eq(MODELS[0])
        & replicates["top_fraction"].eq(FRACTIONS[0])
    ]["baseline_prevalence"]
    baseline_low, baseline_high = interval(baseline_values)
    points["baseline_prevalence_ci_low"] = baseline_low
    points["baseline_prevalence_ci_high"] = baseline_high
    for index, row in points.iterrows():
        subset = replicates[
            replicates["model"].eq(row["model"])
            & replicates["top_fraction"].eq(row["top_fraction"])
        ]
        for metric in ["precision", "recall", "lift"]:
            low, high = interval(subset[metric])
            points.loc[index, f"{metric}_ci_low"] = low
            points.loc[index, f"{metric}_ci_high"] = high
    return points, replicates


def verify_topk(
    tables: dict[str, pd.DataFrame],
    checks: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = tables["units"].copy()
    points, replicates = explicit_topk_bootstrap(units)
    reported = tables["topk"].set_index(["model", "top_fraction"])
    fields = [
        ("k", "k_predefined_from_context_unit_denominator", True),
        ("baseline_prevalence", "baseline_prevalence", False),
        ("baseline_prevalence_ci_low", "baseline_prevalence_ci_low", False),
        ("baseline_prevalence_ci_high", "baseline_prevalence_ci_high", False),
        ("precision", "precision", False),
        ("precision_ci_low", "precision_ci_low", False),
        ("precision_ci_high", "precision_ci_high", False),
        ("recall", "recall", False),
        ("recall_ci_low", "recall_ci_low", False),
        ("recall_ci_high", "recall_ci_high", False),
        ("lift", "lift", False),
        ("lift_ci_low", "lift_ci_low", False),
        ("lift_ci_high", "lift_ci_high", False),
        ("boundary_score", "boundary_score", False),
        ("boundary_tie_count", "boundary_tie_count", True),
        ("boundary_tie_selected", "boundary_tie_selected", True),
        ("boundary_tie_omitted", "boundary_tie_omitted", True),
    ]
    for _, row in points.iterrows():
        key = (row["model"], row["top_fraction"])
        for independent_field, reported_field, exact in fields:
            if exact:
                add_exact_check(
                    checks,
                    (
                        f"topk.{row['model']}.{row['top_fraction']}."
                        f"{independent_field}"
                    ),
                    "topk",
                    int(reported.loc[key, reported_field]),
                    int(row[independent_field]),
                )
            else:
                add_numeric_check(
                    checks,
                    (
                        f"topk.{row['model']}.{row['top_fraction']}."
                        f"{independent_field}"
                    ),
                    "topk",
                    reported.loc[key, reported_field],
                    row[independent_field],
                )
    add_exact_check(
        checks,
        "bootstrap.topk.independent_replicates",
        "bootstrap",
        REPS,
        replicates["bootstrap_replicate"].nunique(),
    )
    add_exact_check(
        checks,
        "bootstrap.topk.reported_replicates",
        "bootstrap",
        REPS,
        tables["main_topk_reps"]["bootstrap_replicate"].nunique(),
    )
    return points, replicates


def write_outputs(
    root: Path,
    checks: list[dict[str, object]],
    primary_reps: pd.DataFrame,
    paired: pd.DataFrame,
    topk_points: pd.DataFrame,
    topk_reps: pd.DataFrame,
    started: datetime,
) -> None:
    verification = root / "verification"
    verification.mkdir(parents=True, exist_ok=True)
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(
        verification / "T4_V2_1_STATISTICAL_VERIFICATION.csv",
        index=False,
    )
    primary_reps.to_csv(
        verification / "T4_V2_1_VERIFIER_PRIMARY_REPLICATES.csv.gz",
        index=False,
        compression="gzip",
    )
    paired.to_csv(
        verification / "T4_V2_1_VERIFIER_PAIRED_DIFFERENCES.csv",
        index=False,
    )
    topk_points.to_csv(
        verification / "T4_V2_1_VERIFIER_TOPK_METRICS.csv",
        index=False,
    )
    topk_reps.to_csv(
        verification / "T4_V2_1_VERIFIER_TOPK_REPLICATES.csv.gz",
        index=False,
        compression="gzip",
    )

    failures = checks_frame[~checks_frame["status"].eq("PASS")]
    summary = {
        "verification": "independent statistical recomputation",
        "independent_script_imports_main_analysis": False,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "checks": len(checks_frame),
        "passed": int(checks_frame["status"].eq("PASS").sum()),
        "failed": len(failures),
        "overall_status": "PASS" if failures.empty else "FAIL",
        "primary_bootstrap_replicates": int(
            primary_reps["bootstrap_replicate"].nunique()
        ),
        "topk_bootstrap_replicates": int(
            topk_reps["bootstrap_replicate"].nunique()
        ),
        "reference_test_design_overlap": 0,
        "strict_external_results": 0,
    }
    (verification / "T4_V2_1_STATISTICAL_VERIFICATION.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    by_category = (
        checks_frame.groupby("category")["status"]
        .agg(total="size", passed=lambda values: values.eq("PASS").sum())
        .reset_index()
    )
    report_lines = [
        "# T4 V2.1 independent statistical QA report",
        "",
        f"- Overall status: **{summary['overall_status']}**",
        f"- Checks: {summary['checks']} total; {summary['passed']} passed; {summary['failed']} failed.",
        "- Verifier implementation is separate and does not import the main analysis script.",
        "- Verifier reads frozen V2.1 joined/source tables and independently rebuilds labels and statistics.",
        "- Primary and top-k intervals were recomputed from 2,000 explicit design-cluster resamples.",
        "- Reference/test design overlap: 0.",
        "- Calibration/held-out design overlap: 0.",
        "- Strict-external results: 0.",
        "",
        "## Checks by category",
        "",
        "| Category | Total | Passed |",
        "| --- | ---: | ---: |",
    ]
    report_lines.extend(
        f"| {row['category']} | {int(row['total'])} | {int(row['passed'])} |"
        for _, row in by_category.iterrows()
    )
    if not failures.empty:
        report_lines.extend(
            [
                "",
                "## Failures",
                "",
                failures.to_markdown(index=False),
            ]
        )
    (root / "T4_FIG5_V2_1_INDEPENDENT_QA_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not failures.empty:
        raise RuntimeError(f"Independent QA failed {len(failures)} checks")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    started = datetime.now(timezone.utc)
    tables = load_tables(root)
    checks: list[dict[str, object]] = []
    clean = verify_denominator_and_labels(tables, checks)
    primary_reps, paired = verify_primary_and_pairs(tables, clean, checks)
    topk_points, topk_reps = verify_topk(tables, checks)
    write_outputs(
        root,
        checks,
        primary_reps,
        paired,
        topk_points,
        topk_reps,
        started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
