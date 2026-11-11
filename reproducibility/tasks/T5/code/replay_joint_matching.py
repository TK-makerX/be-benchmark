"""Recompute the T5 joint matched-unit contrasts from the frozen T1 scores."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BOOTSTRAP_REPLICATES = 2000


MATERIAL_TOLERANCE = 1e-6


NUMERICAL_TOLERANCE = 1e-12


MODEL_ORDER = [
    "crispron_be",
    "deepbe",
    "forecast_be",
    "deepbaseeditor",
]


MODEL_DISPLAY = {
    "crispron_be": "CRISPRon-BE",
    "deepbe": "DeepBE",
    "forecast_be": "FORECasT-BE",
    "deepbaseeditor": "DeepBaseEditor",
}


MODEL_RELATION = {
    "crispron_be": "non-source-associated",
    "deepbe": "unknown-training-overlap",
    "forecast_be": "unknown-training-overlap",
    "deepbaseeditor": "unknown-training-overlap",
}


SPECS = [
    {
        "contrast_id": "T1_editor_ABE_to_ABE-CP1040",
        "short_label": "ABE→ABE-CP1040",
        "axis_name": "editor",
        "axis_label": "Editor",
        "reference_level": "ABE",
        "shift_level": "ABE-CP1040",
        "control_axis": "cell_context",
        "match_key": "design_id x cell_context",
        "seed": 202607301,
        "minimum_joint_units": 200,
        "expected_shared_designs": 1193,
        "expected_matched_designs": 655,
        "expected_unmatched_shared_designs": 538,
    },
    {
        "contrast_id": "T1_cell_HEK293T_to_U2OS",
        "short_label": "HEK293T→U2OS",
        "axis_name": "cell_context",
        "axis_label": "Cell context",
        "reference_level": "HEK293T",
        "shift_level": "U2OS",
        "control_axis": "editor",
        "match_key": "design_id x editor",
        "seed": 202607302,
        "minimum_joint_units": 200,
        "expected_shared_designs": 361,
        "expected_matched_designs": 206,
        "expected_unmatched_shared_designs": 155,
    },
    {
        "contrast_id": "T1_cell_HEK293T_to_mES",
        "short_label": "HEK293T→mES",
        "axis_name": "cell_context",
        "axis_label": "Cell context",
        "reference_level": "HEK293T",
        "shift_level": "mES",
        "control_axis": "editor",
        "match_key": "design_id x editor",
        "seed": 202607303,
        "minimum_joint_units": 200,
        "expected_shared_designs": 749,
        "expected_matched_designs": 442,
        "expected_unmatched_shared_designs": 307,
    },
]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def relation_for_model(model_id: str) -> str:
    return MODEL_RELATION[model_id]


def make_unit_id(design_id: str, control_axis: str, control_level: str) -> str:
    return f"{design_id}||{control_axis}={control_level}"


def audit_matching(
    panel: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[
    dict[str, object],
    set[tuple[str, str]],
    pd.DataFrame,
    pd.DataFrame,
]:
    axis = str(spec["axis_name"])
    control = str(spec["control_axis"])
    ref = str(spec["reference_level"])
    shift = str(spec["shift_level"])
    work = panel.loc[
        panel[axis].isin([ref, shift]),
        ["observation_id", "design_id", axis, control],
    ].dropna()
    work["design_id"] = work["design_id"].astype(str)
    work[axis] = work[axis].astype(str)
    work[control] = work[control].astype(str)

    ref_designs = set(work.loc[work[axis].eq(ref), "design_id"])
    shift_designs = set(work.loc[work[axis].eq(shift), "design_id"])
    shared_designs = ref_designs & shift_designs

    counts = (
        work.groupby(["design_id", control, axis], observed=True)
        .size()
        .unstack(axis, fill_value=0)
    )
    for level in [ref, shift]:
        if level not in counts:
            counts[level] = 0
    counts = counts.reset_index()
    counts["present_reference"] = counts[ref].gt(0)
    counts["present_shift"] = counts[shift].gt(0)
    counts["eligible_joint_unit"] = (
        counts["present_reference"] & counts["present_shift"]
    )
    counts["design_shared_between_levels"] = counts["design_id"].isin(shared_designs)
    matched_design_set = set(
        counts.loc[counts["eligible_joint_unit"], "design_id"].astype(str)
    )
    counts["design_has_any_joint_match"] = counts["design_id"].isin(
        matched_design_set
    )
    counts["unit_id"] = [
        make_unit_id(str(d), control, str(c))
        for d, c in zip(counts["design_id"], counts[control])
    ]

    def exclusion_reason(row: pd.Series) -> str:
        if bool(row["eligible_joint_unit"]):
            return ""
        if str(row["design_id"]) not in ref_designs:
            return "design_absent_from_reference_level"
        if str(row["design_id"]) not in shift_designs:
            return "design_absent_from_shift_level"
        if not bool(row["design_has_any_joint_match"]):
            return f"shared_design_without_common_{control}"
        if not bool(row["present_reference"]):
            return "control_level_absent_from_reference"
        return "control_level_absent_from_shift"

    counts["exclusion_reason"] = counts.apply(exclusion_reason, axis=1)
    counts.insert(0, "contrast_id", str(spec["contrast_id"]))
    counts.insert(1, "axis_name", axis)
    counts.insert(2, "reference_level", ref)
    counts.insert(3, "shift_level", shift)
    counts.insert(4, "control_axis", control)
    counts = counts.rename(
        columns={
            ref: "n_reference_observations",
            shift: "n_shift_observations",
        }
    )

    eligible = counts.loc[counts["eligible_joint_unit"]].copy()
    excluded = counts.loc[~counts["eligible_joint_unit"]].copy()
    unit_set = set(
        zip(
            eligible["design_id"].astype(str),
            eligible[control].astype(str),
        )
    )

    n_ref_eligible_obs = int(eligible["n_reference_observations"].sum())
    n_shift_eligible_obs = int(eligible["n_shift_observations"].sum())
    summary = {
        "task_id": "T1",
        "endpoint": "scalar_efficiency",
        "contrast_id": str(spec["contrast_id"]),
        "axis_name": axis,
        "axis_label": str(spec["axis_label"]),
        "reference_level": ref,
        "shift_level": shift,
        "control_axis": control,
        "joint_match_key": str(spec["match_key"]),
        "bootstrap_unit": str(spec["match_key"]),
        "n_reference_observations_all": int(work[axis].eq(ref).sum()),
        "n_shift_observations_all": int(work[axis].eq(shift).sum()),
        "n_reference_designs": len(ref_designs),
        "n_shift_designs": len(shift_designs),
        "n_shared_designs": len(shared_designs),
        "n_designs_with_joint_match": len(matched_design_set),
        "n_shared_designs_without_joint_match": (
            len(shared_designs) - len(matched_design_set)
        ),
        "n_joint_matched_units": len(unit_set),
        "n_reference_eligible_observations": n_ref_eligible_obs,
        "n_shift_eligible_observations": n_shift_eligible_obs,
        "eligibility_threshold_joint_units": int(spec["minimum_joint_units"]),
        "eligible_primary": len(unit_set) >= int(spec["minimum_joint_units"]),
        "aggregation_rule": (
            "mean measured efficiency and mean model score within "
            "joint unit x contrast level"
        ),
        "claim_scope": "within-BE-Hive fixed-sequence ranking transferability",
        "strict_external": False,
    }

    expected = {
        "n_shared_designs": int(spec["expected_shared_designs"]),
        "n_designs_with_joint_match": int(spec["expected_matched_designs"]),
        "n_shared_designs_without_joint_match": int(
            spec["expected_unmatched_shared_designs"]
        ),
    }
    for field, value in expected.items():
        if int(summary[field]) != value:
            raise AssertionError(
                f"{spec['contrast_id']} {field}: observed {summary[field]}, expected {value}"
            )
    if not bool(summary["eligible_primary"]):
        raise AssertionError(f"{spec['contrast_id']} failed the frozen n threshold")
    return summary, unit_set, eligible, excluded


def aggregate_joint_units(
    common: pd.DataFrame,
    spec: dict[str, object],
    unit_set: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    axis = str(spec["axis_name"])
    control = str(spec["control_axis"])
    ref = str(spec["reference_level"])
    shift = str(spec["shift_level"])
    columns = [
        "observation_id",
        "model_id",
        "design_id",
        axis,
        control,
        "measured_total_efficiency",
        "prediction_for_association",
    ]
    work = common.loc[common[axis].isin([ref, shift]), columns].dropna().copy()
    work["design_id"] = work["design_id"].astype(str)
    work[axis] = work[axis].astype(str)
    work[control] = work[control].astype(str)
    keep = [
        (design_id, control_level) in unit_set
        for design_id, control_level in zip(work["design_id"], work[control])
    ]
    work = work.loc[keep].copy()
    work["measured_total_efficiency"] = pd.to_numeric(
        work["measured_total_efficiency"], errors="coerce"
    )
    work["prediction_for_association"] = pd.to_numeric(
        work["prediction_for_association"], errors="coerce"
    )
    work = work.dropna(
        subset=["measured_total_efficiency", "prediction_for_association"]
    )

    grouped = (
        work.groupby(["model_id", "design_id", control, axis], observed=True)
        .agg(
            measured_efficiency_mean=("measured_total_efficiency", "mean"),
            model_score_mean=("prediction_for_association", "mean"),
            n_valid_observations=("observation_id", "nunique"),
            n_raw_rows=("observation_id", "size"),
            measured_efficiency_min=("measured_total_efficiency", "min"),
            measured_efficiency_max=("measured_total_efficiency", "max"),
            model_score_min=("prediction_for_association", "min"),
            model_score_max=("prediction_for_association", "max"),
        )
        .reset_index()
    )
    grouped["measured_efficiency_range"] = (
        grouped["measured_efficiency_max"] - grouped["measured_efficiency_min"]
    )
    grouped["model_score_range"] = (
        grouped["model_score_max"] - grouped["model_score_min"]
    )
    grouped["unit_id"] = [
        make_unit_id(str(d), control, str(c))
        for d, c in zip(grouped["design_id"], grouped[control])
    ]
    grouped.insert(0, "contrast_id", str(spec["contrast_id"]))
    grouped.insert(1, "axis_name", axis)
    grouped.insert(2, "reference_level", ref)
    grouped.insert(3, "shift_level", shift)
    grouped.insert(4, "control_axis", control)
    grouped["aggregation_rule"] = "arithmetic mean within joint unit x level"

    summaries = []
    for (model_id, level), frame in grouped.groupby(["model_id", axis], sort=True):
        summaries.append(
            {
                "contrast_id": str(spec["contrast_id"]),
                "model_id": model_id,
                "display_name": MODEL_DISPLAY[model_id],
                "axis_name": axis,
                "contrast_level": level,
                "n_joint_unit_level_groups": len(frame),
                "n_duplicate_groups": int(frame["n_valid_observations"].gt(1).sum()),
                "n_extra_observations_from_duplicates": int(
                    (frame["n_valid_observations"] - 1).clip(lower=0).sum()
                ),
                "max_observations_per_group": int(
                    frame["n_valid_observations"].max()
                ),
                "max_measured_efficiency_range": float(
                    frame["measured_efficiency_range"].max()
                ),
                "max_model_score_range": float(frame["model_score_range"].max()),
                "aggregation_rule": "mean measured efficiency; mean model score",
            }
        )
    duplicate_summary = pd.DataFrame(summaries)
    duplicate_detail = grouped.loc[grouped["n_valid_observations"].gt(1)].copy()
    return grouped, duplicate_summary, duplicate_detail


def run_joint_bootstrap(
    aggregated: pd.DataFrame,
    spec: dict[str, object],
    eligibility: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis = str(spec["axis_name"])
    ref = str(spec["reference_level"])
    shift = str(spec["shift_level"])
    seed = int(spec["seed"])
    unit_ids = sorted(aggregated["unit_id"].astype(str).unique())
    if len(unit_ids) != int(eligibility["n_joint_matched_units"]):
        raise AssertionError("Aggregated unit count does not match eligibility audit")

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, len(unit_ids), size=(BOOTSTRAP_REPLICATES, len(unit_ids))
    )
    sample_hashes = [
        hashlib.sha256(sample.astype(np.int32).tobytes()).hexdigest()[:20]
        for sample in sample_indices
    ]
    result_rows = []
    replicate_rows = []

    for model_id in MODEL_ORDER:
        model = aggregated.loc[aggregated["model_id"].eq(model_id)].copy()
        ref_frame = model.loc[model[axis].eq(ref)].set_index("unit_id").loc[unit_ids]
        shift_frame = (
            model.loc[model[axis].eq(shift)].set_index("unit_id").loc[unit_ids]
        )
        ref_measured = ref_frame["measured_efficiency_mean"].to_numpy(dtype=float)
        ref_score = ref_frame["model_score_mean"].to_numpy(dtype=float)
        shift_measured = shift_frame["measured_efficiency_mean"].to_numpy(
            dtype=float
        )
        shift_score = shift_frame["model_score_mean"].to_numpy(dtype=float)

        metric_ref = safe_spearman(ref_measured, ref_score)
        metric_shift = safe_spearman(shift_measured, shift_score)
        replicate_ref = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        replicate_shift = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for index, sampled in enumerate(sample_indices):
            replicate_ref[index] = safe_spearman(
                ref_measured[sampled], ref_score[sampled]
            )
            replicate_shift[index] = safe_spearman(
                shift_measured[sampled], shift_score[sampled]
            )
        replicate_difference = replicate_ref - replicate_shift
        ci_low, ci_high = np.quantile(replicate_difference, [0.025, 0.975])

        result_rows.append(
            {
                "task_id": "T1",
                "endpoint": "scalar_efficiency",
                "model_id": model_id,
                "display_name": MODEL_DISPLAY[model_id],
                "contrast_id": str(spec["contrast_id"]),
                "contrast_label": str(spec["short_label"]),
                "axis_name": axis,
                "axis_label": str(spec["axis_label"]),
                "reference_level": ref,
                "shift_level": shift,
                "control_axis": str(spec["control_axis"]),
                "joint_match_key": str(spec["match_key"]),
                "metric": "spearman",
                "metric_direction": "higher_is_better",
                "difference_definition": "metric_reference_minus_metric_shift",
                "metric_reference": metric_ref,
                "metric_shift": metric_shift,
                "reference_minus_shift": metric_ref - metric_shift,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_method": "paired joint-unit percentile bootstrap",
                "ci_level": 0.95,
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": seed,
                "bootstrap_unit": str(spec["match_key"]),
                "shared_sampling_across_models": True,
                "n_shared_designs_before_joint_matching": int(
                    eligibility["n_shared_designs"]
                ),
                "n_matched_designs": int(
                    eligibility["n_designs_with_joint_match"]
                ),
                "n_joint_matched_units": len(unit_ids),
                "n_reference_valid_observations": int(
                    ref_frame["n_valid_observations"].sum()
                ),
                "n_shift_valid_observations": int(
                    shift_frame["n_valid_observations"].sum()
                ),
                "n_reference_aggregated_units": len(ref_frame),
                "n_shift_aggregated_units": len(shift_frame),
                "aggregation_rule": (
                    "mean measured efficiency and mean model score within "
                    "joint unit x contrast level"
                ),
                "source_relation": relation_for_model(model_id),
                "strict_external": False,
                "claim_tier": "primary",
                "claim_scope": (
                    "within-BE-Hive fixed-sequence ranking transferability"
                ),
            }
        )
        for replicate_id in range(BOOTSTRAP_REPLICATES):
            replicate_rows.append(
                {
                    "task_id": "T1",
                    "model_id": model_id,
                    "display_name": MODEL_DISPLAY[model_id],
                    "contrast_id": str(spec["contrast_id"]),
                    "replicate_id": replicate_id + 1,
                    "bootstrap_seed": seed,
                    "n_sampled_joint_units": len(unit_ids),
                    "sample_index_sha256_prefix": sample_hashes[replicate_id],
                    "metric_reference": replicate_ref[replicate_id],
                    "metric_shift": replicate_shift[replicate_id],
                    "reference_minus_shift": replicate_difference[replicate_id],
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(replicate_rows)


def build_sensitivity(
    joint_results: pd.DataFrame,
    v1_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    old = v1_contrasts.loc[v1_contrasts["task_id"].eq("T1")].copy()
    keep = [
        "model_id",
        "contrast_id",
        "metric_reference",
        "metric_shift",
        "reference_minus_shift",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
        "n_shared_units",
        "bootstrap_estimator",
    ]
    old = old[keep].rename(
        columns={
            "metric_reference": "v1_design_only_metric_reference",
            "metric_shift": "v1_design_only_metric_shift",
            "reference_minus_shift": "v1_design_only_difference",
            "ci_low": "v1_design_only_ci_low",
            "ci_high": "v1_design_only_ci_high",
            "ci_excludes_zero": "v1_design_only_ci_excludes_zero",
            "n_shared_units": "v1_shared_designs",
            "bootstrap_estimator": "v1_bootstrap_estimator",
        }
    )
    new = joint_results[
        [
            "model_id",
            "display_name",
            "contrast_id",
            "contrast_label",
            "metric_reference",
            "metric_shift",
            "reference_minus_shift",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
            "n_matched_designs",
            "n_joint_matched_units",
        ]
    ].rename(
        columns={
            "metric_reference": "v1_1_joint_metric_reference",
            "metric_shift": "v1_1_joint_metric_shift",
            "reference_minus_shift": "v1_1_joint_difference",
            "ci_low": "v1_1_joint_ci_low",
            "ci_high": "v1_1_joint_ci_high",
            "ci_excludes_zero": "v1_1_joint_ci_excludes_zero",
        }
    )
    merged = new.merge(old, on=["model_id", "contrast_id"], how="left", validate="1:1")
    merged["joint_minus_design_only_difference"] = (
        merged["v1_1_joint_difference"] - merged["v1_design_only_difference"]
    )
    merged["v1_direction"] = np.where(
        merged["v1_design_only_difference"].gt(0), "positive", "negative"
    )
    merged["v1_1_direction"] = np.where(
        merged["v1_1_joint_difference"].gt(0), "positive", "negative"
    )
    merged["point_direction_changed"] = (
        merged["v1_direction"] != merged["v1_1_direction"]
    )
    merged["interval_evidence_changed"] = (
        merged["v1_design_only_ci_excludes_zero"].astype(bool)
        != merged["v1_1_joint_ci_excludes_zero"].astype(bool)
    )
    merged["conclusion_reversal"] = (
        merged["point_direction_changed"] & merged["interval_evidence_changed"]
    )
    merged["interpretation"] = np.select(
        [
            merged["conclusion_reversal"],
            merged["point_direction_changed"],
            merged["interval_evidence_changed"],
        ],
        [
            "direction and interval-evidence status changed",
            "point direction changed but both intervals include zero",
            "interval-evidence status changed without direction change",
        ],
        default="direction and interval-evidence status unchanged",
    )
    merged["v1_role"] = "shared-design-only sensitivity"
    merged["v1_1_role"] = "joint-matched primary"
    return merged


def build_context_invariance(
    common: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        axis = str(spec["axis_name"])
        ref = str(spec["reference_level"])
        shift = str(spec["shift_level"])
        panel_work = panel.loc[panel[axis].isin([ref, shift]), ["design_id", axis]]
        ref_designs = set(
            panel_work.loc[panel_work[axis].eq(ref), "design_id"].astype(str)
        )
        shift_designs = set(
            panel_work.loc[panel_work[axis].eq(shift), "design_id"].astype(str)
        )
        shared_designs = ref_designs & shift_designs
        work = common.loc[
            common[axis].isin([ref, shift])
            & common["design_id"].astype(str).isin(shared_designs),
            ["model_id", "design_id", axis, "prediction_for_association"],
        ].dropna()
        work["design_id"] = work["design_id"].astype(str)
        work["prediction_for_association"] = pd.to_numeric(
            work["prediction_for_association"], errors="coerce"
        )
        for model_id in MODEL_ORDER:
            model = work.loc[work["model_id"].eq(model_id)]
            ranges = model.groupby("design_id", observed=True)[
                "prediction_for_association"
            ].agg(lambda values: float(values.max() - values.min()))
            rows.append(
                {
                    "task_id": "T1",
                    "model_id": model_id,
                    "display_name": MODEL_DISPLAY[model_id],
                    "contrast_id": str(spec["contrast_id"]),
                    "contrast_label": str(spec["short_label"]),
                    "axis_name": axis,
                    "reference_level": ref,
                    "shift_level": shift,
                    "score_field": "prediction_for_association",
                    "n_shared_designs_evaluated": len(ranges),
                    "n_designs_with_exact_nonzero_range": int(ranges.gt(0).sum()),
                    "proportion_exact_nonzero_range": float(ranges.gt(0).mean()),
                    "n_designs_changed_gt_1e_12": int(
                        ranges.gt(NUMERICAL_TOLERANCE).sum()
                    ),
                    "proportion_changed_gt_1e_12": float(
                        ranges.gt(NUMERICAL_TOLERANCE).mean()
                    ),
                    "n_designs_materially_changed_gt_1e_6": int(
                        ranges.gt(MATERIAL_TOLERANCE).sum()
                    ),
                    "proportion_materially_changed_gt_1e_6": float(
                        ranges.gt(MATERIAL_TOLERANCE).mean()
                    ),
                    "maximum_absolute_score_range": float(ranges.max()),
                    "material_tolerance": MATERIAL_TOLERANCE,
                    "assessment": (
                        "context-invariant within 1e-6"
                        if int(ranges.gt(MATERIAL_TOLERANCE).sum()) == 0
                        else "material context variation detected"
                    ),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    task = Path(__file__).resolve().parents[1]
    parser.add_argument("--t1-inputs", type=Path, default=task.parent / "T1/inputs")
    parser.add_argument("--prior-contrasts", type=Path,
                        default=task / "inputs/prior_sensitivity/T5_REFERENCE_TO_SHIFT_CONTRASTS_V1.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_csv(args.t1_inputs / "T1_V8_LABEL_BLIND_ANALYSIS_PANEL.csv.gz", low_memory=False)
    common = pd.read_csv(args.t1_inputs / "T1_V8_COMMON_VALID_MODEL_OBSERVATIONS.csv.gz", low_memory=False)
    if len(panel) != 10000 or len(common) != 40000:
        raise ValueError("Unexpected frozen T1 denominator")
    eligibility_rows, eligible_tables, excluded_tables = [], [], []
    aggregated_tables, duplicate_summaries, duplicate_details = [], [], []
    result_tables, replicate_tables = [], []
    for spec in SPECS:
        summary, unit_set, eligible, excluded = audit_matching(panel, spec)
        eligibility_rows.append(summary)
        eligible_tables.append(eligible)
        excluded_tables.append(excluded)
        aggregated, duplicate_summary, duplicate_detail = aggregate_joint_units(common, spec, unit_set)
        aggregated_tables.append(aggregated)
        duplicate_summaries.append(duplicate_summary)
        duplicate_details.append(duplicate_detail)
        results, replicates = run_joint_bootstrap(aggregated, spec, summary)
        result_tables.append(results)
        replicate_tables.append(replicates)

    contrasts = pd.concat(result_tables, ignore_index=True)
    contrasts["contrast_label"] = contrasts["contrast_label"].str.replace("→", " -> ", regex=False)
    prior = pd.read_csv(args.prior_contrasts, low_memory=False)
    outputs = {
        "T5_REFERENCE_TO_SHIFT_CONTRASTS_V1_1.csv": contrasts,
        "T5_T1_JOINT_MATCHING_ELIGIBILITY_V1_1.csv": pd.DataFrame(eligibility_rows),
        "T5_T1_JOINT_MATCHED_ELIGIBLE_UNITS_V1_1.csv": pd.concat(eligible_tables, ignore_index=True),
        "T5_T1_JOINT_MATCHING_EXCLUDED_UNMATCHED_V1_1.csv": pd.concat(excluded_tables, ignore_index=True),
        "T5_T1_JOINT_MATCHED_AGGREGATED_UNITS_V1_1.csv.gz": pd.concat(aggregated_tables, ignore_index=True),
        "T5_T1_JOINT_MATCHED_DUPLICATE_AUDIT_V1_1.csv": pd.concat(duplicate_summaries, ignore_index=True),
        "T5_T1_JOINT_MATCHED_DUPLICATE_GROUPS_V1_1.csv": pd.concat(duplicate_details, ignore_index=True),
        "T5_JOINT_MATCHED_BOOTSTRAP_REPLICATES_V1_1.csv.gz": pd.concat(replicate_tables, ignore_index=True),
        "T5_T1_PREDICTION_CONTEXT_INVARIANCE_V1_1.csv": build_context_invariance(common, panel),
        "T5_SHARED_DESIGN_VS_JOINT_MATCHED_SENSITIVITY_V1_1.csv": build_sensitivity(contrasts, prior),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, index=False)


if __name__ == "__main__":
    main()
