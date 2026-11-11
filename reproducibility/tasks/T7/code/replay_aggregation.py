"""Rebuild T7 unique-design observations and predictions from the frozen pre-aggregation rows."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO = "design_selection_high_efficiency"


KEY = ["scenario", "recommendation_group_id", "design_id"]


MODELS = ["crispron_be", "forecast_be"]


COMPAT_COLS = {
    "crispron_be": "compatible_crispron_be",
    "deepbaseeditor": "compatible_deepbaseeditor",
    "behive_efficiency": "compatible_behive_efficiency",
    "forecast_be": "compatible_forecast_be_t1_derived",
    "deepbe": "compatible_deepbe_t1_derived",
}


def stable_id(values: tuple[object, ...]) -> str:
    payload = "|".join(str(v) for v in values).encode("utf-8")
    return "T7UD_" + hashlib.sha1(payload).hexdigest()[:12]


def bool_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().eq("true")


def aggregate_membership(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    member = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    member = member[member["scenario"].eq(SCENARIO)].copy()
    member["measured_efficiency"] = pd.to_numeric(member["measured_efficiency"], errors="raise")
    member["measured_total_efficiency"] = pd.to_numeric(
        member["measured_total_efficiency"], errors="coerce"
    )
    member = member.sort_values([*KEY, "observation_id"], kind="mergesort")

    grouped = member.groupby(KEY, sort=True, dropna=False)
    unique = grouped.agg(
        design_group_observation_records=("observation_id", "size"),
        measurement_count=("measured_efficiency", "count"),
        measurement_unique_values=("measured_efficiency", "nunique"),
        measured_efficiency=("measured_efficiency", "median"),
        measured_efficiency_min=("measured_efficiency", "min"),
        measured_efficiency_max=("measured_efficiency", "max"),
        measured_total_efficiency=("measured_total_efficiency", "median"),
        first_observation_id=("observation_id", "first"),
        editor_context_id=("editor_context_id", "first"),
        source_family=("source_family", "first"),
        editor=("editor", "first"),
        editor_class=("editor_class", "first"),
        cas_variant=("cas_variant", "first"),
        cell_context=("cell_context", "first"),
        editor_context_values=("editor_context_id", "nunique"),
        source_family_values=("source_family", "nunique"),
        editor_values=("editor", "nunique"),
        cell_context_values=("cell_context", "nunique"),
    ).reset_index()
    unique["design_group_instance_id"] = [
        stable_id(tuple(row)) for row in unique[KEY].itertuples(index=False, name=None)
    ]

    group_size = unique.groupby("recommendation_group_id")["design_id"].transform("size")
    unique["observed_rank_in_group"] = unique.groupby("recommendation_group_id")[
        "measured_efficiency"
    ].rank(method="min", ascending=False)
    unique["high_label_cutoff_rank"] = np.maximum(1, np.ceil(group_size * 0.10).astype(int))
    unique["high_efficiency_label"] = unique["observed_rank_in_group"].le(
        unique["high_label_cutoff_rank"]
    )
    unique["recommendation_top_k"] = np.minimum(10, group_size.astype(int))
    unique["oracle_efficiency_in_group"] = unique.groupby("recommendation_group_id")[
        "measured_efficiency"
    ].transform("max")
    unique["candidate_regret_from_oracle"] = (
        unique["oracle_efficiency_in_group"] - unique["measured_efficiency"]
    )

    group_audit = (
        unique.groupby("recommendation_group_id", sort=True)
        .agg(
            source_family=("source_family", "first"),
            unique_design_group_instances=("design_id", "size"),
            repeated_design_instances=("design_group_observation_records", lambda s: int((s > 1).sum())),
            observation_records=("design_group_observation_records", "sum"),
        )
        .reset_index()
    )
    group_audit["duplicates_removed"] = (
        group_audit["observation_records"] - group_audit["unique_design_group_instances"]
    )
    group_audit["duplicate_fraction"] = (
        group_audit["duplicates_removed"] / group_audit["observation_records"]
    )
    group_audit["affected_by_deduplication"] = group_audit["duplicates_removed"].gt(0)

    metadata_conflicts = unique[
        unique[[
            "editor_context_values",
            "source_family_values",
            "editor_values",
            "cell_context_values",
        ]].gt(1).any(axis=1)
    ].copy()
    return member, unique, group_audit, metadata_conflicts


def load_predictions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = [
        "observation_id",
        "model_id",
        "prediction_value_for_correlation",
        "predicted_efficiency",
        "is_valid_t1_prediction",
        *COMPAT_COLS.values(),
    ]
    pred_chunks: list[pd.DataFrame] = []
    compat_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        dtype=str,
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["model_id"].isin(MODELS)].copy()
        if chunk.empty:
            continue
        score = pd.to_numeric(chunk["prediction_value_for_correlation"], errors="coerce")
        fallback = pd.to_numeric(chunk["predicted_efficiency"], errors="coerce")
        chunk["predicted_recommendation_score"] = score.where(score.notna(), fallback)
        chunk["valid_recommendation_score"] = bool_series(chunk["is_valid_t1_prediction"]) & chunk[
            "predicted_recommendation_score"
        ].notna()
        pred_chunks.append(
            chunk[
                [
                    "observation_id",
                    "model_id",
                    "predicted_recommendation_score",
                    "valid_recommendation_score",
                ]
            ].copy()
        )
        compat_chunks.append(chunk[["observation_id", *COMPAT_COLS.values()]].copy())

    predictions = pd.concat(pred_chunks, ignore_index=True)
    compat = pd.concat(compat_chunks, ignore_index=True)
    for col in COMPAT_COLS.values():
        compat[col] = bool_series(compat[col])
    compat_conflict = (
        compat.groupby("observation_id", sort=True)[list(COMPAT_COLS.values())]
        .nunique(dropna=False)
        .reset_index()
    )
    compat_conflict = compat_conflict[
        compat_conflict[list(COMPAT_COLS.values())].gt(1).any(axis=1)
    ].copy()
    compat = compat.groupby("observation_id", sort=True)[list(COMPAT_COLS.values())].all().reset_index()
    return predictions, compat, compat_conflict


def aggregate_prediction_scores(
    member: pd.DataFrame,
    unique: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = member[["observation_id", *KEY]].copy()
    joined = mapping.merge(predictions, on="observation_id", how="left", validate="one_to_many")
    joined["valid_numeric_score"] = joined["valid_recommendation_score"].fillna(False) & joined[
        "predicted_recommendation_score"
    ].notna()
    valid = joined[joined["valid_numeric_score"]].copy()

    score_agg = (
        valid.groupby([*KEY, "model_id"], sort=True, dropna=False)
        .agg(
            prediction_rows=("predicted_recommendation_score", "size"),
            valid_prediction_rows=("predicted_recommendation_score", "count"),
            unique_numeric_score_values=("predicted_recommendation_score", "nunique"),
            predicted_recommendation_score=("predicted_recommendation_score", "median"),
            predicted_score_min=("predicted_recommendation_score", "min"),
            predicted_score_max=("predicted_recommendation_score", "max"),
        )
        .reset_index()
    )
    score_agg["score_conflict"] = score_agg["unique_numeric_score_values"].gt(1)
    score_agg["score_resolution_rule"] = np.where(
        score_agg["score_conflict"], "median_of_valid_repeated_scores", "single_or_identical_value"
    )
    conflicts = score_agg[score_agg["score_conflict"]].copy()

    model_grid = pd.DataFrame({"model_id": MODELS})
    score_table = (
        unique.merge(model_grid, how="cross")
        .merge(score_agg, on=[*KEY, "model_id"], how="left", validate="one_to_one")
    )

    coverage = (
        score_table.groupby(["scenario", "recommendation_group_id", "model_id"], sort=True)
        .agg(
            raw_unique_design_candidates=("design_id", "size"),
            valid_unique_design_scores=("predicted_recommendation_score", "count"),
            unique_prediction_scores=("predicted_recommendation_score", "nunique"),
            source_family=("source_family", "first"),
        )
        .reset_index()
    )
    coverage["complete_unique_candidate_set"] = coverage["valid_unique_design_scores"].eq(
        coverage["raw_unique_design_candidates"]
    )
    coverage["scoreable_rank_group"] = coverage["complete_unique_candidate_set"] & coverage[
        "unique_prediction_scores"
    ].ge(2)
    return score_table, conflicts, coverage


def main():
    task = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=task)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-full", action="store_true", help="Write the reconstructed unique-design rows")
    args = parser.parse_args()
    inputs = args.task / "inputs"
    member, unique, audit, metadata_conflicts = aggregate_membership(inputs / "T7_design39_observation_membership.csv")
    predictions, _, compatibility_conflicts = load_predictions(inputs / "T7_observation_scores_used.csv.gz")
    scores, prediction_conflicts, coverage = aggregate_prediction_scores(member, unique, predictions)
    if len(member) != 503545 or len(unique) != 438768 or len(scores) != 877536:
        raise ValueError("Unexpected T7 input or unique design denominator")

    columns = ["design_group_instance_id", "design_group_observation_records", "measured_efficiency",
               "high_efficiency_label"]
    frozen = pd.read_csv(inputs / "T7_unique_design_membership.csv", usecols=columns)
    pd.testing.assert_frame_equal(unique[columns].reset_index(drop=True), frozen[columns],
                                  check_dtype=False, rtol=1e-9, atol=1e-10)
    columns = ["design_group_instance_id", "model_id", "predicted_recommendation_score", "score_conflict"]
    frozen = pd.read_csv(inputs / "T7_unique_design_model_scores.csv", usecols=columns)
    pd.testing.assert_frame_equal(scores[columns].reset_index(drop=True), frozen[columns],
                                  check_dtype=False, rtol=1e-9, atol=1e-10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "T7_unique_design_aggregation_audit.csv": audit,
        "T7_unique_design_metadata_conflicts.csv": metadata_conflicts,
        "T7_unique_design_model_group_coverage.csv": coverage,
        "T7_unique_design_compatibility_flag_conflicts.csv": compatibility_conflicts,
        "T7_unique_design_prediction_score_conflicts.csv": prediction_conflicts,
    }
    if args.write_full:
        outputs["T7_unique_design_membership.csv"] = unique
        outputs["T7_unique_design_model_scores.csv"] = scores
    for name, data in outputs.items():
        data.to_csv(args.output_dir / name, index=False)
    print(f"PASS: {len(member):,} original observations, {len(unique):,} unique design-group instances "
          f"and {len(scores):,} frozen model-score rows")


if __name__ == "__main__":
    main()
