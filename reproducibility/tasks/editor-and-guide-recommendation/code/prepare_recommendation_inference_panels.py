#!/usr/bin/env python
"""Prepare raw-derived T7 prediction-completion panels.

The output panel is not a new T7 ground truth. It is a runnable model-input
panel derived from already audited raw T7 recommendation groups. Groups are kept
only when every candidate is compatible with the declared model set, so completed
predictions can later be scored as genuine ranking/recommendation decisions.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "processed" / "T7_editor_design_recommendation_raw_v1"
RAW_CANDIDATES = RAW_DIR / "T7_raw_ground_truth_candidates.csv"
T1_VIEW = ROOT / "data" / "unified_matrix_v5" / "task_views" / "T1_total_efficiency_view_v5.csv"
OUT_DEFAULT = ROOT / "data" / "processed" / "T7_prediction_completion_panels_v1"
REPORT_DEFAULT = ROOT / "reports" / "T7_PREDICTION_COMPLETION_PANELS_V1.md"

MODEL_LABELS = {
    "crispron_be": "CRISPRon-BE",
    "deepbaseeditor": "DeepBaseEditor",
    "behive_efficiency": "BE-Hive efficiency",
    "forecast_be": "FORECasT-BE",
    "deepbe": "DeepBE",
}

RAW_USECOLS = [
    "scenario",
    "objective",
    "candidate_kind",
    "recommendation_group_id",
    "observation_id",
    "design_id",
    "editor_context_id",
    "source_family",
    "source_dataset",
    "source_file",
    "source_row_key",
    "source_associated_models",
    "editor",
    "editor_class",
    "editor_family",
    "cas_variant",
    "pam_scope",
    "cell_context",
    "delivery",
    "timepoint",
    "replicate_strategy",
    "sequence_50nt",
    "protospacer_20nt",
    "pam",
    "target_base",
    "target_position",
    "intended_edit",
    "measurement_scale",
    "qc_level",
    "measured_efficiency",
    "observed_rank_in_group",
    "high_efficiency_label",
    "high_label_cutoff_rank",
    "recommendation_top_k",
    "oracle_efficiency_in_group",
    "candidate_regret_from_oracle",
]

VIEW_USECOLS = [
    "observation_id",
    "source_inventory_id",
    "source_family",
    "source_dataset",
    "source_file",
    "source_sheet",
    "source_row_key",
    "source_relation_to_model",
    "source_associated_models",
    "source_role",
    "label_type",
    "measured_total_efficiency",
    "measurement_scale",
    "qc_level",
    "sequence_context_original",
    "sequence_56nt",
    "sequence_50nt",
    "sequence_30nt",
    "sequence_29nt",
    "protospacer_20nt",
    "pam",
    "pam_type",
    "target_base",
    "target_position",
    "intended_edit",
    "reference_allele",
    "alternate_allele",
    "gene",
    "genomic_locus",
    "sequence_qc_status",
    "design_hash",
    "editor",
    "editor_class",
    "editor_family",
    "cas_variant",
    "pam_scope",
    "cell_context",
    "delivery",
    "timepoint",
    "replicate_strategy",
    "editor_qc_status",
    "editor_context_hash",
    "source_balancing_group",
    "main_scale_group",
]

TRUTH_MEMBERSHIP_COLS = [
    "scenario",
    "objective",
    "candidate_kind",
    "recommendation_group_id",
    "observation_id",
    "design_id",
    "editor_context_id",
    "source_family",
    "editor",
    "editor_class",
    "cas_variant",
    "cell_context",
    "measured_total_efficiency",
    "measured_efficiency",
    "observed_rank_in_group",
    "high_efficiency_label",
    "high_label_cutoff_rank",
    "recommendation_top_k",
    "oracle_efficiency_in_group",
    "candidate_regret_from_oracle",
]

ADAPTER_PANEL_COLS = [
    "observation_id",
    "design_id",
    "editor_context_id",
    "source_inventory_id",
    "source_family",
    "source_dataset",
    "source_file",
    "source_sheet",
    "source_row_key",
    "source_relation_to_model",
    "source_associated_models",
    "source_role",
    "label_type",
    "measured_total_efficiency",
    "measurement_scale",
    "qc_level",
    "sequence_context_original",
    "sequence_56nt",
    "sequence_50nt",
    "sequence_30nt",
    "sequence_29nt",
    "protospacer_20nt",
    "pam",
    "pam_type",
    "target_base",
    "target_position",
    "intended_edit",
    "reference_allele",
    "alternate_allele",
    "gene",
    "genomic_locus",
    "sequence_qc_status",
    "design_hash",
    "editor",
    "editor_class",
    "editor_family",
    "cas_variant",
    "pam_scope",
    "cell_context",
    "delivery",
    "timepoint",
    "replicate_strategy",
    "editor_qc_status",
    "editor_context_hash",
    "source_balancing_group",
    "main_scale_group",
    "compatible_crispron_be",
    "compatible_deepbaseeditor",
    "compatible_behive_efficiency",
    "compatible_forecast_be_t1_derived",
    "compatible_deepbe_t1_derived",
    "t7_panel_set",
    "t7_complete_model_set",
    "t7_group_membership_count",
    "t7_scenarios",
    "t7_recommendation_group_ids",
]


def valid_seq(series: pd.Series, n: int) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.fullmatch(f"[ACGT]{{{n}}}")


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


def safe_set_name(models: list[str]) -> str:
    if models == ["crispron_be", "forecast_be"] or models == ["forecast_be", "crispron_be"]:
        return "priority_pair_crispron_forecast"
    return "modelset_" + "_".join(m.replace("-", "_") for m in models)


def add_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seq30 = valid_seq(out["sequence_30nt"], 30)
    seq50 = valid_seq(out["sequence_50nt"], 50)
    proto20 = valid_seq(out["protospacer_20nt"], 20)
    editor_class = out["editor_class"].fillna("").astype(str)
    pam_type = out["pam_type"].fillna("").astype(str)
    source_family = out["source_family"].fillna("").astype(str)
    target_base = out["target_base"].fillna("").astype(str).str.upper()
    window = out["sequence_30nt"].fillna("").astype(str).str.upper().str.slice(6, 14)
    has_target_in_window = pd.Series(
        [bool(tb) and tb in win for tb, win in zip(target_base, window)],
        index=out.index,
    )
    out["compatible_crispron_be"] = (
        seq30
        & proto20
        & editor_class.isin(["ABE", "CBE"])
        & pam_type.eq("NGG")
        & has_target_in_window
    )
    out["compatible_deepbaseeditor"] = seq30 & proto20 & editor_class.eq("ABE")
    out["compatible_behive_efficiency"] = (
        seq50
        & seq30
        & source_family.eq("BE-Hive")
        & editor_class.isin(["ABE", "CBE", "CGBE"])
    )
    out["compatible_forecast_be_t1_derived"] = proto20 & editor_class.isin(["ABE", "CBE"])
    out["compatible_deepbe_t1_derived"] = seq30 & editor_class.eq("ABE")
    return out


def compat_col(model_id: str) -> str:
    if model_id == "forecast_be":
        return "compatible_forecast_be_t1_derived"
    if model_id == "deepbe":
        return "compatible_deepbe_t1_derived"
    return f"compatible_{model_id}"


def load_joined() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CANDIDATES, dtype=str, usecols=RAW_USECOLS, low_memory=False).fillna("")
    view = pd.read_csv(T1_VIEW, dtype=str, usecols=VIEW_USECOLS, low_memory=False).fillna("")
    view = view.drop_duplicates("observation_id")
    joined = raw.merge(view, on="observation_id", how="left", suffixes=("_raw", ""))
    for col in [
        "source_family",
        "source_dataset",
        "source_file",
        "source_row_key",
        "source_associated_models",
        "editor",
        "editor_class",
        "editor_family",
        "cas_variant",
        "pam_scope",
        "cell_context",
        "delivery",
        "timepoint",
        "replicate_strategy",
        "sequence_50nt",
        "protospacer_20nt",
        "pam",
        "target_base",
        "target_position",
        "intended_edit",
        "measurement_scale",
        "qc_level",
    ]:
        raw_col = f"{col}_raw"
        if raw_col in joined.columns:
            joined[col] = joined[col].where(joined[col].astype(str).ne(""), joined[raw_col])
            joined = joined.drop(columns=[raw_col])
    joined["measured_total_efficiency"] = joined["measured_total_efficiency"].where(
        joined["measured_total_efficiency"].astype(str).ne(""), joined["measured_efficiency"]
    )
    return add_compatibility(joined)


def complete_group_membership(df: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [compat_col(model) for model in models]
    work = df.copy()
    work["compatible_model_set"] = work[cols].all(axis=1)
    grouped = (
        work.groupby(["scenario", "recommendation_group_id"], dropna=False)
        .agg(
            raw_candidates=("observation_id", "size"),
            compatible_candidates=("compatible_model_set", "sum"),
            source_family=("source_family", lambda x: ";".join(sorted(set(map(str, x))))),
            candidate_kind=("candidate_kind", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    grouped["complete_for_model_set"] = grouped["compatible_candidates"].eq(grouped["raw_candidates"])
    keep = grouped[grouped["complete_for_model_set"]][["scenario", "recommendation_group_id"]]
    membership = work.merge(keep, on=["scenario", "recommendation_group_id"], how="inner")
    summary = (
        grouped.groupby(["scenario"], dropna=False)
        .agg(
            raw_groups=("recommendation_group_id", "nunique"),
            complete_groups=("complete_for_model_set", "sum"),
            raw_candidates=("raw_candidates", "sum"),
            candidates_in_complete_groups=(
                "raw_candidates",
                lambda x: int(x[grouped.loc[x.index, "complete_for_model_set"]].sum()),
            ),
        )
        .reset_index()
    )
    return membership, summary


def make_unique_panel(membership: pd.DataFrame, set_name: str, models: list[str]) -> pd.DataFrame:
    membership_info = (
        membership.groupby("observation_id", dropna=False)
        .agg(
            t7_group_membership_count=("recommendation_group_id", "nunique"),
            t7_scenarios=("scenario", lambda x: ";".join(sorted(set(map(str, x))))),
            t7_recommendation_group_ids=(
                "recommendation_group_id",
                lambda x: ";".join(sorted(set(map(str, x)))[:20]),
            ),
        )
        .reset_index()
    )
    panel = (
        membership.sort_values(["source_family", "editor_class", "observation_id"])
        .drop_duplicates("observation_id", keep="first")
        .merge(membership_info, on="observation_id", how="left")
    )
    panel["t7_panel_set"] = set_name
    panel["t7_complete_model_set"] = ";".join(models)
    for col in ADAPTER_PANEL_COLS:
        if col not in panel.columns:
            panel[col] = ""
    return panel[ADAPTER_PANEL_COLS].copy()


def write_batches(panel: pd.DataFrame, out_dir: Path, set_name: str, batch_size: int) -> pd.DataFrame:
    batch_dir = out_dir / f"batches_{set_name}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    panel = panel.sort_values(["source_family", "editor_class", "observation_id"]).reset_index(drop=True)
    rows = []
    n_batches = max(1, math.ceil(len(panel) / batch_size))
    for i in range(n_batches):
        start = i * batch_size
        end = min(len(panel), (i + 1) * batch_size)
        batch = panel.iloc[start:end].copy()
        path = batch_dir / f"T7_{set_name}_batch_{i + 1:03d}.csv"
        batch.to_csv(path, index=False)
        rows.append(
            {
                "batch": i + 1,
                "path": str(path),
                "rows": len(batch),
                "unique_observations": batch["observation_id"].nunique(),
                "source_families": ";".join(sorted(batch["source_family"].dropna().astype(str).unique())),
                "editor_classes": ";".join(sorted(batch["editor_class"].dropna().astype(str).unique())),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / f"T7_{set_name}_batch_manifest.csv", index=False)
    return manifest


def write_report(
    report_path: Path,
    out_dir: Path,
    set_name: str,
    models: list[str],
    group_summary: pd.DataFrame,
    panel: pd.DataFrame,
    batch_manifest: pd.DataFrame,
    smoke_path: Path,
) -> None:
    generated = datetime.now().isoformat(timespec="seconds")
    model_rows = pd.DataFrame(
        [{"model_id": model, "display_name": MODEL_LABELS.get(model, model)} for model in models]
    )
    panel_summary = pd.DataFrame(
        [
            {
                "panel_set": set_name,
                "models": ";".join(models),
                "unique_observations": len(panel),
                "batches": len(batch_manifest),
                "batch_size_target": int(batch_manifest["rows"].max()) if not batch_manifest.empty else 0,
                "smoke_rows": len(pd.read_csv(smoke_path, usecols=["observation_id"], dtype=str)),
            }
        ]
    )
    text = [
        "# T7 prediction-completion panels v1",
        "",
        f"Generated: {generated}",
        "",
        "## Meaning",
        "",
        "These panels are runnable model-input panels for corrected T7. They are derived from raw T7 recommendation groups and keep only groups where every candidate is compatible with the declared model set.",
        "",
        "## Model Set",
        "",
        md_table(model_rows),
        "",
        "## Complete Groups",
        "",
        md_table(group_summary),
        "",
        "## Panel Summary",
        "",
        md_table(panel_summary),
        "",
        "## Outputs",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Unique observation panel: `{out_dir / f'T7_{set_name}_unique_observation_panel.csv'}`",
        f"- Group membership table: `{out_dir / f'T7_{set_name}_complete_group_membership.csv'}`",
        f"- Batch manifest: `{out_dir / f'T7_{set_name}_batch_manifest.csv'}`",
        f"- Smoke panel: `{smoke_path}`",
        "",
        "## Guardrail",
        "",
        "This is not a model leaderboard and not a scalar-regression reformulation. It is a prediction-completion substrate for later T7 ranking metrics.",
        "",
    ]
    report_path.write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="crispron_be,forecast_be")
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=25000)
    parser.add_argument("--smoke-rows", type=int, default=1000)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_LABELS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    set_name = safe_set_name(models)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    joined = load_joined()
    membership, group_summary = complete_group_membership(joined, models)
    if membership.empty:
        raise ValueError(f"No complete groups for model set {models}")
    membership_out = membership[[c for c in TRUTH_MEMBERSHIP_COLS if c in membership.columns]].copy()
    membership_path = args.output_dir / f"T7_{set_name}_complete_group_membership.csv"
    membership_out.to_csv(membership_path, index=False)

    panel = make_unique_panel(membership, set_name, models)
    panel_path = args.output_dir / f"T7_{set_name}_unique_observation_panel.csv"
    panel.to_csv(panel_path, index=False)
    batch_manifest = write_batches(panel, args.output_dir, set_name, args.batch_size)

    smoke = panel.head(min(args.smoke_rows, len(panel))).copy()
    smoke_path = args.output_dir / f"T7_{set_name}_smoke_panel.csv"
    smoke.to_csv(smoke_path, index=False)

    group_summary["panel_set"] = set_name
    group_summary["models"] = ";".join(models)
    group_summary.to_csv(args.output_dir / f"T7_{set_name}_group_summary.csv", index=False)
    write_report(args.report, args.output_dir, set_name, models, group_summary, panel, batch_manifest, smoke_path)

    print(f"Wrote T7 prediction-completion panel set to {args.output_dir}")
    print(f"Wrote report to {args.report}")
    print(group_summary.to_string(index=False))
    print(f"Unique observation panel rows: {len(panel):,}")
    print(f"Batches: {len(batch_manifest):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
