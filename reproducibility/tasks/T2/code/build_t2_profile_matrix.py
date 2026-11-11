#!/usr/bin/env python
"""
Build the T2 outcome and position profile matrices from BE-Hive and related sources.
"""

from __future__ import annotations

import os
import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ROOT = Path(os.environ.get("BE_REPLAY_PROJECT", ".")) / "benchmark_core"
DATA = ROOT / "data"
MANUAL = DATA / "manual_downloads"
BEHIVE_BYSTANDER = MANUAL / "be_hive_bystander"
BEHIVE_EFFICIENCY = MANUAL / "be_hive_efficiency"
CGBE_SMART = MANUAL / "cgbe_smart_processed"
FORECAST_PROCESSED = MANUAL / "forecast_be" / "forecast_be_processed_data.tsv"
BEDICT2_SUPP = ROOT.parent / "BEDICT-V2/resources/paper_supplements/13059_2025_3586_MOESM4_ESM.xlsx"
OUT_DEFAULT = DATA / "processed" / "T2_bystander_matrix"

DNA = set("ACGT")
POSITION_COL_RE = re.compile(r"^([ACGT])(-?\d+)$")


TARGET_COLUMNS = [
    "target_profile_id",
    "source_profile_id",
    "source_family",
    "source_dataset",
    "source_file",
    "source_row_key",
    "source_relation_to_model",
    "source_associated_models",
    "editor",
    "editor_class",
    "editor_family",
    "cell_context",
    "replicate_strategy",
    "sequence_context_original",
    "sequence_50nt",
    "protospacer_20nt",
    "native_50nt_available",
    "protospacer_20nt_available",
    "behive_bystander_input_eligible",
    "forecast_position_input_eligible",
    "pam",
    "pam_type",
    "target_position",
    "intended_edit",
    "target_base",
    "measured_total_efficiency",
    "total_read_count",
    "edited_read_count",
    "outcome_row_count",
    "editable_position_count",
    "observed_outcome_frequency_sum",
    "observed_max_product_frequency",
    "observed_mean_edited_positions_per_product",
    "observed_any_bystander_frequency",
    "observed_desired_product_frequency",
    "measurement_scale",
    "qc_sequence_pass",
    "qc_label_pass",
    "qc_level",
    "exclusion_reason",
    "notes",
]

POSITION_COLUMNS = [
    "position_profile_id",
    "target_profile_id",
    "source_profile_id",
    "source_row_key",
    "editor",
    "editor_class",
    "cell_context",
    "protospacer_position",
    "reference_base",
    "intended_position",
    "is_intended_position",
    "observed_conditional_position_rate",
    "observed_absolute_position_rate_estimate",
    "measurement_scale",
    "outcome_row_count",
    "edited_read_count",
    "source_file",
]

OUTCOME_COLUMNS = [
    "outcome_profile_id",
    "target_profile_id",
    "source_profile_id",
    "source_row_key",
    "editor",
    "editor_class",
    "cell_context",
    "outcome_rank_in_target",
    "observed_outcome_frequency",
    "outcome_count",
    "outcome_edit_count",
    "outcome_events",
    "has_bystander_edit",
    "is_desired_product",
    "measurement_scale",
    "source_file",
]


def stable_hash(*parts: object, n: int = 12) -> str:
    text = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{stable_hash(*parts)}"


def clean_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fmt_float(value: object, digits: int = 12) -> str:
    out = clean_float(value)
    if out is None:
        return ""
    return f"{out:.{digits}g}"


def clean_dna(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = re.sub(r"[^ACGTacgt]", "", str(value)).upper()
    return text


def first_existing(row: pd.Series, candidates: Sequence[str]) -> object:
    for col in candidates:
        if col in row.index:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                if str(val) != "":
                    return val
    return ""


def first_existing_col(columns: Iterable[str], candidates: Sequence[str]) -> str:
    column_set = set(columns)
    for col in candidates:
        if col in column_set:
            return col
    return ""


def numeric_sum(row: pd.Series, prefix: str) -> Optional[float]:
    vals = [clean_float(row.get(col)) for col in row.index if str(col).startswith(prefix)]
    vals = [val for val in vals if val is not None]
    if not vals:
        return None
    return float(sum(vals))


def numeric_mean(row: pd.Series, prefix: str) -> Optional[float]:
    vals = [
        clean_float(row.get(col))
        for col in row.index
        if str(col).startswith(prefix)
        and "logit" not in str(col).lower()
        and "norm" not in str(col).lower()
    ]
    vals = [val for val in vals if val is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def extract_50nt(sequence_context: object, guide: object) -> str:
    seq = clean_dna(sequence_context)
    guide = clean_dna(guide)
    if len(seq) == 50 and set(seq) <= DNA:
        return seq
    if not seq or len(guide) != 20:
        return ""
    idx = seq.find(guide)
    if idx >= 20 and idx + 30 <= len(seq):
        candidate = seq[idx - 20 : idx + 30]
        if len(candidate) == 50 and set(candidate) <= DNA:
            return candidate
    return ""


def infer_pam(sequence_context: object, guide: object) -> str:
    seq = clean_dna(sequence_context)
    guide = clean_dna(guide)
    if not seq or len(guide) != 20:
        return ""
    idx = seq.find(guide)
    if idx >= 0 and idx + 23 <= len(seq):
        return seq[idx + 20 : idx + 23]
    return ""


def pam_type(pam: object) -> str:
    pam_text = str(pam or "").upper()
    if len(pam_text) < 2:
        return "unknown"
    if len(pam_text) >= 3 and pam_text[1:] == "GG":
        return "NGG"
    if pam_text.endswith("G"):
        return "noncanonical_NG_like"
    return "non_NGG_or_PAM_relaxed"


def infer_behive_metadata(stem: str) -> Dict[str, str]:
    parts = stem.split("_")
    if stem.startswith("HEK293T_"):
        cell = "HEK293T"
        rest = stem[len("HEK293T_") :]
    elif stem.startswith("U2OS_"):
        cell = "U2OS"
        rest = stem[len("U2OS_") :]
    elif stem.startswith("mES_"):
        cell = "mES"
        rest = stem[len("mES_") :]
    else:
        cell = parts[0] if parts else ""
        rest = "_".join(parts[1:])

    tokens = rest.split("_")
    dataset = tokens[0] if tokens else ""
    editor = "_".join(tokens[1:]) if len(tokens) > 1 else rest
    if "ABE" in editor or dataset == "AtoG":
        editor_class = "ABE"
    elif dataset == "CtoGA" or "noUGI" in editor or editor.startswith("CG"):
        editor_class = "CGBE"
    else:
        editor_class = "CBE"

    return {
        "cell_context": cell,
        "source_dataset": dataset,
        "editor": editor,
        "editor_class": editor_class,
        "editor_family": "BE-Hive",
    }


def find_efficiency_table(stem: str) -> Path:
    direct = BEHIVE_EFFICIENCY / f"{stem}.csv"
    if direct.exists():
        return direct
    cgbe = CGBE_SMART / f"{stem}.csv"
    if cgbe.exists():
        return cgbe
    return Path()


def load_efficiency_map(path: Path) -> Tuple[Dict[str, pd.Series], Dict[str, object]]:
    if not path:
        return {}, {"efficiency_rows": 0, "efficiency_columns": 0}
    try:
        df = pd.read_csv(path, dtype=object, low_memory=False)
    except Exception as exc:
        return {}, {"efficiency_rows": 0, "efficiency_columns": 0, "efficiency_error": repr(exc)}
    name_col = first_existing_col(df.columns, ["Name (unique)", "Name", "Oligo Id"])
    if not name_col:
        return {}, {"efficiency_rows": len(df), "efficiency_columns": len(df.columns), "efficiency_error": "no target key column"}
    return {str(row[name_col]): row for _, row in df.iterrows()}, {
        "efficiency_rows": len(df),
        "efficiency_columns": len(df.columns),
        "efficiency_name_column": name_col,
    }


def position_columns(df: pd.DataFrame) -> List[str]:
    return [str(col) for col in df.columns if POSITION_COL_RE.match(str(col))]


def outcome_events(row: pd.Series, pos_cols: Sequence[str]) -> Tuple[str, int, Dict[int, bool]]:
    events: List[str] = []
    edited_by_position: Dict[int, bool] = {}
    for col in pos_cols:
        match = POSITION_COL_RE.match(col)
        if not match:
            continue
        ref_base = match.group(1)
        pos = int(match.group(2))
        value = str(row.get(col, "")).upper()
        edited = bool(value and value != ref_base)
        edited_by_position[pos] = edited
        if edited:
            events.append(f"{ref_base}{pos}>{value}")
    return ";".join(events), len(events), edited_by_position


def desired_product_flags(
    row: pd.Series,
    pos_cols: Sequence[str],
    intended_position: Optional[int],
) -> Tuple[str, str]:
    if intended_position is None:
        return "", ""
    edited_other = False
    edited_intended = False
    for col in pos_cols:
        match = POSITION_COL_RE.match(col)
        if not match:
            continue
        ref_base = match.group(1)
        pos = int(match.group(2))
        value = str(row.get(col, "")).upper()
        edited = bool(value and value != ref_base)
        if pos == intended_position:
            edited_intended = edited
        elif edited:
            edited_other = True
    return str(edited_other), str(edited_intended and not edited_other)


def infer_target_position(row: Optional[pd.Series], source_row_key: str) -> Optional[int]:
    if row is not None:
        value = first_existing(row, ["Editing position", "Editing position_x", "Position of SNP in gRNA_x", "target_position"])
        pos = clean_float(value)
        if pos is not None:
            return int(pos)
    match = re.search(r"editpos(-?\d+)", source_row_key)
    if match:
        return int(match.group(1))
    return None


def build_source_inventory_extra() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if FORECAST_PROCESSED.exists():
        try:
            df = pd.read_csv(FORECAST_PROCESSED, sep="\t", dtype=object, low_memory=False)
            rows.append(
                {
                    "source_profile_id": "SRC_T2_FORECAST_PROCESSED",
                    "source_family": "FORECasT-BE",
                    "source_dataset": "processed_position_table",
                    "source_file": str(FORECAST_PROCESSED),
                    "observed_target_count": len(df),
                    "observed_outcome_rows": "",
                    "observed_position_rows": "",
                    "metadata_join_status": "blocked_no_sequence_context_in_table",
                    "t2_role": "candidate_position_profile_after_oligo_sequence_mapping",
                    "notes": "Wide substitution-frequency table has Oligo Id, Total Reads and Editor but no target sequence in the local file.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source_profile_id": "SRC_T2_FORECAST_PROCESSED",
                    "source_family": "FORECasT-BE",
                    "source_dataset": "processed_position_table",
                    "source_file": str(FORECAST_PROCESSED),
                    "metadata_join_status": "read_failed",
                    "t2_role": "candidate_position_profile_after_oligo_sequence_mapping",
                    "notes": repr(exc),
                }
            )
    if BEDICT2_SUPP.exists():
        rows.append(
            {
                "source_profile_id": "SRC_T2_BEDICT2_SUPPLEMENT",
                "source_family": "BEDICT2",
                "source_dataset": "supplement_position_and_proportion",
                "source_file": str(BEDICT2_SUPP),
                "metadata_join_status": "available_via_existing_unified_matrix_v5_T4_and_model_adapter",
                "t2_role": "candidate_position_rate_and_abe_haplotype_profile",
                "notes": "BEDICT-V2 proportion model is executable locally; supplement is source-associated for BEDICT-V2 claims.",
            }
        )
    existing_t3 = DATA / "unified_matrix_v5" / "task_views" / "T3_outcome_bystander_view_v5.csv"
    existing_t4 = DATA / "unified_matrix_v5" / "task_views" / "T4_position_profile_view_v5.csv"
    for path, family, role in [
        (existing_t3, "unified_matrix_v5", "legacy_outcome_view_candidate"),
        (existing_t4, "unified_matrix_v5", "legacy_position_view_candidate"),
    ]:
        if path.exists():
            try:
                row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
            except Exception:
                row_count = ""
            rows.append(
                {
                    "source_profile_id": f"SRC_T2_{path.stem.upper()}",
                    "source_family": family,
                    "source_dataset": path.stem,
                    "source_file": str(path),
                    "observed_target_count": row_count,
                    "metadata_join_status": "available_not_renumbered_as_new_T2",
                    "t2_role": role,
                    "notes": "Kept as source-aware background; new T2 draft does not overwrite old v5 task numbering.",
                }
            )
    return rows


def schema_payload(
    max_targets_per_source: int,
    max_outcomes_per_target: int,
    max_sources: int = 0,
) -> Dict[str, object]:
    return {
        "schema_name": "T2_bystander_editing_pattern_matrix_draft_v0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "scope": "Draft observed matrix for bystander/product and position-profile prediction; no model ranking.",
        "draft_limits": {
            "max_sources": max_sources,
            "max_targets_per_source": max_targets_per_source,
            "max_outcomes_per_target": max_outcomes_per_target,
            "note": "Use 0 for all targets/outcomes after schema review.",
        },
        "measurement_scales": {
            "conditional_outcome_frequency_given_edited_reads": "Outcome/product frequency from BE-Hive bystander pickle tables; frequencies sum to one within a target over edited products.",
            "conditional_position_rate_given_edited_reads": "Position-level edit probability obtained by summing conditional outcome frequencies where that position differs from its reference base.",
            "absolute_position_rate_estimate": "conditional_position_rate * measured_total_efficiency when a matching efficiency table is available; marked as an estimate.",
        },
        "tables": {
            "source_inventory_t2.csv": {
                "primary_key": ["source_profile_id"],
                "purpose": "Source-level audit, including candidate sources not yet materialized into draft profile rows.",
            },
            "T2_observed_target_profiles.csv": {
                "primary_key": ["target_profile_id"],
                "purpose": "One row per target/editor/cell context with sequence metadata and target-level product-profile summaries.",
                "columns": TARGET_COLUMNS,
            },
            "T2_observed_position_profiles.csv": {
                "primary_key": ["position_profile_id"],
                "foreign_keys": {"target_profile_id": "T2_observed_target_profiles.target_profile_id"},
                "purpose": "One row per editable position per target, derived from observed outcome distributions.",
                "columns": POSITION_COLUMNS,
            },
            "T2_observed_outcome_profiles.csv": {
                "primary_key": ["outcome_profile_id"],
                "foreign_keys": {"target_profile_id": "T2_observed_target_profiles.target_profile_id"},
                "purpose": "Long-form product/outcome rows for the sampled draft targets.",
                "columns": OUTCOME_COLUMNS,
            },
        },
        "review_gates_before_full_benchmark": [
            "Confirm conditional vs absolute scales for model comparison.",
            "Confirm whether product purity uses a known intended edit position or target-specific max product when intended position is unavailable.",
            "Confirm source-associated versus external claims for BE-Hive and BEDICT-V2.",
            "Confirm whether FORECasT-BE processed Oligo Ids can be mapped to sequences before inclusion as observed data.",
        ],
    }


def write_report(
    out_dir: Path,
    source_inventory: pd.DataFrame,
    target_df: pd.DataFrame,
    position_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    max_targets_per_source: int,
    max_outcomes_per_target: int,
) -> None:
    source_counts = target_df.groupby(["cell_context", "editor"], dropna=False).size().reset_index(name="targets")
    measurement_counts = target_df["measurement_scale"].value_counts(dropna=False).reset_index()
    eligibility_rows = []
    for col in ["native_50nt_available", "protospacer_20nt_available", "behive_bystander_input_eligible", "forecast_position_input_eligible"]:
        if col in target_df.columns:
            for value, count in target_df[col].value_counts(dropna=False).items():
                eligibility_rows.append({"field": col, "value": value, "count": count})
    eligibility_df = pd.DataFrame(eligibility_rows)
    lines = [
        "# T2 bystander matrix draft report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "This draft starts T2 as bystander/editing-pattern prediction using the unified matrix pattern from T1. It does not rerun scalar T1 and does not use the old Data1/Data2 framing.",
        "",
        "## Draft limits",
        "",
        f"- max_targets_per_source: {max_targets_per_source}",
        f"- max_outcomes_per_target: {max_outcomes_per_target}",
        "- Full benchmark execution is intentionally deferred until the T2 schema is reviewed.",
        "",
        "## Output tables",
        "",
        f"- source_inventory_t2.csv: {len(source_inventory)} source rows",
        f"- T2_observed_target_profiles.csv: {len(target_df)} target rows",
        f"- T2_observed_position_profiles.csv: {len(position_df)} position rows",
        f"- T2_observed_outcome_profiles.csv: {len(outcome_df)} outcome rows",
        "",
        "## Measurement-scale decision",
        "",
        "BE-Hive bystander observed profiles are conditional on edited reads: outcome frequencies sum to one within a target. Position rates in this draft are derived from that conditional product distribution. Absolute position-rate estimates are provided only when a matching total-efficiency table is available.",
        "",
        "## Source summary",
        "",
    ]
    if not source_counts.empty:
        lines.append("```text")
        lines.append(source_counts.head(40).to_string(index=False))
        lines.append("```")
    lines.extend(["", "## Measurement counts", ""])
    if not measurement_counts.empty:
        lines.append("```text")
        lines.append(measurement_counts.to_string(index=False))
        lines.append("```")
    lines.extend(["", "## Input eligibility", ""])
    if not eligibility_df.empty:
        lines.append("```text")
        lines.append(eligibility_df.to_string(index=False))
        lines.append("```")
    lines.extend(
        [
            "",
            "## Review gates",
            "",
            "1. Confirm that T2's main metrics evaluate conditional bystander/profile distributions separately from absolute total-efficiency calibration.",
            "2. Confirm product-purity definitions for rows lacking an intended target position.",
            "3. Confirm that BE-Hive data are source-associated for BE-Hive model evaluation and should not be used for broad external claims by themselves.",
            "4. Confirm whether to run the full unbounded matrix build and then model adapters.",
        ]
    )
    (out_dir / "T2_MATRIX_DRAFT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--max-targets-per-source", type=int, default=200)
    parser.add_argument("--max-outcomes-per-target", type=int, default=50)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        source_inventory = pd.read_csv(out_dir / "source_inventory_t2.csv", dtype=str, low_memory=False).fillna("")
        target_df = pd.read_csv(out_dir / "T2_observed_target_profiles.csv", dtype=str, low_memory=False).fillna("")
        position_df = pd.read_csv(out_dir / "T2_observed_position_profiles.csv", dtype=str, low_memory=False).fillna("")
        outcome_df = pd.read_csv(out_dir / "T2_observed_outcome_profiles.csv", dtype=str, low_memory=False).fillna("")
        (out_dir / "T2_profile_schema.json").write_text(
            json.dumps(
                schema_payload(args.max_targets_per_source, args.max_outcomes_per_target, args.max_sources),
                indent=2,
            ),
            encoding="utf-8",
        )
        write_report(
            out_dir,
            source_inventory,
            target_df,
            position_df,
            outcome_df,
            args.max_targets_per_source,
            args.max_outcomes_per_target,
        )
        manifest = {
            "output_dir": str(out_dir),
            "source_rows": len(source_inventory),
            "target_rows": len(target_df),
            "position_rows": len(position_df),
            "outcome_rows": len(outcome_df),
            "max_sources": args.max_sources,
            "max_targets_per_source": args.max_targets_per_source,
            "max_outcomes_per_target": args.max_outcomes_per_target,
            "report_only": True,
        }
        (out_dir / "T2_profile_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0

    source_rows: List[Dict[str, object]] = []
    target_rows: List[Dict[str, object]] = []
    position_rows: List[Dict[str, object]] = []
    outcome_rows: List[Dict[str, object]] = []

    pkl_paths = sorted(BEHIVE_BYSTANDER.glob("*.pkl"))
    if args.max_sources > 0:
        pkl_paths = pkl_paths[: args.max_sources]

    for pkl_path in pkl_paths:
        meta = infer_behive_metadata(pkl_path.stem)
        source_profile_id = stable_id("SRC_T2_BEHIVE", pkl_path.stem)
        efficiency_path = find_efficiency_table(pkl_path.stem)
        eff_map, eff_stats = load_efficiency_map(efficiency_path)
        print(f"Reading {pkl_path.name} ...", flush=True)
        try:
            outcome_dict = pd.read_pickle(pkl_path)
        except Exception as exc:
            source_rows.append(
                {
                    "source_profile_id": source_profile_id,
                    "source_family": "BE-Hive",
                    "source_dataset": meta["source_dataset"],
                    "source_file": str(pkl_path),
                    "metadata_join_status": "read_failed",
                    "t2_role": "observed_bystander_profile",
                    "notes": repr(exc),
                }
            )
            continue

        keys = list(outcome_dict.keys())
        selected_keys = keys if args.max_targets_per_source == 0 else keys[: args.max_targets_per_source]
        matched_eff = 0
        source_outcome_count = 0
        source_position_count = 0
        source_rows.append(
            {
                "source_profile_id": source_profile_id,
                "source_family": "BE-Hive",
                "source_dataset": meta["source_dataset"],
                "source_file": str(pkl_path),
                "matching_efficiency_file": str(efficiency_path) if efficiency_path else "",
                "cell_context": meta["cell_context"],
                "editor": meta["editor"],
                "editor_class": meta["editor_class"],
                "observed_target_count": len(keys),
                "draft_target_count": len(selected_keys),
                "metadata_join_status": "matched_efficiency_table" if eff_map else "no_matching_efficiency_table",
                "source_relation_to_model": "source_associated",
                "source_associated_models": "behive_bystander;behive_efficiency",
                "t2_role": "observed_bystander_profile",
                "notes": json.dumps(eff_stats, ensure_ascii=True),
            }
        )

        for source_key in selected_keys:
            df = outcome_dict[source_key]
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            pos_cols = position_columns(df)
            if "Frequency" not in df.columns or not pos_cols:
                continue

            eff_row = eff_map.get(str(source_key))
            if eff_row is not None:
                matched_eff += 1
            target_position = infer_target_position(eff_row, str(source_key))
            sequence_context = first_existing(
                eff_row,
                [
                    "Sequence context (56nt)",
                    "Sequence context (35 or 61 nt)",
                    "Sequence context prepended_x",
                    "Designed 61-bp target site (pathogenic)_x",
                    "Designed target_x",
                ],
            ) if eff_row is not None else ""
            guide = first_existing(eff_row, ["gRNA (20nt)", "Designed gRNA (NGG orientation)_x"]) if eff_row is not None else ""
            sequence_50nt = extract_50nt(sequence_context, guide)
            protospacer = clean_dna(guide)
            pam = infer_pam(sequence_context, guide)

            measured_total_eff = numeric_mean(eff_row, "Fraction edited ") if eff_row is not None else None
            total_read_count = numeric_sum(eff_row, "Total count ") if eff_row is not None else None
            edited_read_count = numeric_sum(eff_row, "Edited count ") if eff_row is not None else None
            outcome_read_count = clean_float(df["Count"].sum()) if "Count" in df.columns else None
            if edited_read_count is None:
                edited_read_count = outcome_read_count

            target_profile_id = stable_id(
                "T2T",
                pkl_path.stem,
                source_key,
                sequence_50nt,
                meta["editor"],
                meta["cell_context"],
            )
            freq_sum = clean_float(df["Frequency"].sum())
            max_product = clean_float(df["Frequency"].max())
            weighted_edit_count = 0.0
            bystander_freq = 0.0
            desired_product_freq = 0.0

            sorted_df = df.sort_values("Frequency", ascending=False).reset_index(drop=True)
            for out_rank, (_, outcome) in enumerate(sorted_df.iterrows(), start=1):
                freq = clean_float(outcome.get("Frequency")) or 0.0
                events, event_count, edited_by_pos = outcome_events(outcome, pos_cols)
                weighted_edit_count += freq * event_count
                has_bystander, is_desired = desired_product_flags(outcome, pos_cols, target_position)
                if has_bystander == "True":
                    bystander_freq += freq
                if is_desired == "True":
                    desired_product_freq += freq
                if args.max_outcomes_per_target == 0 or out_rank <= args.max_outcomes_per_target:
                    outcome_rows.append(
                        {
                            "outcome_profile_id": stable_id("T2O", target_profile_id, out_rank, events),
                            "target_profile_id": target_profile_id,
                            "source_profile_id": source_profile_id,
                            "source_row_key": source_key,
                            "editor": meta["editor"],
                            "editor_class": meta["editor_class"],
                            "cell_context": meta["cell_context"],
                            "outcome_rank_in_target": out_rank,
                            "observed_outcome_frequency": fmt_float(freq),
                            "outcome_count": fmt_float(outcome.get("Count")),
                            "outcome_edit_count": event_count,
                            "outcome_events": events or "no_tracked_substrate_change",
                            "has_bystander_edit": has_bystander,
                            "is_desired_product": is_desired,
                            "measurement_scale": "conditional_outcome_frequency_given_edited_reads",
                            "source_file": str(pkl_path),
                        }
                    )
            source_outcome_count += len(df)

            for col in pos_cols:
                match = POSITION_COL_RE.match(col)
                if not match:
                    continue
                ref_base = match.group(1)
                pos = int(match.group(2))
                position_rate = clean_float(df.loc[df[col].astype(str).str.upper() != ref_base, "Frequency"].sum())
                absolute_estimate = (
                    position_rate * measured_total_eff
                    if position_rate is not None and measured_total_eff is not None
                    else None
                )
                position_rows.append(
                    {
                        "position_profile_id": stable_id("T2P", target_profile_id, pos, ref_base),
                        "target_profile_id": target_profile_id,
                        "source_profile_id": source_profile_id,
                        "source_row_key": source_key,
                        "editor": meta["editor"],
                        "editor_class": meta["editor_class"],
                        "cell_context": meta["cell_context"],
                        "protospacer_position": pos,
                        "reference_base": ref_base,
                        "intended_position": "" if target_position is None else target_position,
                        "is_intended_position": "" if target_position is None else str(pos == target_position),
                        "observed_conditional_position_rate": fmt_float(position_rate),
                        "observed_absolute_position_rate_estimate": fmt_float(absolute_estimate),
                        "measurement_scale": "conditional_position_rate_given_edited_reads",
                        "outcome_row_count": len(df),
                        "edited_read_count": fmt_float(edited_read_count),
                        "source_file": str(pkl_path),
                    }
                )
            source_position_count += len(pos_cols)

            native_50nt_available = bool(sequence_50nt)
            protospacer_20nt_available = bool(protospacer and len(protospacer) == 20)
            qc_sequence_pass = protospacer_20nt_available or native_50nt_available
            qc_label_pass = freq_sum is not None and abs(freq_sum - 1.0) < 0.02
            target_rows.append(
                {
                    "target_profile_id": target_profile_id,
                    "source_profile_id": source_profile_id,
                    "source_family": "BE-Hive",
                    "source_dataset": meta["source_dataset"],
                    "source_file": str(pkl_path),
                    "source_row_key": source_key,
                    "source_relation_to_model": "source_associated",
                    "source_associated_models": "behive_bystander;behive_efficiency",
                    "editor": meta["editor"],
                    "editor_class": meta["editor_class"],
                    "editor_family": meta["editor_family"],
                    "cell_context": meta["cell_context"],
                    "replicate_strategy": "merged_replicate_outcome_profile",
                    "sequence_context_original": clean_dna(sequence_context),
                    "sequence_50nt": sequence_50nt,
                    "protospacer_20nt": protospacer,
                    "native_50nt_available": str(native_50nt_available),
                    "protospacer_20nt_available": str(protospacer_20nt_available),
                    "behive_bystander_input_eligible": str(native_50nt_available),
                    "forecast_position_input_eligible": str(protospacer_20nt_available),
                    "pam": pam,
                    "pam_type": pam_type(pam),
                    "target_position": "" if target_position is None else target_position,
                    "intended_edit": first_existing(eff_row, ["Correction edit_x", "Correction edit"]) if eff_row is not None else "",
                    "target_base": "" if not pos_cols else POSITION_COL_RE.match(pos_cols[0]).group(1),
                    "measured_total_efficiency": fmt_float(measured_total_eff),
                    "total_read_count": fmt_float(total_read_count),
                    "edited_read_count": fmt_float(edited_read_count),
                    "outcome_row_count": len(df),
                    "editable_position_count": len(pos_cols),
                    "observed_outcome_frequency_sum": fmt_float(freq_sum),
                    "observed_max_product_frequency": fmt_float(max_product),
                    "observed_mean_edited_positions_per_product": fmt_float(weighted_edit_count),
                    "observed_any_bystander_frequency": fmt_float(bystander_freq) if target_position is not None else "",
                    "observed_desired_product_frequency": fmt_float(desired_product_freq) if target_position is not None else "",
                    "measurement_scale": "conditional_outcome_frequency_given_edited_reads",
                    "qc_sequence_pass": str(qc_sequence_pass),
                    "qc_label_pass": str(qc_label_pass),
                    "qc_level": "B" if qc_sequence_pass and qc_label_pass else "C",
                    "exclusion_reason": "" if qc_sequence_pass and qc_label_pass else "missing_sequence_or_frequency_sum_check",
                    "notes": "Draft T2 observed BE-Hive bystander product profile; source-associated for BE-Hive model evaluation.",
                }
            )

        source_rows[-1]["draft_matched_efficiency_targets"] = matched_eff
        source_rows[-1]["draft_outcome_rows_total_before_cap"] = source_outcome_count
        source_rows[-1]["draft_position_columns_total"] = source_position_count

    source_rows.extend(build_source_inventory_extra())
    source_inventory = pd.DataFrame(source_rows).fillna("")
    target_df = pd.DataFrame(target_rows, columns=TARGET_COLUMNS).fillna("")
    position_df = pd.DataFrame(position_rows, columns=POSITION_COLUMNS).fillna("")
    outcome_df = pd.DataFrame(outcome_rows, columns=OUTCOME_COLUMNS).fillna("")

    source_inventory.to_csv(out_dir / "source_inventory_t2.csv", index=False)
    target_df.to_csv(out_dir / "T2_observed_target_profiles.csv", index=False)
    position_df.to_csv(out_dir / "T2_observed_position_profiles.csv", index=False)
    outcome_df.to_csv(out_dir / "T2_observed_outcome_profiles.csv", index=False)
    (out_dir / "T2_profile_schema.json").write_text(
        json.dumps(
            schema_payload(args.max_targets_per_source, args.max_outcomes_per_target, args.max_sources),
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        out_dir,
        source_inventory,
        target_df,
        position_df,
        outcome_df,
        args.max_targets_per_source,
        args.max_outcomes_per_target,
    )

    manifest = {
        "output_dir": str(out_dir),
        "source_rows": len(source_inventory),
        "target_rows": len(target_df),
        "position_rows": len(position_df),
        "outcome_rows": len(outcome_df),
        "max_sources": args.max_sources,
        "max_targets_per_source": args.max_targets_per_source,
        "max_outcomes_per_target": args.max_outcomes_per_target,
    }
    (out_dir / "T2_profile_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
