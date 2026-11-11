#!/usr/bin/env python
"""
Readiness check for unified base-editing benchmark matrix v5.

This script distinguishes candidate-source audit files from the actual benchmark
matrix files that task runners can consume.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from runtime_paths import CORE_ROOT, DATA_ROOT, REPORT_ROOT

import pandas as pd


ROOT = CORE_ROOT
DATA = DATA_ROOT / "unified_matrix_v5"
REPORT = REPORT_ROOT / "UNIFIED_MATRIX_V5_READINESS_CHECK.md"
READINESS_CSV = DATA / "readiness_check_v5.csv"


REQUIRED_FILES = [
    ("design_doc", ROOT / "docs" / "UNIFIED_MATRIX_V5_DESIGN.md", "planning"),
    ("schema", ROOT / "config" / "unified_matrix_schema_v5.json", "planning"),
    ("task_definitions", ROOT / "config" / "benchmark_tasks_v5.csv", "planning"),
    ("candidate_inventory", DATA / "source_candidate_inventory_v5.csv", "audit"),
    ("candidate_summary", DATA / "source_candidate_summary_v5.csv", "audit"),
    ("source_inventory", DATA / "source_inventory_v5.csv", "matrix_core"),
    ("design_master", DATA / "design_master_v5.csv", "matrix_core"),
    ("editor_context_master", DATA / "editor_context_master_v5.csv", "matrix_core"),
    ("observation_matrix", DATA / "observation_matrix_v5.csv", "matrix_core"),
    ("qc_report", ROOT / "reports" / "UNIFIED_MATRIX_V5_QC_REPORT.md", "qc"),
    ("task_view_T1", DATA / "task_views" / "T1_total_efficiency_view_v5.csv", "task_view"),
    ("task_view_T2", DATA / "task_views" / "T2_high_low_efficiency_view_v5.csv", "task_view"),
    ("task_view_T3", DATA / "task_views" / "T3_outcome_bystander_view_v5.csv", "task_view"),
    ("task_view_T4", DATA / "task_views" / "T4_position_profile_view_v5.csv", "task_view"),
    ("task_view_T5", DATA / "task_views" / "T5_generalization_view_v5.csv", "task_view"),
    ("task_view_T6", DATA / "task_views" / "T6_reproducibility_applicability_view_v5.csv", "task_view"),
    ("task_view_T7", DATA / "task_views" / "T7_editor_recommendation_view_v5.csv", "task_view"),
]


def file_rows(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() != ".csv":
        return None
    try:
        return len(pd.read_csv(path, dtype=str, low_memory=False))
    except Exception:
        return None


def build_status() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name, path, stage in REQUIRED_FILES:
        exists = path.exists()
        rows.append(
            {
                "component": name,
                "stage": stage,
                "exists": exists,
                "rows": file_rows(path),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def summarize_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_path = DATA / "source_candidate_inventory_v5.csv"
    summary_path = DATA / "source_candidate_summary_v5.csv"
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    return inventory, summary


def readiness_decision(status: pd.DataFrame) -> tuple[str, str]:
    core = status[status["stage"].eq("matrix_core")]
    views = status[status["stage"].eq("task_view")]
    audit_ready = bool(status[status["stage"].eq("audit")]["exists"].all())
    core_ready = bool(core["exists"].all())
    views_ready = bool(views["exists"].all())

    if core_ready and views_ready:
        return "READY_FOR_MODEL_RUNS", "Core matrix and all task views exist."
    if audit_ready and not core_ready:
        return (
            "NOT_READY_AUDIT_ONLY",
            "Candidate-source audit exists, but core matrix tables are not built yet.",
        )
    if core_ready and not views_ready:
        return (
            "PARTIALLY_READY_CORE_ONLY",
            "Core matrix exists, but task-specific views are incomplete.",
        )
    return "NOT_READY", "Planning/audit files are incomplete."


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy().fillna("")
    lines = [
        "| " + " | ".join(map(str, out.columns)) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[c]).replace("|", "\\|").replace("\n", " ") for c in out.columns)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    status = build_status()
    status.to_csv(READINESS_CSV, index=False)
    inventory, summary = summarize_candidates()
    decision, reason = readiness_decision(status)
    if decision == "READY_FOR_MODEL_RUNS":
        interpretation = (
            "The unified matrix core tables and all task views have been constructed. "
            "The matrix is ready for downstream model-run planning, with source-aware "
            "and measurement-scale-aware filtering applied at evaluation time."
        )
        next_step = (
            "Proceed to model-run planning on task views. Recommended order: start with "
            "T1 using a source-balanced subset, then T2/T5, then T3/T4/T7 as model adapters allow."
        )
    elif decision == "NOT_READY_AUDIT_ONLY":
        interpretation = (
            "The project currently has a validated design/schema and a candidate-source audit. "
            "That is not the same as having a completed unified benchmark matrix. The core "
            "tables and task views still need to be constructed before full-scale T1-T7 model runs."
        )
        next_step = (
            "Build the core matrix tables: source_inventory_v5.csv, design_master_v5.csv, "
            "editor_context_master_v5.csv, observation_matrix_v5.csv, QC report and T1-T7 task views."
        )
    else:
        interpretation = (
            "The matrix is partially prepared. Inspect the missing component table before running models."
        )
        next_step = "Complete the missing core/task-view components, then rerun this readiness check."

    task_status = status[status["stage"].eq("task_view")].copy()
    task_status["ready_for_run"] = task_status["exists"]

    if not summary.empty:
        summary_show = summary.copy()
    else:
        summary_show = pd.DataFrame()

    missing_core = status[
        status["stage"].isin(["matrix_core", "qc", "task_view"]) & ~status["exists"]
    ][["component", "stage", "path"]]

    text = f"""# Unified matrix v5 readiness check

Generated: {datetime.now().isoformat(timespec="seconds")}

## Decision

**{decision}**

{reason}

## Important interpretation

{interpretation}

## Candidate source summary

{md_table(summary_show)}

## Component status

{md_table(status[["component", "stage", "exists", "rows", "path"]])}

## Missing components before full model runs

{md_table(missing_core)}

## Next required step

{next_step}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(decision)
    print(reason)
    print(f"Status CSV: {READINESS_CSV}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
