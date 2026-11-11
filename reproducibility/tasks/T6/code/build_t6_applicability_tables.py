#!/usr/bin/env python
"""Summarize model applicability, runtime and coverage from the T6 registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "model_reproducibility_registry_T6.csv"
OUT = ROOT / "reports" / "t6_applicability_tables"


TASK_COLUMNS = [
    "T1_applicability",
    "T2_applicability",
    "T3_applicability",
    "T4_applicability",
    "T5_applicability",
    "T6_applicability",
    "T7_applicability",
]


def status_to_level(value: str) -> str:
    text = str(value).lower()
    if "completed" in text:
        return "completed"
    if "possible" in text:
        return "possible"
    if "pilot" in text or "derived" in text:
        return "pilot_or_derived"
    if "optional" in text:
        return "optional"
    if "not_applicable" in text or "not_run" in text or "not_primary" in text:
        return "not_primary"
    if not text:
        return "not_recorded"
    return "limited"


def status_to_numeric(level: str) -> int:
    return {
        "completed": 4,
        "pilot_or_derived": 3,
        "possible": 2,
        "optional": 1,
        "limited": 1,
        "not_primary": 0,
        "not_recorded": 0,
    }.get(level, 0)


def environment_complexity(row: pd.Series) -> dict[str, object]:
    env_sensitive = "activation_sensitive" in str(row["windows_compatibility"])
    wsl = "wsl" in str(row["wsl_or_external_tool_requirement"]).lower()
    external = any(
        token in str(row["wsl_or_external_tool_requirement"]).lower()
        for token in ["bedtools", "endo_factors", "external"]
    )
    cuda = "upstream_docs" in str(row["cuda_gpu_requirement"]).lower() or "may_need_gpu" in str(row["cuda_gpu_requirement"]).lower()
    legacy = any(token in str(row["runtime_family"]).lower() for token in ["legacy", "python2", "tf1"])
    method_only = row["benchmark_tier"] == "method_only"
    score = sum(int(x) for x in [env_sensitive, wsl, external, cuda, legacy, method_only])
    return {
        "model_id": row["model_id"],
        "display_name": row["display_name"],
        "runtime_family": row["runtime_family"],
        "activation_sensitive": env_sensitive,
        "requires_wsl_helper": wsl,
        "requires_external_resource": external,
        "upstream_or_scale_gpu_risk": cuda,
        "legacy_runtime_risk": legacy,
        "method_only_pipeline": method_only,
        "complexity_score": score,
    }


def burden_level(text: str) -> int:
    low = str(text).lower()
    if low.startswith("low"):
        return 1
    if low.startswith("moderate"):
        return 2
    if low.startswith("high"):
        return 3
    return 0


def reproducibility_class(row: pd.Series) -> str:
    status = str(row["install_run_reuse_status"]).lower()
    tier = str(row["benchmark_tier"]).lower()
    if str(row["access_mode"]).startswith("web"):
        return "web_only_or_candidate"
    if "method_only" in tier:
        return "method_only_pipeline"
    if "t1_t2_t3" in status:
        return "formal_multi_task_validated"
    if "t1_v6" in status:
        return "formal_t1_validated"
    if "t2_t3" in status:
        return "formal_profile_tasks_validated"
    if "pilot" in status:
        return "pilot_validated"
    if "partial" in tier or "sequence_only" in status:
        return "partial_reproduction"
    return "smoke_or_unclassified"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REGISTRY, dtype=str).fillna("")
    local = df[df["access_mode"].eq("local")].copy()

    task_rows = []
    for _, row in local.iterrows():
        for col in TASK_COLUMNS:
            task = col.split("_", 1)[0]
            level = status_to_level(row[col])
            task_rows.append(
                {
                    "model_id": row["model_id"],
                    "display_name": row["display_name"],
                    "task": task,
                    "status_label": row[col],
                    "applicability_level": level,
                    "applicability_score": status_to_numeric(level),
                }
            )
    task_matrix = pd.DataFrame(task_rows)
    task_matrix.to_csv(OUT / "T6_model_task_applicability_matrix.csv", index=False)

    env = pd.DataFrame([environment_complexity(row) for _, row in local.iterrows()])
    env.to_csv(OUT / "T6_environment_complexity_matrix.csv", index=False)

    burden = local[
        [
            "model_id",
            "display_name",
            "primary_output_kind",
            "input_contract",
            "input_burden",
            "practical_limits",
        ]
    ].copy()
    burden["input_burden_score"] = burden["input_burden"].map(burden_level)
    burden.to_csv(OUT / "T6_input_requirement_burden.csv", index=False)

    status = local[
        [
            "model_id",
            "display_name",
            "benchmark_tier",
            "install_run_reuse_status",
            "pilot_schema_validated",
            "T1_v6_all_returncode_zero",
            "T2_run_status",
            "T3_include_label",
            "T6_timing_status",
            "T6_timing_elapsed_seconds",
            "T6_timing_seconds_per_1000_panel_rows",
            "runtime_per_1000_examples",
            "failure_rate_or_coverage_evidence",
        ]
    ].copy()
    status["reproducibility_class"] = local.apply(reproducibility_class, axis=1)
    status.to_csv(OUT / "T6_local_reproducibility_status_summary.csv", index=False)

    coverage = local[
        [
            "model_id",
            "display_name",
            "T1_input_coverage",
            "T2_skipped_rows",
            "T3_position_coverage",
            "T3_missing_positions_zero_filled",
            "T3_unmatched_prediction_positions_outside_universe",
            "T6_timing_skipped_rows",
            "T6_timing_elapsed_seconds",
            "T6_timing_seconds_per_1000_panel_rows",
            "failure_rate_or_coverage_evidence",
        ]
    ].copy()
    coverage.to_csv(OUT / "T6_coverage_and_failure_evidence.csv", index=False)

    license_audit = local[
        [
            "model_id",
            "display_name",
            "license_review_status",
            "license_evidence_type",
            "inferred_license_family",
            "redistribution_reuse_assessment",
            "license_review_note",
            "local_license_files",
        ]
    ].copy()
    license_audit.to_csv(OUT / "T6_license_redistribution_audit.csv", index=False)

    print(f"Wrote T6 applicability tables to {OUT}")
    print(task_matrix.head().to_string(index=False))


if __name__ == "__main__":
    main()
