"""Benchmark-informed model routing.

This module intentionally does not fabricate numeric editing predictions.
It decides which reproduced model backends should be used, under what caveats,
and which benchmark evidence supports the recommendation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .registry import load_registry
from .sequence import assess_sequence, infer_editor_family


TASK_LABELS = {
    "auto": "Auto-select task",
    "scalar_efficiency": "Scalar editing efficiency",
    "outcome_profile": "Edited-product / bystander profile",
    "position_profile": "Position-level editing profile",
    "high_low_screening": "High-vs-low screening",
    "design_recommendation": "Design recommendation / top-k ranking",
}


def _task_matches(model: Dict[str, Any], task: str) -> bool:
    if task == "auto":
        return True
    return task in model.get("tasks", {})


def _editor_matches(model: Dict[str, Any], editor_family: str) -> bool:
    supported = set(model.get("editor_families", []))
    if "Any" in supported or editor_family == "Unknown":
        return True
    return editor_family in supported


def _input_penalties(model: Dict[str, Any], assessment: Dict[str, Any], editor_family: str) -> List[str]:
    penalties: List[str] = []
    input_type = assessment["inferred_input_type"]
    model_inputs = set(model.get("input_types", []))
    if "flexible_or_adapter_generated" not in model_inputs and input_type not in model_inputs:
        penalties.append(f"Input type {input_type} may need adapter-specific preprocessing.")

    if editor_family == "ABE" and not assessment["abe_candidate_positions_1based"]:
        penalties.append("ABE route warning: no editable A in positions 4-8.")
    if editor_family == "CBE" and not assessment["cbe_candidate_positions_1based"]:
        penalties.append("CBE route warning: no editable C in positions 4-8.")
    if assessment["is_ngg_like"] is False and model.get("common_core_ngg_sensitive", False):
        penalties.append("Benchmark evidence is strongest for NGG/SpCas9-like inputs.")
    return penalties


def _recommended_tasks(assessment: Dict[str, Any], editor_family: str) -> List[str]:
    tasks = ["scalar_efficiency", "high_low_screening"]
    if editor_family in {"ABE", "CBE", "Unknown"}:
        tasks.extend(["outcome_profile", "position_profile"])
    if assessment["length"] >= 20:
        tasks.append("design_recommendation")
    return tasks


def route_models(sequence: str, editor: str = "ABE", task: str = "auto") -> Dict[str, Any]:
    registry = load_registry()
    assessment_obj = assess_sequence(sequence)
    assessment = assessment_obj.to_dict()
    editor_family = infer_editor_family(editor)
    selected_tasks = _recommended_tasks(assessment, editor_family) if task == "auto" else [task]

    rows: List[Dict[str, Any]] = []
    for model in registry["models"]:
        task_entries = []
        for current_task in selected_tasks:
            if not _task_matches(model, current_task):
                continue
            if not _editor_matches(model, editor_family):
                continue
            role = model.get("tasks", {}).get(current_task, {})
            penalties = _input_penalties(model, assessment, editor_family)
            base_tier = role.get("confidence_tier", model.get("default_confidence_tier", "research_only"))
            if penalties and base_tier == "high":
                display_tier = "moderate_with_input_caveat"
            elif penalties and base_tier == "moderate":
                display_tier = "low_with_input_caveat"
            else:
                display_tier = base_tier
            task_entries.append(
                {
                    "task": current_task,
                    "task_label": TASK_LABELS.get(current_task, current_task),
                    "role": role.get("role", "candidate"),
                    "confidence_tier": display_tier,
                    "benchmark_basis": role.get("benchmark_basis", ""),
                    "benchmark_summary": role.get("benchmark_summary", ""),
                    "warnings": penalties,
                }
            )
        if not task_entries:
            continue
        rows.append(
            {
                "model_id": model["model_id"],
                "display_name": model["display_name"],
                "integration_status": model["integration_status"],
                "runtime_family": model["runtime_family"],
                "license_reuse_note": model["license_reuse_note"],
                "local_path": model["local_path"],
                "task_routes": task_entries,
                "system_use": model.get("system_use", "research-use routing only"),
                "major_caveat": model.get("major_caveat", ""),
            }
        )

    rows.sort(key=_route_sort_key)
    return {
        "system_name": registry["system_name"],
        "system_version": registry["system_version"],
        "input": {
            "editor": editor,
            "editor_family": editor_family,
            "requested_task": task,
            "selected_tasks": selected_tasks,
        },
        "sequence_assessment": assessment,
        "routes": rows,
        "global_interpretation": registry["global_interpretation"],
        "canonical_evidence": registry["canonical_evidence"],
    }


def _route_sort_key(row: Dict[str, Any]) -> tuple:
    tier_order = {
        "high": 0,
        "moderate_with_input_caveat": 1,
        "moderate": 2,
        "low_with_input_caveat": 3,
        "restricted": 4,
        "research_only": 5,
    }
    best = 99
    for entry in row["task_routes"]:
        best = min(best, tier_order.get(entry["confidence_tier"], 50))
    return best, row["display_name"]

