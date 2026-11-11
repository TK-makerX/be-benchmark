"""Service layer for CLI and local web UI."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .model_execution import run_selected_backends
from .router import route_models


def analyze(
    sequence: str,
    editor: str = "ABE",
    task: str = "auto",
    run_models: bool = False,
    backend_ids: List[str] | None = None,
    cell_context: str = "",
    cas_variant: str = "",
    reference_outcome: str = "",
) -> Dict[str, Any]:
    result = route_models(sequence=sequence, editor=editor, task=task)
    result["input"].update(
        {
            "cell_context": cell_context,
            "cas_variant": cas_variant,
            "reference_outcome": reference_outcome,
        }
    )
    if run_models:
        result["model_outputs"] = run_selected_backends(
            sequence=result["sequence_assessment"]["sequence"],
            editor=editor,
            editor_family=result["input"]["editor_family"],
            backend_ids=backend_ids,
            cell_context=cell_context,
            cas_variant=cas_variant,
            reference_outcome=reference_outcome,
        )
    else:
        result["model_outputs"] = []
    return result


def _route_line(route: Dict[str, Any]) -> str:
    parts: List[str] = []
    for task in route["task_routes"]:
        warn = ""
        if task["warnings"]:
            warn = " | warnings: " + "; ".join(task["warnings"])
        parts.append(
            f"  - {task['task_label']}: {task['confidence_tier']} | {task['benchmark_summary']}{warn}"
        )
    caveat = f"\n    caveat: {route['major_caveat']}" if route.get("major_caveat") else ""
    return (
        f"* {route['display_name']} ({route['model_id']})\n"
        f"  integration: {route['integration_status']} | runtime: {route['runtime_family']}\n"
        + "\n".join(parts)
        + caveat
    )


def format_text_report(result: Dict[str, Any], max_models: int = 12) -> str:
    assessment = result["sequence_assessment"]
    lines = [
        f"# {result['system_name']} v{result['system_version']}",
        "",
        f"Editor: {result['input']['editor']} -> inferred family: {result['input']['editor_family']}",
        f"Task request: {result['input']['requested_task']}",
        f"Cell context: {result['input'].get('cell_context') or '(not provided)'}",
        f"Cas/PAM variant: {result['input'].get('cas_variant') or '(inferred/default)'}",
        f"Reference outcome: {result['input'].get('reference_outcome') or '(not provided)'}",
        f"Sequence length: {assessment['length']} ({assessment['inferred_input_type']})",
        f"PAM: {assessment['pam']} | NGG-like: {assessment['is_ngg_like']}",
        f"ABE editable positions 4-8: {assessment['abe_candidate_positions_1based']}",
        f"CBE editable positions 4-8: {assessment['cbe_candidate_positions_1based']}",
    ]
    if assessment["warnings"]:
        lines.extend(["", "Input warnings:"])
        lines.extend([f"- {warning}" for warning in assessment["warnings"]])
    lines.extend(["", "Recommended model routes:"])
    if not result["routes"]:
        lines.append("- No route matched this task/editor/input combination.")
    else:
        for route in result["routes"][:max_models]:
            lines.append(_route_line(route))
    if result.get("model_outputs"):
        lines.extend(["", "Real model outputs:"])
        for output in result["model_outputs"]:
            if output.get("status") == "completed":
                name = output.get("display_name", output["model_id"])
                if "predicted_efficiency_score" in output:
                    lines.append(
                        f"- {name}: completed | score={output.get('predicted_efficiency_score')} "
                        f"scale={output.get('score_scale')}"
                    )
                    scored_positions = [
                        f"{row['position']}:{row['score']:.4g}"
                        for row in output.get("position_scores", [])
                        if row.get("score") is not None
                    ]
                    if scored_positions:
                        lines.append("  position scores: " + ", ".join(scored_positions))
                else:
                    summary = output.get("prediction_summary", {})
                    lines.append(
                        f"- {name}: completed | predictions={output.get('prediction_count')} "
                        f"skipped={output.get('skipped_count', 0)}"
                    )
                    eff_items = summary.get("efficiency") or []
                    if eff_items:
                        eff_text = []
                        for item in eff_items[:3]:
                            eff_text.append(
                                f"eff={item.get('predicted_efficiency')} score={item.get('predicted_score')}"
                            )
                        lines.append("  efficiency: " + "; ".join(eff_text))
                    pos_items = summary.get("position_scores") or []
                    if pos_items:
                        pos_text = [
                            f"{item['position']}:{item.get('score'):.4g}"
                            for item in pos_items[:10]
                            if item.get("score") is not None
                        ]
                        if pos_text:
                            lines.append("  position scores: " + ", ".join(pos_text))
                    outcome_items = summary.get("top_outcomes") or []
                    if outcome_items:
                        outcome_text = []
                        for item in outcome_items[:5]:
                            value = item.get("proportion")
                            if value is None:
                                value = item.get("frequency")
                            if value is None:
                                value = item.get("score")
                            outcome_text.append(f"{item.get('outcome')}={value}")
                        lines.append("  top outcomes: " + "; ".join(outcome_text))
            else:
                lines.append(f"- {output['model_id']}: {output.get('status')} | {output.get('reason', '')}")
    lines.extend(
        [
            "",
            "Interpretation guardrail:",
            result["global_interpretation"],
            "",
            "Canonical evidence:",
        ]
    )
    for key, value in result["canonical_evidence"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def to_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)
