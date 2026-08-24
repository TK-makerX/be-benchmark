from __future__ import annotations

from pathlib import Path

from .contracts import PredictionRequest
from .registry import load_backend_specs
from .router import select_backend


def evaluate_request(
    sequence: str,
    editor_family: str,
    task: str,
    registry: Path | None = None,
) -> dict:
    request = PredictionRequest.build(sequence, editor_family, task)
    decision = select_backend(request, load_backend_specs(registry))
    return {
        "request": request.public_summary(),
        **decision,
        "message": "This public scaffold validates and routes requests but does not run numerical inference.",
    }
