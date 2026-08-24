from __future__ import annotations

from .backends import BackendSpec
from .contracts import PredictionRequest


def select_backend(request: PredictionRequest, backends: list[BackendSpec]) -> dict:
    if not backends:
        return {"status": "backend_not_configured", "backend_id": None}
    compatible = [backend for backend in backends if backend.supports(request)]
    if not compatible:
        return {"status": "no_compatible_backend", "backend_id": None}
    selected = sorted(compatible, key=lambda item: item.backend_id)[0]
    return {"status": "backend_declared_but_not_connected", "backend_id": selected.backend_id}
