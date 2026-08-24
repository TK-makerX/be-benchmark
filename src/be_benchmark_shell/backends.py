from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import PredictionRequest


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    enabled: bool
    editor_families: tuple[str, ...]
    tasks: tuple[str, ...]
    connector: str

    def supports(self, request: PredictionRequest) -> bool:
        return (
            self.enabled
            and request.editor_family in self.editor_families
            and request.task in self.tasks
        )


class BackendConnector(Protocol):
    def predict(self, request: PredictionRequest) -> dict:
        ...


class UnavailableConnector:
    def predict(self, request: PredictionRequest) -> dict:
        raise RuntimeError("No numerical backend is installed in the public scaffold")
