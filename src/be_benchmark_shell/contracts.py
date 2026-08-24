from __future__ import annotations

from dataclasses import dataclass


DNA_ALPHABET = frozenset("ACGTN")


@dataclass(frozen=True)
class PredictionRequest:
    sequence: str
    editor_family: str
    task: str

    @classmethod
    def build(cls, sequence: str, editor_family: str, task: str) -> "PredictionRequest":
        request = cls(
            sequence="".join(sequence.upper().split()),
            editor_family=editor_family.strip().lower(),
            task=task.strip().lower(),
        )
        request.validate()
        return request

    def validate(self) -> None:
        if not 20 <= len(self.sequence) <= 4096:
            raise ValueError("sequence length must be between 20 and 4096 bases")
        invalid = sorted(set(self.sequence) - DNA_ALPHABET)
        if invalid:
            raise ValueError(f"sequence contains unsupported symbols: {''.join(invalid)}")
        if not self.editor_family:
            raise ValueError("editor_family is required")
        if not self.task:
            raise ValueError("task is required")

    def public_summary(self) -> dict:
        return {
            "sequence_length": len(self.sequence),
            "editor_family": self.editor_family,
            "task": self.task,
        }
