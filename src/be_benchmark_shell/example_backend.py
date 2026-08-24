"""Connector template. It contains no model implementation or prediction value."""

from be_benchmark_shell.contracts import PredictionRequest


def predict(request: PredictionRequest) -> dict:
    raise NotImplementedError("Implement this connector only in a licensed private deployment")
