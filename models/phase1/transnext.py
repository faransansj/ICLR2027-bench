"""Official TransNeXt-Micro adapter: load weights, then Linear(11)."""
from medical_benchmark.models.registry import build_model


def create_model():
    return build_model("transnext")
