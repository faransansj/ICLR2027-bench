"""CARZero-FT image-representation adapter declaration (not zero-shot).

Construction remains blocked until both official Drive checkpoints have
locally recorded and verified SHA256 values.
"""
from medical_benchmark.models.registry import build_model


def create_model():
    return build_model("carzero")
