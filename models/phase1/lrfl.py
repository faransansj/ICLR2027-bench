"""LRFL adapter declaration preserving the official low-rank mechanism.

The official project released no LRFL checkpoint, so construction is BLOCKED
rather than replacing its mechanism.
"""
from medical_benchmark.models.registry import build_model


def create_model():
    return build_model("lrfl")
