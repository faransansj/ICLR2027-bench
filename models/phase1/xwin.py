"""X-WIN CXR-only adapter declaration.

Official downstream feature extraction and pretrained weights are unavailable;
CT input or an unofficial substitute is never introduced.
"""
from medical_benchmark.models.registry import build_model


def create_model():
    return build_model("xwin")
