"""CheXWorld target-encoder adapter declaration.

Construction remains blocked until the official checkpoint has a locally
recorded and verified SHA256; no alternative encoder is substituted.
"""
from medical_benchmark.models.registry import build_model


def create_model():
    return build_model("chexworld")
