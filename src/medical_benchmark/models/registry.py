from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from medical_benchmark.config import ROOT, load_yaml

MODEL_REGISTRY = {
    "mambavision": "configs/models/mambavision.yaml",
    "transnext": "configs/models/transnext.yaml",
    "chexworld": "configs/models/chexworld.yaml",
    "lrfl": "configs/models/lrfl.yaml",
    "x_win": "configs/models/x_win.yaml",
    "carzero": "configs/models/carzero.yaml",
}

ALLOWED_BLOCK_REASONS = {
    "missing_source",
    "source_revision_mismatch",
    "missing_checkpoint",
    "checkpoint_hash_unconfigured",
    "checkpoint_hash_mismatch",
    "unavailable_implementation",
}


class BlockedModelError(RuntimeError):
    def __init__(self, model: str, reason: str, detail: str) -> None:
        if reason not in ALLOWED_BLOCK_REASONS:
            raise ValueError(f"unknown blocked reason: {reason}")
        self.model = model
        self.reason = reason
        self.detail = detail
        super().__init__(json.dumps({"status": "BLOCKED", "model": model, "reason": reason, "detail": detail}))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {model}")
    model_config = load_yaml(MODEL_REGISTRY[model])
    sources = load_yaml("configs/source_lock.yaml")["sources"]
    return model_config, sources[model_config["source"]]


def validate_model(model: str) -> dict[str, Any]:
    model_config, source = _source_config(model)
    if model in {"lrfl"}:
        raise BlockedModelError(model, "missing_checkpoint", "the official project has released no LRFL checkpoint")
    if model == "x_win":
        raise BlockedModelError(model, "unavailable_implementation", "official downstream extraction and a checkpoint are unavailable")

    source_path = ROOT / source["path"]
    if not (source_path / ".git").exists():
        raise BlockedModelError(model, "missing_source", f"clone the locked source at {source_path}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BlockedModelError(model, "missing_source", f"cannot inspect source: {exc}") from exc
    if revision != source["commit"]:
        raise BlockedModelError(model, "source_revision_mismatch", f"expected {source['commit']}, found {revision}")

    checkpoints = source.get("checkpoints") or ([source["checkpoint"]] if source.get("checkpoint") else [])
    if not checkpoints:
        raise BlockedModelError(model, "missing_checkpoint", "the official project has released no checkpoint")
    resolved: list[Path] = []
    for checkpoint in checkpoints:
        expected = checkpoint.get("sha256")
        if not expected:
            raise BlockedModelError(model, "checkpoint_hash_unconfigured", f"set a verified sha256 for {checkpoint['path']}")
        path = ROOT / checkpoint["path"]
        if not path.is_file():
            raise BlockedModelError(model, "missing_checkpoint", f"checkpoint not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise BlockedModelError(model, "checkpoint_hash_mismatch", f"{path}: expected {expected}, found {actual}")
        resolved.append(path)
    return {"model": model_config, "source": source, "source_path": source_path, "checkpoints": resolved}


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint does not contain a state dictionary")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def _replace_linear_head(model: nn.Module, attribute: str, classes: int) -> None:
    head = getattr(model, attribute, None)
    if not isinstance(head, nn.Linear):
        raise TypeError(f"expected {attribute} to be torch.nn.Linear, found {type(head).__name__}")
    setattr(model, attribute, nn.Linear(head.in_features, classes))


def build_model(name: str) -> nn.Module:
    locked = validate_model(name)  # Availability and hashes are checked before any upstream import.
    config, source_path, checkpoint = locked["model"], locked["source_path"], locked["checkpoints"][0]
    if name == "mambavision":
        sys.path.insert(0, str(source_path))
        try:
            module = importlib.import_module("mambavision.models.mamba_vision")
            model = getattr(module, config["factory"])(pretrained=False)
        finally:
            sys.path.remove(str(source_path))
    elif name == "transnext":
        module_path, factory = config["factory"].split(":", 1)
        sys.path.insert(0, str((source_path / module_path).parent))
        try:
            module = importlib.import_module(Path(module_path).stem)
            model = getattr(module, factory)(pretrained=False)
        finally:
            sys.path.remove(str((source_path / module_path).parent))
    else:
        # CheXWorld/CARZero are intentionally unreachable until local no-hash artifacts are pinned.
        raise BlockedModelError(name, "unavailable_implementation", "verified upstream feature adapter is not available")

    state = _state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    model.load_state_dict(state, strict=True)  # Load the official 1000-class head before replacing it.
    _replace_linear_head(model, config["head"], int(config["num_classes"]))
    return model
