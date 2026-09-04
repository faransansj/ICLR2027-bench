from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from medical_benchmark.config import ROOT, load_yaml

MODEL_REGISTRY = {
    "mambavision": {
        "milk10k": "configs/models/phase1/mambavision.yaml",
        "chexchonet": "configs/models/phase2/mambavision_chexchonet.yaml",
    },
    "transnext": {
        "milk10k": "configs/models/phase1/transnext.yaml",
        "chexchonet": "configs/models/phase2/transnext_chexchonet.yaml",
    },
    "chexworld": {"chexchonet": "configs/models/phase1/chexworld.yaml"},
    "lrfl": {"chexchonet": "configs/models/phase1/lrfl.yaml"},
    "xwin": {"chexchonet": "configs/models/phase1/xwin.yaml"},
    "carzero": {"chexchonet": "configs/models/phase1/carzero.yaml"},
}

ALLOWED_BLOCK_REASONS = {
    "missing source",
    "missing checkpoint",
    "dependency incompatibility",
    "dataset incompatibility",
    "unavailable implementation",
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


def model_config_path(model: str, dataset: str | None = None) -> str:
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {model}")
    configs = MODEL_REGISTRY[model]
    dataset = dataset or ("milk10k" if model in {"mambavision", "transnext"} else "chexchonet")
    if dataset not in configs:
        raise ValueError(f"{model} is not configured for {dataset}")
    return configs[dataset]


def _source_config(model: str, dataset: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    model_config = load_yaml(model_config_path(model, dataset))
    sources = load_yaml("configs/source_lock.yaml")["sources"]
    return model_config, sources[model_config["source"]]


def validate_model(model: str, dataset: str | None = None) -> dict[str, Any]:
    model_config, source = _source_config(model, dataset)
    if model in {"lrfl"}:
        raise BlockedModelError(model, "missing checkpoint", "the official project has released no LRFL checkpoint")
    if model == "xwin":
        raise BlockedModelError(model, "unavailable implementation", "official downstream extraction and a checkpoint are unavailable")

    source_path = ROOT / source["path"]
    if not (source_path / ".git").exists():
        raise BlockedModelError(model, "missing source", f"clone the locked source at {source_path}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BlockedModelError(model, "missing source", f"cannot inspect source: {exc}") from exc
    if revision != source["commit"]:
        raise BlockedModelError(model, "missing source", f"source revision mismatch: expected {source['commit']}, found {revision}")

    checkpoints = source.get("checkpoints") or ([source["checkpoint"]] if source.get("checkpoint") else [])
    if not checkpoints:
        raise BlockedModelError(model, "missing checkpoint", "the official project has released no checkpoint")
    resolved: list[Path] = []
    for checkpoint in checkpoints:
        expected = checkpoint.get("sha256")
        if not expected:
            raise BlockedModelError(model, "missing checkpoint", f"set a verified sha256 for {checkpoint['path']}")
        path = ROOT / checkpoint["path"]
        if not path.is_file():
            raise BlockedModelError(model, "missing checkpoint", f"checkpoint not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise BlockedModelError(model, "missing checkpoint", f"checkpoint hash mismatch for {path}: expected {expected}, found {actual}")
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


def _make_transnext_pooling_deterministic(model: nn.Module) -> None:
    for module in model.modules():
        pool = getattr(module, "pool", None)
        ratio = getattr(module, "sr_ratio", None)
        if isinstance(pool, nn.AdaptiveAvgPool2d) and isinstance(ratio, int):
            module.pool = nn.AvgPool2d(ratio, ratio)


def _chexworld_modules(source_path: Path) -> tuple[Any, Any]:
    package_name = "_medical_benchmark_chexworld"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_path / "models")]
        sys.modules[package_name] = package
    return (
        importlib.import_module(f"{package_name}.jepa_vit"),
        importlib.import_module(f"{package_name}.finetune"),
    )


def build_model(name: str, dataset: str | None = None) -> nn.Module:
    locked = validate_model(name, dataset)  # Availability and hashes are checked before any upstream import.
    config, source_path, checkpoint = locked["model"], locked["source_path"], locked["checkpoints"][0]
    try:
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
        elif name == "chexworld":
            jepa_vit, fine_tune = _chexworld_modules(source_path)
            encoder = jepa_vit.vit_base(
                img_size=int(config["input_size"]), patch_size=16, drop_path_rate=0.0, drop_path_uniform=False
            )
            raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
                raise ValueError("CheXWorld checkpoint must contain model state")
            target = {
                key.removeprefix("target_encoder."): value
                for key, value in raw["model"].items()
                if key.startswith("target_encoder.")
            }
            if not target:
                raise ValueError("CheXWorld checkpoint contains no target_encoder weights")
            encoder.load_state_dict(target, strict=True)
            model = fine_tune.FineTuner(
                encoder, feature_dim=768, num_classes=int(config["num_classes"]), tune_type="fc", with_cls_token=False
            )
        else:
            raise BlockedModelError(name, "unavailable implementation", "verified upstream feature adapter is not available")
    except ImportError as exc:
        raise BlockedModelError(name, "dependency incompatibility", str(exc)) from exc

    if name == "chexworld":
        return model
    state = _state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    model.load_state_dict(state, strict=True)  # Load the official 1000-class head before replacing it.
    _replace_linear_head(model, config["head"], int(config["num_classes"]))
    if name == "transnext":
        _make_transnext_pooling_deterministic(model)
    return model
