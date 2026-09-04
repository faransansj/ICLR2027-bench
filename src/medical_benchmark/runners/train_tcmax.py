from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from medical_benchmark.config import ROOT, load_yaml, merge_runtime
from medical_benchmark.datasets.milk10k_phase2 import build_milk10k_phase2_datasets
from medical_benchmark.metrics import compute_metrics, require_finite
from medical_benchmark.models.registry import sha256_file
from medical_benchmark.runners.train import _git, _json_hash, _provenance, _write_predictions, set_deterministic


class TCMaxMilk10k(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.clinical = self._encoder(classes)
        self.dermoscopic = self._encoder(classes)

    @staticmethod
    def _encoder(classes: int) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model

    def forward(self, clinical: torch.Tensor, dermoscopic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clinical_logits = self.clinical(clinical)
        dermoscopic_logits = self.dermoscopic(dermoscopic)
        return clinical_logits + dermoscopic_logits, clinical_logits, dermoscopic_logits


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def paired_transforms(input_size: int) -> dict[str, Any]:
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return {"clinical": transform, "dermoscopic": transform}


def tcmax_loss(clinical_logits: torch.Tensor, dermoscopic_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch = labels.shape[0]
    joint = clinical_logits[:, None, :] + dermoscopic_logits[None, :, :]
    prob = torch.exp(joint - joint.mean())
    prob = prob / prob.sum().clamp_min(1e-12)
    index = torch.arange(batch, device=labels.device)
    return -torch.log(prob[index, index, labels].clamp_min(1e-12)).mean() - math.log(batch)


def _loader(dataset: Any, runtime: dict[str, Any], shuffle: bool) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(int(runtime["seed"]))
    return DataLoader(dataset, batch_size=int(runtime["batch_size"]), shuffle=shuffle, num_workers=int(runtime["num_workers"]), pin_memory=True, generator=generator)


def _run_loader(model: TCMaxMilk10k, loader: DataLoader[Any], device: torch.device, precision: str, max_batches: int | None, optimizer: torch.optim.Optimizer | None = None) -> tuple[dict[str, float], list[str], np.ndarray, np.ndarray]:
    model.train(optimizer is not None)
    total_loss = 0.0
    count = 0
    ids: list[str] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    context = torch.enable_grad if optimizer is not None else torch.no_grad
    with context():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            clinical = batch["clinical_image"].to(device, non_blocking=True)
            dermoscopic = batch["dermoscopic_image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits, clinical_logits, dermoscopic_logits = model(clinical, dermoscopic)
                loss = tcmax_loss(clinical_logits, dermoscopic_logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss is NaN or Inf")
            if optimizer:
                loss.backward()
                optimizer.step()
            size = target.shape[0]
            total_loss += float(loss.detach()) * size
            count += size
            ids.extend(batch["sample_id"])
            targets.append(target.detach().cpu().numpy())
            probabilities.append(logits.detach().softmax(dim=1).float().cpu().numpy())
    if not count:
        raise ValueError("no batches were evaluated")
    target_array = np.concatenate(targets)
    probability_array = np.concatenate(probabilities)
    values = {"loss": total_loss / count, **compute_metrics("multiclass", target_array, probability_array)}
    require_finite(values, "metrics")
    return values, ids, target_array, probability_array


def train(args: argparse.Namespace) -> int:
    if args.dataset != "milk10k":
        raise ValueError("TCMax is configured for MILK10k paired images")
    runtime = merge_runtime(args.runtime, {"fold": args.fold, "max_epochs": args.max_epochs, "max_train_batches": args.max_train_batches, "max_val_batches": args.max_val_batches, "seed": args.seed})
    fold = int(runtime["fold"])
    output = Path(args.output_root) / "phase2" / "milk10k" / "tcmax" / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)

    model_config = load_yaml("configs/models/phase2/tcmax_milk10k.yaml")
    dataset_config = load_yaml("configs/datasets/milk10k_phase2.yaml")
    source = load_yaml("configs/source_lock.yaml")["sources"]["tcmax"]
    epochs = runtime["max_epochs"] if runtime["max_epochs"] is not None else load_yaml("configs/training.yaml")["epochs"]
    identity_payload = {"dataset": dataset_config, "model": model_config, "source_commit": source["commit"], "fold": fold, "runtime": runtime, "manifest_sha256": sha256_file(ROOT / dataset_config["manifest"]), "benchmark_commit": _git("rev-parse", "HEAD")}
    identity = _json_hash(identity_payload)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED" and existing.get("run_id") == identity:
            print(f"SKIPPED completed run: {output}")
            return 0

    if runtime["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TCMax training requires CUDA")
    if runtime["precision"] not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    set_deterministic(int(runtime["seed"]))
    started = utc_now()
    start_clock = time.perf_counter()
    device = torch.device("cuda")
    model = TCMaxMilk10k(int(dataset_config["num_classes"])).to(device)
    loaders = {name: _loader(dataset, runtime, name == "train") for name, dataset in build_milk10k_phase2_datasets(fold, paired_transforms(int(model_config.get("input_size", 224)))).items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime["learning_rate"]), weight_decay=float(runtime["weight_decay"]))

    best_loss = math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        train_metrics, _, _, _ = _run_loader(model, loaders["train"], device, runtime["precision"], runtime["max_train_batches"], optimizer)
        validation_metrics, _, _, _ = _run_loader(model, loaders["validation"], device, runtime["precision"], runtime["max_val_batches"])
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        checkpoint = {"run_identity": identity, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_validation_loss": min(best_loss, float(validation_metrics["loss"])), "history": history}
        torch.save(checkpoint, output / "last.pt")
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            torch.save(checkpoint, output / "best.pt")

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation_metrics, _, _, _ = _run_loader(model, loaders["validation"], device, runtime["precision"], runtime["max_val_batches"])
    test_metrics, ids, targets, probabilities = _run_loader(model, loaders["test"], device, runtime["precision"], runtime["max_val_batches"])
    predictions_path = output / "predictions.csv"
    _write_predictions(predictions_path, "multiclass", ids, targets, probabilities)
    completed = utc_now()
    fold_metrics = {"schema_version": "1.0", "run_id": identity, "status": "completed", "best_epoch": int(best["epoch"]), "validation": validation_metrics, "test": {"macro_f1": test_metrics["macro_f1"], "class_f1": test_metrics["class_f1"], "state_f1": {}, "emr": None, "macro_auroc": None}, "compute": {"params_million": sum(p.numel() for p in model.parameters()) / 1_000_000, "gflops": 0.0, "inference_ms_per_sample": 0.0, "peak_gpu_memory_mb": int(torch.cuda.max_memory_allocated() / 1024**2), "training_time_sec": time.perf_counter() - start_clock}, "artifacts": {"checkpoint_best": str(output / "best.pt"), "checkpoint_last": str(output / "last.pt"), "predictions": str(predictions_path), "log": str(output / "train.log")}}
    (output / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": "1.0", "run_id": identity, "status": "COMPLETED", "phase": 2, "mode": "tcmax_paired_image_training", "model": {"name": "tcmax", "source": {"repo": source["repository"], "commit": source["commit"], "local_path": source["path"]}, "adaptation": "TCMax loss with two ResNet-18 image streams trained from scratch on paired MILK10k"}, "dataset": {"name": "milk10k", "fold": fold, "split_manifest": dataset_config["manifest"], "split_hash": identity_payload["manifest_sha256"], "modalities": ["clinical_image", "dermoscopic_image"]}, "training": {"seed": runtime["seed"], "epochs": int(epochs), "batch_size": runtime["batch_size"], "optimizer": "AdamW", "learning_rate": runtime["learning_rate"], "weight_decay": runtime["weight_decay"], "loss": "TCMax"}, "runtime": {"environment": runtime["profile"], "device": str(device), "gpu": torch.cuda.get_device_name(0), "precision": runtime["precision"]}, "provenance": {**_provenance([ROOT / "configs/models/phase2/tcmax_milk10k.yaml", ROOT / "configs/datasets/milk10k_phase2.yaml", ROOT / "configs/source_lock.yaml", ROOT / dataset_config["manifest"]]), "timestamp_start": started, "timestamp_end": completed}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "train.log").write_text(f"start={started}\nend={completed}\nstatus=completed\n", encoding="utf-8")
    print(f"COMPLETED: {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCMax on paired MILK10k")
    parser.add_argument("--dataset", default="milk10k", choices=("milk10k",))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--runtime", choices=("local", "server"), default="local")
    parser.add_argument("--output-root", default=str(ROOT / "results"))
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(train(parse_args()))


if __name__ == "__main__":
    main()
