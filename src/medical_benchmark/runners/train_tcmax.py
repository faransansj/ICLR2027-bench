from __future__ import annotations

import argparse
import json
import math
import subprocess
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
from medical_benchmark.datasets.chexchonet_phase2 import build_chexchonet_phase2_datasets
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


class TCMaxChexchonet(nn.Module):
    """Explicit image/tabular adaptation with additive per-modality class logits."""

    def __init__(self, classes: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.image = TCMaxMilk10k._encoder(classes)
        self.tabular = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, classes))

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_logits, tabular_logits = self.image(image), self.tabular(tabular)
        return image_logits + tabular_logits, image_logits, tabular_logits


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
    # Same pinned official objective/constant, in fp32 log-space to avoid bf16 overflow.
    joint = clinical_logits.float()[:, None, :] + dermoscopic_logits.float()[None, :, :]
    index = torch.arange(batch, device=labels.device)
    return torch.logsumexp(joint.reshape(-1), dim=0) - joint[index, index, labels].mean() - math.log(batch)


def chexchonet_outputs(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.stack((targets % 2, targets // 2), axis=1), np.stack(
        (probabilities[:, [1, 3]].sum(axis=1), probabilities[:, [2, 3]].sum(axis=1)), axis=1
    )


def _loader(dataset: Any, runtime: dict[str, Any], shuffle: bool) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(int(runtime["seed"]))
    return DataLoader(dataset, batch_size=int(runtime["batch_size"]), shuffle=shuffle, num_workers=int(runtime["num_workers"]), pin_memory=True, generator=generator)


def _run_loader(model: nn.Module, loader: DataLoader[Any], device: torch.device, precision: str, max_batches: int | None, optimizer: torch.optim.Optimizer | None = None, task: str = "multiclass") -> tuple[dict[str, float], list[str], np.ndarray, np.ndarray]:
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
            first, second = ("image", "tabular") if task == "multilabel" else ("clinical_image", "dermoscopic_image")
            clinical = batch[first].to(device, non_blocking=True)
            dermoscopic = batch[second].to(device, non_blocking=True)
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
            probabilities.append(logits.detach().float().softmax(dim=1).cpu().numpy())
    if not count:
        raise ValueError("no batches were evaluated")
    target_array = np.concatenate(targets)
    probability_array = np.concatenate(probabilities)
    if task == "multilabel":
        target_array, probability_array = chexchonet_outputs(target_array, probability_array)
    values = {"loss": total_loss / count, **compute_metrics(task, target_array, probability_array)}
    require_finite(values, "metrics")
    return values, ids, target_array, probability_array


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint(path: Path, identity: str) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("run_identity") != identity:
        raise RuntimeError(f"{path.name} belongs to a different run identity; refusing overwrite/resume")
    return saved


def _save_checkpoint(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> int:
    runtime = merge_runtime(args.runtime, {key: getattr(args, key, None) for key in (
        "fold", "max_epochs", "max_train_batches", "max_val_batches", "seed", "batch_size", "num_workers", "precision"
    )})
    fold = int(runtime["fold"])
    model_path = f"configs/models/phase2/tcmax_{args.dataset}.yaml"
    dataset_path = f"configs/datasets/{args.dataset}_phase2.yaml"
    model_config, dataset_config = load_yaml(model_path), load_yaml(dataset_path)
    source = load_yaml("configs/source_lock.yaml")["sources"]["tcmax"]
    epochs = int(runtime["max_epochs"] if runtime["max_epochs"] is not None else load_yaml("configs/training.yaml")["epochs"])
    if fold not in range(int(dataset_config["num_folds"])):
        raise ValueError("fold must be in [0, 4]")
    if epochs < 1 or int(runtime["batch_size"]) < 1 or int(runtime["num_workers"]) < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative")
    for key in ("max_train_batches", "max_val_batches"):
        if runtime[key] is not None and int(runtime[key]) < 1:
            raise ValueError(f"{key} must be positive")
    source_root = ROOT / source["path"]
    source_commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != source["commit"] or subprocess.check_output(["git", "-C", str(source_root), "diff", "HEAD", "--"], text=True).strip():
        raise RuntimeError("TCMax source must match its clean pinned revision")
    config_paths = [ROOT / path for path in (model_path, dataset_path, "configs/source_lock.yaml", "configs/training.yaml", f"configs/runtime/{args.runtime}.yaml", "uv.lock", dataset_config["manifest"])]
    config_paths += sorted((ROOT / "src/medical_benchmark").rglob("*.py"))
    config_paths += [source_root / "src/utils/losses.py"]
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in config_paths}
    identity_payload = {"dataset": dataset_config, "model": model_config, "source_commit": source_commit, "fold": fold, "runtime": runtime, "epochs": epochs, "files_sha256": hashes, "benchmark_commit": _git("rev-parse", "HEAD")}
    identity = _json_hash(identity_payload)
    output = Path(args.output_root) / "phase2" / args.dataset / "tcmax" / f"fold_{fold}"
    manifest_path = output / "run_manifest.json"
    existing = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_id") != identity:
            raise RuntimeError("output belongs to a different run identity; refusing overwrite")
        if existing.get("status") == "COMPLETED":
            from medical_benchmark.runners.validate_run import validate_run
            validate_run(output)
            print(f"SKIPPED completed run: {output}")
            return 0
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("output has artifacts without a run manifest; refusing overwrite")
    saved = _checkpoint(output / "last.pt", identity) if (output / "last.pt").is_file() else None
    if (output / "best.pt").is_file():
        _checkpoint(output / "best.pt", identity)
    if runtime["precision"] not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    transform = paired_transforms(int(model_config.get("input_size", 224)))
    datasets = (build_milk10k_phase2_datasets(fold, transform) if args.dataset == "milk10k"
                else build_chexchonet_phase2_datasets(fold, transform["clinical"]))
    if runtime["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TCMax training requires CUDA; datasets validated but no GPU run performed")
    set_deterministic(int(runtime["seed"]))
    started = existing["provenance"]["timestamp_start"] if existing else utc_now()
    start_clock = time.perf_counter()
    previous_time = float(saved["training_time_sec"]) if saved else 0.0
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model = (TCMaxMilk10k(int(dataset_config["num_classes"])) if args.dataset == "milk10k"
             else TCMaxChexchonet(int(model_config["num_classes"]), int(model_config["tabular_hidden"]))).to(device)
    loaders = {name: _loader(dataset, runtime, name == "train") for name, dataset in datasets.items()}
    task = dataset_config["task"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime["learning_rate"]), weight_decay=float(runtime["weight_decay"]))
    provenance = _provenance(config_paths)
    provenance.update({"benchmark_commit": identity_payload["benchmark_commit"], "uv_lock_hash": hashes["uv.lock"], "dataset_manifest_hash": hashes[dataset_config["manifest"]], "split_manifest_hash": hashes[dataset_config["manifest"]], "cuda_version": torch.version.cuda, "pytorch_version": torch.__version__, "timestamp_start": started, "timestamp_end": None})
    manifest = {
        "schema_version": "1.0", "run_id": identity, "status": "RUNNING", "phase": 2,
        "mode": "tcmax_training", "identity": identity_payload,
        "model": {"name": "tcmax", "source": {"repo": source["repository"], "commit": source_commit, "local_path": source["path"]}, "config": model_config},
        "dataset": {"name": args.dataset, "fold": fold, "split_manifest": dataset_config["manifest"], "split_hash": hashes[dataset_config["manifest"]], "modalities": model_config["modalities"]},
        "training": {"seed": runtime["seed"], "epochs": epochs, "batch_size": runtime["batch_size"], "optimizer": "AdamW", "learning_rate": runtime["learning_rate"], "weight_decay": runtime["weight_decay"], "loss": "TCMax"},
        "runtime": {**runtime, "gpu": torch.cuda.get_device_name(0)}, "provenance": provenance,
    }
    if args.dataset == "chexchonet":
        manifest["dataset"]["age_statistics_train_only"] = datasets["train"].age_stats
    output.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, manifest)
    start_epoch, best_loss, history = 0, math.inf, []
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best_loss, history = int(saved["epoch"]) + 1, float(saved["best_validation_loss"]), saved["history"]
        for name, state in saved["loader_generator_states"].items():
            loaders[name].generator.set_state(state)
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        # Recover a best checkpoint if interrupted between the two atomic saves.
        if saved["is_best"]:
            _save_checkpoint(saved, output / "best.pt")
    for epoch in range(start_epoch, epochs):
        train_metrics, _, _, _ = _run_loader(model, loaders["train"], device, runtime["precision"], runtime["max_train_batches"], optimizer, task)
        validation_metrics, _, _, _ = _run_loader(model, loaders["validation"], device, runtime["precision"], runtime["max_val_batches"], task=task)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        improved = float(validation_metrics["loss"]) < best_loss
        best_loss = min(best_loss, float(validation_metrics["loss"]))
        checkpoint = {"run_identity": identity, "epoch": epoch, "is_best": improved, "training_time_sec": previous_time + time.perf_counter() - start_clock, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_validation_loss": best_loss, "history": history, "loader_generator_states": {name: loader.generator.get_state() for name, loader in loaders.items()}, "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all()}
        _save_checkpoint(checkpoint, output / "last.pt")
        if improved:
            _save_checkpoint(checkpoint, output / "best.pt")
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.6f} validation_loss={validation_metrics['loss']:.6f}", flush=True)
    best = _checkpoint(output / "best.pt", identity)
    model.load_state_dict(best["model"])
    validation_metrics, _, _, _ = _run_loader(model, loaders["validation"], device, runtime["precision"], runtime["max_val_batches"], task=task)
    test_metrics, ids, targets, probabilities = _run_loader(model, loaders["test"], device, runtime["precision"], runtime["max_val_batches"], task=task)
    predictions_path = output / "predictions.csv"
    _write_predictions(predictions_path, task, ids, targets, probabilities)
    completed = utc_now()
    fold_metrics = {"schema_version": "1.0", "run_id": identity, "status": "completed", "best_epoch": int(best["epoch"]), "validation": validation_metrics, "test": {"class_f1": {}, "label_f1": {}, "state_f1": {}, "emr": None, "macro_auroc": None, **{key: value for key, value in test_metrics.items() if key != "loss"}}, "compute": {"params_million": sum(p.numel() for p in model.parameters()) / 1_000_000, "gflops": None, "inference_ms_per_sample": None, "peak_gpu_memory_mb": int(torch.cuda.max_memory_allocated() / 1024**2), "training_time_sec": previous_time + time.perf_counter() - start_clock}, "artifacts": {"checkpoint_best": str(output / "best.pt"), "checkpoint_last": str(output / "last.pt"), "predictions": str(predictions_path), "log": str(output / "train.log")}}
    (output / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2) + "\n", encoding="utf-8")
    (output / "train.log").write_text(f"start={started}\nend={completed}\nstatus=completed\n", encoding="utf-8")
    manifest["status"] = "COMPLETED"
    manifest["provenance"]["timestamp_end"] = completed
    _write_manifest(manifest_path, manifest)
    print(f"COMPLETED: {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCMax on MILK10k paired images or CheXchoNET image/age/sex")
    parser.add_argument("--dataset", default="milk10k", choices=("milk10k", "chexchonet"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--runtime", choices=("local", "server"), default="local")
    parser.add_argument("--output-root", default=str(ROOT / "results"))
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    return parser.parse_args()


def main() -> None:
    raise SystemExit(train(parse_args()))


if __name__ == "__main__":
    main()
