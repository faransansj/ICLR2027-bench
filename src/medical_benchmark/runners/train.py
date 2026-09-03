from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from medical_benchmark.config import ROOT, load_yaml, merge_runtime
from medical_benchmark.datasets import build_fold_datasets
from medical_benchmark.metrics import compute_metrics, require_finite
from medical_benchmark.models import BlockedModelError, build_model
from medical_benchmark.models.registry import sha256_file

COMPATIBLE_MODELS = {
    "milk10k": ("mambavision", "transnext"),
    "chexchonet": ("chexworld", "lrfl", "x_win", "carzero"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _provenance(paths: list[Path]) -> dict[str, Any]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_diff": _git("diff", "--no-ext-diff", "HEAD"),
        "git_status": _git("status", "--short"),
        "files_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _loader(dataset: Any, runtime: dict[str, Any], shuffle: bool) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(int(runtime["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(runtime["batch_size"]),
        shuffle=shuffle,
        num_workers=int(runtime["num_workers"]),
        pin_memory=True,
        generator=generator,
    )


def _probabilities(task: str, logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=1) if task == "multiclass" else logits.sigmoid()


def _run_loader(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    task: str,
    device: torch.device,
    precision: str,
    max_batches: int | None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float | None], list[str], np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(f"model output must be a Tensor, found {type(logits).__name__}")
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("logits contain NaN or Inf")
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss is NaN or Inf")
            if optimizer:
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("gradients contain NaN or Inf")
                optimizer.step()
                for parameter in model.parameters():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError("model parameters contain NaN or Inf")
            batch_size = targets.shape[0]
            total_loss += float(loss.detach()) * batch_size
            count += batch_size
            all_targets.append(targets.detach().cpu().numpy())
            all_probabilities.append(_probabilities(task, logits.detach()).float().cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    if count == 0:
        raise ValueError("no batches were evaluated")
    target_array = np.concatenate(all_targets)
    probability_array = np.concatenate(all_probabilities)
    values = {"loss": total_loss / count, **compute_metrics(task, target_array, probability_array)}
    require_finite([value for value in values.values() if value is not None], "metrics")
    return values, sample_ids, target_array, probability_array


def _write_predictions(path: Path, task: str, ids: list[str], targets: np.ndarray, probabilities: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if task == "multiclass":
            writer.writerow(["sample_id", "target", "prediction", *[f"probability_{i}" for i in range(probabilities.shape[1])]])
            for sample_id, target, probability in zip(ids, targets, probabilities, strict=True):
                writer.writerow([sample_id, int(target), int(probability.argmax()), *map(float, probability)])
        else:
            header = ["sample_id"]
            for index in range(probabilities.shape[1]):
                header.extend([f"target_{index}", f"prediction_{index}", f"probability_{index}"])
            writer.writerow(header)
            for sample_id, target, probability in zip(ids, targets, probabilities, strict=True):
                row: list[Any] = [sample_id]
                for truth, score in zip(target, probability, strict=True):
                    row.extend([int(truth), int(score >= 0.5), float(score)])
                writer.writerow(row)


def train(args: argparse.Namespace) -> int:
    if args.model not in COMPATIBLE_MODELS.get(args.dataset, ()):
        raise ValueError(f"{args.model} is not configured for {args.dataset}")
    overrides = {
        "max_epochs": args.max_epochs,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "fold": args.fold,
        "seed": args.seed,
    }
    runtime = merge_runtime(args.runtime, overrides)
    fold = int(runtime["fold"])
    output = Path(args.output_root) / runtime["profile"] / args.dataset / args.model / f"fold-{fold}"
    output.mkdir(parents=True, exist_ok=True)

    model_config = load_yaml(f"configs/models/{args.model}.yaml")
    dataset_config = load_yaml(f"configs/datasets/{args.dataset}.yaml")
    training_config = load_yaml("configs/training.yaml")
    epochs = runtime["max_epochs"] if runtime["max_epochs"] is not None else training_config["epochs"]
    identity_payload = {
        "dataset": args.dataset,
        "model": args.model,
        "fold": fold,
        "runtime": runtime,
        "model_config": model_config,
        "dataset_config": dataset_config,
        "training_config": training_config,
    }
    identity = _json_hash(identity_payload)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            if existing.get("run_identity") != identity:
                raise RuntimeError("completed output belongs to a different run identity")
            print(f"SKIPPED completed run: {output}")
            return 0

    if int(epochs) < 1:
        raise ValueError("effective epoch count must be at least 1")
    for key in ("max_train_batches", "max_val_batches"):
        if runtime[key] is not None and int(runtime[key]) < 1:
            raise ValueError(f"{key} must be null or at least 1")
    set_deterministic(int(runtime["seed"]))
    started = utc_now()
    try:
        model = build_model(args.model)
    except BlockedModelError as exc:
        blocked = {"status": "BLOCKED", "run_identity": identity, "dataset": args.dataset, "model": args.model, "fold": fold,
                   "reason": exc.reason, "detail": exc.detail, "timestamp": utc_now()}
        (output / "blocked.json").write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        print(str(exc), file=os.sys.stderr)
        return 2

    (output / "blocked.json").unlink(missing_ok=True)
    if runtime["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark configuration requires CUDA; no training was run")
    if runtime["precision"] not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    device = torch.device("cuda")
    model.to(device)
    input_size = int(model_config["input_size"])
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    datasets = build_fold_datasets(args.dataset, fold, transform)
    loaders = {name: _loader(dataset, runtime, name == "train") for name, dataset in datasets.items()}
    task = dataset_config["task"]
    criterion: nn.Module = nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime["learning_rate"]), weight_decay=float(runtime["weight_decay"]))

    start_epoch = 0
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    last_path = output / "last.pt"
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("run_identity") != identity:
            raise RuntimeError("last.pt belongs to a different run identity; refusing resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best_loss = float(saved["best_validation_loss"])
        history = list(saved["history"])
        for name, state in saved.get("loader_generator_states", {}).items():
            loaders[name].generator.set_state(state)

    max_train = runtime["max_train_batches"]
    max_val = runtime["max_val_batches"]
    for epoch in range(start_epoch, int(epochs)):
        train_metrics, _, _, _ = _run_loader(model, loaders["train"], criterion, task, device, runtime["precision"], max_train, optimizer)
        validation_metrics, _, _, _ = _run_loader(model, loaders["validation"], criterion, task, device, runtime["precision"], max_val)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        checkpoint = {
            "run_identity": identity,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": min(best_loss, float(validation_metrics["loss"])),
            "history": history,
            "loader_generator_states": {name: loader.generator.get_state() for name, loader in loaders.items()},
        }
        torch.save(checkpoint, last_path)
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            checkpoint["best_validation_loss"] = best_loss
            torch.save(checkpoint, output / "best.pt")

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    if best.get("run_identity") != identity:
        raise RuntimeError("best.pt belongs to a different run identity")
    model.load_state_dict(best["model"])
    validation_metrics, _, _, _ = _run_loader(model, loaders["validation"], criterion, task, device, runtime["precision"], max_val)
    test_metrics, ids, targets, probabilities = _run_loader(model, loaders["test"], criterion, task, device, runtime["precision"], max_val)
    metrics = {"train": history[-1]["train"], "validation": validation_metrics, "test": test_metrics,
               "history": history, "best_epoch": int(best["epoch"])}
    require_finite([value for split in (validation_metrics, test_metrics) for value in split.values() if value is not None], "final metrics")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_predictions(output / "predictions.csv", task, ids, targets, probabilities)

    source = load_yaml("configs/source_lock.yaml")["sources"][model_config["source"]]
    checkpoint_entries = source.get("checkpoints") or ([source["checkpoint"]] if source.get("checkpoint") else [])
    config_paths = [
        ROOT / f"configs/models/{args.model}.yaml",
        ROOT / f"configs/datasets/{args.dataset}.yaml",
        ROOT / f"configs/runtime/{args.runtime}.yaml",
        ROOT / "configs/training.yaml",
        ROOT / "configs/source_lock.yaml",
        ROOT / dataset_config["manifest"],
        *[ROOT / checkpoint["path"] for checkpoint in checkpoint_entries],
    ]
    manifest = {
        "status": "COMPLETED",
        "run_identity": identity,
        "dataset": args.dataset,
        "model": args.model,
        "fold": fold,
        "started_at": started,
        "completed_at": utc_now(),
        "resolved_runtime": runtime,
        "source_commit": source["commit"],
        "provenance": _provenance(config_paths),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETED: {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one independent dataset/model/fold training job")
    parser.add_argument("--dataset", required=True, choices=tuple(COMPATIBLE_MODELS))
    parser.add_argument("--model", required=True, choices=tuple(model for values in COMPATIBLE_MODELS.values() for model in values))
    parser.add_argument("--fold", type=int)
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
