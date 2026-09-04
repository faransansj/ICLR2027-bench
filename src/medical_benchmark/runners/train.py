from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
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
from medical_benchmark.models.registry import model_config_path, sha256_file

COMPATIBLE_MODELS = {
    "milk10k": ("mambavision", "transnext"),
    "chexchonet": ("mambavision", "transnext", "chexworld", "lrfl", "xwin", "carzero"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def model_transforms(model: str, input_size: int) -> dict[str, Any]:
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if model == "chexworld":
        post = [transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), normalize]
        evaluation = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((input_size, input_size)),
            *post,
        ])
        return {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                *post,
            ]),
            "validation": evaluation,
            "test": evaluation,
        }
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize])
    return {"train": transform, "validation": transform, "test": transform}


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
    output = Path(args.output_root) / "phase1" / args.dataset / args.model / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)

    model_config_file = model_config_path(args.model, args.dataset)
    model_config = load_yaml(model_config_file)
    dataset_config = load_yaml(f"configs/datasets/{args.dataset}.yaml")
    training_config = load_yaml("configs/training.yaml")
    source = load_yaml("configs/source_lock.yaml")["sources"][model_config["source"]]
    checkpoint_entries = source.get("checkpoints") or ([source["checkpoint"]] if source.get("checkpoint") else [])
    checkpoint_meta = checkpoint_entries[0] if checkpoint_entries else {}
    dataset_manifest = ROOT / dataset_config["manifest"]
    dataset_manifest_hash = sha256_file(dataset_manifest) if dataset_manifest.is_file() else ""
    uv_lock_hash = sha256_file(ROOT / "uv.lock")
    benchmark_commit = _git("rev-parse", "HEAD")
    benchmark_diff = _git("diff", "--no-ext-diff", "HEAD")
    epochs = runtime["max_epochs"] if runtime["max_epochs"] is not None else training_config["epochs"]
    identity_payload = {
        "dataset": args.dataset,
        "model": args.model,
        "fold": fold,
        "runtime": runtime,
        "model_config": model_config,
        "dataset_config": dataset_config,
        "dataset_manifest_hash": dataset_manifest_hash,
        "training_config": training_config,
        "source_commit": source["commit"],
        "checkpoint_hashes": [entry.get("sha256") for entry in checkpoint_entries],
        "uv_lock_hash": uv_lock_hash,
        "benchmark_commit": benchmark_commit,
        "benchmark_diff_hash": hashlib.sha256(benchmark_diff.encode()).hexdigest(),
    }
    identity = _json_hash(identity_payload)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            if existing.get("run_id") != identity:
                raise RuntimeError("completed output belongs to a different run identity")
            print(f"SKIPPED completed run: {output}")
            return 0
        manifest_path.unlink()

    if int(epochs) < 1:
        raise ValueError("effective epoch count must be at least 1")
    for key in ("max_train_batches", "max_val_batches"):
        if runtime[key] is not None and int(runtime[key]) < 1:
            raise ValueError(f"{key} must be null or at least 1")
    set_deterministic(int(runtime["seed"]))
    started = utc_now()
    start_clock = time.perf_counter()
    try:
        model = build_model(args.model, args.dataset)
    except BlockedModelError as exc:
        blocked = {"status": "BLOCKED", "run_identity": identity, "dataset": args.dataset, "model": args.model, "fold": fold,
                   "reason": exc.reason, "detail": exc.detail, "timestamp": utc_now()}
        (output / "blocked.json").write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps({
            "schema_version": "1.0", "run_id": identity, "status": "BLOCKED", "phase": 1,
            "model": {
                "name": args.model,
                "source": {"repo": source["repository"], "commit": source["commit"], "local_path": source["path"]},
                "pretrained": {"enabled": True, "checkpoint": checkpoint_meta.get("path", ""),
                               "checkpoint_url": checkpoint_meta.get("url") or checkpoint_meta.get("drive_id", ""),
                               "checkpoint_sha256": checkpoint_meta.get("sha256") or ""},
                "adaptation": model_config.get("adaptation", f"official encoder with Linear({model_config['num_classes']}) head"),
            },
            "dataset": {"name": args.dataset, "version": dataset_config.get("version", "unspecified"), "fold": fold,
                        "split_manifest": dataset_config["manifest"], "split_hash": dataset_manifest_hash,
                        "modalities": ["image"]},
            "training": {"seed": runtime["seed"], "epochs": int(epochs), "batch_size": runtime["batch_size"],
                         "optimizer": "AdamW", "learning_rate": runtime["learning_rate"],
                         "weight_decay": runtime["weight_decay"],
                         "loss": "CrossEntropyLoss" if dataset_config["task"] == "multiclass" else "BCEWithLogitsLoss"},
            "runtime": {"environment": runtime["profile"], "device": runtime["device"],
                        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "precision": runtime["precision"]},
            "blocked": {"reason": exc.reason, "detail": exc.detail},
            "provenance": {"benchmark_commit": benchmark_commit, "timestamp_start": started,
                           "timestamp_end": utc_now(), "uv_lock_hash": uv_lock_hash,
                           "dataset_manifest_hash": dataset_manifest_hash,
                           "split_manifest_hash": dataset_manifest_hash,
                           "pretrained_checkpoint_sha256": checkpoint_meta.get("sha256") or "",
                           "cuda_version": torch.version.cuda, "pytorch_version": torch.__version__,
                           "git_diff": benchmark_diff, "git_status": _git("status", "--short")},

        }, indent=2) + "\n", encoding="utf-8")
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
    datasets = build_fold_datasets(args.dataset, fold, model_transforms(args.model, input_size))
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
    require_finite([value for split in (validation_metrics, test_metrics) for value in split.values() if value is not None], "final metrics")
    predictions_path = output / "predictions.csv"
    _write_predictions(predictions_path, task, ids, targets, probabilities)

    dataset_manifest = ROOT / dataset_config["manifest"]
    config_paths = [
        ROOT / model_config_file,
        ROOT / f"configs/datasets/{args.dataset}.yaml",
        ROOT / f"configs/runtime/{args.runtime}.yaml",
        ROOT / "configs/training.yaml",
        ROOT / "configs/source_lock.yaml",
        dataset_manifest,
        *[ROOT / checkpoint["path"] for checkpoint in checkpoint_entries],
    ]
    completed = utc_now()
    elapsed = time.perf_counter() - start_clock
    params_million = sum(parameter.numel() for parameter in model.parameters()) / 1_000_000
    test_payload = {
        "macro_f1": test_metrics["macro_f1"],
        "class_f1": test_metrics.get("class_f1", {}),
        "label_f1": test_metrics.get("label_f1", {}),
        "state_f1": {},
        "emr": test_metrics.get("emr"),
        "macro_auroc": test_metrics.get("macro_auroc"),
    }
    fold_metrics = {
        "schema_version": "1.0", "run_id": identity, "status": "completed",
        "best_epoch": int(best["epoch"]), "validation": validation_metrics, "test": test_payload,
        "compute": {
            "params_million": params_million, "gflops": 0.0, "inference_ms_per_sample": 0.0,
            "peak_gpu_memory_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "training_time_sec": elapsed,
        },
        "artifacts": {
            "checkpoint_best": str(output / "best.pt"), "checkpoint_last": str(last_path),
            "predictions": str(predictions_path), "log": str(output / "train.log"),
        },
    }
    (output / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2) + "\n", encoding="utf-8")
    split_hash = sha256_file(dataset_manifest)
    provenance = _provenance(config_paths)
    manifest = {
        "schema_version": "1.0", "run_id": identity, "status": "COMPLETED", "phase": 1,
        "model": {
            "name": args.model,
            "source": {"repo": source["repository"], "commit": source["commit"], "local_path": source["path"]},
            "pretrained": {"enabled": True, "checkpoint": checkpoint_meta.get("path", ""),
                           "checkpoint_url": checkpoint_meta.get("url") or checkpoint_meta.get("drive_id", ""),
                           "checkpoint_sha256": checkpoint_meta.get("sha256", "")},
            "adaptation": model_config.get("adaptation", f"official encoder with Linear({model_config['num_classes']}) head"),
        },
        "dataset": {"name": args.dataset, "version": dataset_config.get("version", "unspecified"), "fold": fold,
                    "split_manifest": dataset_config["manifest"], "split_hash": split_hash, "modalities": ["image"]},
        "training": {"seed": runtime["seed"], "epochs": int(epochs), "batch_size": runtime["batch_size"],
                     "optimizer": "AdamW", "learning_rate": runtime["learning_rate"],
                     "weight_decay": runtime["weight_decay"], "loss": type(criterion).__name__},
        "runtime": {"environment": runtime["profile"], "device": str(device),
                    "gpu": torch.cuda.get_device_name(0), "precision": runtime["precision"]},
        "provenance": {"benchmark_commit": provenance["git_commit"], "uv_lock_hash": sha256_file(ROOT / "uv.lock"),
                       "dataset_manifest_hash": split_hash, "split_manifest_hash": split_hash,
                       "pretrained_checkpoint_sha256": checkpoint_meta.get("sha256", ""),
                       "cuda_version": torch.version.cuda, "pytorch_version": torch.__version__,
                       "git_diff": provenance["git_diff"], "git_status": provenance["git_status"],
                       "timestamp_start": started, "timestamp_end": completed},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "train.log").write_text(f"start={started}\nend={completed}\nstatus=completed\n", encoding="utf-8")
    print(f"COMPLETED: {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one independent dataset/model/fold training job")
    parser.add_argument("--dataset", required=True, choices=tuple(COMPATIBLE_MODELS))
    parser.add_argument("--model", required=True, choices=tuple(model for values in COMPATIBLE_MODELS.values() for model in values))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--runtime", choices=("local", "local-full", "server"), default="local")
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
