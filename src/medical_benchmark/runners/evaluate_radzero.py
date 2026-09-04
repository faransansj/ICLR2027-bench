from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from medical_benchmark.config import ROOT, load_yaml, merge_runtime
from medical_benchmark.datasets import ManifestDataset
from medical_benchmark.metrics import compute_metrics, require_finite
from medical_benchmark.models.radzero import load_radzero_chexchonet
from medical_benchmark.models.registry import sha256_file
from medical_benchmark.runners.train import (
    _git,
    _json_hash,
    _provenance,
    _write_predictions,
    set_deterministic,
    utc_now,
)


def evaluate(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("RadZero evaluation requires CUDA")

    model_config = load_yaml("configs/models/phase2/radzero.yaml")
    dataset_config = load_yaml("configs/datasets/chexchonet.yaml")
    source = load_yaml("configs/source_lock.yaml")["sources"]["radzero"]
    checkpoint = ROOT / source["checkpoint"]["path"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != source["checkpoint"]["sha256"]:
        raise FileNotFoundError(f"missing or invalid RadZero checkpoint: {checkpoint}")

    prompts = [model_config["prompts"][name] for name in dataset_config["columns"]["targets"]]
    runtime = merge_runtime(args.runtime, {
        "fold": args.fold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_val_batches": args.max_batches,
    })
    fold = int(runtime["fold"])
    if fold not in range(int(dataset_config["num_folds"])):
        raise ValueError("fold must be in [0, 4]")

    identity_payload = {
        "dataset": dataset_config,
        "model": model_config,
        "source_revision": source["model_revision"],
        "checkpoint_sha256": source["checkpoint"]["sha256"],
        "fold": fold,
        "batch_size": runtime["batch_size"],
        "max_batches": runtime["max_val_batches"],
        "manifest_sha256": sha256_file(ROOT / dataset_config["manifest"]),
    }
    identity = _json_hash(identity_payload)
    output = Path(args.output_root) / "phase2" / "chexchonet" / "radzero" / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            if existing.get("run_id") != identity:
                raise RuntimeError("completed output belongs to a different run identity")
            print(f"SKIPPED completed run: {output}")
            return 0

    set_deterministic(int(runtime["seed"]))
    started = utc_now()
    start_clock = time.perf_counter()
    snapshot = ROOT / "checkpoints/radzero/hf"
    model, transform = load_radzero_chexchonet(snapshot, prompts)
    device = torch.device("cuda")
    model.to(device).eval()
    dataset = ManifestDataset(dataset_config, {fold}, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(runtime["batch_size"]),
        shuffle=False,
        num_workers=int(runtime["num_workers"]),
        pin_memory=True,
    )

    torch.cuda.reset_peak_memory_stats()
    ids: list[str] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    inference_seconds = 0.0
    samples = 0
    max_batches = runtime["max_val_batches"]
    warmed_up = False
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images = batch["image"].to(device, non_blocking=True)
            if not warmed_up:
                model(images)
                torch.cuda.synchronize()
                warmed_up = True
            start = time.perf_counter()
            logits = model(images)
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            probability = logits.sigmoid()
            require_finite(probability.detach().cpu().numpy(), "probabilities")
            ids.extend(batch["sample_id"])
            targets.append(batch["target"].numpy())
            probabilities.append(probability.cpu().numpy())
            samples += images.shape[0]

    if not samples:
        raise ValueError("no samples were evaluated")
    target_array = np.concatenate(targets)
    probability_array = np.concatenate(probabilities)
    metrics = compute_metrics("multilabel", target_array, probability_array)
    predictions_path = output / "predictions.csv"
    _write_predictions(predictions_path, "multilabel", ids, target_array, probability_array)

    completed = utc_now()
    compute = {
        "params_million": sum(parameter.numel() for parameter in model.parameters()) / 1_000_000,
        "gflops": None,
        "inference_ms_per_sample": inference_seconds * 1000 / samples,
        "peak_gpu_memory_mb": int(torch.cuda.max_memory_allocated() / 1024**2),
        "evaluation_time_sec": time.perf_counter() - start_clock,
    }
    test = {
        "macro_f1": metrics["macro_f1"],
        "macro_auroc": metrics["macro_auroc"],
        "label_f1": metrics["label_f1"],
        "class_f1": {},
        "state_f1": {},
        "emr": metrics["emr"],
    }
    fold_metrics = {
        "schema_version": "1.0",
        "run_id": identity,
        "status": "completed",
        "best_epoch": None,
        "validation": {},
        "test": test,
        "compute": compute,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "predictions": str(predictions_path),
            "log": str(output / "eval.log"),
        },
    }
    (output / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2) + "\n", encoding="utf-8")
    (output / "eval.log").write_text(f"start={started}\nend={completed}\nstatus=completed\n", encoding="utf-8")

    config_paths = [
        ROOT / "configs/models/phase2/radzero.yaml",
        ROOT / "configs/datasets/chexchonet.yaml",
        ROOT / "configs/source_lock.yaml",
        ROOT / dataset_config["manifest"],
        checkpoint,
        *sorted(path for path in snapshot.iterdir() if path.is_file() and path.name != checkpoint.name),
    ]
    manifest = {
        "schema_version": "1.0",
        "run_id": identity,
        "status": "COMPLETED",
        "phase": 2,
        "mode": "zero_shot_evaluation",
        "model": {
            "name": "radzero",
            "source": {
                "repo": source["model_repository"],
                "revision": source["model_revision"],
                "local_path": str(snapshot.relative_to(ROOT)),
            },
            "checkpoint_sha256": source["checkpoint"]["sha256"],
            "prompts": dict(zip(dataset_config["columns"]["targets"], prompts, strict=True)),
        },
        "dataset": {"name": "chexchonet", "fold": fold, "manifest": dataset_config["manifest"]},
        "runtime": {
            "batch_size": runtime["batch_size"],
            "num_workers": runtime["num_workers"],
            "max_batches": max_batches,
            "device": torch.cuda.get_device_name(0),
        },
        "provenance": {
            **_provenance(config_paths),
            "benchmark_commit": _git("rev-parse", "HEAD"),
            "uv_lock_hash": sha256_file(ROOT / "uv.lock"),
            "dataset_manifest_hash": identity_payload["manifest_sha256"],
            "split_manifest_hash": identity_payload["manifest_sha256"],
            "timestamp_start": started,
            "timestamp_end": completed,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "fold": fold, "test": test, "compute": compute}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pinned RadZero zero-shot on one CheXchoNet fold")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--runtime", choices=("local", "server"), default="local")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-root", default=str(ROOT / "results"))
    return parser.parse_args()


def main() -> None:
    raise SystemExit(evaluate(parse_args()))


if __name__ == "__main__":
    main()
