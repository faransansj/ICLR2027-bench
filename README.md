# Medical Benchmark — Phase 1

A small, auditable scaffold for independent dataset × model × fold runs. It never substitutes an architecture or reports a blocked job as successful.

## Layout

- `configs/datasets/`: explicit manifest/image-root contracts for MILK10k and CheXchoNET
- `configs/models/`: model/task pairing and official feature/head metadata
- `configs/runtime/`: exact local smoke and server profiles
- `configs/source_lock.yaml`: authoritative repositories, commits, checkpoint locations, hashes, and blocking status
- `src/medical_benchmark/{datasets,models,metrics,runners}/`: adapters, guarded registry, metrics, training/validation/aggregation
- `scripts/`: source/checkpoint helpers and GPU job scheduler
- `data/{milk10k,chexchonet}/`: local manifests/images (ignored)
- `results/`: generated run artifacts (ignored)

## Environment

Python 3.11 is required.

```bash
uv sync
```

`uv.lock` is generated with `uv lock`; if `uv` is unavailable it must not be fabricated. Upstream code is not a package dependency and is imported only after source revision and checkpoint hashes pass validation.

## Data contracts

`data/milk10k/manifest.csv` must contain `sample_id,image_path,label,fold`. `label` is an integer in 0–10 and `fold` is 0–4.

`data/chexchonet/manifest.csv` must contain `sample_id,image_path,SLVH,DLV,fold`. Both targets are binary and `fold` is 0–4.

Paths are relative to each configured image root. IDs must be non-empty and unique, paths may not escape the root, and every referenced image must exist. Samples are returned as:

```python
{"image": RGB_tensor, "target": class_or_two_binary_targets, "sample_id": str, "fold": int}
```

For test fold `f`, validation is `(f + 1) % 5` and the remaining folds train. This is recorded in `configs/training.yaml`.

## Official sources and checkpoints

Run `scripts/fetch_sources.sh` to clone each exact locked commit into ignored `.upstream/` directories. The registry checks `HEAD` before importing. To download the only artifacts with immutable URLs and published hashes:

```bash
uv run python scripts/download_checkpoint.py mambavision
uv run python scripts/download_checkpoint.py transnext
```

Drive artifacts are never downloaded automatically. CheXWorld remains blocked until its local checkpoint hash is entered in `configs/source_lock.yaml`; CARZero remains blocked until both hashes are entered. LRFL has no released checkpoint. X-WIN has neither a released checkpoint nor a downstream extraction implementation. No substitute is implemented.

MambaVision and TransNeXt load their official 1000-class checkpoint strictly, then replace `head` with `Linear(11)`; TransNeXt's broken `reset_classifier` is not used. The locked metadata records CheXWorld's target-encoder ViT-B/16 pooled feature and CARZero's image feature as the intended inputs to `Linear(2)`.

## Runs

One job:

```bash
uv run medical-benchmark-train --dataset milk10k --model mambavision --fold 0 --runtime local
```

Smoke schedules exactly six compatible fold-0 jobs. Full schedules every compatible model for requested folds. GPU workers dynamically pull the next job, so a free GPU receives work immediately.

```bash
scripts/run_benchmark.sh --mode smoke --gpus 0,1
scripts/run_benchmark.sh --mode full --gpus 0,1,2,3 --folds 0,1,2,3,4
```

Local is CUDA, debug, one epoch, 20 train batches, 10 validation/test batches, fold 0, fp32. Server is CUDA, non-debug, bf16 with null runtime caps; the uncapped epoch count comes from `configs/training.yaml`. Missing data, CUDA, official source, or verified weights fails or emits a structured `blocked.json`; it never creates metrics or a PASS marker.

Completed runs contain `manifest.json`, `metrics.json`, `predictions.csv`, `best.pt`, and `last.pt`. Checkpoint resume requires an identical hashed run identity. A completed identical run is skipped. Manifests capture timestamps, resolved runtime, Git commit/diff/status, input/config hashes, and CUDA/PyTorch versions.

```bash
uv run python scripts/validate_run.py results/local/milk10k/mambavision/fold-0
uv run python scripts/aggregate.py --results results
```

Aggregation validates completed runs and writes `results/summary.json`, listing completed, blocked, and invalid runs separately.
