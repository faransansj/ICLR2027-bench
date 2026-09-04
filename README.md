# Medical Benchmark — Phase 1

Reproducible, independent `model × dataset × fold` jobs for the six Phase 1 baselines. The pipeline never shares checkpoints across runs, substitutes architectures, or reports blocked work as successful.

## Environment

The parent `flake.nix` provides Python 3.12, uv, build tools, and NixOS runtime libraries for PyPI wheels.

```bash
nix develop
cd medical-benchmark
uv sync --frozen
```

On a CUDA server the repository is self-contained through `pyproject.toml` and `uv.lock`:

```bash
git clone --recurse-submodules <repo>
cd medical-benchmark
uv sync --frozen
```

## Source and checkpoint pins

Official repositories are exact-commit Git submodules under `third_party/phase1/`. `configs/source_lock.yaml` records repository URLs, commits, local paths, checkpoint names/URLs, hashes, and blockers. No upstream file is modified; `patches/phase1/` is currently empty.

```bash
./scripts/fetch_sources.sh
uv run python scripts/download_checkpoint.py mambavision
uv run python scripts/download_checkpoint.py transnext
```

Only immutable, published-hash downloads are automated. CheXWorld and CARZero use mutable Google Drive artifacts; their downloaded bytes, sizes, and locally verified SHA256 hashes are recorded in `configs/source_lock.yaml`. CARZero remains blocked on a verified downstream adapter, LRFL has no released LRFL checkpoint, and X-WIN has neither a released checkpoint nor official downstream extraction code. MambaVision requires the official CUDA `mamba-ssm==2.2.4` dependency; the Nix shell exposes the NVIDIA driver paths required by PyTorch and Triton.

## Dataset manifests

- `data/milk10k/manifest.csv`: `sample_id,image_path,label,fold`, where label is 0–10.
- `data/chexchonet/manifest.csv`: `sample_id,image_path,SLVH,DLV,fold`, where both targets are binary.

Images live below the corresponding `data/<dataset>/images/` root. Test fold is `f`, validation fold is `(f + 1) % 5`, and all remaining folds train. Dataset items have exactly `sample_id`, `image`, `target`, and `fold` fields.

## Run commands

One independent job:

```bash
uv run python runners/train.py --dataset milk10k --model transnext --fold 0 --runtime local
```

All fold-0 smoke jobs:

```bash
./scripts/run_phase1.sh --mode smoke --gpus 0
# equivalent
./scripts/test_local.sh
```

MambaVision and TransNeXt on CheXchoNet:

```bash
./scripts/run_phase1.sh --mode smoke --gpus 0 --dataset chexchonet --models mambavision,transnext
./scripts/run_phase1.sh --mode full --gpus 0 --dataset chexchonet --models mambavision,transnext --folds 0,1,2,3,4
```

Server full scheduling (30 independent jobs dynamically assigned at job level):

```bash
./scripts/run_phase1.sh --mode full --gpus 0,1,2,3 --folds 0,1,2,3,4
# or: GPUS=0,1,2,3 ./scripts/run_server.sh
```

`local.yaml` is CUDA/fp32, one epoch, 20 train and 10 validation batches, fold 0. `server.yaml` is CUDA/bf16 with uncapped batches and epochs from `configs/training.yaml`. Completed identical runs skip; matching `last.pt` runs resume; identity mismatches fail closed.

## Results

Each run writes:

```text
results/phase1/<dataset>/<model>/fold_<n>/
├── run_manifest.json
├── fold_metrics.json
├── predictions.csv
├── best.pt
├── last.pt
└── train.log
```

Validation and five-fold aggregation:

```bash
uv run python runners/validate_run.py results/phase1/milk10k/transnext/fold_0
uv run python runners/aggregate.py --results results
```

Aggregation writes `summary.json` under each dataset/model directory. Blocked runs write `blocked.json` plus `run_manifest.json` and never fabricate metrics or checkpoints. When a dataset manifest exists, blocked manifests record its SHA256 and include it in the run identity.

A lightweight Phase 1 archive keeps the code/config snapshot, non-checkpoint results, metrics recomputed from predictions, and SHA256/size records for excluded checkpoints:

```bash
uv run python scripts/archive_phase1.py
```

The command writes `archives/phase1-<UTC timestamp>.tar.gz` and a matching `.sha256` file. It does not copy datasets or checkpoint tensors; retain the original checkpoint paths listed in `checkpoint_inventory.json` if later inference is required.

Verify every official CheXchoNET image and persist the verification evidence on the server:

```bash
./scripts/verify_chexchonet.sh /path/to/chexchonet-a-chest-radiograph-dataset-with-gold-standard-echocardiography-labels-1.0.0
```

## Later-phase preparation

`configs/phases.yaml` records the approved experiment waves. Phase 2 contains candidate methods with author-published code: MM-Skin-FS, TCMax, DyMo, MedRegA, and RadZero. Their exact source revisions and available model revisions/hashes are recorded in `configs/source_lock.yaml`. RadZero has a local-snapshot, zero-shot CheXchoNet evaluator. TCMax has a MILK10k adapter using paired clinical/dermoscopic ResNet-18 streams and the official TCMax objective; its source is pinned at `third_party/phase2/tcmax`. The remaining target/model combinations stay blocked until their adapters, modalities, runtimes, and required artifacts are reproducible.

Evaluate the pinned RadZero snapshot with fixed SLVH/DLV prompts (batch 8 peaks at about 2.4 GiB on an RTX 2080):

```bash
uv run python scripts/evaluate_radzero.py --fold 0 --runtime local --batch-size 8 --max-batches 2 --output-root results/smoke
uv run python scripts/evaluate_radzero.py --fold 0 --runtime server --batch-size 8 --num-workers 4
```

Phase 3 contains SkinM2Former and DARC. Both remain blocked because the official publication surfaces currently provide no author-published implementation or checkpoint. These preparation records do not make the Phase 1 image-only trainer compatible with multimodal, few-shot, zero-shot, or causal methods, and no later-phase run is reported as successful.

The immutable, hash-pinned RadZero checkpoint can be prepared without running a GPU job:

```bash
uv run python scripts/download_checkpoint.py radzero
uv run python scripts/smoke_radzero.py  # synthetic-image CUDA smoke
```

Build the Phase 2 paired-image MILK10k manifest from the official training directory:

```bash
uv run python scripts/build_milk10k_phase2_manifest.py /path/to/MILK_ISIC_SKIN/TrainingData
```

The builder requires one clinical and one dermoscopic image per lesion, matching five-fold and one-hot labels, and consistent core metadata. It writes `data/milk10k/manifest_phase2.csv`; images remain under the existing `data/milk10k/images` root.

Run TransNeXt and TCMax together after preparing that manifest:

```bash
TMPDIR=~/tmp ./scripts/run_phase2.sh --mode full --gpus 0 --folds 0,1,2,3,4 --models transnext,tcmax
```

## Current local status

The source pins, UV lock, dataset contracts, metric/result schemas, checkpoint verification, resume policy, and scheduler are unit-tested. Real smoke completion still requires the private dataset manifests/images, CUDA, and the official model artifacts/dependencies listed above. Full five-fold training has not been run.
