# Medical Benchmark

## Current benchmark tracking

- [Phase별 구현·학습·실행 위치·데이터 현황](reports/model_status.md) ([CSV](reports/model_status.csv))
- [논문 형식 결과표와 출처](reports/benchmark_tables.md): [MILK10k CSV](reports/milk10k_benchmark.csv), [CheXchoNET CSV](reports/chexchonet_benchmark.csv)
- Management **Phase 1:** MambaVision, TransNeXt, TCMax — both datasets.
- Management **Phase 2:** IVON, ABNN — both datasets; NCG, HENN — MILK10k; LATA — CheXchoNET.
- Management **Phase 3:** MedRegA, DyMo — MILK10k; DARC, RadZero — CheXchoNET.
- Management **Phase-ex:** NCG, HENN — CheXchoNET; LATA — MILK10k (additional adaptations under investigation; metrics/protocols not yet available).

Existing `configs/phases.yaml`, scripts and `results/phase*` retain **historical execution phases** for provenance/resume compatibility; they are not the new management grouping. Refresh aggregate-only reports with `python reports/update_tables.py`. Missing final metrics remain blank; paper-transcribed, legacy and incomplete runs are distinguished.

## Historical pipeline documentation

Reproducible, independent `model × dataset × fold` jobs for the original baselines. The pipeline never shares checkpoints across runs, substitutes architectures, or reports blocked work as successful.

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

Images live below the corresponding `data/<dataset>/images/` root, which may be a symlink to external storage (for example, `ln -s /path/to/MILK10k_Training_Input data/milk10k/images`). Test fold is `f`, validation fold is `(f + 1) % 5`, and all remaining folds train. Dataset items have exactly `sample_id`, `image`, `target`, and `fold` fields.

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

Public aggregate metrics are tracked in [`reports/benchmark_results.json`](reports/benchmark_results.json). Restricted datasets and sample-level predictions are never published.

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
TMPDIR=~/tmp ./scripts/run_phase2.sh --mode full --gpus 0 --folds 0,1,2,3,4 --models transnext,tcmax --datasets milk10k
```

## TCMax on both datasets (GPU 5 and 6)

TCMax reuses the pinned official additive-logit objective (including its `-log(batch_size)` constant), evaluated in stable fp32 log-space. MILK10k uses the existing two scratch ResNet-18 streams. CheXchoNET is an **explicit adaptation, not an exact paper architecture reproduction**: scratch ResNet-18 plus a `2 → 64 → 4` ReLU MLP for age/sex, with additive class logits. Age uses the training folds' population mean/std (constant std becomes 1); sex is strictly `F=0, M=1`. Missing/nonfinite age, unknown sex, and patients spanning folds fail closed. Echo measurements are never model inputs. The four-class target is `SLVH + 2*DLV`; softmax marginals `[1,3]` and `[2,3]` feed the existing multilabel metrics at threshold 0.5. Compare with the research model only after matching input modalities and evaluation protocol.

Prepare the existing image roots and separate manifests; these commands do not overwrite the Phase 1 manifests:

```bash
uv run python scripts/build_milk10k_phase2_manifest.py /path/to/MILK_ISIC_SKIN/TrainingData
uv run python scripts/build_chexchonet_phase2_manifest.py /path/to/CheXchoNET/metadata.csv
```

The Chest builder joins official `cxr_filename` metadata onto the existing benchmark manifest, checks labels and patient-level split isolation, and writes only age/sex plus identifiers, targets and existing folds. Neither builder invents missing modalities. Keep private images/metadata on the authorized server.

From the repository root, run **smoke first** (two fold-0 jobs, one epoch, two batches per split):

```bash
./scripts/run_phase2.sh --mode smoke --models tcmax --datasets milk10k,chexchonet --gpus 5,6 --batch-size 8 --num-workers 4
```

Check both `results/smoke/phase2/logs/*tcmax*.log` and the two completed `fold_metrics.json` files. Only after both smoke jobs pass, run the ten independent full-fold jobs (100 epochs, uncapped batches):

```bash
./scripts/run_phase2.sh --mode full --models tcmax --datasets milk10k,chexchonet --gpus 5,6 --folds 0,1,2,3,4 --batch-size 8 --num-workers 4
```

The scheduler assigns one job per listed physical GPU, not DDP; other GPUs are excluded. `--datasets milk10k` or `--datasets chexchonet` limits the queue. Smoke outputs are isolated from full results under `results/phase2/<dataset>/tcmax/fold_<n>`. Identical completed runs validate and skip; interrupted matching runs restore model, optimizer and RNG/loader states. Changed code/config/data/runtime identities refuse overwrite; retain existing outputs and use the runner's `--output-root` for a deliberately new experiment. Configs, source revision, code hashes, manifests and train-only age statistics are recorded in run provenance. The existing aggregate CLI still targets Phase 1; TCMax metrics are available per fold.

Initial implementation checks included CPU forward/backward for both real architectures, official-loss/gradient parity, metadata validation, mocked interruption/resume and scheduler routing. Subsequent user-provided Yonsei logs confirm legacy TCMax GPU smoke on both datasets; local Chest legacy TCMax training is also now observable. These do not validate the separate paper-budget deployment package. `READY` in a model config denotes implemented support, not provisioned data or GPU validation.

## Current local status

See the timestamped [status snapshot](reports/model_status.md) and [benchmark tables](reports/benchmark_tables.md). Legacy TransNeXt has five-fold results on both datasets; legacy MambaVision and RadZero have five-fold Chest results. Local MILK image storage remains unpopulated at the latest check, despite its manifest being available. Current server execution is only known through the dated user-provided logs, not a live connection. No raw results, dataset files or running-job configurations are changed by report updates.
