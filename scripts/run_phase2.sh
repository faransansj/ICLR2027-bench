#!/usr/bin/env bash
set -uo pipefail
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=""
GPUS="0"
FOLDS="0,1,2,3,4"
MODELS=""
DATASETS="milk10k,chexchonet"
BATCH_SIZE="8"
NUM_WORKERS="4"
MAX_BATCHES=""
while (($#)); do
  (($# >= 2)) || { echo "missing value for $1" >&2; exit 64; }
  case "$1" in
    --mode) MODE=${2:-}; shift 2 ;;
    --gpus) GPUS=${2:-}; shift 2 ;;
    --folds) FOLDS=${2:-}; shift 2 ;;
    --models) MODELS=${2:-}; shift 2 ;;
    --datasets) DATASETS=${2:-}; shift 2 ;;
    --batch-size) BATCH_SIZE=${2:-}; shift 2 ;;
    --num-workers) NUM_WORKERS=${2:-}; shift 2 ;;
    --max-batches) MAX_BATCHES=${2:-}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo "usage: $0 --mode smoke|full --gpus 0,1 [--folds 0,1,2,3,4] [--models transnext,tcmax,radzero,medrega] [--datasets milk10k,chexchonet] [--batch-size 8] [--num-workers 4] [--max-batches N]" >&2
  exit 64
fi
[[ "$BATCH_SIZE" =~ ^[0-9]+$ && "$BATCH_SIZE" =~ [1-9] ]] || { echo "--batch-size must be a positive integer" >&2; exit 64; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "--num-workers must be a nonnegative integer" >&2; exit 64; }
[[ -z "$MAX_BATCHES" || ( "$MAX_BATCHES" =~ ^[0-9]+$ && "$MAX_BATCHES" =~ [1-9] ) ]] || { echo "--max-batches must be a positive integer" >&2; exit 64; }
cd "$ROOT"
IFS=, read -r -a dataset_list <<< "$DATASETS"
((${#dataset_list[@]})) || { echo "--datasets cannot be empty" >&2; exit 64; }
for dataset in "${dataset_list[@]}"; do
  case "$dataset" in milk10k|chexchonet) ;; *) echo "unknown dataset: $dataset" >&2; exit 64 ;; esac
done
IFS=, read -r -a gpu_list <<< "$GPUS"
if [[ -n "$MODELS" ]]; then IFS=, read -r -a model_list <<< "$MODELS"; else model_list=(transnext tcmax radzero medrega); fi
for model in "${model_list[@]}"; do
  case "$model" in transnext|tcmax|radzero|medrega) ;; *) echo "unknown model: $model" >&2; exit 64 ;; esac
done
((${#gpu_list[@]})) || { echo "--gpus cannot be empty" >&2; exit 64; }
for gpu in "${gpu_list[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }; done
# Duplicate queue keys would concurrently overwrite the same independent run.
for list in "$GPUS" "$DATASETS" "$MODELS" "$FOLDS"; do
  IFS=, read -r -a entries <<< "$list"
  seen=","
  for entry in "${entries[@]}"; do
    [[ "$seen" != *",$entry,"* ]] || { echo "duplicate selection: $entry" >&2; exit 64; }
    seen+="$entry,"
  done
done

if command -v uv >/dev/null 2>&1; then runner=(uv run python); elif command -v python3 >/dev/null 2>&1; then runner=(python3); else
  echo "BLOCKED: neither uv nor python3 is available" >&2
  exit 69
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
jobs="$work/jobs"
: > "$jobs"
if [[ "$MODE" == smoke ]]; then
  fold_list=(0)
  output_root="$ROOT/results/smoke"
  runtime=local
  [[ -n "$MAX_BATCHES" ]] || MAX_BATCHES=2
else
  IFS=, read -r -a fold_list <<< "$FOLDS"
  output_root="$ROOT/results"
  runtime=server
fi
for fold in "${fold_list[@]}"; do
  [[ "$fold" =~ ^[0-4]$ ]] || { echo "invalid fold: $fold (expected 0-4)" >&2; exit 64; }
  for model in "${model_list[@]}"; do
    case "$model" in
      transnext) [[ ",$DATASETS," != *,milk10k,* ]] || printf 'milk10k\t%s\t%s\tphase1\n' "$model" "$fold" >> "$jobs" ;;
      radzero) [[ ",$DATASETS," != *,chexchonet,* ]] || printf 'chexchonet\t%s\t%s\tradzero\n' "$model" "$fold" >> "$jobs" ;;
      medrega) [[ ",$DATASETS," != *,milk10k,* ]] || printf 'milk10k\t%s\t%s\tblock\n' "$model" "$fold" >> "$jobs" ;;
      tcmax)
        for dataset in "${dataset_list[@]}"; do
          printf '%s\t%s\t%s\ttcmax\n' "$dataset" "$model" "$fold" >> "$jobs"
        done
        ;;
    esac
  done
done

write_blocked() {
  local dataset=$1 model=$2 fold=$3
  PYTHONPATH="$ROOT/src" "${runner[@]}" - "$ROOT" "$dataset" "$model" "$fold" "$output_root" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

root = Path(sys.argv[1])
dataset, model, fold, output_root = sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
config_path = root / "configs" / "models" / "phase2" / ("tcmax_chexchonet.yaml" if model == "tcmax" and dataset == "chexchonet" else f"{model}.yaml" if model != "tcmax" else "tcmax_milk10k.yaml")
source_lock = yaml.safe_load((root / "configs/source_lock.yaml").read_text())["sources"][model]
config = yaml.safe_load(config_path.read_text())
out = output_root / "phase2" / dataset / model / f"fold_{fold}"
out.mkdir(parents=True, exist_ok=True)
now = datetime.now(timezone.utc).isoformat()
blocked = {"status": "BLOCKED", "dataset": dataset, "model": model, "fold": fold, "reason": config["blocked_reason"], "detail": config["detail"], "timestamp": now}
(out / "blocked.json").write_text(json.dumps(blocked, indent=2) + "\n")
manifest = {"schema_version": "1.0", "status": "BLOCKED", "phase": 2, "model": {"name": model, "source": {"repo": source_lock["repository"], "commit": source_lock["commit"], "local_path": source_lock["path"]}}, "dataset": {"name": dataset, "fold": fold}, "blocked": {"reason": config["blocked_reason"], "detail": config["detail"]}, "provenance": {"timestamp_start": now, "timestamp_end": now}}
(out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
}

log_dir="$output_root/phase2/logs"
mkdir -p "$log_dir"
printf '0\n' > "$work/counter"
: > "$work/status"
: > "$work/lock"
worker() {
  local gpu=$1 index line dataset model fold action rc args=()
  exec 9<>"$work/lock"
  while true; do
    flock 9
    index=$(<"$work/counter")
    line=$(sed -n "$((index + 1))p" "$jobs")
    if [[ -n "$line" ]]; then printf '%s\n' "$((index + 1))" > "$work/counter"; fi
    flock -u 9
    [[ -n "$line" ]] || break
    IFS=$'\t' read -r dataset model fold action <<< "$line"
    case "$action" in
      phase1)
        args=(-m medical_benchmark.runners.train --dataset "$dataset" --model "$model" --fold "$fold" --runtime "$runtime" --output-root "$output_root")
        [[ -z "$MAX_BATCHES" ]] || args+=(--max-train-batches "$MAX_BATCHES" --max-val-batches "$MAX_BATCHES")
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src" "${runner[@]}" "${args[@]}" >"$log_dir/${dataset}-${model}-${fold}.log" 2>&1
        rc=$?
        ;;
      radzero)
        args=(scripts/evaluate_radzero.py --fold "$fold" --runtime "$runtime" --batch-size "$BATCH_SIZE" --output-root "$output_root")
        [[ -z "$NUM_WORKERS" ]] || args+=(--num-workers "$NUM_WORKERS")
        [[ -z "$MAX_BATCHES" ]] || args+=(--max-batches "$MAX_BATCHES")
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src" "${runner[@]}" "${args[@]}" >"$log_dir/${dataset}-${model}-${fold}.log" 2>&1
        rc=$?
        ;;
      tcmax)
        args=(-m medical_benchmark.runners.train_tcmax --dataset "$dataset" --fold "$fold" --runtime "$runtime" --output-root "$output_root" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
        [[ -z "$MAX_BATCHES" ]] || args+=(--max-train-batches "$MAX_BATCHES" --max-val-batches "$MAX_BATCHES")
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src" "${runner[@]}" "${args[@]}" >"$log_dir/${dataset}-${model}-${fold}.log" 2>&1
        rc=$?
        ;;
      block)
        write_blocked "$dataset" "$model" "$fold" >"$log_dir/${dataset}-${model}-${fold}.log" 2>&1
        rc=2
        ;;
    esac
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$model" "$fold" "$gpu" "$rc" >> "$work/status"
    flock -u 9
  done
}
for gpu in "${gpu_list[@]}"; do worker "$gpu" & done
wait

printf 'Model | Dataset | Source | Data Load | Pretrained | Forward | Metric | JSON | Verdict\n'
while IFS=$'\t' read -r dataset model fold gpu rc; do
  if [[ "$rc" == 0 ]]; then
    printf '%s | %s | PASS | PASS | PASS | PASS | PASS | PASS | PASS\n' "$model" "$dataset"
  elif [[ "$rc" == 2 ]]; then
    reason=$(PYTHONPATH="$ROOT/src" "${runner[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["reason"].replace("_", " "))' \
      "$output_root/phase2/$dataset/$model/fold_$fold/blocked.json" 2>/dev/null || printf 'blocked')
    printf '%s | %s | PASS | NOT RUN | %s | NOT RUN | NOT RUN | PASS | BLOCKED\n' "$model" "$dataset" "$reason"
  else
    printf '%s | %s | UNKNOWN | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL\n' "$model" "$dataset"
  fi
done < "$work/status"
awk -F '\t' '$5 != 0 && $5 != 2 {bad=1} END {exit bad}' "$work/status"
