#!/usr/bin/env bash
set -uo pipefail
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=""
GPUS="0"
FOLDS="0,1,2,3,4"
MODELS=""
DATASET=""
while (($#)); do
  case "$1" in
    --mode) MODE=${2:-}; shift 2 ;;
    --gpus) GPUS=${2:-}; shift 2 ;;
    --folds) FOLDS=${2:-}; shift 2 ;;
    --models) MODELS=${2:-}; shift 2 ;;
    --dataset) DATASET=${2:-}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo "usage: $0 --mode smoke|full --gpus 0,1 [--folds 0,1,2,3,4] [--models mambavision,transnext] [--dataset milk10k|chexchonet]" >&2
  exit 64
fi
IFS=, read -r -a gpu_list <<< "$GPUS"
if [[ -n "$MODELS" ]]; then IFS=, read -r -a model_list <<< "$MODELS"; else model_list=(mambavision transnext chexworld lrfl xwin carzero); fi
for model in "${model_list[@]}"; do
  case "$model" in mambavision|transnext|chexworld|lrfl|xwin|carzero) ;; *) echo "unknown model: $model" >&2; exit 64 ;; esac
  if [[ -n "$DATASET" ]]; then
    case "$DATASET:$model" in
      milk10k:mambavision|milk10k:transnext|chexchonet:mambavision|chexchonet:transnext|chexchonet:chexworld|chexchonet:lrfl|chexchonet:xwin|chexchonet:carzero) ;;
      *) echo "$model is not configured for $DATASET" >&2; exit 64 ;;
    esac
  fi
done
((${#gpu_list[@]})) || { echo "--gpus cannot be empty" >&2; exit 64; }
for gpu in "${gpu_list[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }; done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
jobs="$work/jobs"
if [[ "$MODE" == smoke ]]; then
  fold_list=(0)
  runtime=local
else
  IFS=, read -r -a fold_list <<< "$FOLDS"
  runtime=server
fi
: > "$jobs"
for fold in "${fold_list[@]}"; do
  [[ "$fold" =~ ^[0-4]$ ]] || { echo "invalid fold: $fold (expected 0-4)" >&2; exit 64; }
  for model in "${model_list[@]}"; do
    if [[ -n "$DATASET" ]]; then
      dataset=$DATASET
    else
      case "$model" in
        mambavision|transnext) dataset=milk10k ;;
        chexworld|lrfl|xwin|carzero) dataset=chexchonet ;;
      esac
    fi
    printf '%s\t%s\t%s\n' "$dataset" "$model" "$fold" >> "$jobs"
  done
done

if command -v uv >/dev/null 2>&1; then runner=(uv run python); elif command -v python3 >/dev/null 2>&1; then runner=(python3); else
  echo "BLOCKED: neither uv nor python3 is available" >&2
  exit 69
fi
printf '0\n' > "$work/counter"
: > "$work/status"
: > "$work/lock"

worker() {
  local gpu=$1 index line dataset model fold rc
  exec 9<>"$work/lock"
  while true; do
    flock 9
    index=$(<"$work/counter")
    line=$(sed -n "$((index + 1))p" "$jobs")
    if [[ -n "$line" ]]; then printf '%s\n' "$((index + 1))" > "$work/counter"; fi
    flock -u 9
    [[ -n "$line" ]] || break
    IFS=$'\t' read -r dataset model fold <<< "$line"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src" "${runner[@]}" -m medical_benchmark.runners.train \
      --dataset "$dataset" --model "$model" --fold "$fold" --runtime "$runtime" --output-root "$ROOT/results" \
      >"$work/${dataset}-${model}-${fold}.log" 2>&1
    rc=$?
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$model" "$fold" "$gpu" "$rc" >> "$work/status"
    flock -u 9
  done
}
for gpu in "${gpu_list[@]}"; do worker "$gpu" & done
wait

printf 'Model | Dataset | Source | Data Load | Pretrained | Forward | Backward | Metric | Checkpoint | JSON | Verdict\n'
while IFS=$'\t' read -r dataset model fold gpu rc; do
  if [[ "$rc" == 0 ]]; then
    printf '%s | %s | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS\n' "$model" "$dataset"
  elif [[ "$rc" == 2 ]]; then
    reason=$(PYTHONPATH="$ROOT/src" "${runner[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["reason"].replace("_", " "))' \
      "$ROOT/results/phase1/$dataset/$model/fold_$fold/blocked.json" 2>/dev/null || printf 'blocked')
    printf '%s | %s | PASS | NOT RUN | %s | NOT RUN | NOT RUN | NOT RUN | NOT RUN | PASS | BLOCKED\n' "$model" "$dataset" "$reason"
  else
    printf '%s | %s | UNKNOWN | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL\n' "$model" "$dataset"
  fi
done < "$work/status"
awk -F '\t' '$5 != 0 {bad=1} END {exit bad}' "$work/status"
