#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=""
GPUS="0"
FOLDS="0,1,2,3,4"
while (($#)); do
  case "$1" in
    --mode) MODE=${2:-}; shift 2 ;;
    --gpus) GPUS=${2:-}; shift 2 ;;
    --folds) FOLDS=${2:-}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo "usage: $0 --mode smoke|full --gpus 0,1 [--folds 0,1,2,3,4]" >&2
  exit 64
fi
IFS=, read -r -a gpu_list <<< "$GPUS"
((${#gpu_list[@]})) || { echo "--gpus cannot be empty" >&2; exit 64; }
for gpu in "${gpu_list[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }; done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
jobs="$work/jobs"
if [[ "$MODE" == smoke ]]; then
  printf '%s\n' \
    $'milk10k\tmambavision\t0' $'milk10k\ttransnext\t0' \
    $'chexchonet\tchexworld\t0' $'chexchonet\tlrfl\t0' \
    $'chexchonet\tx_win\t0' $'chexchonet\tcarzero\t0' > "$jobs"
  runtime=local
else
  IFS=, read -r -a fold_list <<< "$FOLDS"
  : > "$jobs"
  for fold in "${fold_list[@]}"; do
    [[ "$fold" =~ ^[0-4]$ ]] || { echo "invalid fold: $fold (expected 0-4)" >&2; exit 64; }
    printf 'milk10k\tmambavision\t%s\nmilk10k\ttransnext\t%s\n' "$fold" "$fold" >> "$jobs"
    for model in chexworld lrfl x_win carzero; do printf 'chexchonet\t%s\t%s\n' "$model" "$fold" >> "$jobs"; done
  done
  runtime=server
fi

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
    echo "START gpu=$gpu dataset=$dataset model=$model fold=$fold"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src" "${runner[@]}" -m medical_benchmark.runners.train \
      --dataset "$dataset" --model "$model" --fold "$fold" --runtime "$runtime" --output-root "$ROOT/results"
    rc=$?
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$model" "$fold" "$gpu" "$rc" >> "$work/status"
    flock -u 9
  done
}
for gpu in "${gpu_list[@]}"; do worker "$gpu" & done
wait

blocked=$(awk -F '\t' '$5 == 2 {n++} END {print n+0}' "$work/status")
failed=$(awk -F '\t' '$5 != 0 && $5 != 2 {n++} END {print n+0}' "$work/status")
completed=$(awk -F '\t' '$5 == 0 {n++} END {print n+0}' "$work/status")
total=$(wc -l < "$jobs")
echo "SUMMARY total=$total completed_or_skipped=$completed blocked=$blocked failed=$failed"
if ((blocked || failed)); then
  echo "BENCHMARK INCOMPLETE: inspect results/*/blocked.json and job output" >&2
  exit 1
fi
