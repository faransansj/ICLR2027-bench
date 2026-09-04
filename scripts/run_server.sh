#!/usr/bin/env bash
set -euo pipefail
: "${GPUS:?Set GPUS, e.g. GPUS=0,1,2,3}"
exec "$(dirname "$0")/run_phase1.sh" --mode full --gpus "$GPUS" --folds "${FOLDS:-0,1,2,3,4}"
