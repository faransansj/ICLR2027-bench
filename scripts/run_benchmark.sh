#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_phase1.sh" "$@"
