#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
git submodule sync --recursive
git submodule update --init --recursive
if git submodule status --recursive | grep -qE '^[+-U]'; then
  echo "BLOCKED: an upstream submodule is missing or differs from its pinned gitlink" >&2
  exit 1
fi
git submodule status --recursive
