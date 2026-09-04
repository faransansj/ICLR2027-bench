#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=${1:?usage: $0 /path/to/chexchonet-1.0.0}
OUT=${2:-$ROOT/results/phase1/chexchonet/dataset_integrity.json}
LOG=${OUT%.json}.log
mkdir -p "$(dirname "$OUT")"

started=$(date --iso-8601=seconds)
(cd "$DATASET" && sha256sum --quiet -c SHA256SUMS.txt) >"$LOG" 2>&1
rc=$?
ended=$(date --iso-8601=seconds)
set -e

uv run python - "$DATASET" "$ROOT/data/chexchonet/manifest.csv" "$OUT" "$LOG" "$started" "$ended" "$rc" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
log = Path(sys.argv[4]).resolve()
returncode = int(sys.argv[7])


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

sum_file = source / "SHA256SUMS.txt"
result = {
    "schema_version": "1.0",
    "dataset": "chexchonet",
    "status": "PASS" if returncode == 0 else "FAIL",
    "verified_entries": sum(1 for line in sum_file.read_text().splitlines() if line.strip()),
    "sha256sums_sha256": sha256(sum_file),
    "source_files": {
        name: sha256(source / name)
        for name in ("metadata.csv", "metadata_4class.csv", "fivefold.csv")
    },
    "benchmark_manifest_sha256": sha256(manifest),
    "timestamp_start": sys.argv[5],
    "timestamp_end": sys.argv[6],
    "verification_log": str(log),
}
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

exit "$rc"
