from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

REQUIRED = ("manifest.json", "metrics.json", "predictions.csv", "best.pt", "last.pt")


def _check_finite(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite metric at {location}")


def validate_run(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if missing:
        raise ValueError(f"missing required artifacts: {missing}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"run status is not COMPLETED: {manifest.get('status')}")
    identity = manifest.get("run_identity")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("manifest has no valid run identity")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    for required_split in ("train", "validation", "test"):
        if not isinstance(metrics.get(required_split), dict) or "loss" not in metrics[required_split]:
            raise ValueError(f"metrics missing {required_split} split")
    _check_finite(metrics, "metrics")
    for checkpoint_name in ("best.pt", "last.pt"):
        checkpoint = torch.load(path / checkpoint_name, map_location="cpu", weights_only=False)
        if checkpoint.get("run_identity") != identity:
            raise ValueError(f"{checkpoint_name} run identity mismatch")
    with (path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or not rows[0] or rows[0][0] != "sample_id":
        raise ValueError("predictions.csv is empty or malformed")
    provenance = manifest.get("provenance", {})
    for field in ("git_commit", "git_diff", "files_sha256", "torch", "cuda_runtime"):
        if field not in provenance:
            raise ValueError(f"manifest provenance missing {field}")
    return {"status": "VALID", "path": str(path), "run_identity": identity}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one completed benchmark run")
    parser.add_argument("run")
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run), indent=2))


if __name__ == "__main__":
    main()
