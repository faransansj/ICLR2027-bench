from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from medical_benchmark.config import ROOT, load_yaml
from medical_benchmark.models.registry import sha256_file


def download(model: str) -> Path:
    source = load_yaml("configs/source_lock.yaml")["sources"].get(model)
    checkpoint = source and source.get("checkpoint")
    if not checkpoint or not checkpoint.get("url") or not checkpoint.get("sha256"):
        raise ValueError(f"{model}: automatic download requires an immutable URL and configured sha256")
    destination = ROOT / checkpoint["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == checkpoint["sha256"]:
        return destination
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(checkpoint["url"], temporary)
        actual = sha256_file(temporary)
        if actual != checkpoint["sha256"]:
            raise ValueError(f"sha256 mismatch: expected {checkpoint['sha256']}, found {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only immutable, hash-pinned official checkpoints")
    parser.add_argument("model", choices=("mambavision", "transnext"))
    args = parser.parse_args()
    print(download(args.model))


if __name__ == "__main__":
    main()
