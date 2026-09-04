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
    if not destination.is_file() or sha256_file(destination) != checkpoint["sha256"]:
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
    if model == "radzero":
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            source["model_repository"], revision=source["model_revision"], local_dir=destination.parent / "hf",
            allow_patterns=["*.py", "*.json", "*.txt"], ignore_patterns=["data/*"],
        ))
        local_checkpoint = snapshot / destination.name
        local_checkpoint.unlink(missing_ok=True)
        local_checkpoint.symlink_to(Path("..") / destination.name)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only immutable, hash-pinned official checkpoints")
    parser.add_argument("model", choices=("mambavision", "transnext", "radzero"))
    args = parser.parse_args()
    print(download(args.model))


if __name__ == "__main__":
    main()
