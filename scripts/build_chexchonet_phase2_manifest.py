#!/usr/bin/env python3
"""Add official age/sex metadata without changing the benchmark split."""
import argparse

from medical_benchmark.config import ROOT
from medical_benchmark.datasets.chexchonet_phase2 import build_chexchonet_phase2_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", help="Official CheXchoNET metadata.csv")
    parser.add_argument("--manifest", default=str(ROOT / "data/chexchonet/manifest.csv"))
    parser.add_argument("--output", default=str(ROOT / "data/chexchonet/manifest_phase2.csv"))
    args = parser.parse_args()
    print(build_chexchonet_phase2_manifest(args.metadata, args.manifest, args.output))
