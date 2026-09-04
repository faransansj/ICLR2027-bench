#!/usr/bin/env python3
from __future__ import annotations

import argparse

from medical_benchmark.config import ROOT
from medical_benchmark.datasets.milk10k_phase2 import build_milk10k_phase2_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the validated paired-image MILK10k Phase 2 manifest")
    parser.add_argument("source_root", help="Path to the official MILK10k TrainingData directory")
    parser.add_argument("--output", default=str(ROOT / "data/milk10k/manifest_phase2.csv"))
    args = parser.parse_args()
    print(build_milk10k_phase2_manifest(args.source_root, args.output))


if __name__ == "__main__":
    main()
