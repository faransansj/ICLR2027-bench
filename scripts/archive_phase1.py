#!/usr/bin/env python3
from __future__ import annotations

import argparse

from medical_benchmark.runners.archive_phase1 import archive_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive Phase 1 results without copying checkpoint tensors")
    parser.add_argument("--output", help="Output .tar.gz path; defaults to archives/phase1-<UTC timestamp>.tar.gz")
    args = parser.parse_args()
    print(archive_phase1(output=args.output))


if __name__ == "__main__":
    main()
