#!/usr/bin/env python3
"""Evaluation is run from the best checkpoint at the end of each train job.

Use validate_run.py to validate its persisted metrics and predictions.
"""
from medical_benchmark.runners.validate_run import main

if __name__ == "__main__":
    main()
