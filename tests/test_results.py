import json
import tempfile
import unittest
from pathlib import Path

import torch

from medical_benchmark.runners.aggregate import aggregate
from medical_benchmark.runners.validate_run import validate_run


class ResultToolsTest(unittest.TestCase):
    def test_validate_and_aggregate_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "local" / "milk10k" / "mambavision" / "fold-0"
            run.mkdir(parents=True)
            identity = "a" * 64
            provenance = {"git_commit": "abc", "git_diff": "", "files_sha256": {}, "torch": torch.__version__, "cuda_runtime": None}
            manifest = {"status": "COMPLETED", "run_identity": identity, "dataset": "milk10k", "model": "mambavision",
                        "fold": 0, "provenance": provenance}
            metrics = {"train": {"loss": 1.0}, "validation": {"loss": 0.8}, "test": {"loss": 0.7, "accuracy": 0.5}}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (run / "predictions.csv").write_text("sample_id,target\none,0\n", encoding="utf-8")
            for name in ("best.pt", "last.pt"):
                torch.save({"run_identity": identity}, run / name)
            self.assertEqual(validate_run(run)["status"], "VALID")
            summary = aggregate(root)
            self.assertEqual(len(summary["completed_runs"]), 1)
            self.assertEqual(summary["groups"][0]["test_mean"]["accuracy"], 0.5)
            self.assertTrue((root / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
