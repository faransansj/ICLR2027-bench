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
            run = root / "phase1" / "milk10k" / "mambavision" / "fold_0"
            run.mkdir(parents=True)
            identity = "a" * 64
            provenance = {
                "benchmark_commit": "abc", "uv_lock_hash": "b" * 64,
                "dataset_manifest_hash": "c" * 64, "split_manifest_hash": "c" * 64,
                "cuda_version": None, "pytorch_version": torch.__version__,
                "timestamp_start": "start", "timestamp_end": "end",
            }
            manifest = {
                "schema_version": "1.0", "status": "COMPLETED", "run_id": identity,
                "dataset": {"name": "milk10k", "fold": 0}, "model": {"name": "mambavision"},
                "provenance": provenance,
            }
            metrics = {
                "schema_version": "1.0", "status": "completed", "run_id": identity,
                "validation": {"loss": 0.8},
                "test": {"macro_f1": 0.5, "macro_auroc": 0.75, "class_f1": {}, "label_f1": {}, "state_f1": {}, "emr": None},
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "fold_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (run / "predictions.csv").write_text("sample_id,target\none,0\n", encoding="utf-8")
            (run / "train.log").write_text("ok\n", encoding="utf-8")
            for name in ("best.pt", "last.pt"):
                torch.save({"run_identity": identity}, run / name)
            self.assertEqual(validate_run(run)["status"], "VALID")
            summary = aggregate(root)["summaries"][0]
            self.assertEqual(summary["completed_folds"], 1)
            self.assertEqual(summary["aggregate"]["macro_f1"]["mean"], 0.5)
            self.assertEqual(summary["aggregate"]["macro_auroc"]["mean"], 0.75)
            self.assertTrue((run.parent / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
