import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from medical_benchmark.runners.archive_phase1 import archive_phase1


class Phase1ArchiveTest(unittest.TestCase):
    def test_archives_results_and_checkpoint_hashes_without_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            run = project / "results/phase1/chexchonet/chexworld/fold_0"
            run.mkdir(parents=True)
            (project / "configs").mkdir()
            (project / "README.md").write_text("phase 1\n", encoding="utf-8")
            (project / "configs/example.yaml").write_text("version: 1\n", encoding="utf-8")
            identity = "a" * 64
            (run / "run_manifest.json").write_text(
                json.dumps({"status": "COMPLETED", "run_id": identity, "dataset": {"fold": 0}}),
                encoding="utf-8",
            )
            (run / "fold_metrics.json").write_text("{}\n", encoding="utf-8")
            (run / "train.log").write_text("status=completed\n", encoding="utf-8")
            (run / "predictions.csv").write_text(
                "sample_id,target_0,prediction_0,probability_0,target_1,prediction_1,probability_1\n"
                "a,0,0,0.1,1,1,0.9\n"
                "b,1,1,0.9,0,0,0.1\n",
                encoding="utf-8",
            )
            checkpoint = b"checkpoint"
            for name in ("best.pt", "last.pt"):
                (run / name).write_bytes(checkpoint)

            output = archive_phase1(project, project / "archive.tar.gz")
            checksum = output.with_suffix(output.suffix + ".sha256").read_text(encoding="utf-8")
            self.assertEqual(checksum, f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n")
            with tarfile.open(output) as archive:
                names = archive.getnames()
                self.assertFalse(any(name.endswith(".pt") for name in names))
                report = json.load(archive.extractfile("phase1/phase1_report.json"))
                inventory = json.load(archive.extractfile("phase1/checkpoint_inventory.json"))
                derived = json.load(
                    archive.extractfile("phase1/results/phase1/chexchonet/chexworld/fold_0/derived_metrics.json")
                )
            self.assertEqual(report["completed_runs"], 1)
            self.assertEqual(derived["macro_auroc"], 1.0)
            self.assertEqual(len(inventory), 2)
            self.assertEqual({item["sha256"] for item in inventory}, {hashlib.sha256(checkpoint).hexdigest()})


if __name__ == "__main__":
    unittest.main()
