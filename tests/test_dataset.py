import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from medical_benchmark.datasets import ManifestDataset


class ManifestDatasetTest(unittest.TestCase):
    def test_milk10k_contract_and_return_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2)).save(root / "one.png")
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "label", "fold"])
                writer.writeheader()
                writer.writerow({"sample_id": "one", "image_path": "one.png", "label": "10", "fold": "0"})
            config = {"task": "multiclass", "num_classes": 11, "num_folds": 5, "manifest": str(manifest),
                      "image_root": str(root), "columns": {"image": "image_path", "target": "label", "fold": "fold", "id": "sample_id"}}
            sample = ManifestDataset(config, {0})[0]
            self.assertEqual(set(sample), {"image", "target", "sample_id", "fold"})
            self.assertEqual(sample["target"].item(), 10)

    def test_rejects_nonbinary_chexchonet_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2)).save(root / "one.png")
            manifest = root / "manifest.csv"
            manifest.write_text("sample_id,image_path,SLVH,DLV,fold\none,one.png,2,0,0\n", encoding="utf-8")
            config = {"task": "multilabel", "num_classes": 2, "num_folds": 5, "manifest": str(manifest),
                      "image_root": str(root), "columns": {"image": "image_path", "targets": ["SLVH", "DLV"], "fold": "fold", "id": "sample_id"}}
            with self.assertRaisesRegex(ValueError, "binary"):
                ManifestDataset(config, {0})


if __name__ == "__main__":
    unittest.main()
