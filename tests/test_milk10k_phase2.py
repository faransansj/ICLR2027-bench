import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from medical_benchmark.datasets import (
    Milk10kPairedDataset,
    build_milk10k_phase2_datasets,
    build_milk10k_phase2_manifest,
)
from medical_benchmark.datasets.milk10k_phase2 import LABELS


class Milk10kPhase2Test(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "TrainingData"
        image_root = source / "MILK10k_Training_Input"
        source.mkdir()
        folds = []
        truths = []
        metadata = []
        for index in range(5):
            lesion = f"IL_{index:07d}"
            clinical = f"ISIC_{index * 2:07d}"
            dermoscopic = f"ISIC_{index * 2 + 1:07d}"
            folder = image_root / lesion
            folder.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(folder / f"{clinical}.jpg")
            Image.new("RGB", (2, 2)).save(folder / f"{dermoscopic}.jpg")
            folds.append({"lesion_id": lesion, "fold": index, "diagnosis": LABELS[index]})
            truths.append({"lesion_id": lesion, **{name: float(i == index) for i, name in enumerate(LABELS)}})
            common = {"lesion_id": lesion, "age_approx": "70", "sex": "male", "skin_tone_class": "1", "site": "head_neck_face"}
            metadata.extend([
                {**common, "image_type": "clinical: close-up", "isic_id": clinical},
                {**common, "image_type": "dermoscopic", "isic_id": dermoscopic},
            ])
        self._write(source / "milk10k_5fold_seed42.csv", folds)
        self._write(source / "MILK10k_Training_GroundTruth.csv", truths)
        self._write(source / "MILK10k_Training_Metadata.csv", metadata)
        return source

    @staticmethod
    def _write(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_builds_and_loads_paired_manifest_with_existing_fold_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            manifest = build_milk10k_phase2_manifest(source, root / "manifest.csv")
            config = {
                "num_folds": 5,
                "manifest": str(manifest),
                "image_root": str(source / "MILK10k_Training_Input"),
            }
            sample = Milk10kPairedDataset(config, {0})[0]
            self.assertEqual(
                set(sample),
                {"clinical_image", "dermoscopic_image", "metadata", "target", "sample_id", "fold"},
            )
            self.assertEqual(sample["sample_id"], "IL_0000000")
            self.assertEqual(sample["target"].item(), 0)
            self.assertEqual(sample["metadata"]["age_approx"], 70.0)

            with patch("medical_benchmark.datasets.milk10k_phase2.load_yaml", return_value=config):
                splits = build_milk10k_phase2_datasets(0)
            self.assertEqual({name: len(value) for name, value in splits.items()}, {"train": 3, "validation": 1, "test": 1})
            self.assertEqual(splits["validation"][0]["fold"], 1)
            self.assertEqual(splits["test"][0]["fold"], 0)

    def test_rejects_incomplete_image_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            metadata = source / "MILK10k_Training_Metadata.csv"
            rows = list(csv.DictReader(metadata.open(newline="", encoding="utf-8")))
            self._write(metadata, rows[1:])
            with self.assertRaisesRegex(ValueError, "expected 2 images"):
                build_milk10k_phase2_manifest(source, root / "manifest.csv")


if __name__ == "__main__":
    unittest.main()
