from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from medical_benchmark.config import ROOT, load_yaml


class ManifestDataset(Dataset[dict[str, Any]]):
    """Dataset backed only by an explicit, validated CSV manifest."""

    def __init__(
        self,
        config: dict[str, Any],
        folds: Iterable[int],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.task = str(config["task"])
        self.num_classes = int(config["num_classes"])
        self.num_folds = int(config["num_folds"])
        self.image_root = (ROOT / config["image_root"]).resolve()
        self.rows = self._read_manifest((ROOT / config["manifest"]).resolve(), set(folds))

    def _read_manifest(self, manifest: Path, folds: set[int]) -> list[dict[str, Any]]:
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        columns = self.config["columns"]
        targets = [columns["target"]] if self.task == "multiclass" else list(columns["targets"])
        required = {columns["image"], columns["id"], columns["fold"], *targets}
        parsed: list[dict[str, Any]] = []
        seen: set[str] = set()
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{manifest}: missing columns {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                sample_id = row[columns["id"]].strip()
                if not sample_id or sample_id in seen:
                    raise ValueError(f"{manifest}:{line}: sample_id is empty or duplicated")
                seen.add(sample_id)
                try:
                    fold = int(row[columns["fold"]])
                except ValueError as exc:
                    raise ValueError(f"{manifest}:{line}: fold must be an integer") from exc
                if fold not in range(self.num_folds):
                    raise ValueError(f"{manifest}:{line}: fold must be in [0, {self.num_folds - 1}]")
                image_path = (self.image_root / row[columns["image"]]).resolve()
                if self.image_root not in image_path.parents:
                    raise ValueError(f"{manifest}:{line}: image escapes image_root")
                if not image_path.is_file():
                    raise FileNotFoundError(f"{manifest}:{line}: image not found: {image_path}")
                if self.task == "multiclass":
                    try:
                        target: int | list[float] = int(row[targets[0]])
                    except ValueError as exc:
                        raise ValueError(f"{manifest}:{line}: label must be an integer") from exc
                    if target not in range(self.num_classes):
                        raise ValueError(f"{manifest}:{line}: label must be in [0, {self.num_classes - 1}]")
                elif self.task == "multilabel":
                    try:
                        target = [float(row[name]) for name in targets]
                    except ValueError as exc:
                        raise ValueError(f"{manifest}:{line}: targets must be 0 or 1") from exc
                    if len(target) != self.num_classes or any(value not in (0.0, 1.0) for value in target):
                        raise ValueError(f"{manifest}:{line}: targets must be {self.num_classes} binary values")
                else:
                    raise ValueError(f"unsupported task: {self.task}")
                if fold in folds:
                    parsed.append({"sample_id": sample_id, "path": image_path, "target": target, "fold": fold})
        if not parsed:
            raise ValueError(f"manifest has no samples for folds {sorted(folds)}")
        return parsed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            value = self.transform(image) if self.transform else image.copy()
        dtype = torch.long if self.task == "multiclass" else torch.float32
        return {
            "image": value,
            "target": torch.tensor(row["target"], dtype=dtype),
            "sample_id": row["sample_id"],
            "fold": row["fold"],
        }


def build_fold_datasets(
    name: str, fold: int, transform: Callable[[Image.Image], torch.Tensor] | dict[str, Callable[[Image.Image], torch.Tensor]]
) -> dict[str, ManifestDataset]:
    config = load_yaml(f"configs/datasets/{name}.yaml")
    num_folds = int(config["num_folds"])
    if fold not in range(num_folds):
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    validation_fold = (fold + 1) % num_folds
    train_folds = set(range(num_folds)) - {fold, validation_fold}
    transforms = transform if isinstance(transform, dict) else {"train": transform, "validation": transform, "test": transform}
    if set(transforms) != {"train", "validation", "test"}:
        raise ValueError("split transforms must define train, validation, and test")
    return {
        "train": ManifestDataset(config, train_folds, transforms["train"]),
        "validation": ManifestDataset(config, {validation_fold}, transforms["validation"]),
        "test": ManifestDataset(config, {fold}, transforms["test"]),
    }
