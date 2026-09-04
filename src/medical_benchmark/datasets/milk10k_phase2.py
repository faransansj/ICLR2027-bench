from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from medical_benchmark.config import ROOT, load_yaml

LABELS = ("AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC")
METADATA_FIELDS = ("age_approx", "sex", "skin_tone_class", "site")
OUTPUT_FIELDS = (
    "sample_id",
    "clinical_path",
    "dermoscopic_path",
    *METADATA_FIELDS,
    "label",
    "diagnosis",
    "fold",
)


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def _index_unique(rows: list[dict[str, str]], key: str, source: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for line, row in enumerate(rows, 2):
        value = row[key].strip()
        if not value or value in indexed:
            raise ValueError(f"{source}:{line}: {key} is empty or duplicated")
        indexed[value] = row
    return indexed


def _shared(rows: list[dict[str, str]], field: str, lesion_id: str) -> str:
    values = {row[field].strip() for row in rows if row[field].strip()}
    if len(values) > 1:
        raise ValueError(f"{lesion_id}: conflicting {field}: {sorted(values)}")
    return next(iter(values), "")


def _image_path(image_root: Path, lesion_id: str, isic_id: str) -> str:
    if not re.fullmatch(r"IL_\d+", lesion_id) or not re.fullmatch(r"ISIC_\d+", isic_id):
        raise ValueError(f"invalid MILK10k identifiers: {lesion_id}, {isic_id}")
    relative = Path(lesion_id) / f"{isic_id}.jpg"
    path = (image_root / relative).resolve()
    if image_root.resolve() not in path.parents:
        raise ValueError(f"image escapes input root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return relative.as_posix()


def build_milk10k_phase2_manifest(source_root: str | Path, output: str | Path) -> Path:
    """Join the official MILK10k training tables into one paired-image manifest."""
    source_root = Path(source_root)
    image_root = source_root / "MILK10k_Training_Input"
    fold_path = source_root / "milk10k_5fold_seed42.csv"
    truth_path = source_root / "MILK10k_Training_GroundTruth.csv"
    metadata_path = source_root / "MILK10k_Training_Metadata.csv"

    folds = _index_unique(
        _read_csv(fold_path, {"lesion_id", "fold", "diagnosis"}), "lesion_id", fold_path
    )
    truths = _index_unique(
        _read_csv(truth_path, {"lesion_id", *LABELS}), "lesion_id", truth_path
    )
    metadata_rows = _read_csv(
        metadata_path, {"lesion_id", "image_type", "isic_id", *METADATA_FIELDS}
    )
    metadata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        metadata[row["lesion_id"].strip()].append(row)

    id_sets = {"folds": set(folds), "ground truth": set(truths), "metadata": set(metadata)}
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        counts = {name: len(ids) for name, ids in id_sets.items()}
        raise ValueError(f"MILK10k lesion IDs do not match: {counts}")

    output_rows: list[dict[str, str | int]] = []
    for lesion_id in sorted(folds):
        pair = metadata[lesion_id]
        if len(pair) != 2:
            raise ValueError(f"{lesion_id}: expected 2 images, found {len(pair)}")
        by_type = {row["image_type"].strip(): row for row in pair}
        if set(by_type) != {"clinical: close-up", "dermoscopic"}:
            raise ValueError(f"{lesion_id}: unexpected image types {sorted(by_type)}")

        truth = truths[lesion_id]
        try:
            one_hot = [float(truth[label]) for label in LABELS]
        except ValueError as exc:
            raise ValueError(f"{lesion_id}: ground truth must be numeric") from exc
        if any(value not in (0.0, 1.0) for value in one_hot) or sum(one_hot) != 1.0:
            raise ValueError(f"{lesion_id}: ground truth must be one-hot")
        label = one_hot.index(1.0)
        diagnosis = folds[lesion_id]["diagnosis"].strip()
        if diagnosis != LABELS[label]:
            raise ValueError(f"{lesion_id}: diagnosis and ground truth disagree")
        try:
            fold = int(folds[lesion_id]["fold"])
        except ValueError as exc:
            raise ValueError(f"{lesion_id}: fold must be an integer") from exc
        if fold not in range(5):
            raise ValueError(f"{lesion_id}: fold must be in [0, 4]")

        clinical = by_type["clinical: close-up"]
        dermoscopic = by_type["dermoscopic"]
        output_rows.append({
            "sample_id": lesion_id,
            "clinical_path": _image_path(image_root, lesion_id, clinical["isic_id"].strip()),
            "dermoscopic_path": _image_path(image_root, lesion_id, dermoscopic["isic_id"].strip()),
            **{field: _shared(pair, field, lesion_id) for field in METADATA_FIELDS},
            "label": label,
            "diagnosis": diagnosis,
            "fold": fold,
        })

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


class Milk10kPairedDataset(Dataset[dict[str, Any]]):
    """Validated clinical/dermoscopic pairs with raw core metadata."""

    def __init__(
        self,
        config: dict[str, Any],
        folds: Iterable[int],
        transform: Callable[[Image.Image], Any] | dict[str, Callable[[Image.Image], Any]] | None = None,
    ) -> None:
        self.image_root = (ROOT / config["image_root"]).resolve()
        self.transforms = transform if isinstance(transform, dict) else {"clinical": transform, "dermoscopic": transform}
        if set(self.transforms) != {"clinical", "dermoscopic"}:
            raise ValueError("paired transforms must define clinical and dermoscopic")
        self.rows = self._read_manifest((ROOT / config["manifest"]).resolve(), set(folds), int(config["num_folds"]))

    def _resolve_image(self, value: str, manifest: Path, line: int) -> Path:
        path = (self.image_root / value).resolve()
        if self.image_root not in path.parents:
            raise ValueError(f"{manifest}:{line}: image escapes image_root")
        if not path.is_file():
            raise FileNotFoundError(f"{manifest}:{line}: image not found: {path}")
        return path

    def _read_manifest(self, manifest: Path, folds: set[int], num_folds: int) -> list[dict[str, Any]]:
        required = set(OUTPUT_FIELDS)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line, row in enumerate(_read_csv(manifest, required), 2):
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"{manifest}:{line}: sample_id is empty or duplicated")
            seen.add(sample_id)
            try:
                fold = int(row["fold"])
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"{manifest}:{line}: fold and label must be integers") from exc
            if fold not in range(num_folds):
                raise ValueError(f"{manifest}:{line}: fold must be in [0, {num_folds - 1}]")
            if label not in range(len(LABELS)) or row["diagnosis"].strip() != LABELS[label]:
                raise ValueError(f"{manifest}:{line}: label and diagnosis disagree")
            age_text = row["age_approx"].strip()
            try:
                age = float(age_text) if age_text else 0.0
            except ValueError as exc:
                raise ValueError(f"{manifest}:{line}: age_approx must be numeric or empty") from exc
            if age_text and not 0 <= age <= 120:
                raise ValueError(f"{manifest}:{line}: age_approx must be in [0, 120]")
            if fold in folds:
                rows.append({
                    "sample_id": sample_id,
                    "clinical_path": self._resolve_image(row["clinical_path"], manifest, line),
                    "dermoscopic_path": self._resolve_image(row["dermoscopic_path"], manifest, line),
                    "metadata": {
                        "age_approx": age,
                        "age_approx_present": bool(age_text),
                        "sex": row["sex"].strip(),
                        "skin_tone_class": row["skin_tone_class"].strip(),
                        "site": row["site"].strip(),
                    },
                    "target": label,
                    "fold": fold,
                })
        if not rows:
            raise ValueError(f"manifest has no samples for folds {sorted(folds)}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        images: dict[str, Any] = {}
        for name in ("clinical", "dermoscopic"):
            with Image.open(row[f"{name}_path"]) as image:
                image = image.convert("RGB")
                transform = self.transforms[name]
                images[name] = transform(image) if transform else image.copy()
        return {
            "clinical_image": images["clinical"],
            "dermoscopic_image": images["dermoscopic"],
            "metadata": row["metadata"],
            "target": torch.tensor(row["target"], dtype=torch.long),
            "sample_id": row["sample_id"],
            "fold": row["fold"],
        }


def build_milk10k_phase2_datasets(
    fold: int,
    transform: Callable[[Image.Image], Any] | dict[str, Callable[[Image.Image], Any]] | None = None,
) -> dict[str, Milk10kPairedDataset]:
    config = load_yaml("configs/datasets/milk10k_phase2.yaml")
    num_folds = int(config["num_folds"])
    if fold not in range(num_folds):
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    validation_fold = (fold + 1) % num_folds
    train_folds = set(range(num_folds)) - {fold, validation_fold}
    return {
        "train": Milk10kPairedDataset(config, train_folds, transform),
        "validation": Milk10kPairedDataset(config, {validation_fold}, transform),
        "test": Milk10kPairedDataset(config, {fold}, transform),
    }
