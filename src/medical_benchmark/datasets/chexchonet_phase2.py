from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medical_benchmark.config import ROOT, load_yaml
from medical_benchmark.datasets.manifest import ManifestDataset
from medical_benchmark.datasets.milk10k_phase2 import _index_unique, _read_csv

SEX_ENCODING = {"F": 0.0, "M": 1.0}
FIELDS = ("sample_id", "image_path", "SLVH", "DLV", "fold", "patient_id", "age", "sex")


def _metadata(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    patients: dict[str, int] = {}
    features = {}
    for row in rows:
        sample = row["sample_id"]
        patient = row["patient_id"].strip()
        fold = int(row["fold"])
        if not patient or patients.setdefault(patient, fold) != fold:
            raise ValueError(f"{sample}: empty patient_id or patient occurs in multiple folds")
        try:
            age = float(row["age"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{sample}: age must be present and numeric") from exc
        if not math.isfinite(age) or not 0 <= age <= 120:
            raise ValueError(f"{sample}: age must be finite and in [0, 120]")
        sex = row["sex"]
        if sex not in SEX_ENCODING:
            raise ValueError(f"{sample}: sex must be F or M, got {sex!r}")
        features[sample] = [age, SEX_ENCODING[sex]]
    return features


def build_chexchonet_phase2_manifest(metadata: str | Path, manifest: str | Path, output: str | Path) -> Path:
    """Join only age/sex onto the existing split; never use echo measurements as inputs."""
    metadata, manifest, output = Path(metadata), Path(manifest), Path(output)
    if output.resolve() in {metadata.resolve(), manifest.resolve()}:
        raise ValueError("output must not overwrite an input manifest")
    base = _read_csv(manifest, set(FIELDS[:5]))
    _index_unique(base, "sample_id", manifest)
    source = _index_unique(_read_csv(metadata, {"cxr_filename", "patient_id", "age", "sex", "slvh", "dlv"}), "cxr_filename", metadata)
    rows = []
    for row in base:
        filename = Path(row["image_path"]).name
        if filename not in source:
            raise ValueError(f"{row['sample_id']}: metadata missing for {filename}")
        extra = source[filename]
        if any(float(row[target]) != float(extra[target.lower()]) for target in ("SLVH", "DLV")):
            raise ValueError(f"{row['sample_id']}: metadata and manifest targets disagree")
        if int(row["fold"]) not in range(5) or any(row[target] not in {"0", "1"} for target in ("SLVH", "DLV")):
            raise ValueError(f"{row['sample_id']}: invalid fold or binary target")
        rows.append({**{key: row[key] for key in FIELDS[:5]}, **{key: extra[key] for key in FIELDS[5:]}})
    _metadata(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


class ChexchonetTabularDataset(ManifestDataset):
    def __init__(self, config: dict[str, Any], folds: set[int], transform: Any, age_stats: dict[str, float] | None = None) -> None:
        # Validate metadata before images, so a missing modality gives an actionable error.
        manifest = ROOT / config["manifest"]
        raw = _read_csv(manifest, set(FIELDS))
        features = _metadata(raw)
        super().__init__(config, folds, transform)
        ages = np.array([features[row["sample_id"]][0] for row in self.rows])
        self.age_stats = age_stats if age_stats is not None else {"mean": float(ages.mean()), "std": float(ages.std()) or 1.0}
        for row in self.rows:
            age, sex = features[row["sample_id"]]
            row["tabular"] = [(age - self.age_stats["mean"]) / self.age_stats["std"], sex]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["tabular"] = torch.tensor(self.rows[index]["tabular"], dtype=torch.float32)
        # Configured state order: neither, SLVH only, DLV only, both.
        item["target"] = (item["target"][0] + 2 * item["target"][1]).long()
        return item


def build_chexchonet_phase2_datasets(fold: int, transform: Any) -> dict[str, ChexchonetTabularDataset]:
    config = load_yaml("configs/datasets/chexchonet_phase2.yaml")
    if fold not in range(int(config["num_folds"])):
        raise ValueError("fold must be in [0, 4]")
    validation = (fold + 1) % int(config["num_folds"])
    train = ChexchonetTabularDataset(config, set(range(int(config["num_folds"]))) - {fold, validation}, transform)
    return {
        "train": train,
        "validation": ChexchonetTabularDataset(config, {validation}, transform, train.age_stats),
        "test": ChexchonetTabularDataset(config, {fold}, transform, train.age_stats),
    }
