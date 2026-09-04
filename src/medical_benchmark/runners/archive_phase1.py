from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from medical_benchmark.config import ROOT
from medical_benchmark.metrics import compute_metrics

SNAPSHOT_FILES = ("README.md", "pyproject.toml", "uv.lock")
SNAPSHOT_DIRS = ("configs", "src", "scripts", "runners", "models", "datasets", "metrics", "patches", "tests")
RESULT_FILES = {"run_manifest.json", "fold_metrics.json", "predictions.csv", "train.log", "blocked.json", "summary.json", "dataset_integrity.json"}
COMPLETED_FILES = {"run_manifest.json", "fold_metrics.json", "predictions.csv", "train.log", "best.pt", "last.pt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_snapshot(project: Path, destination: Path) -> None:
    code = destination / "code"
    code.mkdir()
    for name in SNAPSHOT_FILES:
        source = project / name
        if source.is_file():
            shutil.copy2(source, code / name)
    for name in SNAPSHOT_DIRS:
        source = project / name
        if source.is_dir():
            shutil.copytree(source, code / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    environment = destination / "environment"
    environment.mkdir()
    for name in ("flake.nix", "flake.lock"):
        source = project.parent / name
        if source.is_file():
            shutil.copy2(source, environment / name)


def _copy_results(project: Path, destination: Path) -> None:
    source_root = project / "results" / "phase1"
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    result_root = destination / "results" / "phase1"
    for source in source_root.rglob("*"):
        if source.is_file() and source.name in RESULT_FILES:
            target = result_root / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _derived_metrics(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty predictions: {path}")
    columns = set(rows[0])
    probability_columns = sorted(
        (name for name in columns if name.startswith("probability_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if not probability_columns:
        raise ValueError(f"no probability columns: {path}")
    probabilities = np.asarray([[float(row[name]) for name in probability_columns] for row in rows])
    if "target" in columns:
        targets = np.asarray([int(row["target"]) for row in rows])
        task = "multiclass"
    else:
        target_columns = [f"target_{index}" for index in range(len(probability_columns))]
        if not set(target_columns) <= columns:
            raise ValueError(f"target/probability columns disagree: {path}")
        targets = np.asarray([[float(row[name]) for name in target_columns] for row in rows])
        task = "multilabel"
    return {"task": task, "samples": len(rows), **compute_metrics(task, targets, probabilities)}


def _run_report(project: Path, destination: Path) -> dict[str, Any]:
    source_root = project / "results" / "phase1"
    report: dict[str, Any] = {"completed_runs": 0, "blocked_runs": 0, "models": []}
    for model_dir in sorted(source_root.glob("*/*")):
        if not model_dir.is_dir():
            continue
        model: dict[str, Any] = {
            "dataset": model_dir.parent.name,
            "model": model_dir.name,
            "completed_folds": 0,
            "blocked_folds": 0,
            "derived_aggregate": {},
        }
        derived: list[dict[str, Any]] = []
        for fold_dir in sorted(model_dir.glob("fold_*")):
            manifest_path = fold_dir / "run_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = manifest.get("status")
            if status == "COMPLETED":
                missing = sorted(name for name in COMPLETED_FILES if not (fold_dir / name).is_file())
                if missing:
                    raise ValueError(f"{fold_dir}: completed run missing {missing}")
                metrics = _derived_metrics(fold_dir / "predictions.csv")
                metrics.update({"fold": manifest["dataset"]["fold"], "run_id": manifest["run_id"]})
                archived = destination / "results" / "phase1" / fold_dir.relative_to(source_root) / "derived_metrics.json"
                archived.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                derived.append(metrics)
                model["completed_folds"] += 1
                report["completed_runs"] += 1
            elif status == "BLOCKED":
                if not (fold_dir / "blocked.json").is_file():
                    raise ValueError(f"{fold_dir}: blocked run missing blocked.json")
                model["blocked_folds"] += 1
                report["blocked_runs"] += 1
        for metric in ("accuracy", "macro_f1", "macro_auroc", "emr"):
            values = [float(item[metric]) for item in derived if item.get(metric) is not None]
            if values:
                model["derived_aggregate"][metric] = {"mean": mean(values), "std": pstdev(values)}
        model["status"] = "COMPLETED" if model["completed_folds"] == 5 else "BLOCKED" if model["blocked_folds"] == 5 else "PARTIAL"
        report["models"].append(model)
    return report


def archive_phase1(project_root: str | Path = ROOT, output: str | Path | None = None) -> Path:
    """Create a portable Phase 1 archive without copying checkpoint tensors."""
    project = Path(project_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output) if output else project / "archives" / f"phase1-{timestamp}.tar.gz"
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        archive_root = Path(temporary) / "phase1"
        archive_root.mkdir()
        _copy_snapshot(project, archive_root)
        _copy_results(project, archive_root)
        report = _run_report(project, archive_root)

        checkpoints = [
            {"path": str(path.relative_to(project)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted((project / "results" / "phase1").rglob("*.pt"))
        ]
        (archive_root / "phase1_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (archive_root / "checkpoint_inventory.json").write_text(json.dumps(checkpoints, indent=2) + "\n", encoding="utf-8")

        files = [
            {"path": path.relative_to(archive_root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(archive_root.rglob("*"))
            if path.is_file()
        ]
        manifest = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(), "files": files}
        (archive_root / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with tarfile.open(output, "w:gz") as archive:
            archive.add(archive_root, arcname="phase1")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
    return output
