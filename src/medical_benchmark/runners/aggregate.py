from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from medical_benchmark.config import ROOT
from medical_benchmark.runners.validate_run import validate_run


def aggregate(root: str | Path) -> dict[str, Any]:
    """Validate completed folds and write one schema-1.0 summary per model."""
    root = Path(root)
    summaries: list[dict[str, Any]] = []
    for model_dir in sorted((root / "phase1").glob("*/*")):
        if not model_dir.is_dir():
            continue
        folds: list[dict[str, Any]] = []
        for manifest_path in sorted(model_dir.glob("fold_*/run_manifest.json")):
            run_dir = manifest_path.parent
            try:
                validate_run(run_dir)
            except ValueError:
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "fold_metrics.json").read_text(encoding="utf-8"))
            folds.append({"fold": manifest["dataset"]["fold"], "run_id": manifest["run_id"], "test": metrics["test"]})
        macro = [float(item["test"]["macro_f1"]) for item in folds]
        auroc = [float(item["test"]["macro_auroc"]) for item in folds if item["test"].get("macro_auroc") is not None]
        emr = [float(item["test"]["emr"]) for item in folds if item["test"].get("emr") is not None]
        summary = {
            "schema_version": "1.0",
            "experiment_id": f"phase1-{model_dir.parent.name}-{model_dir.name}",
            "phase": 1,
            "model": model_dir.name,
            "dataset": model_dir.parent.name,
            "completed_folds": len(folds),
            "folds": folds,
            "aggregate": {
                "macro_f1": {"mean": mean(macro) if macro else 0.0, "std": pstdev(macro) if macro else 0.0},
                "macro_auroc": {"mean": mean(auroc) if auroc else None, "std": pstdev(auroc) if auroc else None},
                "emr": {"mean": mean(emr) if emr else None, "std": pstdev(emr) if emr else None},
            },
        }
        (model_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
    return {"summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate validated Phase 1 folds")
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args()
    result = aggregate(args.results)
    print(json.dumps({"models": len(result["summaries"])}, indent=2))


if __name__ == "__main__":
    main()
