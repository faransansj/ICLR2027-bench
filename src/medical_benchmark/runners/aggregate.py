from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from medical_benchmark.config import ROOT
from medical_benchmark.runners.validate_run import validate_run


def aggregate(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    runs: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for manifest_path in sorted(root.glob("**/manifest.json")):
        run_path = manifest_path.parent
        try:
            validate_run(run_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            run = {
                "path": str(run_path.relative_to(root)),
                "dataset": manifest["dataset"],
                "model": manifest["model"],
                "fold": manifest["fold"],
                "test": metrics["test"],
            }
            runs.append(run)
            for key, value in metrics["test"].items():
                if isinstance(value, (int, float)):
                    grouped[(manifest["dataset"], manifest["model"])][key].append(float(value))
        except Exception as exc:
            invalid.append({"path": str(run_path), "error": str(exc)})
    groups = []
    for (dataset, model), values in sorted(grouped.items()):
        groups.append({"dataset": dataset, "model": model, "folds": sum(1 for run in runs if run["dataset"] == dataset and run["model"] == model),
                       "test_mean": {key: mean(items) for key, items in sorted(values.items())}})
    blocked = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("**/blocked.json"))]
    summary = {"completed_runs": runs, "groups": groups, "blocked_runs": blocked, "invalid_runs": invalid}
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate validated benchmark runs")
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args()
    summary = aggregate(args.results)
    print(json.dumps({"completed": len(summary["completed_runs"]), "blocked": len(summary["blocked_runs"]), "invalid": len(summary["invalid_runs"])}, indent=2))


if __name__ == "__main__":
    main()
