from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def merge_runtime(profile: str, overrides: dict[str, Any]) -> dict[str, Any]:
    config = load_yaml(f"configs/runtime/{profile}.yaml")
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config
