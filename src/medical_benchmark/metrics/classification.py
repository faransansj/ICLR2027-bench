from __future__ import annotations

from typing import Any

import numpy as np


def require_finite(value: Any, name: str = "value") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{name}[{index}]")
        return
    if value is not None and not np.isfinite(np.asarray(value, dtype=float)).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _binary_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    true_positive = int(((target == 1) & (prediction == 1)).sum())
    false_positive = int(((target == 0) & (prediction == 1)).sum())
    false_negative = int(((target == 1) & (prediction == 0)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _binary_auroc(target: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[target == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def compute_metrics(task: str, targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    require_finite(targets, "targets")
    require_finite(probabilities, "probabilities")
    if task == "multiclass":
        prediction = probabilities.argmax(axis=1)
        classes = probabilities.shape[1]
        f1 = [_binary_f1((targets == index).astype(int), (prediction == index).astype(int)) for index in range(classes)]
        return {
            "accuracy": float((prediction == targets).mean()),
            "macro_f1": float(np.mean(f1)),
            "class_f1": {str(index): float(value) for index, value in enumerate(f1)},
        }
    if task == "multilabel":
        prediction = (probabilities >= 0.5).astype(int)
        f1 = [_binary_f1(targets[:, index], prediction[:, index]) for index in range(targets.shape[1])]
        aucs = [_binary_auroc(targets[:, index], probabilities[:, index]) for index in range(targets.shape[1])]
        valid_aucs = [value for value in aucs if value is not None]
        label_names = ("SLVH", "DLV")
        return {
            "emr": float((prediction == targets).all(axis=1).mean()),
            "macro_f1": float(np.mean(f1)),
            "label_f1": {name: float(value) for name, value in zip(label_names, f1, strict=True)},
            "macro_auroc": float(np.mean(valid_aucs)) if valid_aucs else None,
        }
    raise ValueError(f"unsupported task: {task}")
