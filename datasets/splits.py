"""Deterministic five-fold split policy."""


def split_folds(test_fold: int, num_folds: int = 5) -> dict[str, set[int]]:
    if test_fold not in range(num_folds):
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    validation_fold = (test_fold + 1) % num_folds
    return {"train": set(range(num_folds)) - {test_fold, validation_fold}, "validation": {validation_fold}, "test": {test_fold}}
