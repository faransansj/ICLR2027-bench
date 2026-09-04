"""Shared Phase 1 model contract."""
from typing import Protocol

import torch


class BenchmarkModel(Protocol):
    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...
