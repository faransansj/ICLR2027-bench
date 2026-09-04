from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn


class RadZeroCheXchoNetAdapter(nn.Module):
    """Expose RadZero's text-conditioned similarities as two CheXchoNet logits."""

    def __init__(self, backbone: nn.Module, encoded_prompts: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("prompt_input_ids", encoded_prompts["input_ids"])
        self.register_buffer("prompt_attention_mask", encoded_prompts["attention_mask"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = {
            "input_ids": self.prompt_input_ids,
            "attention_mask": self.prompt_attention_mask,
        }
        logits = self.backbone.compute_logits(images, [encoded])["logits"]
        if logits.ndim == 1 and images.shape[0] == 1:
            logits = logits.unsqueeze(0)
        expected = (images.shape[0], 2)
        if tuple(logits.shape) != expected:
            raise ValueError(f"RadZero logits shape must be {expected}, found {tuple(logits.shape)}")
        return logits


class RadZeroImageTransform:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def load_radzero_chexchonet(
    snapshot: str | Path, prompts: Sequence[str]
) -> tuple[RadZeroCheXchoNetAdapter, RadZeroImageTransform]:
    if len(prompts) != 2:
        raise ValueError("CheXchoNet requires exactly two prompts ordered as SLVH, DLV")

    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    snapshot = Path(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt")
    model = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()
    return RadZeroCheXchoNetAdapter(model, encoded), RadZeroImageTransform(processor)
