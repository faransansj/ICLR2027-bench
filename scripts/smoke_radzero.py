#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from medical_benchmark.config import ROOT


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    path = ROOT / "checkpoints/radzero/hf"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32
    ).eval().cuda()
    image = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    pixels = torch.as_tensor(np.array(processor(image)["pixel_values"]), device="cuda")
    text = tokenizer("There is fibrosis", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.compute_logits(pixels, [text])
        probability = output["logits"].sigmoid()
        similarity = output["similarity_scores"]
    if not torch.isfinite(probability).all() or not torch.isfinite(similarity).all():
        raise FloatingPointError("RadZero produced NaN or Inf")
    print(json.dumps({
        "probability_shape": list(probability.shape),
        "similarity_shape": list(similarity.shape),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }))


if __name__ == "__main__":
    main()
