import unittest

import torch
from PIL import Image
from torch import nn

from medical_benchmark.models.radzero import RadZeroCheXchoNetAdapter, RadZeroImageTransform


class _Backbone(nn.Module):
    def compute_logits(self, images, encoded_prompts):
        prompts = encoded_prompts[0]["input_ids"].shape[0]
        return {"logits": torch.zeros(images.shape[0], prompts)}


class _Processor:
    def __call__(self, images, return_tensors):
        self.called = (images.size, return_tensors)
        return {"pixel_values": torch.ones(1, 3, 518, 518)}


class RadZeroAdapterTest(unittest.TestCase):
    def test_two_prompts_produce_two_chexchonet_logits(self) -> None:
        encoded = {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        }
        adapter = RadZeroCheXchoNetAdapter(_Backbone(), encoded)
        self.assertEqual(tuple(adapter(torch.ones(3, 3, 518, 518)).shape), (3, 2))

    def test_single_image_keeps_batch_dimension(self) -> None:
        class SqueezingBackbone(nn.Module):
            def compute_logits(self, images, encoded_prompts):
                return {"logits": torch.zeros(2)}

        encoded = {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        }
        adapter = RadZeroCheXchoNetAdapter(SqueezingBackbone(), encoded)
        self.assertEqual(tuple(adapter(torch.ones(1, 3, 518, 518)).shape), (1, 2))

    def test_image_processor_output_drops_singleton_batch(self) -> None:
        processor = _Processor()
        value = RadZeroImageTransform(processor)(Image.new("RGB", (32, 24)))
        self.assertEqual(tuple(value.shape), (3, 518, 518))
        self.assertEqual(processor.called, ((32, 24), "pt"))


if __name__ == "__main__":
    unittest.main()
