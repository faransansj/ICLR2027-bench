import unittest

import torch
from torch import nn

from medical_benchmark.config import load_yaml
from medical_benchmark.models import BlockedModelError, validate_model
from medical_benchmark.models.registry import _make_transnext_pooling_deterministic, model_config_path


class RegistryTest(unittest.TestCase):
    def test_xwin_is_structurally_blocked(self) -> None:
        with self.assertRaises(BlockedModelError) as caught:
            validate_model("xwin")
        self.assertEqual(caught.exception.reason, "unavailable implementation")

    def test_lrfl_is_blocked_without_official_checkpoint(self) -> None:
        with self.assertRaises(BlockedModelError) as caught:
            validate_model("lrfl")
        self.assertEqual(caught.exception.reason, "missing checkpoint")

    def test_image_backbones_have_chexchonet_heads(self) -> None:
        for model in ("mambavision", "transnext"):
            config = load_yaml(model_config_path(model, "chexchonet"))
            self.assertEqual(config["dataset"], "chexchonet")
            self.assertEqual(config["num_classes"], 2)

    def test_transnext_pooling_has_deterministic_equivalent(self) -> None:
        block = nn.Module()
        block.sr_ratio = 2
        block.pool = nn.AdaptiveAvgPool2d((2, 2))
        model = nn.Sequential(block)
        value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        expected = block.pool(value)
        _make_transnext_pooling_deterministic(model)
        self.assertIsInstance(block.pool, nn.AvgPool2d)
        torch.testing.assert_close(block.pool(value), expected)


if __name__ == "__main__":
    unittest.main()
