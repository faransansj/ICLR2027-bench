import unittest

from medical_benchmark.config import load_yaml
from medical_benchmark.models import BlockedModelError, validate_model
from medical_benchmark.models.registry import model_config_path


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


if __name__ == "__main__":
    unittest.main()
