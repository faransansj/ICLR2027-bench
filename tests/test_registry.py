import unittest

from medical_benchmark.models import BlockedModelError, validate_model


class RegistryTest(unittest.TestCase):
    def test_xwin_is_structurally_blocked(self) -> None:
        with self.assertRaises(BlockedModelError) as caught:
            validate_model("x_win")
        self.assertEqual(caught.exception.reason, "unavailable_implementation")

    def test_lrfl_is_blocked_without_official_checkpoint(self) -> None:
        with self.assertRaises(BlockedModelError) as caught:
            validate_model("lrfl")
        self.assertEqual(caught.exception.reason, "missing_checkpoint")

    def test_pinned_models_require_their_real_artifacts(self) -> None:
        expected = {
            "mambavision": "missing_checkpoint",
            "transnext": "missing_checkpoint",
            "chexworld": "checkpoint_hash_unconfigured",
            "carzero": "checkpoint_hash_unconfigured",
        }
        for model, reason in expected.items():
            with self.subTest(model=model), self.assertRaises(BlockedModelError) as caught:
                validate_model(model)
            self.assertEqual(caught.exception.reason, reason)


if __name__ == "__main__":
    unittest.main()
