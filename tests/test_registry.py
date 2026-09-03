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


if __name__ == "__main__":
    unittest.main()
