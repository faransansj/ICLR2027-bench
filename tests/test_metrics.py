import unittest

import numpy as np

from medical_benchmark.metrics import compute_metrics


class MetricsTest(unittest.TestCase):
    def test_multiclass_perfect(self) -> None:
        result = compute_metrics("multiclass", np.array([0, 1]), np.array([[0.9, 0.1], [0.2, 0.8]]))
        self.assertEqual(result, {"accuracy": 1.0, "macro_f1": 1.0})

    def test_multilabel_perfect_auc(self) -> None:
        result = compute_metrics("multilabel", np.array([[0, 1], [1, 0]]), np.array([[0.1, 0.9], [0.8, 0.2]]))
        self.assertEqual(result["macro_auroc"], 1.0)
        self.assertEqual(result["exact_match_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
