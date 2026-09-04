import unittest

import torch

from medical_benchmark.runners.train_tcmax import tcmax_loss


class TCMaxTest(unittest.TestCase):
    def test_loss_is_finite_and_differentiable(self) -> None:
        clinical = torch.randn(3, 4, requires_grad=True)
        dermoscopic = torch.randn(3, 4, requires_grad=True)
        loss = tcmax_loss(clinical, dermoscopic, torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(clinical.grad)
        self.assertIsNotNone(dermoscopic.grad)


if __name__ == "__main__":
    unittest.main()
