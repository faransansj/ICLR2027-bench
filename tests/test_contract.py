import unittest

from medical_benchmark.config import load_yaml


class ContractTest(unittest.TestCase):
    def test_exact_runtime_profiles(self) -> None:
        local = load_yaml("configs/runtime/local.yaml")
        self.assertEqual(
            {key: local[key] for key in ("device", "profile", "debug", "max_epochs", "max_train_batches", "max_val_batches", "fold", "precision")},
            {"device": "cuda", "profile": "local", "debug": True, "max_epochs": 1, "max_train_batches": 20,
             "max_val_batches": 10, "fold": 0, "precision": "fp32"},
        )
        server = load_yaml("configs/runtime/server.yaml")
        self.assertEqual(
            {key: server[key] for key in ("device", "profile", "debug", "max_epochs", "max_train_batches", "max_val_batches", "fold", "precision")},
            {"device": "cuda", "profile": "server", "debug": False, "max_epochs": None, "max_train_batches": None,
             "max_val_batches": None, "fold": None, "precision": "bf16"},
        )

    def test_published_checkpoint_hashes_are_locked(self) -> None:
        sources = load_yaml("configs/source_lock.yaml")["sources"]
        self.assertEqual(sources["mambavision"]["checkpoint"]["sha256"], "952a3e486f94bbe863c753a7ecabe282b2e3b8adbb0d98057e047e4f554c2a9b")
        self.assertEqual(sources["transnext"]["checkpoint"]["sha256"], "18dbc88e9c1eb3b3f7801f91f61ef25db56a276a2ee85c4ae3e940c832410d8f")
        self.assertIsNone(sources["chexworld"]["checkpoint"]["sha256"])
        self.assertTrue(all(item["sha256"] is None for item in sources["carzero"]["checkpoints"]))


if __name__ == "__main__":
    unittest.main()
