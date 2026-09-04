import unittest

import torch
from PIL import Image

from medical_benchmark.config import load_yaml
from medical_benchmark.runners.train import model_transforms


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

    def test_chexworld_uses_distinct_official_train_and_eval_transforms(self) -> None:
        transforms = model_transforms("chexworld", 224)
        self.assertEqual(set(transforms), {"train", "validation", "test"})
        self.assertIs(transforms["validation"], transforms["test"])
        self.assertIsNot(transforms["train"], transforms["test"])
        for transform in transforms.values():
            image = transform(Image.new("RGB", (320, 240)))
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertTrue(torch.isfinite(image).all())

    def test_published_checkpoint_hashes_are_locked(self) -> None:
        sources = load_yaml("configs/source_lock.yaml")["sources"]
        self.assertEqual(sources["mambavision"]["checkpoint"]["sha256"], "952a3e486f94bbe863c753a7ecabe282b2e3b8adbb0d98057e047e4f554c2a9b")
        self.assertEqual(sources["transnext"]["checkpoint"]["sha256"], "18dbc88e9c1eb3b3f7801f91f61ef25db56a276a2ee85c4ae3e940c832410d8f")
        self.assertEqual(sources["chexworld"]["checkpoint"]["sha256"], "175d2c75971daea772d1f338a8cc81540db60fa48f55cbbd9006b2b9b3576a9a")
        self.assertEqual(sources["chexworld"]["checkpoint"]["sha256_provenance"], "locally_verified_from_official_drive")
        self.assertEqual(
            [item["sha256"] for item in sources["carzero"]["checkpoints"]],
            [
                "34ac3e67dc937e9371c541e8a734c417e17bd991f7150684560c5d46a706b9bb",
                "d9257c4865898becf3b2ff9ea4bdc4c5ec57da28bc99f4a714812cc589052b33",
            ],
        )

    def test_later_phase_plan_is_explicit_and_fail_closed(self) -> None:
        phases = load_yaml("configs/phases.yaml")["phases"]
        self.assertEqual(phases[2]["datasets"]["milk10k"], ["mm_skin_fs", "tcmax", "dymo", "medrega"])
        self.assertEqual(phases[2]["datasets"]["chexchonet"], ["radzero", "tcmax"])
        self.assertEqual(phases[3]["datasets"], {"milk10k": ["skinm2former"], "chexchonet": ["darc"]})

        sources = load_yaml("configs/source_lock.yaml")["sources"]
        expected_commits = {
            "mm_skin_fs": "f2fde3e6f5f51fa5b9776742f95fec02e535389a",
            "tcmax": "a950d9b4e9cd5447c3a58d78dddb02930bf2d47f",
            "dymo": "4562aeceec1b5dfcd18a84d02d35a0abed9d016a",
            "medrega": "f003c701b0b146e5ca7233ccfd627ee5f9eb3993",
            "radzero": "656ae5f1af3f106e96c95542ce3ee5c0ee8777fc",
        }
        for name, commit in expected_commits.items():
            with self.subTest(source=name):
                self.assertEqual(sources[name]["commit"], commit)
                expected_status = "READY_" if name in {"radzero", "tcmax"} else "BLOCKED_"
                self.assertTrue(sources[name]["status"].startswith(expected_status))
        self.assertEqual(
            sources["radzero"]["checkpoint"]["sha256"],
            "7f8edcce8dbc42db753b0b110ab7eb7d5814825733b796f930973c46ee95c622",
        )
        tcmax = load_yaml("configs/models/phase2/tcmax_chexchonet.yaml")
        self.assertEqual(tcmax["modalities"], ["image", "clinical_tabular"])
        self.assertEqual(tcmax["tabular_features"], ["age", "sex"])
        self.assertEqual(tcmax["multilabel_probabilities"], {"SLVH": [1, 3], "DLV": [2, 3]})
        self.assertEqual(
            load_yaml("configs/models/phase2/radzero.yaml")["prompts"],
            {
                "SLVH": "There is severe left ventricular hypertrophy",
                "DLV": "There is a dilated left ventricle",
            },
        )
        for name in ("skinm2former", "darc"):
            with self.subTest(source=name):
                self.assertIsNone(sources[name]["repository"])
                self.assertTrue(sources[name]["status"].startswith("BLOCKED_NO_AUTHOR_PUBLISHED_"))


if __name__ == "__main__":
    unittest.main()
