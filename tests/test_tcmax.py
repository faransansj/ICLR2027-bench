import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from medical_benchmark.config import ROOT, load_yaml
from medical_benchmark.datasets.chexchonet_phase2 import (
    ChexchonetTabularDataset, build_chexchonet_phase2_datasets, build_chexchonet_phase2_manifest,
)
from medical_benchmark.runners import train_tcmax as runner
from medical_benchmark.runners.validate_run import validate_run


class TCMaxTest(unittest.TestCase):
    def test_manifest_failed_write_preserves_previous_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            runner._write_manifest(path, {"status": "RUNNING"})
            with patch.object(Path, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    runner._write_manifest(path, {"status": "COMPLETED"})
            self.assertEqual(json.loads(path.read_text()), {"status": "RUNNING"})
            runner._write_manifest(path, {"status": "COMPLETED"})
            self.assertEqual(json.loads(path.read_text()), {"status": "COMPLETED"})

    def test_scheduler_rejects_invalid_numbers_before_launch(self):
        for option, value in [("--batch-size", "invalid"), ("--batch-size", "0"),
                              ("--num-workers", "1.5"), ("--num-workers", "-1"),
                              ("--max-batches", "0")]:
            with self.subTest(option=option, value=value):
                result = subprocess.run(["bash", str(ROOT / "scripts/run_phase2.sh"),
                                         "--mode", "smoke", "--models", "tcmax",
                                         option, value], capture_output=True, text=True)
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn(option, result.stderr)

    def test_loss_matches_pinned_official_value_and_gradients(self):
        spec = importlib.util.spec_from_file_location("official_losses", ROOT / "third_party/phase2/tcmax/src/utils/losses.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        a, b = torch.randn(3, 4, requires_grad=True), torch.randn(3, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        model = SimpleNamespace(module=SimpleNamespace(fusion_module=SimpleNamespace(forward_a=lambda x: x, forward_v=lambda x: x)))
        expected = official.TCMax_loss(SimpleNamespace(fusion_method="sum"), model, a, b, labels)[0]
        actual = runner.tcmax_loss(a, b, labels)
        torch.testing.assert_close(actual, expected)
        for got, want in zip(torch.autograd.grad(actual, (a, b)), torch.autograd.grad(expected, (a, b))):
            torch.testing.assert_close(got, want)
        extreme = (a.detach() * 10000).requires_grad_()
        loss = runner.tcmax_loss(extreme, b, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(extreme.grad).all())

    def test_joint_mapping_and_marginals(self):
        targets, probabilities = runner.chexchonet_outputs(np.arange(4), np.eye(4))
        np.testing.assert_array_equal(targets, [[0, 0], [1, 0], [0, 1], [1, 1]])
        np.testing.assert_array_equal(probabilities, targets)
        _, p = runner.chexchonet_outputs(np.array([3]), np.array([[.1, .2, .3, .4]]))
        np.testing.assert_allclose(p, [[.6, .7]])
        self.assertEqual(load_yaml("configs/models/phase2/tcmax_chexchonet.yaml")["multilabel_probabilities"], {"SLVH": [1, 3], "DLV": [2, 3]})

    def test_both_real_architectures_cpu_forward_backward_and_metrics(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for task, model, second_key in [("multiclass", runner.TCMaxMilk10k(11), "dermoscopic_image"), ("multilabel", runner.TCMaxChexchonet(), "tabular")]:
                with self.subTest(task=task):
                    first_key = "image" if task == "multilabel" else "clinical_image"
                    samples = [{first_key: torch.randn(3, 32, 32), second_key: torch.randn(2) if task == "multilabel" else torch.randn(3, 32, 32), "target": torch.tensor(i), "sample_id": str(i)} for i in range(2)]
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    metrics, ids, targets, probabilities = runner._run_loader(model, DataLoader(samples, batch_size=2), torch.device("cpu"), "fp32", 1, optimizer, task)
                    self.assertTrue(np.isfinite(metrics["loss"]))
                    self.assertEqual(len(ids), 2)
                    self.assertEqual(probabilities.shape, (2, 2 if task == "multilabel" else 11))
                    self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        finally:
            torch.set_num_threads(threads)

    @staticmethod
    def _write(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_manifest_join_train_only_normalization_and_fail_closed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = [{"sample_id": str(i), "image_path": f"{i}.jpg", "SLVH": str(i % 2), "DLV": str((i // 2) % 2), "fold": str(i)} for i in range(5)]
            metadata = [{"cxr_filename": f"{i}.jpg", "patient_id": f"p{i}", "age": str([100, 90, 20, 30, 40][i]), "sex": "F" if i % 2 == 0 else "M", "slvh": row["SLVH"], "dlv": row["DLV"], "ivsd": "999"} for i, row in enumerate(base)]
            for i in range(5):
                Image.new("RGB", (32, 32)).save(root / f"{i}.jpg")
            self._write(root / "base.csv", base)
            self._write(root / "metadata.csv", metadata)
            output = build_chexchonet_phase2_manifest(root / "metadata.csv", root / "base.csv", root / "paired.csv")
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertNotIn("ivsd", rows[0])
            self.assertEqual([{key: row[key] for key in base[0]} for row in rows], base)
            config = {**load_yaml("configs/datasets/chexchonet_phase2.yaml"), "manifest": str(output), "image_root": str(root)}
            with patch("medical_benchmark.datasets.chexchonet_phase2.load_yaml", return_value=config):
                splits = build_chexchonet_phase2_datasets(0, runner.paired_transforms(32)["clinical"])
            stats = splits["train"].age_stats
            self.assertEqual(stats["mean"], 30)
            self.assertAlmostEqual(stats["std"], np.std([20, 30, 40]))
            self.assertIs(splits["test"].age_stats, stats)
            self.assertEqual([row["fold"] for row in splits["train"].rows], [2, 3, 4])
            self.assertAlmostEqual(splits["test"][0]["tabular"][0].item(), 70 / stats["std"], places=5)
            self.assertEqual(splits["validation"][0]["target"].item(), 1)
            for field, value, message in [("age", "", "age"), ("age", "nan", "finite"), ("sex", "", "sex"), ("sex", "unknown", "sex"), ("patient_id", "p1", "multiple folds")]:
                broken = [dict(row) for row in rows]
                broken[0][field] = value
                self._write(output, broken)
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                    ChexchonetTabularDataset(config, {2, 3, 4}, None)
            constant = [{**row, "age": "30"} for row in rows]
            self._write(output, constant)
            self.assertEqual(ChexchonetTabularDataset(config, {2, 3, 4}, None).age_stats["std"], 1)
            self._write(root / "metadata.csv", [{**row, "slvh": "1"} for row in metadata])
            with self.assertRaisesRegex(ValueError, "targets disagree"):
                build_chexchonet_phase2_manifest(root / "metadata.csv", root / "base.csv", output)
            self.assertEqual(list(csv.DictReader(output.read_text().splitlines())), constant)

    def test_checkpoint_identity_and_cpu_mocked_interrupt_resume(self):
        # CUDA boundary and tiny model are mocked; this is NOT a GPU smoke test.
        class Tiny(nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.a, self.b = nn.Linear(2, 4), nn.Linear(2, 4)

            def forward(self, a, b):
                a, b = self.a(a), self.b(b)
                return a + b, a, b

        samples = [{"image": torch.tensor([float(i), 1.]), "tabular": torch.tensor([1., float(i)]), "target": torch.tensor(i), "sample_id": str(i)} for i in range(4)]
        cpu_torch = MagicMock(wraps=torch)
        cpu_torch.__version__ = torch.__version__
        cpu_torch.version = torch.version
        cpu_torch.bfloat16 = torch.bfloat16
        cpu_torch.device.side_effect = lambda *args: torch.device("cpu")
        cpu_torch.cuda.is_available.return_value = True
        cpu_torch.cuda.get_device_name.return_value = "MOCK_CPU"
        cpu_torch.cuda.get_rng_state_all.return_value = []
        cpu_torch.cuda.max_memory_allocated.return_value = 0
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "torch", cpu_torch), patch.object(runner, "TCMaxChexchonet", Tiny), patch.object(runner, "sha256_file", return_value="a" * 64), patch.object(runner, "_provenance", return_value={}), patch.object(runner, "build_chexchonet_phase2_datasets") as build:
            class Samples(list):
                age_stats = {"mean": 30., "std": 1.}
            build.return_value = {name: Samples(samples) for name in ("train", "validation", "test")}
            args = argparse.Namespace(dataset="chexchonet", fold=0, runtime="local", output_root=directory, max_epochs=2, max_train_batches=1, max_val_batches=1, seed=42, batch_size=2, num_workers=0)
            run = Path(directory) / "phase2/chexchonet/tcmax/fold_0"
            original = runner._run_loader
            calls = 0
            def interrupted(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated interruption")
                return original(*a, **kw)
            with patch.object(runner, "_run_loader", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "simulated"):
                runner.train(args)
            self.assertEqual(json.loads((run / "run_manifest.json").read_text())["status"], "RUNNING")
            with self.assertRaisesRegex(RuntimeError, "different run identity"):
                runner._checkpoint(run / "last.pt", "wrong")
            self.assertEqual(runner.train(args), 0)
            self.assertEqual(validate_run(run)["status"], "VALID")
            checkpoint = torch.load(run / "last.pt", weights_only=False)
            self.assertEqual([item["epoch"] for item in checkpoint["history"]], [0, 1])
            with patch.object(runner, "_run_loader", side_effect=AssertionError("must skip")):
                self.assertEqual(runner.train(args), 0)
            args.seed = 99
            with self.assertRaisesRegex(RuntimeError, "different run identity"):
                runner.train(args)
            args.seed, args.output_root = 42, str(Path(directory) / "uninterrupted")
            runner.train(args)
            baseline = torch.load(Path(args.output_root) / "phase2/chexchonet/tcmax/fold_0/last.pt", weights_only=False)
            for name, value in checkpoint["model"].items():
                torch.testing.assert_close(value, baseline["model"][name], rtol=0, atol=0)

    def test_scheduler_routes_both_datasets_only_to_requested_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            shutil.copy(ROOT / "scripts/run_phase2.sh", root / "scripts/run_phase2.sh")
            binary = root / "bin"
            binary.mkdir()
            (binary / "uv").write_text('#!/usr/bin/env bash\nprintf "%s %s\\n" "$CUDA_VISIBLE_DEVICES" "$*" >> "$RECEIPT"\n')
            (binary / "uv").chmod(0o755)
            env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"], "RECEIPT": str(root / "receipt")}
            for mode, count in [("smoke", 2), ("full", 10)]:
                result = subprocess.run(["bash", str(root / "scripts/run_phase2.sh"), "--mode", mode, "--models", "tcmax", "--datasets", "milk10k,chexchonet", "--gpus", "5,6", "--num-workers", "0"], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = (root / "receipt").read_text().splitlines()
                self.assertEqual(len(lines), count)
                self.assertTrue(all(line.split()[0] in {"5", "6"} for line in lines))
                self.assertEqual(sum("--dataset milk10k" in line for line in lines), count // 2)
                self.assertTrue(all("--batch-size 8 --num-workers 0" in line for line in lines))
                self.assertEqual(all("--max-train-batches 2 --max-val-batches 2" in line for line in lines), mode == "smoke")
                (root / "receipt").unlink()
            for extra in (["--gpus", "5,5"], ["--datasets", "milk10k,milk10k"], ["--datasets"]):
                result = subprocess.run(["bash", str(root / "scripts/run_phase2.sh"), "--mode", "smoke", "--models", "tcmax", *extra], env=env, text=True, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 64)
                self.assertFalse((root / "receipt").exists())


if __name__ == "__main__":
    unittest.main()
