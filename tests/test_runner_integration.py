"""Small CPU integrations of the real runner; synthetic data replaces only I/O."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from evaluation.validation import CODValidationMetrics
from tools import evaluate_cod
from tools import run_triprd_ab as runner
from tools.triprd_training import sha256_file


ROOT = Path(__file__).resolve().parents[1]
GT_SHAPE = (40, 48)


def synthetic_images(number):
    rng = np.random.default_rng(100 + number)
    rgb = rng.integers(0, 256, size=(*GT_SHAPE, 3), dtype=np.uint8)
    mask = np.zeros(GT_SHAPE, dtype=np.uint8)
    mask[8 + number % 3:30, 11:35 + number % 3] = 255
    thermal = (rgb[:, :, 0] // 2 + mask // 2).astype(np.uint8)
    depth = np.flipud(thermal).copy()
    return {name: Image.fromarray(array) for name, array in
            (("rgb", rgb), ("mask", mask), ("thermal", thermal), ("depth", depth))}


class MemoryCODIndex:
    def __init__(self, dataset, split="train", count=4):
        self.dataset, self.split = dataset, split
        self.assets = {}
        self.entries = []
        self.closed = False
        for number in range(count):
            key = f"{dataset.lower()}_{number}"
            images = synthetic_images(number)
            image = SimpleNamespace(sample_id=key, location=f"{key}_rgb.png")
            mask = SimpleNamespace(sample_id=key, location=f"{key}.png")
            self.assets[image.location], self.assets[mask.location] = images["rgb"], images["mask"]
            self.entries.append(SimpleNamespace(dataset=dataset, split=split, sample_id=key,
                                                image=image, mask=mask))

    def samples(self, limit=None):
        return self.entries if not limit else self.entries[:limit]

    def read_image(self, asset):
        return self.assets[asset.location].copy()

    def read_pair(self, sample, strict_size=False):
        return self.read_image(sample.image), self.read_image(sample.mask)

    def close(self):
        self.closed = True


class MemoryVDTIndex:
    def __init__(self, count=4):
        self.assets = {}
        self.entries = []
        for number in range(count):
            key = f"vdt_{number}"
            sample = {}
            for modality, image in synthetic_images(number).items():
                asset = SimpleNamespace(sample_id=key, location=f"{key}_{modality}.png")
                self.assets[asset.location] = image
                sample[modality] = asset
            self.entries.append(sample)

    def read_image(self, asset):
        return self.assets[asset.location].copy()


class RunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.temp = tempfile.TemporaryDirectory(prefix=".integration-test-", dir=ROOT)
        cls.work = Path(cls.temp.name).resolve()
        assert cls.work.parent == ROOT
        cls.addClassCleanup(cls.temp.cleanup)
        cls.vdt_index = MemoryVDTIndex()
        cls.vdt_items = [runner.Item(f"VDT/{sample['rgb'].sample_id}", (sample, cls.vdt_index), "vdt")
                         for sample in cls.vdt_index.entries]
        cls.cod_indexes = [MemoryCODIndex(name) for name in ("COD10K", "CAMO")]
        cls.cod_items = [runner.Item(f"{index.dataset}/{sample.sample_id}", (sample, index), "cod")
                         for index in cls.cod_indexes for sample in index.samples()]
        cls.original_sizes = []
        original_update = CODValidationMetrics.update

        def checked_update(metrics, source, prediction, ground_truth):
            if prediction.shape != GT_SHAPE or ground_truth.shape != GT_SHAPE:
                raise AssertionError("Validation must retain the original 40x48 GT dimensions")
            cls.original_sizes.append((source, prediction.shape, ground_truth.shape))
            return original_update(metrics, source, prediction, ground_truth)

        cls.source = cls.work / "fresh"
        with patch.object(CODValidationMetrics, "update", new=checked_update):
            cls.invoke_training(cls.source, ["--groups", "A,B,C,D"])
        cls.comparison = cls.read_json(cls.source / "comparison.json")

    @classmethod
    def base_args(cls, output):
        return [
            "--out", str(output), "--vdt-root", str(cls.work / "vdt"),
            "--cod-root", str(cls.work / "cod"), "--model-variant", "tiny",
            "--device", "cpu", "--base-channels", "2", "--image-size", "32",
            "--batch-size", "2", "--epochs", "2", "--vdt-epochs", "2",
            "--max-samples", "0", "--max-test-samples", "2", "--val-fraction", "0.25",
            "--train-seed", "11", "--split-seed", "7", "--validation-metrics",
            "--val-metrics-every", "2", "--selection-metric", "S_alpha",
            "--cod-encoder-lr-scale", "0.1", "--cod-scheduler", "cosine",
            "--cod-warmup-epochs", "1", "--cod-min-lr-ratio", "0.05",
            "--boundary-kd-type", "prob_mse", "--relation-sampling", "uniform",
            "--lambda-boundary-kd", "0.1", "--test-datasets", "COD10K",
        ]

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def invoke_training(cls, output, extra, *, forbid_vdt=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", [str(runner.__file__)] + cls.base_args(output)
                                              + ["--skip-test"] + extra))
            stack.enter_context(patch.object(runner, "_cod_items", return_value=cls.cod_items))
            if forbid_vdt:
                vdt = stack.enter_context(patch.object(runner, "_vdt_items",
                                                       side_effect=AssertionError("VDT loading must be skipped")))
            else:
                vdt = stack.enter_context(patch.object(runner, "_vdt_items", return_value=cls.vdt_items))
            prediction = stack.enter_context(patch.object(runner, "_predict_test",
                                                           side_effect=AssertionError("Test prediction is forbidden")))
            test_index = stack.enter_context(patch.object(runner, "_index_for_dataset",
                                                          side_effect=AssertionError("Test GT loading is forbidden")))
            stages = stack.enter_context(patch.object(runner, "_fit_stage", wraps=runner._fit_stage))
            stream = stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                runner.main()
            except Exception:
                print(stream.getvalue(), file=sys.stderr)
                raise
            prediction.assert_not_called()
            test_index.assert_not_called()
            if forbid_vdt:
                vdt.assert_not_called()
            return [call.args[0] for call in stages.call_args_list]

    def test_real_abcd_training_selects_original_size_sm_without_test_data(self):
        result = self.comparison
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["test_status"], "skipped")
        self.assertEqual(result["test"], {})
        self.assertEqual(set(result["groups"]), set("ABCD"))
        self.assertEqual(len(result["teacher"]["history"]), 2)
        self.assertEqual(len(self.original_sizes), 8)
        self.assertEqual({row[0] for row in self.original_sizes}, {"COD10K", "CAMO"})
        for group in "ABCD":
            with self.subTest(group=group):
                summary = result["groups"][group]["cod_finetune"]
                first, second = summary["history"]
                self.assertIsNone(first["selection_score"])
                self.assertNotIn("val_official", first)
                self.assertEqual(summary["best_epoch"], 2)
                official = second["val_official"]
                self.assertEqual(official["num_samples"], 2)
                self.assertEqual({name: value["num_samples"] for name, value in official["sources"].items()},
                                 {"COD10K": 1, "CAMO": 1})
                self.assertEqual(summary["best_selection_score"], official["macro"]["S_alpha"])
                checkpoint = torch.load(self.source / group / "cod" / "best.pt",
                                        map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["epoch"], 2)
                self.assertEqual(checkpoint["selection_metric"], "S_alpha")
                self.assertEqual(checkpoint["selection_score"], official["macro"]["S_alpha"])
                self.assertIn("scheduler", checkpoint)
                self.assertTrue((self.source / group / "cod" / "metrics.jsonl").is_file())
                self.assertFalse((self.source / group / "predictions").exists())
        self.assertEqual(result["protocol"]["sample_counts"],
                         {"vdt_train": 3, "vdt_val": 1, "cod_train": 6, "cod_val": 2})
        self.assertEqual(result["protocol"]["boundary_kd_type"], "prob_mse")
        self.assertEqual(result["protocol"]["relation_sampling"], "uniform")

    def test_reuse_skips_teacher_and_vdt_stages_and_preserves_fixed_splits(self):
        original_manifest = self.read_json(self.source / "split_manifest.json")
        teacher = self.source / "teacher" / "best.pt"
        for scope, train_seed in (("full", 21), ("encoder", 22)):
            with self.subTest(transfer_scope=scope):
                output = self.work / f"reuse_{scope}"
                extra = ["--groups", "B,C", "--transfer-scope", scope, "--train-seed", str(train_seed),
                         "--teacher-checkpoint", str(teacher),
                         "--vdt-checkpoint", f"B={self.source / 'B/vdt/best.pt'}",
                         "--vdt-checkpoint", f"C={self.source / 'C/vdt/best.pt'}",
                         "--split-manifest", str(self.source / "split_manifest.json")]
                with patch.object(runner, "_fit_calibration_temperature",
                                  side_effect=AssertionError("Teacher calibration must be skipped")):
                    stages = self.invoke_training(output, extra, forbid_vdt=True)
                self.assertEqual(stages, ["B/cod", "C/cod"])
                comparison = self.read_json(output / "comparison.json")
                manifest = self.read_json(output / "split_manifest.json")
                self.assertEqual(manifest["train_seed"], train_seed)
                self.assertEqual(manifest["split_seed"], 7)
                self.assertTrue(manifest["vdt_stage_skipped"])
                for key in ("cod_train", "cod_val"):
                    self.assertEqual(manifest[key], original_manifest[key])
                self.assertNotIn("teacher", comparison)
                for group in ("B", "C"):
                    reused = comparison["groups"][group]["vdt_pretrain"]
                    self.assertTrue(reused["reused"])
                    self.assertEqual(reused["transfer_scope"], scope)
                    self.assertEqual(reused["sha256"], sha256_file(self.source / group / "vdt/best.pt"))
                    self.assertFalse((output / group / "vdt").exists())
                self.assertFalse((output / "teacher").exists())

        output = self.work / "reuse_teacher"
        stages = self.invoke_training(output, ["--groups", "D", "--teacher-checkpoint", str(teacher)])
        self.assertEqual(stages, ["D/vdt", "D/cod"])
        comparison = self.read_json(output / "comparison.json")
        self.assertTrue(comparison["teacher"]["reused"])
        self.assertEqual(comparison["teacher"]["sha256"], sha256_file(teacher))
        self.assertFalse((output / "teacher").exists())

    def test_eval_only_writes_real_predictions_and_metrics_without_training(self):
        output = self.work / "frozen_evaluation"
        checkpoint = self.source / "A/cod/best.pt"
        before = sha256_file(checkpoint)
        test_indexes = []

        def make_index(dataset="COD10K", split="test"):
            index = MemoryCODIndex(dataset, split)
            test_indexes.append(index)
            return index

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", [str(runner.__file__)] + self.base_args(output)
                                              + ["--groups", "A", "--eval-only", "--cod-checkpoint", f"A={checkpoint}"]))
            for name in ("_fit_stage", "_run_epoch", "_fit_calibration_temperature", "_vdt_items", "_cod_items"):
                stack.enter_context(patch.object(runner, name,
                                                 side_effect=AssertionError(f"{name} forbidden in eval-only")))
            stack.enter_context(patch.object(runner, "COD10KIndex", side_effect=lambda root, split: make_index(split=split)))
            stack.enter_context(patch.object(runner, "_index_for_dataset", side_effect=lambda root, dataset: make_index(dataset)))
            stack.enter_context(patch.object(evaluate_cod, "tqdm", side_effect=lambda items, **_: items))
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner.main()
        result = self.read_json(output / "comparison.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["test_status"], "complete")
        official = result["test"]["A"]["COD10K"]["official"]
        self.assertEqual(official["num_samples"], 2)
        self.assertTrue(all(np.isfinite(official[key]) for key in ("S_alpha", "F_beta_w", "E_phi", "MAE")))
        paths = list((output / "A/predictions/COD10K").glob("*.png"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            with Image.open(path) as image:
                self.assertEqual(image.size, (GT_SHAPE[1], GT_SHAPE[0]))
        self.assertEqual(sha256_file(checkpoint), before)
        self.assertTrue(result["groups"]["A"]["cod_finetune"]["reused"])
        self.assertEqual(result["protocol"]["sample_counts"],
                         {"vdt_train": 0, "vdt_val": 0, "cod_train": 0, "cod_val": 0})
        self.assertTrue(test_indexes and all(index.closed for index in test_indexes))
        self.assertFalse((output / "A/cod").exists())


if __name__ == "__main__":
    unittest.main()
