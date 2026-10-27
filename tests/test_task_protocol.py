"""Benchmark boundaries and original-size inference for the v4 pipeline."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from evaluation.task_protocol import (DATASET_COUNTS, PREDICTION_PROTOCOL,
                                      evaluate_dataset_strict, prediction_image)
from tools import evaluate_triprd_task as exporter


ROOT = Path(__file__).resolve().parents[1]


def bilinear_logit_oracle(array, width, height):
    """Independent scalar half-pixel interpolation, then sigmoid/quantization."""
    source_h, source_w = array.shape
    result = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        sy = max(0., (y + .5) * source_h / height - .5)
        y0 = int(sy)
        y1 = min(y0 + 1, source_h - 1)
        fy = sy - y0
        for x in range(width):
            sx = max(0., (x + .5) * source_w / width - .5)
            x0 = int(sx)
            x1 = min(x0 + 1, source_w - 1)
            fx = sx - x0
            value = (float(array[y0, x0]) * (1-fx) + float(array[y0, x1]) * fx) * (1-fy)
            value += (float(array[y1, x0]) * (1-fx) + float(array[y1, x1]) * fx) * fy
            result[y, x] = round(255 / (1 + math.exp(-value)))
    return result


class PredictionProtocolTests(unittest.TestCase):
    def test_nonsquare_fp16_logits_match_independent_operation_order(self):
        logits = torch.tensor([[[[-7., 1.25, 4.], [2., -3., .5]]]], dtype=torch.float16)
        actual = prediction_image(logits, (7, 5))
        self.assertEqual(actual.mode, "L")
        self.assertEqual(actual.size, (7, 5))
        np.testing.assert_array_equal(np.asarray(actual), bilinear_logit_oracle(logits[0, 0].numpy(), 7, 5))
        # Sigmoid and interpolation do not commute; quantizing first is also different.
        legacy = Image.fromarray((logits.float().sigmoid()[0, 0].numpy() * 255).astype(np.uint8))
        legacy = np.asarray(legacy.resize((7, 5), Image.Resampling.BILINEAR))
        self.assertGreater(np.max(np.abs(np.asarray(actual).astype(int) - legacy.astype(int))), 20)

    def test_constant_logits_round_without_prediction_minmax_normalization(self):
        actual = np.asarray(prediction_image(torch.zeros(1, 1, 2, 3), (5, 4)))
        np.testing.assert_array_equal(actual, np.full((4, 5), 128, dtype=np.uint8))

    def test_invalid_logits_and_sizes_fail(self):
        cases = [(torch.zeros(2, 1, 2, 3), (5, 4)), (torch.zeros(1, 2, 2, 3), (5, 4)),
                 (torch.full((1, 1, 2, 3), float("nan")), (5, 4)),
                 (torch.zeros(1, 1, 2, 3), (0, 4)), (torch.zeros(1, 1, 2, 3), (True, 4))]
        for logits, size in cases:
            with self.subTest(shape=logits.shape, size=size), self.assertRaises(ValueError):
                prediction_image(logits, size)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix=".task-protocol-", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory = self.root / "CAMO"
        self.directory.mkdir()
        self.samples, self.masks = [], {}
        for i in range(2):
            stem = f"fixture_{i}"
            gt = np.zeros((9, 13), dtype=np.uint8)
            gt[2:7, 3+i:9+i] = 255
            self.masks[stem + ".png"] = Image.fromarray(gt)
            self.samples.append(SimpleNamespace(sample_id=stem, mask=SimpleNamespace(location=stem + ".png")))
            Image.fromarray(gt).save(self.directory / (stem + ".png"))
        self.index = SimpleNamespace(samples=lambda: self.samples,
                                     read_image=lambda asset: self.masks[asset.location])

    def score(self, **kwargs):
        return evaluate_dataset_strict(self.index, self.root, "CAMO", "fixture", **kwargs)

    def test_matching_fixture_runs_real_metrics_and_is_marked_limited(self):
        result = self.score(expected_count=2)
        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["MAE"], 0.)
        self.assertEqual(result["coverage"], {"verified_samples": 2, "expected_full_samples": 250,
                                              "limited": True, "exact_id_match": True})
        self.assertIn("E_phi_adaptive", result)
        self.assertIn("E_phi_mean", result)
        self.assertEqual(result["prediction_protocol"], PREDICTION_PROTOCOL)

    def test_official_count_is_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "expected 250 GT"):
            self.score()

    def test_missing_prediction_rejected_before_metrics(self):
        (self.directory / "fixture_1.png").unlink()
        with patch("tools.evaluate_cod.evaluate_dataset") as score:
            with self.assertRaisesRegex(ValueError, "missing=.*fixture_1"):
                self.score(expected_count=2)
            score.assert_not_called()

    def test_extra_prediction_rejected(self):
        self.masks["fixture_0.png"].save(self.directory / "extra.png")
        with self.assertRaisesRegex(ValueError, "extra=.*extra"):
            self.score(expected_count=2)

    def test_duplicate_prediction_stems_across_extensions_rejected(self):
        self.masks["fixture_0.png"].save(self.directory / "fixture_0.jpg")
        with self.assertRaisesRegex(ValueError, "duplicate prediction stems"):
            self.score(expected_count=2)

    def test_duplicate_gt_and_missing_gt_rejected(self):
        self.samples[1] = self.samples[0]
        with self.assertRaisesRegex(ValueError, "duplicate GT"):
            self.score(expected_count=2)
        self.samples[1] = SimpleNamespace(sample_id="fixture_1", mask=None)
        with self.assertRaisesRegex(ValueError, "missing GT"):
            self.score(expected_count=2)

    def test_train_split_and_ambiguous_layout_rejected(self):
        self.index.split = "train"
        with self.assertRaisesRegex(ValueError, "official test split"):
            self.score(expected_count=2)
        self.index.split = "test"
        self.index.split_assertion = "unknown"
        with self.assertRaisesRegex(ValueError, "official test dataset layout"):
            self.score(expected_count=2)


class FinalReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix=".task-export-", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "comparison.json"
        config = {"stage": "student", "group": "T0", "cod_train_mode": "full",
                  "model_variant": "dinov3_vitb16", "epochs": 100, "max_samples": 0,
                  "pretrained": True, "image_size": 384, "base_channels": 128,
                  "train_backbone_blocks": 4, "detail_branch": True, "batch_size": 8,
                  "num_workers": 4, "train_seed": 21, "split_seed": 21, "lr": .0001,
                  "backbone_lr_scale": .1, "weight_decay": .0001, "warmup_epochs": 5,
                  "amp": True, "grad_clip": 1., "augmentation": "crop_flip",
                  "boundary_weight": .2, "aux_weight": .5, "teacher_epochs": 100,
                  "source_weight": .5, "source_final_ratio": .1, "kd_weight": 1.,
                  "feature_weight": 1., "temperature": 2., "val_metrics_every": 5}
        members = [f"COD10K/fixture_{i}" for i in range(3040)] + [f"CAMO/fixture_{i}" for i in range(1000)]
        contract = {"method": exporter.METHOD, "stage": "student", "group": "T0",
                    "prediction_protocol": PREDICTION_PROTOCOL,
                    **{name: config[name] for name in exporter.TRAINING_CONFIG_FIELDS},
                    "cod_train": members, "cod_val": [], "vdt_train": [], "vdt_val": []}
        history = [{"epoch": epoch, "train": {"optimizer_steps": 1}} for epoch in range(1, 101)]
        self.checkpoint = {"method": exporter.METHOD, "stage": "student", "group": "T0",
                           "contract": copy.deepcopy(contract), "contract_sha256": exporter._digest(contract),
                           "config": copy.deepcopy(config), "epoch": 100, "best_epoch": 100,
                           "best_score": 100., "model": {"fixture": torch.zeros(1)}, "history": history}
        self.good = {"method": "triprd_task_v4", "status": "complete", "stage": "student", "formal": True,
                     "group": "T0", "contract_sha256": exporter._digest(contract), "protocol": contract,
                     "prediction_protocol": PREDICTION_PROTOCOL, "selection_metric": "last_epoch",
                     "best_epoch": 100, "completed_epochs": 100, "best_score": 100., "history": copy.deepcopy(history),
                     "sample_counts": {"cod_train": 4040, "cod_val": 0, "vdt_train": 0, "vdt_val": 0},
                     "config": config}
        self.write_checkpoint(self.checkpoint, self.good)
        self.write(self.good)

    def write(self, report):
        self.path.write_text(json.dumps(report), encoding="utf-8")

    def write_checkpoint(self, checkpoint, report):
        path = self.root / "best.pt"
        torch.save(checkpoint, path)
        report["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_complete_full_pool_run_passes(self):
        self.assertEqual(exporter.validate_run_report(self.path), self.good)

    def test_development_and_wrong_provenance_rejected(self):
        mutations = [lambda r: r["config"].update(cod_train_mode="split"),
                     lambda r: r["config"].update(model_variant="tiny"),
                     lambda r: r["config"].update(pretrained=False),
                     lambda r: r["config"].update(max_samples=2),
                     lambda r: r["sample_counts"].update(cod_train=3232, cod_val=808),
                     lambda r: r["sample_counts"].update(cod_val=1),
                     lambda r: r.update(best_epoch=80), lambda r: r.update(formal=False),
                     lambda r: r.pop("formal"), lambda r: r.update(formal=1),
                     lambda r: r.update(method="triprd_correction_v3"), lambda r: r.update(status="running"),
                     lambda r: r.update(stage="probe"), lambda r: r.update(selection_metric="macro_S_alpha"),
                     lambda r: r.update(prediction_protocol="legacy"),
                     lambda r: r.update(checkpoint_sha256="incorrect")]
        for n, mutate in enumerate(mutations):
            with self.subTest(case=n):
                report = copy.deepcopy(self.good)
                mutate(report)
                self.write(report)
                with self.assertRaises(ValueError):
                    exporter.validate_run_report(self.path)

    def test_changed_selected_checkpoint_rejected(self):
        (self.root / "best.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            exporter.validate_run_report(self.path)

    def test_arbitrary_bytes_cannot_be_promoted_by_updating_the_report_hash(self):
        payload = b"not a PyTorch training checkpoint"
        (self.root / "best.pt").write_bytes(payload)
        report = copy.deepcopy(self.good)
        report["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
        self.write(report)
        with self.assertRaisesRegex(ValueError, "weights-only PyTorch"):
            exporter.validate_run_report(self.path)

    def test_checkpoint_stage_epoch_config_and_history_are_bound_to_report(self):
        mutations = [lambda c: c.update(stage="teacher"), lambda c: c.update(group="T4"),
                     lambda c: c.update(epoch=80), lambda c: c.update(best_epoch=80),
                     lambda c: c["config"].update(lr=.1), lambda c: c["config"].update(max_samples=2),
                     lambda c: c["config"].update(stage="teacher"), lambda c: c["history"][0].update(epoch=2),
                     lambda c: c.update(model={}), lambda c: c.update(model={"fixture": "not a weight"}),
                     lambda c: c["contract"].update(cod_train_mode="split")]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                state, report = copy.deepcopy(self.checkpoint), copy.deepcopy(self.good)
                mutate(state)
                self.write_checkpoint(state, report)
                self.write(report)
                with self.assertRaises(ValueError):
                    exporter.validate_run_report(self.path)

    def test_report_cannot_relabel_checkpoint_configuration_or_protocol(self):
        mutations = [lambda r: r["config"].update(image_size=512),
                     lambda r: r["config"].update(detail_branch=False),
                     lambda r: r["config"].update(amp=1),
                     lambda r: r["config"].pop("backbone_lr_scale"),
                     lambda r: r["protocol"].update(epochs=80),
                     lambda r: r.update(group="T4"), lambda r: r.update(completed_epochs=80),
                     lambda r: r["history"][0]["train"].update(optimizer_steps=2),
                     lambda r: r.update(contract_sha256="incorrect")]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                report = copy.deepcopy(self.good)
                mutate(report)
                self.write(report)
                with self.assertRaises(ValueError):
                    exporter.validate_run_report(self.path)

    def test_full_pool_requires_actual_unique_members_not_only_reported_count(self):
        mutations = [lambda c: c["cod_train"].__setitem__(0, c["cod_train"][1]),
                     lambda c: c["cod_train"].__setitem__(0, "CAMO/extra"),
                     lambda c: c["cod_train"].__setitem__(0, "COD10K/"),
                     lambda c: c["cod_train"].__setitem__(0, c["cod_train"][1].upper()),
                     lambda c: c["cod_val"].append("COD10K/heldout"),
                     lambda c: c["vdt_train"].append("VDT/unreported")]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                state, report = copy.deepcopy(self.checkpoint), copy.deepcopy(self.good)
                mutate(state["contract"])
                state["contract_sha256"] = exporter._digest(state["contract"])
                report["protocol"] = copy.deepcopy(state["contract"])
                report["contract_sha256"] = state["contract_sha256"]
                self.write_checkpoint(state, report)
                self.write(report)
                with self.assertRaises(ValueError):
                    exporter.validate_run_report(self.path)

    def result(self):
        return {"method": "triprd_task_v4", "status": "complete", "prediction_protocol": PREDICTION_PROTOCOL,
                "group": "T0", "selected_epoch": 100, "cod_train_samples": 4040,
                "checkpoint_sha256": self.good["checkpoint_sha256"], "test": {
                    name: {"official": {"num_samples": count, "S_alpha": .8, "E_phi_adaptive": .9,
                                         "F_beta_w": .75, "MAE": .03, "E_phi_mean": .85, "E_phi_max": .95},
                           "coverage": {"verified_samples": count, "expected_full_samples": count,
                                        "limited": False, "exact_id_match": True}}
                    for name, count in DATASET_COUNTS.items()}}

    def test_csv_exports_separate_e_variants_in_official_dataset_order(self):
        out = self.root / "export"
        exporter.export_results(self.result(), out)
        with (out / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["dataset"] for row in rows], list(DATASET_COUNTS))
        self.assertEqual(float(rows[0]["E_phi_adaptive"]), .9)
        self.assertEqual(float(rows[0]["E_phi_mean"]), .85)
        self.assertEqual(float(rows[0]["E_phi_max"]), .95)
        self.assertEqual({row["selected_epoch"] for row in rows}, {"100"})

    def test_incomplete_or_nonfinite_results_rejected_before_output(self):
        mutations = [lambda r: r["test"].pop("CAMO"),
                     lambda r: r["test"]["CAMO"]["coverage"].update(limited=True),
                     lambda r: r["test"]["CAMO"]["official"].update(MAE=float("nan"))]
        for n, mutate in enumerate(mutations):
            result = self.result()
            mutate(result)
            out = self.root / f"bad-{n}"
            with self.subTest(case=n), self.assertRaises(ValueError):
                exporter.export_results(result, out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
