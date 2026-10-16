"""Tests for original-size, source-balanced COD validation; no torch needed."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from evaluation.validation import CODValidationMetrics
from tools import evaluate_cod


METRICS = ("S_alpha", "F_beta_w", "E_phi", "MAE")


def rectangular_mask(height=24, width=32):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 255
    return mask


def official_reference(pairs):
    """Run the existing evaluator with only file I/O replaced by memory data."""
    paths = [Path(f"memory-only/CAMO/{i}.png") for i in range(len(pairs))]
    samples = [SimpleNamespace(sample_id=str(i), mask=SimpleNamespace(location=f"{i}.png"))
               for i in range(len(pairs))]
    index = SimpleNamespace(
        samples=lambda: samples,
        read_image=lambda asset: Image.fromarray(pairs[int(Path(asset.location).stem)][1]),
    )
    with patch.object(evaluate_cod, "_prediction_paths", return_value=paths), \
            patch.object(evaluate_cod, "tqdm", side_effect=lambda items, **_: items), \
            patch.object(evaluate_cod, "_read_prediction",
                         side_effect=lambda path: pairs[int(Path(path).stem)][0].copy()):
        return evaluate_cod.evaluate_dataset(
            index=index, prediction_root=Path("memory-only"), dataset="CAMO",
            model="validation-reference", boundary=False, boundary_rate=15, max_samples=None,
        )


class CODValidationMetricsTests(unittest.TestCase):
    def test_macro_weights_sources_equally_instead_of_pooling_images(self):
        mask = rectangular_mask()
        metrics = CODValidationMetrics()
        for _ in range(3):
            metrics.update("COD10K", mask, mask)
        metrics.update("CAMO", 255 - mask, mask)
        result = metrics.compute()
        self.assertEqual(result["num_samples"], 4)
        self.assertEqual(result["sources"]["COD10K"]["num_samples"], 3)
        self.assertEqual(result["sources"]["CAMO"]["num_samples"], 1)
        self.assertAlmostEqual(result["macro"]["MAE"], 0.5)
        pooled = sum(row["MAE"] * row["num_samples"] for row in result["sources"].values()) / 4
        self.assertAlmostEqual(pooled, 0.25)
        self.assertNotAlmostEqual(result["macro"]["MAE"], pooled)
        for name in METRICS:
            expected = sum(row[name] for row in result["sources"].values()) / 2
            self.assertAlmostEqual(result["macro"][name], expected)

    def test_matches_existing_evaluator_at_each_original_gt_size(self):
        rng = np.random.default_rng(7)
        pairs = [(rng.integers(0, 256, size=(height, width), dtype=np.uint8),
                  rectangular_mask(height, width)) for height, width in ((31, 47), (18, 29))]
        originals = [(prediction.copy(), gt.copy()) for prediction, gt in pairs]
        metrics = CODValidationMetrics()
        for prediction, gt in pairs:
            metrics.update("CAMO", prediction, gt)
        result = metrics.compute()
        reference = official_reference(pairs)
        self.assertEqual(set(result["macro"]), set(METRICS))
        self.assertEqual(result["sources"]["CAMO"]["num_samples"], 2)
        for name in METRICS:
            self.assertEqual(result["sources"]["CAMO"][name], reference[name])
            self.assertEqual(result["macro"][name], reference[name])
        self.assertNotAlmostEqual(result["macro"]["E_phi"], reference["E_phi_adaptive"])
        for (prediction, gt), (original_prediction, original_gt) in zip(pairs, originals):
            np.testing.assert_array_equal(prediction, original_prediction)
            np.testing.assert_array_equal(gt, original_gt)
        self.assertEqual(metrics.compute(), result)

    def test_empty_ground_truth_is_a_valid_sample(self):
        mask = np.zeros((16, 23), dtype=np.uint8)
        metrics = CODValidationMetrics()
        metrics.update("CAMO", mask, mask)
        result = metrics.compute()
        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["macro"]["S_alpha"], 1.0)
        self.assertEqual(result["macro"]["MAE"], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in result["macro"].values()))

    def test_compute_rejects_an_empty_collection(self):
        with self.assertRaisesRegex(ValueError, "without samples"):
            CODValidationMetrics().compute()

    def test_update_rejects_mismatched_or_invalid_arrays(self):
        metrics = CODValidationMetrics()
        mask = rectangular_mask()
        cases = [
            (mask[::2, ::2], mask, ValueError, "original ground_truth shape"),
            (mask[..., None], mask[..., None], ValueError, "2-D"),
            (np.empty((0, 4), dtype=np.uint8), np.empty((0, 4), dtype=np.uint8), ValueError, "empty"),
            (mask.astype(np.float32), mask, TypeError, "uint8"),
            (mask, mask.astype(np.float32), TypeError, "uint8"),
        ]
        for prediction, gt, error, message in cases:
            with self.subTest(shape=prediction.shape, dtype=prediction.dtype):
                with self.assertRaisesRegex(error, message):
                    metrics.update("CAMO", prediction, gt)
        with self.assertRaisesRegex(ValueError, "without samples"):
            metrics.compute()

    def test_update_rejects_an_empty_source(self):
        mask = rectangular_mask()
        for source in ("", " ", None):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    CODValidationMetrics().update(source, mask, mask)

    def test_compute_rejects_nonfinite_official_metrics(self):
        mask = rectangular_mask()
        metrics = CODValidationMetrics()
        metrics.update("CAMO", mask, mask)
        cases = [
            ("Smeasure", {"sm": float("nan")}, "S_alpha"),
            ("WeightedFmeasure", {"wfm": float("inf")}, "F_beta_w"),
            ("Emeasure", {"em": {"curve": np.array([float("nan")])}}, "E_phi"),
            ("MAE", {"mae": float("inf")}, "MAE"),
        ]
        for cls, value, name in cases:
            with self.subTest(metric=name), \
                    patch(f"evaluation.validation.{cls}.get_results", return_value=value):
                with self.assertRaisesRegex(ValueError, f"non-finite validation metric {name}"):
                    metrics.compute()


if __name__ == "__main__":
    unittest.main()
