"""Numerical regression against the independently retrieved CamoFormer evaluator.

Goldens were produced from the official evaltools/metrics.py retrieved via
https://raw.githubusercontent.com/HVision-NKU/CamoFormer/main/evaltools/metrics.py
on 2026-10-19. Its normalized Python AST SHA256 was
ddc2a1e481c03d72e415dd7a77c692578699ad35663397c383f42edbd4ad57ae,
identical to the local vendor AST (comments/formatting differ). No network or
downloaded reference file is required to run these tests.
"""

import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tools import evaluate_cod


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = {
    "S_alpha": 0.4031579242659509,
    "F_beta_w": 0.46605071331222037,
    "E_phi_mean": 0.4391021251149694,
    "E_phi_adaptive": 0.37776459518012756,
    "E_phi_max": 0.6089660974439832,
    "MAE": 0.5056388141889001,
    "FNR": 0.633859050045684,
    "F_beta_adaptive": 0.006541528558389778,
}


def reference_pairs():
    """Non-square, unequal-size, threshold-edge, empty and full GT examples."""
    result = []
    for i, (height, width) in enumerate(((19, 31), (23, 17), (15, 21), (17, 25))):
        y, x = np.indices((height, width))
        gt = np.where((x + 2*y) % 7 < 3, 255, 0).astype(np.uint8)
        if i == 0:
            gt[2:5, 3:8] = 128
            gt[8:11, 5:12] = 129
        if i == 2:
            gt[:] = 0
        if i == 3:
            gt[:] = 255
        ph, pw = (height, width) if i == 0 else (11+i, 13+i)
        py, px = np.indices((ph, pw))
        pred = ((px*37 + py*19 + i*23) % 181 + 37).astype(np.uint8)
        if i == 2:
            pred[:] = 128
        result.append((pred, gt))
    return result


class CamoFormerEvaluationTests(unittest.TestCase):
    def evaluate(self, pairs):
        parent = ROOT / "results" / "local_checks"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="camoformer_", dir=parent) as folder:
            root = Path(folder) / "非ASCII"
            directory = root / "CAMO"
            directory.mkdir(parents=True)
            masks, samples = {}, []
            for i, (prediction, gt) in enumerate(pairs):
                name = f"image_{i}.png"
                ok, encoded = cv2.imencode(".png", prediction)
                self.assertTrue(ok)
                encoded.tofile(directory / name)
                ok, encoded_gt = cv2.imencode(".png", gt)
                self.assertTrue(ok)
                masks[name] = encoded_gt.tobytes()
                samples.append(SimpleNamespace(sample_id=f"image_{i}",
                                               mask=SimpleNamespace(location=name)))
            index = SimpleNamespace(samples=lambda: samples,
                                    source=SimpleNamespace(read_bytes=masks.__getitem__))
            original_files = {path: path.read_bytes() for path in directory.iterdir()}
            with patch.object(evaluate_cod, "tqdm", side_effect=lambda items, **_: items):
                result = evaluate_cod.evaluate_dataset(
                    index=index, prediction_root=root, dataset="CAMO", model="reference",
                    boundary=False, boundary_rate=15, max_samples=None)
            self.assertEqual(original_files, {path: path.read_bytes() for path in directory.iterdir()})
            return result

    def test_all_metrics_match_independent_upstream_goldens(self):
        result = self.evaluate(reference_pairs())
        self.assertEqual(result["num_samples"], 4)
        for name, expected in GOLDEN.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(result[name], expected, places=12)
        self.assertEqual(result["alpha_E"], result["E_phi_adaptive"])
        self.assertEqual(result["E_phi"], result["E_phi_mean"])
        self.assertNotAlmostEqual(result["alpha_E"], result["E_phi"])

    def test_constant_map_preserves_upstream_empty_gt_behavior(self):
        result = self.evaluate([reference_pairs()[2]])
        self.assertAlmostEqual(result["MAE"], 128/255, places=12)
        self.assertEqual(result["F_beta_w"], 0.0)
        # Upstream uses N-1 in E-measure, so this edge case exceeds one.
        self.assertAlmostEqual(result["alpha_E"], 315/314, places=12)
        self.assertAlmostEqual(result["E_phi_mean"], 0.4976736664012741, places=12)

    def test_gt_palette_and_16bit_decoding_matches_opencv(self):
        palette = Image.fromarray(np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)).convert("P")
        palette.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * (768 - 9))
        high_depth = Image.fromarray(np.array([[0, 128, 65535], [256, 32768, 65000]], dtype=np.uint16))
        for image in (palette, high_depth):
            with self.subTest(mode=image.mode):
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                encoded = stream.getvalue()
                expected = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                source_index = SimpleNamespace(source=SimpleNamespace(read_bytes=lambda _: encoded))
                memory_index = SimpleNamespace(read_image=lambda _: image)
                for index in (source_index, memory_index):
                    actual = evaluate_cod._read_ground_truth(index, SimpleNamespace(location="gt.png"))
                    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
