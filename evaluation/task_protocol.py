"""Versioned prediction export and complete-coverage COD evaluation for v4.

Legacy experiment helpers are intentionally unchanged. Metrics still come from
the vendored CamoFormer implementation; this module freezes the input to it.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


PREDICTION_PROTOCOL = "triprd_task_v4_fp32_logits_bilinear_align_false_sigmoid_round_uint8_png_v1"
DATASET_COUNTS = {"NC4K": 4121, "COD10K": 2026, "CAMO": 250, "CHAMELEON": 76}
METRIC_FIELDS = ("S_alpha", "E_phi_adaptive", "F_beta_w", "MAE", "E_phi_mean", "E_phi_max")


def prediction_image(logits: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    """Return one grayscale prediction at original ``(width, height)``.

    Interpolate FP32 logits before the sigmoid and quantize exactly once. No
    min/max normalization is applied here: the metric vendor owns that step.
    """
    if logits.ndim != 4 or logits.shape[:2] != (1, 1) or min(logits.shape[-2:]) < 1:
        raise ValueError("Expected logits with shape (1, 1, height, width)")
    if len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in size):
        raise ValueError("size must contain positive integer (width, height)")
    values = logits.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Prediction logits contain non-finite values")
    resized = F.interpolate(values, size=(size[1], size[0]), mode="bilinear", align_corners=False)
    probability = resized.sigmoid()[0, 0].cpu().numpy()
    pixels = np.rint(probability * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def evaluate_dataset_strict(index, prediction_root: str | Path, dataset: str, model: str,
                            *, expected_count: int | None = None) -> dict:
    """Score an exact one-to-one prediction/GT set, without a partial-run flag.

    ``expected_count`` exists for synthetic fixtures only. An override differing
    from the official count always marks the result limited/non-benchmark.
    Production callers must omit it. All metric arithmetic is delegated to the
    unchanged CamoFormer evaluator.
    """
    from tools.evaluate_cod import _prediction_paths, evaluate_dataset

    dataset = dataset.upper()
    if dataset not in DATASET_COUNTS:
        raise ValueError(f"Unsupported COD dataset: {dataset}")
    official_count = DATASET_COUNTS[dataset]
    expected = official_count if expected_count is None else expected_count
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("expected_count must be a positive integer")
    if getattr(index, "dataset", dataset).upper() != dataset or getattr(index, "split", "test") != "test":
        raise ValueError("Index must identify the requested official test split")
    if getattr(index, "layout_error", None) or getattr(index, "split_assertion", "passed") != "passed":
        raise ValueError("Cannot prove the requested official test dataset layout")
    samples = list(index.samples())
    if len(samples) != expected:
        raise ValueError(f"{dataset}: expected {expected} GT samples, found {len(samples)}")
    identifiers = []
    for sample in samples:
        if sample.mask is None:
            raise ValueError(f"{dataset}: missing GT for {sample.sample_id}")
        sample_id = sample.sample_id.casefold()
        mask_id = Path(sample.mask.location).stem.casefold()
        if sample_id != mask_id:
            raise ValueError(f"{dataset}: sample/GT stem mismatch for {sample.sample_id}")
        identifiers.append(sample_id)
    duplicate_gt = [key for key, count in Counter(identifiers).items() if count > 1]
    if duplicate_gt:
        raise ValueError(f"{dataset}: duplicate GT stems: {duplicate_gt[:5]}")
    paths = _prediction_paths(Path(prediction_root), dataset)
    predicted = [path.stem.casefold() for path in paths]
    duplicate_pred = [key for key, count in Counter(predicted).items() if count > 1]
    if duplicate_pred:
        raise ValueError(f"{dataset}: duplicate prediction stems: {duplicate_pred[:5]}")
    missing, extra = set(identifiers) - set(predicted), set(predicted) - set(identifiers)
    if missing or extra:
        raise ValueError(f"{dataset}: prediction coverage mismatch; missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    if any(path.suffix.lower() != ".png" for path in paths):
        raise ValueError("The v4 prediction protocol requires lossless PNG files")
    result = evaluate_dataset(index=index, prediction_root=Path(prediction_root), dataset=dataset,
                              model=model, boundary=False, boundary_rate=15, max_samples=None)
    if result.get("num_samples") != expected:
        raise ValueError(f"{dataset}: evaluator returned an inconsistent sample count")
    for field in METRIC_FIELDS:
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{dataset}: missing/non-finite metric {field}")
    return {**result, "prediction_protocol": PREDICTION_PROTOCOL,
            "coverage": {"verified_samples": expected, "expected_full_samples": official_count,
                         "limited": expected != official_count, "exact_id_match": True}}
