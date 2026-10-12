"""COD evaluation helpers following the CamoFormer evaluation convention."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - optional compatibility backend
    cv2 = None


def resize_prediction_to_gt(prediction: Any, ground_truth: Any) -> np.ndarray:
    """Resize a prediction map to the GT spatial size before metric computation.

    The CamoFormer evaluator treats the ground-truth mask as the spatial
    reference and calls ``cv2.resize`` without an interpolation argument. That
    means OpenCV ``INTER_LINEAR`` (bilinear) interpolation. When OpenCV is
    installed, this helper follows that implementation directly; otherwise it
    uses Pillow's bilinear resampler as a dependency-light fallback. Binary GT
    data is never resized or modified.
    """

    pred = np.asarray(prediction)
    gt = np.asarray(ground_truth)
    if pred.ndim != 2:
        raise ValueError(f"prediction must be a 2-D grayscale map, got shape {pred.shape}")
    if gt.ndim < 2:
        raise ValueError(f"ground_truth must have at least two dimensions, got shape {gt.shape}")

    target_height, target_width = gt.shape[:2]
    if pred.shape == (target_height, target_width):
        return pred.copy()

    if cv2 is not None:
        return cv2.resize(
            pred,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    if np.issubdtype(pred.dtype, np.floating):
        image = Image.fromarray(pred.astype(np.float32), mode="F")
        resized = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=pred.dtype)

    image = Image.fromarray(pred.astype(np.uint8), mode="L")
    resized = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


__all__ = ["resize_prediction_to_gt"]
