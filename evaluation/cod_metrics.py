"""COD evaluation helpers following the official CODToolbox convention."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def resize_prediction_to_gt(prediction: Any, ground_truth: Any) -> np.ndarray:
    """Resize a prediction map to the GT spatial size before metric computation.

    The CODToolbox treats the ground-truth mask as the spatial reference. This
    helper applies bicubic interpolation to a grayscale prediction map, which
    matches MATLAB ``imresize`` used by the official evaluator. Binary GT data
    is never resized or modified.
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

    if np.issubdtype(pred.dtype, np.floating):
        image = Image.fromarray(pred.astype(np.float32), mode="F")
        resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=pred.dtype)

    image = Image.fromarray(pred.astype(np.uint8), mode="L")
    resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


__all__ = ["resize_prediction_to_gt"]
