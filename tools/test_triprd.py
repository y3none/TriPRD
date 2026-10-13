#!/usr/bin/env python3
"""Run an RGB-only TriPRD student and export COD prediction PNGs.

Predictions are written as 8-bit grayscale maps under ``<pred-root>/<dataset>``
with the official sample stem.  The output layout is directly consumable by
``tools/evaluate_triprd.py`` (and by the existing ``evaluate_cod.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    from PIL import Image
except ImportError as exc:  # pragma: no cover - depends on experiment environment
    raise SystemExit("TriPRD testing requires torch, numpy and Pillow") from exc

from datasets.cod_dataset import CAMOIndex, CHAMELEONIndex, COD10KIndex, NC4KIndex
from models.triprd import TriPRDStudent


DATASET_FACTORIES = {
    "COD10K": COD10KIndex,
    "CAMO": CAMOIndex,
    "NC4K": NC4KIndex,
    "CHAMELEON": CHAMELEONIndex,
}


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is false")
    return torch.device(value)


def _to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return torch.from_numpy(array.copy())


def _load_student(path: str | Path, device: torch.device, base_channels: int) -> TriPRDStudent:
    model = TriPRDStudent(base_channels=base_channels).to(device)
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _index(root: str | Path, dataset: str):
    return DATASET_FACTORIES[dataset](root, "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="datasets/COD_benchmark")
    parser.add_argument("--pred-root", default="results/predictions/triprd")
    parser.add_argument("--dataset", action="append", choices=tuple(DATASET_FACTORIES), dest="datasets")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all test samples")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary", default="results/triprd_test_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    device = _device(args.device)
    model = _load_student(args.checkpoint, device, args.base_channels)
    datasets = args.datasets or list(DATASET_FACTORIES)
    pred_root = Path(args.pred_root)
    summary: list[dict[str, Any]] = []
    with torch.inference_mode():
        for dataset_name in datasets:
            index = _index(args.data_root, dataset_name)
            samples = list(index.samples(args.max_samples or None))
            if not samples:
                raise RuntimeError(f"{dataset_name} test split is empty or unavailable")
            output_dir = pred_root / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for sample in samples:
                if sample.image is None:
                    raise ValueError(f"{dataset_name}/{sample.sample_id} has no RGB image")
                image = index.read_image(sample.image)
                output = model(_to_tensor(image, args.image_size).to(device))["logits"]
                probability = torch.sigmoid(output)[0, 0].cpu().numpy()
                prediction = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
                prediction = Image.fromarray(prediction, mode="L").resize(image.size, Image.Resampling.BILINEAR)
                prediction.save(output_dir / f"{sample.sample_id}.png")
                count += 1
            summary.append({"dataset": dataset_name, "split": "test", "num_predictions": count, "output": str(output_dir)})
            index.close()
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
