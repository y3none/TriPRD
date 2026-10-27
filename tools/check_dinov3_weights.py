#!/usr/bin/env python3
"""Strictly load local DINOv3 weights and check four feature maps, without datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-repo", type=Path, default=os.environ.get("DINO_REPO"))
    parser.add_argument("--dino-weights", type=Path, default=os.environ.get("DINO_WEIGHTS"))
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args(argv)
    if args.dino_repo is None or args.dino_weights is None:
        parser.error("Set DINO_REPO/DINO_WEIGHTS or pass --dino-repo/--dino-weights")
    if args.image_size < 16 or args.image_size % 16:
        parser.error("--image-size must be a positive multiple of 16")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from models.triprd_correction import GenericFeatureEncoder

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: install a CUDA PyTorch build, or use --device cpu")
    torch.manual_seed(0)
    encoder = GenericFeatureEncoder("dinov3_vitb16", 128, True,
                                    args.dino_repo, args.dino_weights, True).to(device).eval()
    pixels = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    with torch.inference_mode():
        features = encoder((pixels - mean) / std)
    expected = (1, 768, args.image_size // 16, args.image_size // 16)
    for index, feature in enumerate(features):
        if tuple(feature.shape) != expected or not torch.isfinite(feature).all().item():
            raise RuntimeError(f"Invalid DINOv3 feature map {index}: {tuple(feature.shape)}")
    digest = hashlib.sha256()
    with args.dino_weights.expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = {
        "status": "ok", "model_variant": "dinov3_vitb16", "device": str(device),
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "generic_weights_sha256": digest.hexdigest(),
        "weight_loading": encoder.weight_load_metadata,
        "layer_indices": list(encoder.layer_indices),
        "feature_shapes": [list(feature.shape) for feature in features],
        "finite_features": True, "frozen": not any(p.requires_grad for p in encoder.parameters()),
        "scope": "Local weight loading and random-input forward check; no training or accuracy evaluation",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
