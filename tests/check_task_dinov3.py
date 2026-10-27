"""Optional real DINOv3 ViT-B/16 gradient audit for the task-oriented student.

Run from the repository root, with the project's PyTorch environment:
    python tests/check_task_dinov3.py --dino-repo ./dinov3 --device cpu

This manual check is intentionally outside automatic unittest discovery. It
constructs the full official ViT-B/16 with random weights and uses several
hundred MB of memory; CUDA memory depends on the selected dimensions. It never
downloads weights and does not measure segmentation accuracy or benchmark
training memory. ``--output`` optionally saves its small JSON result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.triprd_task import TaskStudent  # noqa: E402


def _has_gradient(module: torch.nn.Module) -> bool:
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in module.parameters())


def run(args) -> dict:
    if (args.height < 1 or args.width < 1 or args.batch_size < 1 or args.base_channels < 4
            or args.cpu_threads < 1 or not 0 <= args.train_backbone_blocks <= 12):
        raise ValueError("Positive image/batch/thread counts, width >= 4, and 0-12 trainable blocks required")
    repo = args.dino_repo.expanduser().resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"Local official DINOv3 checkout is missing: {repo}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = TaskStudent(
        model_variant="dinov3_vitb16", pretrained=False, dino_repo=repo,
        base_channels=args.base_channels, train_backbone_blocks=args.train_backbone_blocks,
        detail_branch=True,
    ).to(device).train()
    rgb = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    output = model(rgb, task="cod")
    loss = (output["logits"].square().mean() + output["boundary_logits"].square().mean()
            + sum(value.square().mean() for value in output["aux_logits"]))
    loss.backward()
    backbone = model.encoder.backbone
    frozen_count = len(backbone.blocks) - args.train_backbone_blocks
    frozen, tail = list(backbone.blocks)[:frozen_count], list(backbone.blocks)[frozen_count:]
    result = {
        "status": "pending",
        "scope": "official local random-weight model gradient audit; not accuracy or training-memory evidence",
        "model": "dinov3_vitb16",
        "pretrained": False,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "seed": args.seed,
        "dino_repo": str(repo),
        "input_shape": list(rgb.shape),
        "output_shape": list(output["logits"].shape),
        "task_feature_shapes": [list(feature.shape) for feature in output["task_features"]],
        "train_backbone_blocks": args.train_backbone_blocks,
        "frozen_block_count": frozen_count,
        "frozen_blocks_eval": all(not block.training for block in frozen),
        "early_frozen_and_no_gradients": all(not p.requires_grad and p.grad is None
                                              for block in frozen for p in block.parameters()),
        "tail_blocks_trainable": all(p.requires_grad for block in tail for p in block.parameters()),
        "tail_nonzero_gradients": [_has_gradient(block) for block in tail],
        "norm_nonzero_gradient": _has_gradient(backbone.norm),
        "detail_nonzero_gradient": _has_gradient(model.detail),
        "cod_decoder_nonzero_gradient": _has_gradient(model.decoders["cod"]),
        "sod_decoder_no_gradients": all(p.grad is None for p in model.decoders["sod"].parameters()),
        "all_gradients_finite": all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                    for p in model.parameters()),
        "finite_loss": bool(torch.isfinite(loss)),
        "encoder_fully_frozen": not any(p.requires_grad for p in model.encoder.parameters()),
        "cuda_peak_allocated_bytes": None,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    expected_shape = [args.batch_size, 1, args.height, args.width]
    checks = [result["output_shape"] == expected_shape, result["finite_loss"],
              result["early_frozen_and_no_gradients"], result["frozen_blocks_eval"],
              result["tail_blocks_trainable"], all(result["tail_nonzero_gradients"]),
              result["detail_nonzero_gradient"], result["cod_decoder_nonzero_gradient"],
              result["sod_decoder_no_gradients"], result["all_gradients_finite"],
              result["norm_nonzero_gradient"] == (args.train_backbone_blocks > 0),
              result["encoder_fully_frozen"] == (args.train_backbone_blocks == 0)]
    result["status"] = "passed" if all(checks) else "failed"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-backbone-blocks", type=int, default=4)
    parser.add_argument("--height", type=int, default=35)
    parser.add_argument("--width", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(args)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
