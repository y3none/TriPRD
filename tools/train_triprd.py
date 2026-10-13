#!/usr/bin/env python3
"""Train the small, runnable TriPRD feasibility model.

The script intentionally keeps the data protocol explicit:

* ``teacher`` uses RGB/Thermal/Depth from the VDT-2048 train split;
* ``student`` uses the same VDT pairs and distils detached outputs from a frozen teacher;
* ``cod`` fine-tunes an RGB-only student on COD10K-train and CAMO-train.

This is a feasibility implementation, not the final CVPR training recipe.
It does not download weights and defaults to a CPU-safe single-process loader.
Run from the repository root, for example::

    python tools/train_triprd.py --stage teacher --vdt-root datasets/VDT-2048 \
        --epochs 2 --max-samples 64 --out checkpoints/TripPRD_teacher.pt
    python tools/train_triprd.py --stage student --vdt-root datasets/VDT-2048 \
        --teacher-checkpoint checkpoints/TripPRD_teacher.pt --epochs 2 \
        --max-samples 64 --out checkpoints/TripPRD_student.pt
    python tools/train_triprd.py --stage cod --cod-root datasets/COD_benchmark \
        --student-checkpoint checkpoints/TripPRD_student.pt --epochs 5 \
        --out checkpoints/TripPRD_cod.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import ConcatDataset, DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - depends on experiment environment
    raise SystemExit(
        "TriPRD training requires torch, numpy and Pillow; "
        "install them in the active Python environment."
    ) from exc

from datasets.cod_dataset import CAMOIndex, COD10KIndex
from datasets.vdt2048 import VDT2048Index
from models.triprd import TriPRDLoss, TriPRDStudent, TriPRDTeacher, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resize(image: Image.Image, size: int, *, mask: bool = False) -> Image.Image:
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    return image.resize((size, size), resampling)


def _to_tensor(image: Image.Image, size: int, *, mask: bool = False, rgb: bool = False) -> torch.Tensor:
    """Convert VDT/COD images without destroying 16-bit depth values."""

    if rgb:
        image = image.convert("RGB")
        image = _resize(image, size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    elif mask:
        image = _resize(image.convert("L"), size, mask=True)
        array = np.asarray(image, dtype=np.float32) / 255.0
    elif image.mode in {"I;16", "I", "F"}:
        # VDT depth is stored as I;16.  Converting it to ``L`` would saturate
        # most pixels at 255 and remove the privileged depth signal.
        image = _resize(image, size)
        array = np.asarray(image, dtype=np.float32)
        scale = 65535.0 if image.mode == "I;16" else max(float(array.max()), 1.0)
        array = array / scale
    else:
        image = _resize(image.convert("L"), size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, ...]
    else:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array.copy())


class VDTFeasibilityDataset(Dataset):
    """A lazy VDT reader that returns ``rgb, thermal, depth, mask`` tensors."""

    def __init__(self, root: str | Path, image_size: int, max_samples: int = 0) -> None:
        self.index = VDT2048Index(root, split="train")
        self.samples = list(self.index.samples(max_samples or None))
        if not self.samples:
            report = self.index.validate(max_samples=1)
            raise RuntimeError(f"VDT-2048 train split is empty or unavailable: {report}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.samples[item]
        required = ("rgb", "thermal", "depth", "mask")
        missing = [name for name in required if name not in sample]
        if missing:
            raise ValueError(f"VDT sample {next(iter(sample.values())).sample_id} missing {missing}")
        return {
            "rgb": _to_tensor(self.index.read_image(sample["rgb"]), self.image_size, rgb=True),
            "thermal": _to_tensor(self.index.read_image(sample["thermal"]), self.image_size),
            "depth": _to_tensor(self.index.read_image(sample["depth"]), self.image_size),
            "mask": _to_tensor(self.index.read_image(sample["mask"]), self.image_size, mask=True),
        }


class CODFeasibilityDataset(Dataset):
    """A lazy COD reader returning RGB/mask and original image size."""

    def __init__(self, index: Any, image_size: int, max_samples: int = 0) -> None:
        self.index = index
        self.samples = list(index.samples(max_samples or None))
        if not self.samples:
            raise RuntimeError(f"{index.dataset} train split is empty or unavailable")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.samples[item]
        image, mask = self.index.read_pair(sample, strict_size=False)
        return {
            "rgb": _to_tensor(image, self.image_size, rgb=True),
            "mask": _to_tensor(mask, self.image_size, mask=True),
        }


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is false")
    return torch.device(value)


def _load_state(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}")
    return checkpoint if isinstance(checkpoint, dict) else {"model": state}


def _build_loader(stage: str, args: argparse.Namespace) -> Dataset:
    if stage in {"teacher", "student"}:
        return VDTFeasibilityDataset(args.vdt_root, args.image_size, args.max_samples)
    if not args.cod_root:
        raise ValueError("--cod-root is required for --stage cod")
    cod10k = CODFeasibilityDataset(COD10KIndex(args.cod_root, "train"), args.image_size, args.max_samples)
    camo = CODFeasibilityDataset(CAMOIndex(args.cod_root, "train"), args.image_size, args.max_samples)
    return ConcatDataset([cod10k, camo])


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _train_epoch(
    *, stage: str, model: nn.Module, teacher: nn.Module | None, loader: DataLoader,
    criterion: TriPRDLoss, optimizer: torch.optim.Optimizer, device: torch.device,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if stage == "teacher":
            outputs = model(batch["rgb"], batch["thermal"], batch["depth"])
            teacher_outputs = None
        elif stage == "student":
            if teacher is None:
                raise RuntimeError("student stage needs a frozen teacher")
            with torch.no_grad():
                teacher_outputs = teacher(batch["rgb"], batch["thermal"], batch["depth"])
            # The corrected student is RGB-only even during distillation.  The
            # teacher is used only to form detached loss targets; injecting a
            # teacher relation tensor into the forward path would create a
            # privileged training shortcut that is unavailable at inference.
            outputs = model(batch["rgb"])
        else:
            outputs = model(batch["rgb"])
            teacher_outputs = None
        losses = criterion(outputs, batch["mask"], teacher_outputs)
        losses["total"].backward()
        if getattr(optimizer, "param_groups", None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        n = int(batch["mask"].shape[0])
        count += n
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * n
    if not count:
        raise RuntimeError("training loader produced no batches")
    return {name: value / count for name, value in sums.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=("teacher", "student", "cod"), required=True)
    parser.add_argument("--vdt-root", default="datasets/VDT-2048")
    parser.add_argument("--cod-root", default="datasets/COD_benchmark")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--student-checkpoint", default="", help="initial student checkpoint for COD fine-tuning")
    parser.add_argument("--out", default="checkpoints/triprd_{stage}_last.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128, help="small default keeps feasibility smoke tests fast")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples; per COD dataset in cod stage")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log", default="logs/triprd_train.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("epochs, batch-size must be positive and image-size must be at least 32")
    seed_everything(args.seed)
    device = _device(args.device)
    stage = args.stage
    model: nn.Module
    teacher: nn.Module | None = None
    if stage == "teacher":
        model = TriPRDTeacher(base_channels=args.base_channels)
    else:
        model = TriPRDStudent(base_channels=args.base_channels)
        initial = args.student_checkpoint if stage == "cod" else ""
        if initial:
            _load_state(model, initial, device)
        if stage == "student":
            if not args.teacher_checkpoint:
                raise ValueError("--teacher-checkpoint is required for --stage student")
            teacher = TriPRDTeacher(base_channels=args.base_channels).to(device)
            _load_state(teacher, args.teacher_checkpoint, device)
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
    model.to(device)
    dataset = _build_loader(stage, args)
    if args.num_workers:
        raise ValueError("Use --num-workers 0 for ZIP-backed datasets; nonzero workers are intentionally disabled")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # This legacy single-stage utility has no validation split.  Keep its
    # teacher/student smoke behavior compatible with the corrected model API;
    # the controlled A/B/C/D runner is the entry point for conclusions.
    criterion = TriPRDLoss(mode="none" if stage in {"teacher", "cod"} else "full")
    out_path = Path(args.out.format(stage=stage))
    log_path = Path(args.log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        metrics = _train_epoch(stage=stage, model=model, teacher=teacher, loader=loader, criterion=criterion, optimizer=optimizer, device=device)
        metrics.update(epoch=epoch, seconds=time.time() - started, stage=stage, parameters=count_parameters(model))
        print(json.dumps(metrics, ensure_ascii=False))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "stage": stage, "args": vars(args), "metrics": metrics}
        torch.save(state, out_path)
        if metrics["total"] < best:
            best = metrics["total"]
            torch.save(state, out_path.with_name(out_path.stem + "_best" + out_path.suffix))
    print(f"saved checkpoint: {out_path}")


if __name__ == "__main__":
    main()
