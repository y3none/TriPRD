#!/usr/bin/env python3
"""Run a controlled A/B/C/D feasibility experiment for TriPRD.

The four groups use the same RGB student, fixed deterministic splits, the same
optimizer/epoch budget and the same RGB-only inference path:

``A`` COD RGB supervision; ``B`` VDT RGB supervised pretraining then COD;
``C`` tri-modal teacher + logit KD on VDT then COD; ``D`` full relation,
boundary and uncertainty-gated KD on VDT then COD.  The VDT teacher is trained
once and reused by C/D. COD checkpoints use validation Dice by default, or
original-size source-macro official metrics when requested. Use --skip-test
during screening and --eval-only after the experiment protocol is frozen.

Use ``--model-variant tiny`` for a fast feasibility check or
``--model-variant resnet50 --pretrained --image-size 352`` for the
paper-comparable path.  Both variants use the same A/B/C/D protocol and
official COD evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Install torch, numpy and Pillow before running this script") from exc

from datasets.cod_dataset import CAMOIndex, CHAMELEONIndex, COD10KIndex, NC4KIndex
from datasets.vdt2048 import VDT2048Index
from tools.triprd_training import build_optimizer, build_scheduler, load_checkpoint, sha256_file
try:
    # Dice-only screening can run without OpenCV/scipy. Formal validation
    # checks dependencies before training; failed test evaluation exits nonzero.
    from tools.evaluate_cod import _index_for_dataset, evaluate_dataset
except BaseException:  # evaluate_cod raises SystemExit when optional deps miss
    _index_for_dataset = None
    evaluate_dataset = None


def _load_model_api(variant: str = "tiny") -> tuple[Any, Any, Any, Any]:
    """Resolve the corrected API and tolerate either v2 naming convention."""

    errors: list[str] = []
    modules = ("models.triprd_full",) if variant == "resnet50" else ("models.triprd_v2", "models.triprd")
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=["*"])
            teacher_cls = next(
                getattr(module, name) for name in (("TriPRDTeacherFull", "TriPRDTeacher") if variant == "resnet50" else ("TriPRDTeacherV2", "TriPRDTeacher"))
                if hasattr(module, name)
            )
            student_cls = next(
                getattr(module, name) for name in (("TriPRDStudentFull", "TriPRDStudent") if variant == "resnet50" else ("TriPRDStudentV2", "TriPRDStudent"))
                if hasattr(module, name)
            )
            loss_cls = next(
                getattr(module, name) for name in ("TriPRDLossV2", "TriPRDLoss")
                if hasattr(module, name)
            )
            count_fn = getattr(module, "count_parameters", lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad))
            return teacher_cls, student_cls, loss_cls, count_fn
        except (ImportError, StopIteration, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError("No TriPRD v2 model API found. " + " | ".join(errors))


Teacher, Student, Loss, count_parameters = _load_model_api("tiny")
NORMALIZE_RGB = False
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resize(image: Image.Image, size: int, mask: bool = False) -> Image.Image:
    mode = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    return image.resize((size, size), mode)


def _tensor(image: Image.Image, size: int, *, rgb: bool = False, mask: bool = False) -> torch.Tensor:
    if rgb:
        image = _resize(image.convert("RGB"), size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if NORMALIZE_RGB:
            array = (np.transpose(array, (2, 0, 1)) - RGB_MEAN) / RGB_STD
            return torch.from_numpy(array.copy())
    elif mask:
        image = _resize(image.convert("L"), size, True)
        array = np.asarray(image, dtype=np.float32) / 255.0
    elif image.mode in {"I;16", "I", "F"}:
        image = _resize(image, size)
        array = np.asarray(image, dtype=np.float32)
        array = array / (65535.0 if image.mode == "I;16" else max(float(array.max()), 1.0))
    else:
        image = _resize(image.convert("L"), size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, ...]
    else:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array.copy())


@dataclass(frozen=True)
class Item:
    key: str
    source: Any
    kind: str


class PairDataset(Dataset):
    def __init__(self, items: list[Item], size: int, augment: bool = False) -> None:
        self.items, self.size, self.augment = items, size, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.items[index]
        if item.kind == "vdt":
            sample, data_index = item.source
            images = {key: data_index.read_image(sample[key]) for key in ("rgb", "thermal", "depth", "mask")}
            if self.augment and random.random() < 0.5:
                images = {key: value.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for key, value in images.items()}
            return {
                "key": item.key,
                "rgb": _tensor(images["rgb"], self.size, rgb=True),
                "thermal": _tensor(images["thermal"], self.size),
                "depth": _tensor(images["depth"], self.size),
                "mask": _tensor(images["mask"], self.size, mask=True),
            }
        sample, data_index = item.source
        image, mask = data_index.read_pair(sample, strict_size=False)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return {"key": item.key, "rgb": _tensor(image, self.size, rgb=True), "mask": _tensor(mask, self.size, mask=True)}


def _split(items: list[Item], val_fraction: float, seed: int) -> tuple[list[Item], list[Item]]:
    if not items:
        raise RuntimeError("Cannot split an empty dataset")
    # Split each source prefix separately.  This prevents the larger COD10K
    # pool from consuming all validation slots and leaving CAMO untested.
    buckets: dict[str, list[Item]] = {}
    for item in items:
        prefix = item.key.split("/", 1)[0]
        buckets.setdefault(prefix, []).append(item)
    val_keys: set[str] = set()
    for prefix, bucket in buckets.items():
        ranked = sorted(bucket, key=lambda x: hashlib.sha256(f"{seed}:{prefix}:{x.key}".encode()).hexdigest())
        n_val = max(1, min(len(bucket) - 1, int(round(len(bucket) * val_fraction)))) if len(bucket) > 1 else 0
        val_keys.update(item.key for item in ranked[:n_val])
    train = [item for item in items if item.key not in val_keys]
    val = [item for item in items if item.key in val_keys]
    return train, val


def _vdt_items(root: str, max_samples: int) -> list[Item]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise RuntimeError(f"VDT root does not exist: {root_path}. Pass --vdt-root with the server's absolute VDT-2048 directory.")
    index = VDT2048Index(root_path, split="train")
    # print(root_path)
    # Filter completeness before applying the smoke-test cap.  Some VDT
    # releases contain auxiliary or partially copied entries near the start
    # of the lexicographic index; slicing first can therefore produce zero
    # items even when complete samples exist later in the dataset.
    samples = [
        sample
        for sample in index.samples(None)
        if all(key in sample for key in ("rgb", "thermal", "depth", "mask"))
    ]
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
    items = [Item(f"VDT/{next(iter(s.values())).sample_id}", (s, index), "vdt") for s in samples]
    if not items:
        raise RuntimeError(f"No complete VDT samples found under {root}")
    return items


def _cod_items(root: str, max_per_dataset: int) -> list[Item]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise RuntimeError(f"COD root does not exist: {root_path}. Pass --cod-root with the server's absolute COD benchmark directory.")
    result: list[Item] = []
    for cls, name in ((COD10KIndex, "COD10K"), (CAMOIndex, "CAMO")):
        index = cls(root_path, "train")
        for sample in index.samples(max_per_dataset or None):
            if sample.image is not None and sample.mask is not None:
                result.append(Item(f"{name}/{sample.sample_id}", (sample, index), "cod"))
    if not result:
        raise RuntimeError(f"No complete COD training samples found under {root}")
    return result


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _construct(cls: Any, base: int, pretrained: bool = False) -> nn.Module:
    candidates = ({"base_channels": base}, {"base": base}, {"width": base}, {})
    for kwargs in tuple(dict(item, pretrained=pretrained) for item in candidates) + candidates:
        try:
            return cls(**kwargs)
        except TypeError:
            continue
    raise TypeError(f"Cannot construct {cls}")


def _criterion(mode: str, calibration_temperature: float = 1.0, **options: Any) -> Any:
    # Fail on unsupported options instead of silently dropping ablation flags.
    return Loss(mode=mode, calibration_temperature=calibration_temperature, **options)


def _loss(criterion: Any, output: Mapping[str, torch.Tensor], mask: torch.Tensor, teacher: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    for args, kwargs in (((output, mask, teacher), {}), ((), {"student": output, "mask": mask, "teacher": teacher}), ((output, mask), {"teacher": teacher})):
        try:
            value = criterion(*args, **kwargs)
            if isinstance(value, Mapping) and "total" in value:
                return dict(value)
            if torch.is_tensor(value):
                return {"total": value}
        except TypeError:
            continue
    raise TypeError("TriPRDLoss call signature is incompatible with the runner")


def _dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.max() > 1:
        target = target / 255.0
    prediction = torch.sigmoid(logits)
    dims = tuple(range(1, prediction.ndim))
    return ((2 * (prediction * target).sum(dims) + 1e-6) / (prediction.sum(dims) + target.sum(dims) + 1e-6)).mean()


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: Any, *, optimizer: Any = None, teacher: nn.Module | None = None, use_modalities: bool = False) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if teacher is not None:
        teacher.eval()
    sums: dict[str, float] = {}
    n_seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = _move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if use_modalities:
                if teacher is None:
                    output = model(batch["rgb"], batch["thermal"], batch["depth"])
                    teacher_output = None
                else:
                    with torch.no_grad():
                        teacher_output = teacher(batch["rgb"], batch["thermal"], batch["depth"])
                    output = model(batch["rgb"])
            else:
                output = model(batch["rgb"])
                teacher_output = None
            losses = _loss(criterion, output, batch["mask"], teacher_output)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite loss; stopping before saving a misleading checkpoint")
            if training:
                losses["total"].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                losses["gradient_norm_before_clip"] = gradient_norm.detach()
                optimizer.step()
            count = int(batch["mask"].shape[0])
            n_seen += count
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count
            dice = _dice(output["logits"], batch["mask"])
            sums["dice"] = sums.get("dice", 0.0) + float(dice.detach().cpu()) * count
    if not n_seen:
        raise RuntimeError("empty loader")
    return {key: value / n_seen for key, value in sums.items()}


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _prediction_image(logits: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    prediction = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return Image.fromarray(np.uint8(np.clip(prediction, 0, 1) * 255)).resize(size, Image.Resampling.BILINEAR)


def _validate_original_size(model: nn.Module, items: list[Item], size: int, device: torch.device) -> dict[str, Any]:
    from evaluation.validation import CODValidationMetrics
    metrics = CODValidationMetrics()
    model.eval()
    with torch.no_grad():
        for item in items:
            if item.kind != "cod":
                raise ValueError("Official COD validation accepts COD training-split items only")
            sample, index = item.source
            image, mask = index.read_pair(sample, strict_size=False)
            rgb = _tensor(image, size, rgb=True).unsqueeze(0).to(device)
            prediction = _prediction_image(model(rgb)["logits"], mask.size)
            metrics.update(item.key.split("/", 1)[0], np.asarray(prediction, dtype=np.uint8),
                           np.asarray(mask.convert("L"), dtype=np.uint8))
    return metrics.compute()


def _fit_stage(name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, criterion: Any, out_dir: Path, epochs: int, lr: float, weight_decay: float, seed: int, teacher: nn.Module | None = None, use_modalities: bool = False, *, encoder_lr_scale: float = 1.0, scheduler_name: str = "none", warmup_epochs: int = 0, min_lr_ratio: float = 0.05, original_val_items: list[Item] | None = None, image_size: int = 352, selection_metric: str = "dice", val_metrics_every: int = 5, checkpoint_metadata: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    seed_everything(seed)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if teacher is not None:
        teacher.to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    if selection_metric not in {"dice", "S_alpha", "F_beta_w", "MAE"}:
        raise ValueError(f"Unknown checkpoint selection metric: {selection_metric}")
    if selection_metric != "dice" and not original_val_items:
        raise ValueError("Official checkpoint selection requires COD validation items")
    if val_metrics_every < 1:
        raise ValueError("val_metrics_every must be positive")
    optimizer = build_optimizer(model, lr, weight_decay, encoder_lr_scale)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs, warmup_epochs, min_lr_ratio)
    stage_dir = out_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / "metrics.jsonl"
    best_score = float("inf") if selection_metric == "MAE" else -float("inf")
    best_epoch = 0
    stage_started = time.time()
    best_path = stage_dir / "best.pt"
    history: list[dict[str, Any]] = []
    print(f"\n=== {name}: {epochs} epochs ===", flush=True)
    print(f"Samples: train={len(train_loader.dataset)}, val={len(val_loader.dataset)} | log: {log_path}", flush=True)
    print(f"Selection: {selection_metric} | Epoch  Loss T/V  SoftDice T/V  Best(epoch)  Time / stage ETA", flush=True)
    for epoch in range(1, epochs + 1):
        started = time.time()
        learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        train = _run_epoch(model, train_loader, device, criterion, optimizer=optimizer, teacher=teacher, use_modalities=use_modalities)
        val = _run_epoch(model, val_loader, device, criterion, optimizer=None, teacher=teacher, use_modalities=use_modalities)
        official = None
        if original_val_items is not None and (epoch % val_metrics_every == 0 or epoch == epochs):
            official = _validate_original_size(model, original_val_items, image_size, device)
        score = float(val["dice"]) if selection_metric == "dice" else (
            float(official["macro"][selection_metric]) if official is not None else None)
        if score is not None and not np.isfinite(score):
            raise RuntimeError(f"Non-finite validation {selection_metric} in {name} epoch {epoch}")
        row = {"stage": name, "epoch": epoch, "seconds": time.time() - started, "train": train,
               "val": val, "learning_rates": learning_rates, "selection_metric": selection_metric, "selection_score": score}
        if official is not None:
            row["val_official"] = official
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if scheduler is not None:
            scheduler.step()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "val_dice": float(val["dice"]), "stage": name, "seed": seed,
                 "selection_metric": selection_metric, "selection_score": score,
                 "protocol": checkpoint_metadata or {}}
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, stage_dir / "last.pt")
        improved = score is not None and (score < best_score if selection_metric == "MAE" else score > best_score)
        if improved:
            best_score = score
            best_epoch = epoch
            torch.save(state, best_path)
        elapsed = time.time() - stage_started
        eta = elapsed / epoch * (epochs - epoch)
        print(
            f"{epoch:03d}/{epochs:03d}  {train['total']:.4f}/{val['total']:.4f}  "
            f"{train['dice']:.4f}/{val['dice']:.4f}  "
            f"{best_score:.4f}({best_epoch:03d})  "
            f"{time.time() - started:.0f}s / {eta / 60:.1f}m{' *' if improved else ''}",
            flush=True,
        )
        if official is not None:
            print("  validation source-macro: " + " | ".join(f"{key}={value:.6f}" for key, value in official["macro"].items()), flush=True)
    _save_json(stage_dir / "history.json", history)
    print(f"[{name}] complete | best {selection_metric} {best_score:.6f} at epoch {best_epoch} | checkpoint {best_path}", flush=True)
    return best_path, {"best_val_dice": max(row["val"]["dice"] for row in history),
                       "selected_val_dice": history[best_epoch - 1]["val"]["dice"],
                       "selection_metric": selection_metric, "best_selection_score": best_score,
                       "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       "best_epoch": best_epoch, "epochs": epochs, "history": history}


def _load(model: nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    return load_checkpoint(model, checkpoint, device)


def _fit_calibration_temperature(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Select one temperature on VDT validation without touching COD test data."""

    candidates = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    values = {value: 0.0 for value in candidates}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            output = model(batch["rgb"], batch["thermal"], batch["depth"])
            target = batch["mask"].float().clamp(0.0, 1.0).expand(-1, 3, -1, -1)
            n = int(target.shape[0])
            count += n
            for temperature in candidates:
                values[temperature] += float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        output["modality_logits"] / temperature, target
                    ).detach().cpu()
                ) * n
    if not count:
        raise RuntimeError("cannot calibrate teacher without validation samples")
    return min(candidates, key=lambda value: values[value] / count)


def _predict_test(model: nn.Module, root: str, dataset_name: str, out_dir: Path, size: int, device: torch.device, limit: int) -> dict[str, float]:
    classes = {"COD10K": COD10KIndex, "CAMO": CAMOIndex, "CHAMELEON": CHAMELEONIndex, "NC4K": NC4KIndex}
    index = classes[dataset_name](root, "test")
    samples = list(index.samples(limit or None))
    pred_dir = out_dir / "predictions" / dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dices: list[float] = []
    maes: list[float] = []
    with torch.no_grad():
        for sample in samples:
            if sample.image is None or sample.mask is None:
                continue
            image, mask = index.read_pair(sample, strict_size=False)
            rgb = _tensor(image, size, rgb=True).unsqueeze(0).to(device)
            gt = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
            pred_img = _prediction_image(model(rgb)["logits"], mask.size)
            pred = np.asarray(pred_img, dtype=np.float32) / 255.0
            gt_bin = (gt > 0.5).astype(np.float32)
            dices.append(float((2 * (pred * gt_bin).sum() + 1e-6) / (pred.sum() + gt_bin.sum() + 1e-6)))
            maes.append(float(np.abs(pred - gt).mean()))
            pred_img.save(pred_dir / f"{sample.sample_id}.png")
    index.close()
    return {"dice": float(np.mean(dices)) if dices else float("nan"), "mae": float(np.mean(maes)) if maes else float("nan"), "count": len(dices)}


def _official_metrics(root: str, prediction_root: Path, dataset_name: str, limit: int) -> dict[str, Any]:
    if _index_for_dataset is None or evaluate_dataset is None:
        return {"status": "unavailable", "error": "install scipy/opencv-python/tqdm for official COD metrics"}
    index = None
    try:
        index = _index_for_dataset(Path(root), dataset_name)
        return evaluate_dataset(index=index, prediction_root=prediction_root, dataset=dataset_name,
                               model="TriPRD-AB", boundary=True, boundary_rate=15,
                               max_samples=limit or None)
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
    finally:
        if index is not None:
            index.close()


def _checkpoint_map(values: list[str], allowed: set[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Checkpoint arguments must use GROUP=PATH")
        group, path = value.split("=", 1)
        group = group.upper()
        if group not in allowed or group in result:
            raise ValueError(f"Invalid or duplicate checkpoint group: {group}")
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result[group] = checkpoint
    return result


def _manifest_split(items: list[Item], manifest: dict[str, Any] | None, prefix: str,
                    fraction: float, seed: int) -> tuple[list[Item], list[Item]]:
    if manifest is None:
        return _split(items, fraction, seed)
    train_keys, val_keys = manifest[f"{prefix}_train"], manifest[f"{prefix}_val"]
    if len(train_keys) != len(set(train_keys)) or len(val_keys) != len(set(val_keys)):
        raise ValueError(f"Duplicate {prefix} keys in split manifest")
    if set(train_keys) & set(val_keys):
        raise ValueError(f"Train/validation overlap in {prefix} split manifest")
    by_key = {item.key: item for item in items}
    if len(by_key) != len(items) or set(train_keys) | set(val_keys) != set(by_key):
        raise ValueError(f"Manifest {prefix} sample pool does not match the current data/limits")
    if not train_keys or not val_keys:
        raise ValueError(f"Empty {prefix} training or validation split")
    return [by_key[key] for key in train_keys], [by_key[key] for key in val_keys]


def _checkpoint_provenance(path: Path, state: Mapping[str, Any], scope: str = "full") -> dict[str, Any]:
    return {"checkpoint": str(path.resolve()), "sha256": sha256_file(path),
            "source_stage": state.get("stage"), "source_seed": state.get("seed"),
            "source_epoch": state.get("epoch"), "transfer_scope": scope,
            "source_protocol": state.get("protocol"), "reused": True,
            "source_protocol_origin": state.get("_source_protocol_origin"),
            "preprocessing_verified": state.get("_protocol_verified")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vdt-root", default="datasets/VDT-2048")
    parser.add_argument("--cod-root", default="datasets/COD_benchmark")
    parser.add_argument("--out", default="results/triprd_ab")
    parser.add_argument("--groups", default="A,B,C,D")
    parser.add_argument("--model-variant", choices=("tiny", "resnet50"), default="tiny")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--vdt-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=0, help="Per source dataset; 0 means all")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--test-datasets", default="COD10K,CAMO,CHAMELEON,NC4K")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261012, help="Legacy fallback for both seed types")
    parser.add_argument("--train-seed", type=int, help="Initialization, augmentation and shuffle seed")
    parser.add_argument("--split-seed", type=int, help="Fix this across training seeds")
    parser.add_argument("--split-manifest", type=Path, help="Reuse an exact split instead of regenerating it")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-metrics", action="store_true", help="Report original-size COD validation metrics")
    parser.add_argument("--val-metrics-every", type=int, default=5)
    parser.add_argument("--selection-metric", choices=("dice", "S_alpha", "F_beta_w", "MAE"), default="dice",
                        help="COD checkpoint criterion; official metrics use equal source weights")
    parser.add_argument("--cod-encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--cod-scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--cod-warmup-epochs", type=int, default=0)
    parser.add_argument("--cod-min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--teacher-checkpoint", type=Path, help="Reuse an existing teacher/best.pt")
    parser.add_argument("--vdt-checkpoint", action="append", default=[], metavar="GROUP=PATH",
                        help="Skip that student's VDT stage; expects a matching GROUP/vdt checkpoint")
    parser.add_argument("--transfer-scope", choices=("full", "encoder"), default="full")
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--lambda-relation", type=float, default=1.0)
    parser.add_argument("--lambda-boundary-kd", type=float, default=0.5)
    parser.add_argument("--boundary-kd-type", choices=("logit_mse", "prob_mse"), default="logit_mse")
    parser.add_argument("--relation-sampling", choices=("legacy", "uniform"), default="legacy")
    parser.add_argument("--gate-mode", choices=("disagreement", "none"), default="disagreement")
    parser.add_argument("--skip-test", action="store_true", help="Validation-only screening; never load COD test sets")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate frozen COD checkpoints without training")
    parser.add_argument("--cod-checkpoint", action="append", default=[], metavar="GROUP=PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global Teacher, Student, Loss, count_parameters, NORMALIZE_RGB
    Teacher, Student, Loss, count_parameters = _load_model_api(args.model_variant)
    NORMALIZE_RGB = bool(args.model_variant == "resnet50" and args.pretrained)
    model_protocol = {"model_variant": args.model_variant, "base_channels": args.base_channels,
                      "image_size": args.image_size, "rgb_normalization": "ImageNet" if NORMALIZE_RGB else "[0,1]"}
    train_seed = args.seed if args.train_seed is None else args.train_seed
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if min(args.epochs, args.vdt_epochs, args.batch_size, args.base_channels, args.val_metrics_every) < 1 or args.image_size < 32:
        raise ValueError("Epochs, batch/channels and validation interval must be positive; image size >= 32")
    if args.max_samples < 0 or args.max_test_samples < 0 or args.num_workers != 0:
        raise ValueError("Sample limits must be nonnegative; archive-backed datasets require --num-workers 0")
    if not 0 <= args.cod_warmup_epochs < args.epochs or not 0 <= args.cod_min_lr_ratio <= 1:
        raise ValueError("Require warmup < COD epochs and min LR ratio in [0,1]")
    if args.cod_scheduler == "none" and args.cod_warmup_epochs:
        raise ValueError("Warmup requires --cod-scheduler cosine")
    if not all(np.isfinite(value) and value > 0 for value in (args.learning_rate, args.cod_encoder_lr_scale)):
        raise ValueError("Learning rate and encoder LR scale must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Weight decay must be finite and nonnegative")
    if args.selection_metric != "dice":
        args.validation_metrics = True
    if args.validation_metrics and not args.eval_only:
        from evaluation.validation import CODValidationMetrics  # fail before training if dependencies are absent
    groups = [value.strip().upper() for value in args.groups.split(",") if value.strip()]
    if not groups or set(groups) - set("ABCD") or len(groups) != len(set(groups)):
        raise ValueError("--groups must be unique entries from A,B,C,D")
    test_datasets = [value.strip().upper() for value in args.test_datasets.split(",") if value.strip()]
    if set(test_datasets) - {"COD10K", "CAMO", "CHAMELEON", "NC4K"} or len(test_datasets) != len(set(test_datasets)):
        raise ValueError("Unknown or duplicate test dataset")
    if not args.skip_test and not test_datasets:
        raise ValueError("Use --skip-test to disable test evaluation explicitly")
    reused_vdt = _checkpoint_map(args.vdt_checkpoint, set(groups) - {"A"})
    reused_cod = _checkpoint_map(args.cod_checkpoint, set(groups))
    if args.eval_only and (args.skip_test or set(reused_cod) != set(groups) or reused_vdt):
        raise ValueError("--eval-only requires one --cod-checkpoint per group and no --skip-test/VDT reuse")
    if not args.eval_only and reused_cod:
        raise ValueError("--cod-checkpoint is only for frozen --eval-only runs")
    if args.transfer_scope != "full" and (args.eval_only or not any(group in "BCD" for group in groups)):
        raise ValueError("Encoder-only transfer requires a VDT-pretrained student")
    if args.teacher_checkpoint is not None and not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    loss_options = {"lambda_logit": args.lambda_logit, "lambda_relation": args.lambda_relation,
                    "lambda_boundary_kd": args.lambda_boundary_kd, "boundary_kd_type": args.boundary_kd_type,
                    "relation_sampling": args.relation_sampling, "gate_mode": args.gate_mode}
    _criterion("none", **loss_options)  # validate options before any expensive stage
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    seed_everything(train_seed)
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists() and any(path.name not in {"run.log", "launch.json"} for path in out_dir.iterdir()):
        raise FileExistsError(f"Output directory contains an existing experiment; choose a new --out: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_source = json.loads(args.split_manifest.read_text(encoding="utf-8")) if args.split_manifest else None
    if manifest_source is not None and manifest_source.get("split_seed", manifest_source.get("seed")) != split_seed:
        raise ValueError("--split-seed differs from the supplied split manifest")
    needs_vdt = not args.eval_only and any(group in "BCD" and group not in reused_vdt for group in groups)
    needs_teacher = not args.eval_only and any(group in "CD" and group not in reused_vdt for group in groups)
    vdt_train, vdt_val, cod_train, cod_val = [], [], [], []
    if needs_vdt:
        vdt_train, vdt_val = _manifest_split(_vdt_items(args.vdt_root, args.max_samples), manifest_source, "vdt", args.val_fraction, split_seed)
    if not args.eval_only:
        cod_train, cod_val = _manifest_split(_cod_items(args.cod_root, args.max_samples), manifest_source, "cod", args.val_fraction, split_seed)
        if not cod_train or not cod_val or (needs_vdt and (not vdt_train or not vdt_val)):
            raise ValueError("Training and validation must be nonempty; increase the sample limit")
    manifest = {"seed": split_seed, "split_seed": split_seed, "train_seed": train_seed,
                "val_fraction": args.val_fraction,
                "vdt_train": [item.key for item in vdt_train], "vdt_val": [item.key for item in vdt_val],
                "cod_train": [item.key for item in cod_train], "cod_val": [item.key for item in cod_val],
                "vdt_stage_skipped": not needs_vdt,
                "source_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None}
    _save_json(out_dir / "split_manifest.json", manifest)
    loaders = {}
    for key, items in (("vdt_train", vdt_train), ("vdt_val", vdt_val), ("cod_train", cod_train), ("cod_val", cod_val)):
        if items:
            training = key.endswith("_train")
            loaders[key] = DataLoader(PairDataset(items, args.image_size, augment=training), args.batch_size,
                                      shuffle=training, num_workers=0)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    source_files = ("tools/run_triprd_ab.py", "tools/triprd_training.py", "models/triprd.py",
                    "models/triprd_full.py", "evaluation/validation.py", "evaluation/cod_metrics.py",
                    "evaluation/camoformer_metrics.py", "tools/evaluate_cod.py")
    common = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "seed": train_seed, "train_seed": train_seed, "split_seed": split_seed,
              "rgb_normalization": "ImageNet" if NORMALIZE_RGB else "[0,1]",
              "train_augmentation": "horizontal_flip_p=0.5", "optimizer": "AdamW",
              "validation_source_weights": "equal", "validation_prediction": "uint8_then_PIL_bilinear_to_original_GT",
              "split_manifest_sha256": sha256_file(out_dir / "split_manifest.json"),
              "sample_counts": {key: len(value) for key, value in (("vdt_train", vdt_train), ("vdt_val", vdt_val), ("cod_train", cod_train), ("cod_val", cod_val))},
              "vdt_root": str(Path(args.vdt_root).expanduser().resolve()), "cod_root": str(Path(args.cod_root).expanduser().resolve()),
              "code_commit": commit, "code_dirty": dirty,
              "source_sha256": {name: sha256_file(ROOT / name) for name in source_files},
              "python_version": platform.python_version(), "torch_version": str(torch.__version__),
              "cuda_version": torch.version.cuda, "device": str(device),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    comparison: dict[str, Any] = {"protocol": common, "groups": {}, "test": {},
                                  "started_at": datetime.now(timezone.utc).isoformat(),
                                  "status": "running", "test_status": "skipped" if args.skip_test else "pending"}
    _save_json(out_dir / "comparison.json", comparison)
    print(f"TriPRD | {args.model_variant} | device={device} | train_seed={train_seed} | split_seed={split_seed} | groups={','.join(groups)}", flush=True)
    print(f"Output: {out_dir} | COD checkpoint criterion: {args.selection_metric}", flush=True)
    teacher = None
    calibration_temperature = 1.0
    if needs_teacher:
        seed_everything(train_seed)
        teacher = _construct(Teacher, args.base_channels, args.pretrained if args.teacher_checkpoint is None else False)
        if args.teacher_checkpoint is not None:
            state = load_checkpoint(teacher, args.teacher_checkpoint, device, expected_stage="teacher", expected_protocol=model_protocol)
            teacher_path = args.teacher_checkpoint
            teacher_summary = _checkpoint_provenance(teacher_path, state)
        else:
            teacher_path, teacher_summary = _fit_stage("teacher", teacher, loaders["vdt_train"], loaders["vdt_val"],
                device, _criterion("none"), out_dir, args.vdt_epochs, args.learning_rate, args.weight_decay,
                train_seed, use_modalities=True, checkpoint_metadata=common)
            _load(teacher, teacher_path, device)
        teacher.to(device)
        calibration_temperature = _fit_calibration_temperature(teacher, loaders["vdt_val"], device)
        comparison["teacher"] = {**teacher_summary, "checkpoint": str(teacher_path),
                                 "parameters": count_parameters(teacher), "calibration_temperature": calibration_temperature}
        _save_json(out_dir / "comparison.json", comparison)
    failures = []
    for group in groups:
        seed_everything(train_seed)
        initialize_pretrained = args.pretrained and not args.eval_only and not (group in reused_vdt and args.transfer_scope == "full")
        model = _construct(Student, args.base_channels, initialize_pretrained)
        summary: dict[str, Any] = {"group": group, "parameters": count_parameters(model), "definition": {
            "A": "COD RGB supervised", "B": "VDT RGB supervised pretrain -> COD",
            "C": "VDT logit KD -> COD", "D": "VDT configurable relation/boundary/gated KD -> COD"}[group]}
        if args.eval_only:
            cod_ckpt = reused_cod[group]
            state = load_checkpoint(model, cod_ckpt, device, expected_stage=f"{group}/cod", expected_protocol=model_protocol)
            summary["cod_finetune"] = _checkpoint_provenance(cod_ckpt, state)
        else:
            if group == "A":
                summary["cod_pretrain"] = None
            else:
                if group in reused_vdt:
                    ckpt = reused_vdt[group]
                    state = load_checkpoint(model, ckpt, device, scope=args.transfer_scope, expected_stage=f"{group}/vdt", expected_protocol=model_protocol)
                    pre_summary = _checkpoint_provenance(ckpt, state, args.transfer_scope)
                else:
                    criterion = _criterion({"B": "none", "C": "logit", "D": "full"}[group],
                                           calibration_temperature, **loss_options)
                    ckpt, pre_summary = _fit_stage(f"{group}/vdt", model, loaders["vdt_train"], loaders["vdt_val"],
                        device, criterion, out_dir, args.vdt_epochs, args.learning_rate, args.weight_decay,
                        train_seed, teacher=teacher if group in "CD" else None, use_modalities=group in "CD",
                        checkpoint_metadata=common)
                    if args.transfer_scope == "encoder":
                        seed_everything(train_seed)
                        model = _construct(Student, args.base_channels, args.pretrained)
                    load_checkpoint(model, ckpt, device, scope=args.transfer_scope, expected_stage=f"{group}/vdt")
                    pre_summary.update({"checkpoint": str(ckpt), "sha256": sha256_file(ckpt), "transfer_scope": args.transfer_scope})
                summary["vdt_pretrain"] = pre_summary
            cod_ckpt, cod_summary = _fit_stage(f"{group}/cod", model, loaders["cod_train"], loaders["cod_val"],
                device, _criterion("none"), out_dir, args.epochs, args.learning_rate, args.weight_decay, train_seed,
                encoder_lr_scale=args.cod_encoder_lr_scale, scheduler_name=args.cod_scheduler,
                warmup_epochs=args.cod_warmup_epochs, min_lr_ratio=args.cod_min_lr_ratio,
                original_val_items=cod_val if args.validation_metrics else None, image_size=args.image_size,
                selection_metric=args.selection_metric, val_metrics_every=args.val_metrics_every, checkpoint_metadata=common)
            summary["cod_finetune"] = cod_summary
            _load(model, cod_ckpt, device)
        model.to(device)
        summary["checkpoint"] = str(cod_ckpt)
        comparison["groups"][group] = summary
        _save_json(out_dir / "comparison.json", comparison)
        if not args.skip_test:
            for dataset_name in test_datasets:
                print(f"\n[{group}/test] {dataset_name}: frozen RGB prediction + official metrics", flush=True)
                try:
                    prediction = _predict_test(model, args.cod_root, dataset_name, out_dir / group,
                                               args.image_size, device, args.max_test_samples)
                    official = _official_metrics(args.cod_root, out_dir / group / "predictions", dataset_name, args.max_test_samples)
                    if not all(key in official and np.isfinite(official[key]) for key in ("S_alpha", "F_beta_w", "E_phi", "MAE")):
                        raise RuntimeError(official.get("error", "Missing/non-finite official metrics"))
                    comparison["test"].setdefault(group, {})[dataset_name] = {**prediction, "official": official}
                    print(" | ".join(f"{key}={official[key]:.6f}" for key in ("S_alpha", "F_beta_w", "E_phi", "MAE")), flush=True)
                except Exception as exc:
                    comparison["test"].setdefault(group, {})[dataset_name] = {"status": "unavailable", "error": str(exc)}
                    failures.append(f"{group}/{dataset_name}: {exc}")
                    print(f"ERROR: {failures[-1]}", flush=True)
                _save_json(out_dir / "comparison.json", comparison)
    comparison["finished_at"] = datetime.now(timezone.utc).isoformat()
    comparison["status"] = "failed_evaluation" if failures else "complete"
    comparison["test_status"] = "skipped" if args.skip_test else ("failed" if failures else "complete")
    _save_json(out_dir / "comparison.json", comparison)
    if failures:
        raise RuntimeError("Official evaluation failed: " + "; ".join(failures))
    print(f"\nRun finished. Full histories and results: {out_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()

