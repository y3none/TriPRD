#!/usr/bin/env python3
"""Run a controlled A/B/C/D feasibility experiment for TriPRD.

The four groups use the same RGB student, fixed deterministic splits, the same
optimizer/epoch budget and the same RGB-only inference path:

``A`` COD RGB supervision; ``B`` VDT RGB supervised pretraining then COD;
``C`` tri-modal teacher + logit KD on VDT then COD; ``D`` full relation,
boundary and uncertainty-gated KD on VDT then COD.  The VDT teacher is trained
once and reused by C/D.  A checkpoint is selected by validation Dice, never by
training loss.  Test predictions are written only after all model selection is
finished.

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
import sys
import time
from dataclasses import dataclass
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
try:
    # Keep OpenCV/scipy optional for training.  Official metrics are attempted
    # after prediction export, but a missing evaluator must not block A/B/C/D.
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


def _criterion(mode: str, calibration_temperature: float = 1.0) -> Any:
    for kwargs in (
        {"mode": mode, "calibration_temperature": calibration_temperature},
        {"mode": mode},
        {"kind": mode},
        {},
    ):
        try:
            return Loss(**kwargs)
        except TypeError:
            continue
    raise TypeError("Cannot construct TriPRD loss")


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
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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


def _fit_stage(name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, criterion: Any, out_dir: Path, epochs: int, lr: float, weight_decay: float, seed: int, teacher: nn.Module | None = None, use_modalities: bool = False) -> tuple[Path, dict[str, Any]]:
    seed_everything(seed)
    model.to(device)
    if teacher is not None:
        teacher.to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stage_dir = out_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / "metrics.jsonl"
    best_score = -float("inf")
    best_path = stage_dir / "best.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        train = _run_epoch(model, train_loader, device, criterion, optimizer=optimizer, teacher=teacher, use_modalities=use_modalities)
        val = _run_epoch(model, val_loader, device, criterion, optimizer=None, teacher=teacher, use_modalities=use_modalities)
        row = {"stage": name, "epoch": epoch, "seconds": time.time() - started, "train": train, "val": val}
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        score = float(val["dice"])
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "val_dice": score, "stage": name, "seed": seed}
        torch.save(state, stage_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(state, best_path)
        print(json.dumps(row, ensure_ascii=False))
    _save_json(stage_dir / "history.json", history)
    return best_path, {"best_val_dice": best_score, "epochs": epochs, "history": history}


def _load(model: nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location=device)
    raw = state.get("model", state) if isinstance(state, Mapping) else state
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch {checkpoint}: missing={missing}, unexpected={unexpected}")
    return dict(state) if isinstance(state, Mapping) else {"model": raw}


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
            prediction = torch.sigmoid(model(rgb)["logits"])[0, 0].cpu().numpy()
            gt = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
            pred_img = Image.fromarray(np.uint8(np.clip(prediction, 0, 1) * 255)).resize(mask.size, Image.Resampling.BILINEAR)
            pred = np.asarray(pred_img, dtype=np.float32) / 255.0
            gt_bin = (gt > 0.5).astype(np.float32)
            dices.append(float((2 * (pred * gt_bin).sum() + 1e-6) / (pred.sum() + gt_bin.sum() + 1e-6)))
            maes.append(float(np.abs(pred - gt).mean()))
            pred_img.save(pred_dir / f"{sample.sample_id}.png")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vdt-root", default="datasets/VDT-2048")
    parser.add_argument("--cod-root", default="datasets/COD_benchmark")
    parser.add_argument("--out", default="results/triprd_ab")
    parser.add_argument("--groups", default="A,B,C,D")
    parser.add_argument("--model-variant", choices=("tiny", "resnet50"), default="tiny",
                        help="tiny is a fast feasibility model; resnet50 is the paper-comparable protocol")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False,
                        help="use ImageNet weights for the ResNet-50 variant")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--vdt-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=0, help="per source dataset; 0 means all")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--test-datasets", default="COD10K,CAMO,CHAMELEON,NC4K")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global Teacher, Student, Loss, count_parameters, NORMALIZE_RGB
    Teacher, Student, Loss, count_parameters = _load_model_api(args.model_variant)
    NORMALIZE_RGB = bool(args.model_variant == "resnet50" and args.pretrained)
    if not (0 < args.val_fraction < 1):
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.epochs < 1 or args.vdt_epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.base_channels < 1:
        raise ValueError("epochs, vdt-epochs, batch-size and base-channels must be positive; image-size must be at least 32")
    if args.num_workers != 0:
        raise ValueError("Use --num-workers 0 for archive-backed datasets")
    seed_everything(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [item.strip().upper() for item in args.groups.split(",") if item.strip()]
    if not groups:
        raise ValueError("--groups must contain at least one of A,B,C,D")
    unknown = set(groups) - set("ABCD")
    if unknown:
        raise ValueError(f"Unknown groups: {sorted(unknown)}")
    if len(groups) != len(set(groups)):
        raise ValueError("--groups must not contain duplicates")

    # Construct indices once.  The manifests are identical for all groups.
    vdt_train, vdt_val = _split(_vdt_items(args.vdt_root, args.max_samples), args.val_fraction, args.seed)
    cod_train, cod_val = _split(_cod_items(args.cod_root, args.max_samples), args.val_fraction, args.seed)
    manifest = {"seed": args.seed, "val_fraction": args.val_fraction, "vdt_train": [x.key for x in vdt_train], "vdt_val": [x.key for x in vdt_val], "cod_train": [x.key for x in cod_train], "cod_val": [x.key for x in cod_val]}
    _save_json(out_dir / "split_manifest.json", manifest)
    loaders = {
        "vdt_train": DataLoader(PairDataset(vdt_train, args.image_size, augment=True), args.batch_size, shuffle=True, num_workers=0),
        "vdt_val": DataLoader(PairDataset(vdt_val, args.image_size), args.batch_size, shuffle=False, num_workers=0),
        "cod_train": DataLoader(PairDataset(cod_train, args.image_size, augment=True), args.batch_size, shuffle=True, num_workers=0),
        "cod_val": DataLoader(PairDataset(cod_val, args.image_size), args.batch_size, shuffle=False, num_workers=0),
    }

    common = {"model_variant": args.model_variant, "pretrained": args.pretrained,
              "rgb_normalization": "ImageNet" if NORMALIZE_RGB else "[0,1]",
              "train_augmentation": "horizontal_flip_p=0.5",
              "base_channels": args.base_channels, "image_size": args.image_size,
              "seed": args.seed, "epochs": args.epochs, "vdt_epochs": args.vdt_epochs}
    comparison: dict[str, Any] = {"protocol": common, "groups": {}, "test": {}}
    teacher_path: Path | None = None
    calibration_temperature = 1.0
    test_datasets = [item.strip().upper() for item in args.test_datasets.split(",") if item.strip()]
    supported_test = {"COD10K", "CAMO", "CHAMELEON", "NC4K"}
    unknown_test = set(test_datasets) - supported_test
    if unknown_test:
        raise ValueError(f"Unknown --test-datasets entries: {sorted(unknown_test)}")
    if "C" in groups or "D" in groups:
        teacher = _construct(Teacher, args.base_channels, args.pretrained)
        teacher_parameters = count_parameters(teacher)
        teacher_criterion = _criterion("none")
        teacher_path, teacher_summary = _fit_stage("teacher", teacher, loaders["vdt_train"], loaders["vdt_val"], device, teacher_criterion, out_dir, args.vdt_epochs, args.learning_rate, args.weight_decay, args.seed, use_modalities=True)
        _load(teacher, teacher_path, device)
        calibration_temperature = _fit_calibration_temperature(teacher, loaders["vdt_val"], device)
        comparison["teacher"] = {"checkpoint": str(teacher_path), "parameters": teacher_parameters,
                                  "calibration_temperature": calibration_temperature, **teacher_summary}

    for group in groups:
        seed_everything(args.seed)
        model = _construct(Student, args.base_channels, args.pretrained)
        group_dir = out_dir / group
        summary: dict[str, Any] = {"group": group, "parameters": count_parameters(model), "definition": {"A": "COD RGB supervised", "B": "VDT RGB supervised pretrain -> COD", "C": "VDT logit KD -> COD", "D": "VDT full relation/boundary/uncertainty-gated KD -> COD"}[group]}
        if group == "A":
            summary["cod_pretrain"] = None
        else:
            modes = {"B": "none", "C": "logit", "D": "full"}
            vdt_criterion = _criterion(modes[group], calibration_temperature)
            teacher = None
            if group in {"C", "D"}:
                teacher = _construct(Teacher, args.base_channels, args.pretrained)
                if teacher_path is None:
                    raise RuntimeError("teacher checkpoint is missing")
                _load(teacher, teacher_path, device)
            ckpt, pre_summary = _fit_stage(f"{group}/vdt", model, loaders["vdt_train"], loaders["vdt_val"], device, vdt_criterion, group_dir.parent, args.vdt_epochs, args.learning_rate, args.weight_decay, args.seed, teacher=teacher, use_modalities=group in {"C", "D"})
            _load(model, ckpt, device)
            summary["vdt_pretrain"] = pre_summary
        cod_ckpt, cod_summary = _fit_stage(f"{group}/cod", model, loaders["cod_train"], loaders["cod_val"], device, _criterion("none"), out_dir, args.epochs, args.learning_rate, args.weight_decay, args.seed)
        summary["cod_finetune"] = cod_summary
        summary["checkpoint"] = str(cod_ckpt)
        comparison["groups"][group] = summary
        _load(model, cod_ckpt, device)
        for dataset_name in test_datasets:
            try:
                prediction_summary = _predict_test(model, args.cod_root, dataset_name, out_dir / group, args.image_size, device, args.max_test_samples)
                official = _official_metrics(args.cod_root, out_dir / group / "predictions", dataset_name, args.max_test_samples)
                comparison["test"].setdefault(group, {})[dataset_name] = {**prediction_summary, "official": official}
            except Exception as exc:
                comparison["test"].setdefault(group, {})[dataset_name] = {"status": "unavailable", "error": str(exc)}
        _save_json(out_dir / "comparison.json", comparison)
    _save_json(out_dir / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

