#!/usr/bin/env python3
"""TriPRD v4: task-space transfer with an independently trained RGB COD student.

No old COD/source student checkpoint initializes this version. Utilities reused
from v3 handle data archives, generic weights and atomic epoch checkpoints only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tools import run_triprd_correction as infrastructure
from tools import run_triprd_ab as shared
from tools.triprd_structure_training import observation_tensor, state_digest, SENSOR_PREPROCESSING
from tools.triprd_training import build_optimizer, build_scheduler, sha256_file
from evaluation.task_protocol import PREDICTION_PROTOCOL, prediction_image, evaluate_dataset_strict
from evaluation.validation import CODValidationMetrics
from models.triprd_task import TaskStudent, TaskTeacher, TASK_MODEL_SPEC
from models.triprd_task_loss import supervised_loss, teacher_loss, transfer_loss

METHOD = "triprd_task_v4"
GROUPS = {"T0": "COD only", "T1": "COD + source RGB/GT",
          "T2": "T1 + capacity-matched RGB teacher task KD",
          "T3": "T1 + multimodal teacher ordinary output KD",
          "T4": "T1 + multimodal teacher reliable output and task-feature KD"}
TEST_COUNTS = {"COD10K": 2026, "CAMO": 250, "CHAMELEON": 76, "NC4K": 4121}
CODE_FILES = tuple(dict.fromkeys((
    "models/triprd_task.py", "models/triprd_task_loss.py", "tools/run_triprd_task.py",
    "evaluation/task_protocol.py", "tools/evaluate_triprd_task.py",
    *infrastructure.CODE_FILES)))
_digest, _json = infrastructure._digest, infrastructure._json


def _replace_saved(temporary, destination):
    """Keep atomic writes while tolerating brief Windows file-reader locks."""
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _replace_saved(temporary, path)


def _save_pt(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    _replace_saved(temporary, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("teacher", "student"), default="student")
    p.add_argument("--group", choices=tuple(GROUPS), default="T0")
    p.add_argument("--teacher-mode", choices=("rgb", "multimodal"), default="multimodal")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vdt-root", default="datasets/VDT-2048")
    p.add_argument("--cod-root", default="datasets/COD_benchmark")
    p.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-backbone-blocks", type=int, default=4)
    p.add_argument("--detail-branch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--teacher-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-seed", type=int, default=20261012)
    p.add_argument("--split-seed", type=int, default=20261012)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--val-fraction", type=float, default=.2)
    p.add_argument("--val-metrics-every", type=int, default=5)
    p.add_argument("--cod-train-mode", choices=("split", "full"), default="split")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--augmentation", choices=("flip", "crop_flip"), default="crop_flip")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-scale", type=float, default=.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--source-weight", type=float, default=.5)
    p.add_argument("--source-final-ratio", type=float, default=.1,
                   help="Cosine anneal source supervision/KD together; COD weight stays one")
    p.add_argument("--kd-weight", type=float, default=1.)
    p.add_argument("--feature-weight", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=2.)
    p.add_argument("--boundary-weight", type=float, default=.2)
    p.add_argument("--aux-weight", type=float, default=.5)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="Evaluate this existing full4040 run; no training/source data access")
    p.add_argument("--test-after-training", action="store_true", help="Explicitly test only a full4040 final run")
    return p.parse_args(argv)


def validate_args(a):
    for name in ("epochs", "teacher_epochs", "batch_size", "val_metrics_every"):
        if getattr(a, name) < 1:
            raise ValueError(f"{name} must be positive")
    if a.base_channels < 4 or a.image_size < 32 or a.train_backbone_blocks < 0:
        raise ValueError("base-channels >= 4, image-size >= 32, train-backbone-blocks >= 0 required")
    if a.max_samples < 0 or a.num_workers < 0 or a.warmup_epochs < 0:
        raise ValueError("Sample/worker/warmup counts must be nonnegative")
    for name in ("lr", "backbone_lr_scale", "temperature", "grad_clip"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("source_weight", "kd_weight", "feature_weight", "boundary_weight", "aux_weight", "weight_decay"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < a.val_fraction < 1 or not 0 <= a.source_final_ratio <= 1:
        raise ValueError("Invalid validation fraction/source final ratio")
    if a.model_variant == "tiny" and a.pretrained:
        raise ValueError("tiny is diagnostic only; use --no-pretrained")
    if a.model_variant != "tiny" and not a.pretrained:
        raise ValueError("Real v4 runs require explicit generic pretrained weights")
    if a.cod_train_mode == "full" and a.max_samples:
        raise ValueError("full4040 forbids max-samples")
    if a.test_after_training and (a.stage != "student" or a.cod_train_mode != "full" or a.model_variant == "tiny"):
        raise ValueError("Official tests require a real full4040 student")
    if a.eval_only and (a.resume or a.prepare_only or a.test_after_training):
        raise ValueError("eval-only cannot be combined with resume/prepare/test-after-training")
    if not (a.eval_only or a.prepare_only):
        needs = a.stage == "student" and a.group in {"T2", "T3", "T4"}
        if needs != bool(a.teacher_checkpoint):
            raise ValueError("Exactly student T2/T3/T4 require --teacher-checkpoint")


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "task_run.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Run lock exists: {path}; check the recorded process before recovery") from exc
    with handle:
        json.dump({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare_data(a):
    old = json.loads(a.split_manifest.read_text(encoding="utf-8")) if a.split_manifest else None
    if old and old.get("split_seed", old.get("seed")) != a.split_seed:
        raise ValueError("split seed differs from manifest")
    splits = {name: [] for name in ("vdt_train", "vdt_val", "cod_train", "cod_val")}
    prefixes = []
    if a.prepare_only or a.stage == "teacher" or a.group != "T0":
        prefixes.append(("vdt", shared._vdt_items(a.vdt_root, a.max_samples)))
    if a.prepare_only or a.stage == "student":
        prefixes.append(("cod", shared._cod_items(a.cod_root, a.max_samples)))
    for prefix, items in prefixes:
        keys = [x.key for x in items]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate {prefix} training IDs")
        if not a.max_samples:
            if prefix == "vdt" and len(items) != 1048:
                raise ValueError(f"Expected 1048 official VDT training images, got {len(items)}")
            if prefix == "cod" and Counter(k.split("/", 1)[0] for k in keys) != {"COD10K": 3040, "CAMO": 1000}:
                raise ValueError("Expected COD10K 3040 + CAMO 1000 training images")
        if prefix == "cod" and a.cod_train_mode == "full":
            if old:
                pool = old.get("cod_train", []) + old.get("cod_val", [])
                if len(pool) != len(set(pool)) or set(pool) != set(keys):
                    raise ValueError("Full training membership differs from supplied disjoint development pool")
            train, val = items, []
        else:
            train, val = shared._manifest_split(items, old, prefix, a.val_fraction, a.split_seed)
            if not val:
                raise ValueError(f"Empty {prefix} validation set")
        if not train:
            raise ValueError(f"Empty {prefix} training set")
        splits[prefix + "_train"], splits[prefix + "_val"] = train, val
    manifest = {"method": METHOD, "split_seed": a.split_seed, "cod_train_mode": a.cod_train_mode,
                **{k: [item.key for item in v] for k, v in splits.items()}}
    return splits, manifest


class TaskDataset(infrastructure.CorrectionDataset):
    """Deterministic paired crop before resize; worker-safe archives inherited."""
    def __init__(self, *args, augmentation="crop_flip", **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation = augmentation

    def __getitem__(self, index):
        self._ensure_local_archives()
        item = self.items[index]
        sample, data_index = item.source
        if item.kind == "vdt":
            images = {k: data_index.read_image(sample[k]) for k in ("rgb", "thermal", "depth", "mask")}
        else:
            rgb, mask = data_index.read_pair(sample, strict_size=False)
            images = {"rgb": rgb, "mask": mask}
        if self.use_augmentation:
            seed = int(hashlib.sha256(f"{self.seed}:{self.epoch}:{item.key}:v4".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            if self.augmentation == "crop_flip" and rng.random() < .5:
                rh, rw = rng.uniform(.8, 1., size=2)
                top, left = rng.random(2) * np.asarray([1 - rh, 1 - rw])
                cropped = {}
                for key, image in images.items():
                    w, h = image.size
                    box = (int(left * w), int(top * h), max(int((left + rw) * w), int(left * w) + 1),
                           max(int((top + rh) * h), int(top * h) + 1))
                    cropped[key] = image.crop(box)
                images = cropped
            if rng.random() < .5:
                images = {k: v.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for k, v in images.items()}
        shared.NORMALIZE_RGB = False
        result = {"key": item.key, "rgb": shared._tensor(images["rgb"], self.size, rgb=True),
                  "mask": shared._tensor(images["mask"], self.size, mask=True)}
        if self.normalize:
            result["rgb"] = (result["rgb"] - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        for kind in ("thermal", "depth"):
            if kind in images:
                result[kind], result[kind + "_valid"], _ = observation_tensor(images[kind], self.size, depth=kind == "depth")
        return result


def _loaders(a, splits, device):
    loaders = {}
    # Stable offsets independent of which partitions are present (T0 vs T1).
    for n, key in enumerate(("vdt_train", "vdt_val", "cod_train", "cod_val")):
        if not splits[key]:
            continue
        train = key.endswith("train")
        dataset = TaskDataset(splits[key], a.image_size, normalize=a.pretrained, augment=train,
                              seed=a.train_seed, augmentation=a.augmentation)
        loaders[key] = DataLoader(dataset, batch_size=a.batch_size, shuffle=train, num_workers=a.num_workers,
            generator=torch.Generator().manual_seed(a.train_seed + n * 1009),
            pin_memory=device.type == "cuda", persistent_workers=False)
    return loaders


def model_options(a, *, pretrained=None):
    return dict(model_variant=a.model_variant, base_channels=a.base_channels,
                pretrained=a.pretrained if pretrained is None else pretrained,
                dino_repo=a.dino_repo, dino_weights=a.dino_weights,
                train_backbone_blocks=a.train_backbone_blocks, detail_branch=a.detail_branch)


def dino_source_digest(repo):
    """Use platform-independent paths; inference never needs generic weights."""
    repo = Path(repo).expanduser().resolve() if repo else None
    if repo is None or not (repo / "hubconf.py").is_file():
        raise FileNotFoundError("Evaluation requires the original local DINOv3 source checkout")
    files = [repo / "hubconf.py", *sorted((repo / "dinov3").rglob("*.py"))]
    return _digest({p.relative_to(repo).as_posix(): sha256_file(p) for p in files})


def _contracts(a, manifest, resources):
    common_names = ("model_variant", "base_channels", "image_size", "pretrained", "train_backbone_blocks",
                    "detail_branch", "batch_size", "num_workers", "train_seed", "split_seed", "lr",
                    "backbone_lr_scale", "weight_decay", "warmup_epochs", "amp", "grad_clip", "augmentation",
                    "boundary_weight", "aux_weight", "teacher_epochs")
    common = {"method": METHOD, **{k: getattr(a, k) for k in common_names}, **resources,
              "source_sha256": {p: sha256_file(ROOT / p) for p in CODE_FILES},
              "model_spec": TASK_MODEL_SPEC, "prediction_protocol": PREDICTION_PROTOCOL,
              "sensor_preprocessing": SENSOR_PREPROCESSING}
    teacher_signature = {**common, "vdt_train": manifest["vdt_train"], "vdt_val": manifest["vdt_val"]}
    if a.stage == "teacher":
        contract = {**teacher_signature, "stage": "teacher", "teacher_mode": a.teacher_mode,
                    "selection": "source_validation_segmentation_boundary_loss"}
    else:
        contract = {**common, "stage": "student", "group": a.group,
                    **{k: getattr(a, k) for k in ("epochs", "source_weight", "source_final_ratio", "kd_weight",
                                                "feature_weight", "temperature", "val_metrics_every", "cod_train_mode")},
                    **{k: manifest[k] for k in ("vdt_train", "vdt_val", "cod_train", "cod_val")},
                    "teacher_sha256": sha256_file(a.teacher_checkpoint) if a.teacher_checkpoint else None}
    return teacher_signature, contract


def read_checkpoint(path, contract=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("method") != METHOD or state.get("contract_sha256") != _digest(state.get("contract")):
        raise ValueError("Not a valid task-v4 checkpoint")
    if contract is not None and state["contract"] != contract:
        raise ValueError("Checkpoint contract mismatch; start a new output directory")
    return state


def _load_teacher(a, signature, device):
    if a.stage != "student" or a.group not in {"T2", "T3", "T4"}:
        return None
    expected_mode = "rgb" if a.group == "T2" else "multimodal"
    state = read_checkpoint(a.teacher_checkpoint)
    report_path = a.teacher_checkpoint.parent / "comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (state.get("stage") != "teacher" or state.get("teacher_mode") != expected_mode or
            state.get("teacher_signature") != signature or report.get("status") != "complete" or
            report.get("checkpoint_sha256") != sha256_file(a.teacher_checkpoint) or
            report.get("contract_sha256") != state["contract_sha256"]):
        raise ValueError("Teacher mode, source partition, resources or completed checkpoint mismatch")
    teacher = TaskTeacher(**model_options(a, pretrained=False)).to(device)
    teacher.load_state_dict(state["model"], strict=True)
    teacher.requires_grad_(False).eval()
    return teacher


def source_factor(a, epoch):
    progress = (epoch - 1) / max(a.epochs - 1, 1)
    return a.source_weight * (a.source_final_ratio + (1 - a.source_final_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch):
    model.train()
    if teacher is not None:
        teacher.eval()
    for loader in loaders.values():
        loader.dataset.epoch = epoch
    primary = loaders["vdt_train" if a.stage == "teacher" else "cod_train"]
    source_iter = iter(loaders["vdt_train"]) if a.stage == "student" and a.group != "T0" else None
    totals, count, updates, skipped, source_images, consecutive = {}, 0, 0, 0, 0, 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    norm_sum = 0.
    for batch in primary:
        batch = shared._move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with infrastructure._autocast(a, device):
            if a.stage == "teacher":
                coals = model.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"], input_mode=a.teacher_mode)
                losses = teacher_loss(coals, batch["mask"], boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
                total, logged = losses["total"], losses
            else:
                target = supervised_loss(model(batch["rgb"], task="cod"), batch["mask"],
                    boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
                total, logged = target["total"], {"cod_total": target["total"], "cod_segmentation": target["segmentation"]}
                if source_iter is not None:
                    try:
                        source_batch = next(source_iter)
                    except StopIteration:
                        source_iter = iter(loaders["vdt_train"])
                        source_batch = next(source_iter)
                    source_batch = shared._move(source_batch, device)
                    source_images += len(source_batch["rgb"])
                    prediction = model(source_batch["rgb"], task="sod")
                    source = supervised_loss(prediction, source_batch["mask"], boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
                    kd = total * 0
                    if teacher is not None:
                        with torch.no_grad():
                            coals = teacher.forward_coalitions(source_batch["rgb"], source_batch["thermal"], source_batch["depth"],
                                input_mode="rgb" if a.group == "T2" else "multimodal")
                        transfer = transfer_loss(prediction, coals["RTD"], source_batch["mask"], teacher_rgb=coals["R"],
                            mode="ordinary" if a.group == "T3" else "task", temperature=a.temperature, feature_weight=a.feature_weight)
                        kd = transfer["total"]
                        logged.update({k: v for k, v in transfer.items() if k != "total"})
                    total = total + source_factor(a, epoch) * (source["total"] + a.kd_weight * kd)
                    logged["source_supervised"] = source["total"]
        infrastructure._finite(total, "task v4 total")
        scaler.scale(total).backward()
        norm, overflow = infrastructure._optimizer_step(model, optimizer, scaler, a.grad_clip)
        if overflow is not None:
            skipped += 1
            consecutive += 1
            print(f"AMP skipped update epoch={epoch} details={overflow}", flush=True)
            if consecutive >= infrastructure.MAX_CONSECUTIVE_AMP_SKIPS:
                raise FloatingPointError("Repeated nonfinite AMP updates")
        else:
            updates += 1
            consecutive = 0
            norm_sum += float(norm)
        for key, value in {**logged, "total": total}.items():
            totals[key] = totals.get(key, 0.) + float(value.detach())
        count += 1
    if updates == 0:
        raise FloatingPointError("No optimizer updates in epoch")
    return {**{k: v / count for k, v in totals.items()}, "steps": count, "optimizer_steps": updates,
            "amp_skipped_steps": skipped, "amp_scale": float(scaler.get_scale()), "grad_norm": norm_sum / updates,
            "source_weight": source_factor(a, epoch) if a.stage == "student" else 0.,
            "primary_images": len(primary.dataset), "source_images": source_images, "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


@torch.no_grad()
def _validate_teacher(model, loader, device, a):
    model.eval()
    total, n = 0., 0
    for batch in loader:
        batch = shared._move(batch, device)
        with infrastructure._autocast(a, device):
            coals = model.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"], input_mode=a.teacher_mode)
            losses = teacher_loss(coals, batch["mask"], boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
        infrastructure._finite(losses["total"], "teacher validation")
        total += float(losses["total"]) * len(batch["rgb"])
        n += len(batch["rgb"])
    return {"loss": total / n, "num_samples": n}


@torch.no_grad()
def _validate_cod(model, items, device, a):
    model.eval()
    metrics = CODValidationMetrics()
    shared.NORMALIZE_RGB = a.pretrained
    for item in items:
        sample, index = item.source
        rgb, mask = index.read_pair(sample, strict_size=False)
        rgb = shared._tensor(rgb, a.image_size, rgb=True).unsqueeze(0).to(device)
        with infrastructure._autocast(a, device):
            output = model(rgb, task="cod")
        pred = prediction_image(output["logits"], mask.size)
        metrics.update(item.key.split("/", 1)[0], np.asarray(pred, dtype=np.uint8), np.asarray(mask.convert("L"), dtype=np.uint8))
    return metrics.compute()


def _fit(model, teacher, loaders, splits, a, device, signature, contract, report):
    optimizer = build_optimizer(model, a.lr, a.weight_decay, a.backbone_lr_scale)
    epochs = a.teacher_epochs if a.stage == "teacher" else a.epochs
    scheduler = build_scheduler(optimizer, "cosine", epochs, min(a.warmup_epochs, epochs - 1), .05)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
    start, best, best_epoch, history = 1, None, None, []
    initial = state_digest(model)
    if a.resume and (a.out / "last.pt").exists():
        checkpoint = read_checkpoint(a.out / "last.pt", contract)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        infrastructure._restore_rng(checkpoint["rng"], loaders)
        start, best, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_score"], checkpoint["best_epoch"]
        initial, history = checkpoint["initial_state_sha256"], checkpoint["history"]
        path = a.out / "best.pt"
        saved = read_checkpoint(path, contract)["epoch"] if path.exists() else None
        if saved != best_epoch:
            if checkpoint["epoch"] != best_epoch:
                raise RuntimeError("Selected checkpoint missing; cannot reconstruct from last")
            _save_pt(path, checkpoint)
    report["initial_state_sha256"] = initial
    for epoch in range(start, epochs + 1):
        row = {"epoch": epoch, "lr": [g["lr"] for g in optimizer.param_groups]}
        row["train"] = _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch)
        score = None
        if a.stage == "teacher":
            row["validation"] = _validate_teacher(model, loaders["vdt_val"], device, a)
            score = -row["validation"]["loss"]
        elif a.cod_train_mode == "full":
            score = float(epoch)
        elif epoch % a.val_metrics_every == 0 or epoch == epochs:
            row["validation"] = _validate_cod(model, splits["cod_val"], device, a)
            score = row["validation"]["macro"]["S_alpha"]
        selected = score is not None and (best is None or score > best)
        if selected:
            best, best_epoch = score, epoch
        scheduler.step()
        history.append(row)
        checkpoint = {"method": METHOD, "stage": a.stage, "group": a.group, "teacher_mode": a.teacher_mode,
            "contract": contract, "contract_sha256": _digest(contract), "teacher_signature": signature,
            "config": _json(a), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
            "best_score": best, "best_epoch": best_epoch, "history": history, "initial_state_sha256": initial,
            "rng": infrastructure._rng_state(loaders)}
        _save_pt(a.out / "last.pt", checkpoint)
        if selected:
            _save_pt(a.out / "best.pt", checkpoint)
        report.update(completed_epochs=epoch, best_epoch=best_epoch, best_score=best, history=history)
        _save_json(a.out / "comparison.json", report)
        _save_json(a.out / "history.json", history)
        print(f"[{a.stage}/{a.group}] {epoch}/{epochs} loss={row['train']['total']:.5f} "
              f"best={best_epoch} updates={row['train']['optimizer_steps']} time={row['train']['seconds']:.1f}s", flush=True)
    selected = read_checkpoint(a.out / "best.pt", contract)
    model.load_state_dict(selected["model"], strict=True)
    report.update(completed_epochs=epochs, best_epoch=best_epoch, best_score=best, history=history,
                  checkpoint_sha256=sha256_file(a.out / "best.pt"))


@torch.no_grad()
def _formal_tests(model, a):
    from tools.evaluate_cod import _index_for_dataset
    model.eval()
    device = next(model.parameters()).device
    shared.NORMALIZE_RGB = a.pretrained
    results = {}
    for dataset, count in TEST_COUNTS.items():
        index = _index_for_dataset(Path(a.cod_root), dataset)
        try:
            samples = list(index.samples())
            ids = [s.sample_id.casefold() for s in samples]
            if len(ids) != count or len(set(ids)) != count or any(s.image is None or s.mask is None for s in samples):
                raise ValueError(f"Incomplete official {dataset} test set")
            destination = a.out / "predictions" / dataset
            destination.mkdir(parents=True, exist_ok=True)
            # Reject stale/extra files before writing any predictions.
            existing = [p.stem.casefold() for p in destination.iterdir() if p.is_file()]
            if len(existing) != len(set(existing)) or set(existing) - set(ids):
                raise ValueError("Stale/duplicate prediction directory; use a clean output")
            for sample in samples:
                rgb, mask = index.read_pair(sample, strict_size=False)
                tensor = shared._tensor(rgb, a.image_size, rgb=True).unsqueeze(0).to(device)
                with infrastructure._autocast(a, device):
                    output = model(tensor, task="cod")
                prediction_image(output["logits"], mask.size).save(destination / (sample.sample_id + ".png"))
            metrics = evaluate_dataset_strict(index, a.out / "predictions", dataset, "TriPRD-Task-v4")
            coverage = metrics.pop("coverage")
            results[dataset] = {"official": metrics, "coverage": coverage, "count": count}
            _save_json(a.out / "test_metrics.json", results)
        finally:
            index.close()
    return results


def _evaluate_existing(cli):
    from tools.evaluate_triprd_task import validate_run_report
    report = validate_run_report(cli.out / "comparison.json")
    state = read_checkpoint(cli.out / "best.pt")
    if state["contract"]["source_sha256"] != {p: sha256_file(ROOT / p) for p in CODE_FILES}:
        raise ValueError("Evaluation source differs from trained v4 contract")
    a = argparse.Namespace(**state["config"])
    a.out, a.device, a.cod_root = cli.out, cli.device, cli.cod_root
    a.dino_repo = cli.dino_repo or a.dino_repo
    if dino_source_digest(a.dino_repo) != state["contract"]["dino_source_sha256"]:
        raise ValueError("Evaluation DINOv3 source differs from trained v4 contract")
    device = torch.device(a.device)
    model = TaskStudent(**model_options(a, pretrained=False)).to(device)
    model.load_state_dict(state["model"], strict=True)
    report["test_status"] = "running"
    _save_json(a.out / "comparison.json", report)
    report["test"] = _formal_tests(model, a)
    report["test_status"] = "complete"
    _save_json(a.out / "comparison.json", report)
    return 0


def main(argv=None):
    a = parse_args(argv)
    validate_args(a)
    a.out = a.out.expanduser().resolve()
    if a.eval_only:
        with run_lock(a.out):
            return _evaluate_existing(a)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    resources = infrastructure._resources(a)
    if a.model_variant != "tiny":
        resources["dino_source_sha256"] = dino_source_digest(a.dino_repo)
    splits, manifest = prepare_data(a)
    with run_lock(a.out):
        previous_path = a.out / "comparison.json"
        if a.prepare_only:
            target = a.out / "split_manifest.json"
            if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Refusing to overwrite a different manifest")
            _save_json(target, manifest)
            return 0
        signature, contract = _contracts(a, manifest, resources)
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
        if previous:
            if not a.resume or previous.get("contract_sha256") != _digest(contract):
                raise ValueError("Existing run requires --resume and an identical contract")
            if previous.get("status") == "complete":
                checkpoint = read_checkpoint(a.out / "best.pt", contract)
                if sha256_file(a.out / "best.pt") != previous["checkpoint_sha256"] or checkpoint["best_epoch"] != previous["best_epoch"]:
                    raise ValueError("Completed checkpoint differs from report")
                if a.test_after_training and previous.get("test_status") != "complete":
                    return _evaluate_existing(a)
                print(f"Verified completed run: {a.out}")
                return 0
        elif any(p.name not in {"task_run.lock", "run.log", "launch.json"} for p in a.out.iterdir()):
            raise FileExistsError("Nonempty output directory is not a resumable v4 run")
        # Teacher construction must not perturb matched student initialization.
        teacher = _load_teacher(a, signature, device)
        shared.seed_everything(a.train_seed)
        model = (TaskTeacher if a.stage == "teacher" else TaskStudent)(**model_options(a)).to(device)
        loaders = _loaders(a, splits, device)
        benchmark = a.stage == "student" and a.cod_train_mode == "full" and a.model_variant != "tiny" and not a.max_samples
        report = {"method": METHOD, "stage": a.stage, "group": a.group, "teacher_mode": a.teacher_mode,
                  "status": "running", "config": _json(a), "contract_sha256": _digest(contract), "protocol": contract,
                  "prediction_protocol": PREDICTION_PROTOCOL, "formal": benchmark,
                  "result_scope": "benchmark" if benchmark else "development", "sample_counts": {k: len(v) for k, v in splits.items()},
                  "selection_metric": "source_validation_loss" if a.stage == "teacher" else "last_epoch" if a.cod_train_mode == "full" else "macro_S_alpha",
                  "test_status": "skipped", "test": {}, "torch_version": str(torch.__version__),
                  "cuda_version": torch.version.cuda, "started_at": previous.get("started_at") if previous else datetime.now(timezone.utc).isoformat(),
                  "parameters": {"total": sum(p.numel() for p in model.parameters()),
                                 "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
        _save_json(a.out / "split_manifest.json", manifest)
        _save_json(previous_path, report)
        _fit(model, teacher, loaders, splits, a, device, signature, contract, report)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        _save_json(previous_path, report)
        if a.test_after_training:
            from tools.evaluate_triprd_task import validate_run_report
            validate_run_report(previous_path)
            report["test"] = _formal_tests(model, a)
            report["test_status"] = "complete"
            _save_json(previous_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
