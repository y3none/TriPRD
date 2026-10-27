#!/usr/bin/env python3
"""Independent correction-transfer experiments. No E2Net components/checkpoints.

One four-coalition source teacher; six identically initialized RGB students.
Internal validation selects macro Sm; full-pool runs use the fixed final epoch.
Checkpoints include optimizer, scheduler, scaler, RNG and loader states.
The separate probe stage optimizes only on source TRAIN, diagnoses source VAL,
and uses the predefined final epoch without selecting or initializing COD runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools import run_triprd_ab as shared
from tools.triprd_structure_training import StructurePairDataset, SENSOR_PREPROCESSING, state_digest
from tools.triprd_training import build_scheduler, sha256_file

METHOD = "triprd_correction_v3"
MAX_CONSECUTIVE_AMP_SKIPS = 8
GROUPS = {
    "A": "COD only",
    "B": "RGB/GT source joint training",
    "C": "Source joint training plus full multimodal relation KD",
    "D": "Source joint training plus reliable modality correction KD",
    "E": "D plus COD-training-gradient selection of correction KD",
    "F": "C plus the same COD-training-gradient selection of full KD",
}
TEST_COUNTS = {"COD10K": 2026, "CAMO": 250, "CHAMELEON": 76, "NC4K": 4121}
CODE_FILES = (
    "models/triprd_correction.py", "models/triprd_correction_loss.py",
    "models/dinov3_weights.py",
    "tools/run_triprd_correction.py", "tools/triprd_structure_training.py",
    "tools/run_triprd_ab.py", "tools/triprd_training.py",
    "models/triprd.py", "evaluation/validation.py", "evaluation/cod_metrics.py",
    "evaluation/camoformer_metrics.py", "tools/evaluate_cod.py",
    "datasets/vdt2048.py", "datasets/cod_dataset.py",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("teacher", "student", "probe"), default="student",
                   help="probe is a source-only student diagnostic; it never initializes formal COD training")
    p.add_argument("--group", choices=tuple(GROUPS), default="E")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vdt-root", default="datasets/VDT-2048")
    p.add_argument("--cod-root", default="datasets/COD_benchmark")
    p.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--relation-kernel", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--teacher-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-seed", type=int, default=20261012)
    p.add_argument("--split-seed", type=int, default=20261012)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-metrics-every", type=int, default=5)
    p.add_argument("--cod-train-mode", choices=("split", "full"), default="split")
    p.add_argument("--max-samples", type=int, default=0, help="Nonzero is explicitly a limited diagnostic, not a formal result")
    p.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--source-weight", type=float, default=0.5)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--relation-weight", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--skip-test", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--test-datasets", default="COD10K,CAMO,CHAMELEON,NC4K")
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--intervention-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--cod-checkpoint", type=Path)
    return p.parse_args(argv)


def validate_args(a):
    for key in ("epochs", "teacher_epochs", "batch_size", "base_channels", "val_metrics_every"):
        if getattr(a, key) < 1:
            raise ValueError(f"{key} must be positive")
    if a.image_size < 32 or a.image_size % 32:
        raise ValueError("image-size must be a multiple of 32, at least 32")
    if a.base_channels < 4:
        raise ValueError("base-channels must be at least 4")
    if a.relation_kernel < 3 or a.relation_kernel % 2 != 1:
        raise ValueError("relation-kernel must be odd and at least 3")
    if min(a.num_workers, a.max_samples, a.max_test_samples, a.warmup_epochs) < 0:
        raise ValueError("workers, limits and warmup must be nonnegative")
    if not 0 < a.val_fraction < 1:
        raise ValueError("val-fraction must be in (0,1)")
    if any(not 0 <= v < 2**32 for v in (a.train_seed, a.split_seed)):
        raise ValueError("Seeds must be in [0,2**32)")
    for key in ("lr", "grad_clip", "source_weight", "kd_weight", "relation_weight"):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(a.weight_decay) or a.weight_decay < 0:
        raise ValueError("weight-decay must be finite and nonnegative")
    if a.model_variant == "tiny" and a.pretrained:
        raise ValueError("Tiny requires --no-pretrained and is only a pipeline diagnostic")
    if a.model_variant != "tiny" and not a.pretrained:
        raise ValueError("Formal DINO runs require generic pretrained weights; no silent random initialization")
    if a.cod_train_mode == "full" and a.max_samples:
        raise ValueError("Full COD training cannot use a sample limit")
    names = a.test_datasets.upper().split(",")
    if len(set(names)) != len(names) or set(names) - set(TEST_COUNTS):
        raise ValueError("Unknown or duplicate test datasets")
    if a.eval_only and (a.stage != "student" or not a.cod_checkpoint or a.skip_test or a.resume):
        raise ValueError("Eval-only requires student, --cod-checkpoint, --no-skip-test, without --resume")
    if a.cod_checkpoint and not a.eval_only:
        raise ValueError("--cod-checkpoint is only for evaluation")
    if a.stage == "probe":
        if a.group not in "CD":
            raise ValueError("Source probe only supports C (full KD) or D (correction KD)")
        if not a.skip_test or a.cod_train_mode != "split":
            raise ValueError("Source probe requires --skip-test and does not support full COD training")
    if not a.prepare_only and not a.eval_only:
        if a.stage in {"student", "probe"} and a.group in "CDEF" and not a.teacher_checkpoint:
            raise ValueError("C/D/E/F require a completed --teacher-checkpoint")
        if a.teacher_checkpoint and (a.stage not in {"student", "probe"} or a.group not in "CDEF"):
            raise ValueError("Teacher checkpoint is unused for this stage/group")
    return names


def _json(a):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _save_pt(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def prepare_data(a):
    old = json.loads(a.split_manifest.read_text(encoding="utf-8")) if a.split_manifest else None
    if old and old.get("split_seed", old.get("seed")) != a.split_seed:
        raise ValueError("split-seed differs from manifest")
    vdt = shared._vdt_items(a.vdt_root, a.max_samples)
    if a.stage == "probe":
        # Do not even index COD: this diagnostic's optimization and model
        # assessment must be independent of every COD image/label.
        if not a.max_samples and len(vdt) != 1048:
            raise ValueError(f"Formal source coverage mismatch: VDT={len(vdt)}")
        train, val = shared._manifest_split(vdt, old, "vdt", a.val_fraction, a.split_seed)
        if not train or not val:
            raise ValueError("Empty required source probe partition")
        splits = {"vdt_train": train, "vdt_val": val, "cod_train": [], "cod_val": []}
        manifest = {"split_seed": a.split_seed, "cod_train_mode": "split", "scope": "source_probe_only",
                    **{key: [x.key for x in items] for key, items in splits.items()}}
        return splits, manifest
    cod = shared._cod_items(a.cod_root, a.max_samples)
    if not a.max_samples:
        counts = Counter(x.key.split("/", 1)[0] for x in cod)
        if counts != {"COD10K": 3040, "CAMO": 1000} or len(vdt) != 1048:
            raise ValueError(f"Formal source/train coverage mismatch: VDT={len(vdt)}, COD={dict(counts)}")
    splits = {}
    for prefix, items in (("vdt", vdt), ("cod", cod)):
        if prefix == "cod" and old and old.get("cod_train_mode") == "full":
            if a.cod_train_mode != "full" or set(old["cod_train"]) != {x.key for x in items} or old["cod_val"]:
                raise ValueError("Full-pool manifest cannot be converted to an internal split")
            splits["cod_train"], splits["cod_val"] = items, []
        else:
            splits[f"{prefix}_train"], splits[f"{prefix}_val"] = shared._manifest_split(
                items, old, prefix, a.val_fraction, a.split_seed)
    if a.cod_train_mode == "full":
        splits["cod_train"], splits["cod_val"] = cod, []
    required = ("vdt_train", "vdt_val", "cod_train") + (("cod_val",) if a.cod_train_mode == "split" else ())
    if any(not splits[k] for k in required):
        raise ValueError("Empty required data partition")
    manifest = {"split_seed": a.split_seed, "cod_train_mode": a.cod_train_mode,
                **{k: [x.key for x in v] for k, v in splits.items()}}
    return splits, manifest


class CorrectionDataset(StructurePairDataset):
    """Explicit normalization and stateless aligned augmentation across groups/workers."""
    def __init__(self, items, size, *, normalize, augment=False, seed=0):
        super().__init__(items, size, augment=False)
        self.normalize, self.use_augmentation, self.seed, self.epoch = normalize, augment, seed, 0
        self._archive_pid = None

    def __getstate__(self):
        # ZipFile handles cannot be pickled on Windows. Strip only handles,
        # preserving the already built index; workers reopen them locally.
        state = self.__dict__.copy()
        indexes, items = {}, []
        for item in self.items:
            sample, index = item.source
            if id(index) not in indexes:
                cloned = copy.copy(index)
                if hasattr(index, "source"):
                    cloned.source = copy.copy(index.source)
                    if hasattr(cloned.source, "archive"):
                        cloned.source.archive = None
                indexes[id(index)] = cloned
            items.append(shared.Item(item.key, (sample, indexes[id(index)]), item.kind))
        state["items"], state["_archive_pid"] = items, None
        return state

    def _ensure_local_archives(self):
        if self._archive_pid == os.getpid():
            return
        seen = set()
        for item in self.items:
            source = getattr(item.source[1], "source", None)
            if source is not None and id(source) not in seen and hasattr(source, "archive"):
                seen.add(id(source))
                if source.archive is not None:
                    source.archive.close()
                source.archive = zipfile.ZipFile(source.root)
        self._archive_pid = os.getpid()

    def __getitem__(self, index):
        self._ensure_local_archives()
        # The legacy reader uses a module flag. Set it explicitly inside each
        # worker, then normalize here; Windows spawn never inherits this flag.
        shared.NORMALIZE_RGB = False
        result = super().__getitem__(index)
        if self.use_augmentation:
            flip = int(hashlib.sha256(f"{self.seed}:{self.epoch}:{result['key']}".encode()).hexdigest()[:8], 16) % 2
            if flip:
                result = {k: v.flip(-1) if torch.is_tensor(v) and v.ndim >= 3 else v for k, v in result.items()}
        if self.normalize:
            result["rgb"] = (result["rgb"] - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        return result


def _loaders(a, splits, device):
    result = {}
    for n, (key, items) in enumerate(splits.items()):
        if not items:
            continue
        train = key.endswith("train")
        dataset = CorrectionDataset(items, a.image_size, normalize=a.pretrained,
                                    augment=train, seed=a.train_seed)
        generator = torch.Generator().manual_seed(a.train_seed + n * 1009)
        result[key] = DataLoader(dataset, batch_size=a.batch_size, shuffle=train,
            num_workers=a.num_workers, pin_memory=device.type == "cuda", generator=generator,
            persistent_workers=False)
    return result


def _autocast(a, device):
    return torch.autocast("cuda", dtype=torch.float16) if a.amp and device.type == "cuda" else nullcontext()


def _model_options(a, *, pretrained=None):
    return dict(model_variant=a.model_variant, base_channels=a.base_channels,
        pretrained=a.pretrained if pretrained is None else pretrained,
        dino_repo=a.dino_repo, dino_weights=a.dino_weights,
        relation_kernel=a.relation_kernel, freeze_backbone=True)


def _resources(a):
    if a.model_variant == "tiny":
        return {"generic_weights_sha256": None, "generic_config_sha256": None, "dino_source_sha256": None}
    a.dino_repo = a.dino_repo.expanduser().resolve() if a.dino_repo else None
    a.dino_weights = a.dino_weights.expanduser().absolute() if a.dino_weights else None
    if not a.dino_repo or not (a.dino_repo / "hubconf.py").is_file():
        raise FileNotFoundError("--dino-repo must be an official local DINOv3 checkout with hubconf.py")
    if not a.dino_weights or not a.dino_weights.is_file():
        raise FileNotFoundError("--dino-weights must be a local generic DINOv3 ViT-B/16 .pth or .safetensors file")
    files = [a.dino_repo / "hubconf.py", *sorted((a.dino_repo / "dinov3").rglob("*.py"))]
    config = a.dino_weights.with_name("config.json")
    return {"generic_weights_sha256": sha256_file(a.dino_weights),
            "generic_config_sha256": sha256_file(config) if a.dino_weights.suffix.lower() == ".safetensors" and config.is_file() else None,
            "dino_source_sha256": _digest({str(p.relative_to(a.dino_repo)): sha256_file(p) for p in files})}


def _contracts(a, manifest, resources):
    source = {p: sha256_file(ROOT / p) for p in CODE_FILES}
    common = {"method": METHOD, "model_variant": a.model_variant, "base_channels": a.base_channels,
        "relation_kernel": a.relation_kernel, "image_size": a.image_size, "pretrained": a.pretrained,
        "freeze_backbone": True, "source_sha256": source, **resources,
        "rgb_normalization": "ImageNet" if a.pretrained else "[0,1]",
        "sensor_preprocessing": SENSOR_PREPROCESSING, "train_seed": a.train_seed,
        "split_seed": a.split_seed, "batch_size": a.batch_size, "num_workers": a.num_workers,
        "amp": a.amp, "lr": a.lr, "weight_decay": a.weight_decay,
        "relation_weight": a.relation_weight, "grad_clip": a.grad_clip,
        "teacher_epochs": a.teacher_epochs, "warmup_epochs": a.warmup_epochs,
        "vdt_train": manifest["vdt_train"], "vdt_val": manifest["vdt_val"]}
    teacher = {**common, "stage": "teacher", "selection": "minimum_four_coalition_relation_loss"}
    if a.stage == "probe":
        probe = {**common, "stage": "probe", "group": a.group, "epochs": a.epochs,
            "source_weight": a.source_weight, "kd_weight": a.kd_weight,
            "val_metrics_every": a.val_metrics_every, "selection": "fixed_last_epoch",
            "data_scope": "VDT_train_optimization_VDT_val_diagnostic_only",
            "initialization": "generic_pretrained_only_no_probe_warm_start",
            "teacher_sha256": sha256_file(a.teacher_checkpoint)}
        return teacher, probe
    student = {**common, "stage": "student", "group": a.group, "epochs": a.epochs,
        "source_weight": a.source_weight, "kd_weight": a.kd_weight,
        "cod_train": manifest["cod_train"], "cod_val": manifest["cod_val"],
        "cod_train_mode": a.cod_train_mode, "val_metrics_every": a.val_metrics_every,
        "intervention_eval": a.intervention_eval, "skip_test": a.skip_test,
        "test_datasets": a.test_datasets, "max_test_samples": a.max_test_samples,
        "teacher_sha256": sha256_file(a.teacher_checkpoint) if a.teacher_checkpoint else None}
    return teacher, teacher if a.stage == "teacher" else student


def _rng_state(loaders):
    nr = np.random.get_state()
    return {"python": random.getstate(), "numpy": [nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]],
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": {k: v.generator.get_state() for k, v in loaders.items()}}


def _restore_rng(state, loaders):
    random.setstate(state["python"])
    nr = state["numpy"]
    np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    for k, v in loaders.items():
        v.generator.set_state(state["loaders"][k].cpu())


def _read_checkpoint(path, contract=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("method") != METHOD or state.get("contract_sha256") != _digest(state.get("contract")):
        raise ValueError("Not a valid correction_v3 checkpoint")
    if contract is not None and state["contract"] != contract:
        changed = sorted(k for k in set(contract) | set(state["contract"]) if contract.get(k) != state["contract"].get(k))
        raise ValueError(f"Checkpoint contract mismatch: {changed}")
    return state


def _add_stats(acc, stats):
    for key, value in stats.items():
        value = value.detach()
        acc[key] = value.clone() if key not in acc else acc[key] + value


def _finite(loss, label):
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"Non-finite {label}")


def _optimizer_step(model, optimizer, scaler, grad_clip):
    """Clip finite unscaled gradients; let GradScaler skip/back off on overflow.

    A nonfinite *norm* alone is insufficient evidence of a scaler overflow:
    the norm reduction can overflow even when every gradient is finite. In
    that case GradScaler would step, so fail before modifying any gradients.
    """
    named = [(name, p) for name, p in model.named_parameters()
             if p.requires_grad and p.grad is not None]
    if not named:
        raise FloatingPointError("No trainable gradients were produced")
    before = float(scaler.get_scale())
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError(f"Invalid gradient scale: {before}")
    # For scales below one, unscale_ can itself overflow a finite gradient.
    # The fused scaler check tests values BEFORE multiplication, so it may not
    # record this new overflow. Never let that case reach optimizer.step().
    scaled_nonfinite = (any(not torch.isfinite(p.grad).all() for _, p in named)
                        if scaler.is_enabled() and before < 1 else None)
    scaler.unscale_(optimizer)
    parameters = [p for _, p in named]
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters], error_if_nonfinite=False)
    if torch.isfinite(norm).all():
        torch.nn.utils.clip_grads_with_norm_(parameters, grad_clip, norm)
        scaler.step(optimizer)
        scaler.update()
        return norm.detach(), None

    bad = [name for name, p in named if not torch.isfinite(p.grad).all()]
    if not bad:
        raise FloatingPointError("Non-finite gradient norm with finite gradients; optimizer was not stepped")
    if not scaler.is_enabled():
        raise FloatingPointError(f"Non-finite gradients without AMP scaling: {bad[:8]}")
    if scaled_nonfinite is False:
        raise FloatingPointError("Unscaling finite gradients produced non-finite values; optimizer was not stepped")
    # unscale_ has already recorded the nonfinite gradients. step() now skips
    # optimizer.step(), including AdamW weight decay and moment updates.
    # Do not clip or replace those gradients before this decision.
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    if not math.isfinite(after) or not 0 < after < before:
        raise FloatingPointError(f"AMP failed to reduce a valid scale after overflow: {before} -> {after}")
    return None, {"scale_before": before, "scale_after": after,
                  "nonfinite_parameters": bad[:8], "nonfinite_parameter_count": len(bad)}


def _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch):
    from models.triprd_correction_loss import teacher_loss, source_loss, positive_gradient_alignment
    model.train()
    for loader in loaders.values():
        loader.dataset.epoch = epoch
    if teacher is not None:
        teacher.eval()
    primary = loaders["vdt_train" if a.stage in {"teacher", "probe"} else "cod_train"]
    source_iter = iter(loaders["vdt_train"]) if a.stage == "student" and a.group != "A" else None
    stats, count, source_count = {}, 0, 0
    optimizer_steps, consecutive_skips = 0, 0
    norm_sum = torch.zeros((), device=device)
    overflow_events = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(primary, 1):
        batch = shared._move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(a, device):
            if a.stage == "teacher":
                outputs = model.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
                losses = teacher_loss(outputs, batch["mask"], relation_kernel=a.relation_kernel, relation_weight=a.relation_weight)
                total, logged = losses["total"], losses
            elif a.stage == "probe":
                with torch.no_grad():
                    coals = teacher.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
                prediction = model(batch["rgb"], task="sod")
                source = source_loss(prediction, batch["mask"], teacher_rgb=coals["R"],
                    teacher_full=coals["RTD"], mode="ordinary" if a.group == "C" else "correction",
                    relation_kernel=a.relation_kernel, relation_weight=a.relation_weight, kd_weight=a.kd_weight)
                # The same weighted source term as the formal student, without
                # COD supervision or COD-gradient selection.
                total = a.source_weight * source["total"]
                logged = {"source_supervised": source["supervised"], "source_transfer": source["transfer"],
                    "source_positive_coverage": source["positive_coverage"],
                    "source_reliability_mean": source["reliability_mean"]}
                source_count += batch["rgb"].shape[0]
            else:
                output = model(batch["rgb"], task="cod")
                target = source_loss(output, batch["mask"], mode="none",
                    relation_kernel=a.relation_kernel, relation_weight=a.relation_weight)
                total = target["total"]
                logged = {"cod_total": total, "cod_segmentation": target["segmentation"]}
                if source_iter is not None:
                    try:
                        source_batch = next(source_iter)
                    except StopIteration:
                        source_iter = iter(loaders["vdt_train"])
                        source_batch = next(source_iter)
                    source_batch = shared._move(source_batch, device)
                    source_count += source_batch["rgb"].shape[0]
                    with torch.no_grad():
                        coals = teacher.forward_coalitions(source_batch["rgb"], source_batch["thermal"], source_batch["depth"]) if teacher is not None else None
                    prediction = model(source_batch["rgb"], task="sod")
                    mode = "correction" if a.group in "DE" else "ordinary" if a.group in "CF" else "none"
                    source = source_loss(prediction, source_batch["mask"],
                        teacher_rgb=coals["R"] if coals else None, teacher_full=coals["RTD"] if coals else None,
                        mode=mode, relation_kernel=a.relation_kernel,
                        relation_weight=a.relation_weight, kd_weight=a.kd_weight)
                    alignment = positive_gradient_alignment(source["transfer"], total,
                        model.alignment_parameters(), gradient_scale=scaler.get_scale()) if a.group in "EF" else total.detach().new_tensor(1.)
                    total = total + a.source_weight * (source["supervised"] + a.kd_weight * alignment * source["transfer"])
                    logged.update(source_supervised=source["supervised"], source_transfer=source["transfer"],
                        source_positive_coverage=source["positive_coverage"],
                        source_reliability_mean=source["reliability_mean"],
                        alignment=alignment, accepted=(alignment > 0).float())
        _finite(total, f"training loss at {a.stage}/{a.group} epoch={epoch} batch={batch_index}")
        scaler.scale(total).backward()
        norm, overflow = _optimizer_step(model, optimizer, scaler, a.grad_clip)
        if overflow is not None:
            consecutive_skips += 1
            overflow.update(batch=batch_index, primary_keys=list(batch.get("key", [])))
            if source_iter is not None:
                overflow["source_keys"] = list(source_batch.get("key", []))
            overflow_events.append(overflow)
            print(f"[{a.stage}/{a.group}] AMP skipped update: epoch={epoch} batch={batch_index} "
                  f"scale={overflow['scale_before']:g}->{overflow['scale_after']:g} "
                  f"nonfinite_parameters={overflow['nonfinite_parameters']} "
                  f"primary_keys={overflow['primary_keys']} "
                  f"source_keys={overflow.get('source_keys', [])}", flush=True)
            if consecutive_skips >= MAX_CONSECUTIVE_AMP_SKIPS:
                raise FloatingPointError(f"{consecutive_skips} consecutive AMP updates skipped; "
                                         "persistent non-finite gradients require investigation")
        else:
            consecutive_skips = 0
            optimizer_steps += 1
            norm_sum += norm
        logged = {**logged, "total": total.detach()}
        _add_stats(stats, logged)
        count += 1
    if optimizer_steps == 0:
        raise FloatingPointError("No optimizer updates in this epoch; refusing to save an untrained epoch")
    elapsed = time.perf_counter() - started
    return {**{k: float(v / count) for k, v in stats.items()}, "steps": count,
            "optimizer_steps": optimizer_steps, "amp_skipped_steps": len(overflow_events),
            "amp_overflow_events": overflow_events, "amp_scale": float(scaler.get_scale()),
            "grad_norm": float(norm_sum / optimizer_steps),
            "primary_images": len(primary.dataset), "source_images": source_count,
            "seconds": elapsed, "images_per_second": len(primary.dataset) / elapsed}


@torch.no_grad()
def _teacher_validation(model, loader, device, a):
    from models.triprd_correction_loss import teacher_loss, correction_targets
    model.eval()
    sums, n, positive, valid = {}, 0, 0., 0.
    for batch in loader:
        batch = shared._move(batch, device)
        with _autocast(a, device):
            outputs = model.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
            losses = teacher_loss(outputs, batch["mask"], relation_kernel=a.relation_kernel, relation_weight=a.relation_weight)
        weight = len(batch["rgb"])
        _add_stats(sums, {k: v * weight for k, v in losses.items()})
        from models.triprd_correction_loss import relation_supervision
        for name, pred in outputs.items():
            loss = relation_supervision(pred["relation_logits"], batch["mask"], relation_kernel=a.relation_kernel)
            _add_stats(sums, {f"relation_{name}": loss * weight})
        targets = correction_targets(outputs["R"], outputs["RTD"], batch["mask"], relation_kernel=a.relation_kernel)
        for rel, mask in zip(targets["reliability"], targets["valid"]):
            positive += float(((rel > 0) & mask.bool()).sum())
            valid += float(mask.expand_as(rel).sum())
        n += weight
    result = {k: float(v / n) for k, v in sums.items()}
    result.update(mean_relation_gain=result["relation_R"] - result["relation_RTD"],
                  positive_coverage=positive / max(valid, 1), num_samples=n)
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Nonfinite teacher validation")
    return result


@torch.no_grad()
def _constant_probe(teacher, loader, device, a):
    from models.triprd_correction_loss import correction_targets, constant_correction_statistics, finalize_constant_correction
    teacher.eval()
    stats = None
    for batch in loader:
        batch = shared._move(batch, device)
        with _autocast(a, device):
            outputs = teacher.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
        current = constant_correction_statistics(correction_targets(outputs["R"], outputs["RTD"], batch["mask"], relation_kernel=a.relation_kernel))
        stats = current if stats is None else {k: tuple(x + y for x, y in zip(stats[k], current[k])) for k in current}
    return {name: tuple(x.detach().cpu() for x in finalize_constant_correction(stats, positive_only=positive))
            for name, positive in (("all", False), ("positive", True))}


@torch.no_grad()
def _student_probe(student, teacher, loader, constants, device, a):
    from models.triprd_correction_loss import correction_targets, correction_probe, finalize_correction_probe
    student.eval()
    teacher.eval()
    stats = {}
    constants = {name: tuple(x.to(device) for x in values) for name, values in constants.items()}
    for batch in loader:
        batch = shared._move(batch, device)
        with _autocast(a, device):
            coals = teacher.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
            pred = student(batch["rgb"], task="sod")
        targets = correction_targets(coals["R"], coals["RTD"], batch["mask"], relation_kernel=a.relation_kernel)
        _add_stats(stats, correction_probe(pred, targets, constants["all"], relation_kernel=a.relation_kernel,
                                         positive_constant_correction=constants["positive"]))
    result = {k: float(v) for k, v in finalize_correction_probe(stats).items()}
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Nonfinite source recoverability probe")
    return result


@torch.no_grad()
def _validate_cod(model, items, device, a, intervention="none"):
    from evaluation.validation import CODValidationMetrics
    metrics = CODValidationMetrics()
    model.eval()
    shared.NORMALIZE_RGB = a.pretrained
    for item in items:
        sample, index = item.source
        rgb, mask = index.read_pair(sample, strict_size=False)
        image = shared._tensor(rgb, a.image_size, rgb=True).unsqueeze(0).to(device)
        with _autocast(a, device):
            logits = model(image, task="cod", intervention=intervention)["logits"]
        pred = shared._prediction_image(logits.float(), mask.size)
        metrics.update(item.key.split("/", 1)[0], np.asarray(pred, dtype=np.uint8), np.asarray(mask.convert("L"), dtype=np.uint8))
    return metrics.compute()


def _formal_tests(model, a, out, device):
    from tools.run_triprd_structure import _evaluate_complete
    class Inference(torch.nn.Module):
        def __init__(self, student):
            super().__init__()
            self.student = student

        def forward(self, rgb):
            with _autocast(a, device):
                logits = self.student(rgb, task="cod")["logits"]
            return {"logits": logits.float()}

    inference = Inference(model)
    shared.NORMALIZE_RGB = a.pretrained
    result = {}
    for dataset in a.test_datasets.upper().split(","):
        result[dataset] = _evaluate_complete(inference, a, dataset, out, device)
        result[dataset]["official"]["model"] = "TriPRD-Correction"
        _save_json(out / "test_metrics.json", result)
    return result


def _fit(model, teacher, loaders, splits, a, device, contract, report):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    epochs = a.teacher_epochs if a.stage == "teacher" else a.epochs
    scheduler = build_scheduler(optimizer, "cosine", epochs, min(a.warmup_epochs, epochs - 1), .05)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
    start, best, best_epoch, history = 1, None, None, []
    initial_digest = state_digest(model)
    if a.resume and (a.out / "last.pt").exists():
        checkpoint = _read_checkpoint(a.out / "last.pt", contract)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"], loaders)
        start, best, best_epoch, history = checkpoint["epoch"] + 1, checkpoint["best_score"], checkpoint["best_epoch"], checkpoint["history"]
        initial_digest = checkpoint["initial_state_sha256"]
        # Repair the two-file transaction BEFORE another epoch can overwrite
        # last.pt. A crash may have happened between saving last and best.
        best_path = a.out / "best.pt"
        saved_best_epoch = _read_checkpoint(best_path, contract)["epoch"] if best_path.exists() else None
        if best_epoch is not None and saved_best_epoch != best_epoch:
            if checkpoint["epoch"] != best_epoch:
                raise RuntimeError("Selected checkpoint is missing and cannot be reconstructed from last.pt")
            _save_pt(best_path, checkpoint)
    report["initial_state_sha256"] = initial_digest
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, epochs + 1):
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row["train"] = _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch)
        selected, score = False, None
        if a.stage == "teacher":
            row["validation"] = _teacher_validation(model, loaders["vdt_val"], device, a)
            score = -row["validation"]["relation"]
        elif a.cod_train_mode == "split" and (epoch % a.val_metrics_every == 0 or epoch == epochs):
            row["validation"] = _validate_cod(model, splits["cod_val"], device, a)
            score = row["validation"]["macro"]["S_alpha"]
        elif a.cod_train_mode == "full":
            score = float(epoch)
        if score is not None and (best is None or score > best):
            best, best_epoch, selected = score, epoch, True
        scheduler.step()
        history.append(row)
        checkpoint = {"method": METHOD, "stage": a.stage, "group": a.group,
            "contract": contract, "contract_sha256": _digest(contract), "config": _json(a),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_score": best, "best_epoch": best_epoch, "history": history,
            "initial_state_sha256": initial_digest, "rng": _rng_state(loaders)}
        # Last is the atomic epoch boundary. If a crash occurs before best is
        # written, resume can reconstruct it whenever the last epoch is best.
        _save_pt(a.out / "last.pt", checkpoint)
        if selected:
            _save_pt(a.out / "best.pt", checkpoint)
        _save_json(a.out / "history.json", history)
        report.update(completed_epochs=epoch, best_epoch=best_epoch, best_score=best,
                      latest=row, history=history)
        _save_json(a.out / "comparison.json", report)
        print(f"[{a.stage}/{a.group}] {epoch:03d}/{epochs} loss={row['train']['total']:.5f} "
              f"time={row['train']['seconds']:.1f}s selected={best_epoch} "
              f"updates={row['train']['optimizer_steps']} amp_skips={row['train']['amp_skipped_steps']} "
              f"scale={row['train']['amp_scale']:g}", flush=True)
    if not (a.out / "best.pt").exists() or _read_checkpoint(a.out / "best.pt", contract)["epoch"] != best_epoch:
        last = _read_checkpoint(a.out / "last.pt", contract)
        if last["epoch"] != best_epoch:
            raise RuntimeError("Selected checkpoint is missing; refusing to substitute another epoch")
        _save_pt(a.out / "best.pt", last)
    selected = _read_checkpoint(a.out / "best.pt", contract)
    model.load_state_dict(selected["model"], strict=True)
    report.update(completed_epochs=epochs, best_epoch=best_epoch, best_score=best, history=history,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)


def _fit_probe(model, teacher, constants, loaders, a, device, contract, report):
    """Short source-only optimization, fixed final state, exact epoch resume.

    Validation diagnostics are recorded but never determine the checkpoint.
    Keeping a distinct stage/contract and only last.pt prevents this state from
    being mistaken for a selected formal student or a reusable teacher.
    """
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=a.lr, weight_decay=a.weight_decay)
    scheduler = build_scheduler(optimizer, "cosine", a.epochs, min(a.warmup_epochs, a.epochs - 1), .05)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
    start, history = 1, []
    initial_digest = state_digest(model)
    if a.resume and (a.out / "last.pt").exists():
        checkpoint = _read_checkpoint(a.out / "last.pt", contract)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"], loaders)
        start, history = checkpoint["epoch"] + 1, checkpoint["history"]
        initial_digest = checkpoint["initial_state_sha256"]
        initial_probe = checkpoint["initial_source_recoverability"]
    else:
        initial_probe = _student_probe(model, teacher, loaders["vdt_val"], constants, device, a)
    report.update(initial_state_sha256=initial_digest, initial_source_recoverability=initial_probe)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, a.epochs + 1):
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row["train"] = _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch)
        if epoch % a.val_metrics_every == 0 or epoch == a.epochs:
            row["source_recoverability"] = _student_probe(model, teacher, loaders["vdt_val"], constants, device, a)
        scheduler.step()
        history.append(row)
        checkpoint = {"method": METHOD, "stage": "probe", "group": a.group, "diagnostic": True,
            "contract": contract, "contract_sha256": _digest(contract), "config": _json(a),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "history": history, "selection_metric": "fixed_last_epoch",
            "initial_state_sha256": initial_digest, "initial_source_recoverability": initial_probe,
            "rng": _rng_state(loaders)}
        _save_pt(a.out / "last.pt", checkpoint)
        _save_json(a.out / "history.json", history)
        report.update(completed_epochs=epoch, latest=row, history=history)
        _save_json(a.out / "comparison.json", report)
        probe = row.get("source_recoverability")
        detail = (f" mse={probe['mse']:.6f} zero={probe['zero_mse']:.6f} "
                  f"constant={probe['constant_mse']:.6f}" if probe else "")
        print(f"[probe/{a.group} DIAGNOSTIC] {epoch:03d}/{a.epochs} "
              f"source_loss={row['train']['total']:.5f}{detail} "
              f"updates={row['train']['optimizer_steps']} amp_skips={row['train']['amp_skipped_steps']} "
              f"scale={row['train']['amp_scale']:g}; no checkpoint selection", flush=True)
    last = _read_checkpoint(a.out / "last.pt", contract)
    if last["epoch"] != a.epochs:
        raise RuntimeError("Source probe did not reach its predefined final epoch")
    model.load_state_dict(last["model"], strict=True)
    report.update(completed_epochs=a.epochs, selected_epoch=a.epochs, history=history,
        source_recoverability=history[-1]["source_recoverability"],
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)


def main(argv=None):
    a = parse_args(argv)
    validate_args(a)
    a.out = a.out.expanduser().resolve()
    if a.eval_only:
        return _eval_only(a)
    splits, manifest = prepare_data(a)
    if a.prepare_only:
        path = a.out / "split_manifest.json"
        if path.exists() and (not a.resume or json.loads(path.read_text(encoding="utf-8")) != manifest):
            raise FileExistsError("Prepared manifest exists or differs; use matching --resume")
        _save_json(path, manifest)
        print(f"Prepared matched split: {path}", flush=True)
        return 0
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Formal training requires a CUDA PyTorch environment")
    resources = _resources(a)
    teacher_contract, contract = _contracts(a, manifest, resources)
    report_path = a.out / "comparison.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    result_checkpoint = a.out / ("last.pt" if a.stage == "probe" else "best.pt")
    if previous:
        if not a.resume or previous.get("contract_sha256") != _digest(contract):
            raise ValueError("Existing output requires --resume with an identical contract")
        if previous.get("status") == "complete":
            _read_checkpoint(result_checkpoint, contract)
            if previous.get("checkpoint_sha256") != sha256_file(result_checkpoint):
                raise ValueError("Completed checkpoint digest differs from saved report")
            print(f"Verified completed experiment: {a.out}", flush=True)
            return 0
    elif a.out.exists() and any(p.name not in {"launch.json", "run.log"} for p in a.out.iterdir()):
        raise FileExistsError(f"Refusing existing non-experiment output: {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    _save_json(a.out / "split_manifest.json", manifest)
    from models.triprd_correction import CorrectionStudent, CorrectionTeacher
    if a.stage != "probe":
        from evaluation.validation import CODValidationMetrics  # fail before training if evaluator unavailable
    report = {"method": METHOD, "stage": a.stage, "status": "running", "config": _json(a),
        "contract_sha256": _digest(contract), "protocol": contract,
        "definition": "Shared four-coalition teacher" if a.stage == "teacher" else GROUPS[a.group],
        "sample_counts": {k: len(v) for k, v in splits.items()},
        "formal": a.stage != "probe" and a.model_variant != "tiny" and a.max_samples == 0,
        "diagnostic": a.stage == "probe",
        "selection_metric": "fixed_last_epoch" if a.stage == "probe" else "minimum_relation_loss" if a.stage == "teacher" else "last_epoch" if a.cod_train_mode == "full" else "macro_S_alpha",
        "started_at": previous.get("started_at") if previous else datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "test": {}, "test_status": "skipped" if a.skip_test or a.stage == "teacher" else "pending"}
    if a.stage == "probe":
        report.update(validation_used_for_selection=False, cod_access=False,
            definition="Source-only RGB student learnability diagnostic: " +
                ("full multimodal relation KD" if a.group == "C" else "reliable modality correction KD"),
            initialization="Fresh generic-pretrained student; probe weights are not used by formal student training",
            diagnostic_scope="VDT TRAIN optimization, VDT VAL recoverability; teacher constants fitted only on VDT TRAIN")
    _save_json(report_path, report)
    try:
        teacher, constants = None, None
        if a.teacher_checkpoint:
            state = _read_checkpoint(a.teacher_checkpoint, teacher_contract)
            teacher_report_path = a.teacher_checkpoint.parent / "comparison.json"
            teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
            if teacher_report.get("status") != "complete" or teacher_report.get("contract_sha256") != state["contract_sha256"]:
                raise ValueError("Teacher checkpoint does not belong to a completed, matched source run")
            if teacher_report.get("checkpoint_sha256") != sha256_file(a.teacher_checkpoint):
                raise ValueError("Teacher checkpoint digest differs from its completion report")
            if not state.get("constant_correction") or not state.get("positive_constant_correction"):
                raise ValueError("Teacher checkpoint lacks both all-edge and positive-edge TRAIN-fitted probe constants; regenerate the teacher with the current code")
            constants = {"all": state["constant_correction"], "positive": state["positive_constant_correction"]}
            teacher = CorrectionTeacher(**_model_options(a)).to(device)
            teacher.load_state_dict(state["model"], strict=True)
            teacher.requires_grad_(False).eval()
            del state
        # Reset after teacher construction: all student initial states match A.
        shared.seed_everything(a.train_seed)
        model = (CorrectionTeacher if a.stage == "teacher" else CorrectionStudent)(**_model_options(a)).to(device)
        frozen_encoder = state_digest(model.encoder)
        report.update(parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      generic_weight_loading=model.encoder.weight_load_metadata)
        loaders = _loaders(a, splits, device)
        if a.stage == "probe":
            _fit_probe(model, teacher, constants, loaders, a, device, contract, report)
        else:
            _fit(model, teacher, loaders, splits, a, device, contract, report)
        if state_digest(model.encoder) != frozen_encoder:
            raise RuntimeError("Frozen generic encoder changed during training")
        report["frozen_generic_encoder_verified"] = True
        if a.stage == "teacher":
            # Fixed training images only; no source-val constants, no augmentation.
            loaders["vdt_train"].dataset.use_augmentation = False
            constants = _constant_probe(model, loaders["vdt_train"], device, a)
            probe = _teacher_validation(model, loaders["vdt_val"], device, a)
            state = _read_checkpoint(a.out / "best.pt", contract)
            state["constant_correction"] = constants["all"]
            state["positive_constant_correction"] = constants["positive"]
            state["constant_correction_scope"] = "VDT_TRAIN_only_all_valid_and_positive_reliability_per_scale_offset"
            _save_pt(a.out / "best.pt", state)
            report["teacher_probe"] = probe
            _save_json(a.out / "teacher_probe.json", {"status": "complete", "contract_sha256": _digest(contract), **probe})
        elif a.stage == "probe":
            _save_json(a.out / "source_probe.json", {"status": "complete", "stage": "probe", "diagnostic": True,
                "group": a.group,
                "contract_sha256": _digest(contract), "teacher_sha256": contract["teacher_sha256"],
                "selected_epoch": a.epochs, "selection_metric": "fixed_last_epoch",
                "source_recoverability": report["source_recoverability"],
                "initial_source_recoverability": report["initial_source_recoverability"]})
        else:
            if teacher is not None:
                report["source_recoverability"] = _student_probe(model, teacher, loaders["vdt_val"], constants, device, a)
            if a.cod_train_mode == "split" and a.intervention_eval:
                report["intervention_validation"] = {"selection_used": False, "metrics": {
                    mode: _validate_cod(model, splits["cod_val"], device, a, mode)
                    for mode in ("none", "no_correction", "shuffle_correction")}}
            if not a.skip_test:
                report["test"] = _formal_tests(model, a, a.out, device)
                report["test_status"] = "complete"
        report.update(status="complete", checkpoint=str(result_checkpoint),
                      checkpoint_sha256=sha256_file(result_checkpoint))
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_json(report_path, report)
    return 0


def _eval_only(a):
    """RGB inference needs no VDT, teacher, or COD training image access."""
    from models.triprd_correction import CorrectionStudent
    state = _read_checkpoint(a.cod_checkpoint)
    if state["stage"] != "student":
        raise ValueError("Evaluation requires a student checkpoint")
    for key in ("model_variant", "base_channels", "relation_kernel", "image_size", "pretrained"):
        if state["contract"][key] != getattr(a, key):
            raise ValueError(f"Evaluation model mismatch: {key}")
    if a.out.exists():
        raise FileExistsError(a.out)
    device = torch.device(a.device)
    # Official architecture is still needed, but the self-contained checkpoint
    # contains its frozen backbone; no generic weight file is downloaded.
    model = CorrectionStudent(**_model_options(a, pretrained=False)).to(device)
    model.load_state_dict(state["model"], strict=True)
    a.out.mkdir(parents=True)
    report = {"method": METHOD, "status": "running", "config": _json(a), "test": {},
              "checkpoint_group": state["group"], "checkpoint_epoch": state["epoch"],
              "checkpoint_sha256": sha256_file(a.cod_checkpoint), "training_protocol": state["contract"]}
    try:
        report["test"] = _formal_tests(model, a, a.out, device)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _save_json(a.out / "comparison.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
