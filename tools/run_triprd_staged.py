#!/usr/bin/env python3
"""TriPRD v5: source pretraining followed by independent COD-only optimization."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from tools import run_triprd_task as v4
from models.triprd_task import TaskStudent, TASK_MODEL_SPEC
from models.triprd_task_loss import supervised_loss, transfer_loss
from tools.triprd_structure_training import state_digest, SENSOR_PREPROCESSING
from tools.triprd_training import build_optimizer, build_scheduler, sha256_file
METHOD = "triprd_staged_v5"
GROUPS = {"S0": "COD only", "S1": "source GT then COD", "S2": "source RGB KD then COD", "S3": "source multimodal KD then COD"}
CODE_FILES = tuple(dict.fromkeys(("tools/run_triprd_staged.py", "tools/evaluate_triprd_staged.py", *v4.CODE_FILES)))
COMMON_FIELDS = ("model_variant", "base_channels", "image_size", "pretrained", "train_backbone_blocks", "detail_branch", "batch_size", "num_workers", "train_seed", "split_seed", "lr", "backbone_lr_scale", "weight_decay", "warmup_epochs", "amp", "grad_clip", "augmentation", "boundary_weight", "aux_weight", "teacher_epochs", "source_epochs", "kd_weight", "feature_weight", "temperature")
_digest, _json = v4._digest, v4._json
_save_json, _save_pt, run_lock = v4._save_json, v4._save_pt, v4.run_lock
infrastructure, shared = v4.infrastructure, v4.shared
model_options, _loaders = v4.model_options, v4._loaders
PREDICTION_PROTOCOL, TEST_COUNTS = v4.PREDICTION_PROTOCOL, v4.TEST_COUNTS

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("source", "cod"), default="cod")
    p.add_argument("--group", choices=tuple(GROUPS), default="S0")
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
    p.add_argument("--source-epochs", type=int, default=20)
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
    p.add_argument("--kd-weight", type=float, default=1.)
    p.add_argument("--feature-weight", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=2.)
    p.add_argument("--boundary-weight", type=float, default=.2)
    p.add_argument("--aux-weight", type=float, default=.5)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--source-checkpoint", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="Evaluate this existing full4040 run; no training/source data access")
    p.add_argument("--test-after-training", action="store_true", help="Explicitly test only a full4040 final run")
    return p.parse_args(argv)

def _teacher_args(a):
    bridge = copy.copy(a)
    bridge.stage = "student"
    bridge.group = "T2" if a.group == "S2" else "T4"
    bridge.teacher_mode = "rgb" if a.group == "S2" else "multimodal"
    bridge.source_weight, bridge.source_final_ratio = .5, .1
    return bridge

def validate_args(a):
    bridge = _teacher_args(a)
    bridge.stage, bridge.group, bridge.teacher_checkpoint = "student", "T0", None
    v4.validate_args(bridge)
    if a.source_epochs < 1:
        raise ValueError("source_epochs must be positive")
    if a.stage == "source" and a.group == "S0":
        raise ValueError("S0 has no source stage")
    if a.stage == "source" and (a.cod_train_mode != "split" or a.test_after_training or a.eval_only):
        raise ValueError("source stage cannot access COD validation or official tests")
    if a.model_variant != "tiny" and not a.split_manifest and not a.eval_only:
        raise ValueError("Real staged runs require the original --split-manifest")
    if (a.kd_weight, a.feature_weight, a.temperature) != (1., 1., 2.):
        raise ValueError("v5 fixes source KD weight=1, feature weight=1 and temperature=2")
    if not (a.eval_only or a.prepare_only):
        if (a.stage == "source" and a.group in {"S2", "S3"}) != bool(a.teacher_checkpoint):
            raise ValueError("Only source S2/S3 require --teacher-checkpoint")
        if (a.stage == "cod" and a.group != "S0") != bool(a.source_checkpoint):
            raise ValueError("Only cod S1/S2/S3 require --source-checkpoint")
    if a.teacher_checkpoint and a.stage == "cod":
        raise ValueError("COD stage must not receive a teacher checkpoint")

def prepare_data(a):
    old = json.loads(a.split_manifest.read_text(encoding="utf-8")) if a.split_manifest else None
    if old and old.get("split_seed", old.get("seed")) != a.split_seed:
        raise ValueError("split seed differs from manifest")
    names = ("vdt_train", "vdt_val", "cod_train", "cod_val")
    if old:
        for prefix in ("vdt", "cod"):
            train, val = old.get(prefix + "_train"), old.get(prefix + "_val")
            if not isinstance(train, list) or not isinstance(val, list) or not train or not val:
                raise ValueError("Original development manifest requires both nonempty partitions")
            pool = train + val
            if any(not isinstance(k, str) for k in pool) or len(set(pool)) != len(pool):
                raise ValueError("Duplicate/invalid IDs or overlap in supplied manifest")
        if not a.max_samples and a.model_variant != "tiny":
            if (len(old["vdt_train"]), len(old["vdt_val"])) != (838, 210):
                raise ValueError("Original source split must have 838/210 members")
            if Counter(k.split("/", 1)[0] for k in old["cod_train"] + old["cod_val"]) != {"COD10K": 3040, "CAMO": 1000}:
                raise ValueError("Original COD pool must have 3040 COD10K and 1000 CAMO members")
    splits = {name: [] for name in names}
    prefix = "vdt" if a.stage == "source" else "cod"
    items = shared._vdt_items(a.vdt_root, a.max_samples) if prefix == "vdt" else shared._cod_items(a.cod_root, a.max_samples)
    keys = [item.key for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate training IDs")
    if not a.max_samples:
        if prefix == "vdt" and len(items) != 1048:
            raise ValueError("Expected 1048 official VDT training images")
        if prefix == "cod" and Counter(k.split("/", 1)[0] for k in keys) != {"COD10K": 3040, "CAMO": 1000}:
            raise ValueError("Expected COD10K 3040 + CAMO 1000 training images")
    if prefix == "cod" and a.cod_train_mode == "full":
        if old and set(keys) != set(old["cod_train"] + old["cod_val"]):
            raise ValueError("Full training membership differs from original development pool")
        train, val = items, []
    else:
        train, val = shared._manifest_split(items, old, prefix, a.val_fraction, a.split_seed)
        if not val:
            raise ValueError("Empty stage validation set")
    if not train:
        raise ValueError("Empty stage training set")
    splits[prefix + "_train"], splits[prefix + "_val"] = train, val
    manifest = {"method": METHOD, "split_seed": a.split_seed, "cod_train_mode": a.cod_train_mode,
                **{name: list(old[name]) if old else [] for name in names}}
    for suffix, entries in (("train", train), ("val", val)):
        manifest[prefix + "_" + suffix] = [item.key for item in entries]
    return splits, manifest

def _common_contract(a, resources):
    return {"method": METHOD, **{k: getattr(a, k) for k in COMMON_FIELDS}, **resources,
            "source_sha256": {p: sha256_file(ROOT / p) for p in CODE_FILES}, "model_spec": TASK_MODEL_SPEC,
            "prediction_protocol": PREDICTION_PROTOCOL, "sensor_preprocessing": SENSOR_PREPROCESSING,
            "split_manifest_sha256": sha256_file(a.split_manifest) if a.split_manifest else None}

def stage_exposure(a):
    return {"optimization": ["VDT_train"] if a.stage == "source" else ["COD_train"],
            "selection": ["VDT_val"] if a.stage == "source" else [] if a.cod_train_mode == "full" else ["COD_val"],
            "teacher_in_stage": a.stage == "source" and a.group in {"S2", "S3"}, "source_batches_in_cod": 0}

def _contracts(a, manifest, resources, source_provenance=None):
    signature = None
    if a.stage == "source" and a.group in {"S2", "S3"}:
        signature, _ = v4._contracts(_teacher_args(a), manifest, resources)
    active = ("vdt_train", "vdt_val") if a.stage == "source" else ("cod_train", "cod_val")
    contract = {**_common_contract(a, resources), "stage": a.stage, "group": a.group, "epochs": a.epochs,
                "val_metrics_every": a.val_metrics_every, "cod_train_mode": a.cod_train_mode,
                **{key: manifest[key] if key in active else [] for key in ("vdt_train", "vdt_val", "cod_train", "cod_val")},
                "teacher_sha256": sha256_file(a.teacher_checkpoint) if a.teacher_checkpoint else None,
                "source_provenance": source_provenance, "stage_exposure": stage_exposure(a),
                "selection": "source_validation_gt_loss" if a.stage == "source" else "last_epoch" if a.cod_train_mode == "full" else "macro_S_alpha",
                "transfer_scope": "shared_only_exclude_decoders", "source_gt_weight": 1.,
                "cod_optimizer_reset": True, "cod_training_rng_reset": True}
    return signature, contract

def read_checkpoint(path, contract=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("method") != METHOD or not isinstance(state.get("contract"), dict):
        raise ValueError("Not a staged-v5 checkpoint")
    saved = state["contract"]
    if state.get("contract_sha256") != _digest(saved) or saved.get("method") != METHOD:
        raise ValueError("Invalid staged-v5 checkpoint contract hash")
    if state.get("stage") != saved.get("stage") or state.get("group") != saved.get("group"):
        raise ValueError("Checkpoint stage/group differs from its contract")
    if contract is not None and saved != contract:
        raise ValueError("Checkpoint contract mismatch; start a new output directory")
    return state

def _load_source_shared(model, a, manifest, resources):
    if a.stage != "cod" or a.group == "S0":
        return None
    path = Path(a.source_checkpoint)
    state = read_checkpoint(path)
    report = json.loads((path.parent / "comparison.json").read_text(encoding="utf-8"))
    contract = state["contract"]
    common = _common_contract(a, resources)
    if any(contract.get(key) != value for key, value in common.items()):
        raise ValueError("Source resources, initialization, source budget or training settings differ")
    if (state.get("stage") != "source" or state.get("group") != a.group or contract.get("cod_train") or contract.get("cod_val") or
            report.get("status") != "complete" or report.get("method") != METHOD or report.get("stage") != "source" or report.get("group") != a.group or
            report.get("protocol") != contract or report.get("contract_sha256") != state["contract_sha256"] or
            report.get("checkpoint_sha256") != sha256_file(path) or report.get("completed_epochs") != a.source_epochs or
            report.get("best_epoch") != state.get("epoch") or state.get("epoch") != state.get("best_epoch")):
        raise ValueError("Source checkpoint must be the completed matching source-stage selection")
    for container in (state.get("config", {}), report.get("config", {})):
        if any(container.get(key) != contract.get(key) for key in (*COMMON_FIELDS, "epochs", "val_metrics_every", "cod_train_mode")):
            raise ValueError("Source checkpoint/report configuration differs from its contract")
    generic = state_digest(model)
    for key in ("generic_initial_state_sha256", "initial_state_sha256"):
        if state.get(key) != generic or report.get(key) != generic:
            raise ValueError("Source and COD students must share the exact generic initialization")
    head = state_digest(model.decoders["cod"])
    for key in ("cod_decoder_before_sha256", "cod_decoder_after_sha256"):
        if state.get(key) != head or report.get(key) != head:
            raise ValueError("Source stage changed or mismatched the common COD head initialization")
    history = report.get("history", [])
    if [r.get("epoch") for r in history] != list(range(1, a.source_epochs + 1)):
        raise ValueError("Source report does not prove its complete fixed budget")
    if any(row.get("validation", {}).get("objective") != "supervised_gt_only" for row in history):
        raise ValueError("Source checkpoint selection must use GT supervision without KD")
    last_path = path.parent / "last.pt"
    completed = read_checkpoint(last_path, contract)
    if (completed.get("epoch") != a.source_epochs or completed.get("history") != history or
            completed.get("best_epoch") != state.get("epoch")):
        raise ValueError("Completed source checkpoint does not prove the reported fixed budget")
    del completed
    expected_epoch = min(history, key=lambda r: r["validation"]["loss"])["epoch"]
    if expected_epoch != state["epoch"] or state.get("history") != history[:state["epoch"]]:
        raise ValueError("Source selection is not minimum source GT validation loss")
    for key in ("vdt_train", "vdt_val"):
        if manifest[key] and contract.get(key) != manifest[key]:
            raise ValueError("Source checkpoint membership differs from original split manifest")
    current = model.state_dict()
    shared_keys = {key for key in current if not key.startswith("decoders.")}
    saved_keys = {key for key in state["model"] if not key.startswith("decoders.")}
    if shared_keys != saved_keys:
        raise ValueError("Source shared module keys differ from the COD student")
    for key in shared_keys:
        if current[key].shape != state["model"][key].shape:
            raise ValueError(f"Source shared tensor shape differs: {key}")
    before = state_digest(model.decoders["cod"])
    current.update({key: state["model"][key] for key in shared_keys})
    model.load_state_dict(current, strict=True)
    if state_digest(model.decoders["cod"]) != before:
        raise RuntimeError("Source transfer changed the fresh COD decoder")
    return {"checkpoint_sha256": sha256_file(path), "contract_sha256": state["contract_sha256"],
            "group": a.group, "epoch": state["epoch"], "completed_epochs": report["completed_epochs"],
            "contract": contract, "generic_initial_state_sha256": state.get("generic_initial_state_sha256"),
            "initial_state_sha256": state.get("initial_state_sha256"), "transfer_scope": "shared_only_exclude_decoders",
            "completed_checkpoint_sha256": sha256_file(last_path)}

def _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch):
    model.train()
    if teacher is not None:
        teacher.eval()
    for loader in loaders.values():
        loader.dataset.epoch = epoch
    primary = loaders["vdt_train" if a.stage == "source" else "cod_train"]
    totals, count, updates, skipped, consecutive, norm_sum = {}, 0, 0, 0, 0, 0.
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in primary:
        batch = shared._move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with infrastructure._autocast(a, device):
            prediction = model(batch["rgb"], task="sod" if a.stage == "source" else "cod")
            losses = supervised_loss(prediction, batch["mask"], boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
            total = losses["total"]
            logged = {("source_" if a.stage == "source" else "cod_") + key: value for key, value in losses.items()}
            if teacher is not None:
                if a.stage != "source":
                    raise RuntimeError("A teacher must never enter COD optimization")
                with torch.no_grad():
                    coals = teacher.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"], input_mode="rgb" if a.group == "S2" else "multimodal")
                kd = transfer_loss(prediction, coals["RTD"], batch["mask"], teacher_rgb=coals["R"], mode="task", temperature=a.temperature, feature_weight=a.feature_weight)
                total = total + a.kd_weight * kd["total"]
                logged.update({key: value for key, value in kd.items() if key != "total"})
        infrastructure._finite(total, "staged v5 total")
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
    return {**{key: value / count for key, value in totals.items()}, "steps": count, "optimizer_steps": updates,
            "amp_skipped_steps": skipped, "amp_scale": float(scaler.get_scale()), "grad_norm": norm_sum / updates,
            "primary_images": len(primary.dataset), "source_images": len(primary.dataset) if a.stage == "source" else 0,
            "cod_images": len(primary.dataset) if a.stage == "cod" else 0, "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}

@torch.no_grad()
def _validate_source(model, loader, device, a):
    model.eval()
    total, n = 0., 0
    for batch in loader:
        batch = shared._move(batch, device)
        with infrastructure._autocast(a, device):
            losses = supervised_loss(model(batch["rgb"], task="sod"), batch["mask"], boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
        infrastructure._finite(losses["total"], "source GT validation")
        total += float(losses["total"]) * len(batch["rgb"])
        n += len(batch["rgb"])
    return {"loss": total / n, "num_samples": n, "objective": "supervised_gt_only"}

def _fit(model, teacher, loaders, splits, a, device, signature, contract, report):
    optimizer = build_optimizer(model, a.lr, a.weight_decay, a.backbone_lr_scale)
    epochs = a.source_epochs if a.stage == "source" else a.epochs
    scheduler = build_scheduler(optimizer, "cosine", epochs, min(a.warmup_epochs, epochs - 1), .05)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
    start, best, best_epoch, history = 1, None, None, []
    metadata_keys = ("generic_initial_state_sha256", "initial_state_sha256", "cod_decoder_before_sha256", "cod_decoder_after_sha256", "source_provenance", "stage_exposure")
    if a.resume and (a.out / "last.pt").exists():
        checkpoint = read_checkpoint(a.out / "last.pt", contract)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        infrastructure._restore_rng(checkpoint["rng"], loaders)
        start, best, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_score"], checkpoint["best_epoch"]
        history = checkpoint["history"]
        report.update({key: checkpoint[key] for key in metadata_keys})
        selected_path = a.out / "best.pt"
        saved = read_checkpoint(selected_path, contract)["epoch"] if selected_path.exists() else None
        if saved != best_epoch:
            if checkpoint["epoch"] != best_epoch:
                raise RuntimeError("Selected checkpoint missing; cannot reconstruct from last")
            _save_pt(selected_path, checkpoint)
    for epoch in range(start, epochs + 1):
        row = {"epoch": epoch, "lr": [group["lr"] for group in optimizer.param_groups]}
        row["train"] = _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch)
        score = None
        if a.stage == "source":
            row["validation"] = _validate_source(model, loaders["vdt_val"], device, a)
            score = -row["validation"]["loss"]
        elif a.cod_train_mode == "full":
            score = float(epoch)
        elif epoch % a.val_metrics_every == 0 or epoch == epochs:
            row["validation"] = v4._validate_cod(model, splits["cod_val"], device, a)
            score = row["validation"]["macro"]["S_alpha"]
        selected = score is not None and (best is None or score > best)
        if selected:
            best, best_epoch = score, epoch
        scheduler.step()
        history.append(row)
        checkpoint = {"method": METHOD, "stage": a.stage, "group": a.group, "contract": contract,
            "contract_sha256": _digest(contract), "teacher_signature": signature, "config": _json(a),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best, "best_epoch": best_epoch,
            "history": history, "rng": infrastructure._rng_state(loaders), **{key: report[key] for key in metadata_keys}}
        _save_pt(a.out / "last.pt", checkpoint)
        if selected:
            _save_pt(a.out / "best.pt", checkpoint)
        report.update(completed_epochs=epoch, best_epoch=best_epoch, best_score=best, history=history)
        _save_json(a.out / "comparison.json", report)
        _save_json(a.out / "history.json", history)
        print(f"[{a.stage}/{a.group}] {epoch}/{epochs} loss={row['train']['total']:.5f} best={best_epoch} updates={row['train']['optimizer_steps']} time={row['train']['seconds']:.1f}s", flush=True)
    selected = read_checkpoint(a.out / "best.pt", contract)
    model.load_state_dict(selected["model"], strict=True)
    report.update(completed_epochs=epochs, best_epoch=best_epoch, best_score=best, history=history, checkpoint_sha256=sha256_file(a.out / "best.pt"))

def _evaluate_existing(cli):
    from tools.evaluate_triprd_staged import validate_run_report
    report = validate_run_report(cli.out / "comparison.json")
    state = read_checkpoint(cli.out / "best.pt")
    if state["contract"]["source_sha256"] != {p: sha256_file(ROOT / p) for p in CODE_FILES}:
        raise ValueError("Evaluation source differs from trained v5 contract")
    a = argparse.Namespace(**state["config"])
    a.out, a.device, a.cod_root = cli.out, cli.device, cli.cod_root
    a.dino_repo = cli.dino_repo or a.dino_repo
    if v4.dino_source_digest(a.dino_repo) != state["contract"]["dino_source_sha256"]:
        raise ValueError("Evaluation DINO source differs from trained v5 contract")
    model = TaskStudent(**model_options(a, pretrained=False)).to(torch.device(a.device))
    model.load_state_dict(state["model"], strict=True)
    report["test_status"] = "running"
    _save_json(a.out / "comparison.json", report)
    report["test"] = _formal_tests(model, a)
    report["test_status"] = "complete"
    _save_json(a.out / "comparison.json", report)
    return 0

def run(a):
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
        resources["dino_source_sha256"] = v4.dino_source_digest(a.dino_repo)
    splits, manifest = prepare_data(a)
    with run_lock(a.out):
        previous_path = a.out / "comparison.json"
        if a.prepare_only:
            target = a.out / "split_manifest.json"
            if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Refusing to overwrite a different manifest")
            _save_json(target, manifest)
            return 0
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
        if previous and not a.resume:
            raise ValueError("Existing run requires --resume and an identical contract")
        if not previous and any(p.name not in {"task_run.lock", "run.log", "launch.json"} for p in a.out.iterdir()):
            raise FileExistsError("Nonempty output directory is not a resumable v5 run")
        shared.seed_everything(a.train_seed)
        model = TaskStudent(**model_options(a)).to(device)
        generic = state_digest(model)
        decoder_before = state_digest(model.decoders["cod"])
        provenance = _load_source_shared(model, a, manifest, resources)
        decoder_after = state_digest(model.decoders["cod"])
        if decoder_before != decoder_after:
            raise RuntimeError("COD decoder initialization changed during transfer")
        signature, contract = _contracts(a, manifest, resources, provenance)
        if previous:
            if previous.get("contract_sha256") != _digest(contract):
                raise ValueError("Existing run requires an identical contract")
            if previous.get("status") == "complete":
                checkpoint = read_checkpoint(a.out / "best.pt", contract)
                if sha256_file(a.out / "best.pt") != previous["checkpoint_sha256"] or checkpoint["best_epoch"] != previous["best_epoch"]:
                    raise ValueError("Completed checkpoint differs from report")
                if a.test_after_training and previous.get("test_status") != "complete":
                    return _evaluate_existing(a)
                print(f"Verified completed run: {a.out}")
                return 0
        teacher = v4._load_teacher(_teacher_args(a), signature, device) if a.stage == "source" and a.group in {"S2", "S3"} else None
        # Reset after construction and transfer: no source RNG, optimizer state
        # or teacher construction can affect the COD training/data RNG.
        shared.seed_everything(a.train_seed)
        loaders = _loaders(a, splits, device)
        benchmark = a.stage == "cod" and a.cod_train_mode == "full" and a.model_variant != "tiny" and not a.max_samples
        report = {"method": METHOD, "stage": a.stage, "group": a.group, "status": "running", "config": _json(a),
            "contract_sha256": _digest(contract), "protocol": contract, "prediction_protocol": PREDICTION_PROTOCOL,
            "formal": benchmark, "result_scope": "benchmark" if benchmark else "development", "sample_counts": {key: len(value) for key, value in splits.items()},
            "selection_metric": "source_validation_loss" if a.stage == "source" else "last_epoch" if a.cod_train_mode == "full" else "macro_S_alpha",
            "test_status": "skipped", "test": {}, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "started_at": previous.get("started_at") if previous else datetime.now(timezone.utc).isoformat(),
            "generic_initial_state_sha256": generic, "initial_state_sha256": state_digest(model),
            "cod_decoder_before_sha256": decoder_before, "cod_decoder_after_sha256": decoder_after,
            "source_provenance": provenance, "stage_exposure": stage_exposure(a),
            "parameters": {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
        _save_json(a.out / "split_manifest.json", manifest)
        _save_json(previous_path, report)
        _fit(model, teacher, loaders, splits, a, device, signature, contract, report)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        _save_json(previous_path, report)
        if a.test_after_training:
            from tools.evaluate_triprd_staged import validate_run_report
            validate_run_report(previous_path)
            report["test_status"] = "running"
            _save_json(previous_path, report)
            report["test"] = _formal_tests(model, a)
            report["test_status"] = "complete"
            _save_json(previous_path, report)
    return 0

def main(argv=None):
    return run(parse_args(argv))

from tools.run_triprd_task import prediction_image, evaluate_dataset_strict
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
            metrics = evaluate_dataset_strict(index, a.out / "predictions", dataset, "TriPRD-Staged-v5")
            coverage = metrics.pop("coverage")
            results[dataset] = {"official": metrics, "coverage": coverage, "count": count}
            _save_json(a.out / "test_metrics.json", results)
        finally:
            index.close()
    return results


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
