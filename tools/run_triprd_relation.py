#!/usr/bin/env python3
"""Matched TriPRD relation-transfer experiments, independent of legacy A/B/C/D.

A: identical RGB student without VDT; B: VDT RGB supervision;
C: capacity-matched RGB teacher relation KD; D: real RGB/T/D relation KD.
Only the relation branch crosses the VDT -> COD boundary. Screening never
loads test data by default. Full-pool retraining uses a fixed final epoch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.triprd_relation import RelationLoss, RelationStudent, RelationTeacher
from tools import run_triprd_ab as shared
from tools.triprd_training import build_optimizer, build_scheduler, sha256_file
from tools.triprd_relation_training import (
    export_relation_checkpoint, load_full_checkpoint,
    load_relation_checkpoint, state_digest,
)

METHOD = "triprd_relation_v1"
TEST_COUNTS = {"COD10K": 2026, "CAMO": 250, "NC4K": 4121, "CHAMELEON": 76}
DEFINITIONS = {
    "A": "Same relation-aware RGB student, no VDT training",
    "B": "Frozen backbone, VDT RGB supervision, relation-branch transfer",
    "C": "Frozen backbone, capacity-matched RGB teacher relation KD, branch transfer",
    "D": "Frozen backbone, real RGB/T/D teacher relation KD, branch transfer",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--groups", default="A,B,C,D")
    p.add_argument("--out", default="results/optimization/relation_manual")
    p.add_argument("--vdt-root", default="datasets/VDT-2048")
    p.add_argument("--cod-root", default="datasets/COD_benchmark")
    p.add_argument("--model-variant", choices=("tiny", "resnet50"), default="resnet50")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--base-channels", type=int, default=256)
    p.add_argument("--image-size", type=int, default=352)
    p.add_argument("--relation-dim", type=int, default=32)
    p.add_argument("--relation-kernel", type=int, default=3)
    p.add_argument("--relation-weight", type=float, default=1.0)
    p.add_argument("--relation-temperature", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--vdt-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=20261012)
    p.add_argument("--split-seed", type=int, default=20261012)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-metrics-every", type=int, default=5)
    p.add_argument("--selection-metric", choices=("S_alpha",), default="S_alpha")
    p.add_argument("--cod-train-mode", choices=("split", "full"), default="split")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cod-encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--cod-scheduler", choices=("none", "cosine"), default="cosine")
    p.add_argument("--cod-warmup-epochs", type=int, default=5)
    p.add_argument("--cod-min-lr-ratio", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--teacher-checkpoint", action="append", default=[], metavar="GROUP=PATH")
    p.add_argument("--relation-checkpoint", action="append", default=[], metavar="GROUP=PATH")
    p.add_argument("--cod-checkpoint", action="append", default=[], metavar="GROUP=PATH")
    p.add_argument("--skip-test", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--test-datasets", default="COD10K,CAMO,CHAMELEON,NC4K")
    p.add_argument("--max-test-samples", type=int, default=0)
    return p.parse_args(argv)


def _validate_args(args):
    groups = [x.strip().upper() for x in args.groups.split(",")]
    if not groups or set(groups) - set(DEFINITIONS) or len(groups) != len(set(groups)):
        raise ValueError("--groups must be unique entries from A,B,C,D")
    datasets = [x.strip().upper() for x in args.test_datasets.split(",")]
    if set(datasets) - set(TEST_COUNTS) or len(datasets) != len(set(datasets)):
        raise ValueError("Unknown or duplicate test dataset")
    if min(args.epochs, args.vdt_epochs, args.batch_size, args.base_channels,
           args.relation_dim, args.val_metrics_every) < 1 or args.image_size < 32:
        raise ValueError("Epochs, batch, widths and validation interval must be positive; image size >= 32")
    if args.relation_kernel < 3 or args.relation_kernel % 2 != 1:
        raise ValueError("Relation kernel must be odd and >= 3")
    if args.max_samples < 0 or args.max_test_samples < 0 or not 0 < args.val_fraction < 1:
        raise ValueError("Nonnegative sample limits and 0 < val_fraction < 1 required")
    if any(not 0 <= seed < 2**32 for seed in (args.train_seed, args.split_seed)):
        raise ValueError("Seeds must be in [0, 2**32)")
    for name in ("learning_rate", "cod_encoder_lr_scale", "relation_temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "relation_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= args.cod_warmup_epochs < args.epochs or not 0 <= args.cod_min_lr_ratio <= 1:
        raise ValueError("Require warmup < epochs and min LR ratio in [0,1]")
    if args.cod_scheduler == "none" and args.cod_warmup_epochs:
        raise ValueError("Warmup requires cosine")
    if args.model_variant == "tiny" and args.pretrained:
        raise ValueError("Tiny is a synthetic pipeline check; use --no-pretrained")
    if args.cod_train_mode == "full" and args.max_samples:
        raise ValueError("Full COD retraining requires --max-samples 0")
    relations = shared._checkpoint_map(args.relation_checkpoint, set(groups) - {"A"})
    teachers = shared._checkpoint_map(args.teacher_checkpoint, set(groups) & {"C", "D"})
    cod = shared._checkpoint_map(args.cod_checkpoint, set(groups))
    if set(teachers) & set(relations):
        raise ValueError("Teacher reuse is unused when the same group's relation branch is reused")
    if args.eval_only:
        if args.skip_test or set(cod) != set(groups) or relations or teachers:
            raise ValueError("Eval-only requires --no-skip-test, every COD checkpoint, and no source checkpoints")
    elif cod:
        raise ValueError("COD checkpoints are only accepted with --eval-only")
    return groups, datasets, relations, teachers, cod


def _full_cod_pool(items):
    counts = Counter(x.key.split("/", 1)[0] for x in items)
    if counts != {"COD10K": 3040, "CAMO": 1000} or len({x.key for x in items}) != 4040:
        raise ValueError(f"Full COD training requires 3040 COD10K + 1000 CAMO unique samples, got {dict(counts)}")
    return items


def _fit_fixed_cod(model, loader, device, criterion, out_dir, group, args, protocol):
    """No COD validation or model selection; save the prespecified final epoch."""
    shared.seed_everything(args.train_seed)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay, args.cod_encoder_lr_scale)
    scheduler = build_scheduler(optimizer, args.cod_scheduler, args.epochs,
                                args.cod_warmup_epochs, args.cod_min_lr_ratio)
    directory = out_dir / group / "cod"
    directory.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
        train = shared._run_epoch(model, loader, device, criterion, optimizer=optimizer)
        history.append({"epoch": epoch, "train": train, "learning_rates": rates,
                        "seconds": time.time() - started, "stage": f"{group}/cod"})
        with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(history[-1]) + "\n")
        if scheduler is not None:
            scheduler.step()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "stage": f"{group}/cod", "seed": args.train_seed,
                 "selection_metric": "fixed_last_epoch", "protocol": protocol}
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, directory / "last.pt")
        print(f"[{group}/cod full] {epoch:03d}/{args.epochs} loss={train['total']:.4f} "
              f"{history[-1]['seconds']:.0f}s; no validation/selection", flush=True)
    shared._save_json(directory / "history.json", history)
    return directory / "last.pt", {"selection_metric": "fixed_last_epoch", "selected_epoch": args.epochs,
        "epochs": args.epochs, "history": history, "validation_used": False,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def _evaluate_complete(model, args, dataset, output, device):
    """Check dataset and exported prediction coverage before official scoring."""
    if shared._index_for_dataset is None:
        raise RuntimeError("Official evaluator dependencies are unavailable")
    index = shared._index_for_dataset(Path(args.cod_root), dataset)
    try:
        samples = list(index.samples(args.max_test_samples or None))
        ids = [s.sample_id for s in samples]
        if not ids or len({x.casefold() for x in ids}) != len(ids):
            raise ValueError(f"Empty or duplicate {dataset} test IDs")
        if any(s.image is None or s.mask is None for s in samples):
            raise ValueError(f"Incomplete RGB/GT pairs in {dataset}")
        if not args.max_test_samples and len(ids) != TEST_COUNTS[dataset]:
            raise ValueError(f"Incomplete {dataset} test coverage: {len(ids)} != {TEST_COUNTS[dataset]}")
    finally:
        index.close()
    quick = shared._predict_test(model, args.cod_root, dataset, output,
                                 args.image_size, device, args.max_test_samples)
    actual = {p.stem for p in (output / "predictions" / dataset).glob("*.png")}
    if actual != set(ids) or quick["count"] != len(ids):
        raise ValueError(f"Prediction IDs/count do not cover {dataset} test samples")
    official = shared._official_metrics(args.cod_root, output / "predictions", dataset, args.max_test_samples)
    if official.get("num_samples") != len(ids) or not all(
            isinstance(official.get(k), (int, float)) and math.isfinite(official[k])
            for k in ("S_alpha", "F_beta_w", "E_phi", "MAE")):
        raise ValueError(f"Official evaluation failed or count mismatched: {official}")
    official["model"] = "TriPRD-Relation"
    return {**quick, "official": official, "coverage": {
        "verified_samples": len(ids), "expected_full_samples": TEST_COUNTS[dataset],
        "limited": bool(args.max_test_samples)}}


def main(argv=None):
    args = parse_args(argv)
    groups, datasets, reused_relations, reused_teachers, reused_cod = _validate_args(args)
    # Fail before expensive training if formal metrics dependencies are absent.
    from evaluation.validation import CODValidationMetrics  # noqa: F401
    shared.NORMALIZE_RGB = args.model_variant == "resnet50" and args.pretrained
    model_protocol = {"method": METHOD, "model_variant": args.model_variant,
        "base_channels": args.base_channels, "image_size": args.image_size,
        "rgb_normalization": "ImageNet" if shared.NORMALIZE_RGB else "[0,1]",
        "relation_dim": args.relation_dim, "relation_kernel": args.relation_kernel}
    options = dict(model_variant=args.model_variant, base_channels=args.base_channels,
                   relation_dim=args.relation_dim, relation_kernel=args.relation_kernel)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    out = Path(args.out).expanduser().resolve()
    if out.exists() and any(p.name not in {"run.log", "launch.json"} for p in out.iterdir()):
        raise FileExistsError(f"Existing experiment output: {out}")
    manifest_source = json.loads(args.split_manifest.read_text(encoding="utf-8")) if args.split_manifest else None
    if manifest_source and manifest_source.get("split_seed", manifest_source.get("seed")) != args.split_seed:
        raise ValueError("Split seed differs from supplied manifest")
    needs_vdt = not args.eval_only and any(g != "A" and g not in reused_relations for g in groups)
    splits = {k: [] for k in ("vdt_train", "vdt_val", "cod_train", "cod_val")}
    if needs_vdt:
        splits["vdt_train"], splits["vdt_val"] = shared._manifest_split(
            shared._vdt_items(args.vdt_root, args.max_samples), manifest_source,
            "vdt", args.val_fraction, args.split_seed)
    if not args.eval_only:
        items = shared._cod_items(args.cod_root, args.max_samples)
        if args.cod_train_mode == "full":
            # Validate a supplied screening manifest, but train on its entire pool.
            if manifest_source:
                shared._manifest_split(items, manifest_source, "cod", args.val_fraction, args.split_seed)
            splits["cod_train"] = _full_cod_pool(items)
        else:
            splits["cod_train"], splits["cod_val"] = shared._manifest_split(
                items, manifest_source, "cod", args.val_fraction, args.split_seed)
        needed = ["cod_train"] + (["cod_val"] if args.cod_train_mode == "split" else [])
        if needs_vdt:
            needed += ["vdt_train", "vdt_val"]
        if any(not splits[k] for k in needed):
            raise ValueError("Empty required training/validation split; increase sample limits")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"split_seed": args.split_seed, "train_seed": args.train_seed,
        "val_fraction": args.val_fraction, "cod_train_mode": args.cod_train_mode,
        **{k: [x.key for x in v] for k, v in splits.items()},
        "source_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None}
    shared._save_json(out / "split_manifest.json", manifest)
    loaders = {k: DataLoader(shared.PairDataset(v, args.image_size, augment=k.endswith("train")),
                            batch_size=args.batch_size, shuffle=k.endswith("train"), num_workers=0)
               for k, v in splits.items() if v}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    sources = ("models/triprd_relation.py", "models/triprd.py", "models/triprd_full.py",
               "tools/run_triprd_relation.py", "tools/triprd_relation_training.py", "tools/run_triprd_ab.py",
               "tools/triprd_training.py", "evaluation/validation.py", "evaluation/cod_metrics.py",
               "evaluation/camoformer_metrics.py", "tools/evaluate_cod.py", "datasets/cod_dataset.py", "datasets/vdt2048.py")
    protocol = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, **model_protocol,
        "sample_counts": {k: len(v) for k, v in splits.items()},
        "split_manifest_sha256": sha256_file(out / "split_manifest.json"),
        "code_commit": commit, "code_dirty": dirty,
        "source_sha256": {name: sha256_file(ROOT / name) for name in sources},
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "transfer_scope": "relation_branch_only", "source_backbone_frozen": True,
        "source_backbone_bn_frozen": True, "source_head_transferred": False,
        "teacher_modalities": {"C": "RGB/RGB/RGB", "D": "RGB/thermal/depth"},
        "thermal_depth_preprocessing": "[0,1], repeated to 3 channels; RGB uses rgb_normalization",
        "validation_source_weights": "equal", "validation_prediction": "uint8_then_PIL_bilinear_to_original_GT",
        "optimizer": "AdamW", "train_augmentation": "horizontal_flip_p=0.5"}
    comparison = {"protocol": protocol, "groups": {}, "test": {}, "teachers": {},
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "test_status": "skipped" if args.skip_test else "pending"}
    shared._save_json(out / "comparison.json", comparison)
    print(f"TriPRD relation | {device} | groups={groups} | mode={args.cod_train_mode}", flush=True)
    try:
        for group in groups:
            teacher = None
            summary = {"group": group, "definition": DEFINITIONS[group]}
            if not args.eval_only and group in "CD" and group not in reused_relations:
                kind = "rgb" if group == "C" else "multimodal"
                teacher_protocol = {**protocol, "teacher_kind": kind}
                expected_teacher = {**model_protocol, "teacher_kind": kind}
                shared.seed_everything(args.train_seed)
                teacher = RelationTeacher(**options, pretrained=args.pretrained and group not in reused_teachers,
                                          teacher_kind=kind)
                if group in reused_teachers:
                    path = reused_teachers[group]
                    teacher_summary = load_full_checkpoint(teacher, path, expected_stage=f"{group}/teacher",
                                                           expected_protocol=expected_teacher)
                else:
                    path, teacher_summary = shared._fit_stage(f"{group}/teacher", teacher,
                        loaders["vdt_train"], loaders["vdt_val"], device, RelationLoss(), out,
                        args.vdt_epochs, args.learning_rate, args.weight_decay, args.train_seed,
                        use_modalities=True, checkpoint_metadata=teacher_protocol)
                    load_full_checkpoint(teacher, path, expected_stage=f"{group}/teacher", expected_protocol=expected_teacher)
                teacher_summary.update(checkpoint=str(path), sha256=sha256_file(path),
                    parameters=sum(p.numel() for p in teacher.parameters()), teacher_kind=kind)
                comparison["teachers"][group] = teacher_summary
                shared._save_json(out / "comparison.json", comparison)
            shared.seed_everything(args.train_seed)
            student = RelationStudent(**options, pretrained=args.pretrained and not args.eval_only)
            summary["parameters"] = sum(p.numel() for p in student.parameters())
            summary["relation_parameters"] = sum(p.numel() for p in student.relation_branch.parameters())
            if args.eval_only:
                cod_path = reused_cod[group]
                summary["cod_finetune"] = load_full_checkpoint(student, cod_path,
                    expected_stage=f"{group}/cod", expected_protocol=model_protocol)
            else:
                initial = {k: state_digest(getattr(student, k)) for k in ("encoder", "decoder", "heads", "relation_branch")}
                summary["initial_state_sha256"] = initial
                if group != "A":
                    if group in reused_relations:
                        branch_path = reused_relations[group]
                        source_summary = {"reused": True}
                    else:
                        student.freeze_backbone()
                        source_path, source_summary = shared._fit_stage(f"{group}/vdt", student,
                            loaders["vdt_train"], loaders["vdt_val"], device,
                            RelationLoss(relation_weight=args.relation_weight if group in "CD" else 0.0,
                                         relation_temperature=args.relation_temperature),
                            out, args.vdt_epochs, args.learning_rate, args.weight_decay, args.train_seed,
                            teacher=teacher, use_modalities=group in "CD", checkpoint_metadata=protocol)
                        if state_digest(student.encoder) != initial["encoder"]:
                            raise RuntimeError("Source backbone weights/BN buffers changed despite freezing")
                        source_summary["frozen_encoder_verified"] = True
                        branch_path = out / group / "vdt" / "relation.pt"
                        export_relation_checkpoint(source_path, branch_path, expected_protocol=model_protocol)
                        # New COD decoder/head/backbone have exactly A's initialization.
                        del student
                        shared.seed_everything(args.train_seed)
                        student = RelationStudent(**options, pretrained=args.pretrained)
                    transfer = load_relation_checkpoint(student, branch_path,
                        expected_stage=f"{group}/vdt", expected_protocol=model_protocol)
                    for name in ("encoder", "decoder", "heads"):
                        if state_digest(getattr(student, name)) != initial[name]:
                            raise RuntimeError(f"Relation transfer changed {name}")
                    source_summary.update(transfer=transfer, checkpoint=str(branch_path),
                                          transfer_scope="relation_branch_only")
                    summary["vdt_pretrain"] = source_summary
                summary["cod_initial_state_sha256"] = {
                    k: state_digest(getattr(student, k)) for k in ("encoder", "decoder", "heads", "relation_branch")}
                if teacher is not None:
                    teacher.cpu()
                    del teacher
                if args.cod_train_mode == "full":
                    cod_path, fit = _fit_fixed_cod(student, loaders["cod_train"], device,
                                                  RelationLoss(), out, group, args, protocol)
                else:
                    cod_path, fit = shared._fit_stage(f"{group}/cod", student,
                        loaders["cod_train"], loaders["cod_val"], device, RelationLoss(), out,
                        args.epochs, args.learning_rate, args.weight_decay, args.train_seed,
                        encoder_lr_scale=args.cod_encoder_lr_scale, scheduler_name=args.cod_scheduler,
                        warmup_epochs=args.cod_warmup_epochs, min_lr_ratio=args.cod_min_lr_ratio,
                        original_val_items=splits["cod_val"], image_size=args.image_size,
                        selection_metric="S_alpha", val_metrics_every=args.val_metrics_every,
                        checkpoint_metadata=protocol)
                summary["cod_finetune"] = fit
                load_full_checkpoint(student, cod_path, expected_stage=f"{group}/cod", expected_protocol=model_protocol)
            student.to(device)
            summary.update(checkpoint=str(cod_path), checkpoint_sha256=sha256_file(cod_path))
            comparison["groups"][group] = summary
            shared._save_json(out / "comparison.json", comparison)
            if not args.skip_test:
                comparison["test"][group] = {}
                for dataset in datasets:
                    print(f"[{group}/test] {dataset}: verify coverage, export RGB predictions, score", flush=True)
                    comparison["test"][group][dataset] = _evaluate_complete(student, args, dataset, out / group, device)
                    shared._save_json(out / "comparison.json", comparison)
            del student
        comparison.update(status="complete", test_status="skipped" if args.skip_test else "complete")
    except Exception as exc:
        comparison.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if not args.skip_test:
            comparison["test_status"] = "incomplete"
        raise
    finally:
        comparison["finished_at"] = datetime.now(timezone.utc).isoformat()
        shared._save_json(out / "comparison.json", comparison)
    print(f"Run finished: {out / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
