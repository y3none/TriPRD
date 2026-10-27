#!/usr/bin/env python3
"""Plan or execute matched RGB-only students with frozen complete structure-function transfer (v2)."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_triprd_optimization import execute_plan, gpu_list, positive_int, validate_plan


GROUPS = (
    ("A", "A_no_source"),
    ("B", "B_mask_structure"),
    ("C", "C_direct_observed"),
    ("D", "D_rgb_teacher"),
    ("E", "E_multimodal_teacher"),
)


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus", type=gpu_list, default="0,1,2,3")
    parser.add_argument("--output-root", type=Path, default=Path("results/optimization"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", "--train-seed", dest="seed", type=int, default=20261012)
    parser.add_argument("--split-seed", type=int, default=20261012)
    parser.add_argument("--split-manifest", type=Path,
                        help="Frozen VDT and COD split manifest; a COD-only manifest is insufficient")
    parser.add_argument("--vdt-root", type=Path, default=Path("datasets/VDT-2048"))
    parser.add_argument("--cod-root", type=Path, default=Path("datasets/COD_benchmark"))
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--vdt-epochs", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--model-variant", choices=("tiny", "resnet50"), default="resnet50")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=positive_int, default=352)
    parser.add_argument("--base-channels", type=positive_int, default=256)
    parser.add_argument("--relation-dim", type=positive_int, default=32)
    parser.add_argument("--relation-kernel", type=positive_int, default=3)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--structure-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cod-encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--intervention-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--groups", default="A,B,C,D,E")
    parser.add_argument("--max-samples", type=nonnegative_int, default=0)
    parser.add_argument("--val-metrics-every", type=positive_int, default=5)
    parser.add_argument("--cod-train-mode", choices=("split", "full"), default="split")
    parser.add_argument("--execute", action="store_true", help="Start jobs; omission prints a plan without writing files")
    return parser.parse_args(argv)


def build_plan(args: argparse.Namespace) -> dict:
    if any(not 0 <= seed < 2 ** 32 for seed in (args.seed, args.split_seed)):
        raise ValueError("Training and split seeds must be in [0, 2**32)")
    if args.relation_kernel < 3 or args.relation_kernel % 2 != 1:
        raise ValueError("--relation-kernel must be odd and at least 3")
    groups = args.groups.upper().split(",")
    if not groups or len(groups) != len(set(groups)) or set(groups) - dict(GROUPS).keys():
        raise ValueError("--groups must be unique entries A,B,C,D,E")
    for name in ("learning_rate", "cod_encoder_lr_scale", "distillation_weight", "semantic_weight", "structure_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "aux_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.cod_train_mode == "full" and args.max_samples != 0:
        raise ValueError("Full COD training requires --max-samples 0")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.model_variant == "tiny" and args.pretrained:
        raise ValueError("Tiny is a synthetic pipeline check; use --no-pretrained")
    run_dir = args.output_root.expanduser().resolve() / "structure_v2" / f"seed-{args.seed}"
    manifest = args.split_manifest.expanduser().resolve() if args.split_manifest else None
    common = [
        args.python, "-u", str(ROOT / "tools" / "run_triprd_structure.py"),
        "--vdt-root", str(args.vdt_root.expanduser().resolve()),
        "--cod-root", str(args.cod_root.expanduser().resolve()),
        "--model-variant", args.model_variant,
        "--pretrained" if args.pretrained else "--no-pretrained",
        "--image-size", str(args.image_size), "--base-channels", str(args.base_channels),
        "--batch-size", str(args.batch_size),
        "--relation-dim", str(args.relation_dim), "--relation-kernel", str(args.relation_kernel),
        "--distillation-weight", str(args.distillation_weight),
        "--aux-weight", str(args.aux_weight), "--semantic-weight", str(args.semantic_weight),
        "--structure-weight", str(args.structure_weight),
        "--intervention-eval" if args.intervention_eval else "--no-intervention-eval",
        "--epochs", str(args.epochs), "--vdt-epochs", str(args.vdt_epochs),
        "--train-seed", str(args.seed), "--split-seed", str(args.split_seed),
        "--max-samples", str(args.max_samples), "--device", "cuda",
        "--selection-metric", "S_alpha", "--val-metrics-every", str(args.val_metrics_every),
        "--cod-train-mode", args.cod_train_mode,
        "--cod-encoder-lr-scale", str(args.cod_encoder_lr_scale), "--cod-scheduler", "cosine",
        "--cod-warmup-epochs", str(min(5, args.epochs - 1)), "--cod-min-lr-ratio", "0.05",
        "--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay), "--skip-test",
    ]
    if manifest is not None:
        common += ["--split-manifest", str(manifest)]
    jobs = []
    for group, name in GROUPS:
        if group not in groups:
            continue
        output = run_dir / name
        jobs.append({
            "name": name, "gpu": args.gpus[len(jobs) % len(args.gpus)], "group": group,
            "train_seed": args.seed, "split_seed": args.split_seed,
            "transfer_scope": "none" if group == "A" else "complete_structure_function",
            "teacher_inputs": "RGB/RGB/RGB" if group == "D" else "RGB/T/D" if group == "E" else None,
            "output_dir": str(output),
            "command": common + ["--groups", group, "--out", str(output)],
            "required_checkpoints": [],
        })
    plan = {"phase": "structure_v2", "run_dir": str(run_dir), "gpus": args.gpus,
            "cod_train_mode": args.cod_train_mode,
            "selection_rule": "last_epoch" if args.cod_train_mode == "full" else "S_alpha",
            "jobs": jobs, "required_files": [str(manifest)] if manifest is not None else []}
    validate_plan(plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.execute:
            return execute_plan(plan)
        for job in plan["jobs"]:
            command = subprocess.list2cmdline(job["command"]) if os.name == "nt" else shlex.join(job["command"])
            job["display_command"] = f"CUDA_VISIBLE_DEVICES={job['gpu']} {command}"
        print(json.dumps({"mode": "dry-run", **plan}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
