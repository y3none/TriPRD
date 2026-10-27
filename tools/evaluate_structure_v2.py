#!/usr/bin/env python3
"""Evaluate already selected structure-v2 checkpoints on all four COD test sets.

Run from the training environment, e.g.:
    CUDA_VISIBLE_DEVICES=0 python tools/evaluate_structure_v2.py
    python tools/evaluate_structure_v2.py --dry-run

The runner preserves checkpoint inference; the evaluator uses the vendored
CamoFormer metric classes. The paper's alpha-E column is ADAPTIVE E, while
mean E is recorded separately. No training or test-based selection occurs.
This orchestration module only imports the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GROUP_DIRS = {
    "A": "A_no_source", "B": "B_mask_structure", "C": "C_direct_observed",
    "D": "D_rgb_teacher", "E": "E_multimodal_teacher",
}
DATASET_COUNTS = {"NC4K": 4121, "COD10K": 2026, "CAMO": 250, "CHAMELEON": 76}
TABLE_FIELDS = ("S_alpha", "E_phi_adaptive", "F_beta_w", "MAE")
MODEL_FIELDS = ("model_variant", "pretrained", "base_channels", "image_size",
                "relation_dim", "relation_kernel", "train_seed", "split_seed")
SOURCE_NOTES = {
    "A": "No VDT source training",
    "B": "VDT source RGB and masks",
    "C": "VDT source RGB/masks with observed thermal/depth edge supervision",
    "D": "VDT source with observed thermal/depth labels and RGB/RGB/RGB teacher",
    "E": "VDT source with observed thermal/depth labels and RGB/thermal/depth teacher",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path,
                   default=ROOT / "results/optimization/structure_v2/seed-20261012")
    p.add_argument("--out", type=Path, help="new/empty output directory; default is a separate UTC timestamp directory")
    p.add_argument("--cod-root", type=Path, default=ROOT / "datasets/COD_benchmark")
    p.add_argument("--groups", default="A,B,C,D,E")
    p.add_argument("--device", default="cuda", help="runner device; use CUDA_VISIBLE_DEVICES to select a GPU")
    p.add_argument("--dry-run", action="store_true",
                   help="validate metadata and print the command; tolerate missing checkpoints; write nothing")
    return p.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _positive_integer(value, label, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{label} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _load_source(run_dir, group, *, dry_run):
    directory = run_dir / GROUP_DIRS[group]
    source_path = directory / "comparison.json"
    source = _read_json(source_path)
    protocol = source["protocol"]
    if source.get("status") != "complete" or protocol.get("method") != "triprd_structure_v2":
        raise ValueError(f"{source_path}: expected completed structure-v2 training")
    summary = source["groups"][group]
    config = {key: protocol[key] for key in MODEL_FIELDS}
    if type(config["pretrained"]) is not bool:
        raise ValueError(f"{source_path}: pretrained must be a boolean")
    normalization = "ImageNet" if config["model_variant"] == "resnet50" and config["pretrained"] else "[0,1]"
    if protocol.get("rgb_normalization") != normalization:
        raise ValueError(f"{source_path}: pretrained/model variant disagrees with RGB normalization")
    counts = protocol["sample_counts"]
    cod_train = _positive_integer(counts["cod_train"], f"{group} COD train count")
    cod_val = _positive_integer(counts["cod_val"], f"{group} COD val count", allow_zero=True)
    fit = summary["cod_finetune"]
    epoch = _positive_integer(fit["best_epoch"], f"{group} selected epoch")
    # Resolve the known queue layout locally; original records can contain a
    # different machine's absolute path and must not silently select that file.
    checkpoint = (directory / group / "cod/best.pt").resolve()
    expected_hash = summary.get("checkpoint_sha256")
    if expected_hash is not None and (not isinstance(expected_hash, str) or
            len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash.lower())):
        raise ValueError(f"{source_path}: malformed checkpoint_sha256")
    if not checkpoint.is_file():
        if not dry_run:
            raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
        actual_hash = None
        checkpoint_status = "MISSING (dry-run only; evaluation cannot start)"
    else:
        actual_hash = _sha256(checkpoint)
        if expected_hash and actual_hash != expected_hash.lower():
            raise ValueError(f"{group}: checkpoint SHA256 differs from original comparison.json: {checkpoint}")
        checkpoint_status = "SHA256 verified" if expected_hash else "SHA256 recorded (source did not record a hash)"
    return {
        "group": group, "comparison": str(source_path), "comparison_sha256": _sha256(source_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": actual_hash,
        "recorded_checkpoint_sha256": expected_hash, "checkpoint_status": checkpoint_status,
        "model_config": config, "rgb_normalization": normalization,
        "cod_train_mode": protocol.get("cod_train_mode", "split"),
        "cod_train_samples": cod_train, "cod_val_samples": cod_val,
        "selected_epoch": epoch, "selection_metric": fit.get("selection_metric"),
        "source_data_notes": SOURCE_NOTES[group],
        "original_sample_counts": counts,
        "original_split_manifest_sha256": protocol.get("split_manifest_sha256"),
    }


def _validate_output(out, run_dir):
    if out == run_dir or run_dir in out.parents or out in run_dir.parents:
        raise ValueError("--out must be separate from the original --run-dir (not inside it or an ancestor)")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {out}")


def _build_command(args, sources, out):
    command = [sys.executable, "-u", str(ROOT / "tools/run_triprd_structure.py"),
               "--eval-only", "--no-skip-test", "--groups", ",".join(sources),
               "--cod-root", str(args.cod_root.expanduser().resolve()), "--out", str(out),
               "--device", args.device, "--test-datasets", ",".join(DATASET_COUNTS),
               "--max-test-samples", "0"]
    config = next(iter(sources.values()))["model_config"]
    for key, value in config.items():
        if key == "pretrained":
            command.append("--pretrained" if value else "--no-pretrained")
        else:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    for group, source in sources.items():
        command.extend(["--cod-checkpoint", f"{group}={source['checkpoint']}"])
    return command


def _validate_results(result, sources):
    if result.get("status") != "complete" or result.get("test_status") != "complete":
        raise ValueError("Evaluation is not complete; no comparison rows will be exported")
    if set(result.get("groups", {})) != set(sources) or set(result.get("test", {})) != set(sources):
        raise ValueError("Evaluation groups do not match the requested groups")
    for group, source in sources.items():
        if result["groups"][group].get("checkpoint_sha256") != source["checkpoint_sha256"]:
            raise ValueError(f"{group}: evaluated checkpoint hash differs from the preflight hash")
        tests = result["test"][group]
        if set(tests) != set(DATASET_COUNTS):
            raise ValueError(f"{group}: all four test datasets are required")
        for dataset, expected in DATASET_COUNTS.items():
            test = tests[dataset]
            coverage, metrics = test.get("coverage", {}), test.get("official", {})
            if (coverage.get("limited") is not False or
                    coverage.get("verified_samples") != expected or
                    coverage.get("expected_full_samples") != expected or
                    test.get("count") != expected or metrics.get("num_samples") != expected):
                raise ValueError(f"{group}/{dataset}: incomplete or limited test coverage; expected {expected}")
            for field in (*TABLE_FIELDS, "E_phi"):
                value = metrics.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{group}/{dataset}: missing/nonfinite official {field}")


def _export_results(result, sources, out):
    rows, latex = [], [
        "% Dataset order: NC4K, COD10K, CAMO, CHAMELEON.",
        "% Each dataset: S_m, adaptive E (alpha E), weighted F, MAE.",
        "% Values use the same selected checkpoint for all metrics and datasets.",
        "% Training data and source supervision differ from 4040-COD-image methods; disclose them.",
    ]
    raw = {}
    for group, source in sources.items():
        values = []
        raw[group] = {}
        for dataset in DATASET_COUNTS:
            metrics = result["test"][group][dataset]["official"]
            raw[group][dataset] = metrics
            rows.append({
                "group": group, "dataset": dataset, "num_samples": metrics["num_samples"],
                **{field: metrics[field] for field in TABLE_FIELDS}, "E_phi_mean": metrics["E_phi"],
                "cod_train_samples": source["cod_train_samples"], "cod_val_samples": source["cod_val_samples"],
                "selected_epoch": source["selected_epoch"], "source_data_notes": source["source_data_notes"],
                "checkpoint": source["checkpoint"], "checkpoint_sha256": source["checkpoint_sha256"],
            })
            values.extend(f"{metrics[field]:.3f}" for field in TABLE_FIELDS)
        latex.append(f"% {group}: selected epoch {source['selected_epoch']}; "
                     f"{source['cod_val_samples']} COD validation; {source['source_data_notes']}.")
        latex.append(f"TriPRD-{group} ({source['cod_train_samples']} COD train) & "
                     + " & ".join(values) + r" \\")
    _save_json(out / "metrics.json", raw)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "previous_performance_rows.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    groups = [group.strip().upper() for group in args.groups.split(",")]
    if not groups or len(groups) != len(set(groups)) or set(groups) - set(GROUP_DIRS):
        raise ValueError("--groups must contain unique entries from A,B,C,D,E")
    run_dir = args.run_dir.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.out or (ROOT / "results/optimization/structure_v2_test" / f"{run_dir.name}-{stamp}")).expanduser().resolve()
    _validate_output(out, run_dir)
    sources = {group: _load_source(run_dir, group, dry_run=args.dry_run) for group in groups}
    reference = sources[groups[0]]["model_config"]
    if any(source["model_config"] != reference for source in sources.values()):
        raise ValueError("Selected groups have different model/input/seed configurations; evaluate them separately")
    command = _build_command(args, sources, out)
    print(f"Evaluation plan: all {sum(DATASET_COUNTS.values())} images per group; "
          "no training or checkpoint selection", flush=True)
    for group, source in sources.items():
        print(f"{group}: {source['checkpoint_status']} | {source['checkpoint']}", flush=True)
        print(f"   COD train={source['cod_train_samples']}, validation={source['cod_val_samples']}, "
              f"selected epoch={source['selected_epoch']}; {source['source_data_notes']}", flush=True)
    print("Command (POSIX shell): " + shlex.join(command), flush=True)
    print(f"Output: {out}", flush=True)
    if args.dry_run:
        print("Dry run complete: no inference started and no files written.", flush=True)
        return 0
    # Do not create a manifest before launching: the runner rejects nonempty
    # output directories. It writes comparison.json and prediction PNGs itself.
    subprocess.run(command, cwd=ROOT, check=True)
    result = _read_json(out / "comparison.json")
    _validate_results(result, sources)
    _export_results(result, sources, out)
    _save_json(out / "evaluation_manifest.json", {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "out": str(out), "command": command,
        "groups": sources, "test_counts": DATASET_COUNTS, "test_images_per_group": sum(DATASET_COUNTS.values()),
        "table_fields": list(TABLE_FIELDS), "mean_E_field": "E_phi_mean (CSV); E_phi (raw JSON)",
        "evaluation_method": "Vendored CamoFormer metrics; tools/evaluate_cod.py",
        "evaluation_reference": "https://github.com/HVision-NKU/CamoFormer/blob/main/evaltools/eval.py",
        "paper_reference": "https://arxiv.org/pdf/2212.06570",
        "inference": "Original structure-v2 runner inference and checkpoint RGB normalization/input size",
        "training_disclosure": "Existing selected checkpoints only; COD training/validation counts and VDT use are recorded per group. This does not establish equal training data with 4040-image methods.",
        "evaluator_sha256": {name: _sha256(ROOT / name) for name in (
            "tools/evaluate_structure_v2.py", "tools/evaluate_cod.py", "evaluation/camoformer_metrics.py",
            "evaluation/cod_metrics.py", "tools/run_triprd_structure.py", "tools/run_triprd_ab.py")},
        "comparison_sha256": _sha256(out / "comparison.json"),
    })
    print(f"Complete: {out / 'previous_performance_rows.tex'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
