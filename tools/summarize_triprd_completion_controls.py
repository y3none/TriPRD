#!/usr/bin/env python3
"""Verify the four matched completion controls and report descriptive deltas."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.task_protocol import DATASET_COUNTS, METRIC_FIELDS

GROUPS = ("A", "B", "C", "D")
METHOD = "triprd_completion_controls_v1"
METRICS = ("S_alpha", "F_beta_w", "E_phi_mean", "MAE")
PAIRS = (("B", "A"), ("C", "B"), ("D", "C"), ("D", "A"))
MATCHED = (
    "model_variant", "base_channels", "image_size", "pretrained", "train_backbone_blocks",
    "detail_branch", "batch_size", "micro_batch_size", "num_workers", "train_seed", "split_seed",
    "lr", "backbone_lr_scale", "weight_decay", "warmup_epochs", "locator_warmup_epochs",
    "amp", "grad_clip", "augmentation", "boundary_weight", "aux_weight", "cue_perturbation",
    "teacher_epochs", "source_epochs", "epochs", "val_metrics_every", "val_fraction", "max_samples",
    "kd_weight", "temperature", "data_parallel", "generic_weights_sha256", "generic_config_sha256",
    "dino_source_sha256", "source_sha256", "prediction_protocol", "split_manifest_sha256",
    "training_device_count", "compute_parallel", "cod_optimizer_reset", "cod_training_rng_reset",
)
DESCRIPTIONS = {
    "A": "COD only; no VDT source training or teacher",
    "B": "VDT RGB/GT source training; no teacher",
    "C": "VDT RGB/GT + capacity-matched RGB teacher",
    "D": "VDT RGB/GT + real RGB/thermal/depth teacher",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (float, int) and math.isfinite(value), f"Invalid metric: {name}")
    return float(value)


def _matching(left, right, keys, label):
    for key in keys:
        require(key in left and key in right and left[key] == right[key], f"{label} differs: {key}")


def _cost(report):
    training = [row["train"] for row in report["history"]]
    return {"epochs": report["completed_epochs"], "parameters": report["parameters"],
            **{name: sum(row.get(name, 0) for row in training)
               for name in ("seconds", "steps", "optimizer_steps", "amp_skipped_steps")},
            "peak_cuda_allocated_bytes": max((row.get("peak_cuda_allocated_bytes") or 0 for row in training), default=0)}


def _scores(report, full):
    rows = {}
    if full:
        require(report.get("test_status") == "complete" and set(report.get("test", {})) == set(DATASET_COUNTS),
                "Full runs require all four completed official test sets")
        for name, count in DATASET_COUNTS.items():
            value = report["test"][name]
            official, coverage = value["official"], value["coverage"]
            require(value.get("count") == official.get("num_samples") == count
                    and coverage.get("verified_samples") == coverage.get("expected_full_samples") == count
                    and coverage.get("limited") is False and coverage.get("exact_id_match") is True,
                    f"Incomplete test coverage: {name}")
            rows[name] = {"count": count, **{key: finite(official.get(key), key) for key in METRIC_FIELDS}}
    else:
        require(report.get("test_status") == "skipped" and not report.get("test"), "Development run accessed official tests")
        selected = next(row for row in report["history"] if row["epoch"] == report["best_epoch"])["validation"]
        expected = Counter(name.split("/", 1)[0] for name in report["protocol"]["cod_val"])
        require(set(selected.get("sources", {})) == set(expected) == {"COD10K", "CAMO"}, "Missing validation source")
        require(selected.get("num_samples") == sum(expected.values()), "Wrong validation count")
        for name, value in {**selected["sources"], "macro": selected["macro"]}.items():
            count = sum(expected.values()) if name == "macro" else expected[name]
            if name != "macro":
                require(value.get("num_samples") == count, "Wrong per-source validation count")
            rows[name] = {"count": count, **{key: finite(value.get("E_phi" if key == "E_phi_mean" else key), key) for key in METRICS}}
        for key in METRICS:
            require(math.isclose(rows["macro"][key], sum(rows[name][key] for name in expected) / 2,
                                 rel_tol=0, abs_tol=1e-12), "Validation macro does not match sources")
    return rows


def summarize(root):
    from tools.run_triprd_completion_control import validate_completed_run
    root = Path(root).resolve()
    reports = {}
    # Validate best/last checkpoints and the complete provenance before any export.
    for group in GROUPS:
        report = validate_completed_run(root / group / "cod")
        require(report.get("method") == METHOD and report.get("stage") == "cod"
                and report["config"].get("group") == group and report["protocol"].get("group") == group,
                f"Wrong group/method: {group}")
        reports[group] = report
    reference = reports["A"]
    mode = reference["protocol"]["cod_train_mode"]
    require(mode in ("split", "full"), "Unknown COD training mode")
    full = mode == "full"
    costs, scores, sources, teachers = {}, {}, {}, {}
    for group, report in reports.items():
        protocol = report["protocol"]
        _matching(reference["protocol"], protocol, (*MATCHED, "cod_train_mode", "cod_train", "cod_val"), "COD protocol")
        _matching(reference, report, ("generic_initial_state_sha256", "cod_locator_before_sha256", "cod_locator_after_sha256", "parameters"), "Student initialization/capacity")
        require(protocol["data_parallel"] is False and protocol["training_device_count"] == 1, "Each control must use one GPU")
        if full:
            validate_completed_run(root / group / "cod", formal=True)
        scores[group] = _scores(report, full)
        costs[group] = {"cod": _cost(report)}
        if group != "A":
            source = validate_completed_run(root / group / "source")
            proof = report["source_provenance"]
            require(proof["checkpoint_sha256"] == source["checkpoint_sha256"]
                    and proof["contract"] == source["protocol"], "Source artifact differs from consumed predecessor")
            require(source["generic_initial_state_sha256"] == source["initial_state_sha256"]
                    == reference["generic_initial_state_sha256"], "Source student initialization differs")
            _matching(protocol, source["protocol"], MATCHED, "Source/COD resources")
            if sources:
                _matching(sources["B"]["protocol"], source["protocol"], (*MATCHED, "vdt_train", "vdt_val"), "Source controls")
            sources[group] = source
            costs[group]["source"] = _cost(source)
        if group in ("C", "D"):
            teacher = validate_completed_run(root / group / "teacher")
            proof = sources[group]["teacher_provenance"]
            require(proof["checkpoint_sha256"] == teacher["checkpoint_sha256"]
                    and proof["contract"] == teacher["protocol"], "Teacher artifact differs from consumed predecessor")
            _matching(protocol, teacher["protocol"], MATCHED, "Teacher/COD resources")
            if teachers:
                _matching(teachers["C"], teacher, ("generic_initial_state_sha256", "initial_state_sha256", "parameters"), "Teacher initialization/capacity")
                _matching(teachers["C"]["protocol"], teacher["protocol"], (*MATCHED, "vdt_train", "vdt_val"), "Teacher controls")
            teachers[group] = teacher
            costs[group]["teacher"] = _cost(teacher)
    deltas = {f"{left}-{right}": {dataset: {key: scores[left][dataset][key] - scores[right][dataset][key]
                                           for key in METRICS} for dataset in scores[left]}
              for left, right in PAIRS}
    return {"method": METHOD, "scope": "official_tests" if full else "internal_validation",
            "train_seed": reference["protocol"]["train_seed"], "split_seed": reference["protocol"]["split_seed"],
            "matched_artifacts_verified": True, "groups": DESCRIPTIONS,
            "selected_epochs": {group: report["best_epoch"] for group, report in reports.items()},
            "scores": scores, "deltas": deltas, "costs": costs,
            "interpretation": "Raw deltas: positive Sm/Fw/Emean and negative MAE favor the left group. D-C isolates the additional modalities within this protocol. Single-seed descriptive comparison, not statistical significance or a SOTA claim. Total source/teacher training costs differ and are disclosed."}


def export(root, result):
    root = Path(root)
    lines = ["# Completion A–D matched controls", "", f"Scope: {result['scope']}; train seed: {result['train_seed']}.", "",
             result["interpretation"], "", "| Group | Dataset | Epoch | Sm ↑ | Fw ↑ | Emean ↑ | MAE ↓ |",
             "|---|---|---:|---:|---:|---:|---:|"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["group", "dataset", "count", "epoch", *METRICS])
    writer.writeheader()
    for group in GROUPS:
        for dataset, row in result["scores"][group].items():
            epoch = result["selected_epochs"][group]
            writer.writerow({"group": group, "dataset": dataset, "count": row["count"], "epoch": epoch, **{k: row[k] for k in METRICS}})
            lines.append(f"| {group} | {dataset} | {epoch} | " + " | ".join(f"{row[k]:.6f}" for k in METRICS) + " |")
    lines += ["", "## Raw metric deltas", "", "| Contrast | Dataset | ΔSm | ΔFw | ΔEmean | ΔMAE |", "|---|---|---:|---:|---:|---:|"]
    for contrast, datasets in result["deltas"].items():
        for name, row in datasets.items():
            lines.append(f"| {contrast} | {name} | " + " | ".join(f"{row[k]:+.6f}" for k in METRICS) + " |")
    lines += ["", "## Training resources", ""]
    for group in GROUPS:
        lines.append(f"- {group}: {DESCRIPTIONS[group]}. " + "; ".join(
            f"{stage}: {cost['epochs']} epochs, {cost['optimizer_steps']} updates, {cost['seconds'] / 60:.1f} training-loop minutes"
            for stage, cost in result["costs"][group].items()))
    lines += ["", "Training-loop time excludes validation, checkpoint I/O and official testing. GPU peak memory is recorded per stage in the JSON.", ""]
    outputs = {"controls_summary.json": json.dumps(result, ensure_ascii=False, indent=2) + "\n",
               "controls_summary.csv": buffer.getvalue(), "controls_summary.md": "\n".join(lines)}
    for name, text in outputs.items():
        temporary = root / (name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(root / name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.root)
    export(args.root, result)
    print(f"Verified A–D controls: {args.root / 'controls_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
