#!/usr/bin/env python3
"""Compare selected correction_v3 checkpoints without cherry-picking epochs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DATASET_COUNTS = {"NC4K": 4121, "COD10K": 2026, "CAMO": 250, "CHAMELEON": 76}
TABLE_FIELDS = ("S_alpha", "E_phi_adaptive", "F_beta_w", "MAE")


def summarize(root: Path, groups=None):
    groups = tuple(groups or "ABCDEF")
    if not groups or len(set(groups)) != len(groups) or set(groups) - set("ABCDEF"):
        raise ValueError("Expected unique groups from A,B,C,D,E,F")
    reports, records = {}, []
    for path in sorted(root.glob("[A-F]_*/comparison.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("method") != "triprd_correction_v3":
            raise ValueError(f"Wrong method: {path}")
        group = report["config"]["group"]
        if group not in groups:
            continue
        if group in reports:
            raise ValueError(f"Duplicate group {group}")
        reports[group] = report
        record = {"group": group, "status": report.get("status"), "epoch": report.get("best_epoch"),
                  "formal": report.get("formal", False), "metrics": None, "sources": None,
                  "recoverability": report.get("source_recoverability"), "test": report.get("test", {}),
                  "test_status": report.get("test_status"), "report_path": str(path.resolve()),
                  "checkpoint_sha256": report.get("checkpoint_sha256"),
                  "sample_counts": report.get("sample_counts", {}),
                  "cod_train_mode": report["protocol"].get("cod_train_mode"),
                  "selection_metric": report.get("selection_metric"),
                  "budget_epochs": report["protocol"].get("epochs")}
        selected = next((row for row in report.get("history", []) if row["epoch"] == record["epoch"]), None)
        if selected and "validation" in selected:
            record.update(metrics=selected["validation"]["macro"], sources=selected["validation"]["sources"])
        records.append(record)
    if not records:
        raise ValueError(f"No correction_v3 group reports under {root}")
    # Group-specific transfer recipes differ by design; these must be matched.
    keys = ("model_variant", "base_channels", "image_size", "relation_kernel", "pretrained",
            "generic_weights_sha256", "generic_config_sha256", "dino_source_sha256", "source_sha256", "train_seed", "split_seed",
            "cod_train", "cod_val", "vdt_train", "vdt_val", "epochs", "batch_size", "lr", "weight_decay",
            "cod_train_mode", "val_metrics_every", "source_weight", "kd_weight", "relation_weight", "amp",
            "warmup_epochs", "grad_clip", "teacher_epochs", "num_workers")
    first = next(iter(reports.values()))
    for group, report in reports.items():
        mismatches = [key for key in keys if report["protocol"].get(key) != first["protocol"].get(key)]
        if mismatches:
            raise ValueError(f"Unmatched group {group}: {mismatches}")
        if report.get("initial_state_sha256") and first.get("initial_state_sha256") != report["initial_state_sha256"]:
            raise ValueError(f"Group {group} uses different student initialization")
    teacher_hashes = {report["protocol"].get("teacher_sha256") for group, report in reports.items() if group in "CDEF"}
    if None in teacher_hashes or len(teacher_hashes) > 1:
        raise ValueError("C/D/E/F must reuse the same nonempty teacher checkpoint hash")
    by_group = {row["group"]: row for row in records}
    deltas = {}
    for newer, baseline in (("B", "A"), ("C", "B"), ("D", "A"), ("D", "B"), ("D", "C"), ("E", "D"), ("E", "F"), ("E", "A")):
        if all(g in by_group and by_group[g]["status"] == "complete" and by_group[g]["metrics"] for g in (newer, baseline)):
            deltas[f"{newer}-{baseline}"] = {key: by_group[newer]["metrics"][key] - by_group[baseline]["metrics"][key]
                                               for key in ("S_alpha", "F_beta_w", "E_phi", "MAE")}
    return {"status": "complete" if set(reports) == set(groups) and all(x["status"] == "complete" for x in records) else "incomplete",
            "expected_groups": list(groups), "missing_groups": sorted(set(groups) - set(reports)),
            "root": str(root), "records": records, "validation_deltas": deltas,
            "interpretation": "Internal validation differences from one seed are not official test SOTA or statistical significance."}


def export_tests(summary, out: Path):
    """Export only complete four-dataset tests of the selected checkpoint."""
    if summary["status"] != "complete":
        raise ValueError("Cannot export test rows for incomplete selected groups")
    rows, latex = [], [
        "% Dataset order: NC4K, COD10K, CAMO, CHAMELEON.",
        "% Fields: Sm, adaptive E (alpha E), weighted F, MAE; CamoFormer evaluator.",
        "% Correction-v3 group names differ from structure-v2; do not combine their labels.",
        "% Disclose generic DINOv3 pretraining and additional VDT source supervision.",
    ]
    for record in summary["records"]:
        group, tests = record["group"], record["test"]
        if record["test_status"] != "complete" or set(tests) != set(DATASET_COUNTS):
            raise ValueError(f"{group}: all four completed official test sets are required")
        checkpoint = Path(record["report_path"]).parent / "best.pt"
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["checkpoint_sha256"]:
            raise ValueError(f"{group}: selected checkpoint hash differs from report")
        counts = record["sample_counts"]
        if not isinstance(counts.get("cod_train"), int) or counts["cod_train"] <= 0:
            raise ValueError(f"{group}: missing COD training count")
        if record["cod_train_mode"] == "full" and (
                counts["cod_train"] != 4040 or counts.get("cod_val") != 0 or
                record["selection_metric"] != "last_epoch" or record["epoch"] != record["budget_epochs"]):
            raise ValueError(f"{group}: invalid full-4040 fixed-last-epoch protocol")
        values = []
        for dataset, expected in DATASET_COUNTS.items():
            result = tests[dataset]
            coverage, metrics = result.get("coverage", {}), result.get("official", {})
            if (coverage.get("limited") is not False or coverage.get("verified_samples") != expected or
                    coverage.get("expected_full_samples") != expected or result.get("count") != expected or
                    metrics.get("num_samples") != expected):
                raise ValueError(f"{group}/{dataset}: incomplete official coverage; expected {expected}")
            for field in (*TABLE_FIELDS, "E_phi"):
                value = metrics.get(field)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"{group}/{dataset}: missing/nonfinite {field}")
            rows.append({"group": group, "dataset": dataset, "num_samples": expected,
                         **{field: metrics[field] for field in TABLE_FIELDS}, "E_phi_mean": metrics["E_phi"],
                         "cod_train_samples": counts["cod_train"], "selected_epoch": record["epoch"],
                         "checkpoint_sha256": record["checkpoint_sha256"]})
            values.extend(f"{metrics[field]:.3f}" for field in TABLE_FIELDS)
        source = "COD only" if group == "A" else "additional VDT RGB/GT" if group == "B" else "additional VDT RGB/T/D/GT teacher"
        latex.append(f"% {group}: {source}; selected epoch {record['epoch']}; formal={record['formal']}.")
        latex.append(f"TriPRD-v3-{group} ({counts['cod_train']} COD train) & " + " & ".join(values) + r" \\")
    # Validate every record before writing any output.
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "previous_performance_rows.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--groups", default="A,B,C,D,E,F", help="Expected matched groups")
    p.add_argument("--test-output-dir", type=Path, help="Export complete official tests to CSV and LaTeX")
    a = p.parse_args(argv)
    summary = summarize(a.root, [g.strip().upper() for g in a.groups.split(",")])
    print("group status epoch Sm Fw meanE MAE")
    for row in summary["records"]:
        metric = row["metrics"]
        values = " ".join(f"{metric[key]:.6f}" for key in ("S_alpha", "F_beta_w", "E_phi", "MAE")) if metric else "(no internal validation)"
        print(f"{row['group']} {row['status']} {row['epoch']} {values}")
    print(json.dumps(summary["validation_deltas"], indent=2))
    if a.test_output_dir:
        export_tests(summary, a.test_output_dir)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
