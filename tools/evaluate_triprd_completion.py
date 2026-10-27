#!/usr/bin/env python3
"""Evaluate one completed v6 full4040 model and export per-dataset scores."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from evaluation.task_protocol import DATASET_COUNTS, METRIC_FIELDS


def export_summary(run):
    from tools.run_triprd_completion import validate_completed_run
    run = Path(run).resolve()
    report = validate_completed_run(run,formal=True)
    if report.get("test_status") != "complete" or set(report.get("test",{})) != set(DATASET_COUNTS):
        raise ValueError("Four official test datasets must be complete before exporting scores")
    rows = []
    for dataset,count in DATASET_COUNTS.items():
        result = report["test"][dataset]
        metrics,coverage = result["official"],result["coverage"]
        if (result["count"] != count or metrics.get("num_samples") != count or coverage.get("verified_samples") != count or
                coverage.get("expected_full_samples") != count or coverage.get("limited") is not False or
                coverage.get("exact_id_match") is not True):
            raise ValueError(f"Incomplete evaluation coverage: {dataset}")
        import math
        if any(isinstance(metrics.get(k),bool) or not isinstance(metrics.get(k),(int,float)) or
               not math.isfinite(metrics[k]) for k in METRIC_FIELDS):
            raise ValueError(f"Missing/nonfinite official metrics: {dataset}")
        rows.append({"dataset":dataset,"count":count,**{k:metrics[k] for k in METRIC_FIELDS}})
    with (run/"benchmark.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["dataset","count",*METRIC_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    text = ["# TriPRD completion v6 official test results", "",
            "RGB-only inference; full COD10K3040 + CAMO1000 training; fixed last epoch.",
            "Additional VDT source training and a multimodal teacher are part of the training resources.", "",
            "| Dataset | N | Sm ↑ | Fw ↑ | Emean ↑ | Emax ↑ | Eadaptive ↑ | MAE ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values=[row[k] for k in ("S_alpha","F_beta_w","E_phi_mean","E_phi_max","E_phi_adaptive","MAE")]
        text.append(f"| {row['dataset']} | {row['count']} | " + " | ".join(f"{x:.6f}" for x in values)+" |")
    text += ["",f"Checkpoint SHA256: `{report['checkpoint_sha256']}`",
             f"Prediction protocol: `{report['prediction_protocol']}`", "",
             "The E-measure variants are separate; these rows do not establish a SOTA claim.", ""]
    (run/"benchmark.md").write_text("\n".join(text),encoding="utf-8")
    print("\n".join(text))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--cod-root",default="datasets/COD_benchmark")
    p.add_argument("--dino-repo",type=Path)
    p.add_argument("--summary-only",action="store_true")
    a=p.parse_args(argv)
    if not a.summary_only:
        from tools import run_triprd_completion as runner
        command=["--stage","cod","--out",str(a.run),"--eval-only","--device",a.device,
                 "--cod-root",str(a.cod_root),"--cod-train-mode","full"]
        if a.dino_repo:
            command.extend(["--dino-repo",str(a.dino_repo)])
        runner.main(command)
    export_summary(a.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
