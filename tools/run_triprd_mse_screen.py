#!/usr/bin/env python3
"""Matched COD validation screen after a verified joint-positive-MSE source probe.

Reuse completed v3 A/B and teacher evidence; train fresh C/D only. This entry
point cannot evaluate official test sets or launch full-pool final experiments.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tools import diagnose_triprd_correction_probe as diagnostic
from tools import run_triprd_correction as runner
from tools import run_triprd_optimization_phases as phases
from tools import triprd_mse_screen_training as training
from tools.triprd_training import sha256_file

GROUP_DIRS = {"A": "A_cod_only", "B": "B_rgb_joint", "C": "C_full_kd", "D": "D_joint_positive_mse"}
NEW_SOURCES = ("tools/run_triprd_mse_screen.py", "tools/triprd_mse_screen_training.py",
               "tools/run_triprd_mse_screen.sh")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--diagnostics-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", help="Physical GPU IDs; two independent C/D jobs, no DDP")
    p.add_argument("--execute", action="store_true", help="Default is read-only verification and plan")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--group", choices=("C", "D"), help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _saved_args(report):
    a = runner.parse_args(["--out", report["config"]["out"]])
    for key, value in report["config"].items():
        if not hasattr(a, key):
            raise ValueError(f"Unknown saved option: {key}")
        setattr(a, key, value)
    for key in ("out", "dino_repo", "dino_weights", "split_manifest", "teacher_checkpoint", "cod_checkpoint"):
        value = getattr(a, key)
        setattr(a, key, Path(value) if value else None)
    runner.validate_args(a)
    return a


def selected_validation(report):
    rows = [row for row in report["history"] if "validation" in row]
    if not rows:
        raise ValueError("Missing COD validation history")
    for row in rows:
        phases._finite(row["validation"]["macro"]["S_alpha"], "validation macro Sm")
    best = max(rows, key=lambda row: row["validation"]["macro"]["S_alpha"])
    if best["epoch"] != report["best_epoch"] or best["validation"]["macro"]["S_alpha"] != report["best_score"]:
        raise ValueError("Selected epoch/score differs from maximum validation macro Sm")
    return best["validation"]


def _baseline(run_root, group, source_context):
    directory = run_root / GROUP_DIRS[group]
    report = _read(directory / "comparison.json")
    diagnostic._validate_report(report, "student")
    a = _saved_args(report)
    if (a.group != group or a.out.expanduser().resolve() != directory or a.stage != "student"
            or a.cod_train_mode != "split" or not a.skip_test or a.eval_only or a.prepare_only
            or a.teacher_checkpoint or report.get("test") or report.get("test_status") != "skipped"):
        raise ValueError(f"{group} must be a completed, untouched internal-split baseline without tests")
    splits, manifest = runner.prepare_data(a)
    teacher_contract, contract = runner._contracts(a, manifest, runner._resources(a))
    if report["protocol"] != contract or teacher_contract != source_context["teacher"]["protocol"]:
        raise ValueError(f"{group} data, source teacher, resources or training settings differ")
    if report.get("sample_counts") != {key: len(value) for key, value in splits.items()}:
        raise ValueError(f"{group} sample counts differ")
    if report.get("checkpoint_sha256") != sha256_file(directory / "best.pt"):
        raise ValueError(f"{group} selected checkpoint digest differs")
    selected = runner._read_checkpoint(directory / "best.pt", contract)
    last = runner._read_checkpoint(directory / "last.pt", contract)
    if (last.get("epoch") != a.epochs or report.get("completed_epochs") != a.epochs
            or last.get("history") != report.get("history")
            or [row.get("epoch") for row in last["history"]] != list(range(1, a.epochs + 1))
            or selected.get("epoch") != report.get("best_epoch")
            or selected.get("best_score") != report.get("best_score")
            or selected.get("history") != report["history"][:selected["epoch"]]
            or selected.get("initial_state_sha256") != report.get("initial_state_sha256")
            or last.get("initial_state_sha256") != report.get("initial_state_sha256")):
        raise ValueError(f"{group} report and checkpoint budget/history/initialization differ")
    if any(state.get("stage") != "student" or state.get("group") != group for state in (selected, last)):
        raise ValueError(f"{group} checkpoint stage/group differs")
    selected_validation(report)
    return a, splits, manifest, report


def build_context(args):
    for key in ("run_root", "diagnostics_root", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    roots = (args.run_root, args.diagnostics_root, args.output_root)
    if any(left == right or left in right.parents or right in left.parents
           for i, left in enumerate(roots) for right in roots[i + 1:]):
        raise ValueError("Reference, diagnostics and new screen roots must be disjoint")
    evidence_dir = args.diagnostics_root / "joint_positive_mse"
    evidence = _read(evidence_dir / "comparison.json")
    probe_epochs = evidence.get("contract", {}).get("epochs")
    if isinstance(probe_epochs, bool) or not isinstance(probe_epochs, int) or probe_epochs < 1:
        raise ValueError("Invalid source diagnostic budget")
    source = diagnostic.build_context(argparse.Namespace(run_root=args.run_root,
        output_root=args.diagnostics_root, epochs=probe_epochs, resume=False))
    evidence_contract = diagnostic.arm_contract(source, "joint_positive_mse")
    evidence = diagnostic.verify_completed(evidence_dir, evidence_contract)
    phases.student_gate(evidence)
    a_a, _, manifest_a, report_a = _baseline(args.run_root, "A", source)
    a, splits, manifest, report_b = _baseline(args.run_root, "B", source)
    comparable = lambda report: {key: value for key, value in report["protocol"].items() if key != "group"}
    if comparable(report_a) != comparable(report_b) or manifest_a != manifest:
        raise ValueError("A/B protocols, budgets or partitions are not matched")
    # Device placement may differ; train and data semantics must not.
    exclude = {"out", "group", "device", "resume", "split_manifest"}
    if {k: v for k, v in runner._json(a_a).items() if k not in exclude} != {
            k: v for k, v in runner._json(a).items() if k not in exclude}:
        raise ValueError("A/B saved configurations differ")
    initial = report_b.get("initial_state_sha256")
    if not initial or any(report.get("initial_state_sha256") != initial
                          for report in (report_a, source["probe"], evidence)):
        raise ValueError("Baseline/probe initial student states differ")
    # The passed source experiment must use exactly the intended source term.
    for key in ("source_weight", "kd_weight"):
        if source["a"].__dict__[key] != getattr(a, key):
            raise ValueError(f"Passed diagnostic has a different {key}")
    references = {}
    for group, report in (("A", report_a), ("B", report_b)):
        directory = args.run_root / GROUP_DIRS[group]
        references[group] = {"directory": str(directory), "report_sha256": sha256_file(directory / "comparison.json"),
            "best_sha256": report["checkpoint_sha256"], "last_sha256": sha256_file(directory / "last.pt"),
            "protocol": report["protocol"]}
    a.teacher_checkpoint = args.run_root / "teacher" / "best.pt"
    recipe = {"method": training.METHOD, "new_source_sha256": {p: sha256_file(ROOT / p) for p in NEW_SOURCES},
        "baseline_references": references, "teacher_protocol": source["teacher"]["protocol"],
        "teacher_report_sha256": sha256_file(args.run_root / "teacher/comparison.json"),
        "teacher_sha256": source["teacher"]["checkpoint_sha256"],
        "evidence": {"arm": "joint_positive_mse", "contract": evidence_contract,
            "report_sha256": sha256_file(evidence_dir / "comparison.json"),
            "checkpoint_sha256": evidence["checkpoint_sha256"], "source_recoverability": evidence["source_recoverability"]},
        "epochs": a.epochs, "expected_initial_state_sha256": initial, "split_manifest": manifest,
        "selection": "maximum_internal_COD_validation_macro_S_alpha", "official_test_access": False,
        "initialization": "fresh_generic_pretrained_student_no_probe_or_baseline_warm_start",
        "C_objective": "COD_supervised + source_weight * (source_supervised + kd_weight * ordinary_full_relation_KD)",
        "D_objective": "COD_supervised + source_weight * (source_supervised + kd_weight * 5 * pooled_positive_edge_MSE)",
        "mse_coefficient": diagnostic.MSE_COEFFICIENT}
    return {"a": a, "splits": splits, "manifest": manifest, "teacher_state": source["teacher_state"],
            "baselines": {"A": report_a, "B": report_b}, "expected_initial": initial, "recipe": recipe}


def student_args(args, context, group):
    a = copy.deepcopy(context["a"])
    a.group, a.out, a.resume = group, args.output_root / GROUP_DIRS[group], args.resume
    if torch.device(a.device).type == "cuda":
        a.device = "cuda:0"
    runner.validate_args(a)
    return a


def student_contract(context, group):
    if group not in ("C", "D"):
        raise ValueError("Only C/D are new students")
    # Keep the old data/optimization protocol explicit without relabeling it.
    protocol = copy.deepcopy(context["baselines"]["B"]["protocol"])
    protocol.update(group=group, teacher_sha256=context["recipe"]["teacher_sha256"])
    return {"method": training.METHOD, "stage": "student", "group": group,
        "epochs": context["a"].epochs, "matched_v3_protocol": protocol,
        "recipe": context["recipe"], "objective": context["recipe"][f"{group}_objective"]}


def make_summary(context, reports):
    rows = []
    for group, report in {**context["baselines"], **reports}.items():
        metrics = selected_validation(report)
        rows.append({"group": group, "label": GROUP_DIRS[group], "reused": group in "AB",
            "method": report["method"], "selected_epoch": report["best_epoch"],
            "initial_state_sha256": report["initial_state_sha256"], "validation": metrics})
    scores = {row["group"]: row["validation"]["macro"]["S_alpha"] for row in rows}
    deltas = {group: scores["D"] - scores[group] for group in "ABC"}
    return {"method": training.METHOD, "scope": "internal_COD_validation_screen", "formal_test_result": False,
        "rows": rows, "D_minus_baseline_macro_S_alpha": deltas,
        "D_beats_all_ABC": all(value > 0 for value in deltas.values()),
        "note": "Single split/seed selection evidence, not statistical significance or comparison with published official-test scores. No final runs launched."}


def _run_jobs(args, context, gpus):
    pending, active, reports = list("CD"), [], {}
    try:
        while pending or active:
            finished = [(item, item["process"].poll()) for item in active]
            for item, code in finished:
                if code is not None and code != 0:
                    raise RuntimeError(f"{item['group']} exited {code}; see {item['directory'] / 'run.log'}")
            for item, code in finished:
                if code is not None:
                    item["process"].wait()
                    reports[item["group"]] = training.verify_completed(item["directory"], student_contract(context, item["group"]))
                    item["log"].close()
                    active.remove(item)
                    print(f"Completed screen {item['group']}", flush=True)
            available = [gpu for gpu in gpus if all(item["gpu"] != gpu for item in active)]
            while pending and available:
                group, gpu = pending.pop(0), available.pop(0)
                directory = args.output_root / GROUP_DIRS[group]
                if args.resume and (directory / "comparison.json").exists() and _read(directory / "comparison.json").get("status") == "complete":
                    reports[group] = training.verify_completed(directory, student_contract(context, group))
                    available.append(gpu)
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                stream = (directory / "run.log").open("a", encoding="utf-8", buffering=1)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-root", str(args.run_root),
                    "--diagnostics-root", str(args.diagnostics_root), "--output-root", str(args.output_root),
                    "--group", group, "--execute"]
                if args.resume:
                    command.append("--resume")
                environment = os.environ.copy()
                if gpu is not None:
                    environment["CUDA_VISIBLE_DEVICES"] = gpu
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                        **({"start_new_session": True} if os.name == "posix" else {}))
                except BaseException:
                    stream.close()
                    raise
                active.append({"process": process, "log": stream, "gpu": gpu, "group": group, "directory": directory})
                print(f"Started screen {group} on GPU {gpu}; PID {process.pid}; log: {directory / 'run.log'}", flush=True)
            if active:
                time.sleep(.1)
    finally:
        phases._stop_gpu_children(active)
    return reports


def main(argv=None):
    args = parse_args(argv)
    context = build_context(args)
    if args.group:
        if not args.execute:
            raise ValueError("Internal --group requires --execute")
        a = student_args(args, context, args.group)
        training.run_student(a, context["splits"], student_contract(context, args.group),
                             context["teacher_state"], context["expected_initial"])
        return 0
    gpus = phases._parse_gpu_ids(args.gpus)
    device = torch.device(context["a"].device)
    if (gpus and device.type != "cuda") or (device.type == "cuda" and not gpus):
        raise ValueError("Supply physical --gpus for CUDA; omit --gpus for a CPU tiny diagnostic")
    if device.type == "cuda" and context["a"].device not in ("cuda", "cuda:0"):
        raise ValueError("Saved device must be cuda or cuda:0 for single-GPU children")
    plan = {"recipe": context["recipe"], "gpus": gpus, "output_root": str(args.output_root),
        "jobs": [GROUP_DIRS[group] for group in "CD"],
        "note": "Reuse verified A/B and teacher. Train fresh C/D with the original budget. Internal COD validation only."}
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path, lock_path = args.output_root / "manifest.json", args.output_root / "screen.lock"
    if manifest_path.exists():
        if not args.resume or _read(manifest_path).get("recipe") != context["recipe"]:
            raise ValueError("Existing screen requires --resume with identical provenance and budget")
    elif any(args.output_root.iterdir()):
        raise FileExistsError("Unrecognized nonempty screen root")
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    old_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("MSE screen launcher interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        runner._save_json(manifest_path, {**plan, "status": "running"})
        reports = _run_jobs(args, context, gpus or [None])
        summary = make_summary(context, reports)
        runner._save_json(args.output_root / "summary.json", summary)
        print("group  selected_epoch  validation_macro_Sm  reused")
        for row in summary["rows"]:
            print(f"{row['label']:24s} {row['selected_epoch']:4d}  {row['validation']['macro']['S_alpha']:.6f}  {row['reused']}")
        print(f"D beats A/B/C: {summary['D_beats_all_ABC']}; {summary['note']}", flush=True)
        runner._save_json(manifest_path, {**plan, "status": "complete"})
    except BaseException as exc:
        runner._save_json(manifest_path, {**plan, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
