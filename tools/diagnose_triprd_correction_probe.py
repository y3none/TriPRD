#!/usr/bin/env python3
"""Controlled source-only loss diagnostics; never advance the production gates.

Reuse a completed, provenance-verified teacher and source split, then initialize
four fresh identical students. No diagnostic checkpoint is a formal COD result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from tools import run_triprd_correction as runner
from tools import run_triprd_optimization_phases as phases
from tools.triprd_structure_training import state_digest
from tools.triprd_training import build_scheduler, sha256_file

METHOD = "triprd_correction_probe_diagnostic_v1"
MSE_COEFFICIENT = 5.0
ARMS = {
    "joint_balanced": {"supervision": True, "transfer": "original_balanced_smooth_l1"},
    "joint_positive_mse": {"supervision": True, "transfer": "global_positive_mse"},
    "isolated_balanced": {"supervision": False, "transfer": "original_balanced_smooth_l1"},
    "isolated_positive_mse": {"supervision": False, "transfer": "global_positive_mse"},
}


def positive_mse_loss(output, targets, mask, relation_kernel=3):
    """Differentiable SSE / positive count, pooled over scales, images and edges."""
    from models.triprd_correction_loss import _edges, _sequence, annotation_relations
    predicted = _sequence(output["corrections"])
    supplied = output.get("relation_valid_masks")
    if len(predicted) != len(targets["delta"]) or (supplied is not None and len(supplied) != len(predicted)):
        raise ValueError("student/teacher validity and scale counts must match")
    squared, count = [], []
    for index, raw in enumerate(predicted):
        value = _edges(raw, relation_kernel)
        delta = targets["delta"][index].detach()
        if value.shape != delta.shape:
            raise ValueError("student/teacher correction grids must match")
        _, valid, _ = annotation_relations(mask, tuple(value.shape[-2:]), relation_kernel,
                                          None if supplied is None else supplied[index])
        positive = valid & targets["valid"][index].detach().bool() & (targets["reliability"][index].detach() > 0)
        squared.append((value - delta).square().masked_fill(~positive, 0).sum())
        count.append(positive.expand_as(value).sum())
    return torch.stack(squared).sum() / torch.stack(count).sum().clamp_min(1)


def diagnostic_objective(source, positive_mse, arm, source_weight, kd_weight):
    policy = ARMS[arm]
    transfer = MSE_COEFFICIENT * positive_mse if policy["transfer"] == "global_positive_mse" else source["transfer"]
    # Do not include supervised * 0: task-only parameters must have no gradient,
    # avoiding AdamW updates to heads that are absent from isolated objectives.
    return source_weight * (kd_weight * transfer + (source["supervised"] if policy["supervision"] else 0))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True, help="Existing completed teacher/probe seed directory")
    p.add_argument("--output-root", type=Path, required=True, help="New disjoint diagnostics directory")
    p.add_argument("--gpus", help="Physical GPU IDs; one independent student per GPU")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--execute", action="store_true", help="Without this flag, only verify and print the plan")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--arm", choices=tuple(ARMS), help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_report(report, stage):
    if report.get("method") != runner.METHOD or report.get("stage") != stage or report.get("status") != "complete":
        raise ValueError(f"A completed correction_v3 {stage} report is required")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or report.get("contract_sha256") != runner._digest(protocol):
        raise ValueError(f"Invalid {stage} report contract digest")
    current = {name: sha256_file(ROOT / name) for name in runner.CODE_FILES}
    if protocol.get("source_sha256") != current:
        raise ValueError(f"{stage} source fingerprints differ; restore the exact production source before diagnostics")


def load_reference(run_root):
    run_root = Path(run_root).expanduser().resolve()
    probe, teacher = (_read(run_root / stage / "comparison.json") for stage in ("probe", "teacher"))
    _validate_report(probe, "probe")
    _validate_report(teacher, "teacher")
    a = runner.parse_args(["--out", str(run_root / "probe")])
    for key, value in probe["config"].items():
        if not hasattr(a, key):
            raise ValueError(f"Unknown saved probe option: {key}")
        setattr(a, key, value)
    for key in ("out", "dino_repo", "dino_weights", "split_manifest", "teacher_checkpoint", "cod_checkpoint"):
        value = getattr(a, key)
        setattr(a, key, Path(value) if value else None)
    if a.stage != "probe" or a.group != "D" or a.prepare_only or a.eval_only:
        raise ValueError("Reference must be the original source-only group-D probe")
    if a.out.expanduser().resolve() != run_root / "probe" or a.teacher_checkpoint is None or a.teacher_checkpoint.expanduser().resolve() != run_root / "teacher" / "best.pt":
        raise ValueError("Saved probe paths do not match the requested run root")
    runner.validate_args(a)
    return a, probe, teacher


def build_context(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.run_root, args.output_root = args.run_root.expanduser().resolve(), args.output_root.expanduser().resolve()
    if args.run_root == args.output_root or args.run_root in args.output_root.parents or args.output_root in args.run_root.parents:
        raise ValueError("Diagnostics output must be disjoint from the original run root")
    a, probe, teacher = load_reference(args.run_root)
    splits, manifest = runner.prepare_data(a)  # stage=probe: VDT only, never indexes COD.
    resources = runner._resources(a)
    expected_teacher, expected_probe = runner._contracts(a, manifest, resources)
    if teacher["protocol"] != expected_teacher or probe["protocol"] != expected_probe:
        raise ValueError("Saved source membership, resources or configuration differ from the current run")
    for report, path, contract in ((teacher, a.teacher_checkpoint, expected_teacher),
                                   (probe, args.run_root / "probe/last.pt", expected_probe)):
        if report.get("checkpoint_sha256") != sha256_file(path):
            raise ValueError(f"Completed checkpoint digest differs: {path}")
        state = runner._read_checkpoint(path, contract)
        if state.get("stage") != contract["stage"]:
            raise ValueError("Checkpoint stage differs from its contract")
        if contract["stage"] == "probe" and (state.get("epoch") != a.epochs or probe.get("completed_epochs") != a.epochs):
            raise ValueError("Original probe did not complete its configured budget")
        del state
    teacher_state = runner._read_checkpoint(a.teacher_checkpoint, expected_teacher)
    if not teacher_state.get("constant_correction") or not teacher_state.get("positive_constant_correction"):
        raise ValueError("Teacher must contain both fixed TRAIN-fitted correction constants")
    phases.teacher_gate(teacher)
    original_config = runner._json(a)
    a.epochs, a.resume = args.epochs, args.resume
    recipe = {
        "method": METHOD, "diagnostic_script_sha256": sha256_file(Path(__file__)),
        "scheduler_support_sha256": sha256_file(Path(phases.__file__)),
        "original_probe_report_sha256": sha256_file(args.run_root / "probe/comparison.json"),
        "teacher_report_sha256": sha256_file(args.run_root / "teacher/comparison.json"),
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "original_protocol": expected_probe, "original_config": original_config,
        "epochs": a.epochs, "selection": "fixed_last_epoch", "cod_access": False,
        "mse_coefficient": MSE_COEFFICIENT,
        "mse_coefficient_rationale": "Fixed design choice 1/(2*0.1); matches only the local SmoothL1 quadratic coefficient, not balanced loss or gradient magnitudes",
        "source_manifest": manifest,
    }
    return {"a": a, "probe": probe, "teacher": teacher, "teacher_state": teacher_state,
            "splits": splits, "manifest": manifest, "recipe": recipe}


def arm_contract(context, arm):
    return {**context["recipe"], "arm": arm, "policy": ARMS[arm]}


def _read_diagnostic_checkpoint(path, contract):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("method") != METHOD or state.get("contract") != contract or state.get("contract_sha256") != runner._digest(contract):
        raise ValueError("Diagnostic checkpoint provenance differs")
    return state


def verify_completed(directory, contract):
    report = _read(directory / "comparison.json")
    if report.get("method") != METHOD or report.get("status") != "complete" or report.get("contract") != contract or report.get("contract_sha256") != runner._digest(contract):
        raise ValueError("Completed diagnostic report provenance differs")
    if report.get("checkpoint_sha256") != sha256_file(directory / "last.pt"):
        raise ValueError("Diagnostic checkpoint digest differs")
    state = _read_diagnostic_checkpoint(directory / "last.pt", contract)
    if state.get("epoch") != contract["epochs"] or report.get("completed_epochs") != contract["epochs"]:
        raise ValueError("Diagnostic did not reach its predefined final epoch")
    if report.get("initial_state_sha256") != state.get("initial_state_sha256") or report.get("source_recoverability") != state["history"][-1]["source_recoverability"]:
        raise ValueError("Diagnostic report differs from its checkpoint")
    return report


def _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, arm, epoch):
    from models.triprd_correction_loss import source_loss, correction_targets
    model.train()
    teacher.eval()
    for loader in loaders.values():
        loader.dataset.epoch = epoch
    started, stats, count, updates, skips, consecutive = time.perf_counter(), {}, 0, 0, [], 0
    norms = torch.zeros((), device=device)
    for index, batch in enumerate(loaders["vdt_train"], 1):
        batch = runner.shared._move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with runner._autocast(a, device):
            with torch.no_grad():
                coals = teacher.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
            pred = model(batch["rgb"], task="sod")
            source = source_loss(pred, batch["mask"], teacher_rgb=coals["R"], teacher_full=coals["RTD"],
                mode="correction", relation_kernel=a.relation_kernel, relation_weight=a.relation_weight, kd_weight=a.kd_weight)
            targets = correction_targets(coals["R"], coals["RTD"], batch["mask"], a.relation_kernel)
            mse = positive_mse_loss(pred, targets, batch["mask"], a.relation_kernel)
            total = diagnostic_objective(source, mse, arm, a.source_weight, a.kd_weight)
        runner._finite(total, f"diagnostic {arm} epoch={epoch} batch={index}")
        scaler.scale(total).backward()
        norm, overflow = runner._optimizer_step(model, optimizer, scaler, a.grad_clip)
        if overflow is None:
            updates, consecutive = updates + 1, 0
            norms += norm
        else:
            consecutive += 1
            overflow.update(batch=index, primary_keys=list(batch.get("key", [])))
            skips.append(overflow)
            print(f"[{arm}] AMP skipped update: {overflow}", flush=True)
            if consecutive >= runner.MAX_CONSECUTIVE_AMP_SKIPS:
                raise FloatingPointError("Persistent non-finite gradients in diagnostic")
        optimization_transfer = MSE_COEFFICIENT * mse if ARMS[arm]["transfer"] == "global_positive_mse" else source["transfer"]
        runner._add_stats(stats, {"total": total, "source_supervised": source["supervised"],
            "source_transfer": source["transfer"], "raw_positive_mse": mse,
            "optimization_transfer": optimization_transfer,
            "source_positive_coverage": source["positive_coverage"]})
        count += 1
    if not updates:
        raise FloatingPointError("No optimizer updates in diagnostic epoch")
    return {**{key: float(value / count) for key, value in stats.items()}, "steps": count,
        "optimizer_steps": updates, "amp_skipped_steps": len(skips), "amp_overflow_events": skips,
        "amp_scale": float(scaler.get_scale()), "grad_norm": float(norms / updates),
        "seconds": time.perf_counter() - started}


def run_arm(args, context):
    from models.triprd_correction import CorrectionStudent, CorrectionTeacher
    arm, a = args.arm, context["a"]
    a.out = args.output_root / arm
    contract = arm_contract(context, arm)
    report_path = a.out / "comparison.json"
    previous = _read(report_path) if report_path.exists() else None
    if previous:
        if not args.resume or previous.get("contract") != contract or previous.get("contract_sha256") != runner._digest(contract):
            raise ValueError("Existing diagnostic requires --resume with an identical contract")
        if previous.get("status") == "complete":
            return verify_completed(a.out, contract)
    elif a.out.exists() and any(p.name != "run.log" for p in a.out.iterdir()):
        raise FileExistsError(f"Refusing unrecognized diagnostic output: {a.out}")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"method": METHOD, "status": "running", "diagnostic": True, "formal": False,
        "arm": arm, "contract": contract, "contract_sha256": runner._digest(contract),
        "config": runner._json(a), "cod_access": False, "selection_metric": "fixed_last_epoch",
        "validation_used_for_selection": False, "sample_counts": {k: len(v) for k, v in context["splits"].items()},
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_loss_aggregation": "Means over minibatches; raw_positive_mse pools edges within each minibatch, while validation pools all edges across the dataset",
        "warning": "Source-only diagnostic, not COD performance or authorization to advance production gates"}
    runner._save_json(report_path, report)
    try:
        state = context["teacher_state"]
        constants = {"all": state["constant_correction"], "positive": state["positive_constant_correction"]}
        teacher = CorrectionTeacher(**runner._model_options(a, pretrained=False)).to(device)
        teacher.load_state_dict(state["model"], strict=True)
        teacher.requires_grad_(False).eval()
        frozen_teacher = state_digest(teacher)
        runner.shared.seed_everything(a.train_seed)
        model = CorrectionStudent(**runner._model_options(a)).to(device)
        initial_digest, frozen_encoder = state_digest(model), state_digest(model.encoder)
        if initial_digest != context["probe"].get("initial_state_sha256"):
            raise ValueError("Fresh student initialization differs from the completed original probe")
        loaders = runner._loaders(a, context["splits"], device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
        scheduler = build_scheduler(optimizer, "cosine", a.epochs, min(a.warmup_epochs, a.epochs - 1), .05)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
        start, history = 1, []
        if args.resume and (a.out / "last.pt").exists():
            saved = _read_diagnostic_checkpoint(a.out / "last.pt", contract)
            if saved["initial_state_sha256"] != initial_digest:
                raise ValueError("Resumed student initial state differs")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            runner._restore_rng(saved["rng"], loaders)
            start, history, initial = saved["epoch"] + 1, saved["history"], saved["initial_source_recoverability"]
        else:
            initial = runner._student_probe(model, teacher, loaders["vdt_val"], constants, device, a)
        report.update(initial_state_sha256=initial_digest, initial_source_recoverability=initial)
        for epoch in range(start, a.epochs + 1):
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                "train": _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, arm, epoch)}
            if epoch % a.val_metrics_every == 0 or epoch == a.epochs:
                row["source_recoverability"] = runner._student_probe(model, teacher, loaders["vdt_val"], constants, device, a)
            scheduler.step()
            history.append(row)
            checkpoint = {"method": METHOD, "diagnostic": True, "contract": contract,
                "contract_sha256": runner._digest(contract), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "history": history, "initial_state_sha256": initial_digest,
                "initial_source_recoverability": initial, "rng": runner._rng_state(loaders)}
            runner._save_pt(a.out / "last.pt", checkpoint)
            runner._save_json(a.out / "history.json", history)
            report.update(completed_epochs=epoch, history=history)
            runner._save_json(report_path, report)
            validation = row.get("source_recoverability")
            detail = (f" val_positive_mse={validation['positive_mse']:.8f} "
                      f"zero={validation['positive_zero_mse']:.8f} constant={validation['positive_constant_mse']:.8f}"
                      if validation else "")
            print(f"[{arm} DIAGNOSTIC] {epoch:03d}/{a.epochs} loss={row['train']['total']:.6f} "
                  f"train_positive_mse={row['train']['raw_positive_mse']:.6f} "
                  f"updates={row['train']['optimizer_steps']} amp_skips={row['train']['amp_skipped_steps']}{detail}", flush=True)
        if state_digest(model.encoder) != frozen_encoder or state_digest(teacher) != frozen_teacher:
            raise RuntimeError("Frozen encoder or teacher changed during diagnostics")
        report.update(status="complete", completed_epochs=a.epochs, selected_epoch=a.epochs, history=history,
            source_recoverability=history[-1]["source_recoverability"],
            frozen_generic_encoder_verified=True, frozen_teacher_verified=True,
            checkpoint_sha256=sha256_file(a.out / "last.pt"))
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        runner._save_json(report_path, report)
    return report


def _summary_row(name, report):
    metrics = report["source_recoverability"]
    try:
        phases.student_gate(report)
        passed = True
    except phases.EvidenceBlocked:
        passed = False
    return {"arm": name, "initial_positive_mse": report["initial_source_recoverability"]["positive_mse"],
            **{key: metrics[key] for key in ("positive_mse", "positive_zero_mse", "positive_constant_mse")}, "passes_original_gate": passed}


def _run_jobs(args, context, gpus):
    active, pending, results = [], list(ARMS), {}
    try:
        while pending or active:
            finished = [(item, item["process"].poll()) for item in active]
            failures = [(item, code) for item, code in finished if code not in (None, 0)]
            if failures:
                item, code = failures[0]
                raise RuntimeError(f"{item['arm']} exited {code}; see {item['directory'] / 'run.log'}")
            for item, code in finished:
                if code is not None:
                    item["process"].wait()
                    results[item["arm"]] = verify_completed(item["directory"], arm_contract(context, item["arm"]))
                    item["log"].close()
                    active.remove(item)
                    print(f"Completed diagnostic {item['arm']}", flush=True)
            available = [gpu for gpu in gpus if all(item["gpu"] != gpu for item in active)]
            while pending and available:
                arm, gpu = pending.pop(0), available.pop(0)
                directory = args.output_root / arm
                if args.resume and (directory / "comparison.json").exists() and _read(directory / "comparison.json").get("status") == "complete":
                    results[arm] = verify_completed(directory, arm_contract(context, arm))
                    available.append(gpu)
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                stream = (directory / "run.log").open("a", encoding="utf-8", buffering=1)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-root", str(args.run_root),
                    "--output-root", str(args.output_root), "--epochs", str(args.epochs), "--arm", arm, "--execute"]
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
                active.append({"process": process, "log": stream, "gpu": gpu, "arm": arm, "directory": directory})
                print(f"Started {arm} on GPU {gpu}; PID {process.pid}; log: {directory / 'run.log'}", flush=True)
            if active:
                time.sleep(.1)
    finally:
        phases._stop_gpu_children(active)
    return results


def main(argv=None):
    args = parse_args(argv)
    context = build_context(args)
    if args.arm:
        if not args.execute:
            raise ValueError("Internal --arm requires --execute")
        run_arm(args, context)
        return 0
    gpus = phases._parse_gpu_ids(args.gpus)
    device = torch.device(context["a"].device)
    if gpus and device.type != "cuda":
        raise ValueError("GPU scheduling requires a CUDA reference config")
    if device.type == "cuda" and not gpus:
        raise ValueError("Supply --gpus with physical GPU identifiers")
    if device.type == "cuda" and context["a"].device not in ("cuda", "cuda:0"):
        raise ValueError("A single-device child requires saved device cuda or cuda:0")
    plan = {"recipe": context["recipe"], "arms": ARMS, "gpus": gpus, "output_root": str(args.output_root),
            "warning": "VDT-only diagnostic; production gates and original artifacts remain unchanged"}
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path, lock_path = args.output_root / "manifest.json", args.output_root / "diagnostic.lock"
    if manifest_path.exists():
        if not args.resume or _read(manifest_path).get("recipe") != context["recipe"]:
            raise ValueError("Existing diagnostics require --resume with identical provenance and budget")
    elif any(args.output_root.iterdir()):
        raise FileExistsError("Unrecognized nonempty diagnostics root")
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    old_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Diagnostic launcher interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        runner._save_json(manifest_path, {**plan, "status": "running"})
        reports = _run_jobs(args, context, gpus or [None])
        rows = [_summary_row("original_probe_reference", context["probe"])]
        rows += [_summary_row(arm, reports[arm]) for arm in ARMS]
        summary = {"method": METHOD, "diagnostic": True, "rows": rows, "warning": plan["warning"]}
        runner._save_json(args.output_root / "summary.json", summary)
        print("arm                            initial       final         zero          constant      gate")
        for row in rows:
            print(f"{row['arm']:30s} {row['initial_positive_mse']:.8f}  {row['positive_mse']:.8f}  "
                  f"{row['positive_zero_mse']:.8f}  {row['positive_constant_mse']:.8f}  {row['passes_original_gate']}")
        runner._save_json(manifest_path, {**plan, "status": "complete"})
        print(plan["warning"], flush=True)
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
