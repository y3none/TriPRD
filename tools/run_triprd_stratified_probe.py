#!/usr/bin/env python3
"""Paired VDT-only pooled versus stratum-balanced positive-MSE diagnostics.

Fresh identical students, fixed last epoch, fixed teacher and TRAIN constants.
No COD indexing, checkpoint selection, or automatic production advancement.
"""
from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader
from tools import diagnose_triprd_correction_probe as original
from tools import run_triprd_correction as runner
from tools import run_triprd_optimization_phases as phases
from tools.stratified_correction_loss import positive_stratified_mse_loss, evaluate_source_detailed
from tools.triprd_structure_training import state_digest
from tools.triprd_training import build_scheduler, sha256_file

METHOD = "triprd_stratified_positive_mse_probe_v1"
MSE_COEFFICIENT = 5.0
ARMS = {"pooled_mse": "SSE divided by all positive reliable edges",
        "stratified_mse": "Equal nonempty FG/BG/crossing means; pool images/scales within each stratum"}
CODE_FILES = ("tools/run_triprd_stratified_probe.py", "tools/stratified_correction_loss.py",
              "tools/correction_path_diagnostics.py", "tools/run_triprd_stratified_probe.sh")
_read = original._read


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpus", help="Physical GPU IDs; independent arms, no DDP")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--arm", choices=tuple(ARMS), help=argparse.SUPPRESS)
    return p.parse_args(argv)


def build_context(args):
    context = original.build_context(args)
    if any(context["splits"][key] for key in ("cod_train", "cod_val")):
        raise ValueError("Source-only diagnostic must not contain COD samples")
    context["recipe"] = {"method": METHOD, "reference_recipe": context["recipe"],
        "source_sha256": {path: sha256_file(ROOT / path) for path in CODE_FILES},
        "epochs": args.epochs, "selection": "fixed_last_epoch", "cod_access": False,
        "source_manifest": context["manifest"], "expected_initial_state_sha256": context["probe"]["initial_state_sha256"],
        "mse_coefficient": MSE_COEFFICIENT, "supervision": "unchanged source segmentation, auxiliary and GT relation losses",
        "changed_factor": "positive-edge FG/BG/crossing reduction only",
        "unchanged_factors": ["squared error", "reliability > 0", "teacher delta", "MSE coefficient 5",
                              "within-stratum scale/image edge pooling", "initialization", "data", "budget"],
        "gradient_scale_note": "Same coefficient does not imply matched gradient norms under changed reduction",
        "teacher_and_student_weights_reused": "teacher only; students start fresh"}
    return context


def arm_contract(context, arm):
    return {**context["recipe"], "arm": arm, "reduction": ARMS[arm]}


def _read_checkpoint(path, contract):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("method") != METHOD or state.get("contract") != contract
            or state.get("contract_sha256") != runner._digest(contract)):
        raise ValueError("Stratified-probe checkpoint provenance differs")
    return state


def verify_completed(directory, contract):
    directory = Path(directory)
    report = _read(directory / "comparison.json")
    if (report.get("method") != METHOD or report.get("status") != "complete"
            or report.get("contract") != contract or report.get("contract_sha256") != runner._digest(contract)):
        raise ValueError("Completed stratified-probe report provenance differs")
    if report.get("checkpoint_sha256") != sha256_file(directory / "last.pt"):
        raise ValueError("Stratified-probe checkpoint digest differs")
    state = _read_checkpoint(directory / "last.pt", contract)
    history = state.get("history", [])
    if (state.get("epoch") != contract["epochs"] or report.get("completed_epochs") != contract["epochs"]
            or report.get("selected_epoch") != contract["epochs"]
            or [row.get("epoch") for row in history] != list(range(1, contract["epochs"] + 1))
            or report.get("history") != history or _read(directory / "history.json") != history
            or state.get("initial_state_sha256") != contract["expected_initial_state_sha256"]
            or report.get("initial_state_sha256") != state.get("initial_state_sha256")
            or report.get("source_recoverability") != history[-1].get("source_recoverability")
            or report.get("source_diagnostics") != history[-1].get("source_diagnostics")
            or not report.get("source_train_diagnostics")
            or not report.get("frozen_teacher_verified") or not report.get("frozen_generic_encoder_verified")):
        raise ValueError("Stratified-probe history, final metrics or initialization differs")
    if report.get("cod_access") is not False or (directory / "best.pt").exists():
        raise ValueError("Source-only diagnostic must use a fixed last checkpoint without COD")
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
            pooled = original.positive_mse_loss(pred, targets, batch["mask"], a.relation_kernel)
            stratified = positive_stratified_mse_loss(pred, targets, batch["mask"], a.relation_kernel)
            transfer = MSE_COEFFICIENT * (pooled if arm == "pooled_mse" else stratified)
            total = a.source_weight * (source["supervised"] + a.kd_weight * transfer)
        runner._finite(total, f"stratified probe {arm} epoch={epoch} batch={index}")
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
                raise FloatingPointError("Persistent non-finite gradients in stratified probe")
        runner._add_stats(stats, {"total": total, "source_supervised": source["supervised"],
            "original_source_transfer": source["transfer"], "raw_positive_mse": pooled,
            "raw_stratified_mse": stratified, "optimization_transfer": transfer,
            "source_positive_coverage": source["positive_coverage"]})
        count += 1
    if not updates:
        raise FloatingPointError("No optimizer updates in stratified probe epoch")
    return {**{key: float(value / count) for key, value in stats.items()}, "steps": count,
        "optimizer_steps": updates, "amp_skipped_steps": len(skips), "amp_overflow_events": skips,
        "amp_scale": float(scaler.get_scale()), "grad_norm": float(norms / updates),
        "seconds": time.perf_counter() - started}


def run_arm(args, context):
    from models.triprd_correction import CorrectionStudent, CorrectionTeacher
    arm, a = args.arm, copy.deepcopy(context["a"])
    a.out = args.output_root / arm
    contract = arm_contract(context, arm)
    report_path = a.out / "comparison.json"
    previous = _read(report_path) if report_path.exists() else None
    if previous:
        if not args.resume or previous.get("contract") != contract or previous.get("contract_sha256") != runner._digest(contract):
            raise ValueError("Existing stratified probe requires --resume and an identical contract")
        if previous.get("status") == "complete":
            return verify_completed(a.out, contract)
    elif a.out.exists() and any(path.name != "run.log" for path in a.out.iterdir()):
        raise FileExistsError(f"Unrecognized stratified probe output: {a.out}")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"method": METHOD, "status": "running", "diagnostic": True, "formal": False,
        "arm": arm, "contract": contract, "contract_sha256": runner._digest(contract),
        "config": runner._json(a), "cod_access": False, "selection_metric": "fixed_last_epoch",
        "validation_used_for_selection": False, "sample_counts": {key: len(items) for key, items in context["splits"].items()},
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "training_loss_aggregation": "Online minibatch means; final TRAIN/VAL diagnostics pool fixed-model edges without augmentation",
        "warning": "New source-only development comparison; no automatic COD screen or final run"}
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
        if initial_digest != context["probe"]["initial_state_sha256"]:
            raise ValueError("Fresh student initialization differs from the original source probe")
        loaders = runner._loaders(a, context["splits"], device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
        scheduler = build_scheduler(optimizer, "cosine", a.epochs, min(a.warmup_epochs, a.epochs - 1), .05)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
        start, history = 1, []
        if args.resume and (a.out / "last.pt").exists():
            saved = _read_checkpoint(a.out / "last.pt", contract)
            if saved["initial_state_sha256"] != initial_digest:
                raise ValueError("Resumed student initial state differs")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            runner._restore_rng(saved["rng"], loaders)
            start, history, initial = saved["epoch"] + 1, saved["history"], saved["initial_source_recoverability"]
        else:
            # Identical initial validation and RNG consumption to the original
            # joint_positive_mse control; no additional initial loader pass.
            initial = runner._student_probe(model, teacher, loaders["vdt_val"], constants, device, a)
        report.update(initial_state_sha256=initial_digest, initial_source_recoverability=initial)
        for epoch in range(start, a.epochs + 1):
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                "train": _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, arm, epoch)}
            if epoch % a.val_metrics_every == 0 or epoch == a.epochs:
                evaluation = evaluate_source_detailed(model, teacher, loaders["vdt_val"], constants, device, a)
                row.update(source_recoverability=evaluation["source_recoverability"],
                           source_diagnostics=evaluation["path_statistics"])
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
            detail = f" validation_positive_mse={row['source_recoverability']['positive_mse']:.8f}" if "source_recoverability" in row else ""
            print(f"[{arm}] {epoch:03d}/{a.epochs} loss={row['train']['total']:.6f} "
                  f"updates={row['train']['optimizer_steps']} amp_skips={row['train']['amp_skipped_steps']}{detail}", flush=True)
        # A fixed-model TRAIN pass distinguishes fit from held-out recovery.
        # Its independent generator never changes the training-loader state.
        dataset = runner.CorrectionDataset(context["splits"]["vdt_train"], a.image_size,
                                           normalize=a.pretrained, augment=False)
        train_eval = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(a.train_seed + 70001))
        final_train = evaluate_source_detailed(model, teacher, train_eval, constants, device, a)
        if state_digest(model.encoder) != frozen_encoder or state_digest(teacher) != frozen_teacher:
            raise RuntimeError("Frozen encoder or teacher changed")
        report.update(status="complete", completed_epochs=a.epochs, selected_epoch=a.epochs, history=history,
            source_recoverability=history[-1]["source_recoverability"], source_diagnostics=history[-1]["source_diagnostics"],
            source_train_diagnostics=final_train, source_train_evaluation="fixed last checkpoint; no augmentation; all VDT TRAIN",
            frozen_generic_encoder_verified=True, frozen_teacher_verified=True,
            checkpoint_sha256=sha256_file(a.out / "last.pt"))
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        runner._save_json(report_path, report)
    return report


def summary_row(name, report):
    def extract(probe, stats):
        result = {"global": probe, "strata": {}}
        for region in ("fg", "bg", "crossing"):
            rec = f"VDT/recovery/pooled/{region}"
            rel = f"VDT/relation/pooled/{region}"
            values = {key: stats[f"{rec}/{key}"]["mean"] for key in
                      ("positive_mse", "positive_zero_mse", "positive_constant_mse", "positive_coverage",
                       "teacher_rgb_bce", "teacher_rgb_plus_student_bce", "student_base_plus_teacher_delta_bce",
                       "positive_base_bce", "positive_corrected_bce", "positive_teacher_rgb_bce",
                       "positive_teacher_full_bce", "positive_teacher_rgb_plus_student_bce",
                       "positive_student_base_plus_teacher_delta_bce")}
            values.update({key: stats[f"{rel}/{key}"]["mean"] for key in ("base_bce", "corrected_bce", "clip_fraction")})
            values.update(positive_count=stats[f"{rec}/positive_mse"]["count"], all_count=stats[f"{rel}/base_bce"]["count"])
            result["strata"][region] = values
        return result
    return {"arm": name, "selected_epoch": report["selected_epoch"],
        "initial_state_sha256": report["initial_state_sha256"],
        "val": extract(report["source_recoverability"], report["source_diagnostics"]),
        "train": extract(report["source_train_diagnostics"]["source_recoverability"],
                         report["source_train_diagnostics"]["path_statistics"])}


def summarize(reports):
    rows = {arm: summary_row(arm, reports[arm]) for arm in ARMS}
    if len({row["initial_state_sha256"] for row in rows.values()}) != 1:
        raise ValueError("Paired students have different initialization")
    differences = {}
    for split in ("train", "val"):
        pooled, balanced = (rows[arm][split] for arm in ARMS)
        differences[split] = {"global_positive_mse": balanced["global"]["positive_mse"] - pooled["global"]["positive_mse"], "strata": {}}
        for region in ("fg", "bg", "crossing"):
            left, right = pooled["strata"][region], balanced["strata"][region]
            if any(left[key] != right[key] for key in ("positive_count", "all_count")):
                raise ValueError("Paired validation support differs")
            differences[split]["strata"][region] = {key: right[key] - left[key]
                if right[key] is not None and left[key] is not None else None
                for key in ("positive_mse", "corrected_bce", "positive_corrected_bce")}
    return {"method": METHOD, "diagnostic": True, "rows": rows,
        "stratified_minus_pooled": differences, "automatic_advance": False,
        "note": "Lower MSE/BCE differences are better; source recovery is not COD performance. Inspect strata and counterfactual BCE; no new automatic gate."}


def summary_text(summary):
    def fmt(value):
        return "null" if value is None else f"{value:.8g}"
    lines = [summary["note"]]
    for arm, row in summary["rows"].items():
        lines += ["", f"[{arm}] fixed_epoch={row['selected_epoch']}"]
        for split in ("train", "val"):
            data = row[split]
            lines.append(f"{split} global " + " ".join(f"{key}={fmt(data['global'][key])}" for key in
                         ("positive_mse", "positive_zero_mse", "positive_constant_mse")))
            for region, values in data["strata"].items():
                lines.append(f"{split} {region} " + " ".join(f"{key}={fmt(value)}" for key, value in values.items()))
    lines += ["", "stratified_minus_pooled=" + json.dumps(summary["stratified_minus_pooled"], allow_nan=False)]
    return "\n".join(lines) + "\n"


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
                    print(f"Completed {item['arm']}", flush=True)
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
        raise ValueError("Single-device child requires saved device cuda or cuda:0")
    plan = {"recipe": context["recipe"], "arms": ARMS, "gpus": gpus, "output_root": str(args.output_root)}
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
        raise KeyboardInterrupt("Stratified-probe launcher interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        runner._save_json(manifest_path, {**plan, "status": "running"})
        reports = _run_jobs(args, context, gpus or [None])
        summary = summarize(reports)
        runner._save_json(args.output_root / "summary.json", summary)
        (args.output_root / "summary.txt").write_text(summary_text(summary), encoding="utf-8")
        runner._save_json(manifest_path, {**plan, "status": "complete"})
        print(summary_text(summary), flush=True)
    except BaseException as error:
        runner._save_json(manifest_path, {**plan, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
