"""Versioned COD development training for the positive-edge MSE experiment.

Only C/D students are trained here. Existing v3 baselines and teachers retain
their original provenance; diagnostic student weights are never loaded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch

from tools import run_triprd_correction as runner
from tools import diagnose_triprd_correction_probe as diagnostic
from tools.triprd_structure_training import state_digest
from tools.triprd_training import build_scheduler, sha256_file

METHOD = "triprd_correction_positive_mse_screen_v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_checkpoint(path, contract=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("method") != METHOD or state.get("contract_sha256") != runner._digest(state.get("contract")):
        raise ValueError("Not a valid positive-MSE screen checkpoint")
    if contract is not None and state.get("contract") != contract:
        raise ValueError("Positive-MSE screen checkpoint contract differs")
    return state


def verify_completed(directory, contract):
    """Check both epoch-boundary and selected checkpoints against the report."""
    directory = Path(directory)
    report = _read(directory / "comparison.json")
    if (report.get("method") != METHOD or report.get("status") != "complete"
            or report.get("protocol") != contract
            or report.get("contract_sha256") != runner._digest(contract)):
        raise ValueError("Completed screen report provenance differs")
    if report.get("test_status") != "skipped" or report.get("test") != {}:
        raise ValueError("Development screen must not contain official test results")
    if not report.get("frozen_generic_encoder_verified") or not report.get("frozen_teacher_verified"):
        raise ValueError("Screen report lacks frozen encoder/teacher checks")
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    if (report.get("checkpoint_sha256") != sha256_file(best_path)
            or report.get("last_checkpoint_sha256") != sha256_file(last_path)):
        raise ValueError("Completed screen checkpoint digest differs")
    best, last = read_checkpoint(best_path, contract), read_checkpoint(last_path, contract)
    epochs = contract["epochs"]
    history = last.get("history", [])
    if (last.get("epoch") != epochs or report.get("completed_epochs") != epochs
            or [row.get("epoch") for row in history] != list(range(1, epochs + 1))
            or report.get("history") != history):
        raise ValueError("Screen history or completed training budget differs")
    initial = report.get("initial_state_sha256")
    if not initial or initial != last.get("initial_state_sha256") or initial != best.get("initial_state_sha256"):
        raise ValueError("Screen initialization evidence differs")
    expected_initial = contract.get("recipe", {}).get("expected_initial_state_sha256")
    if expected_initial is not None and initial != expected_initial:
        raise ValueError("Screen initialization differs from its matched baseline contract")
    best_epoch, best_score = last.get("best_epoch"), last.get("best_score")
    if (best.get("epoch") != best_epoch or best.get("best_epoch") != best_epoch
            or report.get("best_epoch") != best_epoch
            or best.get("best_score") != best_score or report.get("best_score") != best_score
            or not isinstance(best_epoch, int) or not 1 <= best_epoch <= epochs
            or best.get("history") != history[:best_epoch]):
        raise ValueError("Screen selected checkpoint differs from the final history")
    validation_rows = [(row["epoch"], row["validation"]["macro"]["S_alpha"])
                       for row in history if "validation" in row]
    if not validation_rows or any(not math.isfinite(score) for _, score in validation_rows):
        raise ValueError("Screen lacks finite COD validation scores")
    expected_epoch, expected_score = max(validation_rows, key=lambda item: item[1])
    if (best_epoch, best_score) != (expected_epoch, expected_score):
        raise ValueError("Screen selected checkpoint is not its earliest maximum validation Sm")
    if report.get("selected_validation") != history[best_epoch - 1]["validation"]:
        raise ValueError("Selected validation differs from the checkpoint history")
    for state in (best, last):
        if state.get("stage") != "student" or state.get("group") != contract["group"]:
            raise ValueError("Screen checkpoint stage/group differs")
    return report


def train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch):
    """Keep the v3 COD-primary/source-cycling schedule, replacing only D's KD."""
    from models.triprd_correction_loss import source_loss, correction_targets
    if a.stage != "student" or a.group not in {"C", "D"}:
        raise ValueError("Positive-MSE screen trains only C/D students")
    model.train()
    teacher.eval()
    for loader in loaders.values():
        loader.dataset.epoch = epoch
    primary = loaders["cod_train"]
    source_iter = iter(loaders["vdt_train"])
    stats, count, source_count, updates, consecutive = {}, 0, 0, 0, 0
    norm_sum, overflows = torch.zeros((), device=device), []
    started = time.perf_counter()
    for batch_index, batch in enumerate(primary, 1):
        batch = runner.shared._move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with runner._autocast(a, device):
            output = model(batch["rgb"], task="cod")
            target = source_loss(output, batch["mask"], mode="none",
                                 relation_kernel=a.relation_kernel, relation_weight=a.relation_weight)
            cod_total = target["total"]
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(loaders["vdt_train"])
                source_batch = next(source_iter)
            source_batch = runner.shared._move(source_batch, device)
            source_count += source_batch["rgb"].shape[0]
            with torch.no_grad():
                coals = teacher.forward_coalitions(source_batch["rgb"], source_batch["thermal"], source_batch["depth"])
            prediction = model(source_batch["rgb"], task="sod")
            source = source_loss(prediction, source_batch["mask"], teacher_rgb=coals["R"],
                teacher_full=coals["RTD"], mode="ordinary" if a.group == "C" else "correction",
                relation_kernel=a.relation_kernel, relation_weight=a.relation_weight, kd_weight=a.kd_weight)
            transfer = source["transfer"]
            raw_mse = None
            if a.group == "D":
                targets = correction_targets(coals["R"], coals["RTD"], source_batch["mask"], a.relation_kernel)
                raw_mse = diagnostic.positive_mse_loss(prediction, targets, source_batch["mask"], a.relation_kernel)
                transfer = diagnostic.MSE_COEFFICIENT * raw_mse
            total = cod_total + a.source_weight * (source["supervised"] + a.kd_weight * transfer)
        runner._finite(total, f"screen {a.group} epoch={epoch} batch={batch_index}")
        scaler.scale(total).backward()
        norm, overflow = runner._optimizer_step(model, optimizer, scaler, a.grad_clip)
        if overflow is None:
            updates, consecutive = updates + 1, 0
            norm_sum += norm
        else:
            consecutive += 1
            overflow.update(batch=batch_index, primary_keys=list(batch.get("key", [])),
                            source_keys=list(source_batch.get("key", [])))
            overflows.append(overflow)
            print(f"[screen/{a.group}] AMP skipped update: {overflow}", flush=True)
            if consecutive >= runner.MAX_CONSECUTIVE_AMP_SKIPS:
                raise FloatingPointError("Persistent non-finite gradients in positive-MSE screen")
        logged = {"total": total, "cod_total": cod_total, "cod_segmentation": target["segmentation"],
            "source_supervised": source["supervised"], "source_transfer": transfer,
            "original_source_transfer": source["transfer"],
            "source_positive_coverage": source["positive_coverage"],
            "source_reliability_mean": source["reliability_mean"]}
        if raw_mse is not None:
            logged["raw_positive_mse"] = raw_mse
        runner._add_stats(stats, logged)
        count += 1
    if not updates:
        raise FloatingPointError("No optimizer updates in screen epoch")
    elapsed = time.perf_counter() - started
    return {**{key: float(value / count) for key, value in stats.items()}, "steps": count,
        "optimizer_steps": updates, "amp_skipped_steps": len(overflows), "amp_overflow_events": overflows,
        "amp_scale": float(scaler.get_scale()), "grad_norm": float(norm_sum / updates),
        "primary_images": len(primary.dataset), "source_images": source_count,
        "seconds": elapsed, "images_per_second": len(primary.dataset) / elapsed}


def fit(model, teacher, loaders, splits, a, device, contract, report):
    """Matched student budget and COD validation selection, with exact resume."""
    if a.stage != "student" or a.cod_train_mode != "split" or not a.skip_test:
        raise ValueError("Only internal COD development selection is supported")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    scheduler = build_scheduler(optimizer, "cosine", a.epochs, min(a.warmup_epochs, a.epochs - 1), .05)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
    start, best, best_epoch, history = 1, None, None, []
    initial = state_digest(model)
    if a.resume and (a.out / "last.pt").exists():
        saved = read_checkpoint(a.out / "last.pt", contract)
        if saved.get("initial_state_sha256") != initial:
            raise ValueError("Resumed student did not originate from the matched fresh initialization")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        runner._restore_rng(saved["rng"], loaders)
        start, best, best_epoch, history = saved["epoch"] + 1, saved["best_score"], saved["best_epoch"], saved["history"]
        best_path = a.out / "best.pt"
        saved_best = read_checkpoint(best_path, contract)["epoch"] if best_path.exists() else None
        if best_epoch is not None and saved_best != best_epoch:
            if saved["epoch"] != best_epoch:
                raise RuntimeError("Selected screen checkpoint is missing and cannot be reconstructed")
            runner._save_pt(best_path, saved)
    report["initial_state_sha256"] = initial
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, a.epochs + 1):
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train": train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch)}
        selected = False
        if epoch % a.val_metrics_every == 0 or epoch == a.epochs:
            row["validation"] = runner._validate_cod(model, splits["cod_val"], device, a)
            score = row["validation"]["macro"]["S_alpha"]
            if not math.isfinite(score):
                raise FloatingPointError("Non-finite COD validation Sm")
            if best is None or score > best:
                best, best_epoch, selected = score, epoch, True
        scheduler.step()
        history.append(row)
        state = {"method": METHOD, "stage": "student", "group": a.group,
            "contract": contract, "contract_sha256": runner._digest(contract), "config": runner._json(a),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best, "best_epoch": best_epoch,
            "history": history, "initial_state_sha256": initial, "rng": runner._rng_state(loaders)}
        runner._save_pt(a.out / "last.pt", state)
        if selected:
            runner._save_pt(a.out / "best.pt", state)
        runner._save_json(a.out / "history.json", history)
        report.update(completed_epochs=epoch, best_epoch=best_epoch, best_score=best, latest=row, history=history)
        runner._save_json(a.out / "comparison.json", report)
        print(f"[screen/{a.group}] {epoch:03d}/{a.epochs} loss={row['train']['total']:.5f} "
              f"time={row['train']['seconds']:.1f}s selected={best_epoch} "
              f"updates={row['train']['optimizer_steps']} amp_skips={row['train']['amp_skipped_steps']} "
              f"scale={row['train']['amp_scale']:g}", flush=True)
    best_path = a.out / "best.pt"
    if not best_path.exists() or read_checkpoint(best_path, contract)["epoch"] != best_epoch:
        last = read_checkpoint(a.out / "last.pt", contract)
        if last["epoch"] != best_epoch:
            raise RuntimeError("Selected screen checkpoint is missing; refusing to substitute another epoch")
        runner._save_pt(best_path, last)
    selected = read_checkpoint(best_path, contract)
    model.load_state_dict(selected["model"], strict=True)
    report.update(completed_epochs=a.epochs, best_epoch=best_epoch, best_score=best, history=history,
        selected_validation=history[best_epoch - 1]["validation"],
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)


def run_student(a, splits, contract, teacher_state, expected_initial):
    """Train a fresh C/D student; caller supplies verified source provenance."""
    if (contract.get("method") != METHOD or contract.get("group") != a.group
            or contract.get("epochs") != a.epochs or a.stage != "student" or a.group not in {"C", "D"}
            or a.cod_train_mode != "split" or not a.skip_test or a.prepare_only or a.eval_only):
        raise ValueError("Screen supports only matched C/D student contracts with tests disabled")
    if not expected_initial or not splits.get("cod_train") or not splits.get("cod_val"):
        raise ValueError("Screen requires reference initialization and nonempty COD train/validation")
    a.out = Path(a.out).expanduser().resolve()
    report_path = a.out / "comparison.json"
    previous = _read(report_path) if report_path.exists() else None
    if previous:
        if not a.resume or previous.get("protocol") != contract or previous.get("contract_sha256") != runner._digest(contract):
            raise ValueError("Existing screen requires --resume with an identical contract")
        if previous.get("status") == "complete":
            report = verify_completed(a.out, contract)
            if report["initial_state_sha256"] != expected_initial:
                raise ValueError("Completed screen initialization differs from the original baselines")
            return report
    elif a.out.exists() and any(path.name not in {"launch.json", "run.log"} for path in a.out.iterdir()):
        raise FileExistsError(f"Refusing unrecognized screen output: {a.out}")
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    from models.triprd_correction import CorrectionStudent, CorrectionTeacher
    from evaluation.validation import CODValidationMetrics  # fail before training if unavailable
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"method": METHOD, "stage": "student", "group": a.group, "status": "running",
        "protocol": contract, "contract_sha256": runner._digest(contract), "config": runner._json(a),
        "formal": False, "development_screen": True, "diagnostic": False,
        "selection_metric": "macro_S_alpha", "sample_counts": {key: len(value) for key, value in splits.items()},
        "definition": "Source joint supervision plus " + ("ordinary full relation KD" if a.group == "C" else "5 * pooled positive-edge correction MSE"),
        "initialization": "Fresh generic-pretrained student, no diagnostic checkpoint warm start",
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "test": {}, "test_status": "skipped",
        "started_at": previous.get("started_at") if previous else datetime.now(timezone.utc).isoformat()}
    runner._save_json(report_path, report)
    try:
        if not teacher_state.get("constant_correction") or not teacher_state.get("positive_constant_correction"):
            raise ValueError("Matched teacher lacks TRAIN-fitted probe constants")
        constants = {"all": teacher_state["constant_correction"], "positive": teacher_state["positive_constant_correction"]}
        teacher = CorrectionTeacher(**runner._model_options(a, pretrained=False)).to(device)
        teacher.load_state_dict(teacher_state["model"], strict=True)
        teacher.requires_grad_(False).eval()
        frozen_teacher = state_digest(teacher)
        runner.shared.seed_everything(a.train_seed)
        model = CorrectionStudent(**runner._model_options(a)).to(device)
        if state_digest(model) != expected_initial:
            raise ValueError("Fresh screen student initialization differs from the original A/B baselines")
        frozen_encoder = state_digest(model.encoder)
        report.update(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            generic_weight_loading=model.encoder.weight_load_metadata)
        loaders = runner._loaders(a, splits, device)
        fit(model, teacher, loaders, splits, a, device, contract, report)
        report["source_recoverability"] = runner._student_probe(model, teacher, loaders["vdt_val"], constants, device, a)
        if a.intervention_eval:
            report["intervention_validation"] = {"selection_used": False, "metrics": {
                mode: runner._validate_cod(model, splits["cod_val"], device, a, mode)
                for mode in ("none", "no_correction", "shuffle_correction")}}
        if state_digest(model.encoder) != frozen_encoder or state_digest(teacher) != frozen_teacher:
            raise RuntimeError("Frozen encoder or source teacher changed during screen training")
        report.update(status="complete", frozen_generic_encoder_verified=True, frozen_teacher_verified=True,
            checkpoint=str(a.out / "best.pt"), checkpoint_sha256=sha256_file(a.out / "best.pt"),
            last_checkpoint_sha256=sha256_file(a.out / "last.pt"))
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        runner._save_json(report_path, report)
    return verify_completed(a.out, contract)
