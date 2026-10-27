#!/usr/bin/env python3
"""Run one explicit v4 experiment phase; omitted --execute only prints a plan."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHOD = "triprd_task_v4"
PHASES = ("baseline", "teachers", "screen", "final")
SOURCE_FILES = (
    "tools/run_triprd_task_queue.py", "tools/run_triprd_task.sh", "tools/run_triprd_task.py",
    "models/triprd_task.py", "models/triprd_task_loss.py", "evaluation/task_protocol.py",
    "tools/evaluate_triprd_task.py", "tools/run_triprd_correction.py", "tools/run_triprd_ab.py",
    "tools/triprd_structure_training.py", "tools/triprd_training.py", "models/triprd_correction.py",
    "models/triprd_correction_loss.py", "models/dinov3_weights.py", "models/triprd.py",
    "datasets/cod_dataset.py", "datasets/vdt2048.py", "evaluation/validation.py",
    "evaluation/cod_metrics.py", "evaluation/camoformer_metrics.py", "tools/evaluate_cod.py",
)
BASELINES = (("R0_frozen_no_detail", 0, False), ("R1_blocks4_no_detail", 4, False),
             ("R2_frozen_detail", 0, True), ("R3_blocks4_detail", 4, True))


def gpu_list(value):
    values = [x.strip() for x in value.split(",")]
    if not values or any(not re.fullmatch(r"\d+", x) for x in values):
        raise argparse.ArgumentTypeError("Use unique nonnegative physical GPU IDs, e.g. 0,1")
    values = [str(int(x)) for x in values]
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Use unique nonnegative physical GPU IDs, e.g. 0,1")
    return values


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative")
    return result


def parse_args(argv=None):
    # Explicit options deliberately forbid forwarding managed runner flags or
    # their argparse abbreviations (stage/group/output/teacher/full/test).
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--phase", choices=PHASES, default="baseline")
    p.add_argument("--output-root", type=Path, default=Path("results/task_v4"))
    p.add_argument("--gpus", type=gpu_list, default="0,1")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume only this exact phase plan")
    p.add_argument("--vdt-root", type=Path, default=Path("datasets/VDT-2048"))
    p.add_argument("--cod-root", type=Path, default=Path("datasets/COD_benchmark"))
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", "--train-seed", dest="train_seed", type=int, default=20261012)
    p.add_argument("--split-seed", type=int, default=20261012)
    for name, default in (("epochs", 100), ("teacher-epochs", 100), ("batch-size", 8),
                          ("base-channels", 128), ("image-size", 384), ("val-metrics-every", 5)):
        p.add_argument("--" + name, type=positive_int, default=default)
    for name, default in (("num-workers", 4), ("max-samples", 0), ("warmup-epochs", 5),
                          ("train-backbone-blocks", 4)):
        p.add_argument("--" + name, type=nonnegative_int, default=default)
    p.add_argument("--detail-branch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--augmentation", choices=("flip", "crop_flip"), default="crop_flip")
    for name, default in (("val-fraction", .2), ("lr", 1e-4), ("backbone-lr-scale", .1),
                          ("weight-decay", 1e-4), ("source-weight", .5), ("source-final-ratio", .1),
                          ("kd-weight", 1.), ("feature-weight", 1.), ("temperature", 2.),
                          ("boundary-weight", .2), ("aux-weight", .5), ("grad-clip", 1.)):
        p.add_argument("--" + name, type=float, default=default)
    return p.parse_args(argv)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers/virus scanners can briefly deny replacement of a closed
    # destination. Retain the complete temporary file and retry boundedly.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.025 * (attempt + 1))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _command(python, config):
    command = [python, "-u", str(ROOT / "tools/run_triprd_task.py")]
    for name, value in config.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + name.replace("_", "-"))
        elif value is not None:
            command.extend((flag, str(value)))
    return command


def _visible_gpu_check(gpus, device):
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device == "cuda" and existing is not None:
        allowed = {str(int(x.strip())) if x.strip().isdigit() else x.strip()
                   for x in existing.split(",") if x.strip()}
        if not set(gpus).issubset(allowed):
            raise ValueError("Requested physical GPUs must be inside the existing CUDA_VISIBLE_DEVICES restriction")


def build_plan(a):
    _visible_gpu_check(a.gpus, a.device)
    if any(not 0 <= value < 2 ** 32 for value in (a.train_seed, a.split_seed)):
        raise ValueError("Seeds must be in [0, 2**32)")
    if a.model_variant == "tiny" and a.pretrained:
        raise ValueError("tiny diagnostics require --no-pretrained")
    if a.model_variant != "tiny" and (not a.pretrained or not a.dino_repo or not a.dino_weights):
        raise ValueError("DINOv3 requires --pretrained, --dino-repo and --dino-weights")
    if a.phase == "final" and (a.model_variant == "tiny" or a.max_samples):
        raise ValueError("Final benchmark requires a real model and all 4040 COD training images")
    if a.base_channels < 4 or a.image_size < 32 or a.image_size % 16:
        raise ValueError("base-channels >= 4; image-size >= 32 and divisible by 16 required")
    for name in ("lr", "backbone_lr_scale", "temperature", "grad_clip"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "source_weight", "kd_weight", "feature_weight", "boundary_weight", "aux_weight"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < a.val_fraction < 1 or not 0 <= a.source_final_ratio <= 1:
        raise ValueError("Invalid validation fraction/source final ratio")
    root = a.output_root.expanduser().resolve()
    manifest = root / "_data/split_manifest.json"
    managed = {"phase", "output_root", "gpus", "python", "execute", "resume"}
    common = {key: (str(value.expanduser().resolve()) if isinstance(value, Path) else value)
              for key, value in vars(a).items() if key not in managed}
    # Same split for every phase. Architecture is intentionally absent from the
    # data contract so baseline results can select the later student backbone.
    data_contract = {key: common[key] for key in ("vdt_root", "cod_root", "split_seed", "val_fraction", "max_samples")}
    prepare_config = {**common, "device": "cpu", "stage": "student", "group": "T0",
                      "cod_train_mode": "split", "out": str(manifest.parent)}
    prepare = _command(a.python, prepare_config) + ["--prepare-only"]
    jobs = []
    if a.phase == "baseline":
        recipes = [(name, {"stage": "student", "group": "T0", "train_backbone_blocks": blocks,
                           "detail_branch": detail}) for name, blocks, detail in BASELINES]
    elif a.phase == "teachers":
        recipes = [(mode, {"stage": "teacher", "group": "T0", "teacher_mode": mode})
                   for mode in ("rgb", "multimodal")]
    else:
        recipes = [(f"T{i}", {"stage": "student", "group": f"T{i}"}) for i in range(5)]
    dependencies = {}
    for name, recipe in recipes:
        config = {**common, **recipe, "cod_train_mode": "full" if a.phase == "final" else "split",
                  "out": str(root / a.phase / name), "split_manifest": str(manifest)}
        if recipe["stage"] == "student" and recipe["group"] in {"T2", "T3", "T4"}:
            mode = "rgb" if recipe["group"] == "T2" else "multimodal"
            checkpoint = root / "teachers" / mode / "best.pt"
            config["teacher_checkpoint"] = str(checkpoint)
            dependencies[mode] = {"checkpoint": str(checkpoint),
                                  "sha256": _sha256(checkpoint) if checkpoint.is_file() else None}
        command = _command(a.python, config)
        if a.phase == "final":
            command.append("--test-after-training")
            config["test_after_training"] = True
        jobs.append({"name": name, "output_dir": config["out"], "command": command, "expected_config": config})
    body = {"method": METHOD, "phase": a.phase, "output_root": str(root), "gpus": a.gpus,
            "device": a.device, "data_contract": data_contract, "split_manifest": str(manifest),
            "prepare_command": prepare, "jobs": jobs, "dependencies": dependencies,
            "source_sha256": {name: _sha256(ROOT / name) for name in SOURCE_FILES}}
    return {**body, "plan_sha256": _digest(body)}


def _verify_dependencies(plan):
    for mode, info in plan["dependencies"].items():
        checkpoint = Path(info["checkpoint"])
        report_path = checkpoint.parent / "comparison.json"
        if not checkpoint.is_file() or not report_path.is_file():
            raise ValueError(f"Run the teachers phase first: missing completed {mode} teacher")
        report = _read(report_path)
        digest = _sha256(checkpoint)
        if (report.get("method") != METHOD or report.get("status") != "complete" or
                report.get("stage") != "teacher" or report.get("teacher_mode") != mode or
                digest != report.get("checkpoint_sha256") or digest != info["sha256"]):
            raise ValueError(f"Invalid or changed {mode} teacher checkpoint/report")


def _verify_job(job, final=False):
    directory = Path(job["output_dir"])
    report = _read(directory / "comparison.json")
    config = job["expected_config"]
    if report.get("method") != METHOD or report.get("status") != "complete":
        raise ValueError(f"{job['name']}: child did not produce a completed v4 report")
    for key in ("stage", "group"):
        if report.get(key) != config[key]:
            raise ValueError(f"{job['name']}: report {key} differs")
    mismatch = [key for key, value in config.items() if report.get("config", {}).get(key) != value]
    if mismatch:
        raise ValueError(f"{job['name']}: report configuration differs: {mismatch}")
    if (config["stage"] == "teacher" and report.get("teacher_mode") != config["teacher_mode"]):
        raise ValueError(f"{job['name']}: teacher mode differs")
    checkpoint = directory / "best.pt"
    if not checkpoint.is_file() or report.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError(f"{job['name']}: selected checkpoint hash differs")
    expected_epochs = config["teacher_epochs"] if config["stage"] == "teacher" else config["epochs"]
    if report.get("completed_epochs") != expected_epochs:
        raise ValueError(f"{job['name']}: training budget incomplete")
    if final:
        if (report.get("selection_metric") != "last_epoch" or report.get("best_epoch") != expected_epochs or
                report.get("test_status") != "complete" or set(report.get("test", {})) != {"NC4K", "COD10K", "CAMO", "CHAMELEON"}):
            raise ValueError(f"{job['name']}: final training/testing incomplete")
    elif report.get("test") or report.get("test_status") != "skipped":
        raise ValueError(f"{job['name']}: development jobs must not access official tests")
    return {"checkpoint_sha256": report["checkpoint_sha256"], "selected_epoch": report.get("best_epoch")}


@contextmanager
def _queue_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "task_queue.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Queue lock exists: {path}; inspect its process before recovery") from exc
    with handle:
        json.dump({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _spawn(command, log, gpu, device):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu) if device == "cuda" else ""
    environment["PYTHONUNBUFFERED"] = "1"
    options = {"cwd": ROOT, "env": environment, "stdout": log, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(command, **options)


def _stop(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Includes spawned DataLoader workers; no visible helper window.
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            if process.poll() is None:
                process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


def _prepare(plan, metadata):
    manifest = Path(plan["split_manifest"])
    contract_path = manifest.parent / "data_contract.json"
    if manifest.is_file() and contract_path.is_file():
        saved = _read(contract_path)
        if saved.get("contract") != plan["data_contract"] or saved.get("manifest_sha256") != _sha256(manifest):
            raise ValueError("Existing shared manifest/data configuration differs")
        return _sha256(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    process = None
    with (metadata / "prepare.log").open("a", encoding="utf-8") as log:
        try:
            process = _spawn(plan["prepare_command"], log, "", "cpu")
            code = process.wait()
            if code or not manifest.is_file():
                raise RuntimeError(f"Dataset preparation failed (exit={code}); see {metadata / 'prepare.log'}")
        finally:
            if process is not None:
                _stop(process)
    value = _read(manifest)
    if any(not value.get(key) for key in ("vdt_train", "vdt_val", "cod_train", "cod_val")):
        raise ValueError("Prepared manifest lacks required train/validation partitions")
    digest = _sha256(manifest)
    _write(contract_path, {"contract": plan["data_contract"], "manifest_sha256": digest})
    return digest


def execute_plan(plan, *, resume=False, poll_seconds=.1):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != _digest(body):
        raise ValueError("Plan digest mismatch")
    _visible_gpu_check(plan["gpus"], plan["device"])
    _verify_dependencies(plan)
    root = Path(plan["output_root"])
    metadata = root / "_queue" / plan["phase"]
    saved_plan = metadata / "plan.json"
    if saved_plan.exists():
        if not resume or _read(saved_plan) != plan:
            raise ValueError("Existing phase requires --resume with an identical plan/configuration")
    elif resume:
        raise ValueError("Cannot resume a phase without its saved plan")
    active, state = {}, {"phase": plan["phase"], "status": "running", "plan_sha256": plan["plan_sha256"],
                          "jobs": {job["name"]: {"status": "pending"} for job in plan["jobs"]}}
    previous_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with _queue_lock(root):
            metadata.mkdir(parents=True, exist_ok=True)
            if not saved_plan.exists():
                _write(saved_plan, plan)
            try:
                state["manifest_sha256"] = _prepare(plan, metadata)
                _write(metadata / "status.json", state)
                pending, free = list(plan["jobs"]), list(plan["gpus"])
                while pending or active:
                    # Inspect every completed child before allocating another
                    # slot: a failed experiment must stop all further starts.
                    failure, changed = None, False
                    for name, info in list(active.items()):
                        code = info["process"].poll()
                        if code is None:
                            continue
                        changed = True
                        info["log"].close()
                        del active[name]
                        free.append(info["gpu"])
                        entry = state["jobs"][name]
                        entry["returncode"] = code
                        if code:
                            entry.update(status="failed", error=f"child exit code {code}")
                            failure = entry["error"]
                        else:
                            try:
                                entry.update(_verify_job(info["job"], final=plan["phase"] == "final"), status="complete")
                            except Exception as exc:
                                entry.update(status="failed", error=str(exc))
                                failure = str(exc)
                    if failure:
                        raise RuntimeError(failure)
                    while pending and free:
                        job, gpu = pending.pop(0), free.pop(0)
                        directory = Path(job["output_dir"])
                        directory.mkdir(parents=True, exist_ok=True)
                        command = job["command"] + (["--resume"] if resume else [])
                        _write(directory / "launch.json", {**job, "gpu": gpu, "executed_command": command})
                        log = (directory / "run.log").open("a" if resume else "w", encoding="utf-8")
                        try:
                            process = _spawn(command, log, gpu, plan["device"])
                        except BaseException:
                            log.close()
                            raise
                        active[job["name"]] = {"process": process, "log": log, "gpu": gpu, "job": job}
                        changed = True
                        state["jobs"][job["name"]] = {"status": "running", "pid": process.pid, "gpu": gpu}
                        print(f"[{plan['phase']}] started {job['name']} on GPU {gpu}: {directory / 'run.log'}", flush=True)
                    if changed:
                        _write(metadata / "status.json", state)
                    if active:
                        time.sleep(poll_seconds)
                state["status"] = "complete"
                return 0
            except KeyboardInterrupt:
                state.update(status="interrupted", error="Interrupted; active children terminated")
                return 130
            except Exception as exc:
                state.update(status="failed", error=str(exc))
                print(f"[{plan['phase']}] failed: {exc}", file=sys.stderr, flush=True)
                return 1
            finally:
                for name, info in active.items():
                    try:
                        _stop(info["process"])
                    except Exception as exc:
                        state.setdefault("cleanup_errors", []).append({"job": name, "error": str(exc)})
                    finally:
                        info["log"].close()
                        state["jobs"][name].update(status="terminated", returncode=info["process"].poll())
                for entry in state["jobs"].values():
                    if entry["status"] == "pending":
                        entry["status"] = "not_started"
                _write(metadata / "status.json", state)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **plan}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    return execute_plan(plan, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
