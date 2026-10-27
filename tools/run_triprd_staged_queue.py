#!/usr/bin/env python3
"""Independent v5 queues: source pretraining, COD screen, explicit final."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_triprd_task_queue as old

METHOD = "triprd_staged_v5"
SOURCE_GROUPS = ("S1", "S2", "S3")
COD_GROUPS = ("S0", *SOURCE_GROUPS)
NEW_FILES = ("tools/run_triprd_staged.py", "tools/evaluate_triprd_staged.py",
             "tools/run_triprd_staged_queue.py", "tools/run_triprd_staged.sh",
             "tools/summarize_triprd_staged.py")
SOURCE_FILES = tuple(dict.fromkeys((*NEW_FILES, *old.SOURCE_FILES)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--phase", choices=("source", "screen", "final"), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--gpus", type=old.gpu_list, default=["2", "3", "0", "1", "6"])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--vdt-root", default="datasets/VDT-2048")
    p.add_argument("--cod-root", default="datasets/COD_benchmark")
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--model-variant", choices=("tiny", "dinov3_vitb16"), default="dinov3_vitb16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detail-branch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--augmentation", choices=("flip", "crop_flip"), default="crop_flip")
    p.add_argument("--seed", "--train-seed", dest="train_seed", type=int, default=20261012)
    p.add_argument("--split-seed", type=int, default=20261012)
    for name, value in (("source-epochs", 20), ("epochs", 100), ("teacher-epochs", 100),
                        ("batch-size", 8), ("base-channels", 128), ("image-size", 384), ("val-metrics-every", 5)):
        p.add_argument("--" + name, type=old.positive_int, default=value)
    for name, value in (("num-workers", 4), ("max-samples", 0), ("train-backbone-blocks", 4), ("warmup-epochs", 5)):
        p.add_argument("--" + name, type=old.nonnegative_int, default=value)
    for name, value in (("lr", 1e-4), ("backbone-lr-scale", .1), ("weight-decay", 1e-4),
                        ("val-fraction", .2), ("kd-weight", 1.), ("feature-weight", 1.),
                        ("temperature", 2.), ("boundary-weight", .2), ("aux-weight", .5), ("grad-clip", 1.)):
        p.add_argument("--" + name, type=float, default=value)
    return p.parse_args(argv)


def command(python, config):
    result = [python, "-u", str(ROOT / "tools/run_triprd_staged.py")]
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if key == "test_after_training":
                if value:
                    result.append(flag)
            else:
                result.append(flag if value else "--no-" + key.replace("_", "-"))
        elif value is not None:
            result.extend((flag, str(value)))
    return result


def build_plan(a, phase=None):
    phase = phase or a.phase
    root = a.output_root.expanduser().resolve()
    teachers = a.teacher_root.expanduser().resolve()
    manifest = a.split_manifest.expanduser().resolve()
    if root in (teachers, teachers.parent, manifest.parent, manifest.parent.parent):
        raise ValueError("Use a new v5 output root; do not reuse the existing v4 experiment directory")
    old._visible_gpu_check(a.gpus, a.device)
    if not manifest.is_file():
        raise ValueError("Supply the existing v4 split_manifest.json; v5 must reuse its exact members")
    members = old._read(manifest)
    if members.get("split_seed", members.get("seed")) != a.split_seed:
        raise ValueError("Split seed differs from the frozen manifest")
    for prefix in ("cod", "vdt"):
        tr, va = members.get(prefix + "_train"), members.get(prefix + "_val")
        if (not isinstance(tr, list) or not isinstance(va, list) or not tr or not va
                or any(not isinstance(key, str) for key in tr + va)
                or len(set(tr + va)) != len(tr + va)):
            raise ValueError(f"Invalid/disjointness failure for frozen {prefix} members")
    if phase == "final" and (a.model_variant == "tiny" or a.max_samples or not a.pretrained):
        raise ValueError("Final requires real pretrained full4040 training")
    managed = {"phase", "output_root", "teacher_root", "gpus", "python", "execute", "resume"}
    common = {k: str(v.expanduser().resolve()) if isinstance(v, Path) else v
              for k, v in vars(a).items() if k not in managed}
    jobs, deps = [], {}
    for group in SOURCE_GROUPS if phase == "source" else COD_GROUPS:
        config = {**common, "stage": "source" if phase == "source" else "cod", "group": group,
                  "out": str(root / phase / group), "cod_train_mode": "full" if phase == "final" else "split"}
        if phase == "source" and group in ("S2", "S3"):
            mode = "rgb" if group == "S2" else "multimodal"
            checkpoint = teachers / mode / "best.pt"
            config["teacher_checkpoint"] = str(checkpoint)
            deps[mode] = {"method": old.METHOD, "stage": "teacher", "teacher_mode": mode,
                          "checkpoint": str(checkpoint), "sha256": old._sha256(checkpoint) if checkpoint.is_file() else None}
        if phase != "source" and group != "S0":
            checkpoint = root / "source" / group / "best.pt"
            config["source_checkpoint"] = str(checkpoint)
            deps[group] = {"method": METHOD, "stage": "source", "group": group,
                           "checkpoint": str(checkpoint), "sha256": old._sha256(checkpoint) if checkpoint.is_file() else None}
        if phase == "final":
            config["test_after_training"] = True
        jobs.append({"name": group, "output_dir": config["out"], "expected_config": config, "command": command(a.python, config)})
    plan = {"method": METHOD, "phase": phase, "output_root": str(root), "gpus": a.gpus, "device": a.device,
            "split_manifest": str(manifest), "manifest_sha256": old._sha256(manifest),
            "source_sha256": {name: old._sha256(ROOT / name) for name in SOURCE_FILES}, "dependencies": deps, "jobs": jobs}
    plan["plan_sha256"] = old._digest(plan)
    return plan


def verify_dependencies(plan):
    for name, dep in plan["dependencies"].items():
        checkpoint = Path(dep["checkpoint"])
        if not checkpoint.is_file() or not (checkpoint.parent / "comparison.json").is_file():
            raise ValueError(f"Missing completed dependency {name}: {checkpoint}")
        report = old._read(checkpoint.parent / "comparison.json")
        if (report.get("status") != "complete" or report.get("method") != dep["method"]
                or report.get("stage") != dep["stage"] or dep["sha256"] is None
                or dep["sha256"] != old._sha256(checkpoint) or dep["sha256"] != report.get("checkpoint_sha256")):
            raise ValueError(f"Dependency provenance differs: {name}")
        for key in ("group", "teacher_mode"):
            if key in dep and report.get(key) != dep[key]:
                raise ValueError(f"Dependency {name} has a different {key}")


def verify_job(job, final=False):
    directory, cfg = Path(job["output_dir"]), job["expected_config"]
    report = old._read(directory / "comparison.json")
    if report.get("method") != METHOD or report.get("status") != "complete":
        raise ValueError(f"{job['name']}: expected completed staged-v5 report")
    if any(report.get("config", {}).get(k) != v for k, v in cfg.items()):
        raise ValueError(f"{job['name']}: saved training configuration differs")
    epochs = cfg["source_epochs"] if cfg["stage"] == "source" else cfg["epochs"]
    if report.get("completed_epochs") != epochs:
        raise ValueError(f"{job['name']}: incomplete epoch budget")
    if report.get("contract_sha256") != old._digest(report.get("protocol")):
        raise ValueError(f"{job['name']}: invalid training contract")
    path = directory / "best.pt"
    if not path.is_file() or old._sha256(path) != report.get("checkpoint_sha256"):
        raise ValueError(f"{job['name']}: selected checkpoint hash differs")
    if final:
        from tools.evaluate_triprd_staged import validate_run_report
        validate_run_report(directory / "comparison.json")
        if report.get("test_status") != "complete" or set(report.get("test", {})) != {"COD10K", "CAMO", "CHAMELEON", "NC4K"}:
            raise ValueError(f"{job['name']}: full testing incomplete")
    elif report.get("test") or report.get("test_status") != "skipped":
        raise ValueError("Source and screen phases cannot use official tests")
    return {"checkpoint_sha256": report["checkpoint_sha256"], "selected_epoch": report["best_epoch"]}


@contextmanager
def pipeline_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "staged_pipeline.lock"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
    except FileExistsError as exc:
        raise RuntimeError(f"A staged pipeline lock already exists: {path}; inspect its owner") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def execute_plan(plan, *, resume=False, poll_seconds=.1):
    if plan["plan_sha256"] != old._digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Plan digest mismatch")
    old._visible_gpu_check(plan["gpus"], plan["device"])
    if plan["source_sha256"] != {name: old._sha256(ROOT / name) for name in SOURCE_FILES}:
        raise ValueError("Source files changed after the plan was prepared")
    if plan["manifest_sha256"] != old._sha256(Path(plan["split_manifest"])):
        raise ValueError("Frozen split manifest changed")
    verify_dependencies(plan)
    metadata = Path(plan["output_root"]) / "_queue" / plan["phase"]
    saved = metadata / "plan.json"
    previous = old._read(metadata / "status.json") if (metadata / "status.json").exists() else {}
    if saved.exists():
        if old._read(saved) != plan:
            raise ValueError("Existing phase plan differs; use a new output root")
        if previous.get("status") == "complete":
            for job in plan["jobs"]:
                verify_job(job, final=plan["phase"] == "final")
            print(f"[{plan['phase']}] verified completed phase; reusing its checkpoints", flush=True)
            return 0
        if not resume:
            raise ValueError("Interrupted phase exists; rerun with --resume and the same configuration")
    metadata.mkdir(parents=True, exist_ok=True)
    if not saved.exists():
        old._write(saved, plan)
    state = {"method": METHOD, "phase": plan["phase"], "status": "running", "plan_sha256": plan["plan_sha256"],
             "manifest_sha256": plan["manifest_sha256"], "jobs": {j["name"]: {"status": "pending"} for j in plan["jobs"]}}
    active, pending, free = {}, list(plan["jobs"]), list(plan["gpus"])
    try:
        old._write(metadata / "status.json", state)
        while pending or active:
            for name, info in list(active.items()):
                code = info["process"].poll()
                if code is None:
                    continue
                info["log"].close()
                del active[name]
                free.append(info["gpu"])
                entry = state["jobs"][name]
                entry["returncode"] = code
                if code:
                    entry["status"] = "failed"
                    raise RuntimeError(f"{name} exited {code}; see {info['job']['output_dir']}/run.log")
                entry.update(verify_job(info["job"], final=plan["phase"] == "final"), status="complete")
            while pending and free:
                job = pending.pop(0)
                directory = Path(job["output_dir"])
                existing = old._read(directory / "comparison.json") if (directory / "comparison.json").exists() else {}
                if (existing.get("status") == "complete"
                        and not (plan["phase"] == "final" and existing.get("test_status") != "complete")):
                    state["jobs"][job["name"]].update(verify_job(job, final=plan["phase"] == "final"), status="complete", reused=True)
                    continue
                gpu = free.pop(0)
                directory.mkdir(parents=True, exist_ok=True)
                cmd = job["command"] + (["--resume"] if resume else [])
                old._write(directory / "launch.json", {**job, "gpu": gpu, "executed_command": cmd})
                log = (directory / "run.log").open("a", encoding="utf-8")
                try:
                    process = old._spawn(cmd, log, gpu, plan["device"])
                except BaseException:
                    log.close()
                    raise
                active[job["name"]] = {"process": process, "log": log, "gpu": gpu, "job": job}
                state["jobs"][job["name"]] = {"status": "running", "pid": process.pid, "gpu": gpu}
                print(f"[{plan['phase']}] started {job['name']} on GPU {gpu}: {directory / 'run.log'}", flush=True)
            old._write(metadata / "status.json", state)
            if active:
                time.sleep(poll_seconds)
        state["status"] = "complete"
        return 0
    except KeyboardInterrupt:
        state.update(status="interrupted", error="Interrupted; stopping owned children")
        return 130
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        print(f"[{plan['phase']}] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for name, info in active.items():
            try:
                old._stop(info["process"])
            finally:
                info["log"].close()
                state["jobs"][name].update(status="terminated", returncode=info["process"].poll())
        for entry in state["jobs"].values():
            if entry["status"] == "pending":
                entry["status"] = "not_started"
        old._write(metadata / "status.json", state)


def main(argv=None):
    a = parse_args(argv)
    phases = ("source", "screen") if a.phase == "screen" else (a.phase,)
    if not a.execute:
        print(json.dumps({"mode": "preview", "phases": [build_plan(a, phase) for phase in phases]}, ensure_ascii=False, indent=2))
        return 0
    previous_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with pipeline_lock(a.output_root.expanduser().resolve()):
            for phase in phases:
                code = execute_plan(build_plan(a, phase), resume=a.resume)
                if code:
                    return code
            if a.phase == "screen":
                from tools.summarize_triprd_staged import main as summarize
                return summarize(["--output-root", str(a.output_root.expanduser().resolve())])
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
