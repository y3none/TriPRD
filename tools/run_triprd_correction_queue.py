#!/usr/bin/env python3
"""Plan or execute the six matched correction_v3 experiments and one shared teacher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.run_triprd_optimization import gpu_list, positive_int

GROUPS = (("A", "A_cod_only"), ("B", "B_rgb_joint"), ("C", "C_full_kd"),
          ("D", "D_correction"), ("E", "E_selected_correction"), ("F", "F_selected_full_kd"))
SOURCE_FILES = ("tools/run_triprd_correction_queue.py", "tools/run_triprd_correction.py",
                "models/triprd_correction.py", "models/triprd_correction_loss.py",
                "models/dinov3_weights.py",
                "tools/run_triprd_ab.py", "tools/triprd_training.py", "tools/evaluate_cod.py",
                "datasets/cod_dataset.py", "datasets/vdt2048.py")


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus", type=gpu_list, default="0,1,2,3,6,7")
    parser.add_argument("--output-root", type=Path, default=Path("results/optimization"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", "--train-seed", dest="seed", type=int, default=20261012)
    parser.add_argument("--split-seed", type=int, default=20261012)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--vdt-root", type=Path, default=Path("datasets/VDT-2048"))
    parser.add_argument("--cod-root", type=Path, default=Path("datasets/COD_benchmark"))
    parser.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    parser.add_argument("--dino-repo", type=Path)
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--teacher-epochs", type=positive_int, default=100)
    parser.add_argument("--image-size", type=positive_int, default=384)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--base-channels", type=positive_int, default=128)
    parser.add_argument("--num-workers", type=nonnegative_int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cod-train-mode", choices=("split", "full"), default="split")
    parser.add_argument("--skip-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=nonnegative_int, default=0)
    parser.add_argument("--val-metrics-every", type=positive_int, default=5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--source-weight", type=float, default=0.5)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=nonnegative_int, default=5)
    parser.add_argument("--groups", default="A,B,C,D,E,F")
    parser.add_argument("--resume", action="store_true", help="Revalidate and resume this exact saved plan")
    parser.add_argument("--allow-nonpositive-probe", action="store_true",
                        help="Explicitly allow finite nonpositive teacher gain/coverage; recorded in the plan")
    parser.add_argument("--execute", action="store_true", help="Start training; omission is a read-only dry run")
    return parser.parse_args(argv)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def build_plan(args: argparse.Namespace) -> dict:
    if any(not 0 <= value < 2 ** 32 for value in (args.seed, args.split_seed)):
        raise ValueError("Training and split seeds must be in [0, 2**32)")
    groups = [value.strip().upper() for value in args.groups.split(",")]
    if not groups or len(groups) != len(set(groups)) or set(groups) - dict(GROUPS).keys():
        raise ValueError("--groups must contain unique entries from A,B,C,D,E,F")
    for name in ("lr", "source_weight", "kd_weight", "relation_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if args.image_size < 32 or args.image_size % 32:
        raise ValueError("--image-size must be at least 32 and divisible by 32")
    if args.base_channels < 4:
        raise ValueError("--base-channels must be at least 4")
    if args.cod_train_mode == "full" and args.max_samples:
        raise ValueError("Full COD training requires --max-samples 0")
    if args.model_variant == "tiny" and args.pretrained:
        raise ValueError("Tiny is a pipeline test; use --no-pretrained")
    if args.model_variant == "dinov3_vitb16" and (not args.dino_repo or not args.dino_weights):
        raise ValueError("DINOv3 requires local --dino-repo and --dino-weights")
    if args.model_variant == "dinov3_vitb16" and not args.pretrained:
        raise ValueError("The formal DINOv3 protocol requires --pretrained")
    run_dir = args.output_root.expanduser().resolve() / "correction_v3" / f"seed-{args.seed}"
    manifest = run_dir / "_data" / "split_manifest.json"
    external = args.split_manifest.expanduser().resolve() if args.split_manifest else None
    config = {name: getattr(args, name) for name in (
        "split_seed", "model_variant", "pretrained", "epochs", "teacher_epochs", "image_size", "batch_size",
        "base_channels", "num_workers", "device", "cod_train_mode", "skip_test", "max_samples",
        "val_metrics_every", "amp", "lr", "weight_decay", "source_weight", "kd_weight", "relation_weight")}
    config["train_seed"] = args.seed
    config["warmup_epochs"] = min(args.warmup_epochs, args.epochs - 1, args.teacher_epochs - 1)
    config.update({name: str(getattr(args, name).expanduser().resolve()) for name in ("vdt_root", "cod_root")})
    for name in ("dino_repo", "dino_weights"):
        if getattr(args, name) is not None:
            path = getattr(args, name).expanduser()
            config[name] = str(path.absolute() if name == "dino_weights" else path.resolve())
    common = [args.python, "-u", str(ROOT / "tools" / "run_triprd_correction.py")]
    for name, value in config.items():
        flag = "--" + name.replace("_", "-")
        common += [flag if value else "--no-" + name.replace("_", "-")] if isinstance(value, bool) else [flag, str(value)]
    needs_teacher = bool(set(groups) & set("CDEF"))
    jobs = []
    teacher_out = run_dir / "teacher"
    for stage, group, name, gpu in (
        [("teacher", None, "teacher", args.gpus[-1])] if needs_teacher else []
    ) + [("student", group, name, args.gpus[index % len(args.gpus)])
         for index, (group, name) in enumerate(GROUPS) if group in groups]:
        output = run_dir / name
        expected = {**config, "stage": stage, "out": str(output), "split_manifest": str(manifest)}
        command = common + ["--stage", stage, "--out", str(output), "--split-manifest", str(manifest)]
        if group:
            command += ["--group", group]
            expected["group"] = group
        dependencies = ["teacher"] if group in tuple("CDEF") else []
        if dependencies:
            command += ["--teacher-checkpoint", str(teacher_out / "best.pt")]
            expected["teacher_checkpoint"] = str(teacher_out / "best.pt")
        jobs.append({"name": name, "stage": stage, "group": group, "gpu": gpu,
                     "output_dir": str(output), "dependencies": dependencies,
                     "command": command, "expected_config": expected})
    prepare = common + ["--stage", "teacher", "--prepare-only", "--out", str(manifest.parent)]
    if external:
        prepare += ["--split-manifest", str(external)]
    protocol = {"method": "correction_v3", "run_dir": str(run_dir), "gpus": args.gpus,
                "cod_train_mode": args.cod_train_mode,
                "selection_rule": "last_epoch" if args.cod_train_mode == "full" else "macro_S_alpha",
                "prepare_command": prepare, "split_manifest": str(manifest),
                "input_manifest": str(external) if external else None,
                "input_manifest_sha256": _hash_file(external) if external and external.is_file() else None,
                "source_sha256": {name: _hash_file(ROOT / name) for name in SOURCE_FILES if (ROOT / name).is_file()},
                "jobs": jobs}
    plan = {**protocol, "plan_sha256": _digest(protocol), "resume": args.resume,
            "allow_nonpositive_probe": args.allow_nonpositive_probe}
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    run_dir = Path(plan["run_dir"]).resolve()
    if run_dir.exists() and not plan["resume"]:
        raise ValueError(f"Output already exists; use --resume for this exact plan or a new --output-root: {run_dir}")
    if plan["resume"] and run_dir.exists():
        saved = _read_json(run_dir / "launch.json")
        if saved.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Existing plan differs in protocol, command, input manifest, GPU mapping or source hashes")
    names = set()
    for job in plan["jobs"]:
        if job["name"] in names or Path(job["output_dir"]).resolve().parent != run_dir:
            raise ValueError("Job names must be unique and outputs must be direct run-directory children")
        if job["gpu"] not in plan["gpus"]:
            raise ValueError("Unassigned GPU")
        names.add(job["name"])
    for job in plan["jobs"]:
        if set(job["dependencies"]) - names:
            raise ValueError("Missing teacher dependency")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def validate_result(job: dict, *, allow_nonpositive_probe: bool = False) -> list[str]:
    """Exit zero alone cannot release a teacher dependency or mark a student complete."""
    output = Path(job["output_dir"])
    report = _read_json(output / "comparison.json")
    if report.get("status") != "complete":
        raise ValueError("comparison.json must declare status=complete")
    contract = report.get("contract_sha256", "")
    if not isinstance(contract, str) or not re.fullmatch(r"[0-9a-f]{64}", contract):
        raise ValueError("Missing valid contract_sha256")
    actual = report.get("config", {})
    mismatches = [name for name, value in job["expected_config"].items() if actual.get(name) != value]
    if mismatches:
        raise ValueError("Completed report config differs: " + ", ".join(mismatches))
    if not (output / "best.pt").is_file() or (output / "best.pt").stat().st_size == 0:
        raise ValueError("Completed job has no nonempty best.pt")
    warnings = []
    if job["stage"] == "teacher":
        probe = report.get("teacher_probe", {})
        standalone = _read_json(output / "teacher_probe.json")
        if standalone.get("status") != "complete" or standalone.get("contract_sha256") != contract:
            raise ValueError("Teacher probe is incomplete or belongs to another contract")
        for name in ("mean_relation_gain", "positive_coverage"):
            value = probe.get(name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Teacher probe {name} must be finite")
            if standalone.get(name) != value:
                raise ValueError(f"Teacher probe disagreement for {name}")
        if not 0 <= probe["positive_coverage"] <= 1:
            raise ValueError("Teacher positive_coverage must be in [0,1]")
        if probe["mean_relation_gain"] <= 0 or probe["positive_coverage"] <= 0:
            message = "Teacher has nonpositive relation gain or zero positive coverage"
            if not allow_nonpositive_probe:
                raise ValueError(message + "; C–F require explicit --allow-nonpositive-probe")
            warnings.append(message + "; explicitly allowed in this plan")
    return warnings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_manifest(path: Path, plan: dict) -> str:
    data = _read_json(path)
    for prefix in ("vdt", "cod"):
        train, val = data.get(prefix + "_train"), data.get(prefix + "_val")
        if not isinstance(train, list) or not train or not isinstance(val, list):
            raise ValueError(f"Prepared manifest must contain {prefix} train/val member lists")
        if (prefix == "vdt" or plan["cod_train_mode"] == "split") and not val:
            raise ValueError(f"Prepared manifest has empty {prefix} validation")
        if any(not isinstance(key, str) for key in train + val):
            raise ValueError("Manifest member keys must be strings")
        if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
            raise ValueError(f"Prepared manifest has duplicate or overlapping {prefix} members")
    return _hash_file(path)


def execute_plan(plan: dict, *, popen_factory=None, sleep=None) -> int:
    """Foreground dependency scheduler; failures do not cancel independent experiments."""
    validate_plan(plan)
    popen_factory = subprocess.Popen if popen_factory is None else popen_factory
    sleep = time.sleep if sleep is None else sleep
    required = [Path(plan["prepare_command"][2])]
    if plan["input_manifest"]:
        required.append(Path(plan["input_manifest"]))
    first = plan["jobs"][0]["expected_config"]
    if first["model_variant"] == "dinov3_vitb16":
        if not Path(first["dino_repo"]).is_dir():
            raise ValueError("Local DINO repository directory is missing")
        required += [Path(first["dino_repo"]) / "hubconf.py", Path(first["dino_weights"])]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Required files missing: " + ", ".join(missing))
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "queue_lock.json"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Queue lock exists; verify its PID has exited before removing {lock_path}") from exc
    with lock:
        json.dump({"pid": os.getpid(), "started_at": _now()}, lock)
    states = {job["name"]: {"status": "queued", "gpu": job["gpu"], "dependencies": job["dependencies"]}
              for job in plan["jobs"]}
    running = {}
    prepare_process = None
    prepare_log = None
    failed = False
    summary = {"plan_sha256": plan["plan_sha256"], "started_at": _now(), "launcher_pid": os.getpid(),
               "allow_nonpositive_probe": plan["allow_nonpositive_probe"], "resume": plan["resume"], "jobs": states}

    def save() -> None:
        _write_json(run_dir / "exit_status.json", summary)

    def finish(job: dict, returncode: int, error: str | None = None) -> None:
        nonlocal failed
        state = states[job["name"]]
        state.update(process_returncode=returncode, finished_at=_now())
        if returncode == 0 and error is None:
            try:
                state["warnings"] = validate_result(job, allow_nonpositive_probe=plan["allow_nonpositive_probe"])
            except (ValueError, OSError) as exc:
                error = str(exc)
        state.update(status="complete" if returncode == 0 and error is None else "failed",
                     returncode=returncode if returncode else (3 if error else 0))
        if error:
            state["error"] = error
        failed |= state["status"] != "complete"
        _write_json(Path(job["output_dir"]) / "exit_status.json", state)
        save()
        print(f"[{job['name']}] {state['status']}" + (f": {error}" if error else ""), flush=True)

    try:
        if not (run_dir / "launch.json").exists():
            _write_json(run_dir / "launch.json", plan)
        prepare_dir = run_dir / "_data"
        prepare_dir.mkdir(exist_ok=True)
        prepare_log = (prepare_dir / "run.log").open("a", encoding="utf-8")
        command = plan["prepare_command"] + (["--resume"] if plan["resume"] else [])
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        prepare_process = popen_factory(command, cwd=str(ROOT), env=environment, stdout=prepare_log, stderr=subprocess.STDOUT)
        summary["prepare"] = {"status": "running", "pid": prepare_process.pid, "command": command}
        save()
        while prepare_process.poll() is None:
            sleep(0.25)
        result = prepare_process.poll()
        prepare_log.close()
        if result != 0:
            raise ValueError(f"Manifest preparation failed with exit status {result}; see {prepare_dir / 'run.log'}")
        summary["manifest_sha256"] = _check_manifest(Path(plan["split_manifest"]), plan)
        summary["prepare"]["status"] = "complete"
        save()
        while running or any(state["status"] == "queued" for state in states.values()):
            for gpu, (job, process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                finish(job, code)
                del running[gpu]
            for job in plan["jobs"]:
                state = states[job["name"]]
                if state["status"] != "queued":
                    continue
                if any(states[dep]["status"] in ("failed", "blocked", "interrupted") for dep in job["dependencies"]):
                    state.update(status="blocked", reason="Teacher failed or its completion/probe validation failed")
                    failed = True
                    save()
                    continue
                if job["gpu"] in running or any(states[dep]["status"] != "complete" for dep in job["dependencies"]):
                    continue
                directory = Path(job["output_dir"])
                directory.mkdir(exist_ok=plan["resume"])
                if not (directory / "launch.json").exists():
                    _write_json(directory / "launch.json", job)
                command = job["command"] + (["--resume"] if plan["resume"] else [])
                log = (directory / "run.log").open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=job["gpu"], PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                state.update(status="running", started_at=_now(), command=command)
                try:
                    process = popen_factory(command, cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
                except OSError as exc:
                    log.close()
                    finish(job, 127, str(exc))
                    continue
                state["pid"] = process.pid
                running[job["gpu"]] = (job, process, log)
                save()
                print(f"[{job['name']}] GPU {job['gpu']}, pid {process.pid}, log {directory / 'run.log'}", flush=True)
            if running:
                sleep(0.25)
        summary.update(status="failed" if failed else "complete", finished_at=_now())
        save()
        return 1 if failed else 0
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), finished_at=_now())
        for state in states.values():
            if state["status"] in ("running", "queued"):
                state.update(status="interrupted" if state["status"] == "running" else "blocked", reason=str(exc))
        save()
        raise
    finally:
        for process in ([prepare_process] if prepare_process is not None else []) + [item[1] for item in running.values()]:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for _, _, log in running.values():
            log.close()
        if prepare_log is not None:
            prepare_log.close()
        lock_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        plan = build_plan(args)
        if args.execute:
            return execute_plan(plan)
        visible = dict(plan)
        visible["jobs"] = [{**job, "display_command": f"CUDA_VISIBLE_DEVICES={job['gpu']} " +
                            (subprocess.list2cmdline(job["command"]) if os.name == "nt" else shlex.join(job["command"]))}
                           for job in plan["jobs"]]
        print(json.dumps({"mode": "dry-run", **visible}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
