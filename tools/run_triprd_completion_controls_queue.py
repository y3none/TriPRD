#!/usr/bin/env python3
"""Run four completion controls concurrently, one physical GPU per group.

Each group is a sequence of directly managed child processes. Omit --execute
for a resource-independent, read-only plan. Completed stages are always handed
back to the training runner on resume so their artifacts are revalidated.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHOD = "triprd_completion_controls_v1"
GROUP_STAGES = {"A": ("cod",), "B": ("source", "cod"),
                "C": ("teacher", "source", "cod"), "D": ("teacher", "source", "cod")}
CODE_FILES = (
    "tools/run_triprd_completion_controls_queue.py", "tools/run_triprd_completion_controls.sh",
    "tools/run_triprd_completion_control.py", "tools/evaluate_triprd_completion_control.py",
    "tools/summarize_triprd_completion_controls.py", "tools/run_triprd_completion.py",
    "tools/evaluate_triprd_completion.py", "tools/completion_parallel.py",
    "models/triprd_completion.py", "models/triprd_completion_loss.py", "models/triprd_completion_control.py",
    "tools/run_triprd_staged.py", "tools/run_triprd_task.py", "tools/evaluate_triprd_task.py",
    "models/triprd_task.py", "models/triprd_task_loss.py", "evaluation/task_protocol.py",
    "tools/run_triprd_correction.py", "tools/triprd_structure_training.py",
    "tools/run_triprd_ab.py", "tools/triprd_training.py", "tools/evaluate_cod.py",
    "models/triprd_correction.py", "models/triprd_correction_loss.py", "models/dinov3_weights.py",
    "models/triprd.py", "evaluation/validation.py", "evaluation/cod_metrics.py",
    "evaluation/camoformer_metrics.py", "datasets/vdt2048.py", "datasets/cod_dataset.py",
)
INTEGER_OPTIONS = {
    "teacher_epochs": 100, "source_epochs": 20, "epochs": 100, "batch_size": 8,
    "micro_batch_size": 2, "num_workers": 4, "image_size": 384, "base_channels": 128,
    "train_backbone_blocks": 4, "locator_warmup_epochs": 0, "warmup_epochs": 5,
    "train_seed": 20261012, "split_seed": 20261012, "val_metrics_every": 5, "max_samples": 0,
}
FLOAT_OPTIONS = {
    "lr": 1e-4, "backbone_lr_scale": .1, "weight_decay": 1e-4, "kd_weight": 1.,
    "temperature": 2., "boundary_weight": .2, "aux_weight": .5, "grad_clip": 1., "val_fraction": .2,
}
BOOL_OPTIONS = ("pretrained", "amp", "detail_branch", "cue_perturbation")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--phase", choices=("all", "train", "test"), default="all")
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/completion_controls_v1_seed-20261012"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path)
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("--vdt-root", type=Path, default=Path("datasets/VDT-2048"))
    parser.add_argument("--cod-root", type=Path, default=Path("datasets/COD_benchmark"))
    parser.add_argument("--gpus", default="0,1,2,3", help="Four distinct physical IDs, assigned A/B/C/D")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    parser.add_argument("--cod-train-mode", choices=("full", "split"), default="full")
    parser.add_argument("--augmentation", choices=("flip", "crop_flip"), default="crop_flip")
    parser.add_argument("--python", default=sys.executable)
    for name, default in INTEGER_OPTIONS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in FLOAT_OPTIONS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    for name in BOOL_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually launch jobs; otherwise print a plan only")
    return parser.parse_args(argv)


def gpu_list(value):
    values = [part.strip() for part in value.split(",")]
    if len(values) != 4 or any(not part.isascii() or not part.isdecimal() for part in values):
        raise ValueError("--gpus requires four distinct nonnegative physical GPU indices")
    values = [str(int(part)) for part in values]
    if len(set(values)) != 4:
        raise ValueError("--gpus requires four distinct physical GPU indices")
    return values


def _protect_output(root):
    if root.name.casefold().startswith(("completion_v6", "triprd_completion_v6")):
        raise ValueError("Use a new completion_controls output root, not the old v6 output")
    project = ROOT.resolve()
    for directory in (root, *root.parents):
        # Project-level exports/configuration do not make every new results
        # subdirectory part of an old run. A report archive alone is never
        # evidence of a run directory, including above an external output root.
        if directory == project:
            break
        if (directory / "completion_pipeline.json").exists():
            raise ValueError("Control output must not be inside an existing v6 run")


def build_plan(args):
    root = args.output_root.expanduser().resolve()
    _protect_output(root)
    gpus = gpu_list(args.gpus)
    for name in ("teacher_epochs", "source_epochs", "epochs", "batch_size", "micro_batch_size", "val_metrics_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("num_workers", "train_backbone_blocks", "locator_warmup_epochs", "warmup_epochs", "max_samples"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if args.micro_batch_size > args.batch_size:
        raise ValueError("micro-batch-size must not exceed effective batch-size")
    if args.locator_warmup_epochs >= args.epochs:
        raise ValueError("COD budget must include joint adaptation after locator warmup")
    if args.base_channels < 4 or args.image_size < 32:
        raise ValueError("base-channels >= 4 and image-size >= 32 required")
    if any(not 0 <= getattr(args, name) < 2 ** 32 for name in ("train_seed", "split_seed")):
        raise ValueError("Seeds must be in [0, 2**32)")
    for name in FLOAT_OPTIONS:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if any(getattr(args, name) <= 0 for name in ("lr", "backbone_lr_scale", "temperature", "grad_clip", "aux_weight")):
        raise ValueError("Learning rates, temperature, grad-clip and aux-weight must be positive")
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be in (0, 1)")
    if args.model_variant == "tiny" and args.pretrained:
        raise ValueError("tiny diagnostics require --no-pretrained")
    if args.model_variant != "tiny" and not args.pretrained:
        raise ValueError("Real models require generic pretrained weights")
    if args.cod_train_mode == "full" and args.max_samples:
        raise ValueError("full4040 cannot use max-samples")
    official = args.cod_train_mode == "full" and args.model_variant != "tiny" and not args.max_samples
    if args.phase == "test" and not official:
        raise ValueError("Official tests require a real unrestricted full4040 run")

    common = [args.python, "-u", str(ROOT / "tools/run_triprd_completion_control.py"),
              "--device", args.device, "--model-variant", args.model_variant,
              "--split-manifest", str(args.split_manifest.expanduser().resolve()),
              "--vdt-root", str(args.vdt_root.expanduser().resolve()),
              "--cod-root", str(args.cod_root.expanduser().resolve()),
              "--augmentation", args.augmentation, "--no-data-parallel"]
    for name in (*INTEGER_OPTIONS, *FLOAT_OPTIONS):
        common += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    for name in BOOL_OPTIONS:
        common.append(("--" if getattr(args, name) else "--no-") + name.replace("_", "-"))
    for name in ("dino_repo", "dino_weights"):
        if getattr(args, name):
            common += ["--" + name.replace("_", "-"), str(getattr(args, name).expanduser().resolve())]
    if args.resume:
        common.append("--resume")

    groups = {}
    for group, gpu in zip(GROUP_STAGES, gpus):
        jobs = []
        if args.phase != "test":
            for stage in GROUP_STAGES[group]:
                command = [*common, "--group", group, "--stage", stage,
                           "--out", str(root / group / stage), "--cod-train-mode",
                           args.cod_train_mode if stage == "cod" else "split"]
                if stage == "source" and group in ("C", "D"):
                    command += ["--teacher-checkpoint", str(root / group / "teacher/best.pt")]
                if stage == "cod" and group != "A":
                    command += ["--source-checkpoint", str(root / group / "source/best.pt")]
                jobs.append({"stage": stage, "command": command,
                             "log": str(root / group / stage / "run.log")})
        if official and args.phase in ("all", "test"):
            command = [args.python, "-u", str(ROOT / "tools/evaluate_triprd_completion_control.py"),
                       "--run", str(root / group / "cod"), "--device", args.device,
                       "--cod-root", str(args.cod_root.expanduser().resolve())]
            if args.dino_repo:
                command += ["--dino-repo", str(args.dino_repo.expanduser().resolve())]
            jobs.append({"stage": "test", "command": command, "log": str(root / group / "test.log")})
        groups[group] = {"gpu": gpu, "jobs": jobs}
    summarize = args.cod_train_mode == "split" or official and args.phase in ("all", "test")
    summary = {"command": [args.python, "-u", str(ROOT / "tools/summarize_triprd_completion_controls.py"),
                           "--root", str(root)], "log": str(root / "summary.log")} if summarize else None
    return {"method": METHOD, "output_root": str(root), "phase": args.phase,
            "effective_batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size,
            "groups": groups, "summary": summary}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration(args, plan):
    config = {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in {"phase", "execute", "resume"}}
    config["gpus"] = [details["gpu"] for details in plan["groups"].values()]
    value = {"method": METHOD, "config": config,
             "groups": {group: {"gpu": plan["groups"][group]["gpu"], "stages": list(stages)}
                        for group, stages in GROUP_STAGES.items()},
             "manifest_sha256": _sha256(args.split_manifest.expanduser().resolve()),
             "source_sha256": {name: _sha256(ROOT / name) for name in CODE_FILES}}
    return {**value, "config_sha256": _digest(value)}


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.025 * (attempt + 1))


@contextmanager
def _lock(root):
    path = root / "completion_controls_queue.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid()}, stream)
    except FileExistsError as exc:
        raise RuntimeError(f"Queue lock exists: {path}; inspect its PID before manual recovery") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _environment(gpu):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


class _WindowsJob:
    """Own this child's descendants without credential-dependent taskkill."""
    def __init__(self, child):
        import ctypes
        from ctypes import wintypes
        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]
        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in (
                "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [
                (name, wintypes.DWORD) for name in ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]
        self.ctypes, self.Accounting = ctypes, Accounting
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.api.AssignProcessToJobObject(self.handle, int(child._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def wait_empty(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            info = self.Accounting()
            if not self.api.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(info), self.ctypes.sizeof(info), None):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            if not info.ActiveProcesses:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned Windows process tree did not exit")
            time.sleep(.01)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _launch(job, gpu):
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a", encoding="utf-8")
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    try:
        child = subprocess.Popen(job["command"], cwd=ROOT, env=_environment(gpu),
                                 stdout=stream, stderr=subprocess.STDOUT, **options)
        if os.name == "nt":
            try:
                child._completion_job = _WindowsJob(child)
            except BaseException:
                child.kill()
                child.wait(timeout=5)
                raise
    except BaseException:
        stream.close()
        raise
    return child, stream


def _close_child(child, stream):
    job = getattr(child, "_completion_job", None)
    if job is not None:
        job.close()
    stream.close()


def _stop(children):
    """Terminate process groups, including data-loader workers, then reap children."""
    for child, _ in children:
        if os.name == "nt":
            child._completion_job.terminate()
        else:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for child, stream in children:
        try:
            if os.name == "nt":
                child._completion_job.wait_empty(max(.01, deadline - time.monotonic()))
            child.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                child.kill()
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=5)
        finally:
            _close_child(child, stream)


def execute_plan(plan, args, *, poll_seconds=.1, stop_requested=None):
    """Directly own every child; optional cancellation is useful for embedding."""
    root = Path(plan["output_root"])
    _protect_output(root)
    if args.device == "cuda" and "CUDA_VISIBLE_DEVICES" in os.environ:
        allowed = {part.strip() for part in os.environ["CUDA_VISIBLE_DEVICES"].split(",")}
        if not {group["gpu"] for group in plan["groups"].values()}.issubset(allowed):
            raise ValueError("Requested physical GPUs exceed existing CUDA_VISIBLE_DEVICES restriction")
    contract = _configuration(args, plan)
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        marker = root / "completion_controls_queue.json"
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != contract:
                raise ValueError("Queue configuration, groups, manifest or code changed; use a new output root")
            if args.phase != "test" and not args.resume:
                raise ValueError("Existing control run requires --resume to validate its stage artifacts")
        else:
            permitted = {"completion_controls_queue.lock", "pipeline.log", "queue.log", "launcher.log"}
            if any(path.name not in permitted for path in root.iterdir()):
                raise ValueError("Nonempty output has no matching controls contract; use a new output root")
            if args.phase == "test":
                raise ValueError("Test phase requires an existing matched controls training contract")
            _write(marker, contract)
        status = {"method": METHOD, "status": "running", "phase": args.phase,
                  "config_sha256": contract["config_sha256"], "started_at": _utc(),
                  "groups": {group: {"gpu": details["gpu"], "status": "pending",
                                      "jobs": {job["stage"]: {"status": "pending", "log": job["log"]}
                                               for job in details["jobs"]}}
                             for group, details in plan["groups"].items()},
                  "summary": {"status": "pending" if plan["summary"] else "not_requested"}}
        state_path = root / "queue_status.json"
        _write(state_path, status)
        active = {}
        next_index = {group: 0 for group in GROUP_STAGES}
        summary_child = None
        exit_code = 0
        try:
            while any(details["status"] in ("pending", "running") for details in status["groups"].values()):
                if stop_requested and stop_requested():
                    raise KeyboardInterrupt("Queue cancellation requested")
                changed = False
                for group, spec in plan["groups"].items():
                    group_state = status["groups"][group]
                    if group in active:
                        child, stream, job = active[group]
                        code = child.poll()
                        if code is None:
                            continue
                        _close_child(child, stream)
                        del active[group]
                        group_state["jobs"][job["stage"]].update(
                            status="complete" if code == 0 else "failed", exit_code=code, finished_at=_utc())
                        changed = True
                        if code:
                            group_state["status"] = "failed"
                            exit_code = 1
                            for pending in spec["jobs"][next_index[group]:]:
                                group_state["jobs"][pending["stage"]].update(status="skipped", reason="predecessor_failed")
                            print(f"[{group}/{job['stage']}] failed ({code}); other groups continue", flush=True)
                            continue
                        print(f"[{group}/{job['stage']}] complete", flush=True)
                    if group_state["status"] in ("failed", "complete"):
                        continue
                    if next_index[group] == len(spec["jobs"]):
                        group_state["status"] = "complete"
                        changed = True
                        continue
                    job = spec["jobs"][next_index[group]]
                    next_index[group] += 1
                    try:
                        child, stream = _launch(job, spec["gpu"])
                    except OSError as exc:
                        group_state["status"] = "failed"
                        group_state["jobs"][job["stage"]].update(status="failed", error=str(exc), exit_code=None)
                        for pending in spec["jobs"][next_index[group]:]:
                            group_state["jobs"][pending["stage"]].update(status="skipped", reason="predecessor_failed")
                        exit_code = 1
                    else:
                        active[group] = (child, stream, job)
                        group_state["status"] = "running"
                        group_state["jobs"][job["stage"]].update(status="running", pid=child.pid, started_at=_utc())
                        print(f"[{group}/{job['stage']}] GPU {spec['gpu']}; pid={child.pid}; log={job['log']}", flush=True)
                    changed = True
                if changed:
                    _write(state_path, status)
                if active:
                    time.sleep(poll_seconds)
            if not exit_code and plan["summary"]:
                job = plan["summary"]
                child, stream = _launch(job, "")
                summary_child = (child, stream)
                status["summary"].update(status="running", pid=child.pid, log=job["log"], started_at=_utc())
                _write(state_path, status)
                while child.poll() is None:
                    if stop_requested and stop_requested():
                        raise KeyboardInterrupt("Queue cancellation requested")
                    time.sleep(poll_seconds)
                code = child.returncode
                _close_child(child, stream)
                summary_child = None
                status["summary"].update(status="complete" if code == 0 else "failed", exit_code=code, finished_at=_utc())
                if code:
                    exit_code = 1
            elif exit_code and plan["summary"]:
                status["summary"].update(status="skipped", reason="not_all_groups_complete")
            status["status"] = "failed" if exit_code else "complete"
        except KeyboardInterrupt:
            status["status"] = "interrupted"
            exit_code = 130
            for group, (_, _, job) in active.items():
                status["groups"][group]["status"] = "interrupted"
                status["groups"][group]["jobs"][job["stage"]]["status"] = "interrupted"
            for details in status["groups"].values():
                if details["status"] == "pending":
                    details["status"] = "interrupted"
                for job_state in details["jobs"].values():
                    if job_state["status"] == "pending":
                        job_state.update(status="skipped", reason="queue_interrupted")
            if status["summary"]["status"] in ("pending", "running"):
                status["summary"]["status"] = "interrupted" if summary_child else "skipped"
        except BaseException as exc:
            status["status"] = "failed"
            status["error"] = str(exc)
            raise
        finally:
            _stop([(child, stream) for child, stream, _ in active.values()]
                  + ([summary_child] if summary_child else []))
            status["finished_at"] = _utc()
            _write(state_path, status)
    return exit_code


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **plan}, ensure_ascii=False, indent=2))
        return 0
    return execute_plan(plan, args)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
