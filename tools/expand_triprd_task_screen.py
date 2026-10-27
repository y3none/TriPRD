#!/usr/bin/env python3
"""Move an existing screen queue onto five GPUs without changing training code.

Linux server only for --execute. Default is a read-only preview. The original
plan stays immutable; epoch checkpoints and logs stay in their original paths.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_triprd_task_queue as queue

PHASE = "screen_expanded"


def validate_saved_plan(root):
    root = Path(root).resolve()
    plan = queue._read(root / "_queue/screen/plan.json")
    if plan.get("plan_sha256") != queue._digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Original screen plan digest mismatch")
    if (plan.get("method") != queue.METHOD or plan.get("phase") != "screen"
            or Path(plan["output_root"]).resolve() != root):
        raise ValueError("Expected the original screen plan for this output root")
    source = {name: queue._sha256(ROOT / name) for name in queue.SOURCE_FILES}
    if plan.get("source_sha256") != source:
        raise ValueError("Training/queue source changed; restore the original files before expansion")
    jobs = plan.get("jobs", [])
    if [job.get("name") for job in jobs] != [f"T{i}" for i in range(5)]:
        raise ValueError("Expected exactly T0 through T4 in the original screen plan")
    for job in jobs:
        config = job["expected_config"]
        expected = root / "screen" / job["name"]
        if (Path(job["output_dir"]).resolve() != expected or Path(config["out"]).resolve() != expected
                or config.get("stage") != "student" or config.get("group") != job["name"]
                or config.get("cod_train_mode") != "split" or config.get("test_after_training")
                or job["command"] != queue._command(job["command"][0], config)):
            raise ValueError(f"Unexpected training command or output path for {job['name']}")
    manifest = root / "_data/split_manifest.json"
    saved = queue._read(manifest.parent / "data_contract.json")
    if (Path(plan["split_manifest"]).resolve() != manifest
            or saved.get("contract") != plan["data_contract"]
            or saved.get("manifest_sha256") != queue._sha256(manifest)):
        raise ValueError("Shared data manifest differs from the original training plan")
    queue._verify_dependencies(plan)
    return plan


def make_expansion_plan(original, gpus):
    if len(gpus) != 5 or len(set(gpus)) != 5 or any(not str(g).isdigit() for g in gpus):
        raise ValueError("Specify exactly five distinct physical GPU IDs")
    if not set(original["gpus"]).issubset(gpus):
        raise ValueError("Keep the original GPUs and add the newly available GPUs")
    if gpus[:len(original["gpus"])] != original["gpus"]:
        raise ValueError("List the original GPUs first in their original order, then the new GPUs")
    queue._visible_gpu_check(gpus, original["device"])
    plan = json.loads(json.dumps(original))
    plan.update(phase=PHASE, gpus=gpus, original_plan_sha256=original["plan_sha256"])
    plan["plan_sha256"] = queue._digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def read_process(pid, *, include_cwd=True):
    """Read Linux identity; only targeted identity checks need ptrace-gated cwd."""
    if type(pid) is not int or pid <= 1:
        raise ValueError("Invalid process PID")
    base = Path("/proc") / str(pid)
    try:
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        identity = {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
                    "pgrp": int(fields[2]), "start_ticks": int(fields[19]), "uid": base.stat().st_uid}
        if identity["state"] in {"Z", "X"}:
            return None
        identity["argv"] = [os.fsdecode(x) for x in (base / "cmdline").read_bytes().split(b"\0") if x]
        if include_cwd:
            identity["cwd"] = str((base / "cwd").resolve(strict=True))
        return identity
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        raise RuntimeError(
            f"Cannot verify process PID {pid}: permission denied reading {exc.filename or base}. "
            "The process is not treated as exited; no new training started."
        ) from exc


def _option(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError(f"Expected one explicit process argument {flag}")
    return argv[argv.index(flag) + 1]


def assert_process_owner(process, root, *, job=None, phase="screen"):
    if process["uid"] != os.getuid() or Path(process["cwd"]).resolve() != ROOT:
        raise ValueError("Refusing to touch a process owned by another user/workspace")
    argv = process["argv"]
    script = "expand_triprd_task_screen.py" if phase == PHASE else "run_triprd_task_queue.py"
    if job is not None:
        script = "run_triprd_task.py"
    candidates = [Path(token) for token in argv if Path(token).name == script]
    if len(candidates) != 1 or (ROOT / candidates[0]).resolve() != ROOT / "tools" / script:
        raise ValueError("Process is not the expected v4 queue/runner")
    if job is None:
        if phase == "screen" and _option(argv, "--phase") != "screen":
            raise ValueError("The active queue is not the screen phase")
        if (ROOT / _option(argv, "--output-root")).resolve() != Path(root).resolve():
            raise ValueError("Active process output root differs")
    elif ((ROOT / _option(argv, "--out")).resolve() != Path(job["output_dir"]).resolve()
          or _option(argv, "--group") != job["name"] or process["pgrp"] != process["pid"]):
        raise ValueError("Active training process does not match this job")


def process_groups(group_ids):
    """Find surviving workers, even if their training parent already exited."""
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) <= 1:
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) in group_ids and fields[0] not in {"Z", "X"}:
                result.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def active_job_processes(original):
    """Also catch a runner spawned just before its queue recorded the PID."""
    outputs = {Path(job["output_dir"]).resolve() for job in original["jobs"]}
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) <= 1:
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            # /proc/<pid>/cwd can be inaccessible even for our own unrelated
            # processes (e.g. a non-dumpable helper). Inspect argv first.
            process = read_process(int(path.name), include_cwd=False)
            if process is None:
                continue
            argv = process["argv"]
            if not any(Path(token).name == "run_triprd_task.py" for token in argv) or "--out" not in argv:
                continue
            output = Path(_option(argv, "--out"))
            if not output.is_absolute():
                current = read_process(process["pid"])
                if current is None:
                    continue
                if current["start_ticks"] != process["start_ticks"]:
                    raise RuntimeError(f"Process PID {process['pid']} changed during the scan; retry expansion")
                output = Path(current["cwd"]) / output
            # Absolute --out is enough to identify a competing writer, even
            # from another checkout. Detection never sends a signal to it.
            if output.resolve() in outputs:
                result.append(process["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def _running_jobs(root, original):
    """Re-read all five jobs, including ones dispatched during handoff."""
    state = queue._read(root / "_queue/screen/status.json")
    if state.get("plan_sha256") != original["plan_sha256"]:
        raise ValueError("Original queue status does not match its plan")
    return state


def safe_stop_queue(root, original, audit):
    lock = root / "task_queue.lock"
    if not lock.exists():
        audit["queue_stop"] = "already_stopped"
        return
    content = lock.read_bytes()
    audit["queue_lock_observed"] = content.decode("utf-8")
    pid = json.loads(content)["pid"]
    process = read_process(pid)
    if process is None:
        audit["queue_stop"] = {"stale_queue_lock": json.loads(content)}
        # Keep this lock until every runner and worker group is proven stopped.
        return
    assert_process_owner(process, root)
    state = _running_jobs(root, original)
    for job in original["jobs"]:
        entry = state.get("jobs", {}).get(job["name"], {})
        child = read_process(entry["pid"]) if entry.get("pid") else None
        if child is not None:
            assert_process_owner(child, root, job=job)
            if child["ppid"] != pid:
                raise ValueError("Training process is not a child of the verified queue")
    current = read_process(pid)
    if current is None:
        audit["queue_stop"] = "exited_before_signal"
        return
    if current["start_ticks"] != process["start_ticks"]:
        raise ValueError("Queue PID was recycled; refusing to signal")
    # SIGTERM invokes the old queue's child-process-group cleanup. No killall,
    # no direct SIGKILL of the coordinator and no checkpoint deletion.
    audit["queue_before"] = process
    audit["status_before"] = state
    queue._write(root / "_queue" / PHASE / "handoff.json", audit)
    print(f"Stopping verified screen queue PID {pid}; waiting for its workers...", flush=True)
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        fd = os.pidfd_open(pid)
        try:
            current = read_process(pid)
            if current is None or current["start_ticks"] != process["start_ticks"]:
                raise ValueError("Queue identity changed before signal")
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    else:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        current = read_process(pid)
        if current is None or current["start_ticks"] != process["start_ticks"]:
            audit["queue_stop"] = "stopped"
            return
        time.sleep(.2)
    raise RuntimeError("Old queue has not exited; no new training started. Inspect its log.")


def recover_stale_locks(root, original, audit):
    state = _running_jobs(root, original)
    audit["status_after_stop"] = state
    active = active_job_processes(original)
    if active:
        raise RuntimeError(f"Training processes still alive (including unrecorded launches): {active}; no new training started")
    pending_locks, group_ids = [], set()
    for job in original["jobs"]:
        entry = state.get("jobs", {}).get(job["name"], {})
        pid = entry.get("pid")
        if pid:
            group_ids.add(pid)
            if read_process(pid) is not None:
                raise RuntimeError(f"{job['name']} still has a live process; no new training started")
        path = Path(job["output_dir"]) / "task_run.lock"
        if path.exists():
            content = path.read_bytes()
            recorded = json.loads(content)
            if recorded.get("pid") != pid or not pid:
                raise ValueError(f"{job['name']}: lock PID does not match the stopped queue status")
            pending_locks.append((path, content, job["name"]))
    survivors = process_groups(group_ids)
    if survivors:
        raise RuntimeError(f"Training/DataLoader process groups still alive: {survivors}; no new training started")
    queue_lock = root / "task_queue.lock"
    stale_queue = None
    if queue_lock.exists():
        stale_queue = queue_lock.read_bytes()
        if (stale_queue.decode("utf-8") != audit.get("queue_lock_observed")
                or read_process(json.loads(stale_queue)["pid"]) is not None):
            raise RuntimeError("A different/live queue lock exists; another coordinator may have started")
        audit["recovered_queue_lock"] = json.loads(stale_queue)
    audit["recovered_run_locks"] = {name: json.loads(content) for _, content, name in pending_locks}
    queue._write(root / "_queue" / PHASE / "handoff.json", audit)
    for path, content, _ in pending_locks:
        if path.read_bytes() != content:
            raise RuntimeError("Run lock changed during recovery")
        path.unlink()
    if stale_queue is not None:
        if queue_lock.read_bytes() != stale_queue:
            raise RuntimeError("Queue lock changed during recovery")
        queue_lock.unlink()


def verify_resume_checkpoints(original):
    """Validate saved epochs after writers stop, using the unchanged runner."""
    from tools import run_triprd_task as runner
    result = {}
    for job in original["jobs"]:
        path = Path(job["output_dir"])
        report_path = path / "comparison.json"
        if not report_path.exists():
            if (path / "last.pt").exists() or (path / "best.pt").exists():
                raise ValueError(f"{job['name']}: checkpoint exists without a run report")
            if path.exists() and any(p.name not in {"run.log", "launch.json"} for p in path.iterdir()):
                raise ValueError(f"{job['name']}: interrupted initialization left files without a report; inspect before restarting")
            result[job["name"]] = {"resume_epoch": 0}
            continue
        report = queue._read(report_path)
        for name, value in job["expected_config"].items():
            if report.get("config", {}).get(name) != value:
                raise ValueError(f"{job['name']}: saved training configuration differs for {name}")
        contract = report.get("protocol")
        if report.get("contract_sha256") != queue._digest(contract):
            raise ValueError(f"{job['name']}: report contract is invalid")
        if report.get("status") == "complete":
            queue._verify_job(job)
            runner.read_checkpoint(path / "best.pt", contract)
            result[job["name"]] = {"complete": True, "resume_epoch": report["completed_epochs"]}
        elif (path / "last.pt").exists():
            checkpoint = runner.read_checkpoint(path / "last.pt", contract)
            best_path = path / "best.pt"
            saved_best = runner.read_checkpoint(best_path, contract)["epoch"] if best_path.exists() else None
            if saved_best != checkpoint["best_epoch"] and checkpoint["epoch"] != checkpoint["best_epoch"]:
                raise ValueError(f"{job['name']}: selected checkpoint cannot be recovered from last.pt")
            result[job["name"]] = {"resume_epoch": checkpoint["epoch"]}
            del checkpoint
        else:
            result[job["name"]] = {"resume_epoch": 0, "note": "No complete saved epoch; starts from initialization"}
    return result


@contextmanager
def migration_guard(root):
    path = root / "task_expansion.lock"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
    except FileExistsError as exc:
        raise RuntimeError(f"Expansion lock exists: {path}; inspect its process before recovery") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def interrupt_guard():
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=queue.gpu_list, required=True,
                        help="Original GPUs first, then new GPUs; e.g. 2,3,0,1,6")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this expansion after interruption")
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().resolve()
    if args.execute:
        print("Checking the original screen plan, code, teacher checkpoints and data manifest...", flush=True)
    original = validate_saved_plan(root)
    plan = make_expansion_plan(original, args.gpus)
    metadata = root / "_queue" / PHASE
    target = metadata / "plan.json"
    if target.exists() and queue._read(target) != plan:
        raise ValueError("Existing expansion plan differs; keep its exact GPU list and configuration")
    if not args.execute:
        print(json.dumps({"mode": "preview", "original_plan_sha256": original["plan_sha256"],
                          "gpus": args.gpus, "training_configuration_changed": False,
                          "assignments": {job["name"]: gpu for job, gpu in zip(plan["jobs"], args.gpus)},
                          "checkpoint_policy": "resume last complete epoch; completed jobs verified and skipped",
                          "original_plan_preserved": True}, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != "linux" or not Path("/proc").is_dir():
        raise RuntimeError("Execute expansion on the Linux training server; preview works elsewhere")
    if target.exists() and not args.resume:
        raise ValueError("Expansion already prepared; rerun this tool with --resume")
    with interrupt_guard(), migration_guard(root):
        metadata.mkdir(parents=True, exist_ok=True)
        previous_audit = metadata / "handoff.json"
        if previous_audit.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            queue._write(metadata / f"handoff_previous_{stamp}.json", queue._read(previous_audit))
        audit = {"started_at": datetime.now(timezone.utc).isoformat(), "tool_sha256": queue._sha256(Path(__file__)),
                 "original_plan_sha256": original["plan_sha256"], "new_plan_sha256": plan["plan_sha256"]}
        if not target.exists():
            safe_stop_queue(root, original, audit)
            recover_stale_locks(root, original, audit)
        else:
            if (root / "task_queue.lock").exists():
                raise RuntimeError("An active/stale queue lock exists; inspect it before resuming expansion")
            # Read the new coordinator's final state for lock ownership on resume.
            expanded = queue._read(metadata / "status.json") if (metadata / "status.json").exists() else None
            if expanded and expanded.get("plan_sha256") != plan["plan_sha256"]:
                raise ValueError("Expansion status differs from its saved plan")
            if expanded:
                recover_expanded_locks(root, original, expanded, audit)
        audit["checkpoints"] = verify_resume_checkpoints(original)
        queue._write(metadata / "handoff.json", audit)
        if not target.exists():
            queue._write(target, plan)
        print(json.dumps({"gpus": args.gpus, "checkpoints": audit["checkpoints"]}, indent=2), flush=True)
        # Pre-save the plan so resume=True preserves logs and adds --resume once.
        return queue.execute_plan(plan, resume=True)


def recover_expanded_locks(root, original, state, audit):
    """Same recovery rules with the expansion's own process identities."""
    active = active_job_processes(original)
    if active:
        raise RuntimeError(f"Training processes still alive (including unrecorded launches): {active}")
    locks, groups = [], set()
    for job in original["jobs"]:
        pid = state.get("jobs", {}).get(job["name"], {}).get("pid")
        if pid:
            groups.add(pid)
            if read_process(pid) is not None:
                raise RuntimeError(f"{job['name']} is still running")
        path = Path(job["output_dir"]) / "task_run.lock"
        if path.exists():
            content = path.read_bytes()
            if not pid or json.loads(content).get("pid") != pid:
                raise ValueError(f"{job['name']}: unknown run lock")
            locks.append((path, content, job["name"]))
    if process_groups(groups):
        raise RuntimeError("Expansion workers are still running")
    audit["recovered_run_locks"] = {name: json.loads(content) for _, content, name in locks}
    queue._write(root / "_queue" / PHASE / "handoff.json", audit)
    for path, content, _ in locks:
        if path.read_bytes() != content:
            raise RuntimeError("Run lock changed during recovery")
        path.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Expansion interrupted; inspect its log and rerun this tool with --resume.", file=sys.stderr)
        raise SystemExit(130)
