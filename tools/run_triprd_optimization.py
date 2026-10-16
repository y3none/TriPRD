#!/usr/bin/env python3
"""Plan or execute isolated COD, transfer and distillation optimization jobs."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def gpu_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",")]
    if not values or any(not re.fullmatch(r"0|[1-9][0-9]*", item) for item in values):
        raise argparse.ArgumentTypeError("GPUs must be comma-separated non-negative integer indices")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return values


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--phase", choices=("cod", "transfer", "distill"), default="cod")
    parser.add_argument("--gpus", type=gpu_list, default="0,1,2,3")
    parser.add_argument("--source-run", default="results/triprd_full")
    parser.add_argument("--output-root", default="results/optimization")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=20261012, help="Training seed")
    parser.add_argument("--repeat-seed", type=int,
                        help="Transfer phase only: paired A repeat seed (default: seed + 1)")
    parser.add_argument("--split-seed", type=int, default=20261012)
    parser.add_argument("--split-manifest", type=Path, help="Reuse the same exact split for every job")
    parser.add_argument("--vdt-root", default="datasets/VDT-2048")
    parser.add_argument("--cod-root", default="datasets/COD_benchmark")
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--vdt-epochs", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--execute", action="store_true", help="Launch training; omission only prints the plan")
    return parser.parse_args(argv)


def validate_plan(plan: dict) -> None:
    """Reject reuse of an output tree and any colliding job destinations."""
    run_dir = Path(plan["run_dir"]).resolve()
    if run_dir.exists():
        raise ValueError(f"Output already exists; choose a new --output-root: {run_dir}")
    seen_paths: set[str] = set()
    seen_names: set[str] = set()
    for job in plan["jobs"]:
        output = Path(job["output_dir"]).resolve()
        key = os.path.normcase(str(output))
        if key in seen_paths or job["name"] in seen_names:
            raise ValueError("Job names and output directories must be unique")
        if output.parent != run_dir:
            raise ValueError("Every job output must be a direct child of the run directory")
        if output.exists():
            raise ValueError(f"Job output already exists: {output}")
        if job["gpu"] not in plan["gpus"]:
            raise ValueError(f"Job references an unassigned GPU: {job['gpu']}")
        seen_paths.add(key)
        seen_names.add(job["name"])


def build_plan(args: argparse.Namespace) -> dict:
    if args.repeat_seed is not None and args.phase != "transfer":
        raise ValueError("--repeat-seed is only supported by --phase transfer")
    repeat_seed = (args.seed + 1 if args.repeat_seed is None else args.repeat_seed) if args.phase == "transfer" else None
    seeds = [args.seed, args.split_seed] + ([repeat_seed] if repeat_seed is not None else [])
    if any(not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Training and split seeds must be in [0, 2**32)")
    if repeat_seed == args.seed:
        raise ValueError("--repeat-seed must differ from --seed")
    source = Path(args.source_run).expanduser().resolve()
    run_name = f"seed-{args.seed}" + (f"-repeat-{repeat_seed}" if repeat_seed is not None else "")
    run_dir = Path(args.output_root).expanduser().resolve() / args.phase / run_name
    manifest = args.split_manifest.expanduser().resolve() if args.split_manifest is not None else None
    common = [
        args.python, "-u", str(ROOT / "tools" / "run_triprd_ab.py"),
        "--vdt-root", str(Path(args.vdt_root).expanduser().resolve()),
        "--cod-root", str(Path(args.cod_root).expanduser().resolve()),
        "--model-variant", "resnet50", "--pretrained", "--base-channels", "256",
        "--image-size", "352", "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs), "--vdt-epochs", str(args.vdt_epochs),
        "--max-samples", "0", "--max-test-samples", "0", "--device", "cuda",
        "--split-seed", str(args.split_seed),
        "--learning-rate", "0.0001", "--weight-decay", "0.0001",
        "--validation-metrics", "--val-metrics-every", "5", "--selection-metric", "S_alpha",
        "--skip-test",
    ]
    if manifest is not None:
        common += ["--split-manifest", str(manifest)]
    control = ["--cod-encoder-lr-scale", "1", "--cod-scheduler", "none",
               "--cod-warmup-epochs", "0", "--cod-min-lr-ratio", "0.05"]
    improved = ["--cod-encoder-lr-scale", "0.1", "--cod-scheduler", "cosine",
                "--cod-warmup-epochs", str(min(5, args.epochs - 1)), "--cod-min-lr-ratio", "0.05"]
    jobs = []

    def add(name: str, group: str, options: list[str], required: list[Path] | None = None,
            *, train_seed: int | None = None, transfer_scope: str = "full") -> None:
        train_seed = args.seed if train_seed is None else train_seed
        output = run_dir / name
        jobs.append({
            "name": name, "gpu": args.gpus[len(jobs) % len(args.gpus)],
            "train_seed": train_seed, "split_seed": args.split_seed, "transfer_scope": transfer_scope,
            "output_dir": str(output),
            "command": common + ["--groups", group, "--out", str(output),
                                  "--seed", str(train_seed), "--train-seed", str(train_seed),
                                  "--transfer-scope", transfer_scope] + options,
            "required_checkpoints": [str(path) for path in (required or [])],
        })

    if args.phase == "cod":
        add("A_control", "A", control)
        add("A_schedule", "A", improved)
        for group in ("B", "C"):
            checkpoint = source / group / "vdt" / "best.pt"
            add(f"{group}_schedule", group,
                improved + ["--vdt-checkpoint", f"{group}={checkpoint}"], [checkpoint])
    elif args.phase == "transfer":
        for group in ("B", "C"):
            checkpoint = source / group / "vdt" / "best.pt"
            add(f"{group}_encoder", group,
                improved + ["--vdt-checkpoint", f"{group}={checkpoint}"], [checkpoint],
                transfer_scope="encoder")
        add("A_control_repeat", "A", control, train_seed=repeat_seed)
        add("A_schedule_repeat", "A", improved, train_seed=repeat_seed)
    else:
        teacher = source / "teacher" / "best.pt"
        base = improved + ["--teacher-checkpoint", str(teacher), "--lambda-logit", "1"]
        # Disable gates in the first three jobs so the final job isolates gating.
        add("C_logit", "C", base + ["--lambda-relation", "0", "--lambda-boundary-kd", "0",
                                    "--gate-mode", "none"], [teacher])
        relation = base + ["--lambda-relation", "1", "--relation-sampling", "uniform"]
        add("D_relation", "D", relation + ["--lambda-boundary-kd", "0", "--gate-mode", "none"], [teacher])
        boundary = relation + ["--boundary-kd-type", "prob_mse", "--lambda-boundary-kd", "0.1"]
        add("D_boundary", "D", boundary + ["--gate-mode", "none"], [teacher])
        add("D_gate", "D", boundary + ["--gate-mode", "disagreement"], [teacher])
    plan = {"phase": args.phase, "run_dir": str(run_dir), "gpus": args.gpus, "jobs": jobs,
            "required_files": [str(manifest)] if manifest is not None else []}
    validate_plan(plan)
    return plan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute_plan(plan: dict, *, popen_factory=None, sleep=None) -> int:
    """Run one FIFO queue per GPU; a failure stops new dispatch, not active jobs."""
    validate_plan(plan)
    required = {path for job in plan["jobs"] for path in job["required_checkpoints"]}
    missing = sorted(path for path in required if not Path(path).is_file())
    if missing:
        raise ValueError("Required source checkpoints are missing: " + ", ".join(missing))
    missing_files = [path for path in plan.get("required_files", []) if not Path(path).is_file()]
    if missing_files:
        raise ValueError("Required input files (split manifest) are missing: " + ", ".join(missing_files))
    popen_factory = subprocess.Popen if popen_factory is None else popen_factory
    sleep = time.sleep if sleep is None else sleep
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "launch.json", plan)
    queues = {gpu: deque(job for job in plan["jobs"] if job["gpu"] == gpu) for gpu in plan["gpus"]}
    states = {job["name"]: {"status": "queued", "gpu": job["gpu"]} for job in plan["jobs"]}
    running = {}
    failed = False

    def finish(job: dict, returncode: int, error: str | None = None) -> None:
        state = states[job["name"]]
        state.update(status="completed" if returncode == 0 else "failed",
                     returncode=returncode, finished_at=_now())
        if error:
            state["error"] = error
        _write_json(Path(job["output_dir"]) / "exit_status.json", state)
        print(f"[{job['name']}] finished with exit status {returncode}", flush=True)

    try:
        while running or (not failed and any(queues.values())):
            # Poll all active jobs before considering a new dispatch.
            for gpu, (job, process, log) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log.close()
                finish(job, returncode)
                failed = failed or returncode != 0
                del running[gpu]

            if not failed:
                for gpu, queue in queues.items():
                    if gpu in running or not queue:
                        continue
                    job = queue.popleft()
                    directory = Path(job["output_dir"])
                    directory.mkdir(exist_ok=False)
                    state = states[job["name"]]
                    state.update(status="running", started_at=_now())
                    # The runner permits these two files in a new output directory.
                    _write_json(directory / "launch.json", job)
                    log = (directory / "run.log").open("w", encoding="utf-8")
                    environment = os.environ.copy()
                    environment.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                    try:
                        process = popen_factory(job["command"], cwd=str(ROOT), env=environment,
                                                stdout=log, stderr=subprocess.STDOUT)
                    except OSError as exc:
                        log.write(f"Unable to start training: {exc}\n")
                        log.close()
                        finish(job, 127, str(exc))
                        failed = True
                        break
                    state["pid"] = process.pid
                    running[gpu] = (job, process, log)
                    print(f"[{job['name']}] GPU {gpu}, pid {process.pid}, log {directory / 'run.log'}", flush=True)
            if running:
                sleep(0.25)
    finally:
        # Unexpected launcher I/O errors must not abandon already running jobs.
        for job, process, log in running.values():
            try:
                process.wait()
            finally:
                log.close()

    for queue in queues.values():
        for job in queue:
            states[job["name"]].update(status="skipped", reason="A prior job failed; no automatic retry")
    _write_json(run_dir / "exit_status.json", {"finished_at": _now(), "jobs": states})
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if not args.execute:
            for job in plan["jobs"]:
                command = subprocess.list2cmdline(job["command"]) if os.name == "nt" else shlex.join(job["command"])
                job["display_command"] = f"CUDA_VISIBLE_DEVICES={job['gpu']} {command}"
            print(json.dumps({"mode": "dry-run", **plan}, ensure_ascii=False, indent=2))
            return 0
        return execute_plan(plan)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
