#!/usr/bin/env python3
"""Explicit, evidence-gated baseline/probe/screen/final correction-v3 runs.

Omit --execute for a read-only plan. Each invocation runs exactly one phase.
One fixed source teacher and data split are shared; students always start fresh.
"""
from __future__ import annotations

import argparse
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
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_triprd_correction.py"
GROUPS = {"A": "A_cod_only", "B": "B_rgb_joint", "C": "C_full_kd", "D": "D_correction"}
TEST_COUNTS = {"NC4K": 4121, "COD10K": 2026, "CAMO": 250, "CHAMELEON": 76}
RUNNER_SOURCES = (
    "models/triprd_correction.py", "models/triprd_correction_loss.py",
    "models/dinov3_weights.py",
    "tools/run_triprd_correction.py", "tools/triprd_structure_training.py",
    "tools/run_triprd_ab.py", "tools/triprd_training.py", "models/triprd.py",
    "evaluation/validation.py", "evaluation/cod_metrics.py", "evaluation/camoformer_metrics.py",
    "tools/evaluate_cod.py", "datasets/vdt2048.py", "datasets/cod_dataset.py",
)


class EvidenceBlocked(ValueError):
    """A completed experiment has not established the next phase's premise."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--phase", required=True, choices=("baseline", "probe", "screen", "final"))
    p.add_argument("--output-root", type=Path, default=Path("results/optimization/correction_phases"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--seed", "--train-seed", dest="seed", type=int, default=20261012)
    p.add_argument("--split-seed", type=int, default=20261012)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--vdt-root", type=Path, default=Path("datasets/VDT-2048"))
    p.add_argument("--cod-root", type=Path, default=Path("datasets/COD_benchmark"))
    p.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--teacher-epochs", type=int, default=100)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--relation-kernel", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpus", help="Comma-separated physical CUDA IDs or full GPU UUIDs; one job per GPU")
    p.add_argument("--max-samples", type=int, default=0, help="Nonzero is only a pipeline diagnostic")
    p.add_argument("--val-fraction", type=float, default=.2)
    p.add_argument("--val-metrics-every", type=int, default=5)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    for name, default in (("lr", 1e-4), ("weight-decay", 1e-4), ("source-weight", .5),
                          ("kd-weight", 1.), ("relation-weight", .2), ("grad-clip", 1.)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true", help="Revalidate and continue exactly the saved recipe")
    return p.parse_args(argv)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    # Match the runner's contract serialization.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvidenceBlocked(f"{label} must be finite")
    return value


def _parse_gpu_ids(value, *, visible=None):
    """Keep physical identifiers explicit; never reinterpret an inherited CUDA mask."""
    if value is None:
        return None

    def parse(raw):
        tokens = [part.strip() for part in raw.split(",")]
        numeric = all(re.fullmatch(r"0|[1-9][0-9]*", token) for token in tokens)
        uuids = all(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", token)
                    for token in tokens)
        if not numeric and not uuids:
            raise ValueError("GPU list must contain either physical numeric IDs or full GPU UUIDs, without empty entries or mixed formats")
        tokens = [token if numeric else "GPU-" + token[4:].lower() for token in tokens]
        if len(set(tokens)) != len(tokens):
            raise ValueError("Duplicate GPU IDs would assign multiple jobs to the same GPU")
        return tokens

    requested = parse(value)
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    if inherited is not None:
        if inherited.strip() in ("", "-1"):
            raise ValueError("CUDA_VISIBLE_DEVICES disables GPUs; --gpus cannot override this restriction")
        permitted = parse(inherited)
        if not set(requested) <= set(permitted):
            raise ValueError("--gpus must be a subset of the physical identifiers in CUDA_VISIBLE_DEVICES; use the same ID format")
    return requested


def _validate_args(a):
    for name in ("epochs", "probe_epochs", "teacher_epochs", "batch_size", "base_channels", "val_metrics_every"):
        if getattr(a, name) < 1:
            raise ValueError(f"{name} must be positive")
    if a.image_size < 32 or a.image_size % 32 or a.base_channels < 4:
        raise ValueError("image-size must be a multiple of 32 and base-channels >= 4")
    if a.relation_kernel < 3 or not a.relation_kernel % 2:
        raise ValueError("relation-kernel must be odd and >= 3")
    if min(a.num_workers, a.max_samples, a.warmup_epochs) < 0 or not 0 < a.val_fraction < 1:
        raise ValueError("Nonnegative workers/limits/warmup and 0 < val-fraction < 1 required")
    if any(not 0 <= seed < 2**32 for seed in (a.seed, a.split_seed)):
        raise ValueError("Seeds must be in [0,2**32)")
    for name in ("lr", "source_weight", "kd_weight", "relation_weight", "grad_clip", "weight_decay"):
        value = getattr(a, name)
        if not math.isfinite(value) or (value < 0 if name == "weight_decay" else value <= 0):
            raise ValueError(f"Invalid {name}")
    if a.model_variant == "tiny" and a.pretrained:
        raise ValueError("Tiny diagnostics require --no-pretrained")
    if a.model_variant != "tiny" and (not a.pretrained or not a.dino_repo or not a.dino_weights):
        raise ValueError("DINOv3 requires --pretrained and explicit --dino-repo/--dino-weights")
    if a.phase == "final" and a.max_samples:
        raise ValueError("Final evaluation requires all 4040 COD training images; --max-samples must be 0")
    a.gpu_ids = _parse_gpu_ids(getattr(a, "gpus", None))
    if a.gpu_ids:
        if a.device not in ("cuda", "cuda:0"):
            raise ValueError("--gpus requires --device cuda or cuda:0; each child sees one local CUDA device")
        a.device = "cuda"


def build_plan(a):
    _validate_args(a)
    root = a.output_root.expanduser().resolve() / f"seed-{a.seed}"
    common = {name: getattr(a, name) for name in (
        "model_variant", "pretrained", "image_size", "base_channels", "relation_kernel", "batch_size",
        "num_workers", "device", "max_samples", "val_fraction", "val_metrics_every", "amp", "lr",
        "weight_decay", "source_weight", "kd_weight", "relation_weight", "grad_clip", "warmup_epochs",
        "teacher_epochs", "split_seed")}
    common["train_seed"] = a.seed
    for name in ("vdt_root", "cod_root", "dino_repo", "dino_weights"):
        if getattr(a, name) is not None:
            path = getattr(a, name).expanduser()
            # HF snapshots may link the weight file to a suffix-less blob.
            common[name] = str(path.absolute() if name == "dino_weights" else path.resolve())
    resources = {}
    weights = Path(common["dino_weights"]) if a.dino_weights else None
    repo = Path(common["dino_repo"]) if a.dino_repo else None
    if weights and weights.is_file():
        resources["generic_weights_sha256"] = _sha(weights)
        config = weights.with_name("config.json")
        resources["generic_config_sha256"] = _sha(config) if weights.suffix.lower() == ".safetensors" and config.is_file() else None
    if repo and repo.is_dir():
        files = [repo / "hubconf.py", *sorted((repo / "dinov3").rglob("*.py"))]
        if all(path.is_file() for path in files):
            resources["dino_source_sha256"] = _digest({str(path.relative_to(repo)): _sha(path) for path in files})
    if a.execute and a.model_variant != "tiny" and not {"generic_weights_sha256", "dino_source_sha256"} <= resources.keys():
        raise ValueError("Execution requires the local DINOv3 repository and generic checkpoint files")
    recipe = {"common": common, "epochs": a.epochs, "probe_epochs": a.probe_epochs,
              "python": a.python, "resources": resources,
              "source_sha256": {name: _sha(ROOT / name) for name in RUNNER_SOURCES},
              "orchestrator_sha256": _sha(Path(__file__))}
    recipe_path = root / "optimization_recipe.json"
    saved = _read(recipe_path) if recipe_path.exists() else None
    if root.exists() and any(root.iterdir()) and saved is None:
        raise ValueError(f"Existing directory has no optimization recipe; use a separate --output-root: {root}")
    if saved and saved["recipe"] != recipe:
        raise ValueError("Saved optimization recipe differs (configuration, source, generic weights, or budget); use a new --output-root")
    manifest = root / "_data/split_manifest.json"
    data_input = None
    if a.split_manifest:
        path = a.split_manifest.expanduser().resolve()
        data_input = {"path": str(path), "sha256": _sha(path)}
    elif saved:
        data_input = saved.get("data_input")
    if saved and data_input != saved.get("data_input"):
        raise ValueError("External split manifest differs from the saved recipe")
    base = [a.python, "-u", str(RUNNER)]
    for name, value in common.items():
        flag = "--" + name.replace("_", "-")
        base += [flag if value else "--no-" + name.replace("_", "-")] if isinstance(value, bool) else [flag, str(value)]

    def job(stage, group, directory, *, full=False):
        epochs = a.probe_epochs if stage == "probe" else a.epochs
        config = {**common, "stage": stage, "group": group, "epochs": epochs,
                  "cod_train_mode": "full" if full else "split", "skip_test": not full,
                  "test_datasets": ",".join(TEST_COUNTS), "max_test_samples": 0,
                  "out": str(directory), "split_manifest": str(manifest)}
        command = base + ["--stage", stage, "--group", group, "--epochs", str(epochs),
            "--out", str(directory), "--split-manifest", str(manifest), "--cod-train-mode", config["cod_train_mode"],
            "--no-skip-test" if full else "--skip-test", "--test-datasets", config["test_datasets"], "--max-test-samples", "0"]
        if stage != "teacher" and group in "CD":
            config["teacher_checkpoint"] = str(root / "teacher/best.pt")
            command += ["--teacher-checkpoint", config["teacher_checkpoint"]]
        return {"stage": stage, "group": group, "out": str(directory), "config": config, "command": command}

    teacher = job("teacher", "D", root / "teacher")
    probe = job("probe", "D", root / "probe")
    screen = [job("student", group, root / name) for group, name in GROUPS.items()]
    jobs = screen[:2] if a.phase == "baseline" else [teacher, probe] if a.phase == "probe" else screen if a.phase == "screen" else [
        job("student", group, root / "final" / name, full=True) for group, name in GROUPS.items()]
    prepare = base + ["--stage", "teacher", "--prepare-only", "--epochs", str(a.epochs),
                      "--cod-train-mode", "split", "--out", str(manifest.parent)]
    if data_input:
        prepare += ["--split-manifest", data_input["path"]]
    summary = [a.python, "-u", str(ROOT / "tools/summarize_triprd_correction.py"), str(root / "final"),
               "--groups", "A,B,C,D", "--output", str(root / "final/summary.json"),
               "--test-output-dir", str(root / "final/table")]
    return {"phase": a.phase, "root": str(root), "recipe": recipe, "recipe_sha256": _digest(recipe),
            "scheduling": {"gpus": a.gpu_ids, "max_parallel_jobs": len(a.gpu_ids) if a.gpu_ids else 1},
            "data_input": data_input, "manifest": str(manifest), "prepare_command": prepare,
            "teacher": teacher, "probe": probe, "screen": screen, "jobs": jobs, "summary_command": summary}


def _manifest(plan):
    path = Path(plan["manifest"])
    receipt = _read(Path(plan["root"]) / "data_manifest.json")
    if receipt.get("sha256") != _sha(path):
        raise EvidenceBlocked("Shared data manifest changed after preparation")
    value = _read(path)
    if value.get("split_seed") != plan["recipe"]["common"]["split_seed"] or value.get("cod_train_mode") != "split":
        raise EvidenceBlocked("Shared manifest must retain the original development split")
    for prefix in ("vdt", "cod"):
        train, val = value.get(prefix + "_train"), value.get(prefix + "_val")
        if not isinstance(train, list) or not train or not isinstance(val, list) or not val:
            raise EvidenceBlocked(f"Missing {prefix} train/validation membership")
        if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
            raise EvidenceBlocked(f"Invalid or overlapping {prefix} partitions")
    return value


def validate_result(plan, job):
    path = Path(job["out"]) / "comparison.json"
    if not path.is_file():
        raise EvidenceBlocked(f"Missing completed {job['stage']}/{job['group']} report: {path}")
    report = _read(path)
    if report.get("status") != "complete" or report.get("method") != "triprd_correction_v3":
        raise EvidenceBlocked(f"Incomplete or wrong-method report: {path}")
    mismatches = [key for key, value in job["config"].items() if report.get("config", {}).get(key) != value]
    if mismatches:
        raise EvidenceBlocked(f"Mismatched evidence configuration in {path}: {mismatches}")
    protocol = report.get("protocol", {})
    if report.get("contract_sha256") != _digest(protocol):
        raise EvidenceBlocked(f"Report contract digest mismatch: {path}")
    for key, value in job["config"].items():
        if key in protocol and protocol[key] != value:
            raise EvidenceBlocked(f"Configuration and checkpoint protocol disagree: {key}")
    if protocol.get("source_sha256") != plan["recipe"]["source_sha256"]:
        raise EvidenceBlocked(f"Evidence was produced by different source code: {path}")
    for key, value in plan["recipe"]["resources"].items():
        if protocol.get(key) != value:
            raise EvidenceBlocked(f"Evidence generic pretraining differs: {key}")
    manifest = _manifest(plan)
    for key in ("vdt_train", "vdt_val"):
        if protocol.get(key) != manifest[key]:
            raise EvidenceBlocked(f"Evidence source partition differs: {key}")
    if job["stage"] == "student":
        full = job["config"]["cod_train_mode"] == "full"
        expected_train = manifest["cod_train"] + manifest["cod_val"] if full else manifest["cod_train"]
        if set(protocol.get("cod_train", [])) != set(expected_train) or protocol.get("cod_val") != ([] if full else manifest["cod_val"]):
            raise EvidenceBlocked("Evidence COD training/validation members differ")
        if full and (len(expected_train) != 4040 or report.get("selection_metric") != "last_epoch" or
                     report.get("best_epoch") != plan["recipe"]["epochs"]):
            raise EvidenceBlocked("Final run must use all 4040 images and the fixed last epoch")
    filename = "last.pt" if job["stage"] == "probe" else "best.pt"
    checkpoint = Path(job["out"]) / filename
    if not checkpoint.is_file() or report.get("checkpoint_sha256") != _sha(checkpoint):
        raise EvidenceBlocked(f"Missing or changed checkpoint: {checkpoint}")
    if job["stage"] != "teacher" and job["group"] in "CD":
        if protocol.get("teacher_sha256") != _sha(Path(plan["root"]) / "teacher/best.pt"):
            raise EvidenceBlocked("Evidence used a different teacher checkpoint")
    if job["config"]["cod_train_mode"] == "full":
        if report.get("test_status") != "complete" or set(report.get("test", {})) != set(TEST_COUNTS):
            raise EvidenceBlocked("Final report lacks all four complete test datasets")
        for dataset, count in TEST_COUNTS.items():
            test = report["test"][dataset]
            metrics = test.get("official", test)
            if metrics.get("num_samples") != count:
                raise EvidenceBlocked(f"Incomplete final {dataset} sample count")
            for field in ("S_alpha", "E_phi_adaptive", "F_beta_w", "MAE"):
                _finite(metrics.get(field), f"{dataset}/{field}")
    return report


def teacher_gate(report):
    probe = report.get("teacher_probe", {})
    gain = _finite(probe.get("mean_relation_gain"), "Teacher mean_relation_gain")
    coverage = _finite(probe.get("positive_coverage"), "Teacher positive_coverage")
    if gain <= 0 or not 0 < coverage <= 1:
        raise EvidenceBlocked("Teacher has no positive validation relation gain/coverage; do not distill it")
    return {"mean_relation_gain": gain, "positive_coverage": coverage}


def student_gate(report):
    probe = report.get("source_recoverability", {})
    values = {key: _finite(probe.get(key), "Source probe " + key) for key in (
        "positive_mse", "positive_zero_mse", "positive_constant_mse", "positive_count", "count")}
    if not 0 < values["positive_count"] <= values["count"] or any(values[key] < 0 for key in (
            "positive_mse", "positive_zero_mse", "positive_constant_mse")):
        raise EvidenceBlocked("Source probe has no valid positive-reliability edges")
    if not values["positive_mse"] < min(values["positive_zero_mse"], values["positive_constant_mse"]):
        raise EvidenceBlocked("Student positive-edge correction MSE does not beat BOTH zero and TRAIN-fitted constant baselines")
    return values


def check_gates(plan):
    if plan["phase"] in ("baseline", "probe"):
        return {}
    teacher = validate_result(plan, plan["teacher"])
    probe = validate_result(plan, plan["probe"])
    evidence = {"teacher": teacher_gate(teacher), "source_probe": student_gate(probe),
                "report_sha256": {"teacher": _sha(Path(plan["teacher"]["out"]) / "comparison.json"),
                                  "probe": _sha(Path(plan["probe"]["out"]) / "comparison.json")}}
    if plan["phase"] == "final":
        screen_receipt = Path(plan["root"]) / "phases/screen.json"
        if not screen_receipt.is_file() or _read(screen_receipt).get("status") != "complete":
            raise EvidenceBlocked("Final phase requires a completed screen phase")
        scores, initialization = {}, set()
        for job in plan["screen"]:
            report = validate_result(plan, job)
            selected = [row for row in report.get("history", []) if row.get("epoch") == report.get("best_epoch")]
            if len(selected) != 1:
                raise EvidenceBlocked("Screen selected epoch is absent or duplicated")
            scores[job["group"]] = _finite(selected[0].get("validation", {}).get("macro", {}).get("S_alpha"), "Screen Sm")
            digest = report.get("initial_state_sha256")
            if not digest:
                raise EvidenceBlocked("Screen report lacks matched student initialization evidence")
            initialization.add(digest)
            evidence["report_sha256"][job["group"]] = _sha(Path(job["out"]) / "comparison.json")
        if len(initialization) != 1:
            raise EvidenceBlocked("Screen groups did not start from the same student initialization")
        if scores["D"] <= max(scores[group] for group in "ABC"):
            raise EvidenceBlocked(f"D must beat A, B and C on selected COD validation macro Sm before final training: {scores}")
        evidence["screen_S_alpha"] = scores
    return evidence


def _stop_gpu_children(active):
    """Terminate only process groups created by this scheduler, then reap them."""
    for item in active:
        process = item["process"]
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass
    for item in active:
        process = item["process"]
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()
        # The parent may exit while a DataLoader worker remains in its group.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if item.get("log") is not None:
            item["log"].close()
        entry = item.get("entry")
        if entry is not None and entry["status"] == "running":
            entry.update(status="cancelled", returncode=process.returncode,
                         finished_at=datetime.now(timezone.utc).isoformat())


def _run_control_command(plan, command):
    """Track preparation and summarization as well as GPU training processes."""
    if not plan.get("scheduling", {}).get("gpus"):
        return subprocess.run(command, cwd=ROOT, check=True)
    options = {"start_new_session": True} if os.name == "posix" else {}
    process = subprocess.Popen(command, cwd=ROOT, **options)
    try:
        while process.poll() is None:
            time.sleep(.1)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
    finally:
        _stop_gpu_children([{"process": process}])


def _run_gpu_batch(plan, entries, gpus, state, receipt_path):
    pending = list(entries)
    active = []
    try:
        while pending or active:
            # Process all exits before allocating freed GPUs. A failure therefore
            # cannot launch more pending jobs in the same scheduler iteration.
            finished = [(item, item["process"].poll()) for item in active]
            failures = [(item, code) for item, code in finished if code not in (None, 0)]
            if failures:
                for item, code in failures:
                    item["entry"].update(status="failed", returncode=code,
                        error=f"Child process exited with status {code}", finished_at=datetime.now(timezone.utc).isoformat())
                _write(receipt_path, state)
                item, code = failures[0]
                raise subprocess.CalledProcessError(code, item["command"])
            for item, code in finished:
                if code is None:
                    continue
                entry, job = item["entry"], item["job"]
                item["process"].wait()
                try:
                    report = validate_result(plan, job)
                    entry.update(status="completed", returncode=0,
                        comparison_sha256=_sha(Path(job["out"]) / "comparison.json"),
                        checkpoint_sha256=report["checkpoint_sha256"],
                        finished_at=datetime.now(timezone.utc).isoformat())
                except BaseException as exc:
                    entry.update(status="failed", returncode=0, error=f"{type(exc).__name__}: {exc}",
                                 finished_at=datetime.now(timezone.utc).isoformat())
                    raise
                item["log"].close()
                active.remove(item)
                _write(receipt_path, state)
                print(f"Completed {job['stage']}/{job['group']} on GPU {entry['gpu']}", flush=True)
            available = [gpu for gpu in gpus if all(item["entry"]["gpu"] != gpu for item in active)]
            while pending and available:
                job, entry, command = pending.pop(0)
                gpu = available.pop(0)
                directory = Path(job["out"])
                directory.mkdir(parents=True, exist_ok=True)
                log_path = directory / "run.log"
                stream = log_path.open("a", encoding="utf-8", buffering=1)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                entry.update(status="running", gpu=gpu, log=str(log_path),
                             started_at=datetime.now(timezone.utc).isoformat())
                _write(receipt_path, state)
                options = {"start_new_session": True} if os.name == "posix" else {}
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT, **options)
                except BaseException as exc:
                    stream.close()
                    entry.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                                 finished_at=datetime.now(timezone.utc).isoformat())
                    raise
                active.append({"job": job, "entry": entry, "command": command, "process": process, "log": stream})
                entry["pid"] = process.pid
                _write(receipt_path, state)
                print(f"Started {job['stage']}/{job['group']} on GPU {gpu}; PID {process.pid}; log: {log_path}", flush=True)
            if active:
                time.sleep(.1)
    finally:
        _stop_gpu_children(active)


def _run_gpu_jobs(plan, jobs, state, receipt_path, *, resume=False):
    """Validate completed runs serially, then schedule independent training jobs."""
    gpus = plan["scheduling"]["gpus"]
    entries = []
    for job in jobs:
        entry = {"stage": job["stage"], "group": job["group"], "out": job["out"],
                 "status": "queued", "reused": False}
        state["jobs"].append(entry)
        entries.append((job, entry, list(job["command"])))
    _write(receipt_path, state)
    try:
        for job, entry, command in entries:
            path = Path(job["out"]) / "comparison.json"
            try:
                reused = path.is_file() and _read(path).get("status") == "complete"
                entry["reused"] = reused
                if reused:
                    validate_result(plan, job)
                    command.append("--resume")  # Runner also checks the saved checkpoint contract.
                elif path.exists():
                    if not resume:
                        raise ValueError(f"Incomplete job requires --resume: {job['out']}")
                    command.append("--resume")
            except BaseException as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                             finished_at=datetime.now(timezone.utc).isoformat())
                raise
        if any(job["stage"] == "probe" for job, _, _ in entries):
            # The probe depends on a trained teacher with positive validation gain.
            for item in entries:
                if item[0]["stage"] == "probe":
                    state["evidence"]["teacher"] = teacher_gate(validate_result(plan, plan["teacher"]))
                    _write(receipt_path, state)
                _run_gpu_batch(plan, [item], gpus[:1], state, receipt_path)
        else:
            reused = [item for item in entries if item[1]["reused"]]
            pending = [item for item in entries if not item[1]["reused"]]
            _run_gpu_batch(plan, reused, gpus[:1], state, receipt_path)
            _run_gpu_batch(plan, pending, gpus, state, receipt_path)
    finally:
        for _, entry, _ in entries:
            if entry["status"] == "queued":
                entry.update(status="cancelled", finished_at=datetime.now(timezone.utc).isoformat())
        _write(receipt_path, state)


def execute_plan(plan, *, resume=False):
    root = Path(plan["root"])
    receipt_path = root / "phases" / (plan["phase"] + ".json")
    if receipt_path.exists() and not resume:
        raise ValueError("This phase already has a receipt; use --resume with the identical recipe")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "optimization_lock.json"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"Another phase may be running; inspect {lock} before removing a stale lock") from exc
    os.close(descriptor)
    state = {"phase": plan["phase"], "status": "running", "recipe_sha256": plan["recipe_sha256"],
             "scheduling": plan.get("scheduling", {"gpus": None, "max_parallel_jobs": 1}),
             "started_at": datetime.now(timezone.utc).isoformat(), "jobs": []}
    signal_installed = False
    previous_sigterm = None

    def interrupt_on_signal(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    try:
        if plan.get("scheduling", {}).get("gpus") and threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.signal(signal.SIGTERM, interrupt_on_signal)
            signal_installed = True
        _write(lock, {"pid": os.getpid(), "phase": plan["phase"]})
        _write(root / "optimization_recipe.json", {"recipe": plan["recipe"],
            "recipe_sha256": plan["recipe_sha256"], "data_input": plan["data_input"]})
        _write(root / "plans" / (plan["phase"] + ".json"), plan)
        _write(receipt_path, state)
        manifest = Path(plan["manifest"])
        if not manifest.exists():
            if plan["phase"] in ("screen", "final"):
                raise EvidenceBlocked("Earlier phases must prepare the shared split before screening or final training")
            _run_control_command(plan, plan["prepare_command"])
            _write(root / "data_manifest.json", {"path": str(manifest), "sha256": _sha(manifest)})
        elif not (root / "data_manifest.json").exists():
            # A crash can occur after the child atomically saves the manifest
            # but before this process saves its receipt. Recompute and compare
            # membership in the runner; never bless an unchecked file by hash.
            if not resume or plan["phase"] not in ("baseline", "probe"):
                raise EvidenceBlocked("Data manifest receipt is missing; resume its baseline/probe preparation first")
            _run_control_command(plan, plan["prepare_command"] + ["--resume"])
            _write(root / "data_manifest.json", {"path": str(manifest), "sha256": _sha(manifest)})
        _manifest(plan)
        state["evidence"] = check_gates(plan)
        _write(receipt_path, state)
        if plan.get("scheduling", {}).get("gpus"):
            _run_gpu_jobs(plan, plan["jobs"], state, receipt_path, resume=resume)
        for job in ([] if plan.get("scheduling", {}).get("gpus") else plan["jobs"]):
            if job["stage"] == "probe":
                state["evidence"]["teacher"] = teacher_gate(validate_result(plan, plan["teacher"]))
            path = Path(job["out"]) / "comparison.json"
            command = list(job["command"])
            reused = path.is_file() and _read(path).get("status") == "complete"
            if reused:
                validate_result(plan, job)
                command.append("--resume")  # Runner verifies checkpoint contract, no training.
            elif path.exists():
                if not resume:
                    raise ValueError(f"Incomplete job requires --resume: {job['out']}")
                command.append("--resume")
            subprocess.run(command, cwd=ROOT, check=True)
            report = validate_result(plan, job)
            state["jobs"].append({"stage": job["stage"], "group": job["group"], "reused": reused,
                "out": job["out"], "comparison_sha256": _sha(path), "checkpoint_sha256": report["checkpoint_sha256"]})
            _write(receipt_path, state)
        if plan["phase"] == "probe":
            state["evidence"]["source_probe"] = student_gate(validate_result(plan, plan["probe"]))
        if plan["phase"] == "final":
            _run_control_command(plan, plan["summary_command"])
        state["status"] = "complete"
        return 0
    except BaseException as exc:
        state.update(status="blocked" if isinstance(exc, EvidenceBlocked) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        print(state["error"], file=sys.stderr)
        return 2 if isinstance(exc, EvidenceBlocked) else 1
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _write(receipt_path, state)
        finally:
            try:
                lock.unlink(missing_ok=True)
            finally:
                if signal_installed:
                    signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv=None):
    a = parse_args(argv)
    plan = build_plan(a)
    if not a.execute:
        try:
            evidence, blocked = check_gates(plan), None
        except (EvidenceBlocked, OSError) as exc:
            evidence, blocked = {}, str(exc)
        print(json.dumps({"mode": "dry-run", "status": "blocked" if blocked else "planned",
                          "blocked_reason": blocked, "evidence": evidence, **plan}, ensure_ascii=False, indent=2))
        return 2 if blocked else 0
    return execute_plan(plan, resume=a.resume)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Optimization failed: {error}", file=sys.stderr)
        raise SystemExit(1)
