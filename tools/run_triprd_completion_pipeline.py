#!/usr/bin/env python3
"""Run the single v6 teacher -> RGB source -> COD completion pipeline."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("teacher", "source", "cod")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--phase", choices=("all", *STAGES, "test"), default="all")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--vdt-root", default="datasets/VDT-2048")
    p.add_argument("--cod-root", default="datasets/COD_benchmark")
    p.add_argument("--gpus", default="2,3,0,1,6")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cod-train-mode", choices=("full", "split"), default="full")
    for name, default in (("teacher-epochs",100),("source-epochs",20),("epochs",100),
                          ("batch-size",8),("num-workers",4),("image-size",384),
                          ("base-channels",128),("train-backbone-blocks",4),
                          ("locator-warmup-epochs",3),("warmup-epochs",5),
                          ("train-seed",20261012),("split-seed",20261012),
                          ("val-metrics-every",5),("max-samples",0)):
        p.add_argument("--"+name, type=int, default=default)
    for name, default in (("lr",1e-4),("backbone-lr-scale",.1),("weight-decay",1e-4),
                          ("kd-weight",1.),("temperature",2.),("boundary-weight",.2),
                          ("aux-weight",.5),("grad-clip",1.)):
        p.add_argument("--"+name, type=float, default=default)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true", help="Without this flag, only print the commands")
    return p.parse_args(argv)


def gpu_list(value):
    values = [x.strip() for x in value.split(",")]
    if not values or any(not x.isdecimal() for x in values) or len(set(values)) != len(values):
        raise ValueError("--gpus requires distinct physical GPU indices, e.g. 2,3,0,1,6")
    return values


def build_plan(a):
    root = a.output_root.expanduser().resolve()
    gpus = gpu_list(a.gpus)
    parallel = a.data_parallel and a.device == "cuda" and len(gpus) > 1
    if parallel and a.batch_size < len(gpus):
        raise ValueError("Global batch size must be at least the GPU count")
    if a.phase == "test" and (a.cod_train_mode != "full" or a.model_variant == "tiny" or a.max_samples):
        raise ValueError("Official tests require a real unrestricted full4040 COD run")
    common = [sys.executable, "-u", str(ROOT / "tools/run_triprd_completion.py"),
              "--device", a.device, "--model-variant", a.model_variant,
              "--split-manifest", str(a.split_manifest.expanduser().resolve()),
              "--vdt-root", str(a.vdt_root), "--cod-root", str(a.cod_root)]
    for key in ("teacher_epochs","source_epochs","epochs","batch_size","num_workers","image_size",
                "base_channels","train_backbone_blocks","locator_warmup_epochs","warmup_epochs",
                "train_seed","split_seed","val_metrics_every","max_samples","lr","backbone_lr_scale",
                "weight_decay","kd_weight","temperature","boundary_weight","aux_weight","grad_clip"):
        common.extend(["--"+key.replace("_","-"), str(getattr(a,key))])
    common.append("--pretrained" if a.pretrained else "--no-pretrained")
    common.append("--amp" if a.amp else "--no-amp")
    if parallel:
        common.append("--data-parallel")
    for key in ("dino_repo", "dino_weights"):
        if getattr(a,key):
            common.extend(["--"+key.replace("_","-"), str(getattr(a,key).expanduser().resolve())])
    jobs = []
    for stage in STAGES:
        command = [*common,"--stage",stage,"--out",str(root/stage),
                   "--cod-train-mode",a.cod_train_mode if stage == "cod" else "split"]
        if stage == "source":
            command += ["--teacher-checkpoint",str(root/"teacher/best.pt")]
        if stage == "cod":
            command += ["--source-checkpoint",str(root/"source/best.pt")]
        if a.resume:
            command.append("--resume")
        jobs.append({"stage":stage,"command":command,"log":str(root/stage/"run.log")})
    evaluate = [sys.executable,"-u",str(ROOT/"tools/evaluate_triprd_completion.py"),
                "--run",str(root/"cod"),"--device",a.device,"--cod-root",str(a.cod_root)]
    if a.dino_repo:
        evaluate += ["--dino-repo",str(a.dino_repo.expanduser().resolve())]
    if a.phase == "all":
        if a.cod_train_mode == "full" and a.model_variant != "tiny" and not a.max_samples:
            jobs.append({"stage":"test","command":evaluate,"log":str(root/"test.log")})
    elif a.phase == "test":
        jobs = [{"stage":"test","command":evaluate,"log":str(root/"test.log")}]
    else:
        jobs = [job for job in jobs if job["stage"] == a.phase]
    return {"method":"triprd_completion_v6","output_root":str(root),"gpus":gpus,
            "data_parallel":parallel,"batch_size_global":a.batch_size,"jobs":jobs}


def _write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _lock(root):
    path = root / "completion_pipeline.lock"
    try:
        with path.open("x",encoding="utf-8") as f:
            json.dump({"pid":os.getpid()},f)
    except FileExistsError as exc:
        raise RuntimeError(f"Pipeline lock exists: {path}; inspect that PID before recovery") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def execute_plan(plan, a):
    root = Path(plan["output_root"])
    root.mkdir(parents=True,exist_ok=True)
    # A phase can be resumed separately, but all stages must retain the same configuration.
    config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()
              if k not in {"phase","resume","execute"}}
    config["manifest_sha256"] = hashlib.sha256(a.split_manifest.read_bytes()).hexdigest()
    config["method"] = "triprd_completion_v6"
    marker = root / "completion_pipeline.json"
    with _lock(root):
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != config:
                raise ValueError("Pipeline configuration changed; use a new output root")
        else:
            if any(p.is_dir() and p.name in {"teachers","screen","final"} for p in root.iterdir()):
                raise ValueError("Use a new v6 output directory, separate from prior experiments")
            _write(marker,config)
        environment = os.environ.copy()
        if a.device == "cuda":
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(plan["gpus"])
        environment["PYTHONUNBUFFERED"] = "1"
        status = {"method":plan["method"],"status":"running","phase":a.phase,"jobs":{}}
        state_path = root / "pipeline_status.json"
        child = None
        try:
            for job in plan["jobs"]:
                stage = job["stage"]
                log = Path(job["log"])
                log.parent.mkdir(parents=True,exist_ok=True)
                print(f"[{stage}] starting; log={log}",flush=True)
                with log.open("a",encoding="utf-8") as stream:
                    child = subprocess.Popen(job["command"],cwd=ROOT,env=environment,
                                             stdout=stream,stderr=subprocess.STDOUT)
                    status["jobs"][stage] = {"status":"running","pid":child.pid,"log":str(log)}
                    _write(state_path,status)
                    code = child.wait()
                    child = None
                status["jobs"][stage].update(status="complete" if code == 0 else "failed",exit_code=code)
                _write(state_path,status)
                if code:
                    raise RuntimeError(f"{stage} failed with exit {code}; inspect {log}")
                print(f"[{stage}] complete",flush=True)
            status["status"] = "complete"
        except BaseException:
            status["status"] = "interrupted" if child is not None else "failed"
            if child is not None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            raise
        finally:
            _write(state_path,status)
    return 0


def main(argv=None):
    a = parse_args(argv)
    plan = build_plan(a)
    if not a.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    return execute_plan(plan,a)


if __name__ == "__main__":
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM,interrupt)
    raise SystemExit(main())
