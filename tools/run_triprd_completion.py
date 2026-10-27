#!/usr/bin/env python3
"""TriPRD v6: full RGB object-conditioned completion transfer.

One route: new multimodal teacher -> RGB source completion -> COD adaptation.
Official tests are opt-in, only after fixed-last-epoch full4040 training.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from tools import run_triprd_task as v4
from tools import run_triprd_staged as v5
from models.triprd_completion import CompletionStudent, CompletionTeacher, COMPLETION_MODEL_SPEC
from models.triprd_completion_loss import supervised_loss, output_distillation, COMPLETION_LOSS_SPEC
from tools.triprd_structure_training import state_digest, _digest_state, SENSOR_PREPROCESSING
from tools.triprd_training import build_optimizer, build_scheduler, sha256_file

METHOD = "triprd_completion_v6"
TRANSFER_SCOPE = "all_student_except_cod_locator"
CODE_FILES = tuple(dict.fromkeys((
    "models/triprd_completion.py", "models/triprd_completion_loss.py",
    "tools/run_triprd_completion.py", "tools/evaluate_triprd_completion.py", "tools/completion_parallel.py",
    "tools/run_triprd_staged.py", *v4.CODE_FILES)))
COMMON_FIELDS = (
    "model_variant", "base_channels", "image_size", "pretrained", "train_backbone_blocks",
    "detail_branch", "batch_size", "num_workers", "train_seed", "split_seed", "lr",
    "backbone_lr_scale", "weight_decay", "warmup_epochs", "amp", "grad_clip", "augmentation",
    "boundary_weight", "aux_weight", "teacher_epochs", "source_epochs", "kd_weight",
    "temperature", "cue_perturbation", "locator_warmup_epochs", "data_parallel", "max_samples", "val_fraction")
METADATA_FIELDS = (
    "generic_initial_state_sha256", "initial_state_sha256", "cod_locator_before_sha256",
    "cod_locator_after_sha256", "refiner_before_sha256", "refiner_after_sha256",
    "source_provenance", "teacher_provenance", "stage_exposure")
_digest, _json = v4._digest, v4._json
_save_json, _save_pt, run_lock = v4._save_json, v4._save_pt, v4.run_lock
infrastructure, shared = v4.infrastructure, v4.shared
model_options, _loaders = v4.model_options, v4._loaders
PREDICTION_PROTOCOL, TEST_COUNTS = v4.PREDICTION_PROTOCOL, v4.TEST_COUNTS
prediction_image, evaluate_dataset_strict = v4.prediction_image, v4.evaluate_dataset_strict


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("teacher", "source", "cod"), default="cod")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vdt-root", default="datasets/VDT-2048")
    p.add_argument("--cod-root", default="datasets/COD_benchmark")
    p.add_argument("--model-variant", choices=("dinov3_vitb16", "tiny"), default="dinov3_vitb16")
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--dino-weights", type=Path)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-backbone-blocks", type=int, default=4)
    p.add_argument("--detail-branch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=100, help="Fixed COD epoch budget")
    p.add_argument("--source-epochs", type=int, default=20)
    p.add_argument("--teacher-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8, help="Global batch across all visible GPUs")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-seed", type=int, default=20261012)
    p.add_argument("--split-seed", type=int, default=20261012)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--val-fraction", type=float, default=.2)
    p.add_argument("--val-metrics-every", type=int, default=5)
    p.add_argument("--cod-train-mode", choices=("split", "full"), default="split")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--augmentation", choices=("flip", "crop_flip"), default="crop_flip")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-scale", type=float, default=.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--locator-warmup-epochs", type=int, default=3,
                   help="COD epochs updating only fresh coarse locator before joint adaptation")
    p.add_argument("--cue-perturbation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kd-weight", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=2.)
    p.add_argument("--boundary-weight", type=float, default=.2)
    p.add_argument("--aux-weight", type=float, default=.5)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=False,
                   help="Train forwards across visible CUDA GPUs; batch-size remains global")
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--source-checkpoint", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--test-after-training", action="store_true")
    return p.parse_args(argv)


def validate_args(a):
    for name in ("epochs", "teacher_epochs", "source_epochs", "batch_size", "val_metrics_every"):
        if getattr(a, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("num_workers", "max_samples", "warmup_epochs", "locator_warmup_epochs", "train_backbone_blocks"):
        if getattr(a, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if a.base_channels < 4 or a.image_size < 32:
        raise ValueError("base-channels >= 4 and image-size >= 32 required")
    for name in ("lr", "backbone_lr_scale", "temperature", "grad_clip"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "kd_weight", "boundary_weight", "aux_weight"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if a.aux_weight <= 0:
        raise ValueError("aux-weight must be positive so detached source cues have coarse GT supervision")
    if not 0 < a.val_fraction < 1:
        raise ValueError("val-fraction must be in (0, 1)")
    if a.model_variant == "tiny" and a.pretrained:
        raise ValueError("tiny is diagnostic only; use --no-pretrained")
    if a.model_variant != "tiny" and not a.pretrained:
        raise ValueError("Real runs require explicit generic pretrained DINO weights")
    if a.model_variant != "tiny" and not a.split_manifest and not a.eval_only:
        raise ValueError("Real runs, including prepare-only, require the original shared --split-manifest")
    if a.cod_train_mode == "full" and (a.stage != "cod" or a.max_samples):
        raise ValueError("full4040 requires COD stage without max-samples")
    if a.stage != "cod" and (a.eval_only or a.test_after_training):
        raise ValueError("Source/teacher stages cannot access official COD tests")
    if a.test_after_training and (a.cod_train_mode != "full" or a.model_variant == "tiny"):
        raise ValueError("Official tests require a real full4040 COD run")
    if a.eval_only and (a.resume or a.prepare_only or a.test_after_training):
        raise ValueError("eval-only cannot combine with resume/prepare/test-after-training")
    if a.stage == "cod" and a.locator_warmup_epochs >= a.epochs:
        raise ValueError("COD budget must include at least one joint-adaptation epoch")
    if not (a.eval_only or a.prepare_only):
        if (a.stage == "source") != bool(a.teacher_checkpoint):
            raise ValueError("Exactly source stage requires a new v6 --teacher-checkpoint")
        if (a.stage == "cod") != bool(a.source_checkpoint):
            raise ValueError("Exactly COD stage requires --source-checkpoint")
    if a.stage != "source" and a.teacher_checkpoint:
        raise ValueError("A teacher checkpoint must never enter COD optimization")
    if a.stage != "cod" and a.source_checkpoint:
        raise ValueError("Only COD stage accepts a source checkpoint")
    if a.data_parallel and torch.device(a.device).type != "cuda":
        raise ValueError("--data-parallel requires CUDA")


def prepare_data(a):
    # Reuse split/coverage checks without changing imported module globals.
    bridge = copy.copy(a)
    bridge.stage = "source" if a.stage in {"teacher", "source"} else "cod"
    splits, manifest = v5.prepare_data(bridge)
    manifest["method"] = METHOD
    return splits, manifest


def _common_contract(a, resources):
    return {"method": METHOD, **{key: getattr(a, key) for key in COMMON_FIELDS}, **resources,
            "source_sha256": {p: sha256_file(ROOT / p) for p in CODE_FILES},
            "model_spec": COMPLETION_MODEL_SPEC, "loss_spec": COMPLETION_LOSS_SPEC,
            "prediction_protocol": PREDICTION_PROTOCOL, "sensor_preprocessing": SENSOR_PREPROCESSING,
            "split_manifest_sha256": sha256_file(a.split_manifest) if a.split_manifest else None}


def stage_exposure(a):
    source = a.stage in {"teacher", "source"}
    return {"optimization": ["VDT_train"] if source else ["COD_train"],
            "selection": ["VDT_val"] if source else [] if a.cod_train_mode == "full" else ["COD_val"],
            "teacher_in_stage": a.stage == "source", "source_batches_in_cod": 0,
            "inference_inputs": "RGB/T/D" if a.stage == "teacher" else "RGB", "oracle_cue": False}


def _budget(a):
    return a.teacher_epochs if a.stage == "teacher" else a.source_epochs if a.stage == "source" else a.epochs


def _contract(a, manifest, resources, source_provenance=None, teacher_provenance=None):
    active = ("vdt_train", "vdt_val") if a.stage in {"teacher", "source"} else ("cod_train", "cod_val")
    return {**_common_contract(a, resources), "stage": a.stage, "epochs": a.epochs,
            "budget_epochs": _budget(a), "val_metrics_every": a.val_metrics_every,
            "cod_train_mode": a.cod_train_mode,
            **{key: manifest[key] if key in active else [] for key in ("vdt_train", "vdt_val", "cod_train", "cod_val")},
            "source_provenance": source_provenance, "teacher_provenance": teacher_provenance,
            "stage_exposure": stage_exposure(a), "transfer_scope": TRANSFER_SCOPE,
            "selection": "source_validation_gt_loss" if a.stage != "cod" else "last_epoch" if a.cod_train_mode == "full" else "macro_S_alpha",
            "cod_optimizer_reset": True, "cod_training_rng_reset": True,
            "kd_target": "same_rgb_cue_stage_masks_and_boundaries", "hidden_feature_kd": False}


def read_checkpoint(path, contract=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("method") != METHOD or not isinstance(state.get("contract"), dict):
        raise ValueError("Not a completion-v6 checkpoint; old v4/v5 teachers and students are incompatible")
    saved = state["contract"]
    if saved.get("method") != METHOD or state.get("contract_sha256") != _digest(saved):
        raise ValueError("Invalid completion-v6 checkpoint contract hash")
    if state.get("stage") not in {"teacher", "source", "cod"} or state.get("stage") != saved.get("stage"):
        raise ValueError("Checkpoint stage differs from its contract")
    if contract is not None and contract != saved:
        changed = sorted(key for key in set(saved) | set(contract) if saved.get(key) != contract.get(key))
        raise ValueError(f"Checkpoint contract mismatch: {changed}")
    if (not isinstance(state.get("model"), dict) or not state["model"] or
            any(not isinstance(key, str) or key.startswith("module.") or not isinstance(value, torch.Tensor)
                for key, value in state["model"].items())):
        raise ValueError("Checkpoint must contain unwrapped completion model tensors")
    return state


def _tensor_subset(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def _module_digest(module):
    return _digest_state(module.state_dict())




def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _check_partition(contract, prefix, allow_empty_val=False):
    train, val = contract.get(prefix + "_train"), contract.get(prefix + "_val")
    if not isinstance(train, list) or not train or not isinstance(val, list) or (not val and not allow_empty_val):
        raise ValueError("Missing training/validation membership")
    values = train + val
    allowed = {"VDT"} if prefix == "vdt" else {"COD10K", "CAMO"}
    if any(not isinstance(key, str) or "/" not in key or key.split("/", 1)[0] not in allowed or not key.split("/", 1)[1].strip() for key in values):
        raise ValueError("Invalid dataset sample ID")
    if len({key.casefold() for key in values}) != len(values):
        raise ValueError("Duplicate IDs or overlapping train/validation membership")


def _check_provenance(proof, stage, consumer, formal=False):
    if not isinstance(proof, dict) or not isinstance(proof.get("contract"), dict):
        raise ValueError("Missing predecessor provenance")
    contract = proof["contract"]
    budget_key = "teacher_epochs" if stage == "teacher" else "source_epochs"
    if (contract.get("method") != METHOD or contract.get("stage") != stage or
            proof.get("contract_sha256") != _digest(contract) or
            proof.get("completed_epochs") != consumer.get(budget_key) or
            contract.get("budget_epochs") != consumer.get(budget_key) or
            type(proof.get("epoch")) is not int or not 1 <= proof["epoch"] <= proof["completed_epochs"]):
        raise ValueError("Invalid predecessor stage, fixed budget or selection provenance")
    for key in ("checkpoint_sha256", "completed_checkpoint_sha256"):
        value = proof.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing predecessor artifact digest")
    common = (*COMMON_FIELDS, "generic_weights_sha256", "generic_config_sha256", "dino_source_sha256",
              "training_device_count", "compute_parallel", "source_sha256", "model_spec", "loss_spec",
              "prediction_protocol", "sensor_preprocessing", "split_manifest_sha256")
    if any(contract.get(key) != consumer.get(key) for key in common):
        raise ValueError("Predecessor common resources/settings differ")
    if (contract.get("cod_train") or contract.get("cod_val") or contract.get("source_provenance") is not None or
            contract.get("selection") != "source_validation_gt_loss"):
        raise ValueError("Predecessor source stages must remain isolated from COD")
    _check_partition(contract, "vdt")
    if formal and (len(contract["vdt_train"]), len(contract["vdt_val"])) != (838, 210):
        raise ValueError("Formal predecessor requires the fixed 838/210 source split")
    exposure = stage_exposure(argparse.Namespace(stage=stage, cod_train_mode="split"))
    if contract.get("stage_exposure") != exposure:
        raise ValueError("Predecessor exposure differs from its stage")
    if stage == "source":
        _check_provenance(contract.get("teacher_provenance"), "teacher", contract, formal=formal)
        teacher_contract = contract["teacher_provenance"]["contract"]
        if any(contract[key] != teacher_contract[key] for key in ("vdt_train", "vdt_val")):
            raise ValueError("Teacher and source membership differs")
    elif contract.get("teacher_provenance") is not None:
        raise ValueError("Teacher cannot have predecessor teacher provenance")

def validate_completed_run(path, formal=False):
    """Verify local best/last artifacts without loading training or test data."""
    path = Path(path)
    directory = path.parent if path.name == "comparison.json" else path
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    state = read_checkpoint(directory / "best.pt")
    contract = state["contract"]
    last = read_checkpoint(directory / "last.pt", contract)
    budget = contract.get("budget_epochs")
    budget_key = "teacher_epochs" if state["stage"] == "teacher" else "source_epochs" if state["stage"] == "source" else "epochs"
    if type(budget) is not int or budget < 1 or budget != contract.get(budget_key):
        raise ValueError("Invalid fixed stage budget")
    if (report.get("method") != METHOD or report.get("stage") != state["stage"] or
            report.get("status") != "complete" or report.get("protocol") != contract or
            report.get("contract_sha256") != state["contract_sha256"] or
            report.get("checkpoint_sha256") != sha256_file(directory / "best.pt") or
            report.get("completed_epochs") != budget or last.get("epoch") != budget):
        raise ValueError("Checkpoint requires a complete matching fixed-budget v6 report")
    for container in (state.get("config", {}), last.get("config", {}), report.get("config", {})):
        if container.get("stage") != state["stage"]:
            raise ValueError("Configuration stage differs from checkpoint")
        for key in (*COMMON_FIELDS, "epochs", "val_metrics_every", "cod_train_mode"):
            if container.get(key) != contract.get(key):
                raise ValueError(f"Configuration differs from contract: {key}")
    history = report.get("history", [])
    if ([row.get("epoch") for row in history] != list(range(1, budget + 1)) or
            last.get("history") != history or state.get("history") != history[:state.get("epoch", 0)]):
        raise ValueError("History does not prove complete fixed-budget optimization")
    if any(row.get("train", {}).get("optimizer_steps", 0) < 1 for row in history):
        raise ValueError("Every epoch must contain an optimizer update")
    if state["stage"] in {"teacher", "source"}:
        if contract.get("cod_train") or contract.get("cod_val"):
            raise ValueError("Source optimization must not access COD data")
        if any(row.get("validation", {}).get("objective") != "supervised_gt_only" for row in history):
            raise ValueError("Source selection must use GT only")
        losses = [row["validation"]["loss"] for row in history]
        if not all(math.isfinite(loss) for loss in losses):
            raise ValueError("Nonfinite source selection objective")
        chosen = min(range(len(losses)), key=losses.__getitem__) + 1
        score = -losses[chosen - 1]
    elif contract.get("cod_train_mode") == "full":
        if contract.get("selection") != "last_epoch" or contract.get("cod_val"):
            raise ValueError("Full4040 must select the fixed last epoch without validation")
        chosen, score = budget, float(budget)
    else:
        scored = [(row["epoch"], row["validation"]["macro"]["S_alpha"]) for row in history if "validation" in row]
        if not scored:
            raise ValueError("Development selection requires original-size validation")
        chosen, score = max(scored, key=lambda item: item[1])
    if any(obj.get("best_epoch") != chosen or obj.get("best_score") != score for obj in (state, last, report)) or state["epoch"] != chosen:
        raise ValueError("Selected checkpoint disagrees with the prescribed selection rule")
    if report.get("selection_metric") != contract.get("selection"):
        raise ValueError("Reported selection metric differs from contract")
    digests = ("generic_initial_state_sha256", "initial_state_sha256", "refiner_before_sha256", "refiner_after_sha256")
    if state["stage"] != "teacher":
        digests += ("cod_locator_before_sha256", "cod_locator_after_sha256")
    if any(not _valid_digest(report.get(key)) for key in digests):
        raise ValueError("Missing initialization/transfer tensor digest")
    for key in METADATA_FIELDS:
        if state.get(key) != last.get(key) or state.get(key) != report.get(key):
            raise ValueError(f"Initialization/provenance evidence differs: {key}")
    for key in ("source_provenance", "teacher_provenance", "stage_exposure"):
        if report.get(key) != contract.get(key):
            raise ValueError(f"Report evidence differs from contract: {key}")
    for artifact in (state, last):
        subset = _tensor_subset(artifact["model"], "refiner.")
        if not subset or artifact.get("trained_refiner_sha256") != _digest_state(subset):
            raise ValueError("Saved refiner tensor hash differs from checkpoint evidence")
    if state["stage"] == "source":
        teacher = contract.get("teacher_provenance")
        if not teacher or teacher.get("contract", {}).get("stage") != "teacher" or teacher.get("contract", {}).get("method") != METHOD:
            raise ValueError("Source run requires a new completed v6 teacher provenance")
        if teacher.get("contract_sha256") != _digest(teacher["contract"]) or teacher.get("completed_epochs") != contract["teacher_epochs"]:
            raise ValueError("Invalid completed teacher provenance")
    if state["stage"] == "cod":
        source = contract.get("source_provenance")
        if contract.get("vdt_train") or contract.get("vdt_val") or contract.get("teacher_provenance") is not None:
            raise ValueError("COD must contain no source data or live teacher")
        if not source or source.get("contract", {}).get("stage") != "source" or source.get("contract", {}).get("method") != METHOD:
            raise ValueError("COD requires completed v6 source provenance")
        if source.get("contract_sha256") != _digest(source["contract"]) or source.get("completed_epochs") != contract["source_epochs"]:
            raise ValueError("Invalid completed source provenance")
        if any(not _valid_digest(source.get(key)) for key in ("refiner_source_sha256", "transferred_state_sha256", "transferred_keys_sha256")):
            raise ValueError("Missing full-completion transfer digest")
        if report["cod_locator_before_sha256"] != report["cod_locator_after_sha256"]:
            raise ValueError("Source transfer overwrote the fresh COD locator")
        if report["refiner_after_sha256"] != source.get("refiner_source_sha256") or source.get("transfer_scope") != TRANSFER_SCOPE:
            raise ValueError("Complete source refiner was not preserved")
        nested_teacher = source["contract"].get("teacher_provenance")
        if not nested_teacher or nested_teacher.get("contract", {}).get("method") != METHOD or nested_teacher.get("contract", {}).get("stage") != "teacher":
            raise ValueError("Missing v6 teacher in source provenance")
        if nested_teacher.get("contract_sha256") != _digest(nested_teacher["contract"]):
            raise ValueError("Invalid nested teacher contract hash")
    if report.get("prediction_protocol") != PREDICTION_PROTOCOL or contract.get("prediction_protocol") != PREDICTION_PROTOCOL:
        raise ValueError("Prediction protocol differs from the fixed COD protocol")
    if contract.get("model_spec") != COMPLETION_MODEL_SPEC or contract.get("loss_spec") != COMPLETION_LOSS_SPEC:
        raise ValueError("Model/loss specification differs from completion v6")
    if any(not isinstance(contract.get(key), list) for key in ("vdt_train", "vdt_val", "cod_train", "cod_val")):
        raise ValueError("Stage memberships must be explicit lists")
    expected_counts = {key: len(contract.get(key, [])) for key in ("vdt_train", "vdt_val", "cod_train", "cod_val")}
    if report.get("sample_counts") != expected_counts:
        raise ValueError("Reported sample counts differ from training membership")
    _check_partition(contract, "vdt" if state["stage"] != "cod" else "cod", allow_empty_val=contract.get("cod_train_mode") == "full")
    expected_exposure = stage_exposure(argparse.Namespace(stage=state["stage"], cod_train_mode=contract["cod_train_mode"]))
    if contract.get("stage_exposure") != expected_exposure:
        raise ValueError("Stage exposure does not match the isolated stage")
    predecessor = contract.get("source_provenance") if state["stage"] == "cod" else contract.get("teacher_provenance")
    if state["stage"] != "teacher":
        _check_provenance(predecessor, "source" if state["stage"] == "cod" else "teacher", contract, formal=formal)
    if formal:
        from collections import Counter
        config = report["config"]
        if (state["stage"] != "cod" or contract.get("cod_train_mode") != "full" or contract.get("model_variant") != "dinov3_vitb16" or
                contract.get("pretrained") is not True or contract.get("max_samples") != 0 or config.get("max_samples") != 0 or
                Counter(key.split("/", 1)[0] for key in contract["cod_train"]) != {"COD10K": 3040, "CAMO": 1000} or
                len(set(key.casefold() for key in contract["cod_train"])) != 4040 or report.get("formal") is not True or
                report.get("result_scope") != "benchmark"):
            raise ValueError("Official evaluation requires a real completed full4040 COD run")
    return report


def _completed_predecessor(path, expected_stage, a, manifest, resources):
    path = Path(path)
    if path.name != "best.pt":
        raise ValueError("Transfer requires selected best.pt with a complete report")
    report = validate_completed_run(path.parent)
    state = read_checkpoint(path)
    contract = state["contract"]
    if state["stage"] != expected_stage:
        raise ValueError(f"Expected completed {expected_stage} checkpoint")
    common = _common_contract(a, resources)
    if any(contract.get(key) != value for key, value in common.items()):
        raise ValueError("Predecessor resources, settings, code, initialization or fixed budgets differ")
    for key in ("vdt_train", "vdt_val"):
        if manifest[key] and contract.get(key) != manifest[key]:
            raise ValueError("Source membership differs from original split manifest")
    provenance = {"checkpoint_sha256": sha256_file(path), "completed_checkpoint_sha256": sha256_file(path.parent / "last.pt"),
                  "contract_sha256": state["contract_sha256"], "contract": contract,
                  "epoch": state["epoch"], "completed_epochs": report["completed_epochs"]}
    return state, provenance


def _load_teacher(a, manifest, resources, device):
    if a.stage != "source":
        raise ValueError("A teacher may be constructed only during source training")
    state, provenance = _completed_predecessor(a.teacher_checkpoint, "teacher", a, manifest, resources)
    teacher = CompletionTeacher(**model_options(a, pretrained=False)).to(device)
    teacher.load_state_dict(state["model"], strict=True)
    teacher.eval().requires_grad_(False)
    return teacher, provenance


def _load_source_complete(model, a, manifest, resources):
    if a.stage != "cod":
        return None
    state, provenance = _completed_predecessor(a.source_checkpoint, "source", a, manifest, resources)
    if state.get("generic_initial_state_sha256") != state_digest(model):
        raise ValueError("Source and COD must use the same generic student initialization")
    current = model.state_dict()
    keys = {key for key in current if not key.startswith("locators.cod.")}
    saved_keys = {key for key in state["model"] if not key.startswith("locators.cod.")}
    if keys != saved_keys or not any(key.startswith("refiner.") for key in keys):
        raise ValueError("Full completion transfer keys differ")
    if any(current[key].shape != state["model"][key].shape for key in keys):
        raise ValueError("Full completion transfer tensor shapes differ")
    before = _module_digest(model.locators["cod"])
    current.update({key: state["model"][key] for key in keys})
    model.load_state_dict(current, strict=True)
    if _module_digest(model.locators["cod"]) != before:
        raise RuntimeError("Transfer overwrote fresh COD locator")
    source_digest = _digest_state(_tensor_subset(state["model"], "refiner."))
    if _module_digest(model.refiner) != source_digest:
        raise RuntimeError("Transfer did not preserve complete source refiner")
    return {**provenance, "transfer_scope": TRANSFER_SCOPE, "refiner_source_sha256": source_digest,
            "transferred_state_sha256": _digest_state({key: state["model"][key] for key in keys}),
            "transferred_keys_sha256": _digest(sorted(keys)), "transferred_tensors": len(keys)}


def _set_training_phase(model, a, epoch, trainable_names):
    warmup = a.stage == "cod" and epoch <= a.locator_warmup_epochs
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable_names and (not warmup or name.startswith("locators.cod.")))
    return "locator_warmup" if warmup else "joint"


def _parallel_forward(model, a, device):
    if not a.data_parallel:
        return model
    from tools.completion_parallel import BalancedDataParallel
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("DataParallel requires --device cuda (visible GPU 0 is the output device)")
    count = torch.cuda.device_count()
    if count < 2 or a.batch_size < count:
        raise ValueError("DataParallel requires >=2 visible CUDA GPUs and global batch-size >= GPU count")
    return BalancedDataParallel(model, device_ids=list(range(count)), output_device=0)


def _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch,
                 *, forward_model=None, forward_teacher=None):
    model.train()
    if teacher is not None:
        teacher.eval()
    forward_model = model if forward_model is None else forward_model
    forward_teacher = teacher if forward_teacher is None else forward_teacher
    for loader in loaders.values():
        loader.dataset.epoch = epoch
    source = a.stage in {"teacher", "source"}
    primary = loaders["vdt_train" if source else "cod_train"]
    totals, count, updates, skipped, consecutive, norm_sum = {}, 0, 0, 0, 0, 0.
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in primary:
        batch = shared._move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with infrastructure._autocast(a, device):
            if a.stage == "teacher":
                prediction = forward_model(batch["rgb"], batch["thermal"], batch["depth"],
                                           input_mode="multimodal", perturb_cue=a.cue_perturbation)
            else:
                prediction = forward_model(batch["rgb"], task="sod" if source else "cod",
                                           perturb_cue=a.cue_perturbation and source)
            losses = supervised_loss(prediction, batch["mask"], boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
            total = losses["total"]
            logged = {"supervised_" + key: value for key, value in losses.items() if key != "total"}
            if teacher is not None:
                if a.stage != "source":
                    raise RuntimeError("Teacher must never enter COD optimization")
                # Share the exact detached RGB cue, including its perturbation.
                with torch.no_grad():
                    target = forward_teacher(batch["rgb"], batch["thermal"], batch["depth"],
                                             cue=prediction["cue"].detach(), input_mode="multimodal")
                kd = output_distillation(prediction, target, batch["mask"], temperature=a.temperature,
                                         boundary_weight=a.boundary_weight)
                total = total + a.kd_weight * kd["total"]
                logged.update({key: value for key, value in kd.items() if key != "total"})
        infrastructure._finite(total, "completion total")
        scaler.scale(total).backward()
        norm, overflow = infrastructure._optimizer_step(model, optimizer, scaler, a.grad_clip)
        if overflow is not None:
            skipped += 1
            consecutive += 1
            print(f"AMP skipped update epoch={epoch} details={overflow}", flush=True)
            if consecutive >= infrastructure.MAX_CONSECUTIVE_AMP_SKIPS:
                raise FloatingPointError("Repeated nonfinite AMP updates")
        else:
            updates += 1
            consecutive = 0
            norm_sum += float(norm)
        for key, value in {**logged, "total": total}.items():
            totals[key] = totals.get(key, 0.) + float(value.detach())
        count += 1
    if updates == 0:
        raise FloatingPointError("No optimizer updates in epoch")
    return {**{key: value / count for key, value in totals.items()}, "steps": count, "optimizer_steps": updates,
            "amp_skipped_steps": skipped, "amp_scale": float(scaler.get_scale()), "grad_norm": norm_sum / updates,
            "primary_images": len(primary.dataset), "source_images": len(primary.dataset) if source else 0,
            "cod_images": len(primary.dataset) if not source else 0, "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


@torch.no_grad()
def _validate_source(model, loader, device, a):
    model.eval()
    total, n = 0., 0
    for batch in loader:
        batch = shared._move(batch, device)
        with infrastructure._autocast(a, device):
            prediction = (model(batch["rgb"], batch["thermal"], batch["depth"], input_mode="multimodal")
                          if a.stage == "teacher" else model(batch["rgb"], task="sod"))
            losses = supervised_loss(prediction, batch["mask"], boundary_weight=a.boundary_weight, aux_weight=a.aux_weight)
        infrastructure._finite(losses["total"], "source GT validation")
        total += float(losses["total"]) * len(batch["rgb"])
        n += len(batch["rgb"])
    if not n:
        raise ValueError("Empty source validation")
    return {"loss": total / n, "num_samples": n, "objective": "supervised_gt_only"}


def _fit(model, teacher, loaders, splits, a, device, contract, report):
    # Include all eventual trainable parameters before temporary locator warmup.
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    optimizer = build_optimizer(model, a.lr, a.weight_decay, a.backbone_lr_scale)
    epochs = _budget(a)
    scheduler = build_scheduler(optimizer, "cosine", epochs, min(a.warmup_epochs, epochs - 1), .05)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=a.amp and device.type == "cuda")
    forward_model = _parallel_forward(model, a, device)
    forward_teacher = _parallel_forward(teacher, a, device) if teacher is not None else None
    start, best, best_epoch, history = 1, None, None, []
    if a.resume and (a.out / "last.pt").exists():
        checkpoint = read_checkpoint(a.out / "last.pt", contract)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        infrastructure._restore_rng(checkpoint["rng"], loaders)
        start, best, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_score"], checkpoint["best_epoch"]
        history = checkpoint["history"]
        report.update({key: checkpoint[key] for key in METADATA_FIELDS})
        selected_path = a.out / "best.pt"
        saved = read_checkpoint(selected_path, contract)["epoch"] if selected_path.exists() else None
        if saved != best_epoch:
            if checkpoint["epoch"] != best_epoch:
                raise RuntimeError("Selected checkpoint missing; cannot reconstruct from last")
            _save_pt(selected_path, checkpoint)
    for epoch in range(start, epochs + 1):
        phase = _set_training_phase(model, a, epoch, trainable_names)
        row = {"epoch": epoch, "phase": phase, "lr": [group["lr"] for group in optimizer.param_groups]}
        row["train"] = _train_epoch(model, teacher, loaders, optimizer, scaler, device, a, epoch,
                                     forward_model=forward_model, forward_teacher=forward_teacher)
        score = None
        if a.stage != "cod":
            row["validation"] = _validate_source(model, loaders["vdt_val"], device, a)
            score = -row["validation"]["loss"]
        elif a.cod_train_mode == "full":
            score = float(epoch)
        elif epoch % a.val_metrics_every == 0 or epoch == epochs:
            row["validation"] = v4._validate_cod(model, splits["cod_val"], device, a)
            score = row["validation"]["macro"]["S_alpha"]
        selected = score is not None and (best is None or score > best)
        if selected:
            best, best_epoch = score, epoch
        scheduler.step()
        history.append(row)
        checkpoint = {"method": METHOD, "stage": a.stage, "contract": contract,
            "contract_sha256": _digest(contract), "config": _json(a), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_score": best, "best_epoch": best_epoch, "history": history,
            "rng": infrastructure._rng_state(loaders), "trained_refiner_sha256": _module_digest(model.refiner),
            **{key: report[key] for key in METADATA_FIELDS}}
        _save_pt(a.out / "last.pt", checkpoint)
        if selected:
            _save_pt(a.out / "best.pt", checkpoint)
        report.update(completed_epochs=epoch, best_epoch=best_epoch, best_score=best, history=history)
        _save_json(a.out / "comparison.json", report)
        _save_json(a.out / "history.json", history)
        print(f"[{a.stage}/{phase}] {epoch}/{epochs} loss={row['train']['total']:.5f} best={best_epoch} updates={row['train']['optimizer_steps']} time={row['train']['seconds']:.1f}s", flush=True)
    selected = read_checkpoint(a.out / "best.pt", contract)
    model.load_state_dict(selected["model"], strict=True)
    report.update(completed_epochs=epochs, best_epoch=best_epoch, best_score=best, history=history,
                  checkpoint_sha256=sha256_file(a.out / "best.pt"))


def _evaluate_existing(cli):
    report = validate_completed_run(cli.out, formal=True)
    state = read_checkpoint(cli.out / "best.pt")
    if state["contract"]["source_sha256"] != {p: sha256_file(ROOT / p) for p in CODE_FILES}:
        raise ValueError("Evaluation source differs from trained v6 contract")
    a = argparse.Namespace(**state["config"])
    a.out, a.device, a.cod_root = cli.out, cli.device, cli.cod_root
    a.dino_repo = cli.dino_repo or a.dino_repo
    if v4.dino_source_digest(a.dino_repo) != state["contract"]["dino_source_sha256"]:
        raise ValueError("Evaluation DINO source differs from trained v6 contract")
    model = CompletionStudent(**model_options(a, pretrained=False)).to(torch.device(a.device))
    model.load_state_dict(state["model"], strict=True)
    report["test_status"] = "running"
    _save_json(a.out / "comparison.json", report)
    report["test"] = _formal_tests(model, a)
    report["test_status"] = "complete"
    _save_json(a.out / "comparison.json", report)
    return 0


def run(a):
    validate_args(a)
    a.out = a.out.expanduser().resolve()
    if a.eval_only:
        with run_lock(a.out):
            return _evaluate_existing(a)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if a.data_parallel and (device.index not in (None, 0) or torch.cuda.device_count() < 2 or a.batch_size < torch.cuda.device_count()):
        raise ValueError("DataParallel requires device cuda, >=2 visible GPUs and global batch >= GPU count")
    resources = infrastructure._resources(a)
    if a.model_variant != "tiny":
        resources["dino_source_sha256"] = v4.dino_source_digest(a.dino_repo)
    resources["training_device_count"] = torch.cuda.device_count() if a.data_parallel else 1
    resources["compute_parallel"] = "balanced_data_parallel" if a.data_parallel else "single_device"
    splits, manifest = prepare_data(a)
    with run_lock(a.out):
        previous_path = a.out / "comparison.json"
        if a.prepare_only:
            target = a.out / "split_manifest.json"
            if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Refusing to overwrite a different manifest")
            _save_json(target, manifest)
            return 0
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
        if previous and not a.resume:
            raise ValueError("Existing run requires --resume and an identical contract")
        allowed = {"task_run.lock", "run.log", "launch.json"}
        if not previous and any(p.name not in allowed for p in a.out.iterdir()):
            raise FileExistsError("Nonempty output directory is not a resumable v6 run")
        shared.seed_everything(a.train_seed)
        model_type = CompletionTeacher if a.stage == "teacher" else CompletionStudent
        model = model_type(**model_options(a)).to(device)
        generic = state_digest(model)
        locator_before = _module_digest(model.locators["cod"]) if a.stage != "teacher" else None
        refiner_before = _module_digest(model.refiner)
        source_provenance = _load_source_complete(model, a, manifest, resources)
        locator_after = _module_digest(model.locators["cod"]) if a.stage != "teacher" else None
        refiner_after = _module_digest(model.refiner)
        teacher, teacher_provenance = _load_teacher(a, manifest, resources, device) if a.stage == "source" else (None, None)
        contract = _contract(a, manifest, resources, source_provenance, teacher_provenance)
        if previous:
            if previous.get("contract_sha256") != _digest(contract):
                raise ValueError("Existing run requires an identical contract")
            if previous.get("status") == "complete":
                validate_completed_run(a.out)
                if a.test_after_training and previous.get("test_status") != "complete":
                    return _evaluate_existing(a)
                print(f"Verified completed run: {a.out}")
                return 0
        # Constructors/transfer cannot influence fresh stage data RNG. Exact
        # resume restores saved epoch-end RNG after all constructors/loaders.
        shared.seed_everything(a.train_seed)
        loaders = _loaders(a, splits, device)
        benchmark = a.stage == "cod" and a.cod_train_mode == "full" and a.model_variant != "tiny" and not a.max_samples
        report = {"method": METHOD, "stage": a.stage, "status": "running", "config": _json(a),
            "contract_sha256": _digest(contract), "protocol": contract, "prediction_protocol": PREDICTION_PROTOCOL,
            "formal": benchmark, "result_scope": "benchmark" if benchmark else "development",
            "sample_counts": {key: len(value) for key, value in splits.items()}, "selection_metric": contract["selection"],
            "test_status": "skipped", "test": {}, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "started_at": previous.get("started_at") if previous else datetime.now(timezone.utc).isoformat(),
            "generic_initial_state_sha256": generic, "initial_state_sha256": state_digest(model),
            "cod_locator_before_sha256": locator_before, "cod_locator_after_sha256": locator_after,
            "refiner_before_sha256": refiner_before, "refiner_after_sha256": refiner_after,
            "source_provenance": source_provenance, "teacher_provenance": teacher_provenance, "stage_exposure": stage_exposure(a),
            "parameters": {"total": sum(p.numel() for p in model.parameters()),
                           "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
        _save_json(a.out / "split_manifest.json", manifest)
        _save_json(previous_path, report)
        _fit(model, teacher, loaders, splits, a, device, contract, report)
        report.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        _save_json(previous_path, report)
        validate_completed_run(a.out)
        if a.test_after_training:
            validate_completed_run(a.out, formal=True)
            report["test_status"] = "running"
            _save_json(previous_path, report)
            report["test"] = _formal_tests(model, a)
            report["test_status"] = "complete"
            _save_json(previous_path, report)
    return 0


@torch.no_grad()
def _formal_tests(model, a):
    from tools.evaluate_cod import _index_for_dataset
    model.eval()
    device = next(model.parameters()).device
    shared.NORMALIZE_RGB = a.pretrained
    results = {}
    for dataset, count in TEST_COUNTS.items():
        index = _index_for_dataset(Path(a.cod_root), dataset)
        try:
            samples = list(index.samples())
            ids = [sample.sample_id.casefold() for sample in samples]
            if len(ids) != count or len(set(ids)) != count or any(sample.image is None or sample.mask is None for sample in samples):
                raise ValueError(f"Incomplete official {dataset} test set")
            destination = a.out / "predictions" / dataset
            destination.mkdir(parents=True, exist_ok=True)
            existing = [p.stem.casefold() for p in destination.iterdir() if p.is_file()]
            if len(existing) != len(set(existing)) or set(existing) - set(ids):
                raise ValueError("Stale/duplicate prediction directory; use a clean output")
            for sample in samples:
                rgb, mask = index.read_pair(sample, strict_size=False)
                tensor = shared._tensor(rgb, a.image_size, rgb=True).unsqueeze(0).to(device)
                with infrastructure._autocast(a, device):
                    output = model(tensor, task="cod")
                prediction_image(output["logits"], mask.size).save(destination / (sample.sample_id + ".png"))
            metrics = evaluate_dataset_strict(index, a.out / "predictions", dataset, "TriPRD-Completion-v6")
            coverage = metrics.pop("coverage")
            results[dataset] = {"official": metrics, "coverage": coverage, "count": count}
            _save_json(a.out / "test_metrics.json", results)
        finally:
            index.close()
    return results


def main(argv=None):
    return run(parse_args(argv))


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
