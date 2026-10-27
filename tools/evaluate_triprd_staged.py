#!/usr/bin/env python3
"""Export four official COD test rows from one completed full-pool staged-v5 run."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.task_protocol import (DATASET_COUNTS, METRIC_FIELDS, PREDICTION_PROTOCOL,
                                      evaluate_dataset_strict)
from tools.evaluate_cod import _index_for_dataset


METHOD = "triprd_staged_v5"
TRAINING_CONFIG_FIELDS = (
    "model_variant", "base_channels", "image_size", "pretrained", "train_backbone_blocks",
    "detail_branch", "batch_size", "num_workers", "train_seed", "split_seed", "lr",
    "backbone_lr_scale", "weight_decay", "warmup_epochs", "amp", "grad_clip", "augmentation",
    "boundary_weight", "aux_weight", "teacher_epochs", "epochs", "source_epochs",
    "kd_weight", "feature_weight", "temperature", "val_metrics_every",
    "cod_train_mode",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _same_json(left, right) -> bool:
    """Do not silently treat JSON true/1 or missing/null as equivalent evidence."""
    return _digest(left) == _digest(right)


def _validate_staged_provenance(state, report):
    """COD initialization and source training are explicit checkpoint evidence."""
    contract = state["contract"]
    for key in ("generic_initial_state_sha256", "initial_state_sha256",
                "cod_decoder_before_sha256", "cod_decoder_after_sha256"):
        value = state.get(key)
        if not isinstance(value, str) or len(value) != 64 or value != report.get(key):
            raise ValueError(f"Missing/inconsistent staged initialization evidence: {key}")
    if state["cod_decoder_before_sha256"] != state["cod_decoder_after_sha256"]:
        raise ValueError("COD decoder was overwritten during source transfer")
    provenance = state.get("source_provenance")
    if not _same_json(provenance, report.get("source_provenance")) or not _same_json(provenance, contract.get("source_provenance")):
        raise ValueError("Checkpoint/report source provenance differs")
    if contract["group"] == "S0":
        if provenance is not None or state["initial_state_sha256"] != state["generic_initial_state_sha256"]:
            raise ValueError("S0 must start from its generic initialization without source transfer")
    else:
        if not isinstance(provenance, dict):
            raise ValueError("Staged source initialization requires provenance")
        source = provenance.get("contract")
        if (not isinstance(source, dict) or source.get("method") != METHOD or source.get("stage") != "source"
                or source.get("group") != contract["group"] or provenance.get("contract_sha256") != _digest(source)
                or provenance.get("group") != contract["group"]):
            raise ValueError("Source contract/method/group is invalid")
        if provenance.get("completed_epochs") != contract["source_epochs"]:
            raise ValueError("Source pretraining budget is incomplete")
        if type(provenance.get("epoch")) is not int or not 1 <= provenance["epoch"] <= contract["source_epochs"]:
            raise ValueError("Invalid selected source epoch")
        if not isinstance(provenance.get("checkpoint_sha256"), str) or len(provenance["checkpoint_sha256"]) != 64:
            raise ValueError("Source checkpoint hash is missing")
        for key in ("train_seed", "split_seed", "model_variant", "image_size", "base_channels",
                    "train_backbone_blocks", "detail_branch", "source_epochs", "prediction_protocol",
                    "pretrained", "generic_weights_sha256", "generic_config_sha256",
                    "dino_source_sha256", "source_sha256"):
            if not _same_json(source.get(key), contract.get(key)):
                raise ValueError(f"Source/COD provenance differs for {key}")
        if provenance.get("generic_initial_state_sha256") != state["generic_initial_state_sha256"]:
            raise ValueError("Source/COD generic initialization evidence differs")
        if (not isinstance(provenance.get("completed_checkpoint_sha256"), str)
                or len(provenance["completed_checkpoint_sha256"]) != 64):
            raise ValueError("Completed source checkpoint evidence is missing")
        tr, va = source.get("vdt_train"), source.get("vdt_val")
        if (not isinstance(tr, list) or not isinstance(va, list) or len(tr) != 838 or len(va) != 210
                or any(not isinstance(v, str) for v in tr + va) or len(set(tr + va)) != 1048
                or source.get("cod_train") != [] or source.get("cod_val") != []):
            raise ValueError("Source stage must identify disjoint VDT838/210 and no COD optimization data")
    exposure = report.get("stage_exposure")
    if (not isinstance(exposure, dict) or exposure.get("optimization") != ["COD_train"]
            or exposure.get("selection") != [] or exposure.get("source_batches_in_cod") != 0
            or exposure.get("teacher_in_stage") is not False):
        raise ValueError("Final COD stage must exclude source batches, teachers and COD validation selection")
    if not _same_json(exposure, state.get("stage_exposure")) or not _same_json(exposure, contract.get("stage_exposure")):
        raise ValueError("Checkpoint/report stage exposure differs")
    if report["sample_counts"].get("vdt_train") != 0 or report["sample_counts"].get("vdt_val") != 0:
        raise ValueError("Source data must not be reintroduced into COD optimization")


def _validate_checkpoint_provenance(checkpoint: Path, report: dict) -> None:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Selected file is not a readable weights-only PyTorch checkpoint") from exc
    if not isinstance(state, dict) or state.get("method") != METHOD or state.get("stage") != "cod":
        raise ValueError("Selected checkpoint is not a staged-v5 student")
    contract = state.get("contract")
    if not isinstance(contract, dict) or state.get("contract_sha256") != _digest(contract):
        raise ValueError("Selected checkpoint training contract/hash is invalid")
    if (report.get("contract_sha256") != state["contract_sha256"]
            or not _same_json(report.get("protocol"), contract)):
        raise ValueError("Run report does not describe the checkpoint training contract")
    if (contract.get("method") != METHOD or contract.get("stage") != "cod"
            or contract.get("prediction_protocol") != PREDICTION_PROTOCOL):
        raise ValueError("Checkpoint method/stage/prediction protocol is inconsistent")
    config, checkpoint_config = report["config"], state.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint lacks its training configuration")
    group = contract.get("group")
    if group not in {"S0", "S1", "S2", "S3"} or any(
            container.get("group") != group for container in (state, report, config, checkpoint_config)):
        raise ValueError("Checkpoint and report student groups differ")
    if any(container.get("stage") != "cod" for container in (config, checkpoint_config)):
        raise ValueError("Checkpoint/report configuration must identify student training")
    for name in TRAINING_CONFIG_FIELDS:
        if name not in contract or name not in config or name not in checkpoint_config or not (
                _same_json(config[name], contract[name]) and _same_json(checkpoint_config[name], contract[name])):
            raise ValueError(f"Checkpoint/report training configuration differs for {name}")
    for container in (config, checkpoint_config):
        if type(container.get("max_samples")) is not int or container["max_samples"] != 0:
            raise ValueError("Checkpoint/report must both specify unrestricted training")
    epochs = config["epochs"]
    if any(type(value) is not int or value != epochs for value in (
            state.get("epoch"), state.get("best_epoch"), report.get("best_epoch"), report.get("completed_epochs"))):
        raise ValueError("Checkpoint and report must identify the completed predefined last epoch")
    history = state.get("history")
    if (not isinstance(history, list) or len(history) != epochs
            or any(not isinstance(row, dict) or type(row.get("epoch")) is not int
                   or row["epoch"] != index for index, row in enumerate(history, 1))
            or not _same_json(report.get("history"), history)):
        raise ValueError("Checkpoint/report completed training history differs")
    weights = state.get("model")
    if not isinstance(weights, dict) or not weights or any(not isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("Checkpoint must contain model tensor weights")
    if contract.get("cod_train_mode") != "full" or contract.get("cod_val") != []:
        raise ValueError("Checkpoint must use full4040 with no internal COD validation set")
    members = contract.get("cod_train")
    if (not isinstance(members, list) or len(members) != 4040
            or any(not isinstance(key, str) or not key.partition("/")[2] for key in members)
            or len({key.casefold() for key in members}) != 4040
            or Counter(key.split("/", 1)[0] for key in members) != {"COD10K": 3040, "CAMO": 1000}):
        raise ValueError("Checkpoint must enumerate unique COD10K 3040 + CAMO 1000 training members")
    for name in ("vdt_train", "vdt_val", "cod_train", "cod_val"):
        values = contract.get(name)
        count = report["sample_counts"].get(name)
        if not isinstance(values, list) or type(count) is not int or count != len(values):
            raise ValueError(f"Report sample count differs from checkpoint members for {name}")
    _validate_staged_provenance(state, report)


def validate_run_report(path: str | Path) -> dict:
    """Fail closed before evaluating or writing a benchmark result."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Run report must be a JSON object")
    for key, expected in (("method", METHOD), ("status", "complete"), ("stage", "cod"),
                          ("selection_metric", "last_epoch"), ("prediction_protocol", PREDICTION_PROTOCOL)):
        if report.get(key) != expected:
            raise ValueError(f"Run report requires {key}={expected!r}")
    config, counts = report.get("config"), report.get("sample_counts")
    if not isinstance(config, dict) or not isinstance(counts, dict):
        raise ValueError("Run report requires config and sample_counts objects")
    if config.get("cod_train_mode") != "full":
        raise ValueError("Benchmark export requires full-4040 training, not an internal split")
    variant = config.get("model_variant")
    if not isinstance(variant, str) or not variant.strip() or "tiny" in variant.casefold():
        raise ValueError("Benchmark export requires a non-tiny model_variant")
    if config.get("pretrained") is not True:
        raise ValueError("Benchmark export requires pretrained=True")
    if (isinstance(config.get("max_samples"), bool) or not isinstance(config.get("max_samples"), int)
            or config["max_samples"] != 0):
        raise ValueError("Benchmark export requires max_samples=0")
    for key, expected in (("cod_train", 4040), ("cod_val", 0)):
        if isinstance(counts.get(key), bool) or not isinstance(counts.get(key), int) or counts[key] != expected:
            raise ValueError(f"Benchmark export requires {key}={expected}")
    epochs, best_epoch = config.get("epochs"), report.get("best_epoch")
    if (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1 or
            isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch != epochs):
        raise ValueError("The selected checkpoint must be the predefined last epoch")
    if report.get("formal") is not True:
        raise ValueError("Benchmark export requires formal=True")
    checkpoint = path.parent / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Selected checkpoint is missing: {checkpoint}")
    if report.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("Selected checkpoint SHA256 differs from the run report")
    _validate_checkpoint_provenance(checkpoint, report)
    return report


def evaluate_run(run_report: str | Path, data_root: str | Path, prediction_root: str | Path) -> dict:
    report_path = Path(run_report).resolve()
    report = validate_run_report(report_path)
    config = report["config"]
    results = {}
    model = f"TriPRD-staged-v5-{config.get('group', 'student')}"
    for dataset in DATASET_COUNTS:
        with _index_for_dataset(Path(data_root), dataset) as index:
            result = evaluate_dataset_strict(index, prediction_root, dataset, model)
        coverage = result.pop("coverage")
        results[dataset] = {"official": result, "coverage": coverage}
    return {"method": METHOD, "status": "complete", "run_report": str(report_path),
            "prediction_root": str(Path(prediction_root).resolve()), "prediction_protocol": PREDICTION_PROTOCOL,
            "checkpoint_sha256": report["checkpoint_sha256"], "selected_epoch": report["best_epoch"],
            "cod_train_samples": report["sample_counts"]["cod_train"], "group": config.get("group"),
            "model_variant": config["model_variant"], "input_size": config.get("image_size"), "test": results}


def export_results(result: dict, out: str | Path) -> None:
    """Write all datasets together, never one independently selected metric."""
    out = Path(out)
    if (result.get("method") != METHOD or result.get("status") != "complete" or
            result.get("prediction_protocol") != PREDICTION_PROTOCOL or
            set(result.get("test", {})) != set(DATASET_COUNTS)):
        raise ValueError("Only a completed four-dataset staged-v5 evaluation can be exported")
    rows = []
    for dataset, expected in DATASET_COUNTS.items():
        metrics = result["test"][dataset]["official"]
        coverage = result["test"][dataset].get("coverage", {})
        if (metrics.get("num_samples") != expected or coverage.get("verified_samples") != expected or
                coverage.get("expected_full_samples") != expected or coverage.get("limited") is not False or
                coverage.get("exact_id_match") is not True):
            raise ValueError(f"{dataset}: incomplete official coverage cannot be exported")
        for key in METRIC_FIELDS:
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{dataset}: missing/non-finite metric {key}")
        rows.append({"dataset": dataset, "num_samples": metrics["num_samples"],
                     **{key: metrics[key] for key in METRIC_FIELDS}, "group": result["group"],
                     "selected_epoch": result["selected_epoch"], "cod_train_samples": result["cod_train_samples"],
                     "checkpoint_sha256": result["checkpoint_sha256"], "prediction_protocol": PREDICTION_PROTOCOL})
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(payload + "\n", encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Directory for metrics.json and metrics.csv")
    args = parser.parse_args(argv)
    result = evaluate_run(args.run_report, args.data_root, args.pred_root)
    export_results(result, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
