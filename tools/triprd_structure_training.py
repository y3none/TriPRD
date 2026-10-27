"""Strict checkpoint boundaries for the typed TriPRD structure experiments.

Source training checkpoints are deliberately incompatible with the old mean-
fusion experiments.  COD transfer uses a separate artifact containing only the
complete structure function (frozen reference encoder and edge predictors).
The trainable COD encoder, propagation operators and readout never transfer.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from tools.triprd_training import sha256_file


METHOD = "triprd_structure_v2"
BRANCH_FORMAT = "triprd_structure_branch_v2"
PROTOCOL_KEYS = frozenset({
    "method", "model_variant", "base_channels", "image_size",
    "rgb_normalization", "relation_dim", "relation_kernel", "structure_target_spec", "reference_encoder_policy",
    "sensor_preprocessing",
})
SOURCE_STAGES = frozenset({"B/vdt", "C/vdt", "D/vdt", "E/vdt"})
SOURCE_CONTRACT_KEYS = frozenset({
    "vdt_train_sha256", "vdt_val_sha256", "vdt_train_count", "vdt_val_count",
    "train_seed", "split_seed", "vdt_epochs", "batch_size", "learning_rate", "weight_decay",
    "aux_weight", "semantic_weight", "structure_weight", "distillation_weight",
    "selection_metric", "augmentation", "scheduler",
})
_ARTIFACT_KEYS = frozenset({
    "format", "structure_branch", "protocol", "source_stage", "source_epoch",
    "source_seed", "source_checkpoint_path", "source_checkpoint_sha256",
    "structure_keys", "structure_state_sha256",
})

SENSOR_PREPROCESSING = {
    "I;16*": "divide by 65535 before resize, preserving original integer mode",
    "I/F": "divide by max(finite original-image maximum, 1) before resize",
    "thermal_color": "preserve RGB color (including palette/RGBA conversion), divide by 255; not calibrated temperature",
    "thermal_scalar": "normalize in original scalar mode and repeat to 3 channels for batching",
    "depth_other": "convert L and divide by 255; retain a single depth channel",
    "output_channels": {"thermal": 3, "depth": 1, "validity": 1},
    "validity": "pixel valid iff every normalized channel finite in [0,1]; zero depth invalid; zero thermal valid",
    "resize": "bilinear valid-weighted value and coverage; align_corners=False",
    "supervision_coverage": "input pixel valid iff coverage >= 0.99; feature-area coverage >= 0.99",
    "teacher_invalid_input": "zero fill; no NaN/Inf enters teacher",
}


def observation_tensor(image, size, *, depth=False):
    """Resize scalar depth or color-preserving thermal with pixel validity.

    Every thermal sample has three channels, including repeated grayscale,
    so mixed grayscale/pseudocolor source files can share a DataLoader batch.
    """
    integer16 = image.mode.startswith("I;16")
    if integer16 or image.mode in {"I", "F"}:
        array = np.asarray(image, dtype=np.float32).copy()
        finite = array[np.isfinite(array)]
        scale = 65535.0 if integer16 else max(float(finite.max()) if finite.size else 1.0, 1.0)
    else:
        array = np.asarray(image.convert("L" if depth else "RGB"), dtype=np.float32).copy()
        scale = 255.0
    array = array[None] if array.ndim == 2 else np.transpose(array, (2, 0, 1))
    value = torch.from_numpy(array / scale)[None]
    if not depth and value.shape[1] == 1:
        value = value.repeat(1, 3, 1, 1)
    valid = (torch.isfinite(value) & (value >= 0) & (value <= 1)).all(dim=1, keepdim=True)
    if depth:
        valid &= value > 0
    coverage = F.interpolate(valid.float(), (size, size), mode="bilinear", align_corners=False)
    numerator = F.interpolate(torch.where(valid, value, 0), (size, size), mode="bilinear", align_corners=False)
    observed = numerator / coverage.clamp_min(1e-6)
    accepted = coverage >= 0.99
    observed = torch.where(accepted, observed, 0).clamp(0, 1)
    return observed[0], accepted[0], scale


class StructurePairDataset(Dataset):
    """Same paired horizontal flip as v1, with explicit sensor validity."""
    def __init__(self, items, size, augment=False):
        self.items, self.size, self.augment = items, size, augment
        self.sensor_records = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from tools import run_triprd_ab as shared
        item = self.items[index]
        sample, data_index = item.source
        if item.kind != "vdt":
            return shared.PairDataset([item], self.size, self.augment)[0]
        images = {key: data_index.read_image(sample[key]) for key in ("rgb", "thermal", "depth", "mask")}
        if self.augment and random.random() < 0.5:
            images = {key: value.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for key, value in images.items()}
        result = {"key": item.key, "rgb": shared._tensor(images["rgb"], self.size, rgb=True),
                  "mask": shared._tensor(images["mask"], self.size, mask=True)}
        record = {}
        for kind in ("thermal", "depth"):
            result[kind], result[f"{kind}_valid"], scale = observation_tensor(
                images[kind], self.size, depth=kind == "depth")
            result[f"{kind}_normalization_divisor"] = torch.tensor(scale, dtype=torch.float64)
            record[kind] = {"original_mode": images[kind].mode, "normalization_divisor": scale,
                            "output_channels": result[kind].shape[0],
                            "validity_channels": result[f"{kind}_valid"].shape[0]}
        self.sensor_records[item.key] = record
        return result


def _json_value(value: Any) -> str:
    """Canonical JSON also rejects tensor/pickle objects in provenance."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint protocol must contain finite JSON metadata") from error


def _check_protocol(protocol: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(expected, Mapping) or not PROTOCOL_KEYS.issubset(expected):
        raise ValueError(f"Expected protocol must include {sorted(PROTOCOL_KEYS)}")
    if expected["method"] != METHOD:
        raise ValueError(f"Expected protocol method must be {METHOD!r}")
    if not isinstance(protocol, Mapping):
        raise ValueError("Checkpoint is missing its structure protocol")
    for key, value in expected.items():
        if key not in protocol:
            raise ValueError(f"Checkpoint protocol is missing {key!r}")
        if _json_value(protocol[key]) != _json_value(value):
            raise ValueError(f"Checkpoint protocol mismatch for {key}: source={protocol[key]!r}, requested={value!r}")
    # Preserve the complete finite JSON run provenance: split hashes and source
    # data/training configuration remain auditable when this artifact is reused.
    # This explicitly excludes tensors and any serialized optimizer state.
    return json.loads(_json_value(dict(protocol)))


def _check_source_protocol(protocol, expected):
    """Source reuse is bound to exact data partitions and optimization budget."""
    contract = expected.get("source_contract") if isinstance(expected, Mapping) else None
    if not isinstance(contract, Mapping) or set(contract) != SOURCE_CONTRACT_KEYS:
        raise ValueError("Source checkpoint reuse requires the complete source_contract, including VDT partition hashes")
    for part in ("train", "val"):
        digest = contract[f"vdt_{part}_sha256"]
        count = contract[f"vdt_{part}_count"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid source VDT {part} partition hash")
        if type(count) is not int or count < 1:
            raise ValueError(f"Source VDT {part} partition must be nonempty")
    return _check_protocol(protocol, expected)


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint must be a mapping: {path}")
    return state


def _check_tensor_state(state: Any, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} must contain a nonempty tensor state")
    checked = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} has an invalid state key")
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_quantized:
            raise ValueError(f"{label} {key!r} must be a dense tensor")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{label} {key!r} has non-finite values")
        checked[key] = value
    return checked


def _check_target_state(module: nn.Module, state: Any, *, label: str) -> dict[str, torch.Tensor]:
    checked = _check_tensor_state(state, label=label)
    target = module.state_dict()
    if set(checked) != set(target):
        missing = sorted(set(target) - set(checked))
        unexpected = sorted(set(checked) - set(target))
        raise ValueError(f"{label} state keys mismatch: missing={missing}, unexpected={unexpected}")
    for key, value in checked.items():
        if value.shape != target[key].shape:
            raise ValueError(f"{label} shape mismatch for {key}: {tuple(value.shape)} != {tuple(target[key].shape)}")
        if value.dtype != target[key].dtype:
            raise ValueError(f"{label} dtype mismatch for {key}: {value.dtype} != {target[key].dtype}")
    return checked


def _digest_state(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = _json_value([key, str(tensor.dtype), list(tensor.shape)]).encode("utf-8")
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def state_digest(module: nn.Module) -> str:
    """Hash actual parameter/buffer values, including nonpersistent buffers."""
    state = {f"parameter:{key}": value for key, value in module.named_parameters(remove_duplicate=False)}
    state.update({f"buffer:{key}": value for key, value in module.named_buffers(remove_duplicate=False)})
    return _digest_state(state)


def _source_metadata(state: Mapping[str, Any], path: Path, *, expected_stage: str) -> dict[str, Any]:
    if state.get("stage") != expected_stage:
        raise ValueError(f"Expected a {expected_stage} checkpoint, got {state.get('stage')!r}: {path}")
    for key in ("epoch", "seed"):
        if type(state.get(key)) is not int or (key == "epoch" and state[key] < 1):
            raise ValueError(f"Checkpoint {key} must be an integer" + (" >= 1" if key == "epoch" else ""))
    return {
        "source_stage": state["stage"], "source_epoch": state["epoch"],
        "source_seed": state["seed"], "source_checkpoint_path": str(path.resolve()),
        "source_checkpoint_sha256": sha256_file(path),
    }


def export_structure_checkpoint(source_checkpoint: Path, destination: Path, *,
                               expected_protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Export only ``structure_branch.*`` from a v2 B/C/D/E source checkpoint."""
    source_checkpoint, destination = Path(source_checkpoint), Path(destination)
    if source_checkpoint.resolve() == destination.resolve():
        raise ValueError("Structure artifact must not overwrite its full source checkpoint")
    state = _read_checkpoint(source_checkpoint)
    stage = state.get("stage")
    if stage not in SOURCE_STAGES:
        raise ValueError(f"Structure export requires B/vdt, C/vdt, D/vdt or E/vdt, got {stage!r}")
    protocol = _check_source_protocol(state.get("protocol"), expected_protocol)
    provenance = _source_metadata(state, source_checkpoint, expected_stage=stage)
    full_state = _check_tensor_state(state.get("model"), label="Source model")
    branch = {key.removeprefix("structure_branch."): value.detach().cpu().clone()
              for key, value in full_state.items() if key.startswith("structure_branch.")}
    branch = _check_tensor_state(branch, label="Structure branch")
    artifact = {
        "format": BRANCH_FORMAT, "structure_branch": branch, "protocol": protocol,
        **provenance, "structure_keys": sorted(branch), "structure_state_sha256": _digest_state(branch),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return {key: value for key, value in artifact.items() if key != "structure_branch"}


def load_structure_checkpoint(model: nn.Module, path: Path, *,
                             expected_protocol: Mapping[str, Any],
                             expected_stage: str | None = None) -> dict[str, Any]:
    """Load a verified branch artifact without touching the other modules."""
    path = Path(path)
    state = _read_checkpoint(path)
    if state.get("format") != BRANCH_FORMAT:
        raise ValueError(f"Expected a {BRANCH_FORMAT} artifact; full and legacy checkpoints are not accepted")
    if set(state) != _ARTIFACT_KEYS:
        raise ValueError("Structure artifact has missing or unexpected fields; only branch tensors and provenance are allowed")
    protocol = _check_source_protocol(state.get("protocol"), expected_protocol)
    stage = state.get("source_stage")
    if stage not in SOURCE_STAGES or (expected_stage is not None and stage != expected_stage):
        raise ValueError(f"Structure artifact source stage mismatch: got {stage!r}, expected {expected_stage or sorted(SOURCE_STAGES)!r}")
    for key in ("source_epoch", "source_seed"):
        if type(state[key]) is not int or (key == "source_epoch" and state[key] < 1):
            raise ValueError(f"Invalid structure artifact metadata: {key}")
    if not isinstance(state["source_checkpoint_path"], str) or not state["source_checkpoint_path"]:
        raise ValueError("Invalid structure artifact metadata: source_checkpoint_path")
    for key in ("source_checkpoint_sha256", "structure_state_sha256"):
        value = state[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid structure artifact metadata: {key}")
    branch_module = getattr(model, "structure_branch", None)
    if not isinstance(branch_module, nn.Module):
        raise ValueError("Structure transfer requires model.structure_branch")
    branch = _check_target_state(branch_module, state.get("structure_branch"), label="Structure branch")
    if state["structure_keys"] != sorted(branch):
        raise ValueError("Structure artifact keys differ from its provenance")
    if state["structure_state_sha256"] != _digest_state(branch):
        raise ValueError("Structure artifact tensor digest differs from its provenance")
    branch_module.load_state_dict(branch, strict=True)
    return {
        **{key: value for key, value in state.items() if key not in {"structure_branch", "protocol"}},
        "protocol": protocol, "artifact_path": str(path.resolve()),
        "artifact_sha256": sha256_file(path), "transfer_scope": "structure_branch",
        "preprocessing_verified": True,
    }


def load_full_checkpoint(model: nn.Module, path: Path, *, expected_stage: str,
                         expected_protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Strict CPU deserialization returning metadata, never optimizer tensors."""
    path = Path(path)
    state = _read_checkpoint(path)
    check = _check_source_protocol if expected_stage.endswith(("/vdt", "/teacher")) else _check_protocol
    protocol = check(state.get("protocol"), expected_protocol)
    provenance = _source_metadata(state, path, expected_stage=expected_stage)
    raw = _check_target_state(model, state.get("model"), label="Full model")
    model.load_state_dict(raw, strict=True)
    return {**provenance, "protocol": protocol, "transfer_scope": "full",
            "preprocessing_verified": True}
