"""Strict checkpoint boundaries for the explicit TriPRD relation experiments.

Source training checkpoints are deliberately incompatible with the old mean-
fusion experiments.  COD transfer uses a separate artifact containing only the
relation branch, so neither an RGB encoder nor a source decoder can leak across
the transfer boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tools.triprd_training import sha256_file


METHOD = "triprd_relation_v1"
BRANCH_FORMAT = "triprd_relation_branch_v1"
PROTOCOL_KEYS = frozenset({
    "method", "model_variant", "base_channels", "image_size",
    "rgb_normalization", "relation_dim", "relation_kernel",
})
SOURCE_STAGES = frozenset({"B/vdt", "C/vdt", "D/vdt"})
_ARTIFACT_KEYS = frozenset({
    "format", "relation_branch", "protocol", "source_stage", "source_epoch",
    "source_seed", "source_checkpoint_path", "source_checkpoint_sha256",
    "relation_keys", "relation_state_sha256",
})


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
        raise ValueError("Checkpoint is missing its relation protocol")
    for key, value in expected.items():
        if key not in protocol:
            raise ValueError(f"Checkpoint protocol is missing {key!r}")
        if _json_value(protocol[key]) != _json_value(value):
            raise ValueError(f"Checkpoint protocol mismatch for {key}: source={protocol[key]!r}, requested={value!r}")
    # Preserve the complete finite JSON run provenance: split hashes and source
    # data/training configuration remain auditable when this artifact is reused.
    # This explicitly excludes tensors and any serialized optimizer state.
    return json.loads(_json_value(dict(protocol)))


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


def export_relation_checkpoint(source_checkpoint: Path, destination: Path, *,
                               expected_protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Export only ``relation_branch.*`` from a new B/C/D source checkpoint."""
    source_checkpoint, destination = Path(source_checkpoint), Path(destination)
    if source_checkpoint.resolve() == destination.resolve():
        raise ValueError("Relation artifact must not overwrite its full source checkpoint")
    state = _read_checkpoint(source_checkpoint)
    stage = state.get("stage")
    if stage not in SOURCE_STAGES:
        raise ValueError(f"Relation export requires B/vdt, C/vdt or D/vdt, got {stage!r}")
    protocol = _check_protocol(state.get("protocol"), expected_protocol)
    provenance = _source_metadata(state, source_checkpoint, expected_stage=stage)
    full_state = _check_tensor_state(state.get("model"), label="Source model")
    branch = {key.removeprefix("relation_branch."): value.detach().cpu().clone()
              for key, value in full_state.items() if key.startswith("relation_branch.")}
    branch = _check_tensor_state(branch, label="Relation branch")
    artifact = {
        "format": BRANCH_FORMAT, "relation_branch": branch, "protocol": protocol,
        **provenance, "relation_keys": sorted(branch), "relation_state_sha256": _digest_state(branch),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return {key: value for key, value in artifact.items() if key != "relation_branch"}


def load_relation_checkpoint(model: nn.Module, path: Path, *,
                             expected_protocol: Mapping[str, Any],
                             expected_stage: str | None = None) -> dict[str, Any]:
    """Load a verified branch artifact without touching the other modules."""
    path = Path(path)
    state = _read_checkpoint(path)
    if state.get("format") != BRANCH_FORMAT:
        raise ValueError(f"Expected a {BRANCH_FORMAT} artifact; full and legacy checkpoints are not accepted")
    if set(state) != _ARTIFACT_KEYS:
        raise ValueError("Relation artifact has missing or unexpected fields; only branch tensors and provenance are allowed")
    protocol = _check_protocol(state.get("protocol"), expected_protocol)
    stage = state.get("source_stage")
    if stage not in SOURCE_STAGES or (expected_stage is not None and stage != expected_stage):
        raise ValueError(f"Relation artifact source stage mismatch: got {stage!r}, expected {expected_stage or sorted(SOURCE_STAGES)!r}")
    for key in ("source_epoch", "source_seed"):
        if type(state[key]) is not int or (key == "source_epoch" and state[key] < 1):
            raise ValueError(f"Invalid relation artifact metadata: {key}")
    if not isinstance(state["source_checkpoint_path"], str) or not state["source_checkpoint_path"]:
        raise ValueError("Invalid relation artifact metadata: source_checkpoint_path")
    for key in ("source_checkpoint_sha256", "relation_state_sha256"):
        value = state[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid relation artifact metadata: {key}")
    branch_module = getattr(model, "relation_branch", None)
    if not isinstance(branch_module, nn.Module):
        raise ValueError("Relation transfer requires model.relation_branch")
    branch = _check_target_state(branch_module, state.get("relation_branch"), label="Relation branch")
    if state["relation_keys"] != sorted(branch):
        raise ValueError("Relation artifact keys differ from its provenance")
    if state["relation_state_sha256"] != _digest_state(branch):
        raise ValueError("Relation artifact tensor digest differs from its provenance")
    branch_module.load_state_dict(branch, strict=True)
    return {
        **{key: value for key, value in state.items() if key not in {"relation_branch", "protocol"}},
        "protocol": protocol, "artifact_path": str(path.resolve()),
        "artifact_sha256": sha256_file(path), "transfer_scope": "relation_branch",
        "preprocessing_verified": True,
    }


def load_full_checkpoint(model: nn.Module, path: Path, *, expected_stage: str,
                         expected_protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Strict CPU deserialization returning metadata, never optimizer tensors."""
    path = Path(path)
    state = _read_checkpoint(path)
    protocol = _check_protocol(state.get("protocol"), expected_protocol)
    provenance = _source_metadata(state, path, expected_stage=expected_stage)
    raw = _check_target_state(model, state.get("model"), label="Full model")
    model.load_state_dict(raw, strict=True)
    return {**provenance, "protocol": protocol, "transfer_scope": "full",
            "preprocessing_verified": True}
