"""Checkpoint provenance and COD optimization helpers shared by the runner."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(model, checkpoint: Path, device, *, scope: str = "full",
                    expected_stage: str | None = None,
                    expected_protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    raw = state.get("model", state) if isinstance(state, Mapping) else state
    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint has no model state: {checkpoint}")
    if expected_stage is not None and (not isinstance(state, Mapping) or state.get("stage") != expected_stage):
        raise ValueError(f"Expected a {expected_stage} checkpoint, got {state.get('stage') if isinstance(state, Mapping) else None}: {checkpoint}")
    state = dict(state)
    if expected_protocol is not None:
        protocol = state.get("protocol")
        origin = "checkpoint"
        if not protocol:
            # E001 checkpoints predate embedded protocol metadata. The raw
            # comparison on the server can still verify their preprocessing.
            for parent in list(checkpoint.resolve().parents)[:3]:
                candidate = parent / "comparison.json"
                if candidate.is_file():
                    protocol = json.loads(candidate.read_text(encoding="utf-8")).get("protocol")
                    if protocol:
                        origin = str(candidate)
                        break
        protocol = protocol if isinstance(protocol, Mapping) else {}
        for key, expected in expected_protocol.items():
            if key in protocol and protocol[key] != expected:
                raise ValueError(f"Checkpoint protocol mismatch for {key}: source={protocol[key]!r}, requested={expected!r}")
        verified = all(key in protocol for key in expected_protocol)
        if not verified:
            warnings.warn(f"Checkpoint preprocessing metadata incomplete: {checkpoint}; verify the source run before confirmation experiments", stacklevel=2)
        state["protocol"] = dict(protocol)
        state["_source_protocol_origin"] = origin if protocol else None
        state["_protocol_verified"] = verified
    if scope == "full":
        model.load_state_dict(raw, strict=True)
    elif scope == "encoder":
        if not hasattr(model, "encoder"):
            raise ValueError("Encoder-only transfer requires a student encoder")
        encoder_state = {key[len("encoder."):]: value for key, value in raw.items() if key.startswith("encoder.")}
        if not encoder_state:
            raise ValueError(f"No student encoder found in {checkpoint}")
        model.encoder.load_state_dict(encoder_state, strict=True)
    else:
        raise ValueError(f"Unknown transfer scope: {scope}")
    return state


def build_optimizer(model, lr: float, weight_decay: float, encoder_lr_scale: float = 1.0):
    if not all(math.isfinite(x) for x in (lr, weight_decay, encoder_lr_scale)) or lr <= 0 or weight_decay < 0 or encoder_lr_scale <= 0:
        raise ValueError("Learning rate and encoder scale must be positive; weight decay must be nonnegative")
    encoder_ids = {id(p) for p in model.encoder.parameters()} if hasattr(model, "encoder") else set()
    if encoder_lr_scale != 1.0 and not encoder_ids:
        raise ValueError("Differential learning rates require a student encoder")
    groups = []
    for name, scale, belongs in (("encoder", encoder_lr_scale, True), ("decoder_heads", 1.0, False)):
        parameters = [p for p in model.parameters() if p.requires_grad and (id(p) in encoder_ids) == belongs]
        if parameters:
            groups.append({"params": parameters, "lr": lr * scale, "name": name})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str, epochs: int, warmup_epochs: int = 0, min_lr_ratio: float = 0.05):
    if name not in {"none", "cosine"}:
        raise ValueError(f"Unknown scheduler: {name}")
    if not 0 <= warmup_epochs < epochs or not 0 <= min_lr_ratio <= 1:
        raise ValueError("Require 0 <= warmup_epochs < epochs and 0 <= min_lr_ratio <= 1")
    if name == "none":
        if warmup_epochs:
            raise ValueError("Warmup requires the cosine scheduler")
        return None

    def factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if warmup_epochs and epochs - warmup_epochs == 1:
            return min_lr_ratio
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)))
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
