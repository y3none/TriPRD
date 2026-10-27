"""Strict local DINOv3 loading, including the HF ViT-B/16 tensor layout.

Only serialization and parameter names change: inference continues to use the
official local ``dinov3`` model. The HF layout is reversed from Hugging Face's
``src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py``. It splits
Q/K/V along dimension zero, drops the masked K bias and deterministic buffers,
and adds one axis to the mask token. Both ``layer.N`` and ``model.layer.N``
exports are supported, but mixing these layouts is an error.

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/
models/dinov3_vit/convert_dinov3_vit_to_hf.py

No learned value is taken from the newly initialized destination model. The
only reconstructed values are the mathematically inactive K bias, its fixed
mask, and the configured non-learned RoPE periods.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


CONVERSION_VERSION = "hf_vitb16_to_official_v1"

# These are the serialized HF architecture defaults, not values guessed from
# checkpoint tensor dimensions. Shape-invisible alternatives are rejected.
_HF_OPTION_DEFAULTS = {
    "num_channels": 3,
    "hidden_act": "gelu",
    "attention_dropout": 0.0,
    "layer_norm_eps": 1e-5,
    "rope_theta": 100.0,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "drop_path_rate": 0.0,
    "use_gated_mlp": False,
    "pos_embed_shift": None,
    "pos_embed_jitter": None,
    "pos_embed_rescale": 2.0,
}
_HF_REQUIRED = {
    "model_type": "dinov3_vit",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "patch_size": 16,
    "num_register_tokens": 4,
}


def _check_values(state: Mapping[str, Tensor], label: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} must be a nonempty raw tensor state_dict")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError(f"{label} must contain only string keys and tensors: {key!r}")
        if value.layout != torch.strided or value.is_meta or value.is_complex():
            raise ValueError(f"Unsupported tensor representation in {label}: {key}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite tensor in {label}: {key}")


def _validate_exact_state(backbone: nn.Module, state: Mapping[str, Tensor]) -> None:
    expected = backbone.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"DINOv3 strict state_dict mismatch; missing={missing}; unexpected={unexpected}")
    parameters = dict(backbone.named_parameters())
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise RuntimeError(
                f"DINOv3 tensor shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)}, model={tuple(expected[key].shape)}"
            )
        if key in parameters and not value.is_floating_point():
            raise ValueError(f"DINOv3 learned parameters must be floating point: {key}")
        if (value.is_floating_point() and expected[key].is_floating_point()
                and value.dtype != expected[key].dtype
                and not torch.isfinite(value.to(dtype=expected[key].dtype)).all().item()):
            raise ValueError(f"DINOv3 tensor overflows the destination dtype: {key}")


def _json_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate DINOv3 config key: {key}")
        result[key] = value
    return result


def _same_config_value(actual: Any, expected: Any) -> bool:
    # Python considers True == 1, which is not a valid architecture declaration.
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual == expected
    return actual == expected


def _validate_hf_config(path: Path, model_variant: str) -> tuple[dict[str, Any], str]:
    if model_variant != "dinov3_vitb16":
        raise ValueError("Hugging Face safetensors conversion currently supports only dinov3_vitb16")
    if not path.is_file():
        raise FileNotFoundError(
            f"Hugging Face DINOv3 weights require the original config.json beside model.safetensors: {path}"
        )
    raw = path.read_bytes()
    config = json.loads(raw, object_pairs_hook=_json_unique_object)
    if not isinstance(config, dict):
        raise ValueError("DINOv3 config.json must contain an object")
    for key, expected in _HF_REQUIRED.items():
        if key not in config or not _same_config_value(config[key], expected):
            raise ValueError(f"Incompatible DINOv3 config {key}: expected {expected!r}, got {config.get(key)!r}")
    for key, expected in _HF_OPTION_DEFAULTS.items():
        actual = config.get(key, expected)
        if not _same_config_value(actual, expected):
            raise ValueError(f"Incompatible DINOv3 config {key}: expected {expected!r}, got {actual!r}")
    architectures = config.get("architectures")
    if architectures is not None and architectures != ["DINOv3ViTModel"]:
        raise ValueError(f"Expected a base DINOv3ViTModel, got architectures={architectures!r}")
    # The public HF ViT model uses this fixed RoPE convention. Refuse custom
    # declarations instead of accepting unknown alternatives through a fallback.
    for key, expected in {
        "pos_embed_rope_normalize_coords": "separate",
        "pos_embed_rope_min_period": None,
        "pos_embed_rope_max_period": None,
        "rope_scaling": None,
        "rope_parameters": None,
    }.items():
        if key in config and not _same_config_value(config[key], expected):
            raise ValueError(f"Unsupported DINOv3 config {key}={config[key]!r}")
    return config, hashlib.sha256(raw).hexdigest()


def _validate_hf_backbone(backbone: nn.Module) -> None:
    for name, expected in {
        "embed_dim": 768, "num_heads": 12, "patch_size": 16, "n_storage_tokens": 4,
    }.items():
        if getattr(backbone, name, None) != expected:
            raise ValueError(f"Local DINOv3 backbone {name} must be {expected}")
    if len(getattr(backbone, "blocks", [])) != 12:
        raise ValueError("Local DINOv3 backbone must have 12 transformer blocks")
    rope = getattr(backbone, "rope_embed", None)
    for name, expected in {
        "base": 100.0, "D_head": 64, "min_period": None, "max_period": None,
        "normalize_coords": "separate", "shift_coords": None,
        "jitter_coords": None, "rescale_coords": 2.0, "dtype": torch.float32,
    }.items():
        if rope is None or not hasattr(rope, name) or getattr(rope, name) != expected:
            raise ValueError(f"Local DINOv3 RoPE {name} must be {expected!r}")
    norms = [getattr(backbone, "norm", None)]
    for block in backbone.blocks:
        norms.extend([getattr(block, "norm1", None), getattr(block, "norm2", None)])
        attention = getattr(block, "attn", None)
        if attention is None or getattr(attention, "num_heads", None) != 12:
            raise ValueError("Every local DINOv3 attention block must use 12 heads")
        if getattr(getattr(attention, "attn_drop", None), "p", None) != 0.0:
            raise ValueError("Local DINOv3 attention dropout must be zero")
        mlp = getattr(block, "mlp", None)
        act = getattr(mlp, "act", None)
        if not isinstance(act, nn.GELU) or act.approximate != "none":
            raise ValueError("Local DINOv3 ViT-B MLP must use exact GELU")
    if any(not isinstance(norm, nn.LayerNorm) or norm.eps != 1e-5 for norm in norms):
        raise ValueError("Local DINOv3 LayerNorm epsilon must be 1e-5")


def _convert_hf_state(
    backbone: nn.Module, source: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], list[str], list[str]]:
    old_layers = any(key.startswith("layer.") for key in source)
    new_layers = any(key.startswith("model.layer.") for key in source)
    if old_layers == new_layers:
        raise ValueError("HF DINOv3 requires exactly one layer prefix: layer.N or model.layer.N; mixed layouts are invalid")
    prefix = "model.layer" if new_layers else "layer"
    expected_keys = {
        "embeddings.cls_token", "embeddings.mask_token", "embeddings.register_tokens",
        "embeddings.patch_embeddings.weight", "embeddings.patch_embeddings.bias",
        "norm.weight", "norm.bias",
    }
    direct_suffixes = {
        "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
        "attention.o_proj.weight": "attn.proj.weight", "attention.o_proj.bias": "attn.proj.bias",
        "mlp.up_proj.weight": "mlp.fc1.weight", "mlp.up_proj.bias": "mlp.fc1.bias",
        "mlp.down_proj.weight": "mlp.fc2.weight", "mlp.down_proj.bias": "mlp.fc2.bias",
        "layer_scale1.lambda1": "ls1.gamma", "layer_scale2.lambda1": "ls2.gamma",
    }
    for index in range(12):
        expected_keys.update(f"{prefix}.{index}.{suffix}" for suffix in direct_suffixes)
        expected_keys.update(f"{prefix}.{index}.attention.{part}_proj.weight" for part in ("q", "k", "v"))
        expected_keys.update(f"{prefix}.{index}.attention.{part}_proj.bias" for part in ("q", "v"))
    missing = sorted(expected_keys - set(source))
    unexpected = sorted(set(source) - expected_keys)
    if missing or unexpected:
        raise RuntimeError(f"HF DINOv3 strict tensor coverage mismatch; missing={missing}; unexpected={unexpected}")
    mask_token = source["embeddings.mask_token"]
    if tuple(mask_token.shape) != (1, 1, 768):
        raise RuntimeError(f"HF DINOv3 embeddings.mask_token must have shape (1, 1, 768), got {tuple(mask_token.shape)}")
    result = {
        "cls_token": source["embeddings.cls_token"],
        "mask_token": mask_token.squeeze(1),
        "storage_tokens": source["embeddings.register_tokens"],
        "patch_embed.proj.weight": source["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": source["embeddings.patch_embeddings.bias"],
        "norm.weight": source["norm.weight"],
        "norm.bias": source["norm.bias"],
    }
    # Derive the same FP32 periods as official RopePositionEmbedding._init_weights.
    # Never reuse the destination's initialized/random tensor contents.
    result["rope_embed.periods"] = 100.0 ** (2 * torch.arange(16, dtype=torch.float32) / 32)
    buffers = ["rope_embed.periods"]
    masked_biases = []
    for index in range(12):
        src = f"{prefix}.{index}"
        dst = f"blocks.{index}"
        for suffix, target_suffix in direct_suffixes.items():
            result[f"{dst}.{target_suffix}"] = source[f"{src}.{suffix}"]
        weights = [source[f"{src}.attention.{part}_proj.weight"] for part in ("q", "k", "v")]
        biases = [source[f"{src}.attention.{part}_proj.bias"] for part in ("q", "v")]
        if any(tuple(weight.shape) != (768, 768) for weight in weights):
            raise RuntimeError(f"HF DINOv3 Q/K/V weight shape must be (768, 768) in block {index}")
        if any(tuple(bias.shape) != (768,) for bias in biases):
            raise RuntimeError(f"HF DINOv3 Q/V bias shape must be (768,) in block {index}")
        if len({tensor.dtype for tensor in weights + biases}) != 1:
            raise ValueError(f"HF DINOv3 Q/K/V weights and biases must have one dtype in block {index}")
        result[f"{dst}.attn.qkv.weight"] = torch.cat(weights, dim=0)
        q_bias, v_bias = biases
        result[f"{dst}.attn.qkv.bias"] = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias), dim=0)
        mask_key = f"{dst}.attn.qkv.bias_mask"
        result[mask_key] = torch.cat((torch.ones_like(q_bias), torch.zeros_like(q_bias), torch.ones_like(q_bias)))
        buffers.append(mask_key)
        masked_biases.append(f"{dst}.attn.qkv.bias[768:1536]")
    return result, buffers, masked_biases


def load_dinov3_weights(backbone: nn.Module, path: str | Path, model_variant: str) -> dict[str, Any]:
    """Validate and strictly load one existing local checkpoint; never download.

    Official raw ``.pth`` state dictionaries retain their native tensor names.
    A ``.safetensors`` file may use native names or the HF ViT-B/16 layout; the
    latter requires the original adjacent ``config.json``. All source tensors
    must be finite, every learned destination tensor must be covered exactly,
    and all shapes are validated before the destination is modified.
    """
    # HF cache snapshots can contain symlinks into a shared blobs directory.
    # The config belongs to the requested snapshot, not the symlink target.
    weights_path = Path(path).expanduser().absolute()
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    is_safe = weights_path.suffix.lower() == ".safetensors"
    if is_safe:
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError("Loading DINOv3 .safetensors requires: python -m pip install safetensors") from error
        source = load_file(str(weights_path), device="cpu")
    else:
        source = torch.load(weights_path, map_location="cpu", weights_only=True)
    _check_values(source, "DINOv3 checkpoint")
    is_hf = is_safe and any(
        key.startswith(("embeddings.", "layer.", "model.layer.")) for key in source
    )
    metadata: dict[str, Any] = {
        "source_path": str(weights_path),
        "source_format": "safetensors" if is_safe else "pytorch",
        "source_layout": "huggingface" if is_hf else "official",
        "source_tensors": len(source),
        "loaded_tensors": 0,
        "reconstructed_buffers": [],
        "zeroed_masked_key_biases": [],
        "config_path": None,
        "config_sha256": None,
        "conversion_version": CONVERSION_VERSION if is_hf else None,
    }
    state = source
    if is_hf:
        config_path = weights_path.with_name("config.json")
        _, digest = _validate_hf_config(config_path, model_variant)
        _validate_hf_backbone(backbone)
        state, buffers, biases = _convert_hf_state(backbone, source)
        metadata.update(config_path=str(config_path), config_sha256=digest,
                        reconstructed_buffers=buffers, zeroed_masked_key_biases=biases)
    _validate_exact_state(backbone, state)
    # This remains mandatory even after our coverage/shape checks.
    backbone.load_state_dict(state, strict=True)
    metadata["loaded_tensors"] = len(state)
    return metadata
