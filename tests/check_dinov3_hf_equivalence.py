"""Optional CPU numerical audit of HF-to-official DINOv3 ViT-B/16 loading.

Run from the project root:
    python tests/check_dinov3_hf_equivalence.py --dino-repo ./dinov3

This intentionally is not a pytest-discovered unit test: it needs the official
local checkout, ~1.5 GB RAM and ~350 MB temporary disk space. It downloads
nothing and does not require Transformers or pretrained weights. A real full
ViT-B/16 with random, nontrivial learned parameters is exported in each HF
layout and loaded via the production loader. Its outputs are checked against
an independent functional implementation of HF's separate-QKV forward pass.

Reference implementation (both use separate Q/K/V and nonpersistent RoPE):
https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py
https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py
This validates conversion mechanics, not the provenance/accuracy of a user's
downloaded checkpoint, nor bitwise equivalence under CUDA mixed precision.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dinov3_weights import load_dinov3_weights  # noqa: E402


CONFIG = {
    "architectures": ["DINOv3ViTModel"],
    "model_type": "dinov3_vit",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "patch_size": 16,
    "num_register_tokens": 4,
    "num_channels": 3,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "rope_theta": 100.0,
    "layerscale_value": 1.0,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "use_gated_mlp": False,
    "pos_embed_rescale": 2.0,
    "pos_embed_shift": None,
    "pos_embed_jitter": None,
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
}
STAGES = (2, 5, 8, 11)


def export_hf(backbone: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Independent export, following the published original-to-HF conversion."""
    source = backbone.state_dict()
    mapping = {
        "cls_token": "embeddings.cls_token",
        "mask_token": "embeddings.mask_token",
        "storage_tokens": "embeddings.register_tokens",
        "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    suffixes = {
        "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
        "attn.proj.weight": "attention.o_proj.weight",
        "attn.proj.bias": "attention.o_proj.bias",
        "ls1.gamma": "layer_scale1.lambda1", "ls2.gamma": "layer_scale2.lambda1",
        "mlp.fc1.weight": "mlp.up_proj.weight", "mlp.fc1.bias": "mlp.up_proj.bias",
        "mlp.fc2.weight": "mlp.down_proj.weight", "mlp.fc2.bias": "mlp.down_proj.bias",
    }
    for index in range(12):
        mapping.update({f"blocks.{index}.{k}": f"layer.{index}.{v}" for k, v in suffixes.items()})
    result = {dst: source[src].detach().clone() for src, dst in mapping.items()}
    result["embeddings.mask_token"] = result["embeddings.mask_token"].unsqueeze(1)
    for index in range(12):
        weights = source[f"blocks.{index}.attn.qkv.weight"].chunk(3, 0)
        biases = source[f"blocks.{index}.attn.qkv.bias"].chunk(3, 0)
        for part, weight, bias in zip(("q", "k", "v"), weights, biases):
            result[f"layer.{index}.attention.{part}_proj.weight"] = weight.detach().clone()
            if part != "k":
                result[f"layer.{index}.attention.{part}_proj.bias"] = bias.detach().clone()
    return result


@torch.no_grad()
def hf_forward(
    pixels: torch.Tensor, state: dict[str, torch.Tensor], mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """HF math, independent of official modules and the production converter."""
    tokens = F.conv2d(pixels, state["embeddings.patch_embeddings.weight"],
                      state["embeddings.patch_embeddings.bias"], stride=16).flatten(2).transpose(1, 2)
    if mask is not None:
        tokens = torch.where(mask[..., None], state["embeddings.mask_token"], tokens)
    tokens = torch.cat((state["embeddings.cls_token"].expand(pixels.shape[0], -1, -1),
                        state["embeddings.register_tokens"].expand(pixels.shape[0], -1, -1), tokens), 1)
    height, width = pixels.shape[-2] // 16, pixels.shape[-1] // 16
    yy = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    xx = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    coords = torch.stack(torch.meshgrid(yy, xx, indexing="ij"), -1).flatten(0, 1) * 2 - 1
    inverse_frequency = 1 / (100.0 ** torch.arange(0, 1, 4 / 64, dtype=torch.float32))
    angles = (2 * math.pi * coords[:, :, None] * inverse_frequency).flatten(1).repeat(1, 2)
    cos, sin = angles.cos(), angles.sin()

    def rotate_patches(value: torch.Tensor) -> torch.Tensor:
        patch = value[:, :, 5:]
        half = patch.shape[-1] // 2
        rotated = torch.cat((-patch[..., half:], patch[..., :half]), -1)
        return torch.cat((value[:, :, :5], patch * cos + rotated * sin), 2)

    def layer_norm(value: torch.Tensor, name: str) -> torch.Tensor:
        return F.layer_norm(value, (768,), state[f"{name}.weight"], state[f"{name}.bias"], 1e-5)

    output = []
    for index in range(12):
        prefix = f"layer.{index}"
        normalized = layer_norm(tokens, f"{prefix}.norm1")
        qkv = []
        for part in ("q", "k", "v"):
            projected = F.linear(normalized, state[f"{prefix}.attention.{part}_proj.weight"],
                                 state.get(f"{prefix}.attention.{part}_proj.bias"))
            qkv.append(projected.reshape(pixels.shape[0], tokens.shape[1], 12, 64).transpose(1, 2))
        query, key, value = rotate_patches(qkv[0]), rotate_patches(qkv[1]), qkv[2]
        # Eager HF attention deliberately differs from official fused SDPA.
        attention = ((query @ key.transpose(-1, -2)) / math.sqrt(64)).softmax(-1)
        attention = (attention @ value).transpose(1, 2).reshape_as(tokens)
        attention = F.linear(attention, state[f"{prefix}.attention.o_proj.weight"],
                             state[f"{prefix}.attention.o_proj.bias"])
        tokens = tokens + attention * state[f"{prefix}.layer_scale1.lambda1"]
        hidden = layer_norm(tokens, f"{prefix}.norm2")
        hidden = F.gelu(F.linear(hidden, state[f"{prefix}.mlp.up_proj.weight"],
                                state[f"{prefix}.mlp.up_proj.bias"]), approximate="none")
        hidden = F.linear(hidden, state[f"{prefix}.mlp.down_proj.weight"],
                          state[f"{prefix}.mlp.down_proj.bias"])
        tokens = tokens + hidden * state[f"{prefix}.layer_scale2.lambda1"]
        if index in STAGES:
            output.append(layer_norm(tokens, "norm"))
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-cli", action="store_true", help="Also smoke-test tools/check_dinov3_weights.py")
    parser.add_argument("--report", type=Path, help="Write the small JSON verification report to this path")
    args = parser.parse_args()
    repo = args.dino_repo.expanduser().resolve()
    torch.set_num_threads(args.threads)
    torch.manual_seed(71283)
    # Call the same official factory without importing unrelated hubconf
    # segmentation/depth models and their optional dependencies.
    sys.path.insert(0, str(repo))
    from dinov3.hub.backbones import dinov3_vitb16
    backbone = dinov3_vitb16(pretrained=False).eval()
    with torch.no_grad():
        # Nontrivial scales/biases prevent small initialization from hiding a
        # wrong QKV order, dropped projection bias or absent transformer block.
        for name, parameter in backbone.named_parameters():
            if name.endswith(("ls1.gamma", "ls2.gamma")):
                parameter.uniform_(0.2, 0.7)
            elif name.endswith("bias"):
                parameter.uniform_(-0.1, 0.1)
            elif "norm" in name and name.endswith("weight"):
                parameter.uniform_(0.8, 1.2)
        backbone.mask_token.uniform_(-0.3, 0.3)
    state = export_hf(backbone)
    cases = []
    for height, width in ((32, 48), (48, 64)):
        pixels = torch.randn(2, 3, height, width)
        mask = torch.arange(height * width // 256)[None].expand(2, -1) % 3 == 0
        cases.append((pixels, mask, hf_forward(pixels, state), hf_forward(pixels, state, mask)))
    rows = []
    cli_result = None
    with tempfile.TemporaryDirectory(prefix="triprd-hf-equivalence-") as temporary:
        folder = Path(temporary)
        (folder / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        weights = folder / "model.safetensors"
        for layout in ("layer", "model.layer"):
            exported = {(f"model.{k}" if layout == "model.layer" and k.startswith("layer.") else k): v
                        for k, v in state.items()}
            save_file(exported, str(weights))
            with torch.no_grad():
                for parameter in backbone.parameters():
                    parameter.fill_(0.123)
                # A missing deterministic buffer must not survive loading.
                for buffer in backbone.buffers():
                    if buffer.is_floating_point():
                        buffer.fill_(math.nan)
            metadata = load_dinov3_weights(backbone, weights, "dinov3_vitb16")
            for pixels, mask, expected, expected_masked in cases:
                with torch.no_grad():
                    actual = backbone.get_intermediate_layers(
                        pixels, n=STAGES, norm=True, return_class_token=True, return_extra_tokens=True,
                    )
                    masked = backbone.forward_features(pixels, masks=mask)
                errors = []
                for stage, ((patches, cls, registers), reference) in enumerate(zip(actual, expected)):
                    sequence = torch.cat((cls[:, None], registers, patches), 1)
                    torch.testing.assert_close(sequence, reference, atol=3e-5, rtol=3e-5,
                                               msg=f"{layout}, shape={tuple(pixels.shape)}, block={STAGES[stage]}")
                    errors.append((sequence - reference).abs().max().item())
                actual_masked = torch.cat((masked["x_norm_clstoken"][:, None],
                                           masked["x_storage_tokens"], masked["x_norm_patchtokens"]), 1)
                torch.testing.assert_close(actual_masked, expected_masked[-1], atol=3e-5, rtol=3e-5)
                rows.append({"layout": layout, "input": list(pixels.shape), "stage_max_abs_errors": errors,
                             "masked_max_abs_error": (actual_masked - expected_masked[-1]).abs().max().item()})
            print(json.dumps({"layout": layout, "loader": metadata}, ensure_ascii=False))
        if args.check_cli:
            completed = subprocess.run([
                sys.executable, str(ROOT / "tools/check_dinov3_weights.py"),
                "--dino-repo", str(repo), "--dino-weights", str(weights),
                "--device", "cpu", "--image-size", "32",
            ], check=True, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            cli_result = json.loads(completed.stdout)
            print(completed.stdout)
    result = {"status": "passed", "reference": "independent HF functional FP32 eager attention",
              "torch_version": str(torch.__version__), "seed": 71283, "cpu_threads": args.threads,
              "production_cli_status": "passed" if args.check_cli else "not_run",
              "pretrained_accuracy_verified": False, "cases": rows}
    if cli_result is not None:
        result["production_cli"] = {key: cli_result[key] for key in (
            "status", "model_variant", "device", "torch_version", "cuda_version",
            "generic_weights_sha256", "layer_indices", "feature_shapes", "finite_features", "frozen",
        )}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
