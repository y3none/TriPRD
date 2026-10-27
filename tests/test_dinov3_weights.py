"""Real-file tests of the strict HF-to-official DINOv3 weight bridge.

The destination fixture keeps the genuine ViT-B Q/K/V dimensions, token
dimensions, twelve blocks and loader-visible architecture attributes. Unused
patch/output/MLP projections are deliberately small so rejection tests do not
repeatedly write a complete 350 MB checkpoint. Full official-model numerical
equivalence is checked separately by ``check_dinov3_hf_equivalence.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from models.dinov3_weights import load_dinov3_weights


class _MaskedLinear(nn.Linear):
    def __init__(self):
        super().__init__(768, 2304, dtype=torch.float16)
        self.register_buffer("bias_mask", torch.cat((torch.ones(768), torch.zeros(768), torch.ones(768))))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-5)
        self.norm2 = nn.LayerNorm(768, eps=1e-5)
        self.attn = nn.Module()
        self.attn.num_heads = 12
        self.attn.attn_drop = nn.Dropout(0)
        self.attn.qkv = _MaskedLinear()
        self.attn.proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.act = nn.GELU()
        self.mlp.fc1 = nn.Linear(4, 8)
        self.mlp.fc2 = nn.Linear(8, 4)
        self.ls1 = nn.Module()
        self.ls1.gamma = nn.Parameter(torch.randn(768))
        self.ls2 = nn.Module()
        self.ls2.gamma = nn.Parameter(torch.randn(768))


class _LoaderDestination(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim, self.num_heads, self.patch_size, self.n_storage_tokens = 768, 12, 16, 4
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.mask_token = nn.Parameter(torch.randn(1, 768))
        self.storage_tokens = nn.Parameter(torch.randn(1, 4, 768))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, 1)
        self.norm = nn.LayerNorm(768, eps=1e-5)
        self.blocks = nn.ModuleList([_Block() for _ in range(12)])
        self.rope_embed = nn.Module()
        for name, value in {
            "base": 100.0, "D_head": 64, "min_period": None, "max_period": None,
            "normalize_coords": "separate", "shift_coords": None,
            "jitter_coords": None, "rescale_coords": 2.0, "dtype": torch.float32,
        }.items():
            setattr(self.rope_embed, name, value)
        self.rope_embed.register_buffer("periods", torch.logspace(0, 15 / 16, 16, base=100.0))


def _hf_export(reference: _LoaderDestination) -> dict[str, torch.Tensor]:
    """An independent explicit HF fixture, without calling production mapping."""
    state = {
        "embeddings.cls_token": reference.cls_token.detach().clone(),
        "embeddings.mask_token": reference.mask_token.detach().unsqueeze(1).clone(),
        "embeddings.register_tokens": reference.storage_tokens.detach().clone(),
        "embeddings.patch_embeddings.weight": reference.patch_embed.proj.weight.detach().clone(),
        "embeddings.patch_embeddings.bias": reference.patch_embed.proj.bias.detach().clone(),
        "norm.weight": reference.norm.weight.detach().clone(),
        "norm.bias": reference.norm.bias.detach().clone(),
    }
    for index, block in enumerate(reference.blocks):
        prefix = f"layer.{index}."
        for name, module in (("norm1", block.norm1), ("norm2", block.norm2),
                             ("attention.o_proj", block.attn.proj),
                             ("mlp.up_proj", block.mlp.fc1), ("mlp.down_proj", block.mlp.fc2)):
            state[prefix + name + ".weight"] = module.weight.detach().clone()
            state[prefix + name + ".bias"] = module.bias.detach().clone()
        state[prefix + "layer_scale1.lambda1"] = block.ls1.gamma.detach().clone()
        state[prefix + "layer_scale2.lambda1"] = block.ls2.gamma.detach().clone()
        for part, weights in zip(("q", "k", "v"), block.attn.qkv.weight.detach().chunk(3, dim=0)):
            state[prefix + f"attention.{part}_proj.weight"] = weights.clone()
        state[prefix + "attention.q_proj.bias"] = block.attn.qkv.bias.detach()[:768].clone()
        state[prefix + "attention.v_proj.bias"] = block.attn.qkv.bias.detach()[1536:].clone()
    return state


class Dinov3WeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(730)
        cls.directory = tempfile.TemporaryDirectory(prefix=".dinov3-weights-", dir=Path(__file__).resolve().parents[1])
        cls.root = Path(cls.directory.name)
        cls.reference = _LoaderDestination()
        cls.hf_state = _hf_export(cls.reference)
        cls.weights = cls.root / "model.safetensors"
        save_file(cls.hf_state, str(cls.weights))
        cls.config = {
            "architectures": ["DINOv3ViTModel"], "model_type": "dinov3_vit",
            "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
            "num_attention_heads": 12, "patch_size": 16, "num_register_tokens": 4,
            "layer_norm_eps": 1e-5, "rope_theta": 100.0, "key_bias": False,
        }

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.target = _LoaderDestination()
        self._write_config(self.config)

    def _write_config(self, config):
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def _load(self, path=None, variant="dinov3_vitb16"):
        return load_dinov3_weights(self.target, path or self.weights, variant)

    def _bad_state(self, update=None, remove=None):
        state = dict(self.hf_state)
        if update:
            state.update(update)
        if remove:
            for key in remove:
                state.pop(key)
        path = self.root / "invalid.safetensors"
        save_file(state, str(path))
        return path

    def test_hf_load_covers_every_parameter_and_reconstructs_only_fixed_values(self):
        # Neither deterministic buffer may be copied from destination contents.
        self.target.rope_embed.periods.fill_(float("nan"))
        for block in self.target.blocks:
            block.attn.qkv.bias_mask.fill_(float("nan"))
        metadata = self._load()
        self.assertEqual(metadata["source_layout"], "huggingface")
        self.assertEqual(metadata["source_format"], "safetensors")
        self.assertEqual(metadata["source_tensors"], len(self.hf_state))
        self.assertEqual(metadata["loaded_tensors"], len(self.target.state_dict()))
        self.assertEqual(len(metadata["reconstructed_buffers"]), 13)
        self.assertEqual(len(metadata["zeroed_masked_key_biases"]), 12)
        self.assertEqual(metadata["config_sha256"], hashlib.sha256((self.root / "config.json").read_bytes()).hexdigest())
        self.assertEqual(tuple(self.target.mask_token.shape), (1, 768))
        expected = dict(self.reference.named_parameters())
        for name, parameter in self.target.named_parameters():
            value = expected[name].detach().clone()
            if name.endswith("attn.qkv.bias"):
                value[768:1536] = 0
            torch.testing.assert_close(parameter, value, rtol=0, atol=0, msg=name)
        torch.testing.assert_close(self.target.rope_embed.periods,
                                   torch.tensor([100 ** (i / 16) for i in range(16)]),
                                   rtol=1e-6, atol=0)
        for block in self.target.blocks:
            torch.testing.assert_close(block.attn.qkv.bias_mask,
                                       torch.cat((torch.ones(768), torch.zeros(768), torch.ones(768))),
                                       rtol=0, atol=0)

    def test_fused_qkv_matches_independent_separate_projection_attention(self):
        self._load()
        x = torch.randn(2, 5, 768)
        block = self.target.blocks[0]
        q, k, v = [F.linear(x, self.hf_state[f"layer.0.attention.{part}_proj.weight"].float(),
                            self.hf_state[f"layer.0.attention.{part}_proj.bias"].float() if part != "k" else None)
                   .reshape(2, 5, 12, 64).transpose(1, 2) for part in ("q", "k", "v")]
        expected = ((q @ k.transpose(-1, -2)) / 8).softmax(-1) @ v
        fused = F.linear(x, block.attn.qkv.weight.float(),
                         (block.attn.qkv.bias * block.attn.qkv.bias_mask).float())
        loaded_q, loaded_k, loaded_v = fused.reshape(2, 5, 3, 12, 64).permute(2, 0, 3, 1, 4)
        actual = F.scaled_dot_product_attention(loaded_q, loaded_k, loaded_v)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_current_model_layer_prefix_is_supported(self):
        state = {("model." + key if key.startswith("layer.") else key): value
                 for key, value in self.hf_state.items()}
        path = self.root / "current.safetensors"
        save_file(state, str(path))
        self._load(path)
        torch.testing.assert_close(self.target.blocks[11].attn.qkv.weight,
                                   self.reference.blocks[11].attn.qkv.weight, rtol=0, atol=0)

    def test_mixed_layer_prefixes_are_rejected(self):
        key = "layer.0.norm1.weight"
        path = self._bad_state({"model." + key: self.hf_state[key]}, remove=[key])
        with self.assertRaisesRegex(ValueError, "prefix|mixed"):
            self._load(path)

    def test_missing_learned_tensor_is_not_filled_from_initialization(self):
        path = self._bad_state(remove=["layer.11.mlp.down_proj.bias"])
        with patch.object(self.target, "load_state_dict", wraps=self.target.load_state_dict) as load:
            with self.assertRaisesRegex(RuntimeError, "missing"):
                self._load(path)
            load.assert_not_called()

    def test_unknown_tensor_is_not_silently_dropped(self):
        path = self._bad_state({"classifier.weight": torch.ones(2, 2)})
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            self._load(path)

    def test_even_zero_key_bias_is_rejected_as_an_unsupported_hf_layout(self):
        path = self._bad_state({"layer.0.attention.k_proj.bias": torch.zeros(768, dtype=torch.float16)})
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            self._load(path)

    def test_wrong_qkv_shape_is_rejected(self):
        path = self._bad_state({"layer.0.attention.q_proj.weight": torch.zeros(767, 768, dtype=torch.float16)})
        with self.assertRaisesRegex(RuntimeError, "shape"):
            self._load(path)

    def test_wrong_non_qkv_shape_is_rejected_before_destination_changes(self):
        path = self._bad_state({"layer.0.norm1.weight": torch.zeros(767)})
        with patch.object(self.target, "load_state_dict", wraps=self.target.load_state_dict) as load:
            with self.assertRaisesRegex(RuntimeError, "shape"):
                self._load(path)
            load.assert_not_called()

    def test_wrong_mask_token_rank_is_rejected(self):
        path = self._bad_state({"embeddings.mask_token": torch.zeros(1, 768)})
        with self.assertRaisesRegex(RuntimeError, "mask_token.*shape"):
            self._load(path)

    def test_nonfinite_learned_tensor_is_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                path = self._bad_state({"norm.bias": torch.full((768,), value)})
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    self._load(path)

    def test_integer_learned_tensor_is_rejected(self):
        path = self._bad_state({"norm.bias": torch.zeros(768, dtype=torch.int64)})
        with self.assertRaisesRegex(ValueError, "floating point"):
            self._load(path)

    def test_mixed_qkv_dtype_is_rejected(self):
        key = "layer.0.attention.q_proj.bias"
        path = self._bad_state({key: self.hf_state[key].float()})
        with self.assertRaisesRegex(ValueError, "dtype"):
            self._load(path)

    def test_hf_requires_adjacent_original_config(self):
        (self.root / "config.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "config.json"):
            self._load()

    def test_config_rejects_shape_invisible_and_architectural_differences(self):
        for key, value in {
            "hidden_size": 384, "intermediate_size": 2048, "num_hidden_layers": 24,
            "num_attention_heads": 6, "patch_size": 14, "num_register_tokens": 0,
            "layer_norm_eps": 1e-6, "rope_theta": 10000.0, "key_bias": True,
            "use_gated_mlp": True, "hidden_act": "gelu_new", "num_channels": 1,
            "attention_dropout": 0.1, "query_bias": False, "value_bias": False,
            "proj_bias": False, "mlp_bias": False, "drop_path_rate": 0.1,
            "pos_embed_shift": 0.1, "pos_embed_jitter": 1.1, "pos_embed_rescale": 1.0,
            "pos_embed_rope_normalize_coords": "max", "rope_scaling": {"factor": 2},
            "architectures": ["DINOv3ViTForImageClassification"],
        }.items():
            with self.subTest(key=key):
                self._write_config({**self.config, key: value})
                with self.assertRaisesRegex(ValueError, key + "|base DINOv3"):
                    self._load()

    def test_missing_required_config_field_is_rejected(self):
        config = dict(self.config)
        del config["num_attention_heads"]
        self._write_config(config)
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            self._load()

    def test_config_boolean_is_not_accepted_as_a_numeric_architecture_value(self):
        self._write_config({**self.config, "hidden_size": True})
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            self._load()

    def test_duplicate_config_key_is_rejected(self):
        serialized = json.dumps(self.config)
        (self.root / "config.json").write_text(serialized[:-1] + ', "hidden_size": 768}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self._load()

    def test_hf_conversion_requires_the_requested_b16_variant(self):
        with self.assertRaisesRegex(ValueError, "only dinov3_vitb16"):
            self._load(variant="dinov3_vits16")

    def test_local_backbone_norm_and_rope_conventions_must_match(self):
        self.target.norm.eps = 1e-6
        with self.assertRaisesRegex(ValueError, "epsilon"):
            self._load()
        self.target.norm.eps = 1e-5
        self.target.rope_embed.normalize_coords = "max"
        with self.assertRaisesRegex(ValueError, "normalize_coords"):
            self._load()

    def test_native_safetensors_and_pth_remain_strict_without_config(self):
        (self.root / "config.json").unlink()
        source = nn.Linear(3, 2)
        for suffix in (".safetensors", ".pth"):
            with self.subTest(suffix=suffix):
                path = self.root / ("native" + suffix)
                if suffix == ".safetensors":
                    save_file(source.state_dict(), str(path))
                else:
                    torch.save(source.state_dict(), path)
                destination = nn.Linear(3, 2)
                metadata = load_dinov3_weights(destination, path, "dinov3_vitb16")
                self.assertEqual(metadata["source_layout"], "official")
                self.assertIsNone(metadata["config_sha256"])
                for name, value in destination.state_dict().items():
                    torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)
                with self.assertRaisesRegex(RuntimeError, "shape"):
                    load_dinov3_weights(nn.Linear(3, 3), path, "dinov3_vitb16")

    def test_native_state_cannot_omit_buffers_or_wrap_an_entire_checkpoint(self):
        destination = nn.Linear(3, 2)
        destination.register_buffer("required_buffer", torch.ones(2))
        path = self.root / "native-incomplete.safetensors"
        save_file({"weight": destination.weight, "bias": destination.bias}, str(path))
        with self.assertRaisesRegex(RuntimeError, "missing.*required_buffer"):
            load_dinov3_weights(destination, path, "dinov3_vitb16")
        path = self.root / "training-checkpoint.pth"
        torch.save({"model": destination.state_dict(), "epoch": 2}, path)
        with self.assertRaisesRegex(ValueError, "only string keys and tensors"):
            load_dinov3_weights(destination, path, "dinov3_vitb16")


if __name__ == "__main__":
    unittest.main()
