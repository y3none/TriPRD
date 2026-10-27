"""Mechanism and strict generic-backbone loading tests for correction models."""

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from models.triprd_correction import (
    COALITIONS, CorrectionStudent, CorrectionTeacher, GenericFeatureEncoder,
    normalize_auxiliary, relation_valid_mask, shuffle_correction,
)


class _FakeOfficialDino(nn.Module):
    """Exercises the documented local hub contract without downloading weights."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.patch = nn.Conv2d(3, 8, 16, stride=16)
        self.seen = None

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        self.seen = (tuple(x.shape), tuple(n), reshape, norm)
        value = self.patch(x)
        return tuple(value * (i + 1) for i in range(4))


class CorrectionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(29)
        self.rgb = torch.randn(2, 3, 33, 41)
        self.thermal = torch.rand(2, 3, 33, 41)
        self.depth = torch.rand(2, 1, 33, 41) * 0.03 + 0.01

    @staticmethod
    def _student(**kwargs):
        return CorrectionStudent(model_variant="tiny", pretrained=False, base_channels=8, **kwargs)

    @staticmethod
    def _teacher(**kwargs):
        return CorrectionTeacher(model_variant="tiny", pretrained=False, base_channels=8, **kwargs)

    def test_student_two_scale_contract_bounds_and_rgb_only_inference(self):
        model = self._student().eval()
        output = model(self.rgb)
        self.assertEqual(output["logits"].shape, (2, 1, 33, 41))
        for index, (height, width) in enumerate(((5, 6), (3, 3))):
            expected = (2, 1, 9, height, width)
            for key in ("relation_logits", "base_relation_logits", "corrections", "relation_probs"):
                self.assertEqual(output[key][index].shape, expected)
                self.assertTrue(torch.isfinite(output[key][index]).all())
            valid = output["relation_valid_masks"][index]
            self.assertEqual(valid.shape, (1, 1, 9, height, width))
            self.assertFalse(valid[:, :, 4].any())
            base = output["base_relation_logits"][index].sigmoid()
            residual = output["corrections"][index]
            self.assertTrue((residual >= -1).all())
            self.assertTrue((residual <= 1).all())
            torch.testing.assert_close((base + residual).clamp(1e-6, 1 - 1e-6), output["relation_probs"][index])
            self.assertFalse(residual.masked_select(~valid.expand_as(residual)).any())
        parameters = inspect.signature(model.forward).parameters
        self.assertNotIn("thermal", parameters)
        self.assertNotIn("depth", parameters)
        with self.assertRaises(TypeError):
            model(self.rgb, thermal=self.thermal)

    def test_cod_backward_reaches_correction_and_adaptation_but_frozen_encoder(self):
        model = self._student().train()
        output = model(self.rgb, task="cod")
        loss = F.binary_cross_entropy_with_logits(output["logits"], torch.rand_like(output["logits"]))
        loss.backward()
        for head in model.correction_heads:
            self.assertIsNotNone(head.edge[-1].weight.grad)
            self.assertGreater(head.edge[-1].weight.grad.abs().sum().item(), 0)
        for head in model.base_heads:
            self.assertGreater(head.edge[-1].weight.grad.abs().sum().item(), 0)
        alignment = tuple(model.alignment_parameters())
        self.assertTrue(alignment)
        self.assertTrue(all(parameter.grad is not None for parameter in alignment))
        self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in alignment), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))
        self.assertFalse(model.encoder.backbone.training)
        self.assertTrue(model.adapters.training)
        self.assertTrue(all(p.grad is None for p in model.task_heads["sod"].parameters()))

    def test_initial_correction_is_zero_and_preserves_rgb_base(self):
        model = self._student().eval()
        normal = model(self.rgb)
        disabled = model(self.rgb, intervention="no_correction")
        torch.testing.assert_close(normal["logits"], disabled["logits"], rtol=0, atol=0)
        for base, residual, probability in zip(normal["base_relation_logits"],
                                                normal["corrections"], normal["relation_probs"]):
            self.assertEqual(residual.count_nonzero().item(), 0)
            torch.testing.assert_close(probability, base.sigmoid(), rtol=0, atol=0)
        # Base/teacher predictors keep their ordinary initialization.
        for head in (*model.base_heads, *self._teacher().relation_heads):
            self.assertGreater(head.edge[-1].weight.count_nonzero().item(), 0)

    def test_zero_correction_projection_learns_then_opens_upstream_gradients(self):
        from models.triprd_correction_loss import source_loss
        model = self._student().train()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        mask = torch.zeros(2, 1, 33, 41)
        mask[:, :, 8:24, 10:30] = 1
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            source_loss(model(self.rgb), mask)["total"].backward()
            for head in model.correction_heads:
                last = head.edge[-1]
                for parameter in (last.weight, last.bias):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                upstream = head.projection.weight.grad
                self.assertIsNotNone(upstream)
                self.assertTrue(torch.isfinite(upstream).all())
                if step == 0:
                    self.assertEqual(upstream.abs().sum().item(), 0)
                else:
                    self.assertGreater(upstream.abs().sum().item(), 0)
            optimizer.step()
        self.assertTrue(any(delta.count_nonzero() for delta in model(self.rgb)["corrections"]))

    def test_sod_and_cod_heads_are_distinct(self):
        model = self._student().eval()
        cod = model(self.rgb, task="cod")
        sod = model(self.rgb, task="sod")
        self.assertGreater((cod["logits"] - sod["logits"]).abs().max().item(), 0)
        for left, right in zip(cod["relation_logits"], sod["relation_logits"]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        sod["logits"].mean().backward()
        self.assertIsNotNone(model.task_heads["sod"].mask.weight.grad)
        self.assertIsNone(model.task_heads["cod"].mask.weight.grad)

    def test_correction_interventions_change_propagation_not_prediction(self):
        model = self._student().eval()
        # Simulate a learned correction; the initial correction is exactly zero.
        with torch.no_grad():
            for head in model.correction_heads:
                nn.init.normal_(head.edge[-1].weight, std=0.1)
        normal = model(self.rgb)
        for intervention in ("no_correction", "shuffle_correction", "identity"):
            altered = model(self.rgb, intervention=intervention)
            self.assertGreater((normal["logits"] - altered["logits"]).abs().max().item(), 1e-8)
            for key in ("relation_logits", "corrections"):
                for expected, actual in zip(normal[key], altered[key]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        no_correction = model(self.rgb, intervention="no_correction")
        for base, actual in zip(normal["base_relation_logits"], no_correction["propagation_probs"]):
            torch.testing.assert_close(base.sigmoid(), actual)

    def test_shuffle_preserves_valid_signed_histograms(self):
        valid = relation_valid_mask(4, 5, 3, device=torch.device("cpu"))
        residual = torch.randn(2, 1, 9, 4, 5).masked_fill(~valid, 0)
        shuffled = shuffle_correction(residual, valid)
        self.assertGreater((shuffled - residual).abs().sum().item(), 0)
        for batch in range(2):
            for offset in range(9):
                keep = valid[0, 0, offset]
                torch.testing.assert_close(residual[batch, 0, offset][keep].sort().values,
                                           shuffled[batch, 0, offset][keep].sort().values,
                                           rtol=0, atol=0)

    def test_teacher_r_is_exactly_auxiliary_independent_and_missing_is_explicit(self):
        teacher = self._teacher().train()
        baseline = teacher(self.rgb, coalition="R")
        changed = teacher(self.rgb, torch.full((1,), float("nan")), torch.full((1,), float("inf")), coalition="R")
        torch.testing.assert_close(baseline["logits"], changed["logits"], rtol=0, atol=0)
        baseline["logits"].square().mean().backward()
        for token in teacher.missing_tokens.values():
            self.assertIsNotNone(token.grad)
            self.assertGreater(token.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in teacher.thermal_adapters.parameters()))
        with self.assertRaisesRegex(ValueError, "thermal"):
            teacher(self.rgb, coalition="RT")
        with self.assertRaisesRegex(ValueError, "depth"):
            teacher(self.rgb, coalition="RD")

    def test_forward_coalitions_encodes_each_modality_once_and_matches_separate_calls(self):
        teacher = self._teacher().eval()
        calls = []
        handle = teacher.encoder.register_forward_hook(lambda *args: calls.append(1))
        outputs = teacher.forward_coalitions(self.rgb, self.thermal, self.depth)
        handle.remove()
        self.assertEqual(len(calls), 3)
        self.assertEqual(tuple(outputs), COALITIONS)
        for coalition in COALITIONS:
            separate = teacher(self.rgb, self.thermal, self.depth, coalition=coalition)
            torch.testing.assert_close(outputs[coalition]["logits"], separate["logits"], rtol=0, atol=0)
            for actual, rgb_base in zip(outputs[coalition]["base_relation_logits"], outputs["R"]["relation_logits"]):
                torch.testing.assert_close(actual, rgb_base, rtol=0, atol=0)
        for coalition in ("RT", "RD", "RTD"):
            self.assertGreater((outputs[coalition]["logits"] - outputs["R"]["logits"]).abs().max().item(), 0)
        sum(output["logits"].square().mean() for output in outputs.values()).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in teacher.thermal_adapters.parameters() if p.grad is not None), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in teacher.depth_adapters.parameters() if p.grad is not None), 0)

    def test_omitted_modalities_do_not_change_rt_rd(self):
        teacher = self._teacher().eval()
        first = teacher(self.rgb, self.thermal, self.depth, coalition="RT")
        second = teacher(self.rgb, self.thermal, 1 - self.depth, coalition="RT")
        torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)
        first = teacher(self.rgb, self.thermal, self.depth, coalition="RD")
        second = teacher(self.rgb, 1 - self.thermal, self.depth, coalition="RD")
        torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)

    def test_thermal_color_depth_scale_and_invalid_observations(self):
        thermal = normalize_auxiliary(self.thermal, depth=False)
        grayscale = normalize_auxiliary(self.thermal.mean(1, keepdim=True), depth=False)
        self.assertGreater((thermal - grayscale).abs().sum().item(), 0)
        first = normalize_auxiliary(self.depth, depth=True)
        second = normalize_auxiliary(self.depth * 3 + 0.1, depth=True)
        torch.testing.assert_close(first, second, atol=3e-6, rtol=2e-5)
        invalid = self.depth.clone()
        invalid[0] = float("nan")
        invalid[1, :, :5] = 0
        invalid[1, :, 5:10] = float("inf")
        self.assertTrue(torch.isfinite(normalize_auxiliary(invalid, depth=True)).all())
        teacher = self._teacher().eval()
        self.assertTrue(torch.isfinite(teacher(self.rgb, self.thermal, invalid)["logits"]).all())

    def test_freeze_survives_train_and_unfreeze_restores_gradients(self):
        model = self._student()
        model.train()
        self.assertFalse(model.encoder.backbone.training)
        before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        model(self.rgb)["logits"].square().mean().backward()
        optimizer.step()
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        model.unfreeze_backbone()
        self.assertTrue(model.encoder.backbone.training)
        model.zero_grad(set_to_none=True)
        model(self.rgb)["logits"].square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.encoder.parameters()))

    def test_formal_default_fails_fast_and_never_silently_uses_tiny(self):
        with patch("torch.hub.load") as loader:
            with self.assertRaisesRegex(ValueError, "dino_repo"):
                CorrectionStudent()
            with self.assertRaisesRegex(ValueError, "dino_repo"):
                CorrectionStudent(pretrained=False)
            with self.assertRaisesRegex(ValueError, "test-only"):
                CorrectionStudent(model_variant="tiny")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                CorrectionStudent(model_variant="unknown")
            loader.assert_not_called()

    def test_local_official_weights_strict_loading_and_nonmultiple_padding(self):
        with tempfile.TemporaryDirectory(prefix=".correction-model-", dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            (root / "hubconf.py").write_text("# test local hub marker", encoding="utf-8")
            pretrained = _FakeOfficialDino()
            torch.save(pretrained.state_dict(), root / "generic.pth")
            with patch("torch.hub.load", side_effect=lambda *args, **kwargs: _FakeOfficialDino()) as loader:
                encoder = GenericFeatureEncoder("dinov3_vitb16", 8, True, root, root / "generic.pth", True)
                features = encoder(self.rgb)
                loader.assert_called_once_with(str(root.resolve()), "dinov3_vitb16", source="local", pretrained=False)
                self.assertEqual(encoder.backbone.seen, ((2, 3, 48, 48), (2, 5, 8, 11), True, True))
                self.assertEqual(features[0].shape, (2, 8, 3, 3))
                for key, value in encoder.backbone.state_dict().items():
                    torch.testing.assert_close(value, pretrained.state_dict()[key], rtol=0, atol=0)
            torch.save({"cod_checkpoint": torch.ones(1)}, root / "wrong.pth")
            with patch("torch.hub.load", return_value=_FakeOfficialDino()):
                with self.assertRaises(RuntimeError):
                    GenericFeatureEncoder("dinov3_vitb16", 8, True, root, root / "wrong.pth", True)
            with patch("torch.hub.load") as loader:
                with self.assertRaises(FileNotFoundError):
                    GenericFeatureEncoder("dinov3_vitb16", 8, True, root, root / "missing.pth", True)
                loader.assert_not_called()

    def test_checkpoint_architecture_does_not_require_generic_weights(self):
        with tempfile.TemporaryDirectory(prefix=".correction-model-", dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            (root / "hubconf.py").write_text("# fixture", encoding="utf-8")
            with patch("torch.hub.load", return_value=_FakeOfficialDino()) as loader:
                model = CorrectionStudent(pretrained=False, dino_repo=root, base_channels=8)
                self.assertEqual(model(self.rgb)["logits"].shape, (2, 1, 33, 41))
                loader.assert_called_once_with(str(root.resolve()), "dinov3_vitb16", source="local", pretrained=False)

    def test_learned_correction_is_not_limited_by_student_base_headroom(self):
        model = self._student().eval()
        with torch.no_grad():
            for head in model.base_heads:
                head.edge[-1].weight.zero_()
                head.edge[-1].bias.fill_(8)
            for head in model.correction_heads:
                head.edge[-1].weight.zero_()
                head.edge[-1].bias.fill_(0.7)
        output = model(self.rgb)
        valid = output["relation_valid_masks"][0].expand_as(output["corrections"][0])
        self.assertGreater(output["corrections"][0][valid].min().item(), 0.5)
        self.assertLess(output["clipped_corrections"][0][valid].max().item(), 0.001)

    def test_seed_recreates_identical_student_across_experiment_groups(self):
        torch.manual_seed(901)
        baseline = self._student()
        torch.manual_seed(901)
        corrected = self._student()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, corrected.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
