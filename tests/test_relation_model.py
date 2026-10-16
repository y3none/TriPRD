"""Mechanism checks for explicit local relations and frozen-backbone transfer."""

import unittest

import torch
from torch import nn

from models.triprd_relation import (
    LocalRelationBranch, RelationLoss, RelationStudent, RelationTeacher,
    local_valid_mask, masked_local_distribution, normalized_relation_entropy,
)


class RelationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.rgb = torch.randn(2, 3, 24, 28)
        self.mask = (torch.rand(2, 1, 24, 28) > 0.65).float()

    @staticmethod
    def _student(**kwargs):
        return RelationStudent(base_channels=4, relation_dim=4, **kwargs)

    @staticmethod
    def _teacher(**kwargs):
        return RelationTeacher(base_channels=4, relation_dim=4, **kwargs)

    def test_masked_neighbours_exclude_padding_and_include_centre(self):
        valid = local_valid_mask(2, 3, 3, device=torch.device("cpu"))
        logits = torch.zeros(1, 2, 9, 2, 3)
        probabilities, logs = masked_local_distribution(logits, valid)
        self.assertEqual(valid[0, 0, :, 0, 0].sum().item(), 4)
        self.assertEqual(valid[0, 0, :, 0, 1].sum().item(), 6)
        self.assertTrue(valid[:, :, 4].all())
        self.assertTrue((probabilities.masked_select(~valid.expand_as(probabilities)) == 0).all())
        torch.testing.assert_close(probabilities.sum(2), torch.ones(1, 2, 2, 3))
        self.assertEqual(probabilities[0, 0, 4, 0, 0].item(), 0.25)
        self.assertTrue(torch.isfinite(logs).all())

    def test_single_pixel_distribution_is_finite_and_has_only_centre(self):
        branch = LocalRelationBranch(8, relation_dim=4, relation_kernel=5)
        output = branch(torch.randn(2, 8, 1, 1, requires_grad=True))
        probabilities = output["relation_probs"]
        self.assertEqual(probabilities[:, :, 12].sum().item(), 4)
        self.assertEqual(probabilities.sum().item(), 4)
        self.assertTrue(torch.isfinite(output["relation_logits"]).all())
        output["residual"].sum().backward()
        self.assertIsNotNone(branch.values[0].weight.grad)

    def test_entropy_normalizes_valid_neighbours_and_ignores_padding(self):
        valid = local_valid_mask(2, 3, 3, device=torch.device("cpu"))
        logits = torch.zeros(2, 2, 9, 2, 3, requires_grad=True)
        uniform = normalized_relation_entropy(logits, valid)
        self.assertFalse(uniform.requires_grad)
        self.assertAlmostEqual(uniform.item(), 1, places=6)
        padded = logits.detach().clone().masked_fill(~valid, 10000)
        torch.testing.assert_close(normalized_relation_entropy(padded, valid), uniform)
        padded[:, :, 4] = 10000
        self.assertEqual(normalized_relation_entropy(padded, valid).item(), 0)
        singleton = local_valid_mask(1, 1, 3, device=torch.device("cpu"))
        self.assertEqual(normalized_relation_entropy(torch.zeros(1, 2, 9, 1, 1), singleton).item(), 0)

    def test_segmentation_supervision_trains_relation_queries_and_keys(self):
        model = self._student()
        output = model(self.rgb)
        self.assertEqual(output["logits"].shape, self.mask.shape)
        self.assertEqual(output["boundary_logits"].shape, self.mask.shape)
        self.assertEqual(output["relation_logits"].shape, (2, 2, 9, 6, 7))
        losses = RelationLoss()(output, self.mask)
        losses["total"].backward()
        for name, parameter in model.relation_branch.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_relation_aggregation_changes_segmentation_forward(self):
        model = self._student().eval()
        with torch.no_grad():
            original = model(self.rgb)
            # Remove the entire relation branch contribution without touching
            # the backbone/decoder; logits must respond to this intervention.
            model.relation_branch.output_projection.weight.zero_()
            ablated = model(self.rgb)
        torch.testing.assert_close(original["relation_logits"], ablated["relation_logits"])
        self.assertGreater((original["logits"] - ablated["logits"]).abs().max().item(), 0)

    def test_rgb_teacher_is_invariant_to_thermal_depth_inputs(self):
        model = self._teacher(teacher_kind="rgb").eval()
        with torch.no_grad():
            expected = model(self.rgb)
            actual = model(self.rgb, torch.randn(7), torch.full((1,), float("nan")))
        for field in ("logits", "relation_logits", "features"):
            torch.testing.assert_close(expected[field], actual[field], rtol=0, atol=0)

    def test_teachers_have_identical_parameterization_and_initialization(self):
        torch.manual_seed(7)
        rgb = self._teacher(teacher_kind="rgb")
        torch.manual_seed(7)
        multimodal = self._teacher(teacher_kind="multimodal")
        self.assertEqual(list(rgb.state_dict()), list(multimodal.state_dict()))
        self.assertEqual(sum(p.numel() for p in rgb.parameters()), sum(p.numel() for p in multimodal.parameters()))
        for name, value in rgb.state_dict().items():
            torch.testing.assert_close(value, multimodal.state_dict()[name], rtol=0, atol=0)
        for model in (rgb, multimodal):
            for encoder in (model.rgb_encoder, model.thermal_encoder, model.depth_encoder):
                self.assertEqual(encoder.stem.block[0].in_channels, 3)

    def test_multimodal_teacher_relation_is_used_and_trained(self):
        model = self._teacher()
        thermal = torch.randn(2, 1, 24, 28)
        depth = torch.randn(2, 1, 24, 28)
        output = model(self.rgb, thermal, depth)
        RelationLoss()(output, self.mask)["total"].backward()
        for index in (0, 1):
            self.assertGreater(model.relation_branch.keys[index].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.thermal_encoder.stem.block[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.depth_encoder.stem.block[0].weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "requires real thermal"):
            model(self.rgb)
        with self.assertRaisesRegex(ValueError, "paired"):
            model(self.rgb, thermal[:, :, :-1], depth)

    def test_source_freeze_preserves_backbone_weights_and_batchnorm(self):
        model = self._student().freeze_backbone()
        before = {name: tensor.clone() for name, tensor in model.encoder.state_dict().items()}
        model.eval()
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.decoder.training)
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not module.training for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)))
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        RelationLoss()(model(self.rgb), self.mask)["total"].backward()
        optimizer.step()
        for name, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))
        self.assertGreater(model.relation_branch.queries[0].weight.grad.abs().sum().item(), 0)
        model.unfreeze_backbone()
        self.assertTrue(model.encoder.training)
        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_relation_kd_detaches_teacher_and_is_nonnegative(self):
        student, teacher = self._student(), self._teacher(teacher_kind="rgb")
        teacher_output = teacher(self.rgb)
        output = student(self.rgb)
        losses = RelationLoss(relation_weight=1, relation_temperature=2)(output, self.mask, teacher_output)
        self.assertGreater(losses["relation_kd"].item(), 0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        for key in ("relation_entropy", "teacher_relation_entropy"):
            self.assertFalse(losses[key].requires_grad)
            self.assertGreaterEqual(losses[key].item(), 0)
            self.assertLessEqual(losses[key].item(), 1)
        losses["total"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertGreater(student.relation_branch.queries[0].weight.grad.abs().sum().item(), 0)

    def test_equal_relations_have_zero_kd_and_disabled_kd_needs_no_targets(self):
        output = self._student()(self.rgb)
        losses = RelationLoss(relation_weight=1)(output, self.mask, output)
        self.assertAlmostEqual(losses["relation_kd"].item(), 0, places=7)
        plain = RelationLoss()({"logits": output["logits"]}, self.mask)
        torch.testing.assert_close(plain["total"], losses["total"])
        with self.assertRaisesRegex(ValueError, "requires teacher"):
            RelationLoss(relation_weight=1)(output, self.mask)

    def test_invalid_hyperparameters_and_relation_grid_mismatch(self):
        for kernel in (0, 2, -1, True):
            with self.subTest(kernel=kernel), self.assertRaises(ValueError):
                self._student(relation_kernel=kernel)
        for kwargs in ({"relation_weight": -1}, {"relation_weight": float("nan")},
                       {"relation_temperature": 0}, {"relation_temperature": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RelationLoss(**kwargs)
        student = self._student()(self.rgb)
        teacher = self._student(relation_kernel=5)(self.rgb)
        with self.assertRaisesRegex(ValueError, "must match exactly"):
            RelationLoss(relation_weight=1)(student, self.mask, teacher)

    def test_resnet_relation_grid_is_stride_16_and_output_has_input_size(self):
        model = RelationStudent(model_variant="resnet50", base_channels=8, pretrained=False, relation_dim=4).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(output["relation_logits"].shape, (1, 2, 9, 4, 4))
        self.assertEqual(output["logits"].shape, (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
