"""Falsifiable mechanism checks for the complete-function structure model."""

import unittest
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from models.triprd_structure import (
    PairwiseEdgeHead, StructureStudent, StructureTeacher, TaskConditionedPropagation,
    _truncate_structure_encoder, normalize_structure_edges, propagate_structure, structure_valid_mask,
)
from models.triprd import TinyEncoder
from models.triprd_full import ResNet50Pyramid


class StructureModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.rgb = torch.randn(2, 3, 32, 40)
        self.thermal = torch.rand(2, 1, 32, 40)
        self.depth = torch.rand(2, 1, 32, 40)

    @staticmethod
    def _student(**kwargs):
        return StructureStudent(base_channels=4, relation_dim=4, **kwargs)

    @staticmethod
    def _teacher(**kwargs):
        return StructureTeacher(base_channels=4, relation_dim=4, **kwargs)

    def test_two_scale_contract_and_strict_rgb_only_student(self):
        model = self._student().eval()
        output = model(self.rgb)
        self.assertEqual(output["logits"].shape, (2, 1, 32, 40))
        self.assertEqual(output["boundary_logits"].shape, output["logits"].shape)
        for scale, (h, w) in enumerate(((16, 20), (8, 10))):
            self.assertEqual(output["structure_logits"][scale].shape, (2, 3, 9, h, w))
            self.assertEqual(output["semantic_logits"][scale].shape, (2, 1, 9, h, w))
            self.assertEqual(output["aux_logits"][scale].shape, (2, 1, h, w))
            self.assertEqual(output["structure_valid_masks"][scale].shape, (1, 1, 9, h, w))
            self.assertEqual(output["structure_valid_masks"][scale].dtype, torch.bool)
            self.assertEqual(output["propagation_gates"][scale].shape, (2, 5, h, w))
            torch.testing.assert_close(output["propagation_gates"][scale].sum(1), torch.ones(2, h, w))
        with self.assertRaises(TypeError):
            model(self.rgb, self.thermal)
        with self.assertRaises(ValueError):
            model(self.rgb, intervention="unknown")

    def test_off_diagonal_padding_and_fixed_self_loop(self):
        valid = structure_valid_mask(2, 3, 3, device=torch.device("cpu"))
        self.assertEqual(valid[0, 0, :, 0, 0].sum().item(), 3)
        self.assertEqual(valid[0, 0, :, 0, 1].sum().item(), 5)
        self.assertFalse(valid[:, :, 4].any())
        probabilities, self_mass = normalize_structure_edges(torch.ones(2, 3, 9, 2, 3), valid)
        torch.testing.assert_close(probabilities.sum(2, keepdim=True) + self_mass, torch.ones(2, 3, 1, 2, 3))
        self.assertEqual(self_mass[0, 0, 0, 0, 0].item(), 0.25)
        self.assertEqual(probabilities.masked_select(~valid.expand_as(probabilities)).count_nonzero().item(), 0)
        values = torch.ones(2, 4, 2, 3)
        aggregated, _ = propagate_structure(values, torch.rand(2, 3, 9, 2, 3), valid, 3)
        torch.testing.assert_close(aggregated, torch.ones(2, 3, 4, 2, 3))

    def test_single_pixel_has_no_learned_edges_and_is_exact_identity(self):
        valid = structure_valid_mask(1, 1, 3, device=torch.device("cpu"))
        self.assertFalse(valid.any())
        values = torch.randn(2, 4, 1, 1, requires_grad=True)
        logits = PairwiseEdgeHead(4, 4, 3, 3)(values, valid)
        aggregate, self_mass = propagate_structure(values, logits.sigmoid(), valid, 3)
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(aggregate, values[:, None].expand(-1, 3, -1, -1, -1), rtol=0, atol=0)
        self.assertTrue((self_mass == 1).all())
        aggregate.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        output = self._student().eval()(torch.randn(1, 3, 1, 1))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertTrue(all(not mask.any() for mask in output["structure_valid_masks"]))

    def test_segmentation_trains_graphs_at_both_scales_but_not_reference(self):
        model = self._student().train()
        output = model(self.rgb)
        loss = F.binary_cross_entropy_with_logits(output["logits"], torch.rand_like(output["logits"]))
        loss.backward()
        for head in model.structure_branch.heads:
            self.assertGreater(head.pair_mlp[-1].weight.grad.abs().sum().item(), 0)
        for graph in model.decoder.graphs:
            for parameter in (graph.semantic_head.projection.weight, graph.gate[-1].weight,
                              graph.value.weight, graph.output.weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.structure_branch.reference_encoder.parameters()))
        self.assertTrue(all(not module.training for module in model.structure_branch.reference_encoder.modules()))

    def test_freeze_backbone_preserves_parameters_and_batchnorm(self):
        model = self._student()
        model.freeze_backbone()
        model.train()
        before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
        reference_before = {key: value.clone() for key, value in model.structure_branch.reference_encoder.state_dict().items()}
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.structure_branch.heads.training)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        output = model(self.rgb)
        (output["logits"].square().mean() + sum(x.square().mean() for x in output["structure_logits"])).backward()
        optimizer.step()
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for key, value in model.structure_branch.reference_encoder.state_dict().items():
            torch.testing.assert_close(value, reference_before[key], rtol=0, atol=0)
        model.unfreeze_backbone()
        self.assertTrue(model.encoder.training)
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))

    def test_branch_transfer_preserves_full_function_while_task_encoder_changes(self):
        source = self._student().eval()
        source_logits, _ = source.structure_branch(self.rgb)
        torch.manual_seed(91)
        target = self._student()
        # Reference tensors must actually be present, not reconstructed from the target encoder.
        state = source.structure_branch.state_dict()
        self.assertTrue(any(name.startswith("reference_encoder.") for name in state))
        target.structure_branch.load_state_dict(state, strict=True)
        target.freeze_structure()
        target.train()
        self.assertFalse(target.structure_branch.training)
        self.assertTrue(all(not p.requires_grad for p in target.structure_branch.parameters()))
        reference_bn = [m for m in target.structure_branch.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertTrue(all(not m.training for m in reference_bn))
        encoder_before = target.encoder.stem.block[0].weight.detach().clone()
        frozen_before = {key: value.clone() for key, value in target.structure_branch.state_dict().items()}
        optimizer = torch.optim.SGD((p for p in target.parameters() if p.requires_grad), lr=1.0)
        output = target(self.rgb)
        output["logits"].square().mean().backward()
        optimizer.step()
        self.assertGreater((target.encoder.stem.block[0].weight - encoder_before).abs().sum().item(), 0)
        target_logits, _ = target.structure_branch(self.rgb)
        for actual, expected in zip(target_logits, source_logits):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in target.structure_branch.state_dict().items():
            torch.testing.assert_close(value, frozen_before[key], rtol=0, atol=0)

    def test_teacher_capacity_matching_and_rgb_control_modal_independence(self):
        torch.manual_seed(51)
        rgb_teacher = self._teacher(teacher_kind="rgb").eval()
        torch.manual_seed(51)
        modal_teacher = self._teacher(teacher_kind="multimodal").eval()
        self.assertEqual(sum(p.numel() for p in rgb_teacher.parameters()), sum(p.numel() for p in modal_teacher.parameters()))
        self.assertEqual(rgb_teacher.state_dict().keys(), modal_teacher.state_dict().keys())
        for key, value in rgb_teacher.state_dict().items():
            torch.testing.assert_close(value, modal_teacher.state_dict()[key], rtol=0, atol=0)
        first = rgb_teacher(self.rgb)
        second = rgb_teacher(self.rgb, self.thermal, self.depth)
        torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)
        for actual, expected in zip(first["structure_logits"], second["structure_logits"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            modal_teacher(self.rgb)

    def test_teacher_modal_effect_is_only_through_graphs(self):
        model = self._teacher().eval()
        first = model(self.rgb, self.thermal, self.depth)
        second = model(self.rgb, 1 - self.thermal, 1 - self.depth)
        self.assertTrue(all((a - b).abs().max().item() > 0 for a, b in zip(first["structure_logits"], second["structure_logits"])))
        self.assertGreater((first["logits"] - second["logits"]).abs().max().item(), 0)
        # Cutting graph edges must cut EVERY thermal/depth-to-mask path.
        for intervention in ("identity", "uniform", "no_source"):
            first = model(self.rgb, self.thermal, self.depth, intervention=intervention)
            second = model(self.rgb, 1 - self.thermal, 1 - self.depth, intervention=intervention)
            torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)
            for actual, expected in zip(first["aux_logits"], second["aux_logits"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_interventions_change_readout_not_predicted_source_edges(self):
        model = self._student().eval()
        baseline = model(self.rgb)
        for intervention in ("identity", "uniform", "shuffle", "no_source", "uniform_source", "shuffle_source"):
            output = model(self.rgb, intervention=intervention)
            self.assertTrue(torch.isfinite(output["logits"]).all())
            self.assertGreater((output["logits"] - baseline["logits"]).abs().max().item(), 0)
            for actual, expected in zip(output["structure_logits"], baseline["structure_logits"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        identity = model(self.rgb, intervention="identity")
        self.assertTrue(all((mass == 1).all() for mass in identity["propagation_self_mass"]))

    def test_source_interventions_preserve_task_semantics_and_gate_within_layer(self):
        graph = TaskConditionedPropagation(4, 4, 3).eval()
        features = torch.randn(2, 4, 5, 7)
        guidance = torch.randn(2, 1, 5, 7)
        structures = torch.randn(2, 3, 9, 5, 7)
        valid = structure_valid_mask(5, 7, 3, device=features.device)
        baseline = graph(features, guidance, structures, valid, "none")
        for intervention in ("no_source", "uniform_source", "shuffle_source"):
            updated = graph(features, guidance, structures, valid, intervention)
            torch.testing.assert_close(updated[1], baseline[1], rtol=0, atol=0)
            torch.testing.assert_close(updated[2], baseline[2], rtol=0, atol=0)
            # The fourth (semantic-only) graph is exactly the same operator.
            torch.testing.assert_close(updated[3][:, 3], baseline[3][:, 3], rtol=0, atol=0)
            self.assertGreater((updated[0] - baseline[0]).abs().max().item(), 0)
        no_source = graph(features, guidance, structures, valid, "no_source")
        self.assertTrue((no_source[3][:, :3] == 1).all())
        uniform = graph(features, guidance, structures, valid, "uniform_source")
        source_mean = structures.sigmoid().masked_fill(~valid, 0).sum((2, 3, 4), keepdim=True) / valid.sum()
        expected = (source_mean * baseline[1].sigmoid()).masked_fill(~valid, 0)
        torch.testing.assert_close(uniform[3][:, :3], (1 + expected.sum(2)).reciprocal())

    def test_invalid_options_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self._student(pretrained=True)
        with self.assertRaises(ValueError):
            self._student(relation_kernel=1)
        with self.assertRaises(ValueError):
            self._student(relation_kernel=2)

    def test_resnet_feature_indices_are_stride_eight_and_sixteen(self):
        model = StructureStudent(model_variant="resnet50", base_channels=4, relation_dim=4, pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(output["structure_logits"][0].shape, (1, 3, 9, 8, 8))
        self.assertEqual(output["structure_logits"][1].shape, (1, 3, 9, 4, 4))
        self.assertEqual(output["logits"].shape, (1, 1, 64, 64))

    def test_truncated_encoders_preserve_exact_graph_features_and_remove_final_stage(self):
        rgb = torch.randn(1, 3, 64, 64)
        for encoder, stage_name in ((TinyEncoder(3, 4), "stage4"), (ResNet50Pyramid(3, False), "layer4")):
            with self.subTest(encoder=type(encoder).__name__):
                encoder.eval()
                truncated = _truncate_structure_encoder(deepcopy(encoder)).eval()
                self.assertIsInstance(getattr(truncated, stage_name), nn.Identity)
                self.assertFalse(any(name.startswith(stage_name + ".") for name, _ in truncated.named_parameters()))
                removed_parameters = sum(p.numel() for p in getattr(encoder, stage_name).parameters())
                self.assertGreater(removed_parameters, 0)
                self.assertEqual(sum(p.numel() for p in encoder.parameters()) - sum(p.numel() for p in truncated.parameters()),
                                 removed_parameters)
                with torch.no_grad():
                    full_features = encoder(rgb)
                    truncated_features = truncated(rgb)
                for index in (1, 2):
                    torch.testing.assert_close(full_features[index], truncated_features[index], rtol=0, atol=0)
                self.assertEqual(truncated_features[3].data_ptr(), truncated_features[2].data_ptr())

    def test_only_structure_encoders_are_truncated(self):
        student = self._student()
        teacher = self._teacher()
        self.assertNotIsInstance(student.encoder.stage4, nn.Identity)
        self.assertIsInstance(student.structure_branch.reference_encoder.stage4, nn.Identity)
        self.assertNotIsInstance(teacher.rgb_encoder.stage4, nn.Identity)
        self.assertIsInstance(teacher.thermal_encoder.stage4, nn.Identity)
        self.assertIsInstance(teacher.depth_encoder.stage4, nn.Identity)


if __name__ == "__main__":
    unittest.main()
