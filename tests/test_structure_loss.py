"""Typed-edge targets must preserve sensor meaning and exclude invalid pairs."""
import unittest

import torch

from models.triprd_relation import local_valid_mask
from models.triprd_structure_loss import (
    StructureLoss, annotation_targets, observed_targets, balanced_edge_mean,
)


class StructureLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def output(self, h=3, w=4):
        return {
            "logits": torch.zeros(2, 1, 12, 16, requires_grad=True),
            "aux_logits": (torch.zeros(2, 1, h, w, requires_grad=True),),
            "structure_logits": (torch.zeros(2, 3, 9, h, w, requires_grad=True),),
            "semantic_logits": (torch.zeros(2, 1, 9, h, w, requires_grad=True),),
            "structure_valid_masks": (local_valid_mask(h, w, 3, device=torch.device("cpu")),),
        }

    def mask(self):
        target = torch.zeros(2, 1, 12, 16)
        target[:, :, :, :8] = 1
        return target

    def test_annotation_relation_distinguishes_crossing_from_same_background(self):
        labels = torch.tensor([[[[0., 0., 1.], [0., 0., 1.]]]])
        target, strata = annotation_targets(labels, (2, 3), 3)
        self.assertEqual(target[0, 0, 5, 0, 0].item(), 1)  # BG -> BG
        self.assertEqual(target[0, 0, 5, 0, 1].item(), 0)  # BG -> FG
        self.assertTrue(strata[1][0, 0, 5, 0, 0])
        self.assertTrue(strata[2][0, 0, 5, 0, 1])

    def test_observed_difference_has_continuous_scale_and_no_self_supervision(self):
        sensor = torch.tensor([[[[0.2, 0.2, 0.3], [0.2, 0.2, 0.3]]]])
        target, valid = observed_targets(sensor, (2, 3), 3, depth=True)
        self.assertEqual(target[0, 0, 5, 0, 0].item(), 1)
        self.assertAlmostEqual(target[0, 0, 5, 0, 1].item(), torch.exp(torch.tensor(-10.)).item(), places=6)
        self.assertFalse(valid[:, :, 4].any())
        self.assertFalse(valid[0, 0, 0, 0, 0])

    def test_depth_zero_invalid_but_thermal_zero_valid(self):
        sensor = torch.zeros(1, 1, 2, 3)
        _, depth = observed_targets(sensor, (2, 3), 3, depth=True)
        target, thermal = observed_targets(sensor, (2, 3), 3)
        self.assertFalse(depth.any())
        self.assertTrue(thermal.any())
        self.assertTrue((target[thermal] == 1).all())

    def test_invalid_observation_does_not_contaminate_pooling_and_masks_both_ends(self):
        sensor = torch.ones(1, 1, 4, 4) * 0.5
        sensor[0, 0, 0, 0] = float("nan")
        sensor[0, 0, 3, 3] = float("inf")
        target, valid = observed_targets(sensor, (2, 2), 3, depth=True)
        self.assertTrue(torch.isfinite(target).all())
        self.assertFalse(valid[:, :, :, 0, 0].any())
        self.assertFalse(valid[:, :, :, 1, 1].any())
        self.assertFalse(valid[0, 0, 3, 0, 1])  # neighbour 0,0 invalid
        validity = torch.zeros_like(sensor)
        _, masked = observed_targets(torch.nan_to_num(sensor), (2, 2), 3, validity=validity)
        self.assertFalse(masked.any())

    def test_balancing_does_not_let_many_background_pairs_swamp_rare_edges(self):
        values = torch.tensor([[[[[2., 2., 2., 2., 8.]]]]], requires_grad=True)
        valid = torch.ones_like(values, dtype=torch.bool)
        rare = torch.tensor([[[[[False, False, False, False, True]]]]])
        result = balanced_edge_mean(values, valid, (~rare, rare))
        self.assertEqual(result.item(), 5.)
        result.backward()
        self.assertAlmostEqual(values.grad.flatten()[-1].item(), 0.5)

    def test_none_supervises_cod_semantic_graph_without_modality_or_source_gradients(self):
        output = self.output()
        loss = StructureLoss(supervision="none")(output, self.mask())
        self.assertEqual(loss["structure_supervision"].item(), 0)
        loss["total"].backward()
        self.assertIsNone(output["structure_logits"][0].grad)
        self.assertGreater(output["semantic_logits"][0].grad.abs().sum().item(), 0)
        self.assertGreater(output["aux_logits"][0].grad.abs().sum().item(), 0)

    def test_mask_only_source_leaves_thermal_depth_unsupervised(self):
        output = self.output()
        loss = StructureLoss(supervision="mask")(output, self.mask())
        loss["total"].backward()
        gradient = output["structure_logits"][0].grad
        self.assertEqual(gradient[:, :2].abs().sum().item(), 0)
        self.assertGreater(gradient[:, 2:].abs().sum().item(), 0)
        self.assertEqual(gradient[:, :, 4].abs().sum().item(), 0)

    def test_observed_supervision_trains_three_independent_types(self):
        output = self.output()
        sensor = torch.rand(2, 1, 12, 16) * 0.8 + 0.1
        losses = StructureLoss(supervision="observed")(
            output, self.mask(), modalities={"thermal": sensor, "depth": sensor})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        for channel in range(3):
            self.assertGreater(output["structure_logits"][0].grad[:, channel].abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "requires real"):
            StructureLoss(supervision="observed")(output, self.mask())

    def test_bernoulli_kd_detaches_teacher_and_detects_uniform_wrong_edges(self):
        student, teacher = self.output(), self.output()
        with torch.no_grad():
            teacher["structure_logits"][0][:] = 2
        criterion = StructureLoss(distillation_weight=1)
        losses = criterion(student, self.mask(), teacher)
        self.assertGreater(losses["relation_kd"].item(), 0.1)
        losses["total"].backward()
        self.assertIsNone(teacher["structure_logits"][0].grad)
        self.assertGreater(student["structure_logits"][0].grad.abs().sum().item(), 0)
        equal = criterion(teacher, self.mask(), teacher)
        self.assertAlmostEqual(equal["relation_kd"].item(), 0, places=7)

    def test_singleton_or_all_invalid_edges_are_finite_with_zero_edge_loss(self):
        output = self.output(h=1, w=1)
        losses = StructureLoss(supervision="observed")(
            output, self.mask(), modalities={"thermal": torch.zeros(2, 1, 12, 16),
                                           "depth": torch.zeros(2, 1, 12, 16)})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertEqual(losses["structure_supervision"].item(), 0)
        self.assertEqual(losses["semantic_affinity"].item(), 0)
        losses["total"].backward()

    def test_missing_depth_is_reported_and_does_not_reduce_source_selection_loss(self):
        output = self.output()
        losses = StructureLoss(supervision="observed")(
            output, self.mask(), modalities={"thermal": torch.zeros(2, 1, 12, 16),
                                           "depth": torch.zeros(2, 1, 12, 16)})
        self.assertEqual(losses["structure_depth_valid_fraction"].item(), 0)
        self.assertEqual(losses["structure_thermal_valid_fraction"].item(), 1)
        expected = (losses["structure_thermal"] + losses["structure_annotation"]) / 2
        torch.testing.assert_close(losses["structure_supervision"], expected)
        self.assertAlmostEqual(expected.item(), torch.log(torch.tensor(2.)).item(), places=6)

    def test_adding_observed_types_does_not_change_annotation_gradient_weight(self):
        mask_output, observed_output = self.output(), self.output()
        sensor = torch.ones(2, 1, 12, 16) * 0.5
        mask_loss = StructureLoss(supervision="mask")(mask_output, self.mask())
        observed_loss = StructureLoss(supervision="observed")(
            observed_output, self.mask(), modalities={"thermal": sensor, "depth": sensor})
        mask_loss["total"].backward()
        observed_loss["total"].backward()
        torch.testing.assert_close(mask_output["structure_logits"][0].grad[:, 2],
                                   observed_output["structure_logits"][0].grad[:, 2])

    def test_depth_edges_are_invariant_to_container_scale_and_valid_offset(self):
        sensor = torch.tensor([[[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]]])
        first, valid = observed_targets(sensor, (2, 3), 3, depth=True)
        compressed, other_valid = observed_targets(sensor * 0.001 + 0.03, (2, 3), 3, depth=True)
        torch.testing.assert_close(valid, other_valid)
        torch.testing.assert_close(first[valid], compressed[valid], rtol=1e-4, atol=2e-6)
        self.assertTrue((compressed[valid] < 0.5).any())

    def test_pseudo_color_thermal_edges_do_not_collapse_equal_mean_colors(self):
        color = torch.zeros(1, 3, 2, 2)
        color[:, 0, :, 0] = 1
        color[:, 1, :, 1] = 1  # same channel average, different appearance
        targets, valid = observed_targets(color, (2, 2), 3)
        gray_targets, _ = observed_targets(color.mean(1, keepdim=True), (2, 2), 3)
        self.assertLess(targets[0, 0, 5, 0, 0].item(), 0.001)
        self.assertEqual(gray_targets[0, 0, 5, 0, 0].item(), 1)
        self.assertTrue(valid[0, 0, 5, 0, 0])


if __name__ == "__main__":
    unittest.main()
