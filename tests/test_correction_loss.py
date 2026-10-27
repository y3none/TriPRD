"""Mathematical checks for annotation edges, correction KD and gradient selection."""
import unittest

import torch
from torch.nn import functional as F

from models.triprd_correction_loss import (
    annotation_relations, segmentation_loss, relation_supervision, teacher_loss, source_loss,
    correction_targets, positive_gradient_alignment, constant_correction_statistics,
    finalize_constant_correction, correction_probe, finalize_correction_probe,
)


class CorrectionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        torch.manual_seed(17)
        self.mask = torch.tensor([[[[0., 0., 1.], [0., 0., 1.]]]])

    def output(self, value=0., requires_grad=True, four_dim=False):
        relation = torch.full((1, 1, 9, 2, 3), value, requires_grad=requires_grad)
        correction = torch.full_like(relation, value, requires_grad=requires_grad)
        base = torch.full_like(relation, value, requires_grad=requires_grad)
        return {"logits": torch.full((1, 1, 2, 3), value, requires_grad=requires_grad),
                "relation_logits": (relation.squeeze(1) if four_dim else relation,),
                "base_relation_logits": (base.squeeze(1) if four_dim else base,),
                "corrections": (correction.squeeze(1) if four_dim else correction,)}

    def teachers(self, improvement=True):
        target, _, _ = annotation_relations(self.mask, (2, 3))
        rgb, full = self.output(requires_grad=False), self.output(requires_grad=False)
        full["relation_logits"] = ((target * 2 - 1) * (2 if improvement else -2),)
        return rgb, full

    def test_validity_excludes_padding_centre_and_keeps_annotation_strata(self):
        target, valid, strata = annotation_relations(self.mask, (2, 3))
        self.assertFalse(valid[:, :, 4].any())
        self.assertEqual(valid[0, 0, :, 0, 0].sum().item(), 3)
        self.assertEqual(target[0, 0, 5, 0, 0].item(), 1)
        self.assertEqual(target[0, 0, 5, 0, 1].item(), 0)
        self.assertTrue(strata[1][0, 0, 5, 0, 0])
        self.assertTrue(strata[2][0, 0, 5, 0, 1])
        logits = torch.zeros_like(target, requires_grad=True)
        loss = relation_supervision((logits,), self.mask)
        self.assertAlmostEqual(loss.item(), 0.69314718056, places=6)
        loss.backward()
        self.assertTrue((logits.grad.masked_select(~valid) == 0).all())
        poisoned = logits.detach().masked_fill(~valid, 1000)
        torch.testing.assert_close(relation_supervision((poisoned,), self.mask), loss.detach())

    def test_relation_balance_equalizes_nonempty_foreground_background_crossing(self):
        target, valid, strata = annotation_relations(self.mask, (2, 3))
        logits = torch.where(strata[0], 1., torch.where(strata[1], -2., 3.))
        errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        expected = torch.stack([errors.masked_select(valid & stratum).mean() for stratum in strata]).mean()
        torch.testing.assert_close(relation_supervision((logits,), self.mask), expected)

    def test_single_pixel_relations_have_finite_zero_loss_and_gradient(self):
        mask = torch.ones(1, 1, 1, 1)
        relation = torch.randn(1, 1, 9, 1, 1, requires_grad=True)
        loss = relation_supervision((relation,), mask)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(relation.grad).all())
        self.assertEqual(relation.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(segmentation_loss(torch.zeros_like(mask), mask)))

    def test_segmentation_weights_boundary_and_handles_constant_masks(self):
        mask = torch.zeros(1, 1, 9, 9)
        mask[:, :, 3:6, 3:6] = 1
        logits = (mask * 2 - 1) * 3
        logits[:, :, 3, 3:6] = -3
        weighted = segmentation_loss(logits, mask, boundary_kernel=3)
        unweighted = segmentation_loss(logits, mask, boundary_weight=0, boundary_kernel=3)
        self.assertGreater(weighted.item(), unweighted.item())
        constant = torch.ones_like(mask)
        torch.testing.assert_close(segmentation_loss(logits, constant, boundary_kernel=3),
                                   segmentation_loss(logits, constant, boundary_weight=0, boundary_kernel=3))
        torch.testing.assert_close(segmentation_loss(logits, mask * 255), segmentation_loss(logits, mask))

    def test_teacher_supervises_four_coalitions_equally(self):
        outputs = {name: self.output(float(index) - 1) for index, name in enumerate(("R", "RT", "RD", "RTD"))}
        result = teacher_loss(outputs, self.mask)
        expected = torch.stack([source_loss(output, self.mask)["total"] for output in outputs.values()]).mean()
        torch.testing.assert_close(result["total"], expected)
        result["total"].backward()
        self.assertTrue(all(output["logits"].grad.abs().sum() > 0 for output in outputs.values()))
        with self.assertRaises(ValueError):
            teacher_loss({"R": outputs["R"], "RTD": outputs["RTD"]}, self.mask)

    def test_teacher_shared_rgb_reference_does_not_multiply_rgb_gradient(self):
        names = ("R", "RT", "RD", "RTD")
        outputs = {name: self.output(0.3) for name in names}
        rgb_relations = outputs["R"]["relation_logits"]
        # This aliasing matches the real teacher, unlike independent test bases.
        for output in outputs.values():
            output["base_relation_logits"] = rgb_relations
        result = teacher_loss(outputs, self.mask, relation_weight=2, aux_weight=0)
        result["total"].backward()
        gradients = [outputs[name]["relation_logits"][0].grad for name in names]
        self.assertGreater(gradients[0].abs().sum().item(), 0)
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient, gradients[0], rtol=0, atol=0)
        standalone = torch.full_like(rgb_relations[0], 0.3, requires_grad=True)
        relation_supervision((standalone,), self.mask).backward()
        # Relation weight 2 divided equally among four coalitions.
        torch.testing.assert_close(gradients[0], standalone.grad * 0.5)

    def test_teacher_ignores_base_reference_in_objective_but_student_supervises_it(self):
        names = ("R", "RT", "RD", "RTD")
        outputs = {name: self.output(float(i) - 1) for i, name in enumerate(names)}
        for output in outputs.values():
            output["base_relation_logits"] = (torch.full_like(output["relation_logits"][0], 8., requires_grad=True),)
        result = teacher_loss(outputs, self.mask, aux_weight=0)
        expected = torch.stack([
            segmentation_loss(output["logits"], self.mask) + relation_supervision(output["relation_logits"], self.mask)
            for output in outputs.values()
        ]).mean()
        torch.testing.assert_close(result["total"], expected)
        result["total"].backward()
        self.assertTrue(all(output["base_relation_logits"][0].grad is None for output in outputs.values()))

        student = self.output(0.3)
        student["base_relation_logits"] = (torch.full_like(student["relation_logits"][0], -2., requires_grad=True),)
        student_loss = source_loss(student, self.mask, aux_weight=0)
        expected = segmentation_loss(student["logits"], self.mask) + 0.5 * (
            relation_supervision(student["relation_logits"], self.mask) +
            relation_supervision(student["base_relation_logits"], self.mask))
        torch.testing.assert_close(student_loss["total"], expected)
        student_loss["total"].backward()
        for key in ("relation_logits", "base_relation_logits"):
            self.assertGreater(student[key][0].grad.abs().sum().item(), 0)

    def test_reliability_is_gt_advantage_and_teachers_are_detached(self):
        rgb, full = self.teachers()
        rgb["relation_logits"][0].requires_grad_(True)
        full["relation_logits"][0].requires_grad_(True)
        target = correction_targets(rgb, full, self.mask)
        valid = target["valid"][0]
        self.assertAlmostEqual(target["reliability"][0].masked_select(valid).mean().item(),
                               0.69314718056 - 0.12692801104, places=6)
        self.assertTrue(all(not tensor.requires_grad for values in target.values() for tensor in values))
        self.assertEqual(target["reliability"][0].masked_select(~valid).sum().item(), 0)
        _, bad = self.teachers(improvement=False)
        self.assertEqual(correction_targets(rgb, bad, self.mask)["reliability"][0].sum().item(), 0)

    def test_no_positive_coverage_returns_zero_transfer_but_keeps_gt(self):
        rgb, full = self.teachers(improvement=False)
        output = self.output()
        result = source_loss(output, self.mask, rgb, full, mode="correction")
        self.assertEqual(result["positive_coverage"].item(), 0)
        self.assertEqual(result["transfer"].item(), 0)
        torch.testing.assert_close(result["total"], result["supervised"])
        result["total"].backward()
        self.assertGreater(output["logits"].grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(output["corrections"][0].grad).all())
        self.assertEqual(output["corrections"][0].grad.abs().sum().item(), 0)

    def test_supplied_student_validity_also_masks_distillation(self):
        rgb, full = self.teachers()
        for mode in ("ordinary", "correction"):
            output = self.output()
            output["relation_valid_masks"] = (torch.zeros(1, 1, 9, 2, 3, dtype=torch.bool),)
            losses = source_loss(output, self.mask, rgb, full, mode=mode)
            self.assertEqual(losses["transfer"].item(), 0.)
            self.assertEqual(losses["relation"].item(), 0.)
            self.assertEqual(losses["reliability_mean"].item(), 0.)
            self.assertTrue(torch.isfinite(losses["total"]))

    def test_ordinary_kd_uses_corrected_head_and_detaches_teacher(self):
        rgb, full = self.teachers()
        full["relation_logits"][0].requires_grad_(True)
        output = self.output()
        base, delta = output["base_relation_logits"][0], output["corrections"][0]
        output["relation_logits"] = (torch.logit((base.sigmoid() + delta).clamp(.001, .999)),)
        result = source_loss(output, self.mask, rgb, full, mode="ordinary")
        self.assertGreater(result["transfer"].item(), 0)
        result["transfer"].backward()
        self.assertGreater(delta.grad.abs().sum().item(), 0)
        self.assertGreater(base.grad.abs().sum().item(), 0)
        self.assertIsNone(full["relation_logits"][0].grad)

    def test_reliable_transfer_updates_correction_and_alignment_only_gates_transfer(self):
        rgb, full = self.teachers()
        output = self.output()
        gate = torch.tensor(0., requires_grad=True)
        result = source_loss(output, self.mask, rgb, full, mode="correction", alignment=gate)
        self.assertGreater(result["transfer"].item(), 0)
        self.assertEqual(result["weighted_transfer"].item(), 0)
        result["total"].backward()
        self.assertIsNone(gate.grad)
        self.assertGreater(output["relation_logits"][0].grad.abs().sum().item(), 0)
        self.assertEqual(output["corrections"][0].grad.abs().sum().item(), 0)

    def test_four_dimensional_and_five_dimensional_relation_interfaces_match(self):
        rgb, full = self.teachers()
        five = self.output()
        four = self.output(four_dim=True)
        expected = source_loss(five, self.mask, rgb, full, mode="correction")
        actual = source_loss(four, self.mask, rgb, full, mode="correction")
        torch.testing.assert_close(actual["total"], expected["total"])

    def test_gradient_alignment_opposing_orthogonal_zero_and_unused_parameters(self):
        p, unused = torch.nn.Parameter(torch.tensor([1., 2.])), torch.nn.Parameter(torch.tensor(4.))
        for source, target, expected in ((p.sum(), p.sum(), 1.), (p.sum(), -p.sum(), 0.),
                                         (p[0], p[1], 0.), (p.sum() * 0, p.sum(), 0.)):
            gate = positive_gradient_alignment(source, target, [p, unused])
            self.assertAlmostEqual(gate.item(), expected, places=6)
            self.assertFalse(gate.requires_grad)
            self.assertIsNone(p.grad)
            (source + target).backward()
            p.grad = None
        self.assertEqual(positive_gradient_alignment(p.sum(), torch.tensor(1.), [p]).item(), 0)
        self.assertEqual(positive_gradient_alignment(p.sum(), p.sum(), []).item(), 0)
        self.assertEqual(positive_gradient_alignment(p.sum(), unused, [p, unused]).item(), 0)

    def test_gradient_scaling_preserves_cosine_and_prevents_half_chain_underflow(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        source, target = p[0] + 2 * p[1], 2 * p[0] + p[1]
        expected = positive_gradient_alignment(source, target, [p])
        scaled = positive_gradient_alignment(source, target, [p], gradient_scale=1024.)
        torch.testing.assert_close(scaled, expected)
        self.assertAlmostEqual(scaled.item(), .8, places=6)
        # Simulate an fp16 intermediate even on CPU; final loss is fp32.
        tiny_source = (p.half()[0] * 1e-4).float() * 1e-4
        tiny_target = (p.half()[0] * 1e-4).float() * 1e-4
        self.assertEqual(positive_gradient_alignment(tiny_source, tiny_target, [p]).item(), 0.)
        self.assertAlmostEqual(positive_gradient_alignment(tiny_source, tiny_target, [p], gradient_scale=1024.).item(), 1.)
        self.assertIsNone(p.grad)

    def test_gradient_scale_validation_and_nonfinite_probe_raise(self):
        p = torch.nn.Parameter(torch.tensor(0.))
        for scale in (0., -1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                positive_gradient_alignment(p, p, [p], gradient_scale=scale)
        # The loss itself is finite, but its derivative at zero is infinite.
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite gradient-alignment"):
            positive_gradient_alignment(p.sqrt(), p, [p], gradient_scale=1024.)

    def test_train_constant_statistics_are_fixed_and_batch_additive(self):
        rgb, full = self.teachers()
        train_targets = correction_targets(rgb, full, self.mask)
        statistics = constant_correction_statistics(train_targets)
        doubled = {key: tuple(value * 2 for value in values) for key, values in statistics.items()}
        constants = finalize_constant_correction(statistics)
        positive_constants = finalize_constant_correction(statistics, positive_only=True)
        for actual, expected in zip(finalize_constant_correction(doubled), constants):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(finalize_constant_correction(doubled, positive_only=True), positive_constants):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(constants[0].shape, (1, 1, 9, 1, 1))
        self.assertEqual(constants[0][:, :, 4].item(), 0)
        validation_full = {**full, "relation_logits": (full["relation_logits"][0] * 2,)}
        validation_targets = correction_targets(rgb, validation_full, self.mask)
        output = self.output()
        stats = correction_probe(output, validation_targets, constants, positive_constant_correction=positive_constants)
        final = finalize_correction_probe(stats)
        doubled_stats = {key: value * 2 for key, value in stats.items()}
        torch.testing.assert_close(final["mse"], finalize_correction_probe(doubled_stats)["mse"])
        torch.testing.assert_close(final["mse"], final["zero_mse"])
        torch.testing.assert_close(final["improved_task_error"], torch.tensor(0.))
        self.assertTrue(all(torch.isfinite(value) for value in final.values()))

    def test_exact_delta_probe_beats_zero_and_fixed_constant_and_improves_task(self):
        rgb, full = self.teachers()
        targets = correction_targets(rgb, full, self.mask)
        output = self.output()
        output["corrections"] = targets["delta"]
        probe = finalize_correction_probe(correction_probe(output, targets, (torch.tensor(0.),),
                                                         positive_constant_correction=(torch.tensor(0.),)))
        self.assertEqual(probe["mse"].item(), 0)
        self.assertGreater(probe["zero_mse"].item(), 0)
        self.assertGreater(probe["improved_task_error"].item(), 0)
        torch.testing.assert_close(probe["corrected_task_error"], probe["teacher_task_error"])

    def test_empty_probe_and_invalid_inputs_are_explicit(self):
        output = {"logits": torch.zeros(1, 1, 1, 1), "relation_logits": (torch.zeros(1, 1, 9, 1, 1),),
                  "corrections": (torch.zeros(1, 1, 9, 1, 1),)}
        targets = correction_targets(output, output, torch.ones(1, 1, 1, 1))
        final = finalize_correction_probe(correction_probe(output, targets, (torch.tensor(0.),),
                                                         positive_constant_correction=(torch.tensor(0.),)))
        self.assertEqual(final["count"].item(), 0)
        self.assertTrue(all(torch.isfinite(value) for value in final.values()))
        with self.assertRaises(ValueError):
            relation_supervision((torch.zeros(1, 8, 2, 3),), self.mask)
        with self.assertRaises(ValueError):
            source_loss(self.output(), self.mask, mode="ordinary")
        with self.assertRaises(ValueError):
            correction_probe(output, targets, (), positive_constant_correction=())

    def test_positive_constant_collapse_cannot_pass_learnability_gate(self):
        from tools.run_triprd_optimization_phases import EvidenceBlocked, student_gate
        # Both TRAIN and VAL have opposite deltas on reliable/unreliable edges.
        # A constant +.1 output learns no image-specific correction but beats
        # the all-edge TRAIN mean (zero) when evaluated only on reliable edges.
        delta = torch.tensor([.1, -.1]).reshape(1, 1, 1, 1, 2).expand(1, 1, 9, 1, 2)
        targets = {"delta": (delta,), "valid": (torch.ones_like(delta, dtype=torch.bool),),
                   "reliability": ((delta > 0).float(),), "target": (torch.ones_like(delta),),
                   "base_probability": (torch.full_like(delta, .5),),
                   "full_probability": (.5 + delta,)}
        train_stats = constant_correction_statistics(targets)
        all_constants = finalize_constant_correction(train_stats)
        positive_constants = finalize_constant_correction(train_stats, positive_only=True)
        output = {"corrections": (torch.full_like(delta, .1),), "relation_logits": (torch.zeros_like(delta),)}
        final = finalize_correction_probe(correction_probe(output, targets, all_constants,
            positive_constant_correction=positive_constants))
        self.assertEqual(final["positive_mse"].item(), 0.)
        self.assertGreater(final["positive_all_edge_constant_mse"].item(), 0.)
        self.assertEqual(final["positive_constant_mse"].item(), 0.)
        with self.assertRaisesRegex(EvidenceBlocked, "BOTH zero"):
            student_gate({"source_recoverability": {key: float(value) for key, value in final.items()}})
        # Reusing the same TRAIN constants on changed VAL targets must NOT fit
        # the new +.2 positive mean using validation labels.
        changed = {**targets, "delta": (delta * 2,)}
        changed_probe = finalize_correction_probe(correction_probe(output, changed, all_constants,
            positive_constant_correction=positive_constants))
        self.assertAlmostEqual(float(changed_probe["positive_constant_mse"]), .01, places=6)


if __name__ == "__main__":
    unittest.main()
