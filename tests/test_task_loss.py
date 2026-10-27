"""Independent arithmetic and gradient checks for task-oriented transfer."""
import math
import unittest

import torch
from torch.nn import functional as F

from models.triprd_correction_loss import segmentation_loss
from models.triprd_task_loss import boundary_target, supervised_loss, teacher_loss, transfer_loss


def edge_oracle(mask):
    result = torch.zeros_like(mask)
    for b in range(mask.shape[0]):
        for y in range(mask.shape[-2]):
            for x in range(mask.shape[-1]):
                window = mask[b, 0, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
                result[b, 0, y, x] = float(window.max() != window.min())
    return result


def balanced_oracle(errors, masses):
    images = []
    for b in range(errors.shape[0]):
        regions = []
        for mass in masses:
            denominator = mass[b].sum()
            if denominator.item() > 0:
                regions.append((errors[b] * mass[b]).sum() / denominator)
        if regions:
            images.append(torch.stack(regions).mean())
    return torch.stack(images).mean() if images else errors.sum() * 0


def area_oracle(value, side):
    b, c, h, w = value.shape
    return value.reshape(b, c, side, h // side, side, w // side).mean((3, 5))


class TaskLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        torch.manual_seed(41)
        self.mask = torch.zeros(2, 1, 8, 8)
        self.mask[0, :, 2:6, 2:6] = 1
        self.mask[1, :, 3:6, 1:5] = 1

    def output(self, *, requires_grad=True):
        leaf = lambda shape: torch.randn(shape, requires_grad=requires_grad)
        return {"logits": leaf(self.mask.shape), "boundary_logits": leaf(self.mask.shape),
                "aux_logits": (leaf((2, 1, 4, 4)), leaf((2, 1, 2, 2))),
                "task_features": (leaf((2, 3, 4, 4)), leaf((2, 3, 2, 2)))}

    def teacher_pair(self):
        rgb, full = self.output(), self.output()
        sign = self.mask * 2 - 1
        magnitudes = torch.linspace(.3, 3., self.mask.numel()).reshape_as(self.mask)
        rgb["logits"] = (sign * .2).requires_grad_()
        full["logits"] = (sign * magnitudes).requires_grad_()
        # One unreliable pixel, so support is neither empty nor artificially all one.
        with torch.no_grad():
            full["logits"][0, 0, 0, 0] = 2
        return rgb, full

    def test_boundary_oracle_retains_single_pixels_and_does_not_invent_frame_edges(self):
        mask = torch.zeros(1, 1, 7, 7)
        mask[0, 0, 3, 3] = 1
        torch.testing.assert_close(boundary_target(mask), edge_oracle(mask))
        self.assertEqual(boundary_target(mask).sum().item(), 9)
        for value in (0., 1.):
            self.assertEqual(boundary_target(torch.full_like(mask, value)).sum().item(), 0)
        mask[0, 0, 0, 0] = 1
        torch.testing.assert_close(boundary_target(mask * 255), edge_oracle(mask))

    def test_supervision_has_class_balanced_boundary_bce_and_expected_mask_losses(self):
        output = self.output()
        result = supervised_loss(output, self.mask, boundary_weight=.3, aux_weight=.7)
        edges = edge_oracle(self.mask)
        logits = output["boundary_logits"]
        # Stable BCE expression independent of the production BCE call.
        errors = torch.logaddexp(torch.zeros_like(logits), logits) - edges * logits
        expected_boundary = balanced_oracle(errors, (edges, 1 - edges))
        expected_aux = torch.stack([segmentation_loss(x, self.mask) for x in output["aux_logits"]]).mean()
        expected = segmentation_loss(output["logits"], self.mask) + .3 * expected_boundary + .7 * expected_aux
        torch.testing.assert_close(result["boundary"], expected_boundary)
        torch.testing.assert_close(result["total"], expected)
        result["total"].backward()
        for value in (output["logits"], output["boundary_logits"], *output["aux_logits"]):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        self.assertIsNone(output["task_features"][0].grad)

    def test_constant_masks_and_absent_auxiliary_are_finite_without_empty_class_bias(self):
        for value in (0., 1.):
            output = self.output()
            output["aux_logits"] = ()
            mask = torch.full_like(self.mask, value)
            result = supervised_loss(output, mask)
            expected = F.softplus(output["boundary_logits"]).mean()
            torch.testing.assert_close(result["boundary"], expected)
            self.assertEqual(result["auxiliary"].item(), 0)
            torch.testing.assert_close(result["total"], supervised_loss(output, mask * 255)["total"])

    def test_teacher_coalitions_have_equal_supervision_and_gradients(self):
        r, full = self.output(), self.output()
        result = teacher_loss({"R": r, "RTD": full}, self.mask)
        expected = (supervised_loss(r, self.mask)["total"] + supervised_loss(full, self.mask)["total"]) / 2
        torch.testing.assert_close(result["total"], expected)
        expected_grads = torch.autograd.grad(expected, (r["logits"], full["logits"]), retain_graph=True)
        result["total"].backward()
        torch.testing.assert_close(r["logits"].grad, expected_grads[0])
        torch.testing.assert_close(full["logits"].grad, expected_grads[1])

    def test_ordinary_bernoulli_kl_and_gradient_match_closed_form(self):
        student, teacher = self.output(), self.output()
        temperature = 2.3
        result = transfer_loss(student, teacher, self.mask, temperature=temperature)
        p = (student["logits"] / temperature).sigmoid()
        q = (teacher["logits"].detach() / temperature).sigmoid()
        expected = (q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()).mean() * temperature ** 2
        torch.testing.assert_close(result["kd_output"], expected)
        result["total"].backward()
        torch.testing.assert_close(student["logits"].grad, (p.detach() - q) * temperature / p.numel())
        self.assertIsNone(teacher["logits"].grad)
        self.assertIsNone(student["task_features"][0].grad)
        self.assertEqual(result["kd_feature"].item(), 0)
        self.assertEqual(result["reliable_coverage"].item(), 1)

    def test_task_loss_matches_independent_weighted_region_and_scale_oracle(self):
        student = self.output()
        rgb, teacher = self.teacher_pair()
        result = transfer_loss(student, teacher, self.mask, rgb, mode="task", feature_weight=.8)
        r, t = rgb["logits"].detach(), teacher["logits"].detach()
        reliability = ((torch.logaddexp(r, torch.zeros_like(r)) - r * self.mask)
                       - (torch.logaddexp(t, torch.zeros_like(t)) - t * self.mask)).clamp_min(0)
        edge = edge_oracle(self.mask)
        masses = (reliability * self.mask * (1 - edge),
                  reliability * (1 - self.mask) * (1 - edge), reliability * edge)
        p, q = (student["logits"] / 2).sigmoid(), (t / 2).sigmoid()
        divergence = (q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()) * 4
        expected_output = balanced_oracle(divergence, masses)
        feature_losses = []
        for sf, tf in zip(student["task_features"], teacher["task_features"]):
            tf = tf.detach()
            sn = sf / sf.square().sum(1, keepdim=True).sqrt()
            tn = tf / tf.square().sum(1, keepdim=True).sqrt()
            distance = (sn - tn).square().sum(1, keepdim=True)
            feature_losses.append(balanced_oracle(distance, tuple(area_oracle(m, sf.shape[-1]) for m in masses)))
        expected_feature = torch.stack(feature_losses).mean()
        expected = expected_output + .8 * expected_feature
        torch.testing.assert_close(result["kd_output"], expected_output)
        torch.testing.assert_close(result["kd_feature"], expected_feature)
        expected_grads = torch.autograd.grad(expected, (student["logits"], *student["task_features"]), retain_graph=True)
        result["total"].backward()
        for value, gradient in zip((student["logits"], *student["task_features"]), expected_grads):
            torch.testing.assert_close(value.grad, gradient, atol=2e-7, rtol=2e-5)
        self.assertEqual(result["reliable_coverage"].item(), 127 / 128)
        for output in (rgb, teacher):
            self.assertIsNone(output["logits"].grad)
            self.assertTrue(all(value.grad is None for value in output["task_features"]))

    def test_no_reliable_pixels_gives_differentiable_zero_for_logits_and_features(self):
        student, rgb, teacher = self.output(), self.output(), self.output()
        rgb["logits"] = (self.mask * 2 - 1) * 2
        teacher["logits"] = -rgb["logits"]
        result = transfer_loss(student, teacher, self.mask, rgb, mode="task")
        self.assertEqual(result["reliable_coverage"].item(), 0)
        self.assertEqual(result["total"].item(), 0)
        result["total"].backward()
        for value in (student["logits"], *student["task_features"]):
            self.assertIsNotNone(value.grad)
            self.assertEqual(value.grad.abs().sum().item(), 0)

    def test_no_rgb_reference_is_an_explicit_all_reliable_control(self):
        student, teacher = self.output(), self.output()
        result = transfer_loss(student, teacher, self.mask, mode="task")
        self.assertEqual(result["reliable_coverage"].item(), 1)
        self.assertGreater(result["kd_feature"].item(), 0)
        with_zero_weight = transfer_loss(student, teacher, self.mask, mode="task", feature_weight=0)
        torch.testing.assert_close(with_zero_weight["total"], result["kd_output"])

    def test_scale_normalized_features_and_equal_images_do_not_depend_on_area(self):
        student, teacher = self.output(), self.output()
        reference = transfer_loss(student, teacher, self.mask, mode="task")
        scaled = {**teacher, "task_features": tuple(value * 17 for value in teacher["task_features"])}
        torch.testing.assert_close(reference["kd_feature"], transfer_loss(student, scaled, self.mask, mode="task")["kd_feature"])
        separate = []
        for b in range(2):
            def sliced(output):
                return {key: tuple(x[b:b + 1] for x in value) if isinstance(value, tuple) else value[b:b + 1]
                        for key, value in output.items()}
            separate.append(transfer_loss(sliced(student), sliced(teacher), self.mask[b:b + 1], mode="task")["total"])
        torch.testing.assert_close(reference["total"], torch.stack(separate).mean())

    def test_feature_joint_support_pooling_preserves_boundary_reliability(self):
        mask = torch.zeros(1, 1, 4, 4)
        mask[0, 0, 0, 0] = 1
        student = {"logits": torch.zeros_like(mask, requires_grad=True),
                   "task_features": (torch.tensor([[[[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]]]], requires_grad=True),)}
        rgb = {"logits": (mask * 2 - 1) * 2}
        full = {"logits": -rgb["logits"],
                "task_features": (torch.tensor([[[[0., 1.], [1., 1.]], [[1., 0.], [0., 0.]]]]),)}
        full["logits"][0, 0, 0, 0] = 3
        result = transfer_loss(student, full, mask, rgb, mode="task")
        self.assertEqual(result["reliable_coverage"].item(), 1 / 16)
        self.assertAlmostEqual(result["kd_feature"].item(), 2., places=6)
        result["total"].backward()
        self.assertEqual(student["task_features"][0].grad[:, :, 1, :].abs().sum().item(), 0)
        self.assertEqual(student["task_features"][0].grad[:, :, 0, 1].abs().sum().item(), 0)

    def test_extreme_logits_and_autocast_remain_finite_and_nonnegative(self):
        for magnitude in (0., 1e-5, 100., 10000.):
            with self.subTest(magnitude=magnitude):
                logits = torch.full_like(self.mask, magnitude, requires_grad=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    same = transfer_loss({"logits": logits}, {"logits": logits.detach()}, self.mask)
                    opposite = transfer_loss({"logits": logits}, {"logits": -logits.detach()}, self.mask)
                self.assertEqual(same["total"].item(), 0)
                self.assertEqual(opposite["total"].dtype, torch.float32)
                self.assertGreaterEqual(opposite["total"].item(), 0)
                self.assertTrue(torch.isfinite(opposite["total"]))
                opposite["total"].backward()
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_nonmatching_native_logit_and_feature_resolutions_are_supported(self):
        student, teacher = self.output(), self.output()
        teacher["logits"] = torch.randn(2, 1, 3, 3, requires_grad=True)
        teacher["task_features"] = (torch.randn(2, 3, 3, 3, requires_grad=True),
                                    torch.randn(2, 3, 1, 1, requires_grad=True))
        student["logits"] = torch.randn(2, 1, 4, 4, requires_grad=True)
        result = transfer_loss(student, teacher, self.mask, mode="task")
        result["total"].backward()
        self.assertTrue(torch.isfinite(student["logits"].grad).all())
        self.assertIsNone(teacher["logits"].grad)
        self.assertTrue(all(x.grad is None for x in teacher["task_features"]))

    def test_argument_validation_rejects_invalid_weights_modes_and_shapes(self):
        student, teacher = self.output(), self.output()
        for kwargs in ({"temperature": 0}, {"temperature": -1}, {"temperature": math.inf},
                       {"feature_weight": -1}, {"feature_weight": math.nan}, {"mode": "delta"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                transfer_loss(student, teacher, self.mask, **kwargs)
        for kwargs in ({"boundary_weight": -1}, {"aux_weight": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                supervised_loss(student, self.mask, **kwargs)
        with self.assertRaises(ValueError):
            teacher_loss({"R": teacher}, self.mask)
        with self.assertRaises(ValueError):
            transfer_loss(student, {**teacher, "task_features": ()}, self.mask, mode="task")
        with self.assertRaises(ValueError):
            transfer_loss(student, {**teacher, "task_features": teacher["task_features"][:1]}, self.mask, mode="task")
        with self.assertRaises(ValueError):
            transfer_loss(student, {**teacher, "task_features": (torch.ones(2, 4, 4, 4),) * 2}, self.mask, mode="task")
        with self.assertRaises(ValueError):
            supervised_loss({**student, "boundary_logits": torch.zeros(1, 1, 8, 8)}, self.mask)
        with self.assertRaises(ValueError):
            supervised_loss({**student, "aux_logits": student["logits"]}, self.mask)


if __name__ == "__main__":
    unittest.main()
