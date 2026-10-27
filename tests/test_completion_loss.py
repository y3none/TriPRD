"""Independent KD arithmetic, gradient isolation, and stage supervision tests."""
import unittest
import torch
from torch.nn import functional as F
from models.triprd_completion_loss import output_distillation, supervised_loss
from models.triprd_task_loss import boundary_target
from models.triprd_correction_loss import segmentation_loss


class CompletionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        torch.manual_seed(51)
        self.mask = torch.zeros(2, 1, 8, 8)
        self.mask[:, :, 2:6, 2:6] = 1

    def output(self):
        masks = tuple(torch.randn(2, 1, 8, 8, requires_grad=True) for _ in range(3))
        edges = tuple(torch.randn(2, 1, 8, 8, requires_grad=True) for _ in range(3))
        return {"logits": masks[-1], "stage_logits": masks, "stage_boundaries": edges,
                "boundary_logits": edges[-1], "coarse_logits": torch.randn_like(self.mask, requires_grad=True)}

    def test_stage_supervision_and_coarse_locator_all_receive_gradients(self):
        output = self.output()
        result = supervised_loss(output, self.mask, aux_weight=.7, boundary_weight=.3)
        expected_aux = sum(segmentation_loss(x, self.mask) for x in output["stage_logits"][:-1]) / 2
        expected_coarse = segmentation_loss(output["coarse_logits"], self.mask)
        torch.testing.assert_close(result["aux"], expected_aux)
        torch.testing.assert_close(result["coarse"], expected_coarse)
        torch.testing.assert_close(result["total"], result["mask"] + .7 * (expected_aux + expected_coarse)
                                   + .3 * result["boundary"])
        result["total"].backward()
        for value in (*output["stage_logits"], *output["stage_boundaries"], output["coarse_logits"]):
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_kd_matches_closed_form_fixed_pixel_mean_and_teacher_has_no_gradient(self):
        student, teacher = self.output(), self.output()
        temperature, edge_weight = 2.3, .4
        result = output_distillation(student, teacher, self.mask, temperature, edge_weight)
        expected = []
        for key, label in (("stage_logits", self.mask), ("stage_boundaries", boundary_target(self.mask))):
            values = []
            for s, t in zip(student[key], teacher[key]):
                p, q = (s / temperature).sigmoid(), (t.detach() / temperature).sigmoid()
                divergence = (q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()) * temperature ** 2
                quality = 1 - (t.detach().sigmoid() - label).abs()
                values.append((divergence * quality).sum() / self.mask.numel())
            expected.append(torch.stack(values).mean())
        torch.testing.assert_close(result["mask_kd"], expected[0])
        torch.testing.assert_close(result["boundary_kd"], expected[1])
        torch.testing.assert_close(result["total"], expected[0] + edge_weight * expected[1])
        result["total"].backward()
        for key in ("stage_logits", "stage_boundaries"):
            for s, t in zip(student[key], teacher[key]):
                self.assertGreater(s.grad.abs().sum().item(), 0)
                self.assertIsNone(t.grad)
        self.assertIsNone(student["coarse_logits"].grad)

    def test_wrong_teacher_suppresses_loss_without_renormalizing_reliable_mass(self):
        # Equally sized KL errors but different source-GT correctness. Correct
        # weights should be sigmoid(4), wrong weights sigmoid(-4), not both 1.
        mask = torch.ones_like(self.mask)
        student = {"stage_logits": (torch.zeros_like(mask, requires_grad=True),) * 3,
                   "stage_boundaries": (torch.zeros_like(mask, requires_grad=True),) * 3}
        correct = {"stage_logits": (torch.full_like(mask, 4.),) * 3,
                   "stage_boundaries": (torch.zeros_like(mask),) * 3}
        wrong = {**correct, "stage_logits": (torch.full_like(mask, -4.),) * 3}
        good = output_distillation(student, correct, mask)["mask_kd"]
        bad = output_distillation(student, wrong, mask)["mask_kd"]
        torch.testing.assert_close(bad / good, torch.exp(torch.tensor(-4.)), atol=1e-6, rtol=1e-5)

    def test_identical_teacher_is_zero_and_extreme_logits_remain_finite_under_amp(self):
        for magnitude in (0., 100., 10000.):
            student = self.output()
            for key in ("stage_logits", "stage_boundaries"):
                student[key] = tuple(torch.full_like(self.mask, magnitude, requires_grad=True) for _ in range(3))
            teacher = {key: tuple(x.detach() for x in student[key]) for key in ("stage_logits", "stage_boundaries")}
            with torch.autocast("cpu", dtype=torch.bfloat16):
                same = output_distillation(student, teacher, self.mask)
                opposite = output_distillation(student, {key: tuple(-x for x in value)
                    for key, value in teacher.items()}, self.mask)
            self.assertEqual(same["total"].item(), 0)
            self.assertEqual(opposite["total"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(opposite["total"]))
            opposite["total"].backward()
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in student["stage_logits"]))

    def test_empty_and_full_mask_and_native_resolutions_are_supported(self):
        for label in (0., 1.):
            student, teacher = self.output(), self.output()
            student["stage_logits"] = tuple(torch.randn(2, 1, side, side, requires_grad=True) for side in (2, 4, 8))
            student["logits"] = student["stage_logits"][-1]
            mask = torch.full_like(self.mask, label)
            result = supervised_loss(student, mask)
            distill = output_distillation(student, teacher, mask)
            self.assertTrue(torch.isfinite(result["total"]))
            self.assertTrue(torch.isfinite(distill["total"]))
            torch.testing.assert_close(result["total"], supervised_loss(student, mask * 255)["total"])

    def test_invalid_parameters_and_missing_stages_fail(self):
        student, teacher = self.output(), self.output()
        for options in ({"temperature": 0}, {"temperature": float("nan")}, {"boundary_weight": -1}):
            with self.assertRaises(ValueError):
                output_distillation(student, teacher, self.mask, **options)
        with self.assertRaises(ValueError):
            supervised_loss(student, self.mask, aux_weight=-1)
        with self.assertRaises(ValueError):
            output_distillation(student, {**teacher, "stage_logits": ()}, self.mask)


if __name__ == "__main__":
    unittest.main()
