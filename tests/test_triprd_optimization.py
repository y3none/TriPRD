"""CPU regression tests for optional KD ablations; run with unittest discover."""

import unittest

import torch
from torch.nn import functional as F

from models.triprd import TriPRDLoss, _category_indices, _gate_diagnostics, _relation_cosine_gram


def _fixture(size=13):
    torch.manual_seed(17)
    shape = (1, 1, size, size)
    mask = torch.zeros(shape)
    mask[:, :, 4:9, 4:9] = 1
    student = {
        "logits": torch.randn(shape, requires_grad=True),
        "boundary_logits": torch.randn(shape, requires_grad=True),
        "features": torch.randn(1, 3, size, size, requires_grad=True),
    }
    teacher = {
        "logits": torch.randn(shape, requires_grad=True),
        "boundary_logits": torch.randn(shape, requires_grad=True),
        "features": torch.randn(1, 3, size, size, requires_grad=True),
        "modality_logits": torch.randn(1, 3, size, size, requires_grad=True),
    }
    return student, teacher, mask


def _historical_full_loss(student, teacher, mask):
    """Frozen E001 formulas, independent of the new sampling/diagnostic code."""
    boundary = (F.max_pool2d(mask, 3, 1, 1) + F.max_pool2d(-mask, 3, 1, 1)).clamp(0, 1)
    pred = student["logits"].sigmoid()
    intersection = (pred * mask).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + mask.sum((1, 2, 3)) - intersection
    supervised = (F.binary_cross_entropy_with_logits(student["logits"], mask)
                  + (1 - (intersection + 1e-6) / (union + 1e-6)).mean()
                  + 0.5 * F.binary_cross_entropy_with_logits(student["boundary_logits"], boundary))
    gate = (1 - teacher["modality_logits"].sigmoid().var(1, keepdim=True, unbiased=False) / 0.25).clamp(0.1, 1).detach()
    q = (teacher["logits"].detach() / 4).sigmoid()
    entropy = -(q * q.clamp_min(1e-7).log() + (1 - q) * (1 - q).clamp_min(1e-7).log())
    kl = (F.binary_cross_entropy_with_logits(student["logits"] / 4, q, reduction="none") - entropy).clamp_min(0) * 16
    logit_kd = (kl * gate).mean()
    target = mask > 0.5
    categories = (target & ~(boundary > 0),
                  (F.max_pool2d(mask, 3, 1, 1) > 0.5) & ~target,
                  (boundary > 0) & target, (boundary > 0) & ~target)
    terms = []
    for category in categories:
        indices = category[0, 0].flatten().nonzero().flatten()
        if not indices.numel():
            indices = torch.arange(mask.shape[-2] * mask.shape[-1])
        indices = indices[torch.arange(8) % indices.numel()]
        s = F.normalize(student["features"][0].flatten(1).T[indices], dim=1, eps=1e-6)
        t = F.normalize(teacher["features"][0].flatten(1).T[indices].detach(), dim=1, eps=1e-6)
        points = gate[0, 0].flatten()[indices]
        pair = points[:, None] * points[None, :]
        terms.append((((s @ s.T) - (t @ t.T)).square() * pair).sum() / pair.sum().clamp_min(1e-6))
    relation_kd = torch.stack(terms).mean()
    boundary_kd = ((student["boundary_logits"] - teacher["boundary_logits"].detach()).square() * gate).mean()
    return supervised + logit_kd + relation_kd + 0.5 * boundary_kd


class TriPRDOptimizationTests(unittest.TestCase):
    def test_default_loss_and_student_gradients_match_historical_full(self):
        student, teacher, mask = _fixture()
        expected = _historical_full_loss(student, teacher, mask)
        actual = TriPRDLoss()(student, mask, teacher)["total"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        tensors = tuple(student.values())
        actual_grad = torch.autograd.grad(actual, tensors, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, tensors)
        for got, wanted in zip(actual_grad, expected_grad):
            # The frozen reference spells the same graph independently;
            # float32 backward accumulation may differ by a few ulps.
            torch.testing.assert_close(got, wanted, rtol=1e-5, atol=1e-8)

    def test_zero_kd_weights_require_no_teacher(self):
        student, _, mask = _fixture()
        student.pop("features")
        plain = TriPRDLoss(mode="none")(student, mask)
        disabled = TriPRDLoss(lambda_logit=0, lambda_relation=0, lambda_boundary_kd=0)(student, mask)
        torch.testing.assert_close(disabled["total"], plain["total"])
        for name in ("logit_kd", "relation_kd", "boundary_kd"):
            self.assertEqual(disabled[name].item(), 0)

    def test_individual_disabled_terms_do_not_read_teacher_fields(self):
        student, teacher, mask = _fixture()
        cases = (
            ({"lambda_relation": 0, "lambda_boundary_kd": 0}, {"logits": teacher["logits"]}),
            ({"lambda_logit": 0, "lambda_boundary_kd": 0}, {"features": teacher["features"]}),
            ({"lambda_logit": 0, "lambda_relation": 0}, {"boundary_logits": teacher["boundary_logits"]}),
        )
        for kwargs, targets in cases:
            with self.subTest(kwargs=kwargs):
                result = TriPRDLoss(gate_mode="none", **kwargs)(student, mask, targets)
                self.assertTrue(torch.isfinite(result["total"]).item())
        gated_targets = {"logits": teacher["logits"], "modality_logits": teacher["modality_logits"]}
        result = TriPRDLoss(lambda_relation=0, lambda_boundary_kd=0)(student, mask, gated_targets)
        self.assertTrue(torch.isfinite(result["total"]).item())

    def test_probability_boundary_loss_is_bounded_and_teacher_detached(self):
        student, _, mask = _fixture()
        student["boundary_logits"] = torch.full_like(mask, 100, requires_grad=True)
        teacher = {"boundary_logits": torch.full_like(mask, -100, requires_grad=True)}
        probability = TriPRDLoss(mode="boundary", boundary_kd_type="prob_mse")(student, mask, teacher)
        raw = TriPRDLoss(mode="boundary")(student, mask, teacher)
        self.assertGreater(raw["boundary_kd"].item(), 1000)
        self.assertGreaterEqual(probability["boundary_kd"].item(), 0)
        self.assertLessEqual(probability["boundary_kd"].item(), 1)
        self.assertTrue(torch.isfinite(probability["total"]).item())
        probability["total"].backward()
        self.assertTrue(torch.isfinite(student["boundary_logits"].grad).all().item())
        self.assertIsNone(teacher["boundary_logits"].grad)

    def test_probability_boundary_identical_predictions_have_zero_loss(self):
        student, _, mask = _fixture()
        teacher = {"boundary_logits": student["boundary_logits"].detach().clone()}
        result = TriPRDLoss(mode="boundary", boundary_kd_type="prob_mse")(student, mask, teacher)
        self.assertEqual(result["boundary_kd"].item(), 0)

    def test_uniform_regions_are_disjoint_with_no_empty_fallback(self):
        _, _, mask = _fixture()
        categories = _category_indices(mask, mask.shape[-2:], 1000, "uniform")[0]
        self.assertEqual([len(x) for x in categories], [9, 16, 24, 32])
        combined = torch.cat(categories)
        self.assertEqual(combined.unique().numel(), combined.numel())
        empty = _category_indices(torch.zeros_like(mask), mask.shape[-2:], 8, "uniform")[0]
        self.assertTrue(all(indices.numel() == 0 for indices in empty))

    def test_uniform_sampling_spans_regions_is_unique_and_deterministic(self):
        _, _, mask = _fixture()
        all_points = _category_indices(mask, mask.shape[-2:], 1000, "uniform")[0]
        rng_before = torch.random.get_rng_state().clone()
        sampled = _category_indices(mask, mask.shape[-2:], 4, "uniform")[0]
        again = _category_indices(mask, mask.shape[-2:], 4, "uniform")[0]
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for indices, full, repeated in zip(sampled, all_points, again):
            self.assertTrue(torch.equal(indices, repeated))
            self.assertEqual(indices.numel(), indices.unique().numel())
            self.assertEqual(indices[0].item(), full[0].item())
            self.assertEqual(indices[-1].item(), full[-1].item())
        single = torch.zeros_like(mask)
        single[:, :, 6, 6] = 1
        tiny = _category_indices(single, single.shape[-2:], 8, "uniform")[0]
        self.assertEqual(tiny[0].numel(), 0)
        self.assertEqual(tiny[1].numel(), 1)

    def test_uniform_gram_detects_cross_category_difference(self):
        _, _, mask = _fixture()
        teacher_features = torch.cat([torch.ones_like(mask), torch.zeros_like(mask)], dim=1)
        student_features = torch.cat([mask, 1 - mask], dim=1).requires_grad_()
        gate = torch.ones_like(mask)
        legacy = _relation_cosine_gram(student_features, teacher_features, mask, gate, 8)
        uniform = _relation_cosine_gram(student_features, teacher_features, mask, gate, 8, "uniform")
        self.assertEqual(legacy.item(), 0)
        self.assertGreater(uniform.item(), 0.1)
        uniform.backward()
        self.assertTrue(torch.isfinite(student_features.grad).all().item())

    def test_empty_uniform_relations_are_finite_zero(self):
        student, teacher, mask = _fixture()
        value = _relation_cosine_gram(student["features"], teacher["features"],
                                      torch.zeros_like(mask), torch.ones_like(mask), 8, "uniform")
        self.assertEqual(value.item(), 0)

    def test_gate_none_matches_fixed_and_all_teacher_gradients_stay_absent(self):
        student, teacher, mask = _fixture()
        teacher.pop("modality_logits")
        disabled_gate = TriPRDLoss(gate_mode="none")(student, mask, teacher)
        fixed = TriPRDLoss(mode="fixed")(student, mask, teacher)
        torch.testing.assert_close(disabled_gate["total"], fixed["total"], rtol=0, atol=0)
        for key in ("gate_mean", "gate_p10", "gate_p50", "gate_p90"):
            self.assertEqual(disabled_gate[key].item(), 1)
            self.assertFalse(disabled_gate[key].requires_grad)
        self.assertEqual(disabled_gate["gate_std"].item(), 0)
        disabled_gate["total"].backward()
        self.assertTrue(all(value.grad is None for value in teacher.values()))

    def test_weighted_components_reconstruct_total(self):
        student, teacher, mask = _fixture()
        result = TriPRDLoss(lambda_logit=0.2, lambda_relation=0.3, lambda_boundary_kd=0.4)(student, mask, teacher)
        torch.testing.assert_close(result["weighted_logit_kd"], 0.2 * result["logit_kd"])
        torch.testing.assert_close(result["weighted_relation_kd"], 0.3 * result["relation_kd"])
        torch.testing.assert_close(result["weighted_boundary_kd"], 0.4 * result["boundary_kd"])
        reconstructed = result["supervised"] + result["weighted_logit_kd"] + result["weighted_relation_kd"] + result["weighted_boundary_kd"]
        torch.testing.assert_close(result["total"], reconstructed, rtol=0, atol=0)

    def test_gate_diagnostics_sample_without_rng_or_gradients(self):
        values = torch.linspace(0.1, 1, 10000).reshape(1, 1, 100, 100).requires_grad_()
        rng_before = torch.random.get_rng_state().clone()
        result = _gate_diagnostics(values)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue(all(not value.requires_grad for value in result.values()))
        self.assertLessEqual(result["gate_p10"].item(), result["gate_p50"].item())
        self.assertLessEqual(result["gate_p50"].item(), result["gate_p90"].item())
        self.assertAlmostEqual(result["gate_mean"].item(), 0.55, places=5)

    def test_invalid_ablation_options_fail_early(self):
        for key in ("boundary_kd_type", "relation_sampling", "gate_mode"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                TriPRDLoss(**{key: "unsupported"})

    def test_loss_weights_must_be_finite_and_nonnegative(self):
        for key in ("lambda_iou", "lambda_boundary", "lambda_logit", "lambda_relation", "lambda_boundary_kd"):
            for value in (-1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    TriPRDLoss(**{key: value})
            TriPRDLoss(**{key: 0})

    def test_temperatures_must_be_finite_and_positive(self):
        for key in ("temperature", "calibration_temperature"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                    TriPRDLoss(**{key: value})


if __name__ == "__main__":
    unittest.main()
