"""Independent reduction/gradient oracles and real tiny source validation."""
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch

from models.triprd_correction import CorrectionStudent, CorrectionTeacher
from tools import run_triprd_correction as runner
from tools.diagnose_triprd_correction_probe import positive_mse_loss
from tools.correction_path_diagnostics import DiagnosticAccumulator, _inspect_relations
from tools.stratified_correction_loss import (
    POSITIVE_BCE_FIELDS, STRATA, _inspect_positive_relation_errors,
    evaluate_source_detailed, positive_stratified_mse_loss, source_summary,
)


def explicit_edges(mask, height, width):
    """Independent nearest-label and directed-neighbor enumeration, without unfold."""
    source_h, source_w = mask.shape[-2:]
    threshold = 127.5 if float(mask.max()) > 1 else .5
    for sample in range(mask.shape[0]):
        for y in range(height):
            for x in range(width):
                centre = bool(mask[sample, 0, y * source_h // height, x * source_w // width] >= threshold)
                for offset in range(9):
                    ny, nx = y + offset // 3 - 1, x + offset % 3 - 1
                    if offset == 4 or not (0 <= ny < height and 0 <= nx < width):
                        continue
                    nearby = bool(mask[sample, 0, ny * source_h // height, nx * source_w // width] >= threshold)
                    name = "fg" if centre and nearby else "bg" if not centre and not nearby else "crossing"
                    yield (sample, 0, offset, y, x), name


class StratifiedObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)

    def fixture(self):
        mask = torch.zeros(3, 1, 12, 16)
        mask[0, :, :9, :12] = 255
        mask[1, :, :5, :7] = 255
        mask[2, :, :3, 8:] = 255
        output = {"corrections": [], "relation_valid_masks": []}
        targets = {"delta": [], "valid": [], "reliability": []}
        coordinates = {name: [] for name in STRATA}
        for scale, (height, width) in enumerate(((2, 3), (4, 5))):
            shape = (3, 1, 9, height, width)
            prediction = torch.full(shape, 10000.)
            delta = torch.full(shape, -.125)
            reliability = torch.ones(shape)
            student_valid = torch.ones(1, 1, 9, height, width, dtype=torch.bool)
            teacher_valid = torch.ones_like(student_valid)
            student_valid[:, :, 5, 0, 0] = False
            teacher_valid[:, :, 7, 0, 1] = False
            for coordinate, name in explicit_edges(mask, height, width):
                sample, channel, offset, y, x = coordinate
                reliability[coordinate] = (-1., 0., .001, 9.)[(sample + offset + y + x) % 4]
                prediction[coordinate] = .1 * (STRATA.index(name) + 1) + .35 * scale + .07 * sample
                if (student_valid[0, channel, offset, y, x] and teacher_valid[0, channel, offset, y, x]
                        and reliability[coordinate] > 0):
                    coordinates[name].append((scale, coordinate))
            output["corrections"].append(prediction.requires_grad_())
            output["relation_valid_masks"].append(student_valid)
            targets["delta"].append(delta.requires_grad_())
            targets["valid"].append(teacher_valid)
            targets["reliability"].append(reliability.requires_grad_())
        return output, targets, mask, coordinates

    def test_manual_minibatch_pooling_and_gradient_change_only_stratum_weights(self):
        output, targets, mask, coordinates = self.fixture()
        sums = {
            name: sum(float((output["corrections"][scale].detach()[coordinate]
                             - targets["delta"][scale].detach()[coordinate]).square())
                      for scale, coordinate in edges)
            for name, edges in coordinates.items()
        }
        count = sum(len(edges) for edges in coordinates.values())
        self.assertTrue(all(coordinates.values()))
        self.assertGreater(len(set(len(edges) for edges in coordinates.values())), 1)
        expected = sum(sums[name] / len(edges) for name, edges in coordinates.items()) / 3
        pooled = positive_mse_loss(output, targets, mask)
        value = positive_stratified_mse_loss(output, targets, mask)
        self.assertAlmostEqual(float(value.detach()), expected, places=7)
        self.assertAlmostEqual(float(pooled.detach()), sum(sums.values()) / count, places=7)
        self.assertGreater(abs(float(value.detach() - pooled.detach())), .005)
        pooled_gradients = torch.autograd.grad(pooled, output["corrections"], retain_graph=True)
        value.backward()
        expected_gradients = [torch.zeros_like(raw) for raw in output["corrections"]]
        for name, edges in coordinates.items():
            for scale, coordinate in edges:
                error = output["corrections"][scale].detach()[coordinate] - targets["delta"][scale].detach()[coordinate]
                expected_gradients[scale][coordinate] = 2 * error / (3 * len(edges))
                self.assertAlmostEqual(float(output["corrections"][scale].grad[coordinate]),
                                       float(pooled_gradients[scale][coordinate]) * count / (3 * len(edges)), places=7)
        for prediction, expected_gradient in zip(output["corrections"], expected_gradients):
            torch.testing.assert_close(prediction.grad, expected_gradient)
        self.assertTrue(all(value.grad is None for value in targets["delta"] + targets["reliability"]))

    def test_pooling_is_not_equal_per_scale_or_per_image(self):
        output, targets, mask, coordinates = self.fixture()
        region_scale_means, region_image_means = [], []
        for edges in coordinates.values():
            for key, groups, results in ((0, range(2), region_scale_means), (1, range(3), region_image_means)):
                means = []
                for group in groups:
                    selected = [(scale, coordinate) for scale, coordinate in edges
                                if (scale if key == 0 else coordinate[0]) == group]
                    if selected:
                        means.append(sum(float((output["corrections"][scale].detach()[coordinate]
                                                - targets["delta"][scale].detach()[coordinate]).square())
                                         for scale, coordinate in selected) / len(selected))
                results.append(sum(means) / len(means))
        actual = float(positive_stratified_mse_loss(output, targets, mask).detach())
        self.assertGreater(abs(actual - sum(region_scale_means) / 3), .01)
        self.assertGreater(abs(actual - sum(region_image_means) / 3), .001)

    def test_positive_reliability_magnitude_does_not_weight_selected_edges(self):
        output, targets, mask, _ = self.fixture()
        expected = positive_stratified_mse_loss(output, targets, mask)
        changed = dict(targets, reliability=[torch.where(value > 0, 1., value) for value in targets["reliability"]])
        torch.testing.assert_close(positive_stratified_mse_loss(output, changed, mask), expected, rtol=0, atol=0)

    def test_one_nonempty_stratum_matches_existing_pooled_loss_and_empty_is_differentiable(self):
        for kind in ("background", "foreground", "no_positive", "no_valid"):
            with self.subTest(kind=kind):
                prediction = torch.randn(2, 1, 9, 3, 4, requires_grad=True)
                mask = torch.full((2, 1, 12, 16), float(kind == "foreground"))
                targets = {"delta": [torch.ones_like(prediction, requires_grad=True)],
                           "valid": [torch.full((1, 1, 9, 3, 4), kind != "no_valid", dtype=torch.bool)],
                           "reliability": [torch.full_like(prediction, float(kind != "no_positive"), requires_grad=True)]}
                output = {"corrections": [prediction]}
                value = positive_stratified_mse_loss(output, targets, mask)
                torch.testing.assert_close(value, positive_mse_loss(output, targets, mask), rtol=0, atol=0)
                value.backward()
                if kind.startswith("no_"):
                    self.assertEqual(float(value.detach()), 0.)
                    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
                self.assertIsNone(targets["delta"][0].grad)
                self.assertIsNone(targets["reliability"][0].grad)

    def test_mismatched_scales_and_grids_fail_explicitly(self):
        output, targets, mask, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "scale counts"):
            positive_stratified_mse_loss(output, dict(targets, valid=targets["valid"][:1]), mask)
        wrong = list(targets["delta"])
        wrong[0] = wrong[0][..., :1]
        with self.assertRaisesRegex(ValueError, "grids"):
            positive_stratified_mse_loss(output, dict(targets, delta=wrong), mask)

    def test_positive_bce_matches_independent_edge_oracle_and_exact_mse_support(self):
        output, targets, mask, coordinates = self.fixture()
        output.update(base_relation_logits=[], relation_logits=[], clipped_corrections=[])
        targets.update(target=[], base_probability=[], full_probability=[])
        expected = {}
        constants = {"all": [torch.tensor(0.), torch.tensor(0.)],
                     "positive": [torch.tensor(0.), torch.tensor(0.)]}
        for scale, raw in enumerate(output["corrections"]):
            logits = torch.full_like(raw, -.8 + .4 * scale)
            base = logits.sigmoid()
            corrected = (base + raw.detach() * .25).clamp(1e-6, 1 - 1e-6)
            output["base_relation_logits"].append(logits)
            output["relation_logits"].append(corrected.logit())
            output["clipped_corrections"].append(corrected - base)
            teacher_base = torch.full_like(raw, .35 + .07 * scale)
            targets["base_probability"].append(teacher_base)
            targets["full_probability"].append(teacher_base + targets["delta"][scale].detach())
            labels = torch.zeros_like(raw)
            for coordinate, name in explicit_edges(mask, *raw.shape[-2:]):
                labels[coordinate] = float(name != "crossing")
            targets["target"].append(labels)

        def scalar_bce(value, label):
            lower, upper = float(torch.tensor(1e-6)), float(torch.tensor(1 - 1e-6))
            probability = min(upper, max(lower, float(value)))
            return -math.log(probability if label else 1 - probability)

        for stratum, edges in coordinates.items():
            for scale, coordinate in edges:
                raw = output["corrections"][scale].detach()[coordinate]
                base = output["base_relation_logits"][scale][coordinate].sigmoid()
                corrected = output["relation_logits"][scale][coordinate].sigmoid()
                teacher_base = targets["base_probability"][scale][coordinate]
                probabilities = (base, corrected, teacher_base,
                                 targets["full_probability"][scale][coordinate],
                                 teacher_base + raw, base + targets["delta"][scale].detach()[coordinate])
                for scale_name in (f"stride{8 * 2 ** scale}", "pooled"):
                    for region in (stratum, "all"):
                        prefix = f"VDT/recovery/{scale_name}/{region}/"
                        for field, probability in zip(POSITIVE_BCE_FIELDS, probabilities):
                            expected.setdefault(prefix + field, []).append(scalar_bce(probability, stratum != "crossing"))
        accumulator = DiagnosticAccumulator()
        with torch.no_grad():
            _inspect_relations(output, mask, 3, accumulator, "VDT", targets, constants)
            _inspect_positive_relation_errors(output, targets, mask, 3, accumulator)
        actual = accumulator.result()
        for name, values in expected.items():
            self.assertEqual(actual[name]["count"], len(values), name)
            self.assertAlmostEqual(actual[name]["mean"], sum(values) / len(values), places=6, msg=name)
        for scale in ("stride8", "stride16", "pooled"):
            for region in ("all", *STRATA):
                prefix = f"VDT/recovery/{scale}/{region}/"
                for field in POSITIVE_BCE_FIELDS:
                    self.assertEqual(actual[prefix + field]["count"], actual[prefix + "positive_mse"]["count"])
        targets["reliability"] = [torch.zeros_like(raw) for raw in output["corrections"]]
        empty = DiagnosticAccumulator()
        _inspect_positive_relation_errors(output, targets, mask, 3, empty)
        for row in empty.result().values():
            self.assertEqual(row, {"mean": None, "sum": 0., "count": 0, "max": None})


class SourceDetailedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)

    def setUp(self):
        torch.manual_seed(53)
        self.student = CorrectionStudent(model_variant="tiny", base_channels=4, pretrained=False)
        self.teacher = CorrectionTeacher(model_variant="tiny", base_channels=4, pretrained=False)
        self.constants = {"all": [torch.tensor(.01), torch.tensor(-.01)],
                          "positive": [torch.tensor(.02), torch.tensor(-.02)]}
        self.args = SimpleNamespace(amp=False, relation_kernel=3)
        self.device = torch.device("cpu")
        self.loader = []
        for count in (2, 1):
            mask = torch.zeros(count, 1, 33, 41)
            mask[:, :, :24, :29] = 1
            self.loader.append({"rgb": torch.randn(count, 3, 33, 41),
                                "thermal": torch.randn(count, 3, 33, 41),
                                "depth": torch.rand(count, 1, 33, 41) * .05 + .01,
                                "mask": mask})

    def evaluate(self):
        return evaluate_source_detailed(self.student, self.teacher, self.loader,
                                        self.constants, self.device, self.args)

    def test_original_probe_agreement_and_single_normal_forward_per_batch(self):
        expected = runner._student_probe(self.student, self.teacher, self.loader,
                                          self.constants, self.device, self.args)
        calls = []
        handle = self.student.register_forward_hook(lambda *args: calls.append(1))
        try:
            with patch("models.triprd_correction_loss.constant_correction_statistics", side_effect=AssertionError("refit")), \
                 patch("models.triprd_correction_loss.finalize_constant_correction", side_effect=AssertionError("refit")):
                actual = self.evaluate()
        finally:
            handle.remove()
        self.assertEqual(len(calls), len(self.loader))
        self.assertEqual(actual["source_recoverability"], expected)
        self.assertFalse(any("intervention/" in key for key in actual["path_statistics"]))
        summary = source_summary(actual)
        self.assertEqual(set(summary), {"global", *STRATA})
        self.assertAlmostEqual(summary["global"]["positive_mse"], expected["positive_mse"], places=7)
        self.assertEqual(summary["global"]["positive_count"], expected["positive_count"])
        self.assertEqual(sum(summary[name]["count"] for name in STRATA), summary["global"]["count"])
        self.assertEqual(sum(summary[name]["positive_count"] for name in STRATA), summary["global"]["positive_count"])
        self.assertAlmostEqual(sum(summary[name]["full_share"] for name in STRATA), 1.)
        self.assertAlmostEqual(sum(summary[name]["positive_share"] for name in STRATA), 1.)
        for row in summary.values():
            self.assertIn("student_base_plus_teacher_delta_bce", row)
            self.assertAlmostEqual(row["positive_coverage"], row["positive_count"] / row["count"])
            self.assertAlmostEqual(row["bce_improvement"], row["base_bce"] - row["corrected_bce"], places=6)
            self.assertTrue(all(field in row for field in POSITIVE_BCE_FIELDS))
        for scale in ("stride8", "stride16", "pooled"):
            for region in ("all", *STRATA):
                prefix = f"VDT/recovery/{scale}/{region}/"
                for field in POSITIVE_BCE_FIELDS:
                    self.assertEqual(actual["path_statistics"][prefix + field]["count"],
                                     actual["path_statistics"][prefix + "positive_mse"]["count"])

    def test_evaluation_restores_flags_and_preserves_weights_gradients_and_constants(self):
        self.student.train()
        self.teacher.train()
        self.student.task_heads["cod"].eval()
        flags = [[module.training for module in root.modules()] for root in (self.student, self.teacher)]
        states = [{name: value.clone() for name, value in root.state_dict().items()}
                  for root in (self.student, self.teacher)]
        parameter = next(self.student.correction_heads.parameters())
        parameter.grad = torch.full_like(parameter, .123)
        gradient = parameter.grad.clone()
        constants = {name: [value.clone() for value in values] for name, values in self.constants.items()}
        self.evaluate()
        for root, before_flags, before_state in zip((self.student, self.teacher), flags, states):
            self.assertEqual([module.training for module in root.modules()], before_flags)
            for name, value in root.state_dict().items():
                torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.teacher.parameters()))
        for name, values in self.constants.items():
            for value, before in zip(values, constants[name]):
                torch.testing.assert_close(value, before, rtol=0, atol=0)

    def test_outer_cpu_autocast_uses_safe_float_probability_bce(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = self.evaluate()
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in actual["source_recoverability"].values()))
        for row in actual["path_statistics"].values():
            if row["count"]:
                self.assertTrue(torch.isfinite(torch.tensor(row["mean"])))

    def test_empty_loader_has_explicit_failure_and_restores_training_flags(self):
        self.loader = []
        self.student.train()
        flags = [module.training for module in self.student.modules()]
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            self.evaluate()
        self.assertEqual([module.training for module in self.student.modules()], flags)


if __name__ == "__main__":
    unittest.main()
