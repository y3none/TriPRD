"""Independent tensor checks for the read-only correction-path inspection."""
from contextlib import ExitStack
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.triprd_correction import CorrectionStudent, CorrectionTeacher
from tools.correction_path_diagnostics import DiagnosticAccumulator, inspect_batch


MODES = ("none", "no_correction", "shuffle_correction", "identity")
STRATA = ("fg", "bg", "crossing")


def explicit_edges(mask, height, width, *, valid=None):
    """Enumerate directed neighbors without calling the production edge helpers."""
    source_h, source_w = mask.shape[-2:]
    for sample in range(mask.shape[0]):
        for y in range(height):
            for x in range(width):
                centre = bool(mask[sample, 0, y * source_h // height, x * source_w // width] >= .5)
                for offset in range(9):
                    dy, dx = offset // 3 - 1, offset % 3 - 1
                    ny, nx = y + dy, x + dx
                    if offset == 4 or not (0 <= ny < height and 0 <= nx < width):
                        continue
                    if valid is not None and not bool(valid[0, 0, offset, y, x]):
                        continue
                    nearby = bool(mask[sample, 0, ny * source_h // height, nx * source_w // width] >= .5)
                    stratum = "fg" if centre and nearby else "bg" if not centre and not nearby else "crossing"
                    yield (sample, 0, offset, y, x), stratum, centre == nearby


class AccumulatorTests(unittest.TestCase):
    def test_weighted_streaming_reduction_uses_elements_not_batch_means(self):
        value = torch.tensor([1., -2., 4., 9., 3.], dtype=torch.float64)
        selected = torch.tensor([True, True, False, True, True])
        full, split = DiagnosticAccumulator(), DiagnosticAccumulator()
        full.add("probe", value, selected)
        for start, end in ((0, 1), (1, 4), (4, 5)):
            split.add("probe", value[start:end], selected[start:end])
        expected = {"mean": 2.75, "sum": 11., "count": 4, "max": 9.}
        self.assertEqual(full.result()["probe"], expected)
        self.assertEqual(split.result(), full.result())

    def test_empty_support_is_missing_and_invalid_nonfinite_values_are_excluded(self):
        acc = DiagnosticAccumulator()
        acc.add("empty", torch.tensor([1., 2.]), torch.tensor([False, False]))
        self.assertEqual(acc.result()["empty"], {"mean": None, "sum": 0., "count": 0, "max": None})
        acc.add("masked", torch.tensor([2., float("nan"), float("inf")]), torch.tensor([True, False, False]))
        self.assertEqual(acc.result()["masked"], {"mean": 2., "sum": 2., "count": 1, "max": 2.})

    def test_selected_nonfinite_values_fail_instead_of_silently_dropping_edges(self):
        with self.assertRaises((ValueError, FloatingPointError)):
            acc = DiagnosticAccumulator()
            acc.add("bad", torch.tensor([1., float("nan")]))
            acc.result()


class CorrectionPathDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        torch.manual_seed(97)
        self.rgb = torch.randn(3, 3, 33, 41)
        self.mask = torch.zeros(3, 1, 33, 41)
        self.mask[0, :, 7:29, 9:34] = 1
        self.mask[1, :, :19, 17:] = 1
        self.mask[2, :, 10:, :24] = 1
        self.model = CorrectionStudent(model_variant="tiny", base_channels=4, pretrained=False).eval()

    def learned_correction(self):
        with torch.no_grad():
            for head in self.model.correction_heads:
                nn.init.normal_(head.edge[-1].weight, std=.2)
                head.edge[-1].bias.fill_(.1)

    def inspect(self, **kwargs):
        acc = DiagnosticAccumulator()
        outputs = inspect_batch(self.model, self.rgb, self.mask, "cod", acc, "COD", **kwargs)
        return outputs, acc.result()

    def test_all_four_modes_match_direct_forward_for_real_tiny_model(self):
        self.learned_correction()
        outputs, _ = self.inspect()
        self.assertEqual(set(outputs), set(MODES))
        with torch.no_grad():
            for mode in MODES:
                direct = self.model(self.rgb, task="cod", intervention=mode)
                torch.testing.assert_close(outputs[mode]["logits"], direct["logits"], rtol=0, atol=0)
                for actual, expected in zip(outputs[mode]["propagation_probs"], direct["propagation_probs"]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_identity_is_exactly_bypassing_both_propagation_modules(self):
        self.learned_correction()
        outputs, _ = self.inspect()
        with ExitStack() as stack:
            for propagation in self.model.decoder.propagation:
                stack.enter_context(patch.object(propagation, "forward", side_effect=lambda feature, probability, valid: feature))
            with torch.no_grad():
                independent = self.model(self.rgb, task="cod")["logits"]
        torch.testing.assert_close(outputs["identity"]["logits"], independent, rtol=0, atol=0)

    def test_zero_correction_has_exactly_zero_removal_and_shuffle_effect(self):
        outputs, metrics = self.inspect()
        for mode in ("no_correction", "shuffle_correction"):
            torch.testing.assert_close(outputs["none"]["logits"], outputs[mode]["logits"], rtol=0, atol=0)
            for field in ("logits_abs_diff", "prob_abs_diff", "decoder_abs_diff"):
                row = metrics[f"COD/intervention/{mode}/{field}"]
                self.assertEqual(row["mean"], 0.)
                self.assertGreater(row["count"], 0)
        self.assertGreater(metrics["COD/intervention/identity/logits_abs_diff"]["mean"], 0.)

    def test_saturated_student_distinguishes_raw_and_deployed_correction(self):
        with torch.no_grad():
            for head in self.model.base_heads:
                head.edge[-1].weight.zero_()
                head.edge[-1].bias.fill_(8.)
            for head in self.model.correction_heads:
                head.edge[-1].weight.zero_()
                head.edge[-1].bias.fill_(.7)
        outputs, metrics = self.inspect()
        for scale in ("stride8", "stride16", "pooled"):
            prefix = f"COD/relation/{scale}/all/"
            self.assertAlmostEqual(metrics[prefix + "raw_abs"]["mean"], math.tanh(.7), places=6)
            self.assertLess(metrics[prefix + "clipped_abs"]["mean"], .001)
            self.assertEqual(metrics[prefix + "clip_fraction"]["mean"], 1.)
        self.assertGreater((outputs["none"]["logits"] - outputs["no_correction"]["logits"]).abs().max().item(), 0.)

    def test_strata_counts_partition_only_valid_off_diagonal_edges(self):
        outputs, metrics = self.inspect()
        pooled = dict.fromkeys(STRATA, 0)
        for scale, correction in zip(("stride8", "stride16"), outputs["none"]["corrections"]):
            expected = dict.fromkeys(STRATA, 0)
            for _, stratum, _ in explicit_edges(self.mask, *correction.shape[-2:]):
                expected[stratum] += 1
                pooled[stratum] += 1
            total = sum(expected.values())
            self.assertLess(total, correction.numel())
            self.assertEqual(metrics[f"COD/relation/{scale}/all/raw_abs"]["count"], total)
            for stratum, count in expected.items():
                self.assertEqual(metrics[f"COD/relation/{scale}/{stratum}/raw_abs"]["count"], count)
        self.assertEqual(metrics["COD/relation/pooled/all/raw_abs"]["count"], sum(pooled.values()))
        for stratum, count in pooled.items():
            self.assertEqual(metrics[f"COD/relation/pooled/{stratum}/raw_abs"]["count"], count)

    def test_same_samples_partitioned_into_different_batches_have_same_metrics(self):
        self.learned_correction()
        _, expected = self.inspect()
        accumulated = DiagnosticAccumulator()
        for indices in (slice(0, 1), slice(1, 3)):
            inspect_batch(self.model, self.rgb[indices], self.mask[indices], "cod", accumulated, "COD")
        actual = accumulated.result()
        self.assertEqual(actual.keys(), expected.keys())
        for name, expected_row in expected.items():
            self.assertEqual(actual[name]["count"], expected_row["count"], name)
            for field in ("mean", "sum", "max"):
                if expected_row[field] is None:
                    self.assertIsNone(actual[name][field], name)
                else:
                    self.assertTrue(math.isclose(actual[name][field], expected_row[field], rel_tol=3e-5, abs_tol=3e-5),
                                    (name, field, actual[name][field], expected_row[field]))

    def test_all_background_has_no_foreground_or_crossing_support(self):
        self.mask.zero_()
        _, metrics = self.inspect()
        for scale in ("stride8", "stride16", "pooled"):
            for stratum in ("fg", "crossing"):
                row = metrics[f"COD/relation/{scale}/{stratum}/base_bce"]
                self.assertEqual(row["count"], 0)
                self.assertIsNone(row["mean"])
                self.assertIsNone(row["max"])

    def test_inspection_preserves_parameters_buffers_gradients_and_existing_hooks(self):
        self.learned_correction()
        self.model.train()
        existing_handle = self.model.decoder.register_forward_hook(lambda *args: None)
        self.addCleanup(existing_handle.remove)
        before_state = {name: value.clone() for name, value in self.model.state_dict().items()}
        before_training = {name: module.training for name, module in self.model.named_modules()}
        before_hooks = {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                        for name, module in self.model.named_modules()}
        parameter = next(self.model.correction_heads.parameters())
        parameter.grad = torch.full_like(parameter, .3)
        before_gradient = parameter.grad.clone()
        self.inspect()
        self.assertEqual(before_training, {name: module.training for name, module in self.model.named_modules()})
        self.assertEqual(before_hooks, {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                                       for name, module in self.model.named_modules()})
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, before_gradient, rtol=0, atol=0)

    def test_inspection_removes_temporary_hooks_when_a_forward_fails(self):
        before = {name: tuple(module._forward_hooks) for name, module in self.model.named_modules()}
        with patch.object(self.model.task_heads["cod"], "forward", side_effect=RuntimeError("injected forward failure")):
            with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
                self.inspect()
        self.assertEqual(before, {name: tuple(module._forward_hooks) for name, module in self.model.named_modules()})

    def source_fixture(self):
        """Vary support, sign, scale and offset so pooled means cannot hide bugs."""
        with torch.no_grad():
            output = self.model(self.rgb, task="sod")
        fields = {name: [] for name in ("delta", "valid", "reliability", "target", "base_probability", "full_probability")}
        constants = {"all": [], "positive": []}
        expected = {}
        for scale_index, (scale, correction) in enumerate(zip(("stride8", "stride16"), output["corrections"])):
            shape = correction.shape
            delta = torch.full_like(correction, 7.)
            reliability = torch.ones_like(correction)
            valid = output["relation_valid_masks"][scale_index].clone()
            # Exclude one real neighbor independently of geometric padding.
            first = next(explicit_edges(self.mask[:1], *shape[-2:]))[0]
            valid[first] = False
            target = torch.zeros_like(correction)
            base = torch.full_like(correction, .5)
            full = base.clone()
            all_constant = (torch.arange(9).float() * .01 - .04).reshape(1, 1, 9, 1, 1)
            positive_constant = (torch.arange(9).float() * -.02 + .09).reshape(1, 1, 9, 1, 1)
            constants["all"].append(all_constant)
            constants["positive"].append(positive_constant)
            for coordinate, stratum, agrees in explicit_edges(self.mask, *shape[-2:], valid=valid):
                sample, _, offset, y, x = coordinate
                value = (.1 + .07 * scale_index + .01 * offset) * (1 if agrees else -1)
                delta[coordinate] = value
                target[coordinate] = float(agrees)
                full[coordinate] = .5 + value
                positive = (sample + offset + y + x) % 3 != 0
                reliability[coordinate] = .25 if positive else 0.
                for scale_key in (scale, "pooled"):
                    for stratum_key in (stratum, "all"):
                        prefix = f"VDT/recovery/{scale_key}/{stratum_key}/"
                        for metric, sample_value in (("mse", value * value), ("zero_mse", value * value),
                                                     ("constant_mse", (float(all_constant[0, 0, offset, 0, 0]) - value) ** 2)):
                            expected.setdefault(prefix + metric, []).append(sample_value)
                        if positive:
                            for metric, sample_value in (("positive_mse", value * value), ("positive_zero_mse", value * value),
                                                         ("positive_constant_mse", (float(positive_constant[0, 0, offset, 0, 0]) - value) ** 2)):
                                expected.setdefault(prefix + metric, []).append(sample_value)
            for name, value in (("delta", delta), ("valid", valid), ("reliability", reliability),
                                ("target", target), ("base_probability", base), ("full_probability", full)):
                fields[name].append(value)
        return {name: tuple(value) for name, value in fields.items()}, {name: tuple(value) for name, value in constants.items()}, expected

    def test_source_mse_and_constants_match_independent_directed_edge_enumeration(self):
        targets, constants, expected = self.source_fixture()
        original = {key: tuple(value.clone() for value in values) for key, values in constants.items()}
        acc = DiagnosticAccumulator()
        with patch("models.triprd_correction_loss.constant_correction_statistics", side_effect=AssertionError("validation constants refitted")), \
             patch("models.triprd_correction_loss.finalize_constant_correction", side_effect=AssertionError("validation constants refitted")):
            inspect_batch(self.model, self.rgb, self.mask, "sod", acc, "VDT", targets=targets, constants=constants)
        metrics = acc.result()
        for name, values in expected.items():
            self.assertEqual(metrics[name]["count"], len(values), name)
            self.assertAlmostEqual(metrics[name]["sum"], sum(values), places=5, msg=name)
            self.assertAlmostEqual(metrics[name]["mean"], sum(values) / len(values), places=7, msg=name)
        for key, values in constants.items():
            for value, before in zip(values, original[key]):
                torch.testing.assert_close(value, before, rtol=0, atol=0)

    def test_zero_reliability_reports_missing_positive_mse(self):
        targets, constants, _ = self.source_fixture()
        targets["reliability"] = tuple(torch.zeros_like(value) for value in targets["reliability"])
        acc = DiagnosticAccumulator()
        inspect_batch(self.model, self.rgb, self.mask, "sod", acc, "VDT", targets=targets, constants=constants)
        for scale in ("stride8", "stride16", "pooled"):
            row = acc.result()[f"VDT/recovery/{scale}/all/positive_mse"]
            self.assertEqual(row["count"], 0)
            self.assertIsNone(row["mean"])
            self.assertIsNone(row["max"])

    def test_real_teacher_path_accepts_fixed_constants_without_changing_teacher(self):
        teacher = CorrectionTeacher(model_variant="tiny", base_channels=4, pretrained=False).eval()
        before = {name: value.clone() for name, value in teacher.state_dict().items()}
        constants = {"all": (torch.tensor(.01), torch.tensor(-.01)),
                     "positive": (torch.tensor(.02), torch.tensor(-.02))}
        acc = DiagnosticAccumulator()
        inspect_batch(self.model, self.rgb, self.mask, "sod", acc, "VDT", teacher=teacher,
                      thermal=torch.rand_like(self.rgb), depth=torch.rand_like(self.mask) * .05 + .01,
                      constants=constants)
        self.assertGreater(acc.result()["VDT/recovery/pooled/all/mse"]["count"], 0)
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_source_diagnostics_remain_finite_under_cpu_bfloat16_autocast(self):
        teacher = CorrectionTeacher(model_variant="tiny", base_channels=4, pretrained=False).eval()
        constants = {"all": (torch.tensor(.01), torch.tensor(-.01)),
                     "positive": (torch.tensor(.02), torch.tensor(-.02))}
        acc = DiagnosticAccumulator()
        calls = []
        handle = self.model.encoder.register_forward_hook(lambda *args: calls.append(1))
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                outputs = inspect_batch(self.model, self.rgb, self.mask, "sod", acc, "VDT", teacher=teacher,
                    thermal=torch.rand_like(self.rgb), depth=torch.rand_like(self.mask) * .05 + .01,
                    constants=constants)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1, "The four student modes should share one RGB feature extraction")
        self.assertTrue(all(torch.isfinite(output["logits"]).all() for output in outputs.values()))
        for name, row in acc.result().items():
            if row["count"]:
                self.assertTrue(math.isfinite(row["mean"]), name)


if __name__ == "__main__":
    unittest.main()
