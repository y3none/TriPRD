"""Exercise real GradScaler overflow recovery, clipping, and epoch reporting.

CPU tests intentionally put a float16 cast in the backward path while retaining
float32 parameters and a finite float32 forward loss. No scaler is mocked.
"""
from contextlib import redirect_stdout
import copy
import io
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from tools import run_triprd_correction as runner


class HalfBackwardModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., device=device))
        self.forward_calls = 0

    def forward(self, rgb=None, task=None):
        self.forward_calls += 1
        value = self.weight.to(torch.float16).float()
        return value if rgb is None else value * rgb.sum()


class ScalarDataset(Dataset):
    def __init__(self, values):
        self.values = values
        self.epoch = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"rgb": torch.tensor([self.values[index]], dtype=torch.float32),
                "mask": torch.zeros(1), "key": f"sample-{index}"}


def scalar_source_loss(output, mask, **kwargs):
    return {"total": output, "segmentation": output}


class CorrectionAMPTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right), (left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for lhs, rhs in zip(left, right):
                self.assert_nested_equal(lhs, rhs)
        else:
            self.assertEqual(left, right)

    @staticmethod
    def make_training(device="cpu", scale=65536., enabled=True):
        model = HalfBackwardModel(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
        scaler = torch.amp.GradScaler(device, init_scale=scale, enabled=enabled)
        return model, optimizer, scaler

    def backward(self, model, optimizer, scaler, *, nan=False, factor=1.):
        optimizer.zero_grad(set_to_none=True)
        # sqrt(0) has an infinite derivative; multiplication by zero then gives
        # a NaN gradient while the forward result remains exactly zero.
        loss = torch.sqrt(model.weight * 0) if nan else model() * factor
        self.assertTrue(torch.isfinite(loss).all())
        scaler.scale(loss).backward()

    def exercise_skip_and_recovery(self, device, nan):
        model, optimizer, scaler = self.make_training(device, scale=32768.)
        # Populate AdamW moments first so a skipped update must preserve both
        # existing moments and parameters, including decoupled weight decay.
        self.backward(model, optimizer, scaler)
        norm, event = runner._optimizer_step(model, optimizer, scaler, 1.)
        self.assertIsNone(event)
        self.assertTrue(torch.isfinite(norm))
        scaler.update(new_scale=65536.)
        previous_model = copy.deepcopy(model.state_dict())
        previous_optimizer = copy.deepcopy(optimizer.state_dict())
        self.backward(model, optimizer, scaler, nan=nan)
        self.assertFalse(torch.isfinite(model.weight.grad))
        norm, event = runner._optimizer_step(model, optimizer, scaler, 1.)
        self.assertIsNone(norm)
        self.assertEqual(event["scale_before"], 65536.)
        self.assertEqual(event["scale_after"], 32768.)
        self.assertEqual(event["nonfinite_parameters"], ["weight"])
        self.assertEqual(event["nonfinite_parameter_count"], 1)
        self.assert_nested_equal(model.state_dict(), previous_model)
        self.assert_nested_equal(optimizer.state_dict(), previous_optimizer)
        self.backward(model, optimizer, scaler)
        norm, event = runner._optimizer_step(model, optimizer, scaler, 1.)
        self.assertIsNone(event)
        self.assertAlmostEqual(float(norm), 1.)
        self.assertFalse(torch.equal(model.weight, previous_model["weight"]))
        self.assertEqual(float(optimizer.state[model.weight]["step"]), 2.)

    def test_real_cpu_scaler_skips_inf_and_nan_without_adamw_mutation(self):
        for nan in (False, True):
            with self.subTest(nonfinite="nan" if nan else "inf"):
                self.exercise_skip_and_recovery("cpu", nan)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a real CUDA device")
    def test_real_cuda_scaler_skips_and_recovers(self):
        for nan in (False, True):
            with self.subTest(nonfinite="nan" if nan else "inf"):
                self.exercise_skip_and_recovery("cuda", nan)

    def test_first_overflow_does_not_initialize_adamw_state(self):
        model, optimizer, scaler = self.make_training()
        self.backward(model, optimizer, scaler)
        before = model.weight.detach().clone()
        norm, event = runner._optimizer_step(model, optimizer, scaler, 1.)
        self.assertIsNone(norm)
        self.assertIsNotNone(event)
        self.assertEqual(optimizer.state, {})
        self.assertTrue(torch.equal(model.weight, before))

    def test_finite_gradients_are_unscaled_then_clipped(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=4.)
        scaler.scale(model(torch.tensor([[3., 4.]])).sum()).backward()
        norm, event = runner._optimizer_step(model, optimizer, scaler, 1.)
        self.assertIsNone(event)
        self.assertAlmostEqual(float(norm), 5.)
        torch.testing.assert_close(model.weight, torch.tensor([[-.06, -.08]]))
        torch.testing.assert_close(model.weight.grad, torch.tensor([[.6, .8]]))

    def test_disabled_scaler_fails_on_inf_or_nan_without_updating(self):
        for nan in (False, True):
            with self.subTest(nonfinite="nan" if nan else "inf"):
                model, optimizer, scaler = self.make_training(enabled=False)
                before = copy.deepcopy(model.state_dict())
                self.backward(model, optimizer, scaler, nan=nan, factor=70000.)
                with self.assertRaisesRegex(FloatingPointError, "without AMP scaling"):
                    runner._optimizer_step(model, optimizer, scaler, 1.)
                self.assert_nested_equal(model.state_dict(), before)
                self.assertEqual(optimizer.state, {})

    def test_norm_reduction_overflow_with_finite_gradients_never_steps(self):
        for enabled in (False, True):
            with self.subTest(amp=enabled):
                model = torch.nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    model.weight.zero_()
                optimizer = torch.optim.AdamW(model.parameters())
                scaler = torch.amp.GradScaler("cpu", enabled=enabled, init_scale=1.)
                loss = model(torch.full((1, 2), 3.e38)).sum()
                self.assertTrue(torch.isfinite(loss))
                scaler.scale(loss).backward()
                self.assertTrue(torch.isfinite(model.weight.grad).all())
                before = model.weight.grad.clone()
                with self.assertRaisesRegex(FloatingPointError, "norm with finite gradients"):
                    runner._optimizer_step(model, optimizer, scaler, 1.)
                self.assertTrue(torch.equal(model.weight.grad, before))
                self.assertTrue(torch.equal(model.weight, torch.zeros_like(model.weight)))
                self.assertEqual(optimizer.state, {})

    def test_missing_gradients_fail_before_optimizer_update(self):
        model, optimizer, scaler = self.make_training()
        unrelated = torch.tensor(1., requires_grad=True)
        scaler.scale(unrelated).backward()
        with self.assertRaisesRegex(FloatingPointError, "No trainable gradients"):
            runner._optimizer_step(model, optimizer, scaler, 1.)
        self.assertEqual(optimizer.state, {})

    def test_unscale_created_overflow_at_subunit_scale_never_steps(self):
        model, optimizer, scaler = self.make_training(scale=.25)
        before = model.weight.detach().clone()
        # The derivative is representable after scaling (.25 * 4e38), but
        # unscaling exceeds float32. GradScaler's fused inf check can miss
        # this because it checks the pre-unscale value.
        loss = model.weight.double() * 4.e38
        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()
        self.assertTrue(torch.isfinite(model.weight.grad))
        with self.assertRaisesRegex(FloatingPointError, "Unscaling finite gradients"):
            runner._optimizer_step(model, optimizer, scaler, 1.)
        self.assertFalse(torch.isfinite(model.weight.grad))
        self.assertTrue(torch.equal(model.weight, before))
        self.assertEqual(optimizer.state, {})

    def test_preexisting_overflow_still_skips_with_subunit_scale(self):
        for nan in (False, True):
            with self.subTest(nonfinite="nan" if nan else "inf"):
                model, optimizer, scaler = self.make_training(scale=.25)
                before = model.weight.detach().clone()
                self.backward(model, optimizer, scaler, nan=nan, factor=1.e6)
                self.assertFalse(torch.isfinite(model.weight.grad))
                norm, event = runner._optimizer_step(model, optimizer, scaler, 1.)
                self.assertIsNone(norm)
                self.assertEqual(event["scale_before"], .25)
                self.assertEqual(event["scale_after"], .125)
                self.assertTrue(torch.equal(model.weight, before))
                self.assertEqual(optimizer.state, {})

    def test_invalid_scaler_scale_is_rejected_without_updates(self):
        for scale in (0., -1., float("nan"), float("inf")):
            with self.subTest(scale=scale):
                model, optimizer, scaler = self.make_training(scale=scale)
                before = model.weight.detach().clone()
                self.backward(model, optimizer, scaler)
                with self.assertRaisesRegex(FloatingPointError, "Invalid gradient scale"):
                    runner._optimizer_step(model, optimizer, scaler, 1.)
                self.assertTrue(torch.equal(model.weight, before))
                self.assertEqual(optimizer.state, {})

    def test_serialized_scaler_and_optimizer_recover_identically(self):
        model, optimizer, scaler = self.make_training(scale=32768.)
        self.backward(model, optimizer, scaler)
        runner._optimizer_step(model, optimizer, scaler, 1.)
        scaler.update(new_scale=65536.)
        self.backward(model, optimizer, scaler)
        runner._optimizer_step(model, optimizer, scaler, 1.)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        restored_model, restored_optimizer, restored_scaler = self.make_training()
        restored_model.load_state_dict(state["model"])
        restored_optimizer.load_state_dict(state["optimizer"])
        restored_scaler.load_state_dict(state["scaler"])
        for subject, opt, scale in ((model, optimizer, scaler),
                                    (restored_model, restored_optimizer, restored_scaler)):
            self.backward(subject, opt, scale)
            norm, event = runner._optimizer_step(subject, opt, scale, 1.)
            self.assertIsNone(event)
            self.assertAlmostEqual(float(norm), 1.)
        self.assert_nested_equal(restored_model.state_dict(), model.state_dict())
        self.assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        self.assert_nested_equal(restored_scaler.state_dict(), scaler.state_dict())

    def epoch(self, values, model, optimizer, scaler):
        dataset = ScalarDataset(values)
        loaders = {"cod_train": DataLoader(dataset, batch_size=1)}
        args = SimpleNamespace(stage="student", group="A", amp=False,
                               relation_kernel=3, relation_weight=.2, grad_clip=1.)
        with patch("models.triprd_correction_loss.source_loss", side_effect=scalar_source_loss), \
                redirect_stdout(io.StringIO()):
            return runner._train_epoch(model, None, loaders, optimizer, scaler,
                                       torch.device("cpu"), args, 3)

    def test_epoch_reports_skips_separately_and_keeps_finite_json_statistics(self):
        model, optimizer, scaler = self.make_training()
        result = self.epoch([1., 1.], model, optimizer, scaler)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["amp_skipped_steps"], 1)
        self.assertTrue(math.isfinite(result["grad_norm"]))
        self.assertEqual(result["grad_norm"], 1.)
        self.assertEqual(result["amp_scale"], 32768.)
        event, = result["amp_overflow_events"]
        self.assertEqual(event["batch"], 1)
        self.assertEqual(event["primary_keys"], ["sample-0"])
        self.assertEqual(event["nonfinite_parameters"], ["weight"])
        self.assertEqual(event["scale_before"], 65536.)
        self.assertEqual(event["scale_after"], 32768.)
        json.dumps(result, allow_nan=False)

    def test_eight_consecutive_skips_fail_without_parameter_or_adamw_updates(self):
        model, optimizer, scaler = self.make_training()
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(FloatingPointError, "8 consecutive AMP updates skipped"):
            self.epoch([10000.] * 9, model, optimizer, scaler)
        self.assertEqual(model.forward_calls, 8)
        self.assertEqual(scaler.get_scale(), 256.)
        self.assertEqual(optimizer.state, {})
        self.assertTrue(torch.equal(model.weight, before))

    def test_short_epoch_with_only_skips_is_not_successful(self):
        model, optimizer, scaler = self.make_training()
        with self.assertRaisesRegex(FloatingPointError, "No optimizer updates in this epoch"):
            self.epoch([1.], model, optimizer, scaler)
        self.assertEqual(optimizer.state, {})

    def test_nonfinite_loss_fails_before_backward_or_scaler_update(self):
        model, optimizer, scaler = self.make_training()
        with self.assertRaisesRegex(FloatingPointError, "Non-finite training loss.*batch=1"):
            self.epoch([float("nan")], model, optimizer, scaler)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(scaler.get_scale(), 65536.)
        self.assertEqual(optimizer.state, {})


if __name__ == "__main__":
    unittest.main()
