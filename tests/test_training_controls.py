"""CPU tests for transfer, optimization and validation-only model selection."""

import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from tools import run_triprd_ab as runner
from tools.triprd_training import build_optimizer, build_scheduler, load_checkpoint


class StudentStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 3)
        self.decoder = nn.Linear(3, 1)
        self.extra = nn.Parameter(torch.tensor(2.0))
        self.frozen = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.register_buffer("checkpoint_marker", torch.tensor(0))


class CheckpointTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="triprd_transfer_test_")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "source.pt"
        torch.manual_seed(4)
        self.source = StudentStub()
        self.target = StudentStub()

    def save(self, model_state=None, stage="B/vdt"):
        state = {"model": self.source.state_dict() if model_state is None else model_state, "stage": stage, "epoch": 3}
        torch.save(state, self.path)
        return state

    def test_full_transfer_restores_all_state_and_metadata(self):
        self.save()
        state = load_checkpoint(self.target, self.path, "cpu", expected_stage="B/vdt")
        self.assertEqual(state["epoch"], 3)
        for name, value in self.source.state_dict().items():
            torch.testing.assert_close(self.target.state_dict()[name], value, rtol=0, atol=0)

    def test_encoder_only_transfer_preserves_decoder_and_other_state(self):
        self.save()
        original = {name: value.clone() for name, value in self.target.state_dict().items()}
        load_checkpoint(self.target, self.path, "cpu", scope="encoder", expected_stage="B/vdt")
        for name, value in self.target.state_dict().items():
            expected = self.source.state_dict()[name] if name.startswith("encoder.") else original[name]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_full_transfer_is_strict_about_missing_and_unexpected_parameters(self):
        for change in ("missing", "unexpected"):
            with self.subTest(change=change):
                raw = dict(self.source.state_dict())
                if change == "missing":
                    raw.pop("decoder.weight")
                else:
                    raw["unknown.weight"] = torch.ones(1)
                self.save(raw)
                with self.assertRaises(RuntimeError):
                    load_checkpoint(self.target, self.path, "cpu")

    def test_encoder_transfer_is_strict_about_encoder_keys_and_shapes(self):
        for change in ("missing", "unexpected", "shape"):
            with self.subTest(change=change):
                raw = dict(self.source.state_dict())
                if change == "missing":
                    raw.pop("encoder.bias")
                elif change == "unexpected":
                    raw["encoder.extra"] = torch.ones(1)
                else:
                    raw["encoder.weight"] = torch.ones(9, 9)
                self.save(raw)
                with self.assertRaises(RuntimeError):
                    load_checkpoint(self.target, self.path, "cpu", scope="encoder")

    def test_encoder_transfer_requires_student_encoder(self):
        self.save({"rgb_encoder.weight": torch.ones(3, 2)})
        with self.assertRaisesRegex(ValueError, "No student encoder"):
            load_checkpoint(self.target, self.path, "cpu", scope="encoder")
        self.save()
        with self.assertRaisesRegex(ValueError, "requires a student encoder"):
            load_checkpoint(nn.Linear(2, 3), self.path, "cpu", scope="encoder")

    def test_stage_mismatch_is_rejected_before_loading_weights(self):
        self.save(stage="B/cod")
        original = {name: value.clone() for name, value in self.target.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "Expected a B/vdt checkpoint"):
            load_checkpoint(self.target, self.path, "cpu", expected_stage="B/vdt")
        for name, value in self.target.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_raw_state_without_stage_cannot_bypass_stage_check(self):
        torch.save(self.source.state_dict(), self.path)
        with self.assertRaisesRegex(ValueError, "Expected a B/vdt checkpoint"):
            load_checkpoint(self.target, self.path, "cpu", expected_stage="B/vdt")
        load_checkpoint(self.target, self.path, "cpu")

    def test_invalid_transfer_scope_is_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "Unknown transfer scope"):
            load_checkpoint(self.target, self.path, "cpu", scope="decoder")

    def test_matching_embedded_protocol_is_verified_without_warning(self):
        expected = {"model_variant": "tiny", "image_size": 64, "rgb_normalization": "[0,1]"}
        state = self.save()
        state["protocol"] = {**expected, "learning_rate": 1e-4}
        torch.save(state, self.path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_checkpoint(self.target, self.path, "cpu", expected_stage="B/vdt", expected_protocol=expected)
        self.assertEqual(caught, [])
        self.assertTrue(loaded["_protocol_verified"])
        self.assertEqual(loaded["_source_protocol_origin"], "checkpoint")
        self.assertEqual(loaded["protocol"], state["protocol"])

    def test_protocol_mismatch_fails_before_weights_are_loaded(self):
        expected = {"model_variant": "tiny", "image_size": 64, "rgb_normalization": "[0,1]"}
        original = {name: value.clone() for name, value in self.target.state_dict().items()}
        for key, incompatible in (("model_variant", "resnet50"), ("image_size", 352),
                                  ("rgb_normalization", "ImageNet")):
            with self.subTest(key=key):
                state = self.save()
                state["protocol"] = {**expected, key: incompatible}
                torch.save(state, self.path)
                with self.assertRaisesRegex(ValueError, f"Checkpoint protocol mismatch for {key}"):
                    load_checkpoint(self.target, self.path, "cpu", expected_protocol=expected)
                for name, value in self.target.state_dict().items():
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_legacy_protocol_is_loaded_from_run_directory_ancestor(self):
        expected = {"model_variant": "tiny", "image_size": 64, "rgb_normalization": "[0,1]"}
        run_root = Path(self.temporary.name) / "legacy_run"
        self.path = run_root / "B" / "vdt" / "best.pt"
        self.path.parent.mkdir(parents=True)
        self.save()
        comparison = run_root / "comparison.json"
        comparison.write_text(json.dumps({"protocol": expected}), encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_checkpoint(self.target, self.path, "cpu", expected_stage="B/vdt", expected_protocol=expected)
        self.assertEqual(caught, [])
        self.assertTrue(loaded["_protocol_verified"])
        self.assertEqual(Path(loaded["_source_protocol_origin"]), comparison.resolve())
        self.assertEqual(loaded["protocol"], expected)

    def test_legacy_ancestor_protocol_mismatch_is_not_silently_accepted(self):
        self.save()
        comparison = self.path.parent / "comparison.json"
        comparison.write_text(json.dumps({"protocol": {"rgb_normalization": "ImageNet"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Checkpoint protocol mismatch for rgb_normalization"):
            load_checkpoint(self.target, self.path, "cpu", expected_protocol={"rgb_normalization": "[0,1]"})

    def test_missing_protocol_warns_and_stays_unverified(self):
        self.save()
        with self.assertWarnsRegex(UserWarning, "preprocessing metadata incomplete"):
            loaded = load_checkpoint(self.target, self.path, "cpu", expected_protocol={"rgb_normalization": "[0,1]"})
        self.assertFalse(loaded["_protocol_verified"])
        self.assertIsNone(loaded["_source_protocol_origin"])
        self.assertEqual(loaded["protocol"], {})
        for name, value in self.target.state_dict().items():
            torch.testing.assert_close(value, self.source.state_dict()[name], rtol=0, atol=0)

    def test_partial_protocol_warns_and_preserves_known_metadata(self):
        state = self.save()
        state["protocol"] = {"model_variant": "tiny"}
        torch.save(state, self.path)
        with self.assertWarnsRegex(UserWarning, "preprocessing metadata incomplete"):
            loaded = load_checkpoint(self.target, self.path, "cpu",
                                     expected_protocol={"model_variant": "tiny", "rgb_normalization": "[0,1]"})
        self.assertFalse(loaded["_protocol_verified"])
        self.assertEqual(loaded["_source_protocol_origin"], "checkpoint")
        self.assertEqual(loaded["protocol"], {"model_variant": "tiny"})


class OptimizationControlTests(unittest.TestCase):
    def test_encoder_multiplier_covers_every_trainable_parameter_exactly_once(self):
        model = StudentStub()
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-4, encoder_lr_scale=0.2)
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertAlmostEqual(groups["encoder"]["lr"], 2e-4)
        self.assertAlmostEqual(groups["decoder_heads"]["lr"], 1e-3)
        grouped = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        self.assertEqual(set(grouped), expected)
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual({id(p) for p in groups["encoder"]["params"]}, {id(p) for p in model.encoder.parameters()})
        self.assertIn(id(model.extra), {id(p) for p in groups["decoder_heads"]["params"]})
        self.assertNotIn(id(model.frozen), grouped)
        self.assertTrue(all(group["weight_decay"] == 1e-4 for group in optimizer.param_groups))

    def test_encoder_scale_requires_encoder_but_default_handles_teacher(self):
        model = nn.Linear(2, 1)
        optimizer = build_optimizer(model, 1e-3, 0, 1)
        self.assertEqual(len(optimizer.param_groups), 1)
        with self.assertRaisesRegex(ValueError, "require a student encoder"):
            build_optimizer(model, 1e-3, 0, 0.1)

    def test_invalid_optimizer_hyperparameters_fail(self):
        for lr, decay, scale in ((0, 0, 1), (float("nan"), 0, 1), (1e-3, -1, 1),
                                 (1e-3, 0, 0), (1e-3, 0, float("inf"))):
            with self.subTest(lr=lr, decay=decay, scale=scale), self.assertRaises(ValueError):
                build_optimizer(StudentStub(), lr, decay, scale)

    def learning_rates(self, epochs, warmup, minimum):
        optimizer = build_optimizer(StudentStub(), 1.0, 0, 0.2)
        scheduler = build_scheduler(optimizer, "cosine", epochs, warmup, minimum)
        rates = []
        for _ in range(epochs):
            rates.append([group["lr"] for group in optimizer.param_groups])
            optimizer.step()
            scheduler.step()
        return rates

    def test_cosine_schedule_spans_base_to_minimum_and_preserves_encoder_ratio(self):
        rates = self.learning_rates(epochs=5, warmup=0, minimum=0.1)
        self.assertAlmostEqual(rates[0][1], 1)
        self.assertAlmostEqual(rates[-1][1], 0.1)
        self.assertTrue(all(left[1] >= right[1] for left, right in zip(rates, rates[1:])))
        for encoder, decoder in rates:
            self.assertAlmostEqual(encoder, decoder * 0.2)

    def test_warmup_and_cosine_endpoints(self):
        rates = self.learning_rates(epochs=6, warmup=2, minimum=0.05)
        self.assertAlmostEqual(rates[0][1], 0.5)
        self.assertAlmostEqual(rates[1][1], 1)
        self.assertAlmostEqual(rates[-1][1], 0.05)
        self.assertTrue(all(0.05 <= decoder <= 1 for _, decoder in rates))

    def test_last_allowed_warmup_boundary_reaches_minimum(self):
        rates = self.learning_rates(epochs=2, warmup=1, minimum=0.1)
        self.assertAlmostEqual(rates[0][1], 1)
        self.assertAlmostEqual(rates[-1][1], 0.1)

    def test_single_epoch_cosine_remains_at_initial_rate(self):
        rates = self.learning_rates(epochs=1, warmup=0, minimum=0.1)
        self.assertAlmostEqual(rates[0][1], 1)

    def test_scheduler_none_and_invalid_boundaries(self):
        optimizer = build_optimizer(StudentStub(), 1e-3, 0)
        self.assertIsNone(build_scheduler(optimizer, "none", 3))
        for name, epochs, warmup, minimum in (("unknown", 3, 0, 0.1), ("none", 3, 1, 0.1),
                                             ("cosine", 3, 3, 0.1), ("cosine", 3, -1, 0.1),
                                             ("cosine", 0, 0, 0.1), ("cosine", 3, 0, -0.1),
                                             ("cosine", 3, 0, 1.1)):
            with self.subTest(name=name, epochs=epochs, warmup=warmup, minimum=minimum), self.assertRaises(ValueError):
                build_scheduler(optimizer, name, epochs, warmup, minimum)


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="triprd_selection_test_")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.model = StudentStub()
        self.train_loader = SimpleNamespace(dataset=["train-a", "train-b"])
        self.val_loader = SimpleNamespace(dataset=["held-out-a", "held-out-b"])
        self.val_items = [runner.Item("COD10K/held-out-a", (None, None), "cod"),
                          runner.Item("CAMO/held-out-b", (None, None), "cod")]

    def fit_mocked(self, selection="S_alpha", every=2, dice=None, official_scores=None):
        dice = [0.1, 0.2, 0.99, 0.4, 0.5] if dice is None else dice
        official_scores = {2: 0.9, 4: 0.7, 5: 0.8} if official_scores is None else official_scores
        current = {"epoch": 0}
        validation_epochs = []

        def epoch_result(model, loader, device, criterion, *, optimizer=None, **kwargs):
            if optimizer is not None:
                self.assertIs(loader, self.train_loader)
                current["epoch"] += 1
                model.checkpoint_marker.fill_(current["epoch"])
                return {"total": 0.2, "dice": 0.9}
            self.assertIs(loader, self.val_loader)
            return {"total": 0.3, "dice": dice[current["epoch"] - 1]}

        def original_validation(model, items, size, device):
            self.assertIs(items, self.val_items)
            self.assertEqual([item.key for item in items], ["COD10K/held-out-a", "CAMO/held-out-b"])
            self.assertEqual(size, 64)
            validation_epochs.append(current["epoch"])
            value = official_scores[current["epoch"]]
            return {"macro": {"S_alpha": value, "F_beta_w": value, "MAE": value}}

        with mock.patch.object(runner, "_run_epoch", side_effect=epoch_result), \
                mock.patch.object(runner, "_validate_original_size", side_effect=original_validation), \
                mock.patch.object(runner, "_predict_test") as predict_test, \
                mock.patch.object(runner, "_official_metrics") as evaluate_test, \
                contextlib.redirect_stdout(io.StringIO()):
            best_path, summary = runner._fit_stage(
                "A/cod", self.model, self.train_loader, self.val_loader,
                torch.device("cpu"), object(), self.output, len(dice), 1e-3, 0, 42,
                original_val_items=self.val_items, image_size=64,
                selection_metric=selection, val_metrics_every=every,
                checkpoint_metadata={"test_run": "fixed-validation"},
            )
            predict_test.assert_not_called()
            evaluate_test.assert_not_called()
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        last = torch.load(best_path.with_name("last.pt"), map_location="cpu", weights_only=True)
        return summary, best, last, validation_epochs

    def test_original_sm_selects_different_checkpoint_from_best_dice(self):
        summary, best, last, evaluated = self.fit_mocked()
        self.assertEqual(evaluated, [2, 4, 5])
        self.assertEqual(best["epoch"], 2)
        self.assertEqual(best["model"]["checkpoint_marker"].item(), 2)
        self.assertEqual(last["epoch"], 5)
        self.assertEqual(last["model"]["checkpoint_marker"].item(), 5)
        self.assertEqual(best["selection_metric"], "S_alpha")
        self.assertAlmostEqual(best["selection_score"], 0.9)
        self.assertEqual(best["protocol"], {"test_run": "fixed-validation"})
        self.assertEqual(summary["best_epoch"], 2)
        self.assertAlmostEqual(summary["best_val_dice"], 0.99)
        self.assertAlmostEqual(summary["selected_val_dice"], 0.2)
        self.assertIsNone(summary["history"][0]["selection_score"])
        self.assertIsNone(summary["history"][2]["selection_score"])
        logged = [json.loads(line) for line in (self.output / "A/cod/metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["epoch"] for row in logged if "val_official" in row], [2, 4, 5])

    def test_dice_selection_keeps_every_epoch_eligible(self):
        summary, best, last, evaluated = self.fit_mocked(selection="dice")
        self.assertEqual(evaluated, [2, 4, 5])
        self.assertEqual(best["epoch"], 3)
        self.assertEqual(last["epoch"], 5)
        self.assertAlmostEqual(summary["best_selection_score"], 0.99)
        self.assertTrue(all(row["selection_score"] is not None for row in summary["history"]))

    def test_mae_selection_minimizes_and_final_epoch_is_forced(self):
        summary, best, last, evaluated = self.fit_mocked(
            selection="MAE", every=2, official_scores={2: 0.07, 4: 0.09, 5: 0.04})
        self.assertEqual(evaluated, [2, 4, 5])
        self.assertEqual(best["epoch"], 5)
        self.assertEqual(last["epoch"], 5)
        self.assertAlmostEqual(summary["best_selection_score"], 0.04)

    def test_evaluation_interval_beyond_training_still_selects_final_checkpoint(self):
        summary, best, last, evaluated = self.fit_mocked(
            every=10, dice=[0.9, 0.1, 0.2], official_scores={3: 0.75})
        self.assertEqual(evaluated, [3])
        self.assertEqual(summary["best_epoch"], 3)
        self.assertEqual(best["epoch"], 3)
        self.assertEqual(last["epoch"], 3)

    def test_official_selection_requires_validation_items(self):
        with self.assertRaisesRegex(ValueError, "requires COD validation items"):
            runner._fit_stage("A/cod", self.model, self.train_loader, self.val_loader,
                              torch.device("cpu"), object(), self.output, 1, 1e-3, 0, 42,
                              selection_metric="S_alpha")


if __name__ == "__main__":
    unittest.main()
