"""Source-only diagnostic controls: objectives, gradients and real tiny training."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from models.triprd_correction import CorrectionStudent
from models.triprd_correction_loss import annotation_relations, correction_targets, source_loss
from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import diagnose_triprd_correction_probe as diagnostic
from tools import run_triprd_correction as runner


def manual_valid(height, width):
    """Independent boundary/centre oracle, deliberately not using unfold."""
    result = torch.zeros(1, 1, 9, height, width, dtype=torch.bool)
    for y in range(height):
        for x in range(width):
            for index, (dy, dx) in enumerate((
                    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                    (0, 1), (1, -1), (1, 0), (1, 1))):
                result[0, 0, index, y, x] = (
                    index != 4 and 0 <= y + dy < height and 0 <= x + dx < width)
    return result


class PositiveCorrectionObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)

    def test_mse_counts_edges_across_both_scales_and_excludes_invalid_nonpositive(self):
        mask = torch.ones(2, 1, 12, 12)
        predictions, deltas, reliabilities, available, student_masks = [], [], [], [], []
        expected_sum, expected_count, scale_means = 0., 0, []
        selections = []
        for scale, (height, width) in enumerate(((2, 2), (3, 4))):
            shape = (2, 1, 9, height, width)
            target_valid = manual_valid(height, width)
            student_valid = target_valid.clone()
            flat_valid = torch.nonzero(target_valid.flatten()).flatten()
            student_valid.flatten()[flat_valid[0]] = False
            reliability = torch.ones(shape)
            reliability.flatten()[::7] = 0
            reliability.flatten()[::11] = -1
            delta = torch.full(shape, .2 - .1 * scale)
            selected = target_valid.expand(shape) & student_valid.expand(shape) & (reliability > 0)
            # Invalid/nonpositive entries must not affect the objective.
            values = torch.full(shape, 10000.)
            values[selected] = .3 + .4 * scale
            prediction = values.requires_grad_()
            error = (values.detach()[selected] - delta[selected]).square()
            expected_sum += float(error.sum())
            expected_count += int(selected.sum())
            scale_means.append(float(error.mean()))
            predictions.append(prediction)
            deltas.append(delta)
            reliabilities.append(reliability)
            available.append(target_valid)
            student_masks.append(student_valid)
            selections.append(selected)
        output = {"corrections": tuple(predictions), "relation_valid_masks": tuple(student_masks)}
        targets = {"delta": tuple(deltas), "valid": tuple(available),
                   "reliability": tuple(reliabilities)}
        value = diagnostic.positive_mse_loss(output, targets, mask, relation_kernel=3)
        expected = expected_sum / expected_count
        self.assertAlmostEqual(float(value.detach()), expected, places=6)
        self.assertGreater(abs(expected - sum(scale_means) / 2), .01)
        value.backward()
        for prediction, delta, selected in zip(predictions, deltas, selections):
            gradient = torch.zeros_like(prediction)
            gradient[selected] = 2 * (prediction.detach()[selected] - delta[selected]) / expected_count
            torch.testing.assert_close(prediction.grad, gradient)

    def test_no_positive_edges_produces_differentiable_zero(self):
        for empty_kind in ("nonpositive", "invalid"):
            with self.subTest(empty_kind=empty_kind):
                prediction = torch.randn(1, 1, 9, 2, 2, requires_grad=True)
                valid = manual_valid(2, 2)
                reliability = torch.zeros_like(prediction)
                if empty_kind == "invalid":
                    valid.zero_()
                    reliability.fill_(1)
                value = diagnostic.positive_mse_loss(
                    {"corrections": (prediction,)},
                    {"delta": (torch.ones_like(prediction),), "valid": (valid,),
                     "reliability": (reliability,)}, torch.ones(1, 1, 8, 8))
                self.assertEqual(float(value.detach()), 0.)
                self.assertTrue(value.requires_grad)
                value.backward()
                torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_teacher_targets_remain_detached_during_positive_mse_backward(self):
        teacher_rgb = torch.zeros(1, 1, 9, 2, 2, requires_grad=True)
        teacher_full = torch.full_like(teacher_rgb, 2., requires_grad=True)
        mask = torch.ones(1, 1, 8, 8)
        targets = correction_targets({"relation_logits": (teacher_rgb,)},
                                     {"relation_logits": (teacher_full,)}, mask)
        prediction = torch.zeros_like(teacher_rgb, requires_grad=True)
        diagnostic.positive_mse_loss({"corrections": (prediction,)}, targets, mask).backward()
        self.assertIsNone(teacher_rgb.grad)
        self.assertIsNone(teacher_full.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.)

    def test_four_control_objectives_preserve_source_and_kd_scaling(self):
        source = {"supervised": torch.tensor(2.), "transfer": torch.tensor(3.)}
        positive = torch.tensor(.4)
        expected = {"joint_balanced": 4., "joint_positive_mse": 3.,
                    "isolated_balanced": 3., "isolated_positive_mse": 2.}
        for arm, result in expected.items():
            with self.subTest(arm=arm):
                self.assertAlmostEqual(float(diagnostic.diagnostic_objective(source, positive, arm, .5, 2.)), result)

    def test_summary_gate_uses_both_baselines_and_requires_positive_support(self):
        for prediction, count, expected in ((.002, 5, True), (.004, 5, False),
                                            (.006, 5, False), (.002, 0, False)):
            with self.subTest(prediction=prediction, count=count):
                report = {"initial_source_recoverability": {"positive_mse": .005},
                    "source_recoverability": {"positive_mse": prediction, "positive_zero_mse": .005,
                        "positive_constant_mse": .004, "positive_count": count, "count": 10}}
                summary = diagnostic._summary_row("fixture", report)
                self.assertEqual(summary["passes_original_gate"], expected)
                self.assertEqual(summary["positive_mse"], prediction)

    def test_joint_and_isolated_objectives_route_gradients_to_intended_parameters(self):
        for arm in diagnostic.ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(31)
                model = CorrectionStudent(model_variant="tiny", base_channels=4, pretrained=False)
                # Zero-initialized final projections block their upstream gradients
                # on the first step. Move them as one real update would.
                with torch.no_grad():
                    for head in model.correction_heads:
                        head.edge[-1].weight.fill_(.03)
                mask = torch.ones(2, 1, 32, 32)
                mask[:, :, 8:24, 8:24] = 0
                output = model(torch.randn(2, 3, 32, 32), task="sod")
                rgb_logits, full_logits = [], []
                for prediction in output["corrections"]:
                    target, _, _ = annotation_relations(mask, prediction.shape[-2:])
                    rgb_logits.append(torch.zeros_like(prediction, requires_grad=True))
                    full_logits.append((2 * (2 * target - 1)).requires_grad_())
                rgb, full = {"relation_logits": tuple(rgb_logits)}, {"relation_logits": tuple(full_logits)}
                source = source_loss(output, mask, rgb, full, mode="correction", relation_weight=.2)
                positive = diagnostic.positive_mse_loss(output, correction_targets(rgb, full, mask), mask)
                objective = diagnostic.diagnostic_objective(source, positive, arm, .5, 1.)
                objective.backward()
                groups = {
                    "adapters": model.adapters,
                    "correction": model.correction_heads,
                    "base": model.base_heads,
                    "decoder": model.decoder,
                    "sod": model.task_heads["sod"],
                }
                for name in ("adapters", "correction"):
                    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                        for p in groups[name].parameters()), name)
                for name in ("base", "decoder", "sod"):
                    gradients = [p.grad for p in groups[name].parameters()]
                    if arm.startswith("isolated_"):
                        self.assertTrue(all(g is None for g in gradients), name)
                    else:
                        self.assertTrue(any(g is not None and bool(g.abs().sum() > 0) for g in gradients), name)
                self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
                self.assertTrue(all(p.grad is None for p in model.task_heads["cod"].parameters()))
                self.assertTrue(all(p.grad is None for p in (*rgb_logits, *full_logits)))


class ProbeDiagnosticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)
        folder = tempfile.TemporaryDirectory(prefix=".probe-diagnostics-", dir=runner.ROOT)
        cls.addClassCleanup(folder.cleanup)
        cls.folder = Path(folder.name)
        cls.reference = cls.folder / "reference"
        cls.output = cls.folder / "diagnostics"
        index = MemoryVDTIndex()
        cls.vdt = [runner.shared.Item(f"VDT/{sample['rgb'].sample_id}", (sample, index), "vdt")
                   for sample in index.entries]
        cls.cod = [runner.shared.Item(f"{index.dataset}/{sample.sample_id}", (sample, index), "cod")
                   for index in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO"))
                   for sample in index.samples()]
        teacher = cls.reference / "teacher"
        probe = cls.reference / "probe"
        with patch.object(runner.shared, "_vdt_items", return_value=cls.vdt), \
             patch.object(runner.shared, "_cod_items", return_value=cls.cod), redirect_stdout(io.StringIO()):
            runner.main(cls.reference_arguments(teacher) + ["--stage", "teacher", "--group", "D"])
        with ExitStack() as stack:
            cls.forbid_cod(stack)
            runner.main(cls.reference_arguments(probe) + ["--stage", "probe", "--group", "D",
                "--teacher-checkpoint", str(teacher / "best.pt")])
        cls.original_hashes = cls.hash_reference()
        cls.reports = {}
        for arm in diagnostic.ARMS:
            cls.invoke(arm)
            cls.reports[arm] = cls.read_report(arm)

    @classmethod
    def reference_arguments(cls, out):
        return ["--out", str(out), "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
            "--base-channels", "4", "--image-size", "32", "--batch-size", "2", "--num-workers", "0",
            "--epochs", "2", "--teacher-epochs", "2", "--max-samples", "4", "--warmup-epochs", "1",
            "--train-seed", "11", "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
            "--skip-test", "--no-amp"]

    @classmethod
    def forbid_cod(cls, stack):
        stack.enter_context(patch.object(runner.shared, "_vdt_items", return_value=cls.vdt))
        for obj, name in ((runner.shared, "_cod_items"), (runner, "_validate_cod"),
                          (runner, "_formal_tests"), (runner, "_constant_probe")):
            stack.enter_context(patch.object(obj, name,
                side_effect=AssertionError(f"Diagnostic must not access/refit {name}")))
        stack.enter_context(redirect_stdout(io.StringIO()))

    @classmethod
    def invoke(cls, arm, *, output=None, extra=()):
        with ExitStack() as stack:
            cls.forbid_cod(stack)
            return diagnostic.main(["--run-root", str(cls.reference), "--output-root", str(output or cls.output),
                "--epochs", "2", "--execute", "--arm", arm, *extra])

    @classmethod
    def read_report(cls, arm, output=None):
        return json.loads(((output or cls.output) / arm / "comparison.json").read_text(encoding="utf-8"))

    @classmethod
    def hash_reference(cls):
        return {path.relative_to(cls.reference).as_posix(): runner.sha256_file(path)
                for path in cls.reference.rglob("*") if path.is_file()}

    def test_real_source_only_arms_share_initialization_and_keep_original_results(self):
        self.assertEqual(self.hash_reference(), self.original_hashes)
        self.assertEqual(len({report["initial_state_sha256"] for report in self.reports.values()}), 1)
        for arm, report in self.reports.items():
            with self.subTest(arm=arm):
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["completed_epochs"], 2)
                self.assertTrue(report["diagnostic"])
                self.assertFalse(report["formal"])
                self.assertFalse(report["cod_access"])
                self.assertFalse(report["validation_used_for_selection"])
                self.assertEqual(report["selection_metric"], "fixed_last_epoch")
                self.assertTrue(report["frozen_generic_encoder_verified"])
                self.assertTrue(report["frozen_teacher_verified"])
                self.assertEqual(len(report["history"]), 2)
                self.assertEqual(report["source_recoverability"], report["history"][-1]["source_recoverability"])
                self.assertEqual(runner.sha256_file(self.output / arm / "last.pt"), report["checkpoint_sha256"])
                self.assertFalse((self.output / arm / "best.pt").exists())
                for row in report["history"]:
                    self.assertGreater(row["train"]["optimizer_steps"], 0)
                    self.assertEqual(row["train"]["amp_skipped_steps"], 0)
                    self.assertNotIn("validation", row)
                    self.assertNotIn("cod_total", row["train"])
                # allow_nan=False requires every persisted telemetry value to be finite.
                json.dumps(report, allow_nan=False)

    def test_completed_resume_preserves_checkpoint_without_training(self):
        arm = "isolated_positive_mse"
        checkpoint = self.output / arm / "last.pt"
        before = runner.sha256_file(checkpoint)
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("completed arm retrained")):
            self.invoke(arm, extra=["--resume"])
        self.assertEqual(runner.sha256_file(checkpoint), before)
        self.assertEqual(self.hash_reference(), self.original_hashes)

    def test_joint_balanced_reproduces_original_probe_weights_and_metrics(self):
        original = runner._read_checkpoint(self.reference / "probe" / "last.pt")
        current = torch.load(self.output / "joint_balanced" / "last.pt", weights_only=True, map_location="cpu")
        self.assertEqual(current["initial_state_sha256"], original["initial_state_sha256"])
        for name, tensor in original["model"].items():
            self.assertTrue(torch.equal(current["model"][name], tensor), name)
        self.assertEqual(current["history"][-1]["source_recoverability"],
                         original["history"][-1]["source_recoverability"])

    def test_interrupted_epoch_resumes_to_identical_weights_and_metrics(self):
        arm, output = "isolated_positive_mse", self.folder / "interrupted"
        train = diagnostic._train_epoch
        def interrupt_at_second_epoch(*args, **kwargs):
            if args[-1] == 2:
                raise RuntimeError("injected diagnostic interruption")
            return train(*args, **kwargs)
        with patch.object(diagnostic, "_train_epoch", side_effect=interrupt_at_second_epoch):
            with self.assertRaisesRegex(RuntimeError, "injected diagnostic interruption"):
                self.invoke(arm, output=output)
        partial = torch.load(output / arm / "last.pt", weights_only=True, map_location="cpu")
        self.assertEqual(partial["epoch"], 1)
        self.invoke(arm, output=output, extra=["--resume"])
        actual = torch.load(output / arm / "last.pt", weights_only=True, map_location="cpu")
        expected = torch.load(self.output / arm / "last.pt", weights_only=True, map_location="cpu")
        self.assertEqual(actual["epoch"], 2)
        for name, tensor in expected["model"].items():
            self.assertTrue(torch.equal(actual["model"][name], tensor), name)
        self.assertEqual(actual["history"][-1]["source_recoverability"],
                         expected["history"][-1]["source_recoverability"])
        self.assertEqual(self.hash_reference(), self.original_hashes)

    def test_tampered_teacher_hash_is_rejected_before_training_or_output_creation(self):
        path = self.reference / "teacher" / "comparison.json"
        original = path.read_bytes()
        out = self.folder / "bad_teacher"
        try:
            report = json.loads(original)
            report["checkpoint_sha256"] = "0" * 64
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("bad reference trained")):
                with self.assertRaises(ValueError):
                    self.invoke("isolated_positive_mse", output=out)
            self.assertFalse(out.exists())
        finally:
            path.write_bytes(original)

    def test_tampered_probe_source_contract_is_rejected_before_training(self):
        path = self.reference / "probe" / "comparison.json"
        original = path.read_bytes()
        out = self.folder / "bad_probe"
        try:
            report = json.loads(original)
            report["contract_sha256"] = "0" * 64
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("bad reference trained")):
                with self.assertRaises(ValueError):
                    self.invoke("isolated_positive_mse", output=out)
            self.assertFalse(out.exists())
        finally:
            path.write_bytes(original)


class ProbeDiagnosticLauncherTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix=".diagnostic-launcher-", dir=runner.ROOT)
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_real_children_receive_single_physical_gpu_and_quoted_paths(self):
        fixture = self.folder / "child.py"
        fixture.write_text('''import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--arm")
p.add_argument("--output-root")
a = p.parse_args()
result = {"arm": a.arm, "visibility": os.environ.get("CUDA_VISIBLE_DEVICES")}
(Path(a.output_root) / a.arm / "observed.json").write_text(json.dumps(result))
print(json.dumps(result), flush=True)
''', encoding="utf-8")
        output = self.folder / "results with spaces"
        args = diagnostic.parse_args(["--run-root", str(self.folder / "reference with spaces"),
            "--output-root", str(output), "--gpus", "2,3,4,5", "--epochs", "2", "--execute"])
        context = {"recipe": {"test": "single physical GPU per child"}}
        captured = []
        real_popen = subprocess.Popen
        def child(command, **kwargs):
            captured.append(command)
            arm = command[command.index("--arm") + 1]
            return real_popen([sys.executable, str(fixture), "--arm", arm,
                               "--output-root", str(output)], **kwargs)
        def observed(directory, contract):
            return json.loads((directory / "observed.json").read_text())
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,4,5"}), \
             patch.object(diagnostic.subprocess, "Popen", side_effect=child), \
             patch.object(diagnostic, "verify_completed", side_effect=observed), \
             redirect_stdout(io.StringIO()):
            reports = diagnostic._run_jobs(args, context, ["2", "3", "4", "5"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3,4,5")
        self.assertEqual(set(reports), set(diagnostic.ARMS))
        self.assertEqual([reports[arm]["visibility"] for arm in diagnostic.ARMS], ["2", "3", "4", "5"])
        self.assertEqual(len(captured), 4)
        for command in captured:
            self.assertIsInstance(command, list)
            self.assertEqual(command[command.index("--run-root") + 1], str(args.run_root))
            self.assertEqual(command[command.index("--output-root") + 1], str(args.output_root))
            self.assertIn("--execute", command)
            self.assertNotIn("--gpus", command)
        for arm in diagnostic.ARMS:
            self.assertIn(reports[arm]["visibility"], (output / arm / "run.log").read_text())

    def test_shell_passes_physical_gpu_list_and_paths_without_expansion(self):
        bash = shutil.which("bash")
        if bash is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
            bash = "C:/Program Files/Git/bin/bash.exe"
        if bash is None:
            self.skipTest("Bash unavailable")
        fixture = self.folder / "fake_python.sh"
        fixture.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FIXTURE_ARGUMENTS"
printf '%s' "${CUDA_VISIBLE_DEVICES-unset}" > "$FIXTURE_VISIBILITY"
''', encoding="utf-8", newline="\n")
        fixture.chmod(0o700)
        env = os.environ.copy()
        env.update(PYTHON=fixture.as_posix(), GPUS="2,3,4,5", CUDA_VISIBLE_DEVICES="2,3,4,5",
            RUN_ROOT="reference with spaces/seed-1", OUTPUT_ROOT="results with spaces/diagnostic",
            PROBE_EPOCHS="2", RESUME="1", DRY_RUN="1",
            FIXTURE_ARGUMENTS=(self.folder / "args.txt").as_posix(),
            FIXTURE_VISIBILITY=(self.folder / "visibility.txt").as_posix())
        result = subprocess.run([bash, "tools/run_triprd_probe_diagnostics.sh"], cwd=runner.ROOT,
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = (self.folder / "args.txt").read_text().splitlines()
        self.assertEqual(args[args.index("--gpus") + 1], "2,3,4,5")
        self.assertEqual(args[args.index("--run-root") + 1], env["RUN_ROOT"])
        self.assertEqual(args[args.index("--output-root") + 1], env["OUTPUT_ROOT"])
        self.assertEqual(args[args.index("--epochs") + 1], "2")
        self.assertIn("--resume", args)
        self.assertNotIn("--execute", args)
        self.assertEqual((self.folder / "visibility.txt").read_text(), "2,3,4,5")


if __name__ == "__main__":
    unittest.main()
