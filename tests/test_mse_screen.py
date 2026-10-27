"""CPU controls for the versioned, matched COD positive-MSE development screen."""
from contextlib import ExitStack, redirect_stdout
import copy
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

from models.triprd_correction import CorrectionStudent, CorrectionTeacher
from models.triprd_correction_loss import annotation_relations, correction_targets, source_loss
from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import diagnose_triprd_correction_probe as diagnostic
from tools import run_triprd_correction as runner
from tools import run_triprd_mse_screen as screen
from tools import triprd_mse_screen_training as training


class MSEScreenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)
        temporary = tempfile.TemporaryDirectory(prefix=".mse-screen-", dir=runner.ROOT)
        cls.addClassCleanup(temporary.cleanup)
        cls.folder = Path(temporary.name)
        cls.reference = cls.folder / "reference"
        cls.diagnostics = cls.folder / "diagnostics"
        cls.output = cls.folder / "screen"
        source = MemoryVDTIndex()
        cls.vdt = [runner.shared.Item(f"VDT/{sample['rgb'].sample_id}", (sample, source), "vdt")
                   for sample in source.entries]
        cls.cod = [runner.shared.Item(f"{index.dataset}/{sample.sample_id}", (sample, index), "cod")
                   for index in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO"))
                   for sample in index.samples()]
        with ExitStack() as stack:
            cls.inputs(stack)
            for stage, group, directory in (("teacher", "D", "teacher"),
                    ("probe", "D", "probe"), ("student", "A", screen.GROUP_DIRS["A"]),
                    ("student", "B", screen.GROUP_DIRS["B"]), ("student", "C", "original_C")):
                args = cls.reference_arguments(cls.reference / directory) + ["--stage", stage, "--group", group]
                if stage == "probe" or group == "C":
                    args += ["--teacher-checkpoint", str(cls.reference / "teacher" / "best.pt")]
                runner.main(args)
            # Source diagnostic performs real optimization and checkpointing.
            diagnostic.main(["--run-root", str(cls.reference), "--output-root", str(cls.diagnostics),
                "--epochs", "2", "--execute", "--arm", "joint_positive_mse"])
        cls.reference_hashes = cls.hash_tree(cls.reference)
        cls.diagnostic_hashes = cls.hash_tree(cls.diagnostics)
        cls.args = cls.arguments(cls.output)
        with ExitStack() as stack:
            cls.inputs(stack, allow_constants=False)
            # A random two-epoch tiny source probe is not evidence about the
            # real experiment. Keep all artifact verification, bypass only its
            # statistical gate for integration; rejection is tested separately.
            stack.enter_context(patch.object(screen.phases, "student_gate"))
            cls.context = screen.build_context(cls.args)
            cls.reports = {group: cls.run_group(group) for group in "CD"}

    @classmethod
    def reference_arguments(cls, out):
        return ["--out", str(out), "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
            "--base-channels", "4", "--image-size", "32", "--batch-size", "2", "--num-workers", "0",
            "--epochs", "2", "--teacher-epochs", "2", "--max-samples", "4", "--warmup-epochs", "1",
            "--train-seed", "11", "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
            "--skip-test", "--no-amp"]

    @classmethod
    def arguments(cls, output, extra=()):
        return screen.parse_args(["--run-root", str(cls.reference), "--diagnostics-root", str(cls.diagnostics),
                                  "--output-root", str(output), *extra])

    @classmethod
    def inputs(cls, stack, *, allow_constants=True):
        stack.enter_context(patch.object(runner.shared, "_vdt_items", return_value=cls.vdt))
        stack.enter_context(patch.object(runner.shared, "_cod_items", return_value=cls.cod))
        stack.enter_context(patch.object(runner, "_formal_tests",
            side_effect=AssertionError("Development screen accessed official tests")))
        if not allow_constants:
            stack.enter_context(patch.object(runner, "_constant_probe",
                side_effect=AssertionError("Screen refit saved TRAIN constants")))
        stack.enter_context(redirect_stdout(io.StringIO()))

    @staticmethod
    def hash_tree(directory):
        return {path.relative_to(directory).as_posix(): runner.sha256_file(path)
                for path in directory.rglob("*") if path.is_file()}

    @classmethod
    def run_group(cls, group, *, output=None, resume=False, expected_initial=None):
        args = cls.arguments(output or cls.output, ["--resume"] if resume else [])
        a = screen.student_args(args, cls.context, group)
        with ExitStack() as stack:
            cls.inputs(stack, allow_constants=False)
            return training.run_student(a, cls.context["splits"], screen.student_contract(cls.context, group),
                cls.context["teacher_state"], expected_initial or cls.context["expected_initial"])

    def test_reused_references_unchanged_and_both_students_start_fresh(self):
        self.assertEqual(self.hash_tree(self.reference), self.reference_hashes)
        self.assertEqual(self.hash_tree(self.diagnostics), self.diagnostic_hashes)
        for group, report in self.reports.items():
            with self.subTest(group=group):
                self.assertEqual(report["method"], training.METHOD)
                self.assertNotEqual(report["method"], runner.METHOD)
                self.assertEqual(report["initial_state_sha256"], self.context["expected_initial"])
                self.assertEqual(report["status"], "complete")
                self.assertFalse(report["formal"])
                self.assertTrue(report["development_screen"])
                self.assertEqual(report["test"], {})
                self.assertEqual(report["test_status"], "skipped")
                self.assertTrue(report["frozen_generic_encoder_verified"])
                self.assertTrue(report["frozen_teacher_verified"])
                self.assertEqual(report["completed_epochs"], self.context["a"].epochs)
                self.assertEqual(report["sample_counts"], self.context["baselines"]["B"]["sample_counts"])
                self.assertEqual(report["selected_validation"], screen.selected_validation(report))
                json.dumps(report, allow_nan=False)
        for group in "AB":
            self.assertEqual(self.context["baselines"][group]["method"], runner.METHOD)

    def test_control_c_exactly_reproduces_original_c_training_and_validation(self):
        old = runner._read_checkpoint(self.reference / "original_C" / "last.pt")
        new = training.read_checkpoint(self.output / screen.GROUP_DIRS["C"] / "last.pt")
        self.assertEqual(old["initial_state_sha256"], new["initial_state_sha256"])
        self.assertEqual(old["best_epoch"], new["best_epoch"])
        self.assertEqual(old["best_score"], new["best_score"])
        for name, value in old["model"].items():
            self.assertTrue(torch.equal(value, new["model"][name]), name)
        for previous, current in zip(old["history"], new["history"]):
            self.assertEqual(previous["validation"], current["validation"])
            for field in ("total", "cod_total", "source_supervised", "source_transfer", "grad_norm",
                          "optimizer_steps", "source_images", "primary_images", "amp_skipped_steps"):
                self.assertEqual(previous["train"][field], current["train"][field], field)

    def test_d_matches_independent_joint_supervised_pooled_mse_gradient(self):
        a = screen.student_args(self.args, self.context, "D")
        a.source_weight, a.kd_weight, a.batch_size = .37, .61, 128
        device = torch.device("cpu")
        teacher = CorrectionTeacher(**runner._model_options(a, pretrained=False))
        teacher.load_state_dict(self.context["teacher_state"]["model"])
        teacher.requires_grad_(False).eval()
        runner.shared.seed_everything(a.train_seed)
        actual = CorrectionStudent(**runner._model_options(a))
        expected = copy.deepcopy(actual)
        actual_loaders = runner._loaders(a, self.context["splits"], device)
        expected_loaders = runner._loaders(a, self.context["splits"], device)
        for loader in expected_loaders.values():
            loader.dataset.epoch = 1
        expected.train()
        # Instantiate the source iterator first, as in the preserved cycling
        # schedule. A single batch makes the complete gradient directly visible.
        source_iterator = iter(expected_loaders["vdt_train"])
        cod_batch = next(iter(expected_loaders["cod_train"]))
        source_batch = next(source_iterator)
        cod = source_loss(expected(cod_batch["rgb"], task="cod"), cod_batch["mask"],
                          mode="none", relation_weight=a.relation_weight)
        with torch.no_grad():
            coals = teacher.forward_coalitions(source_batch["rgb"], source_batch["thermal"], source_batch["depth"])
        prediction = expected(source_batch["rgb"], task="sod")
        supervised = source_loss(prediction, source_batch["mask"], mode="none", relation_weight=a.relation_weight)
        targets = correction_targets(coals["R"], coals["RTD"], source_batch["mask"])
        # Independent gather/concatenate oracle: every eligible edge has equal
        # weight across scales and examples, not equal weight per scale/stratum.
        errors = []
        for scale, delta in enumerate(prediction["corrections"]):
            _, valid, _ = annotation_relations(source_batch["mask"], delta.shape[-2:],
                3, prediction["relation_valid_masks"][scale])
            selected = valid & targets["valid"][scale] & (targets["reliability"][scale] > 0)
            errors.append((delta - targets["delta"][scale])[selected.expand_as(delta)])
        raw_mse = torch.cat(errors).square().mean()
        objective = cod["total"] + .37 * supervised["supervised"] + (.37 * .61 * 5.) * raw_mse
        objective.backward()
        expected_gradients = {name: p.grad.clone() if p.grad is not None else None
                              for name, p in expected.named_parameters()}
        seen = {}
        def inspect_gradient(model, optimizer, scaler, grad_clip):
            for name, parameter in model.named_parameters():
                previous = expected_gradients[name]
                if previous is None:
                    self.assertIsNone(parameter.grad, name)
                else:
                    torch.testing.assert_close(parameter.grad, previous, rtol=1e-5, atol=1e-7, msg=name)
            seen["supervised_sod"] = any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                           for p in model.task_heads["sod"].parameters())
            seen["supervised_cod"] = any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                           for p in model.task_heads["cod"].parameters())
            return torch.tensor(1.), None
        optimizer = torch.optim.AdamW([p for p in actual.parameters() if p.requires_grad], lr=a.lr)
        with patch.object(runner, "_optimizer_step", side_effect=inspect_gradient):
            row = training.train_epoch(actual, teacher, actual_loaders, optimizer,
                torch.amp.GradScaler("cuda", enabled=False), device, a, 1)
        self.assertEqual(row["steps"], 1)
        self.assertAlmostEqual(row["total"], float(objective.detach()), places=6)
        self.assertAlmostEqual(row["source_transfer"], 5 * float(raw_mse.detach()), places=6)
        self.assertTrue(seen["supervised_sod"])
        self.assertTrue(seen["supervised_cod"])
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_epoch_boundary_resume_is_exact_and_completed_resume_does_not_train(self):
        output = self.folder / "interrupted"
        original_train = training.train_epoch
        def interrupt(*args, **kwargs):
            if args[-1] == 2:
                raise RuntimeError("injected screen interruption")
            return original_train(*args, **kwargs)
        with patch.object(training, "train_epoch", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "injected screen interruption"):
                self.run_group("D", output=output)
        directory = output / screen.GROUP_DIRS["D"]
        self.assertEqual(training.read_checkpoint(directory / "last.pt")["epoch"], 1)
        self.run_group("D", output=output, resume=True)
        actual = training.read_checkpoint(directory / "last.pt")
        expected = training.read_checkpoint(self.output / screen.GROUP_DIRS["D"] / "last.pt")
        for name, value in expected["model"].items():
            self.assertTrue(torch.equal(value, actual["model"][name]), name)
        self.assertEqual(actual["best_epoch"], expected["best_epoch"])
        self.assertEqual(actual["best_score"], expected["best_score"])
        self.assertEqual(actual["history"][-1]["validation"], expected["history"][-1]["validation"])
        before = self.hash_tree(directory)
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Complete screen retrained")):
            self.run_group("D", output=output, resume=True)
        self.assertEqual(before, self.hash_tree(directory))

    def test_bad_fresh_initialization_rejected_before_optimizer_update(self):
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Mismatched initialization trained")):
            with self.assertRaisesRegex(ValueError, "initialization differs"):
                self.run_group("D", output=self.folder / "bad_initial", expected_initial="0" * 64)

    def test_checkpoint_contract_and_completed_report_tampering_are_rejected(self):
        contract = screen.student_contract(self.context, "D")
        directory = self.output / screen.GROUP_DIRS["D"]
        changed = copy.deepcopy(contract)
        changed["epochs"] += 1
        with self.assertRaisesRegex(ValueError, "contract differs"):
            training.read_checkpoint(directory / "best.pt", changed)
        with self.assertRaisesRegex(ValueError, "valid positive-MSE screen"):
            training.read_checkpoint(self.reference / "original_C" / "best.pt")
        report_path = directory / "comparison.json"
        original = report_path.read_bytes()
        for field, value in (("checkpoint_sha256", "0" * 64), ("initial_state_sha256", "0" * 64),
                              ("test_status", "complete"), ("completed_epochs", 1)):
            with self.subTest(field=field):
                try:
                    report = json.loads(original)
                    report[field] = value
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        training.verify_completed(directory, contract)
                finally:
                    report_path.write_bytes(original)

    def test_gate_is_enforced_after_artifact_verification_before_training(self):
        args = self.arguments(self.folder / "failed_gate")
        gate = screen.phases.student_gate
        calls = []
        def reject_report(report):
            calls.append(report)
            failed = copy.deepcopy(report)
            failed["source_recoverability"]["positive_mse"] = failed["source_recoverability"]["positive_zero_mse"]
            return gate(failed)
        with ExitStack() as stack:
            self.inputs(stack, allow_constants=False)
            stack.enter_context(patch.object(screen.phases, "student_gate", side_effect=reject_report))
            stack.enter_context(patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Rejected gate trained")))
            with self.assertRaises(screen.phases.EvidenceBlocked):
                screen.build_context(args)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arm"], "joint_positive_mse")
        self.assertFalse(args.output_root.exists())

    def test_tampered_baseline_or_diagnostic_rejected_without_output(self):
        cases = ((self.reference / screen.GROUP_DIRS["B"] / "comparison.json", "checkpoint_sha256"),
                 (self.diagnostics / "joint_positive_mse" / "comparison.json", "checkpoint_sha256"))
        for index, (path, field) in enumerate(cases):
            with self.subTest(path=path):
                original = path.read_bytes()
                args = self.arguments(self.folder / f"bad_reference_{index}")
                try:
                    report = json.loads(original)
                    report[field] = "0" * 64
                    path.write_text(json.dumps(report), encoding="utf-8")
                    with ExitStack() as stack:
                        self.inputs(stack, allow_constants=False)
                        stack.enter_context(patch.object(screen.phases, "student_gate"))
                        with self.assertRaises(ValueError):
                            screen.build_context(args)
                    self.assertFalse(args.output_root.exists())
                finally:
                    path.write_bytes(original)

    def test_summary_preserves_v3_baseline_identity_and_uses_selected_epoch(self):
        summary = screen.make_summary(self.context, self.reports)
        rows = {row["group"]: row for row in summary["rows"]}
        self.assertFalse(summary["formal_test_result"])
        self.assertEqual(set(rows), set("ABCD"))
        for group in "AB":
            self.assertTrue(rows[group]["reused"])
            self.assertEqual(rows[group]["method"], runner.METHOD)
        for group in "CD":
            self.assertFalse(rows[group]["reused"])
            self.assertEqual(rows[group]["method"], training.METHOD)
        for group in "ABC":
            self.assertEqual(summary["D_minus_baseline_macro_S_alpha"][group],
                rows["D"]["validation"]["macro"]["S_alpha"] - rows[group]["validation"]["macro"]["S_alpha"])

    def test_dry_run_is_read_only_and_roots_must_be_disjoint(self):
        output = self.folder / "dry_run"
        with ExitStack() as stack:
            self.inputs(stack, allow_constants=False)
            stack.enter_context(patch.object(screen.phases, "student_gate"))
            stack.enter_context(patch.object(training, "run_student", side_effect=AssertionError("Dry run trained")))
            self.assertEqual(screen.main(["--run-root", str(self.reference), "--diagnostics-root", str(self.diagnostics),
                                         "--output-root", str(output)]), 0)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "disjoint"):
            screen.build_context(self.arguments(self.reference / "nested"))


class MSEScreenLauncherTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix=".screen-launcher-", dir=runner.ROOT)
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_two_real_children_receive_one_physical_gpu_and_preserve_quoted_paths(self):
        fixture = self.folder / "child.py"
        fixture.write_text('''import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--group")
p.add_argument("--directory")
a = p.parse_args()
result = {"group": a.group, "visibility": os.environ.get("CUDA_VISIBLE_DEVICES")}
(Path(a.directory) / "observed.json").write_text(json.dumps(result))
print(json.dumps(result), flush=True)
''', encoding="utf-8")
        output = self.folder / "results with spaces"
        args = screen.parse_args(["--run-root", str(self.folder / "reference with spaces"),
            "--diagnostics-root", str(self.folder / "diagnostics with spaces"),
            "--output-root", str(output), "--gpus", "2,3,4,5", "--execute"])
        captured = []
        original_popen = subprocess.Popen
        def launch(command, **kwargs):
            captured.append(command)
            group = command[command.index("--group") + 1]
            return original_popen([sys.executable, str(fixture), "--group", group,
                "--directory", str(output / screen.GROUP_DIRS[group])], **kwargs)
        def observed(directory, contract):
            return json.loads((directory / "observed.json").read_text())
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,4,5"}), \
             patch.object(screen.subprocess, "Popen", side_effect=launch), \
             patch.object(screen, "student_contract", side_effect=lambda context, group: {"group": group}), \
             patch.object(training, "verify_completed", side_effect=observed), redirect_stdout(io.StringIO()):
            reports = screen._run_jobs(args, {}, ["2", "3", "4", "5"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3,4,5")
        self.assertEqual(len(captured), 2)
        self.assertEqual(reports["C"]["visibility"], "2")
        self.assertEqual(reports["D"]["visibility"], "3")
        for command in captured:
            for flag, value in (("--run-root", args.run_root), ("--diagnostics-root", args.diagnostics_root),
                                ("--output-root", args.output_root)):
                self.assertEqual(command[command.index(flag) + 1], str(value))
            self.assertIn("--execute", command)
            self.assertNotIn("--gpus", command)

    def test_shell_forwards_paths_gpu_list_resume_and_dry_run(self):
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
            RUN_ROOT="reference with spaces/seed-1", DIAGNOSTICS_ROOT="diagnostics with spaces",
            OUTPUT_ROOT="results with spaces/screen", RESUME="1", DRY_RUN="1",
            FIXTURE_ARGUMENTS=(self.folder / "args.txt").as_posix(),
            FIXTURE_VISIBILITY=(self.folder / "visibility.txt").as_posix())
        result = subprocess.run([bash, "tools/run_triprd_mse_screen.sh"], cwd=runner.ROOT,
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = (self.folder / "args.txt").read_text().splitlines()
        self.assertEqual(args[args.index("--gpus") + 1], "2,3,4,5")
        for flag, key in (("--run-root", "RUN_ROOT"), ("--diagnostics-root", "DIAGNOSTICS_ROOT"),
                          ("--output-root", "OUTPUT_ROOT")):
            self.assertEqual(args[args.index(flag) + 1], env[key])
        self.assertIn("--resume", args)
        self.assertNotIn("--execute", args)
        self.assertEqual((self.folder / "visibility.txt").read_text(), "2,3,4,5")


if __name__ == "__main__":
    unittest.main()
