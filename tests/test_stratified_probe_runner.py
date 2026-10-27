"""Real tiny controls for paired, source-only stratified correction probes."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_probe_diagnostics as fixtures
from tools import diagnose_triprd_correction_probe as original
from tools import run_triprd_correction as runner
from tools import run_triprd_stratified_probe as diagnostic


class StratifiedProbeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.ProbeDiagnosticIntegrationTests
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.fixture.setUpClass()
        cls.output = cls.fixture.folder / "paired_stratified"
        cls.old_hashes = cls.hash_references()
        cls.reports = {}
        for arm in diagnostic.ARMS:
            cls.invoke(arm)
            cls.reports[arm] = cls.read_report(arm)

    @staticmethod
    def hash_tree(directory):
        return {path.relative_to(directory).as_posix(): runner.sha256_file(path)
                for path in directory.rglob("*") if path.is_file()}

    @classmethod
    def hash_references(cls):
        return {"reference": cls.hash_tree(cls.fixture.reference),
                "original_diagnostics": cls.hash_tree(cls.fixture.output)}

    @classmethod
    def arguments(cls, output, extra=()):
        return ["--run-root", str(cls.fixture.reference), "--output-root", str(output),
                "--epochs", "2", *extra]

    @classmethod
    def invoke(cls, arm, *, output=None, extra=()):
        with ExitStack() as stack:
            cls.fixture.forbid_cod(stack)
            return diagnostic.main(cls.arguments(output or cls.output,
                ("--execute", "--arm", arm, *extra)))

    @classmethod
    def read_report(cls, arm, output=None):
        path = (output or cls.output) / arm / "comparison.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def tearDown(self):
        self.assertEqual(self.hash_references(), self.old_hashes)

    def assert_state_equal(self, actual, expected, path="state"):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, path)
            self.assertTrue(torch.equal(actual, expected), path)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key, value in expected.items():
                self.assert_state_equal(actual[key], value, f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected), path)
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self.assert_state_equal(actual[index], value, f"{path}[{index}]")
        else:
            self.assertEqual(actual, expected, path)

    def test_both_fresh_arms_have_fixed_budget_complete_histories_and_no_cod_access(self):
        self.assertEqual(set(self.reports), {"pooled_mse", "stratified_mse"})
        initial = self.fixture.reports["joint_positive_mse"]["initial_state_sha256"]
        for arm, report in self.reports.items():
            with self.subTest(arm=arm):
                self.assertEqual(report["method"], diagnostic.METHOD)
                self.assertNotEqual(report["method"], original.METHOD)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["initial_state_sha256"], initial)
                self.assertEqual(report["completed_epochs"], 2)
                self.assertEqual(report["selected_epoch"], 2)
                self.assertTrue(report["diagnostic"])
                self.assertFalse(report["formal"])
                self.assertFalse(report["cod_access"])
                self.assertFalse(report["validation_used_for_selection"])
                self.assertEqual(report["selection_metric"], "fixed_last_epoch")
                self.assertTrue(report["frozen_generic_encoder_verified"])
                self.assertTrue(report["frozen_teacher_verified"])
                self.assertEqual([row["epoch"] for row in report["history"]], [1, 2])
                self.assertEqual(report["source_recoverability"], report["history"][-1]["source_recoverability"])
                self.assertTrue(report["source_diagnostics"])
                self.assertTrue(report["source_train_diagnostics"]["path_statistics"])
                self.assertEqual(report["source_diagnostics"], report["history"][-1]["source_diagnostics"])
                self.assertEqual(json.loads((self.output / arm / "history.json").read_text(encoding="utf-8")),
                                 report["history"])
                self.assertFalse((self.output / arm / "best.pt").exists())
                self.assertEqual(runner.sha256_file(self.output / arm / "last.pt"), report["checkpoint_sha256"])
                for row in report["history"]:
                    self.assertGreater(row["train"]["optimizer_steps"], 0)
                    self.assertEqual(row["train"]["amp_skipped_steps"], 0)
                    self.assertNotIn("validation", row)
                    self.assertNotIn("cod_total", row["train"])
                    self.assertIn("source_diagnostics", row)
                json.dumps(report, allow_nan=False)

    def test_pooled_control_exactly_reproduces_old_joint_positive_mse(self):
        previous = torch.load(self.fixture.output / "joint_positive_mse" / "last.pt",
                              map_location="cpu", weights_only=True)
        current = torch.load(self.output / "pooled_mse" / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(current["initial_state_sha256"], previous["initial_state_sha256"])
        for field in ("model", "optimizer", "scheduler", "scaler", "rng", "initial_source_recoverability"):
            self.assert_state_equal(current[field], previous[field], field)
        for old_row, new_row in zip(previous["history"], current["history"]):
            self.assertEqual(new_row["epoch"], old_row["epoch"])
            self.assertEqual(new_row["lr"], old_row["lr"])
            self.assertEqual(new_row["source_recoverability"], old_row["source_recoverability"])
            for field in ("total", "source_supervised", "grad_norm", "optimizer_steps", "amp_skipped_steps"):
                self.assertEqual(new_row["train"][field], old_row["train"][field], field)

    def test_epoch_resume_restores_weights_optimizer_and_all_rng_states(self):
        arm, output = "stratified_mse", self.fixture.folder / "stratified_interrupted"
        train = diagnostic._train_epoch

        def interrupt_second_epoch(*args, **kwargs):
            if args[-1] == 2:
                raise RuntimeError("injected stratified interruption")
            return train(*args, **kwargs)

        with patch.object(diagnostic, "_train_epoch", side_effect=interrupt_second_epoch):
            with self.assertRaisesRegex(RuntimeError, "injected stratified interruption"):
                self.invoke(arm, output=output)
        partial = torch.load(output / arm / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(partial["epoch"], 1)
        # Recovery must restore these independently of ambient process state.
        random.seed(987)
        np.random.seed(654)
        torch.manual_seed(321)
        self.invoke(arm, output=output, extra=("--resume",))
        actual = torch.load(output / arm / "last.pt", map_location="cpu", weights_only=True)
        expected = torch.load(self.output / arm / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(actual["epoch"], 2)
        for field in ("model", "optimizer", "scheduler", "scaler", "rng", "initial_source_recoverability"):
            self.assert_state_equal(actual[field], expected[field], field)
        for actual_row, expected_row in zip(actual["history"], expected["history"]):
            self.assertEqual(actual_row["source_recoverability"], expected_row["source_recoverability"])
            self.assertEqual(actual_row["source_diagnostics"], expected_row["source_diagnostics"])

    def test_complete_resume_reuses_verified_results_without_training_or_evaluation(self):
        before = self.hash_tree(self.output)
        with patch.object(diagnostic, "_train_epoch", side_effect=AssertionError("Complete arm retrained")), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Complete arm built optimizer")), \
             patch.object(diagnostic, "evaluate_source_detailed", side_effect=AssertionError("Complete arm reevaluated")):
            for arm in diagnostic.ARMS:
                self.invoke(arm, extra=("--resume",))
        self.assertEqual(self.hash_tree(self.output), before)

    def test_dry_run_creates_no_files_and_reads_only_source_evidence(self):
        output = self.fixture.folder / "stratified_plan"
        with ExitStack() as stack:
            self.fixture.forbid_cod(stack)
            stack.enter_context(patch.object(diagnostic, "run_arm", side_effect=AssertionError("Dry run trained")))
            self.assertEqual(diagnostic.main(self.arguments(output)), 0)
        self.assertFalse(output.exists())

    def test_tampered_reference_checkpoint_and_source_fingerprint_are_rejected(self):
        teacher_report = self.fixture.reference / "teacher" / "comparison.json"
        original_bytes = teacher_report.read_bytes()
        output = self.fixture.folder / "stratified_bad_reference"
        for mutation in ("checkpoint", "source"):
            with self.subTest(mutation=mutation):
                try:
                    report = json.loads(original_bytes)
                    if mutation == "checkpoint":
                        report["checkpoint_sha256"] = "0" * 64
                    else:
                        key = next(iter(report["protocol"]["source_sha256"]))
                        report["protocol"]["source_sha256"][key] = "0" * 64
                        report["contract_sha256"] = runner._digest(report["protocol"])
                    teacher_report.write_text(json.dumps(report), encoding="utf-8")
                    with patch.object(torch.optim, "AdamW", side_effect=AssertionError("Bad evidence trained")):
                        with self.assertRaises(ValueError):
                            self.invoke("pooled_mse", output=output)
                    self.assertFalse(output.exists())
                finally:
                    teacher_report.write_bytes(original_bytes)

    def test_completed_report_and_checkpoint_tampering_are_rejected(self):
        directory = self.output / "stratified_mse"
        report_path, checkpoint_path = directory / "comparison.json", directory / "last.pt"
        report_bytes, checkpoint_bytes = report_path.read_bytes(), checkpoint_path.read_bytes()
        for mutation in ("history", "checkpoint"):
            with self.subTest(mutation=mutation):
                try:
                    if mutation == "history":
                        report = json.loads(report_bytes)
                        report["history"] = report["history"][:-1]
                        report_path.write_text(json.dumps(report), encoding="utf-8")
                    else:
                        checkpoint_path.write_bytes(checkpoint_bytes + b"tampered")
                    with patch.object(torch.optim, "AdamW", side_effect=AssertionError("Tampered arm retrained")):
                        with self.assertRaises(ValueError):
                            self.invoke("stratified_mse", extra=("--resume",))
                finally:
                    report_path.write_bytes(report_bytes)
                    checkpoint_path.write_bytes(checkpoint_bytes)

    def test_changed_new_diagnostic_source_invalidates_resume_before_training(self):
        original_hash = diagnostic.sha256_file
        changed_path = (runner.ROOT / "tools/stratified_correction_loss.py").resolve()

        def changed_source(path):
            return "0" * 64 if Path(path).resolve() == changed_path else original_hash(path)

        before = self.hash_tree(self.output)
        with patch.object(diagnostic, "sha256_file", side_effect=changed_source), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Changed source retrained")):
            with self.assertRaisesRegex(ValueError, "identical contract"):
                self.invoke("stratified_mse", extra=("--resume",))
        self.assertEqual(self.hash_tree(self.output), before)


class StratifiedProbeLauncherTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix=".stratified-launcher-", dir=runner.ROOT)
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_two_real_children_receive_distinct_physical_gpus_and_quoted_paths(self):
        child_path = self.folder / "child.py"
        child_path.write_text('''import argparse, json, os
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
            "--output-root", str(output), "--gpus", "2,3", "--epochs", "2", "--execute"])
        context = {"recipe": {"test": "one physical GPU per independent arm"}}
        captured, real_popen = [], subprocess.Popen

        def child(command, **kwargs):
            captured.append(command)
            arm = command[command.index("--arm") + 1]
            return real_popen([sys.executable, str(child_path), "--arm", arm,
                "--output-root", str(output)], **kwargs)

        def observed(directory, contract):
            return json.loads((directory / "observed.json").read_text(encoding="utf-8"))

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}), \
             patch.object(diagnostic.subprocess, "Popen", side_effect=child), \
             patch.object(diagnostic, "verify_completed", side_effect=observed), redirect_stdout(io.StringIO()):
            reports = diagnostic._run_jobs(args, context, ["2", "3"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(set(reports), set(diagnostic.ARMS))
        self.assertEqual([reports[arm]["visibility"] for arm in diagnostic.ARMS], ["2", "3"])
        self.assertEqual(len(captured), 2)
        for command in captured:
            self.assertIsInstance(command, list)
            self.assertEqual(command[command.index("--run-root") + 1], str(args.run_root))
            self.assertEqual(command[command.index("--output-root") + 1], str(args.output_root))
            self.assertEqual(command[command.index("--epochs") + 1], "2")
            self.assertIn("--execute", command)
            self.assertNotIn("--gpus", command)
        for arm in diagnostic.ARMS:
            self.assertIn(reports[arm]["visibility"], (output / arm / "run.log").read_text(encoding="utf-8"))

    def test_shell_preserves_paths_gpu_visibility_and_budget(self):
        bash = shutil.which("bash")
        if bash is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
            bash = "C:/Program Files/Git/bin/bash.exe"
        if bash is None:
            self.skipTest("Bash unavailable")
        fake_python = self.folder / "fake_python.sh"
        fake_python.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FIXTURE_ARGUMENTS"
printf '%s' "${CUDA_VISIBLE_DEVICES-unset}" > "$FIXTURE_VISIBILITY"
''', encoding="utf-8", newline="\n")
        fake_python.chmod(0o700)
        environment = os.environ.copy()
        environment.update(PYTHON=fake_python.as_posix(), GPUS="2,3", CUDA_VISIBLE_DEVICES="2,3",
            RUN_ROOT="reference with spaces/seed-1", OUTPUT_ROOT="results with spaces/stratified",
            PROBE_EPOCHS="2", RESUME="1", DRY_RUN="1",
            FIXTURE_ARGUMENTS=(self.folder / "args.txt").as_posix(),
            FIXTURE_VISIBILITY=(self.folder / "visibility.txt").as_posix())
        result = subprocess.run([bash, "tools/run_triprd_stratified_probe.sh"], cwd=runner.ROOT,
            env=environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        arguments = (self.folder / "args.txt").read_text().splitlines()
        self.assertEqual(arguments[arguments.index("--gpus") + 1], "2,3")
        self.assertEqual(arguments[arguments.index("--run-root") + 1], environment["RUN_ROOT"])
        self.assertEqual(arguments[arguments.index("--output-root") + 1], environment["OUTPUT_ROOT"])
        self.assertEqual(arguments[arguments.index("--epochs") + 1], "2")
        self.assertIn("--resume", arguments)
        self.assertNotIn("--execute", arguments)
        self.assertEqual((self.folder / "visibility.txt").read_text(), "2,3")


if __name__ == "__main__":
    unittest.main()
