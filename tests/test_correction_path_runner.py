"""Real tiny-checkpoint integration controls for read-only path diagnostics."""
from contextlib import ExitStack, contextmanager
import json
import unittest
from unittest.mock import patch

import torch

import test_mse_screen as fixtures
from tools import diagnose_triprd_correction_path as diagnostic
from tools import run_triprd_correction as runner
from tools import run_triprd_mse_screen as screen
from tools import triprd_mse_screen_training as training


class CorrectionPathRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.MSEScreenIntegrationTests
        # Real two-epoch A/B/C/D, teacher and positive-MSE diagnostic artifacts.
        # Invoke setup without inheriting the unrelated screen test methods.
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.fixture.setUpClass()
        cls.original_hashes = cls.reference_hashes()

    @classmethod
    def reference_hashes(cls):
        return {
            name: cls.fixture.hash_tree(directory)
            for name, directory in (
                ("reference", cls.fixture.reference),
                ("source_diagnostics", cls.fixture.diagnostics),
                ("screen", cls.fixture.output),
            )
        }

    def tearDown(self):
        self.assertEqual(self.reference_hashes(), self.original_hashes)

    def arguments(self, output, *extra):
        return [
            "--run-root", str(self.fixture.reference),
            "--diagnostics-root", str(self.fixture.diagnostics),
            "--screen-root", str(self.fixture.output),
            "--output-root", str(output),
            "--device", "cpu", "--num-workers", "0", *extra,
        ]

    @contextmanager
    def evaluation_only(self):
        with ExitStack() as stack:
            self.fixture.inputs(stack, allow_constants=False)
            # Tiny random fixtures are not a claim that the real source gate
            # passes. Artifact and provenance checks remain fully enabled.
            stack.enter_context(patch.object(screen.phases, "student_gate"))
            for module, name in ((runner, "_fit"), (runner, "_fit_probe"),
                                 (runner, "_train_epoch"), (training, "fit"),
                                 (training, "train_epoch")):
                stack.enter_context(patch.object(module, name,
                    side_effect=AssertionError("Read-only diagnostic attempted training")))
            stack.enter_context(patch.object(torch.optim, "AdamW",
                side_effect=AssertionError("Read-only diagnostic constructed an optimizer")))
            yield

    def test_preflight_is_read_only_and_does_not_create_output(self):
        output = self.fixture.folder / "path_preflight"
        self.assertFalse(output.exists())
        with self.evaluation_only(), patch.object(diagnostic, "run_group",
                side_effect=AssertionError("Preflight evaluated a checkpoint")):
            self.assertEqual(diagnostic.main(self.arguments(output)), 0)
        self.assertFalse(output.exists())

    def test_complete_reports_reproduce_selected_checkpoints_and_resume_skips_evaluation(self):
        output = self.fixture.folder / "path_complete"
        arguments = self.arguments(output, "--groups", "C,D", "--execute")
        with self.evaluation_only():
            self.assertEqual(diagnostic.main(arguments), 0)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(set(summary["groups"]), {"C", "D"})
        contract = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["contract"]
        self.assertFalse(contract["reselect_checkpoint"])
        self.assertFalse(contract["fit_constants"])
        self.assertFalse(contract["optimize"])
        for group in "CD":
            with self.subTest(group=group):
                report = json.loads((output / group / "diagnostics.json").read_text(encoding="utf-8"))
                reference = self.fixture.reports[group]
                self.assertEqual(report["selected_epoch"], reference["best_epoch"])
                self.assertTrue(report["student_unchanged"])
                self.assertTrue(report["coverage"]["full_cod_validation"])
                self.assertTrue(report["coverage"]["full_vdt_validation"])
                self.assertEqual(report["metrics"]["none"], reference["selected_validation"])
                for key in ("positive_mse", "positive_zero_mse", "positive_constant_mse"):
                    self.assertAlmostEqual(report["source_recoverability"][key],
                        reference["source_recoverability"][key], places=10, msg=key)
                self.assertAlmostEqual(report["reproduction"]["macro_S_alpha_delta"], 0., places=12)
                self.assertAlmostEqual(report["reproduction"]["positive_mse_delta"], 0., places=10)
                rows = [json.loads(line) for line in
                    (output / group / "cod_per_image.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), reference["selected_validation"]["num_samples"])
                self.assertEqual(len({row["key"] for row in rows}), len(rows))
                self.assertTrue(all(set(row["S_alpha"]) == set(diagnostic.MODES) for row in rows))
                self.assertTrue(report["path_statistics"])
        before = self.fixture.hash_tree(output)
        with self.evaluation_only(), patch.object(diagnostic, "evaluate_cod",
                side_effect=AssertionError("Completed resume reevaluated COD")), \
                patch.object(diagnostic, "evaluate_source",
                side_effect=AssertionError("Completed resume reevaluated VDT")):
            self.assertEqual(diagnostic.main(arguments + ["--resume"]), 0)
        self.assertEqual(self.fixture.hash_tree(output), before)

    def test_interrupted_resume_only_evaluates_the_unfinished_group(self):
        output = self.fixture.folder / "path_interrupted"
        arguments = self.arguments(output, "--groups", "C,D", "--execute")
        original = diagnostic.run_group

        def interrupted(args, context, group, reference, teacher):
            if group == "D":
                raise RuntimeError("injected diagnostic interruption")
            return original(args, context, group, reference, teacher)

        with self.evaluation_only(), patch.object(diagnostic, "run_group", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected diagnostic interruption"):
                diagnostic.main(arguments)
        completed_c = self.fixture.hash_tree(output / "C")
        seen = []

        def unfinished_only(args, context, group, reference, teacher):
            self.assertEqual(group, "D", "Resume reevaluated a completed group")
            seen.append(group)
            return original(args, context, group, reference, teacher)

        with self.evaluation_only(), patch.object(diagnostic, "run_group", side_effect=unfinished_only):
            self.assertEqual(diagnostic.main(arguments + ["--resume"]), 0)
        self.assertEqual(seen, ["D"])
        self.assertEqual(self.fixture.hash_tree(output / "C"), completed_c)
        self.assertFalse((output / "diagnostic.lock").exists())

    def test_positive_sample_limits_are_explicitly_partial_even_on_tiny_validation(self):
        output = self.fixture.folder / "path_partial"
        with self.evaluation_only():
            self.assertEqual(diagnostic.main(self.arguments(output, "--groups", "D", "--execute",
                "--limit-cod-per-source", "1", "--limit-vdt", "1")), 0)
        report = json.loads((output / "D" / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertFalse(report["coverage"]["full_cod_validation"])
        self.assertFalse(report["coverage"]["full_vdt_validation"])
        self.assertLessEqual(report["coverage"]["cod"], 2)
        self.assertLessEqual(report["coverage"]["vdt"], 1)
        self.assertIsNone(report["reproduction"]["macro_S_alpha_delta"])
        self.assertIsNone(report["reproduction"]["positive_mse_delta"])

    def test_output_overlap_with_any_reference_root_is_rejected(self):
        roots = (self.fixture.reference, self.fixture.diagnostics, self.fixture.output)
        for root in roots:
            for output in (root, root / "nested_path_diagnostic"):
                with self.subTest(output=output), self.evaluation_only():
                    with self.assertRaises(ValueError):
                        diagnostic.main(self.arguments(output))
        with self.evaluation_only(), self.assertRaises(ValueError):
            diagnostic.main(self.arguments(self.fixture.folder))

    def test_tampered_screen_report_and_checkpoint_are_rejected(self):
        directory = self.fixture.output / screen.GROUP_DIRS["D"]
        report_path = directory / "comparison.json"
        checkpoint_path = directory / "best.pt"
        report_bytes, checkpoint_bytes = report_path.read_bytes(), checkpoint_path.read_bytes()
        output = self.fixture.folder / "path_reject_tampering"
        try:
            report = json.loads(report_bytes)
            report["checkpoint_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.evaluation_only(), self.assertRaises(ValueError):
                diagnostic.main(self.arguments(output, "--groups", "D"))
        finally:
            report_path.write_bytes(report_bytes)
        try:
            checkpoint_path.write_bytes(checkpoint_bytes + b"tampered-checkpoint")
            with self.evaluation_only(), self.assertRaises(ValueError):
                diagnostic.main(self.arguments(output, "--groups", "D"))
        finally:
            checkpoint_path.write_bytes(checkpoint_bytes)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
