"""Real source-only probe optimization, isolation, constants and resume checks."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_correction as runner


class CorrectionSourceProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)
        cls.temporary = tempfile.TemporaryDirectory(prefix=".correction-probe-", dir=runner.ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        index = MemoryVDTIndex()
        cls.vdt = [runner.shared.Item(f"VDT/{sample['rgb'].sample_id}", (sample, index), "vdt")
                   for sample in index.entries]
        cls.cod = [runner.shared.Item(f"{index.dataset}/{sample.sample_id}", (sample, index), "cod")
                   for index in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO")) for sample in index.samples()]
        cls.teacher = cls.work / "teacher"
        with patch.object(runner.shared, "_vdt_items", return_value=cls.vdt), \
             patch.object(runner.shared, "_cod_items", return_value=cls.cod), redirect_stdout(io.StringIO()):
            runner.main(cls.arguments(cls.teacher) + ["--stage", "teacher"])
        cls.teacher_digest = runner.sha256_file(cls.teacher / "best.pt")
        cls.reports = {}
        for group in "CD":
            out = cls.work / group
            cls.invoke(out, group)
            cls.reports[group] = json.loads((out / "comparison.json").read_text(encoding="utf-8"))

    @classmethod
    def arguments(cls, out):
        return ["--out", str(out), "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
            "--base-channels", "4", "--image-size", "32", "--batch-size", "2", "--num-workers", "0",
            "--epochs", "2", "--teacher-epochs", "2", "--max-samples", "4", "--warmup-epochs", "1",
            "--train-seed", "11", "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
            "--skip-test", "--no-amp"]

    @classmethod
    def invoke(cls, out, group="D", extra=()):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.shared, "_vdt_items", return_value=cls.vdt))
            for obj, name in ((runner.shared, "_cod_items"), (runner, "_validate_cod"),
                              (runner, "_formal_tests"), (runner, "_constant_probe")):
                stack.enter_context(patch.object(obj, name, side_effect=AssertionError(f"Probe must not call {name}")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return runner.main(cls.arguments(out) + ["--stage", "probe", "--group", group,
                "--teacher-checkpoint", str(cls.teacher / "best.pt"), *extra])

    def test_real_training_is_source_only_and_uses_fixed_final_checkpoint(self):
        for group, report in self.reports.items():
            with self.subTest(group=group):
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["stage"], "probe")
                self.assertTrue(report["diagnostic"])
                self.assertFalse(report["formal"])
                self.assertFalse(report["cod_access"])
                self.assertFalse(report["validation_used_for_selection"])
                self.assertEqual(report["selection_metric"], "fixed_last_epoch")
                self.assertEqual(report["selected_epoch"], 2)
                self.assertEqual(report["test"], {})
                self.assertEqual(report["test_status"], "skipped")
                self.assertEqual(report["sample_counts"]["cod_train"], 0)
                self.assertEqual(report["sample_counts"]["cod_val"], 0)
                self.assertFalse((self.work / group / "best.pt").exists())
                state = runner._read_checkpoint(self.work / group / "last.pt")
                self.assertEqual(state["stage"], "probe")
                self.assertEqual(state["epoch"], 2)
                self.assertNotIn("best_score", state)
                for row in report["history"]:
                    self.assertNotIn("validation", row)
                    self.assertNotIn("cod_total", row["train"])
                    self.assertEqual(row["train"]["source_images"], report["sample_counts"]["vdt_train"])
                    self.assertGreater(row["train"]["grad_norm"], 0)
                    self.assertIn("source_recoverability", row)
                probe = report["source_recoverability"]
                self.assertGreater(probe["count"], 0)
                self.assertIn("positive_count", probe)
                self.assertIn("constant_mse", probe)
                self.assertIn("positive_constant_mse", probe)
                self.assertIn("positive_all_edge_constant_mse", probe)
                self.assertIn("student_task_error", probe)
                self.assertEqual(probe, report["history"][-1]["source_recoverability"])
                self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in probe.values()))
                self.assertTrue(report["frozen_generic_encoder_verified"])
        self.assertEqual(self.reports["C"]["initial_state_sha256"], self.reports["D"]["initial_state_sha256"])
        self.assertEqual(runner.sha256_file(self.teacher / "best.pt"), self.teacher_digest)

    def test_probe_changes_adapters_but_keeps_encoder_and_unused_cod_head(self):
        from models.triprd_correction import CorrectionStudent
        a = runner.parse_args(self.arguments(self.work / "unused"))
        runner.shared.seed_everything(a.train_seed)
        initial = CorrectionStudent(**runner._model_options(a)).state_dict()
        final = runner._read_checkpoint(self.work / "D" / "last.pt")["model"]
        changed = {key for key in initial if not torch.equal(initial[key], final[key])}
        self.assertTrue(changed)
        self.assertTrue(any("adapter" in key for key in changed))
        self.assertFalse(any(key.startswith("encoder.") for key in changed))
        self.assertFalse(any(key.startswith("task_heads.cod.") for key in changed))

    def test_resume_is_exact_after_a_real_interrupted_epoch(self):
        out = self.work / "interrupted"
        train = runner._train_epoch
        def interrupted(*args):
            if args[-1] == 2:
                raise RuntimeError("injected probe interruption")
            return train(*args)
        with patch.object(runner, "_train_epoch", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected probe"):
                self.invoke(out)
        self.assertEqual(runner._read_checkpoint(out / "last.pt")["epoch"], 1)
        self.invoke(out, extra=["--resume"])
        actual = runner._read_checkpoint(out / "last.pt")
        expected = runner._read_checkpoint(self.work / "D" / "last.pt")
        for name, tensor in expected["model"].items():
            self.assertTrue(torch.equal(actual["model"][name], tensor), name)
        self.assertEqual(actual["history"][-1]["source_recoverability"], expected["history"][-1]["source_recoverability"])
        self.invoke(out, extra=["--resume"])

    def test_interval_always_records_final_and_validation_never_selects_epoch(self):
        out = self.work / "interval"
        self.invoke(out, extra=["--val-metrics-every", "5"])
        report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
        self.assertNotIn("source_recoverability", report["history"][0])
        self.assertIn("source_recoverability", report["history"][1])
        self.assertEqual(report["selected_epoch"], 2)

    def test_probe_is_not_a_formal_student_or_teacher_checkpoint(self):
        path = self.work / "D" / "last.pt"
        with self.assertRaisesRegex(ValueError, "student checkpoint"):
            runner.main(self.arguments(self.work / "invalid_eval") + [
                "--eval-only", "--no-skip-test", "--cod-checkpoint", str(path)])
        with self.assertRaisesRegex(ValueError, "identical contract"):
            with patch.object(runner.shared, "_vdt_items", return_value=self.vdt), \
                 patch.object(runner.shared, "_cod_items", return_value=self.cod):
                runner.main(self.arguments(self.work / "D") + ["--stage", "student", "--group", "D",
                    "--teacher-checkpoint", str(self.teacher / "best.pt"), "--resume"])

    def test_probe_rejects_tests_other_groups_and_changed_teacher_contract(self):
        for extra in (["--group", "A"], ["--group", "E"], ["--no-skip-test"], ["--cod-train-mode", "full"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.invoke(self.work / "invalid", extra=extra)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.invoke(self.work / "wrong_teacher", extra=["--lr", ".002"])
        manifest = json.loads((self.teacher / "split_manifest.json").read_text(encoding="utf-8"))
        manifest["vdt_train"][0], manifest["vdt_val"][0] = manifest["vdt_val"][0], manifest["vdt_train"][0]
        path = self.work / "changed_source.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.invoke(self.work / "changed_source", extra=["--split-manifest", str(path)])

    def test_teacher_without_positive_train_constants_is_explicitly_rejected(self):
        directory = self.work / "teacher_without_positive_constants"
        directory.mkdir()
        checkpoint = runner._read_checkpoint(self.teacher / "best.pt")
        del checkpoint["positive_constant_correction"]
        runner._save_pt(directory / "best.pt", checkpoint)
        report = json.loads((self.teacher / "comparison.json").read_text(encoding="utf-8"))
        report["checkpoint_sha256"] = runner.sha256_file(directory / "best.pt")
        runner._save_json(directory / "comparison.json", report)
        with self.assertRaisesRegex(ValueError, "positive-edge TRAIN-fitted"):
            self.invoke(self.work / "reject_missing_constant", extra=[
                "--teacher-checkpoint", str(directory / "best.pt")])


if __name__ == "__main__":
    unittest.main()
