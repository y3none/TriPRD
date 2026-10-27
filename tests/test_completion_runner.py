"""Real tiny CPU teacher/source/COD optimization, complete transfer and resume."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_completion as runner
from tools import run_triprd_task as legacy


class CompletionRunnerTests(unittest.TestCase):
    @classmethod
    def base_args(cls, out):
        return ["--out", str(out), "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
                "--base-channels", "4", "--image-size", "32", "--batch-size", "2", "--num-workers", "0",
                "--epochs", "3", "--teacher-epochs", "2", "--source-epochs", "2", "--max-samples", "4",
                "--warmup-epochs", "1", "--locator-warmup-epochs", "1", "--train-seed", "11",
                "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
                "--no-amp", "--train-backbone-blocks", "1"]

    @classmethod
    def args(cls, out):
        return cls.base_args(out) + ["--split-manifest", str(cls.manifest)]

    @classmethod
    def stage_args(cls, stage):
        extra = ["--stage", stage]
        if stage == "source":
            extra += ["--teacher-checkpoint", str(cls.work / "teacher/best.pt")]
        if stage == "cod":
            extra += ["--source-checkpoint", str(cls.work / "source/best.pt")]
        return extra

    @classmethod
    def invoke(cls, out, stage, extra=()):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.shared, "_vdt_items", **(
                {"return_value": cls.vdt} if stage != "cod" else {"side_effect": AssertionError("COD accessed VDT")})))
            stack.enter_context(patch.object(runner.shared, "_cod_items", **(
                {"return_value": cls.cod} if stage == "cod" else {"side_effect": AssertionError("source accessed COD")})))
            stack.enter_context(patch.object(runner, "_formal_tests", side_effect=AssertionError("development accessed tests")))
            stream = stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                return runner.main(cls.args(out) + cls.stage_args(stage) + list(extra))
            except BaseException:
                print(stream.getvalue(), file=sys.stderr)
                raise

    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)
        folder = tempfile.TemporaryDirectory(prefix=".completion-v6-test-", dir=ROOT)
        cls.addClassCleanup(folder.cleanup)
        cls.work = Path(folder.name).resolve()
        vdt = MemoryVDTIndex()
        cls.vdt = [legacy.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, vdt), "vdt") for s in vdt.entries]
        cls.cod = [legacy.shared.Item(f"{idx.dataset}/{s.sample_id}", (s, idx), "cod")
                   for idx in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO")) for s in idx.samples()]
        cls.manifest = cls.work / "shared_manifest.json"
        # Produce the original shared development membership using immutable v4.
        legacy_args = legacy.parse_args(["--out", str(cls.work / "prepare"), "--model-variant", "tiny", "--no-pretrained",
            "--max-samples", "4", "--split-seed", "7", "--val-fraction", ".25", "--group", "T1"])
        with (patch.object(legacy.shared, "_vdt_items", return_value=cls.vdt),
              patch.object(legacy.shared, "_cod_items", return_value=cls.cod)):
            _, manifest = legacy.prepare_data(legacy_args)
        cls.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        cls.initial, cls.epochs, cls.cues = {}, {}, []
        original = runner._train_epoch
        original_kd = runner.output_distillation
        def observe(*args, **kwargs):
            model, teacher, loaders, optimizer, scaler, device, a, epoch = args
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1:
                cls.initial[a.stage] = {
                    "model": before, "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "teacher_frozen": teacher is None or all(not p.requires_grad for p in teacher.parameters()),
                    "rng": torch.get_rng_state().clone(),
                    "loader_keys": set(loaders),
                    "trainable": {name for name, p in model.named_parameters() if p.requires_grad}}
            value = original(*args, **kwargs)
            cls.epochs[a.stage, epoch] = (before, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            return value
        def checked_kd(student, teacher, mask, **kwargs):
            if not torch.equal(student["cue"], teacher["cue"]):
                raise AssertionError("Teacher and student received different source RGB cues")
            if student["cue"].requires_grad or teacher["cue"].requires_grad:
                raise AssertionError("Source cue must be detached")
            cls.cues.append(student["cue"].detach().clone())
            return original_kd(student, teacher, mask, **kwargs)
        with patch.object(runner, "_train_epoch", side_effect=observe), patch.object(runner, "output_distillation", side_effect=checked_kd):
            for stage in ("teacher", "source", "cod"):
                if stage == "cod":
                    with patch.object(runner, "_load_teacher", side_effect=AssertionError("COD constructed teacher")):
                        cls.invoke(cls.work / stage, stage)
                else:
                    cls.invoke(cls.work / stage, stage)
        cls.reports = {stage: runner.validate_completed_run(cls.work / stage) for stage in ("teacher", "source", "cod")}

    def test_real_three_stage_route_and_data_isolation(self):
        self.assertGreater(len(self.cues), 0)
        for stage, report in self.reports.items():
            with self.subTest(stage=stage):
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["completed_epochs"], 3 if stage == "cod" else 2)
                self.assertFalse(report["formal"])
                self.assertEqual(report["test_status"], "skipped")
                self.assertTrue(self.initial[stage]["teacher_frozen"])
                self.assertEqual(self.initial[stage]["loader_keys"], {"cod_train", "cod_val"} if stage == "cod" else {"vdt_train", "vdt_val"})
                self.assertEqual({row["train"]["amp_skipped_steps"] for row in report["history"]}, {0})
                inactive = "vdt_train" if stage == "cod" else "cod_train"
                self.assertEqual(report["sample_counts"][inactive], 0)

    def test_source_transfers_every_tensor_except_fresh_cod_locator(self):
        state = runner.read_checkpoint(self.work / "source/best.pt")["model"]
        before = self.initial["source"]["model"]
        target = self.initial["cod"]["model"]
        self.assertTrue(any(key.startswith("refiner.") for key in target))
        for name, tensor in target.items():
            expected = before[name] if name.startswith("locators.cod.") else state[name]
            self.assertTrue(torch.equal(tensor, expected), name)
        report = self.reports["cod"]
        self.assertEqual(report["cod_locator_before_sha256"], report["cod_locator_after_sha256"])
        self.assertEqual(report["refiner_after_sha256"], report["source_provenance"]["refiner_source_sha256"])
        self.assertNotEqual(report["refiner_before_sha256"], report["refiner_after_sha256"])

    def test_cod_warmup_only_locator_then_refiner_joint_updates(self):
        before, after = self.epochs["cod", 1]
        changed = {name for name in before if not torch.equal(before[name], after[name])}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("locators.cod.") for name in changed), changed)
        before, after = self.epochs["cod", 2]
        changed = {name for name in before if not torch.equal(before[name], after[name])}
        self.assertTrue(any(name.startswith("refiner.") for name in changed))
        self.assertTrue(any(name.startswith("locators.cod.") for name in changed))
        self.assertEqual(self.initial["cod"]["optimizer"]["state"], {})
        self.assertEqual([row["phase"] for row in self.reports["cod"]["history"]], ["locator_warmup", "joint", "joint"])

    def test_exact_epoch_resume_for_every_stage(self):
        original = runner._train_epoch
        def interrupted(*args, **kwargs):
            if args[-1] == 2:
                raise RuntimeError("injected epoch boundary interruption")
            return original(*args, **kwargs)
        for stage in ("teacher", "source", "cod"):
            out = self.work / "resume" / stage
            with self.subTest(stage=stage):
                with patch.object(runner, "_train_epoch", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "injected epoch boundary"):
                        self.invoke(out, stage)
                self.invoke(out, stage, ["--resume"])
                expected = runner.read_checkpoint(self.work / stage / "last.pt")
                actual = runner.read_checkpoint(out / "last.pt")
                self.assertEqual(expected["best_epoch"], actual["best_epoch"])
                self.assertEqual(expected["scheduler"], actual["scheduler"])
                self.assertTrue(torch.equal(expected["rng"]["torch"], actual["rng"]["torch"]))
                for name, tensor in expected["model"].items():
                    self.assertTrue(torch.equal(tensor, actual["model"][name]), name)

    def test_completed_resume_is_noop_and_changed_contract_rejected(self):
        for stage in ("teacher", "source", "cod"):
            path = self.work / stage
            before = runner.sha256_file(path / "best.pt")
            with patch.object(runner, "_train_epoch", side_effect=AssertionError("completed run retrained")):
                self.invoke(path, stage, ["--resume"])
            self.assertEqual(before, runner.sha256_file(path / "best.pt"))
        with self.assertRaises(ValueError):
            self.invoke(self.work / "cod", "cod", ["--resume", "--lr", ".0002"])
        with self.assertRaisesRegex(ValueError, "Existing run"):
            self.invoke(self.work / "cod", "cod")

    def test_source_validation_uses_gt_without_kd(self):
        a = runner.parse_args(self.args(self.work / "validate") + self.stage_args("source"))
        with patch.object(runner.shared, "_vdt_items", return_value=self.vdt):
            splits, _ = runner.prepare_data(a)
        model = runner.CompletionStudent(**runner.model_options(a, pretrained=False))
        model.load_state_dict(runner.read_checkpoint(self.work / "source/best.pt")["model"])
        loader = runner._loaders(a, splits, torch.device("cpu"))["vdt_val"]
        with patch.object(runner, "output_distillation", side_effect=AssertionError("KD in selection")):
            value = runner._validate_source(model, loader, torch.device("cpu"), a)
        self.assertEqual(value["objective"], "supervised_gt_only")
        self.assertAlmostEqual(value["loss"], -self.reports["source"]["best_score"])

    def test_detached_source_cue_requires_positive_coarse_supervision(self):
        for stage in ("teacher", "source", "cod"):
            a = runner.parse_args(self.args(self.work / "never_run") + self.stage_args(stage) + ["--aux-weight", "0"])
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "coarse GT"):
                runner.validate_args(a)

    def test_reports_reject_incomplete_or_wrong_transfer_evidence(self):
        original = self.reports["cod"]
        path = self.work / "cod/comparison.json"
        mutations = (
            lambda report: report.update(status="running"),
            lambda report: report.update(completed_epochs=2),
            lambda report: report.update(cod_locator_after_sha256="0" * 64),
            lambda report: report.update(refiner_after_sha256="0" * 64),
        )
        for mutate in mutations:
            report = copy.deepcopy(original)
            mutate(report)
            path.write_text(json.dumps(report), encoding="utf-8")
            try:
                with self.assertRaises(ValueError):
                    runner.validate_completed_run(path)
            finally:
                path.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaises(ValueError):
            runner.validate_completed_run(self.work / "cod", formal=True)


if __name__ == "__main__":
    unittest.main()
