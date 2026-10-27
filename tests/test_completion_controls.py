"""Real CPU control routes, micro accumulation, provenance and exact resume."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_completion_control as runner
from tools import run_triprd_task as legacy
from tools import evaluate_triprd_completion_control as exporter
from models.triprd_completion import CompletionTeacher as FrozenTeacher
from models.triprd_completion_control import CompletionTeacher
from evaluation.task_protocol import DATASET_COUNTS, METRIC_FIELDS, PREDICTION_PROTOCOL


class CompletionControlTests(unittest.TestCase):
    @classmethod
    def arguments(cls, out, group, stage, extra=()):
        result = ["--out", str(out), "--group", group, "--stage", stage,
            "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
            "--base-channels", "4", "--image-size", "32", "--batch-size", "4",
            "--micro-batch-size", "2", "--num-workers", "0", "--epochs", "2",
            "--teacher-epochs", "2", "--source-epochs", "2", "--max-samples", "8",
            "--warmup-epochs", "1", "--locator-warmup-epochs", "0", "--train-seed", "11",
            "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
            "--no-amp", "--train-backbone-blocks", "1", "--split-manifest", str(cls.manifest)]
        if stage == "source" and group in {"C", "D"}:
            result += ["--teacher-checkpoint", str(cls.work / group / "teacher/best.pt")]
        if stage == "cod" and group != "A":
            result += ["--source-checkpoint", str(cls.work / group / "source/best.pt")]
        return result + list(extra)

    @classmethod
    def invoke(cls, out, group, stage, extra=()):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.shared, "_vdt_items", **(
                {"return_value": cls.vdt} if stage != "cod" else {"side_effect": AssertionError("COD accessed VDT")})))
            stack.enter_context(patch.object(runner.shared, "_cod_items", **(
                {"return_value": cls.cod} if stage == "cod" else {"side_effect": AssertionError("source accessed COD")})))
            stack.enter_context(patch.object(runner, "_formal_tests", side_effect=AssertionError("development accessed tests")))
            if group in {"A", "B"} or stage != "source":
                stack.enter_context(patch.object(runner, "_load_teacher", side_effect=AssertionError("unexpected teacher load")))
            if group in {"A", "B"}:
                stack.enter_context(patch.object(runner, "output_distillation", side_effect=AssertionError("unexpected KD")))
            stream = stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                return runner.main(cls.arguments(out, group, stage, extra))
            except BaseException:
                print(stream.getvalue(), file=sys.stderr)
                raise

    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)
        folder = tempfile.TemporaryDirectory(prefix=".completion-controls-", dir=ROOT)
        cls.addClassCleanup(folder.cleanup)
        cls.work = Path(folder.name)
        vdt = MemoryVDTIndex(count=8)
        cls.vdt = [legacy.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, vdt), "vdt") for s in vdt.entries]
        cls.cod = [legacy.shared.Item(f"{idx.dataset}/{s.sample_id}", (s, idx), "cod")
                   for idx in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO")) for s in idx.samples()]
        cls.manifest = cls.work / "shared_manifest.json"
        a = legacy.parse_args(["--out", str(cls.work / "prepare"), "--model-variant", "tiny", "--no-pretrained",
            "--split-seed", "7", "--val-fraction", ".25", "--group", "T1", "--max-samples", "8"])
        with patch.object(legacy.shared, "_vdt_items", return_value=cls.vdt), patch.object(legacy.shared, "_cod_items", return_value=cls.cod):
            _, manifest = legacy.prepare_data(a)
        cls.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        cls.routes = {"A": ("cod",), "B": ("source", "cod"),
                      "C": ("teacher", "source", "cod"), "D": ("teacher", "source", "cod")}
        cls.initial, cls.first_changed, cls.reports, cls.kd_calls = {}, {}, {}, []
        original_train, original_kd = runner._train_epoch, runner.output_distillation
        def observe(*args, **kwargs):
            model, teacher, loaders, optimizer, scaler, device, config, epoch = args
            key = (config.group, config.stage)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch == 1:
                cls.initial[key] = {"model": before, "rng": torch.get_rng_state().clone(),
                    "optimizer": copy.deepcopy(optimizer.state_dict()), "loaders": set(loaders)}
            result = original_train(*args, **kwargs)
            if epoch == 1:
                cls.first_changed[key] = {name for name, value in model.state_dict().items() if not torch.equal(value, before[name])}
            return result
        def checked_kd(student, teacher, mask, **kwargs):
            torch.testing.assert_close(student["cue"], teacher["cue"], rtol=0, atol=0)
            if student["cue"].requires_grad or teacher["cue"].requires_grad:
                raise AssertionError("source cue must be detached")
            cls.kd_calls.append(1)
            return original_kd(student, teacher, mask, **kwargs)
        with patch.object(runner, "_train_epoch", side_effect=observe), patch.object(runner, "output_distillation", side_effect=checked_kd):
            for group, stages in cls.routes.items():
                for stage in stages:
                    cls.invoke(cls.work / group / stage, group, stage)
                    cls.reports[group, stage] = runner.validate_completed_run(cls.work / group / stage)

    def test_all_routes_isolate_data_and_report_exact_effective_steps(self):
        self.assertGreater(len(self.kd_calls), 0)
        for (group, stage), report in self.reports.items():
            with self.subTest(group=group, stage=stage):
                self.assertEqual(report["completed_epochs"], 2)
                self.assertFalse(report["formal"])
                self.assertEqual(report["test_status"], "skipped")
                self.assertEqual(self.initial[group, stage]["loaders"], {"cod_train", "cod_val"} if stage == "cod" else {"vdt_train", "vdt_val"})
                self.assertEqual(report["stage_exposure"]["teacher_in_stage"], stage == "source" and group in {"C", "D"})
                for row in report["history"]:
                    self.assertEqual(row["train"]["optimizer_steps"], 2)
                    self.assertEqual(row["train"]["micro_steps"], 3)
                    self.assertEqual(row["train"]["observed_images"], 6)
                    self.assertEqual(row["train"]["micro_batch_size"], 2)

    def test_initialization_rng_and_full_transfer_are_paired(self):
        hashes = {self.reports[group, "cod"]["generic_initial_state_sha256"] for group in self.routes}
        self.assertEqual(len(hashes), 1)
        cod_rng = self.initial["A", "cod"]["rng"]
        source_rng = self.initial["B", "source"]["rng"]
        for group in self.routes:
            self.assertTrue(torch.equal(cod_rng, self.initial[group, "cod"]["rng"]))
            self.assertEqual(self.initial[group, "cod"]["optimizer"]["state"], {})
            cod = self.initial[group, "cod"]["model"]
            if group == "A":
                self.assertIsNone(self.reports[group, "cod"]["source_provenance"])
                continue
            self.assertTrue(torch.equal(source_rng, self.initial[group, "source"]["rng"]))
            source = runner.read_checkpoint(self.work / group / "source/best.pt")["model"]
            for name in cod:
                expected = self.initial["A", "cod"]["model"][name] if name.startswith("locators.cod.") else source[name]
                self.assertTrue(torch.equal(expected, cod[name]), (group, name))
        self.assertEqual(self.reports["C", "teacher"]["generic_initial_state_sha256"], self.reports["D", "teacher"]["generic_initial_state_sha256"])
        self.assertEqual(self.reports["C", "teacher"]["parameters"], self.reports["D", "teacher"]["parameters"])
        self.assertIsNone(self.reports["B", "source"]["teacher_provenance"])

    def test_all_cod_groups_jointly_train_first_epoch_with_same_lr(self):
        schedules = []
        for group in self.routes:
            changed = self.first_changed[group, "cod"]
            self.assertTrue(any(k.startswith("refiner.") for k in changed))
            self.assertTrue(any(k.startswith("locators.cod.") for k in changed))
            self.assertEqual({r["phase"] for r in self.reports[group, "cod"]["history"]}, {"joint"})
            schedules.append([row["lr"] for row in self.reports[group, "cod"]["history"]])
        self.assertTrue(all(value == schedules[0] for value in schedules))

    def test_exact_resume_for_every_control_stage(self):
        original = runner._train_epoch
        def interrupt(*args, **kwargs):
            if args[-1] == 2:
                raise RuntimeError("injected epoch interruption")
            return original(*args, **kwargs)
        for group, stages in self.routes.items():
            for stage in stages:
                with self.subTest(group=group, stage=stage):
                    out = self.work / "resume" / group / stage
                    with patch.object(runner, "_train_epoch", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "injected"):
                        self.invoke(out, group, stage)
                    self.invoke(out, group, stage, ["--resume"])
                    expected = runner.read_checkpoint(self.work / group / stage / "last.pt")
                    actual = runner.read_checkpoint(out / "last.pt")
                    self.assertEqual(expected["scheduler"], actual["scheduler"])
                    self.assertTrue(torch.equal(expected["rng"]["torch"], actual["rng"]["torch"]))
                    for name in expected["model"]:
                        self.assertTrue(torch.equal(expected["model"][name], actual["model"][name]), name)

    def test_completed_resume_and_changed_route_rejected(self):
        for group in self.routes:
            out = self.work / group / "cod"
            digest = runner.sha256_file(out / "best.pt")
            with patch.object(runner, "_train_epoch", side_effect=AssertionError("completed group retrained")):
                self.invoke(out, group, "cod", ["--resume"])
            self.assertEqual(digest, runner.sha256_file(out / "best.pt"))
        with self.assertRaises(ValueError):
            self.invoke(self.work / "A/cod", "A", "cod", ["--resume", "--micro-batch-size", "1"])
        with self.assertRaises(ValueError):
            self.invoke(self.work / "wrong-teacher", "C", "source",
                        ["--teacher-checkpoint", str(self.work / "D/teacher/best.pt")])

    def test_route_guards_and_official_tests_before_io(self):
        bad = [("A", "source", []), ("A", "teacher", []), ("B", "teacher", []),
            ("A", "cod", ["--source-checkpoint", "unused.pt"]),
            ("B", "source", ["--teacher-checkpoint", "unused.pt"]),
            ("A", "cod", ["--data-parallel"]),
            ("A", "cod", ["--micro-batch-size", "0"]),
            ("A", "cod", ["--test-after-training"])]
        for group, stage, extra in bad:
            with self.subTest(group=group, stage=stage, extra=extra):
                a = runner.parse_args(self.arguments(self.work / "never", group, stage, extra))
                with self.assertRaises(ValueError):
                    runner.validate_args(a)

    def test_forged_group_teacher_and_missing_transfer_provenance_rejected(self):
        for group, field, replacement in (("A", "source_provenance", {}), ("B", "teacher_provenance", {}),
                                          ("D", "source_provenance", None), ("C", "cod_locator_after_sha256", "0" * 64)):
            path = self.work / group / "cod/comparison.json"
            original = path.read_text(encoding="utf-8")
            mutated = json.loads(original)
            mutated[field] = replacement
            path.write_text(json.dumps(mutated), encoding="utf-8")
            try:
                with self.assertRaises(ValueError):
                    runner.validate_completed_run(path)
            finally:
                path.write_text(original, encoding="utf-8")

    def test_actual_four_group_artifacts_summarize_and_export(self):
        from tools.summarize_triprd_completion_controls import summarize, export
        result = summarize(self.work)
        with redirect_stdout(io.StringIO()):
            export(self.work, result)
        self.assertTrue((self.work / "controls_summary.json").exists())


class CompletionControlMicroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        count = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, count)

    def test_micro_accumulation_matches_unsplit_gradients_including_short_tail(self):
        torch.manual_seed(71)
        template = runner.CompletionStudent(model_variant="tiny", base_channels=4, pretrained=False, train_backbone_blocks=1)
        batches = []
        for count in (5, 3):
            batches.append({"rgb": torch.randn(count, 3, 32, 32), "mask": (torch.rand(count, 1, 32, 32) > .5).float()})
        class Loader(list):
            dataset = type("Dataset", (), {"epoch": 0, "__len__": lambda self: 8})()
        results = []
        for micro in (5, 2):
            network = copy.deepcopy(template)
            optimizer = torch.optim.SGD([p for p in network.parameters() if p.requires_grad], lr=.001)
            a = runner.parse_args(["--out", "unused", "--group", "A", "--batch-size", "5", "--micro-batch-size", str(micro)])
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            with patch.object(runner.infrastructure, "_optimizer_step", wraps=runner.infrastructure._optimizer_step) as step:
                log = runner._train_epoch(network, None, {"cod_train": Loader(batches)}, optimizer, scaler, torch.device("cpu"), a, 1)
            self.assertEqual(step.call_count, 2)
            self.assertEqual(log["optimizer_steps"], 2)
            self.assertEqual(log["observed_images"], 8)
            self.assertEqual(log["micro_steps"], 2 if micro == 5 else 5)
            results.append((network.state_dict(), log))
        for name in results[0][0]:
            torch.testing.assert_close(results[0][0][name], results[1][0][name], rtol=2e-5, atol=2e-7, msg=name)
        self.assertAlmostEqual(results[0][1]["total"], results[1][1]["total"], places=5)

    def test_teacher_rgb_is_three_encoder_calls_and_multimodal_matches_frozen_v6(self):
        options = dict(model_variant="tiny", base_channels=4, pretrained=False, train_backbone_blocks=1)
        torch.manual_seed(2)
        old = FrozenTeacher(**options).eval()
        torch.manual_seed(2)
        teacher = CompletionTeacher(**options).eval()
        for name in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[name], teacher.state_dict()[name]), name)
        rgb, thermal, depth = torch.randn(2, 3, 32, 32), torch.rand(2, 3, 32, 32), torch.rand(2, 1, 32, 32)
        expected = old(rgb, thermal, depth)
        calls = []
        handle = teacher.encoder.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[0].detach().clone()))
        try:
            actual = teacher(rgb, thermal, depth)
            self.assertEqual(len(calls), 3)
            torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
            calls.clear()
            control = teacher(rgb, input_mode="rgb")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(torch.equal(rgb, value) for value in calls))
            changed = teacher(rgb, torch.tensor(float("nan")), torch.zeros(1), input_mode="rgb")
            torch.testing.assert_close(control["logits"], changed["logits"], rtol=0, atol=0)
            control["logits"].square().mean().backward()
            for name in ("T", "D"):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in teacher.modal_adapters[name].parameters()))
        finally:
            handle.remove()


class CompletionControlExportTests(unittest.TestCase):
    def test_each_resource_declaration_is_truthful_and_complete(self):
        for group, phrase in (("A", "no VDT"), ("B", "no teacher"), ("C", "matched RGB teacher"), ("D", "real RGB/thermal/depth")):
            with tempfile.TemporaryDirectory(prefix=".control-export-", dir=ROOT) as folder:
                report = {"protocol": {"group": group}, "test_status": "complete", "checkpoint_sha256": "a" * 64,
                          "prediction_protocol": PREDICTION_PROTOCOL, "test": {}}
                for name, count in DATASET_COUNTS.items():
                    report["test"][name] = {"count": count,
                        "official": {"num_samples": count, **{key: .5 for key in METRIC_FIELDS}},
                        "coverage": {"verified_samples": count, "expected_full_samples": count,
                                     "limited": False, "exact_id_match": True}}
                with patch.object(runner, "validate_completed_run", return_value=report), redirect_stdout(io.StringIO()):
                    exporter.export_summary(folder)
                self.assertIn(phrase, (Path(folder) / "benchmark.md").read_text(encoding="utf-8"))
                report["test"].pop("CAMO")
                with patch.object(runner, "validate_completed_run", return_value=report), self.assertRaises(ValueError):
                    exporter.export_summary(folder)


if __name__ == "__main__":
    unittest.main()
