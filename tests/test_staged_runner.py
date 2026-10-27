"""Real tiny CPU optimization of separate source and COD stages."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_task as legacy
from tools import run_triprd_staged as runner


class StagedRunnerTests(unittest.TestCase):
    @classmethod
    def common_args(cls, out):
        return ["--out", str(out), "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
                "--base-channels", "4", "--image-size", "32", "--batch-size", "2", "--num-workers", "0",
                "--epochs", "4", "--teacher-epochs", "2", "--max-samples", "4", "--warmup-epochs", "1",
                "--train-seed", "11", "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
                "--no-amp", "--train-backbone-blocks", "1"]

    @classmethod
    def args(cls, out):
        return cls.common_args(out) + ["--source-epochs", "3", "--split-manifest", str(cls.manifest)]

    @classmethod
    def group_args(cls, stage, group):
        args = ["--stage", stage, "--group", group]
        if stage == "source" and group in {"S2", "S3"}:
            mode = "rgb" if group == "S2" else "multimodal"
            args += ["--teacher-checkpoint", str(cls.work / "teachers" / mode / "best.pt")]
        elif stage == "cod" and group != "S0":
            args += ["--source-checkpoint", str(cls.work / "source" / group / "best.pt")]
        return args

    @staticmethod
    def report(out):
        return json.loads((out / "comparison.json").read_text(encoding="utf-8"))

    @classmethod
    def invoke(cls, out, extra, *, stage):
        with ExitStack() as stack:
            stack.enter_context(patch.object(legacy.shared, "_vdt_items", **(
                {"return_value": cls.vdt} if stage == "source" else {"side_effect": AssertionError("COD accessed VDT")})))
            stack.enter_context(patch.object(legacy.shared, "_cod_items", **(
                {"return_value": cls.cod} if stage == "cod" else {"side_effect": AssertionError("source accessed COD")})))
            stack.enter_context(patch.object(legacy, "_formal_tests", side_effect=AssertionError("development accessed tests")))
            stack.enter_context(patch.object(runner, "_formal_tests", side_effect=AssertionError("development accessed v5 tests")))
            stream = stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                return runner.main(cls.args(out) + extra)
            except BaseException:
                print(stream.getvalue(), file=sys.stderr)
                raise

    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)
        folder = tempfile.TemporaryDirectory(prefix=".staged-v5-test-", dir=ROOT)
        cls.addClassCleanup(folder.cleanup)
        cls.work = Path(folder.name).resolve()
        vdt = MemoryVDTIndex()
        cls.vdt = [legacy.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, vdt), "vdt") for s in vdt.entries]
        cls.cod = [legacy.shared.Item(f"{idx.dataset}/{s.sample_id}", (s, idx), "cod")
                   for idx in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO")) for s in idx.samples()]
        cls.manifest = cls.work / "shared_manifest.json"
        a = legacy.parse_args(cls.common_args(cls.work / "prepare") + ["--group", "T1"])
        with (patch.object(legacy.shared, "_vdt_items", return_value=cls.vdt),
              patch.object(legacy.shared, "_cod_items", return_value=cls.cod)):
            _, manifest = legacy.prepare_data(a)
        cls.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        for mode in ("rgb", "multimodal"):
            with (patch.object(legacy.shared, "_vdt_items", return_value=cls.vdt),
                  patch.object(legacy.shared, "_cod_items", side_effect=AssertionError("teacher accessed COD")),
                  redirect_stdout(io.StringIO())):
                legacy.main(cls.common_args(cls.work / "teachers" / mode) +
                            ["--stage", "teacher", "--teacher-mode", mode, "--split-manifest", str(cls.manifest)])
        cls.initial = {}
        original_train = runner._train_epoch
        def observe(*args, **kwargs):
            model, teacher, loaders, optimizer, scaler, device, a, epoch = args
            if epoch == 1:
                cls.initial[(a.stage, a.group)] = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer": copy.deepcopy(optimizer.state_dict()), "scaler": copy.deepcopy(scaler.state_dict()),
                    "rng": torch.get_rng_state().clone(),
                    "loader_rng": {k: v.generator.get_state().clone() for k, v in loaders.items()},
                    "teacher_frozen": teacher is None or all(not v.requires_grad for v in teacher.parameters())}
            return original_train(*args, **kwargs)
        cls.reports = {}
        with patch.object(runner, "_train_epoch", side_effect=observe):
            for group in ("S1", "S2", "S3"):
                cls.invoke(cls.work / "source" / group, cls.group_args("source", group), stage="source")
                cls.reports[("source", group)] = cls.report(cls.work / "source" / group)
            for group in ("S0", "S1", "S2", "S3"):
                with patch.object(legacy, "_load_teacher", side_effect=AssertionError("COD loaded teacher")):
                    cls.invoke(cls.work / "screen" / group, cls.group_args("cod", group), stage="cod")
                cls.reports[("cod", group)] = cls.report(cls.work / "screen" / group)

    def test_real_two_teacher_three_source_four_cod_smoke_completes(self):
        for (stage, group), report in self.reports.items():
            with self.subTest(stage=stage, group=group):
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["completed_epochs"], 3 if stage == "source" else 4)
                self.assertFalse(report["formal"])
                self.assertEqual(report["test"], {})
                self.assertEqual(report["test_status"], "skipped")
                self.assertTrue(self.initial[(stage, group)]["teacher_frozen"])
                self.assertEqual(report["sample_counts"]["cod_train" if stage == "source" else "vdt_train"], 0)
                self.assertEqual({r["train"]["amp_skipped_steps"] for r in report["history"]}, {0})

    def test_transfer_preserves_shared_tensors_and_fresh_decoders(self):
        baseline = self.initial[("cod", "S0")]["model"]
        self.assertTrue(any(k.startswith("decoders.cod.") for k in baseline))
        for group in ("S1", "S2", "S3"):
            source = runner.read_checkpoint(self.work / "source" / group / "best.pt")["model"]
            target = self.initial[("cod", group)]["model"]
            with self.subTest(group=group):
                for key in target:
                    expected = baseline[key] if key.startswith("decoders.") else source[key]
                    self.assertTrue(torch.equal(target[key], expected), key)

    def test_cod_optimizer_scaler_scheduler_and_rng_start_fresh(self):
        initial = self.initial[("cod", "S0")]
        for group in ("S0", "S1", "S2", "S3"):
            observed = self.initial[("cod", group)]
            with self.subTest(group=group):
                self.assertEqual(observed["optimizer"]["state"], {})
                self.assertEqual(observed["scaler"], initial["scaler"])
                self.assertTrue(torch.equal(observed["rng"], initial["rng"]))
                self.assertEqual(set(observed["loader_rng"]), {"cod_train", "cod_val"})
                for key in observed["loader_rng"]:
                    self.assertTrue(torch.equal(observed["loader_rng"][key], initial["loader_rng"][key]))
                checkpoint = runner.read_checkpoint(self.work / "screen" / group / "last.pt")
                self.assertEqual(checkpoint["scheduler"]["last_epoch"], 4)
                self.assertEqual({int(s["step"]) for s in checkpoint["optimizer"]["state"].values()}, {12})

    def test_source_selection_uses_only_source_ground_truth_validation(self):
        for group in ("S1", "S2", "S3"):
            report = self.reports[("source", group)]
            losses = [r["validation"]["loss"] for r in report["history"]]
            with self.subTest(group=group):
                self.assertEqual(report["best_epoch"], losses.index(min(losses)) + 1)
                self.assertAlmostEqual(report["best_score"], -min(losses))
                self.assertTrue(all(r["validation"]["objective"] == "supervised_gt_only" for r in report["history"]))

    def test_exact_epoch_resume_for_source_and_cod(self):
        original = runner._train_epoch
        def interrupted(*args, **kwargs):
            if args[-1] == 2:
                raise RuntimeError("injected epoch boundary interruption")
            return original(*args, **kwargs)
        for stage in ("source", "cod"):
            out = self.work / "interrupted" / stage
            extra = self.group_args(stage, "S3")
            with self.subTest(stage=stage):
                with patch.object(runner, "_train_epoch", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "injected epoch boundary"):
                        self.invoke(out, extra, stage=stage)
                self.invoke(out, extra + ["--resume"], stage=stage)
                expected = runner.read_checkpoint(self.work / ("source" if stage == "source" else "screen") / "S3" / "last.pt")
                actual = runner.read_checkpoint(out / "last.pt")
                self.assertEqual(actual["best_epoch"], expected["best_epoch"])
                for name, tensor in expected["model"].items():
                    self.assertTrue(torch.equal(actual["model"][name], tensor), name)

    def test_v4_teacher_rejects_wrong_mode_or_changed_signature(self):
        wrong = self.group_args("source", "S3")
        wrong[wrong.index("--teacher-checkpoint") + 1] = str(self.work / "teachers/rgb/best.pt")
        with self.assertRaisesRegex(ValueError, "Teacher|teacher"):
            self.invoke(self.work / "wrong_teacher", wrong, stage="source")
        with self.assertRaisesRegex(ValueError, "Teacher|teacher"):
            self.invoke(self.work / "wrong_signature", self.group_args("source", "S3") + ["--lr", ".0002"], stage="source")


    def test_source_validation_never_calls_distillation(self):
        a = runner.parse_args(self.args(self.work / "validation_only") + self.group_args("source", "S3"))
        with (patch.object(legacy.shared, "_vdt_items", return_value=self.vdt),
              patch.object(legacy.shared, "_cod_items", side_effect=AssertionError("source validation accessed COD"))):
            splits, _ = runner.prepare_data(a)
        model = runner.TaskStudent(**runner.model_options(a, pretrained=False))
        state = runner.read_checkpoint(self.work / "source/S3/best.pt")
        model.load_state_dict(state["model"])
        loader = runner._loaders(a, splits, torch.device("cpu"))["vdt_val"]
        with patch.object(runner, "transfer_loss", side_effect=AssertionError("KD used for source selection")):
            result = runner._validate_source(model, loader, torch.device("cpu"), a)
        self.assertEqual(result["objective"], "supervised_gt_only")
        self.assertAlmostEqual(result["loss"], -self.reports[("source", "S3")]["best_score"])

    def test_source_provenance_rejects_semantically_wrong_or_incomplete_artifacts(self):
        cases = ("group", "seed", "members", "source_hash", "unfinished", "checkpoint_hash", "last_budget")
        for case in cases:
            with self.subTest(case=case):
                directory = self.work / "tampered" / case
                shutil.copytree(self.work / "source/S3", directory)
                best = runner.read_checkpoint(directory / "best.pt")
                last = runner.read_checkpoint(directory / "last.pt")
                report = self.report(directory)
                contract = copy.deepcopy(best["contract"])
                if case == "group":
                    contract["group"] = "S2"
                    report["group"] = "S2"
                    report["config"]["group"] = "S2"
                elif case == "seed":
                    contract["train_seed"] += 1
                    report["config"]["train_seed"] += 1
                elif case == "members":
                    contract["vdt_train"][0] = "VDT/not_a_member"
                elif case == "source_hash":
                    contract["source_sha256"][next(iter(contract["source_sha256"]))] = "0" * 64
                elif case == "unfinished":
                    report["status"] = "running"
                elif case == "last_budget":
                    last["epoch"] -= 1
                for state, filename in ((best, "best.pt"), (last, "last.pt")):
                    state["contract"] = copy.deepcopy(contract)
                    state["contract_sha256"] = runner._digest(contract)
                    if case == "group":
                        state["group"] = state["config"]["group"] = "S2"
                    elif case == "seed":
                        state["config"]["train_seed"] += 1
                    torch.save(state, directory / filename)
                report["protocol"] = contract
                report["contract_sha256"] = runner._digest(contract)
                report["checkpoint_sha256"] = "0" * 64 if case == "checkpoint_hash" else runner.sha256_file(directory / "best.pt")
                (directory / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
                extra = self.group_args("cod", "S3") + ["--source-checkpoint", str(directory / "best.pt")]
                with self.assertRaises(ValueError):
                    self.invoke(self.work / "rejected" / case, extra, stage="cod")

    def test_official_test_requires_real_full_training_before_data_access(self):
        for extra in (["--test-after-training"], ["--test-after-training", "--cod-train-mode", "full", "--max-samples", "0"]):
            with self.subTest(extra=extra), patch.object(runner, "prepare_data", side_effect=AssertionError("data before official guard")):
                with self.assertRaisesRegex(ValueError, "Official tests"):
                    runner.main(self.args(self.work / "invalid_formal") + ["--stage", "cod", "--group", "S0"] + extra)
        a = runner.parse_args(self.args(self.work / "invalid_count") +
                              ["--stage", "cod", "--group", "S0", "--cod-train-mode", "full", "--max-samples", "0"])
        with patch.object(legacy.shared, "_cod_items", return_value=self.cod):
            with self.assertRaisesRegex(ValueError, "3040"):
                runner.prepare_data(a)

    def test_actual_pipeline_reports_are_accepted_by_summary_cli(self):
        from tools import summarize_triprd_staged as summary
        with redirect_stdout(io.StringIO()):
            self.assertEqual(summary.main(["--output-root", str(self.work)]), 0)
        result = json.loads((self.work / "analysis/staged_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["report_consistency_verified"])
        self.assertEqual(result["evidence_scope"], "smoke_only")
        self.assertEqual(result["verdict"], "not_supported")
        self.assertEqual(len(result["checkpoint_files_checked"]), 7)
        self.assertEqual(set(result["teachers"]), {"rgb", "multimodal"})
        self.assertEqual({x["cost"]["epochs"] for x in result["teachers"].values()}, {2})
        self.assertTrue((self.work / "analysis/staged_summary.md").is_file())


    def test_full_training_selects_final_epoch_without_cod_validation(self):
        original = runner.prepare_data
        def fixture_full(a):
            a.cod_train_mode, a.max_samples = "split", 4
            splits, manifest = original(a)
            a.cod_train_mode, a.max_samples = "full", 0
            splits["cod_train"], splits["cod_val"] = self.cod, []
            manifest.update(cod_train=[x.key for x in self.cod], cod_val=[], cod_train_mode="full")
            return splits, manifest
        out = self.work / "full_training_fixture"
        with (patch.object(runner, "prepare_data", side_effect=fixture_full),
              patch.object(legacy, "_validate_cod", side_effect=AssertionError("full training accessed COD validation"))):
            self.invoke(out, ["--stage", "cod", "--group", "S0", "--cod-train-mode", "full", "--max-samples", "0"], stage="cod")
        report = self.report(out)
        self.assertEqual(report["selection_metric"], "last_epoch")
        self.assertEqual(report["best_epoch"], 4)
        self.assertFalse(report["formal"])
        self.assertTrue(all("validation" not in row for row in report["history"]))

if __name__ == "__main__":
    unittest.main()


