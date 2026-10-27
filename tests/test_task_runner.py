"""Real v4 CPU optimization with synthetic archive I/O only."""
from contextlib import ExitStack, redirect_stdout
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
from tools import run_triprd_task as runner


class TaskRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)
        folder = tempfile.TemporaryDirectory(prefix=".task-v4-test-", dir=ROOT)
        cls.addClassCleanup(folder.cleanup)
        cls.work = Path(folder.name)
        vdt = MemoryVDTIndex()
        cls.vdt = [runner.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, vdt), "vdt") for s in vdt.entries]
        cls.cod = [runner.shared.Item(f"{idx.dataset}/{s.sample_id}", (s, idx), "cod")
                   for idx in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO")) for s in idx.samples()]
        for mode in ("rgb", "multimodal"):
            cls.invoke(cls.work / mode, ["--stage", "teacher", "--teacher-mode", mode])
        cls.reports = {}
        for group in runner.GROUPS:
            cls.invoke(cls.work / group, cls.group_args(group))
            cls.reports[group] = json.loads((cls.work / group / "comparison.json").read_text(encoding="utf-8"))

    @classmethod
    def args(cls, out):
        return ["--out", str(out), "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
                "--base-channels", "4", "--image-size", "32", "--batch-size", "2", "--num-workers", "0",
                "--epochs", "2", "--teacher-epochs", "2", "--max-samples", "4", "--warmup-epochs", "1",
                "--train-seed", "11", "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
                "--no-amp", "--train-backbone-blocks", "1"]

    @classmethod
    def group_args(cls, group):
        result = ["--group", group]
        if group in {"T2", "T3", "T4"}:
            mode = "rgb" if group == "T2" else "multimodal"
            result += ["--teacher-checkpoint", str(cls.work / mode / "best.pt")]
        return result

    @classmethod
    def invoke(cls, out, extra):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.shared, "_vdt_items", return_value=cls.vdt))
            stack.enter_context(patch.object(runner.shared, "_cod_items", return_value=cls.cod))
            stack.enter_context(patch.object(runner, "_formal_tests", side_effect=AssertionError("Official test access during development")))
            stream = stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                return runner.main(cls.args(out) + extra)
            except BaseException:
                print(stream.getvalue(), file=sys.stderr)
                raise

    def test_all_groups_same_initialization_budget_and_selected_checkpoint(self):
        initial = self.reports["T0"]["initial_state_sha256"]
        source_images = []
        for group, report in self.reports.items():
            with self.subTest(group=group):
                self.assertEqual(report["status"], "complete")
                self.assertFalse(report["formal"])
                self.assertEqual(report["test"], {})
                self.assertEqual(report["initial_state_sha256"], initial)
                self.assertEqual(report["prediction_protocol"], runner.PREDICTION_PROTOCOL)
                self.assertEqual(report["best_score"], max(r["validation"]["macro"]["S_alpha"] for r in report["history"]))
                self.assertEqual({r["train"]["optimizer_steps"] for r in report["history"]}, {3})
                self.assertEqual({r["train"]["amp_skipped_steps"] for r in report["history"]}, {0})
                if group != "T0":
                    source_images.append([r["train"]["source_images"] for r in report["history"]])
                self.assertAlmostEqual(report["history"][-1]["train"]["source_weight"], .05)
                cp = runner.read_checkpoint(self.work / group / "best.pt")
                self.assertEqual(cp["epoch"], report["best_epoch"])
        self.assertTrue(all(x == source_images[0] for x in source_images))

    def test_teacher_only_does_not_index_cod_and_t0_does_not_index_vdt(self):
        a = runner.parse_args(self.args(self.work / "noop") + ["--stage", "teacher"])
        with patch.object(runner.shared, "_vdt_items", return_value=self.vdt), patch.object(runner.shared, "_cod_items", side_effect=AssertionError("COD indexed")):
            splits, _ = runner.prepare_data(a)
        self.assertEqual(splits["cod_train"], [])
        a.stage, a.group = "student", "T0"
        with patch.object(runner.shared, "_vdt_items", side_effect=AssertionError("VDT indexed")), patch.object(runner.shared, "_cod_items", return_value=self.cod):
            splits, _ = runner.prepare_data(a)
        self.assertEqual(splits["vdt_train"], [])

    def test_resume_is_exact_for_multimodal_student_and_teacher_stays_frozen(self):
        out = self.work / "interrupted"
        original = runner._train_epoch
        def interrupted(*args):
            if args[-1] == 2:
                raise RuntimeError("injected interruption")
            return original(*args)
        with patch.object(runner, "_train_epoch", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.invoke(out, self.group_args("T4"))
        self.invoke(out, self.group_args("T4") + ["--resume"])
        expected = runner.read_checkpoint(self.work / "T4" / "last.pt")
        actual = runner.read_checkpoint(out / "last.pt")
        self.assertEqual(expected["best_epoch"], actual["best_epoch"])
        for name, value in expected["model"].items():
            self.assertTrue(torch.equal(value, actual["model"][name]), name)
        self.assertFalse((out / "task_run.lock").exists())

    def test_reject_wrong_teacher_mode_and_configuration(self):
        with self.assertRaisesRegex(ValueError, "Teacher mode"):
            self.invoke(self.work / "wrong", ["--group", "T4", "--teacher-checkpoint", str(self.work / "rgb" / "best.pt")])
        with self.assertRaisesRegex(ValueError, "Teacher mode"):
            self.invoke(self.work / "wrong_lr", self.group_args("T4") + ["--lr", ".0002"])
        with self.assertRaisesRegex(ValueError, "identical contract"):
            self.invoke(self.work / "T0", ["--resume", "--lr", ".0002"])
        with patch.object(runner, "_train_epoch", side_effect=AssertionError("retrained completed run")):
            self.invoke(self.work / "T0", ["--resume"])

    def test_resume_repairs_best_transaction(self):
        out = self.work / "transaction"
        save = runner._save_pt
        def interrupted(path, state):
            if path.name == "best.pt":
                raise RuntimeError("best transaction")
            return save(path, state)
        with patch.object(runner, "_save_pt", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "transaction"):
                self.invoke(out, [])
        self.assertFalse((out / "best.pt").exists())
        self.invoke(out, ["--resume"])
        reference = runner.read_checkpoint(self.work / "T0" / "last.pt")
        repaired = runner.read_checkpoint(out / "last.pt")
        for name, value in reference["model"].items():
            self.assertTrue(torch.equal(value, repaired["model"][name]), name)

    def test_full_mode_fixed_last_and_no_validation(self):
        original = runner.prepare_data
        def full(a):
            a.cod_train_mode, a.max_samples = "split", 4
            splits, manifest = original(a)
            a.cod_train_mode, a.max_samples = "full", 0
            splits["cod_train"], splits["cod_val"] = self.cod, []
            manifest.update(cod_train=[x.key for x in self.cod], cod_val=[], cod_train_mode="full")
            return splits, manifest
        out = self.work / "full"
        with patch.object(runner, "prepare_data", side_effect=full), patch.object(runner, "_validate_cod", side_effect=AssertionError("validation in full mode")):
            self.invoke(out, ["--cod-train-mode", "full", "--max-samples", "0"])
        report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["best_epoch"], 2)
        self.assertEqual(report["selection_metric"], "last_epoch")
        self.assertFalse(report["formal"])

    def test_paired_stateless_crop_and_epoch_reproducibility(self):
        dataset = runner.TaskDataset(self.vdt, 32, normalize=False, augment=True, seed=17)
        dataset.epoch = 3
        first, second = dataset[0], dataset[0]
        for key in ("rgb", "mask", "thermal", "depth"):
            self.assertTrue(torch.equal(first[key], second[key]))
            self.assertEqual(first[key].shape[-2:], (32, 32))
        self.assertEqual(set(first["mask"].unique().tolist()), {0., 1.})

    def test_official_test_guard_before_data_access(self):
        with patch.object(runner, "prepare_data", side_effect=AssertionError("data before guard")):
            with self.assertRaisesRegex(ValueError, "Official tests"):
                runner.main(self.args(self.work / "bad_test") + ["--test-after-training"])

    def test_full_manifest_rejects_duplicate_or_overlapping_members(self):
        manifest = self.work / "bad_full_manifest.json"
        keys = [item.key for item in self.cod]
        manifest.write_text(json.dumps({"split_seed": 7, "cod_train": keys, "cod_val": [keys[0]]}), encoding="utf-8")
        a = runner.parse_args(self.args(self.work / "bad_full") +
                             ["--cod-train-mode", "full", "--split-manifest", str(manifest)])
        with patch.object(runner.shared, "_cod_items", return_value=self.cod):
            with self.assertRaisesRegex(ValueError, "disjoint development pool"):
                runner.prepare_data(a)

    def test_dino_source_digest_detects_operator_change_without_weights(self):
        repo = self.work / "mock_dino"
        (repo / "dinov3").mkdir(parents=True)
        (repo / "hubconf.py").write_text("# local entrypoint\n", encoding="utf-8")
        operator = repo / "dinov3" / "operator.py"
        operator.write_text("scale = 1\n", encoding="utf-8")
        before = runner.dino_source_digest(repo)
        operator.write_text("scale = 2\n", encoding="utf-8")
        self.assertNotEqual(before, runner.dino_source_digest(repo))

    def test_atomic_save_retries_transient_file_reader_lock(self):
        path = self.work / "atomic.json"
        replace = Path.replace
        attempts = []
        def transient(source, destination):
            attempts.append(source)
            if len(attempts) < 3:
                raise PermissionError("simulated Windows sharing conflict")
            return replace(source, destination)
        with patch.object(Path, "replace", transient), patch.object(runner.time, "sleep"):
            runner._save_json(path, {"saved": True})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"saved": True})
        with patch.object(Path, "replace", side_effect=PermissionError("persistent conflict")), patch.object(runner.time, "sleep"):
            with self.assertRaises(PermissionError):
                runner._save_json(path, {"saved": False})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"saved": True})


if __name__ == "__main__":
    unittest.main()
