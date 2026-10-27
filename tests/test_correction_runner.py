"""Real optimization, restart and evaluation of the independent correction runner."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_correction as runner


class CorrectionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)
        cls.temp = tempfile.TemporaryDirectory(prefix=".correction-runner-", dir=runner.ROOT)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        vdt = MemoryVDTIndex()
        cls.vdt = [runner.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, vdt), "vdt") for s in vdt.entries]
        cls.cod = [runner.shared.Item(f"{index.dataset}/{s.sample_id}", (s, index), "cod")
                   for index in (MemoryCODIndex("COD10K"), MemoryCODIndex("CAMO")) for s in index.samples()]
        cls.teacher = cls.work / "teacher"
        cls.invoke(cls.teacher, ["--stage", "teacher"])
        cls.reports = {}
        for group in "ABCDEF":
            out = cls.work / group
            cls.invoke(out, cls.group_args(group))
            cls.reports[group] = json.loads((out / "comparison.json").read_text(encoding="utf-8"))

    @classmethod
    def args(cls, out):
        return ["--out", str(out), "--device", "cpu", "--model-variant", "tiny", "--no-pretrained",
                "--base-channels", "4", "--image-size", "32", "--batch-size", "2", "--num-workers", "0",
                "--epochs", "2", "--teacher-epochs", "2", "--max-samples", "4", "--warmup-epochs", "1",
                "--train-seed", "11", "--split-seed", "7", "--val-fraction", ".25", "--val-metrics-every", "1",
                "--skip-test", "--no-amp"]

    @classmethod
    def group_args(cls, group):
        result = ["--stage", "student", "--group", group]
        if group in "CDEF":
            result += ["--teacher-checkpoint", str(cls.teacher / "best.pt")]
        return result

    @classmethod
    def invoke(cls, out, extra, *, allow_test=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.shared, "_vdt_items", return_value=cls.vdt))
            stack.enter_context(patch.object(runner.shared, "_cod_items", return_value=cls.cod))
            if not allow_test:
                stack.enter_context(patch.object(runner, "_formal_tests", side_effect=AssertionError("Test access during development")))
            log = stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                return runner.main(cls.args(out) + extra)
            except Exception:
                print(log.getvalue())
                raise

    def test_matched_six_groups_actual_optimization_and_selection(self):
        initial = self.reports["A"]["initial_state_sha256"]
        counts = []
        for group, report in self.reports.items():
            with self.subTest(group=group):
                self.assertEqual(report["status"], "complete")
                self.assertFalse(report["formal"])
                self.assertEqual(report["test"], {})
                self.assertEqual(report["initial_state_sha256"], initial)
                self.assertEqual(report["completed_epochs"], 2)
                self.assertEqual(report["best_score"], max(x["validation"]["macro"]["S_alpha"] for x in report["history"]))
                self.assertEqual({r["train"]["steps"] for r in report["history"]}, {3})
                self.assertEqual(report["sample_counts"]["cod_val"], 2)
                self.assertFalse(report["intervention_validation"]["selection_used"])
                if group != "A":
                    counts.append([r["train"]["source_images"] for r in report["history"]])
                if group in "CDEF":
                    probe = report["source_recoverability"]
                    self.assertTrue(all(np.isfinite(x) for x in probe.values()))
                    self.assertIn("constant_mse", probe)
                if group in "EF":
                    for row in report["history"]:
                        self.assertTrue(0 <= row["train"]["alignment"] <= 1)
        self.assertTrue(all(x == counts[0] for x in counts))

    def test_teacher_four_coalitions_and_train_constants(self):
        report = json.loads((self.teacher / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        for coalition in ("R", "RT", "RD", "RTD"):
            self.assertIn("relation_" + coalition, report["teacher_probe"])
        checkpoint = runner._read_checkpoint(self.teacher / "best.pt")
        self.assertEqual(len(checkpoint["constant_correction"]), 2)
        self.assertTrue(all(x.shape == (1, 1, 9, 1, 1) for x in checkpoint["constant_correction"]))
        self.assertEqual(len(checkpoint["positive_constant_correction"]), 2)
        self.assertTrue(all(x.shape == (1, 1, 9, 1, 1) for x in checkpoint["positive_constant_correction"]))
        self.assertIn("VDT_TRAIN_only", checkpoint["constant_correction_scope"])

    def test_exact_epoch_resume_after_interruption(self):
        out = self.work / "interrupted"
        original = runner._train_epoch
        def interrupted(*args):
            if args[-1] == 2:
                raise RuntimeError("injected interruption")
            return original(*args)
        with patch.object(runner, "_train_epoch", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.invoke(out, self.group_args("E"))
        self.assertEqual(runner._read_checkpoint(out / "last.pt")["epoch"], 1)
        self.invoke(out, self.group_args("E") + ["--resume"])
        expected = runner._read_checkpoint(self.work / "E" / "last.pt")
        actual = runner._read_checkpoint(out / "last.pt")
        self.assertEqual(actual["best_epoch"], expected["best_epoch"])
        for name, value in expected["model"].items():
            self.assertTrue(torch.equal(actual["model"][name], value), name)

    def test_resume_rejects_protocol_changes_and_never_overwrites(self):
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            self.invoke(self.work / "A", self.group_args("A"))
        with self.assertRaisesRegex(ValueError, "identical contract"):
            self.invoke(self.work / "A", self.group_args("A") + ["--resume", "--lr", ".002"])
        self.invoke(self.work / "A", self.group_args("A") + ["--resume"])

    def test_resume_repairs_best_before_next_epoch_overwrites_last(self):
        out = self.work / "interrupted_best"
        save = runner._save_pt
        def interrupted(path, state):
            if path.name == "best.pt":
                raise RuntimeError("interrupted best transaction")
            return save(path, state)
        with patch.object(runner, "_save_pt", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "best transaction"):
                self.invoke(out, self.group_args("A"))
        self.assertFalse((out / "best.pt").exists())
        train = runner._train_epoch
        def check_repair(*args):
            self.assertEqual(runner._read_checkpoint(out / "best.pt")["epoch"], 1)
            return train(*args)
        with patch.object(runner, "_train_epoch", side_effect=check_repair):
            self.invoke(out, self.group_args("A") + ["--resume"])

    def test_teacher_reuse_rejects_changed_source_partition(self):
        manifest = json.loads((self.teacher / "split_manifest.json").read_text(encoding="utf-8"))
        manifest["vdt_train"][0], manifest["vdt_val"][0] = manifest["vdt_val"][0], manifest["vdt_train"][0]
        path = self.work / "changed.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.invoke(self.work / "changed_source", self.group_args("E") + ["--split-manifest", str(path)])

    def test_full_mode_has_no_cod_validation_and_selects_last(self):
        # Full-pool count enforcement is separately tested; replicate only the
        # indexing boundary here so the actual train/selection path stays real.
        original = runner.prepare_data
        def prepared(args):
            args.cod_train_mode, args.max_samples = "split", 4
            splits, manifest = original(args)
            args.cod_train_mode, args.max_samples = "full", 0
            splits["cod_train"], splits["cod_val"] = self.cod, []
            manifest.update(cod_train=[x.key for x in self.cod], cod_val=[], cod_train_mode="full")
            return splits, manifest
        with patch.object(runner, "prepare_data", side_effect=prepared), patch.object(runner, "_validate_cod", side_effect=AssertionError("COD validation in full mode")):
            out = self.work / "full"
            self.invoke(out, self.group_args("A") + ["--cod-train-mode", "full", "--max-samples", "0"])
        report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["best_epoch"], 2)
        self.assertTrue(all("validation" not in r for r in report["history"]))

    def test_checkpoint_evaluation_never_loads_training_or_teacher(self):
        out = self.work / "evaluation"
        with patch.object(runner, "prepare_data", side_effect=AssertionError("Training data accessed")), patch.object(runner, "_formal_tests", return_value={"checked": True}) as evaluate:
            runner.main(self.args(out) + ["--eval-only", "--no-skip-test", "--cod-checkpoint", str(self.work / "E" / "best.pt")])
        self.assertEqual(evaluate.call_count, 1)

    def test_prepare_resume_and_member_equality(self):
        out = self.work / "prepare"
        self.invoke(out, ["--stage", "teacher", "--prepare-only"])
        self.invoke(out, ["--stage", "teacher", "--prepare-only", "--resume"])
        with self.assertRaises(FileExistsError):
            self.invoke(out, ["--stage", "teacher", "--prepare-only", "--resume", "--split-seed", "8"])


class CorrectionDataTests(unittest.TestCase):
    def test_zip_dataset_pickle_and_worker_local_handles(self):
        from datasets.vdt2048 import VDT2048Index
        with tempfile.TemporaryDirectory(prefix=".correction-zip-", dir=runner.ROOT) as work:
            path = Path(work) / "vdt.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for modality in ("rgb", "thermal", "depth", "mask"):
                    image = Image.new("RGB" if modality in ("rgb", "thermal") else "L", (32, 32), 100)
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    archive.writestr(f"Train/{modality}/1.png", stream.getvalue())
            index = VDT2048Index(path, "train")
            sample = next(iter(index.samples()))
            data = runner.CorrectionDataset([runner.shared.Item("VDT/1", (sample, index), "vdt")], 32, normalize=True)
            try:
                copy = pickle.loads(pickle.dumps(data))
                self.assertTrue(torch.equal(data[0]["rgb"], copy[0]["rgb"]))
                self.assertTrue(torch.isfinite(copy[0]["depth"]).all())
                copy.items[0].source[1].close()
            finally:
                index.close()

    def test_formal_defaults_and_no_random_backbone_fallback(self):
        a = runner.parse_args(["--out", "unused"])
        self.assertEqual((a.epochs, a.teacher_epochs, a.max_samples), (100, 100, 0))
        self.assertEqual(a.model_variant, "dinov3_vitb16")
        with self.assertRaisesRegex(ValueError, "generic pretrained"):
            runner.validate_args(runner.parse_args(["--out", "unused", "--no-pretrained"]))


if __name__ == "__main__":
    unittest.main()
