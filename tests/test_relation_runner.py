"""Exercise the real relation pipeline with tiny networks and in-memory images."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_relation as runner
from tools import record_experiment


ROOT = Path(__file__).resolve().parents[1]


class RelationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)
        cls.temp = tempfile.TemporaryDirectory(prefix=".relation-integration-", dir=ROOT)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.vdt = MemoryVDTIndex()
        cls.vdt_items = [runner.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, cls.vdt), "vdt")
                         for s in cls.vdt.entries]
        cls.cod = [MemoryCODIndex(x) for x in ("COD10K", "CAMO")]
        cls.cod_items = [runner.shared.Item(f"{index.dataset}/{s.sample_id}", (s, index), "cod")
                         for index in cls.cod for s in index.samples()]
        cls.fresh = cls.work / "fresh"
        cls.invoke(cls.fresh, ["--groups", "A,B,C,D"])
        cls.result = json.loads((cls.fresh / "comparison.json").read_text(encoding="utf-8"))

    @classmethod
    def arguments(cls, output):
        return ["--out", str(output), "--model-variant", "tiny", "--no-pretrained",
                "--base-channels", "2", "--image-size", "32", "--relation-dim", "4",
                "--device", "cpu", "--batch-size", "2", "--epochs", "2", "--vdt-epochs", "2",
                "--train-seed", "11", "--split-seed", "7", "--val-fraction", "0.25",
                "--val-metrics-every", "2", "--cod-warmup-epochs", "1", "--skip-test"]

    @classmethod
    def invoke(cls, output, extra, *, no_vdt=False, full=False):
        stream = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.shared, "_vdt_items",
                side_effect=AssertionError("Unexpected VDT read")) if no_vdt else
                patch.object(runner.shared, "_vdt_items", return_value=cls.vdt_items))
            stack.enter_context(patch.object(runner.shared, "_cod_items", return_value=cls.cod_items))
            stack.enter_context(patch.object(runner, "_evaluate_complete",
                                             side_effect=AssertionError("Unexpected test evaluation")))
            if full:
                stack.enter_context(patch.object(runner, "_full_cod_pool", side_effect=lambda x: x))
                stack.enter_context(patch.object(runner.shared, "_validate_original_size",
                                                 side_effect=AssertionError("Full mode must not validate COD")))
            stack.enter_context(redirect_stdout(stream))
            try:
                runner.main(cls.arguments(output) + extra)
            except Exception:
                print(stream.getvalue())
                raise

    def test_real_four_groups_match_initialization_and_isolate_transfer(self):
        result = self.result
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["test"], {})
        self.assertEqual(result["test_status"], "skipped")
        self.assertEqual(result["teachers"]["C"]["parameters"], result["teachers"]["D"]["parameters"])
        self.assertEqual(result["teachers"]["C"]["teacher_kind"], "rgb")
        self.assertEqual(result["teachers"]["D"]["teacher_kind"], "multimodal")
        initial = result["groups"]["A"]["initial_state_sha256"]
        for group, summary in result["groups"].items():
            with self.subTest(group=group):
                self.assertEqual(summary["initial_state_sha256"], initial)
                self.assertEqual(summary["parameters"], result["groups"]["A"]["parameters"])
                for key in ("encoder", "decoder", "heads"):
                    self.assertEqual(summary["cod_initial_state_sha256"][key], initial[key])
                fit = summary["cod_finetune"]
                self.assertEqual(fit["selection_metric"], "S_alpha")
                self.assertEqual(fit["best_epoch"], 2)
                self.assertEqual(fit["history"][-1]["val_official"]["num_samples"], 2)
                if group != "A":
                    self.assertTrue(summary["vdt_pretrain"]["frozen_encoder_verified"])
                    self.assertNotEqual(summary["cod_initial_state_sha256"]["relation_branch"], initial["relation_branch"])
                    self.assertTrue((self.fresh / group / "vdt" / "relation.pt").is_file())
                    losses = summary["vdt_pretrain"]["history"][-1]["train"]
                    if group in "CD":
                        self.assertGreater(losses["relation_kd"], 0)
                    else:
                        self.assertEqual(losses["weighted_relation_kd"], 0)

    def test_branch_reuse_skips_source_and_reproduces_cod_training(self):
        output = self.work / "reuse"
        self.invoke(output, ["--groups", "D", "--relation-checkpoint",
            f"D={self.fresh / 'D/vdt/relation.pt'}", "--split-manifest",
            str(self.fresh / "split_manifest.json")], no_vdt=True)
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        actual = result["groups"]["D"]
        expected = self.result["groups"]["D"]
        self.assertEqual(actual["cod_initial_state_sha256"], expected["cod_initial_state_sha256"])
        self.assertEqual(actual["cod_finetune"]["best_selection_score"], expected["cod_finetune"]["best_selection_score"])
        self.assertEqual(result["teachers"], {})

    def test_full_training_has_no_cod_validation_and_uses_final_epoch(self):
        output = self.work / "full"
        self.invoke(output, ["--groups", "A", "--cod-train-mode", "full"], no_vdt=True, full=True)
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        fit = result["groups"]["A"]["cod_finetune"]
        self.assertEqual(fit["selection_metric"], "fixed_last_epoch")
        self.assertEqual(fit["selected_epoch"], 2)
        self.assertFalse(fit["validation_used"])
        self.assertTrue(result["groups"]["A"]["checkpoint"].endswith("last.pt"))
        self.assertTrue(all("val" not in x and "val_official" not in x for x in fit["history"]))
        self.assertEqual(result["protocol"]["sample_counts"]["cod_val"], 0)
        self.assertIn("not applicable", "\n".join(record_experiment.validation_lines(result)))

    def test_full_pool_rejects_incomplete_data(self):
        with self.assertRaisesRegex(ValueError, "3040 COD10K"):
            runner._full_cod_pool(self.cod_items)

    def test_frozen_checkpoint_evaluation_skips_all_training_io(self):
        output = self.work / "eval"
        index = MemoryCODIndex("COD10K", split="test", count=2)
        stream = io.StringIO()
        with patch.object(runner.shared, "_vdt_items", side_effect=AssertionError("VDT read")), \
             patch.object(runner.shared, "_cod_items", side_effect=AssertionError("COD training read")), \
             patch.object(runner.shared, "_index_for_dataset", side_effect=lambda *a: index), \
             patch.object(runner.shared, "COD10KIndex", side_effect=lambda *a: index), \
             redirect_stdout(stream):
            runner.main(self.arguments(output) + ["--groups", "D", "--eval-only", "--no-skip-test",
                "--cod-checkpoint", f"D={self.fresh / 'D/cod/best.pt'}",
                "--test-datasets", "COD10K", "--max-test-samples", "2"])
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(result["test_status"], "complete")
        self.assertEqual(result["test"]["D"]["COD10K"]["official"]["num_samples"], 2)
        self.assertTrue(result["test"]["D"]["COD10K"]["coverage"]["limited"])

    def test_full_test_coverage_fails_before_export(self):
        args = runner.parse_args(self.arguments(self.work / "unused"))
        index = MemoryCODIndex("COD10K", split="test", count=2)
        with patch.object(runner.shared, "_index_for_dataset", return_value=index), \
             patch.object(runner.shared, "_predict_test") as predict:
            with self.assertRaisesRegex(ValueError, "Incomplete COD10K test coverage"):
                runner._evaluate_complete(None, args, "COD10K", self.work / "unused", torch.device("cpu"))
            predict.assert_not_called()

    def test_failure_is_recorded_and_existing_run_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            self.invoke(self.fresh, ["--groups", "A"])
        output = self.work / "failure"
        with patch.object(runner.shared, "_fit_stage", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.invoke(output, ["--groups", "A"])
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("synthetic failure", result["error"])


if __name__ == "__main__":
    unittest.main()
