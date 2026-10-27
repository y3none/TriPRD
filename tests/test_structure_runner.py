"""Real two-epoch CPU v2 experiments; only dataset I/O is replaced."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_structure as runner
from tools import run_triprd_structure_queue as queue
from tools.triprd_structure_training import load_structure_checkpoint, observation_tensor


ROOT = Path(__file__).resolve().parents[1]


class StructureRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)
        cls.temp = tempfile.TemporaryDirectory(prefix=".structure-integration-", dir=ROOT)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.vdt = MemoryVDTIndex()
        cls.vdt_items = [runner.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, cls.vdt), "vdt")
                         for s in cls.vdt.entries]
        cls.cod = [MemoryCODIndex(name) for name in ("COD10K", "CAMO")]
        cls.cod_items = [runner.shared.Item(f"{index.dataset}/{s.sample_id}", (s, index), "cod")
                         for index in cls.cod for s in index.samples()]
        cls.fresh = cls.work / "fresh"
        cls.invoke(cls.fresh, ["--groups", "A,B,C,D,E"])
        cls.result = json.loads((cls.fresh / "comparison.json").read_text(encoding="utf-8"))

    @classmethod
    def arguments(cls, output):
        return ["--out", str(output), "--model-variant", "tiny", "--no-pretrained",
                "--base-channels", "2", "--image-size", "32", "--relation-dim", "4",
                "--device", "cpu", "--batch-size", "2", "--epochs", "2", "--vdt-epochs", "2",
                "--train-seed", "11", "--split-seed", "7", "--val-fraction", "0.25",
                "--val-metrics-every", "2", "--cod-warmup-epochs", "1", "--skip-test"]

    @classmethod
    def invoke(cls, output, extra, *, no_vdt=False, full=False, source_items=None):
        stream = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.shared, "_vdt_items",
                side_effect=AssertionError("Unexpected VDT read")) if no_vdt else
                patch.object(runner.shared, "_vdt_items", return_value=cls.vdt_items if source_items is None else source_items))
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

    def test_real_five_groups_transfer_complete_function_and_freeze_cod(self):
        result = self.result
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["test_status"], "skipped")
        self.assertEqual(result["test"], {})
        sensor_audit = json.loads((self.fresh / "sensor_preprocessing.json").read_text(encoding="utf-8"))
        self.assertEqual(result["sensor_preprocessing_audit"]["sample_counts"], {"vdt_train": 3, "vdt_val": 1})
        self.assertEqual(next(iter(sensor_audit["samples"]["vdt_train"].values()))["depth"]["normalization_divisor"], 255)
        self.assertEqual(result["teachers"]["D"]["parameters"], result["teachers"]["E"]["parameters"])
        self.assertEqual(result["teachers"]["D"]["teacher_kind"], "rgb")
        self.assertEqual(result["teachers"]["E"]["teacher_kind"], "multimodal")
        initial = result["groups"]["A"]["initial_state_sha256"]
        for group, summary in result["groups"].items():
            with self.subTest(group=group):
                self.assertEqual(summary["initial_state_sha256"], initial)
                self.assertEqual(summary["parameters"], result["groups"]["A"]["parameters"])
                self.assertEqual(summary["cod_trainable_parameters"], result["groups"]["A"]["cod_trainable_parameters"])
                for key in set(initial) - {"structure_branch"}:
                    self.assertEqual(summary["cod_initial_state_sha256"][key], initial[key])
                self.assertTrue(summary["cod_frozen_structure_verified"])
                self.assertEqual(summary["cod_final_structure_sha256"],
                                 summary["cod_initial_state_sha256"]["structure_branch"])
                fit = summary["cod_finetune"]
                self.assertEqual(fit["selection_metric"], "S_alpha")
                self.assertEqual(fit["best_epoch"], 2)
                self.assertEqual(fit["history"][-1]["val_official"]["num_samples"], 2)
                self.assertEqual(fit["history"][-1]["train"]["structure_supervision"], 0)
                if group != "A":
                    source = summary["vdt_pretrain"]
                    self.assertTrue(source["frozen_encoder_verified"])
                    self.assertTrue(source["frozen_reference_encoder_verified"])
                    self.assertEqual(source["selection_metric"], "structure_supervision")
                    self.assertEqual(source["best_selection_score"], min(
                        row["val"]["structure_supervision"] for row in source["history"]))
                    self.assertNotEqual(summary["cod_initial_state_sha256"]["structure_branch"], initial["structure_branch"])
                    artifact = torch.load(self.fresh / group / "vdt/structure.pt", weights_only=True)
                    self.assertTrue(any(k.startswith("reference_encoder.") for k in artifact["structure_branch"]))
                    self.assertFalse(any(k.startswith("decoder.") for k in artifact["structure_branch"]))
                    losses = source["history"][-1]["train"]
                    if group in "DE":
                        self.assertGreater(losses["relation_kd"], 0)
                    else:
                        self.assertEqual(losses["relation_kd"], 0)

    def test_postselection_interventions_record_metrics_without_reselection(self):
        for summary in self.result["groups"].values():
            result = summary["intervention_validation"]
            self.assertFalse(result["selection_used"])
            self.assertEqual(set(result["metrics"]),
                             {"normal", "no_source", "uniform_source", "shuffle_source", "identity"})
            self.assertEqual(result["metrics"]["normal"]["macro"]["S_alpha"],
                             summary["cod_finetune"]["best_selection_score"])

    def test_structure_reuse_skips_source_and_reproduces_cod(self):
        output = self.work / "reuse"
        self.invoke(output, ["--groups", "E", "--structure-checkpoint",
            f"E={self.fresh / 'E/vdt/structure.pt'}", "--split-manifest",
            str(self.fresh / "split_manifest.json")], no_vdt=True)
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        actual, expected = result["groups"]["E"], self.result["groups"]["E"]
        self.assertEqual(actual["cod_initial_state_sha256"], expected["cod_initial_state_sha256"])
        self.assertEqual(actual["cod_finetune"]["best_selection_score"], expected["cod_finetune"]["best_selection_score"])
        self.assertEqual(result["teachers"], {})

    def test_reuse_rejects_changed_seed_budget_loss_and_vdt_members_without_source_io(self):
        base = ["--groups", "E", "--structure-checkpoint", f"E={self.fresh / 'E/vdt/structure.pt'}"]
        with self.assertRaisesRegex(ValueError, "original VDT"):
            self.invoke(self.work / "reuse-no-manifest", base, no_vdt=True)
        manifest = self.fresh / "split_manifest.json"
        for name, extra in (("seed", ["--train-seed", "12"]), ("budget", ["--vdt-epochs", "3"]),
                            ("loss", ["--structure-weight", "2"])):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "source_contract"):
                self.invoke(self.work / f"mismatch-{name}", base + ["--split-manifest", str(manifest)] + extra, no_vdt=True)
        changed = json.loads(manifest.read_text(encoding="utf-8"))
        changed["vdt_train"][0], changed["vdt_val"][0] = changed["vdt_val"][0], changed["vdt_train"][0]
        changed_path = self.work / "changed-source-members.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source_contract"):
            self.invoke(self.work / "mismatch-partition-full", base + ["--split-manifest", str(changed_path),
                        "--cod-train-mode", "full"], no_vdt=True, full=True)
        with self.assertRaisesRegex(ValueError, "source_contract"):
            self.invoke(self.work / "teacher-seed-mismatch", ["--groups", "E", "--teacher-checkpoint",
                        f"E={self.fresh / 'E/teacher/best.pt'}", "--train-seed", "12"])

    def test_observed_source_rejects_entire_train_or_val_split_without_valid_depth(self):
        for bad_split in ("train", "val"):
            data = MemoryVDTIndex()
            items = [runner.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, data), "vdt") for s in data.entries]
            train, val = runner.shared._split(items, 0.25, 7)
            for item in train if bad_split == "train" else val:
                sample, index = item.source
                location = sample["depth"].location
                index.assets[location] = Image.new("L", index.assets[location].size, 0)
            with self.subTest(split=bad_split), self.assertRaisesRegex(ValueError, f"C/vdt/{bad_split}.*no valid depth"):
                self.invoke(self.work / f"empty-depth-{bad_split}", ["--groups", "C"], source_items=items)
            result = json.loads((self.work / f"empty-depth-{bad_split}" / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertFalse((self.work / f"empty-depth-{bad_split}" / "C/vdt/structure.pt").exists())

    def test_full_mode_has_no_validation_and_uses_fixed_final_epoch(self):
        output = self.work / "full"
        self.invoke(output, ["--groups", "A", "--cod-train-mode", "full"], no_vdt=True, full=True)
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        summary = result["groups"]["A"]
        self.assertEqual(summary["cod_finetune"]["selection_metric"], "fixed_last_epoch")
        self.assertEqual(summary["cod_finetune"]["selected_epoch"], 2)
        self.assertNotIn("intervention_validation", summary)
        self.assertTrue(summary["cod_frozen_structure_verified"])
        with self.assertRaisesRegex(ValueError, "3040 COD10K"):
            runner._full_cod_pool(self.cod_items)

    def test_rejects_legacy_full_corrupt_and_wrong_stage_artifacts(self):
        model = runner.StructureStudent(model_variant="tiny", base_channels=2, relation_dim=4)
        protocol = self.result["protocol"]
        for name, state in (("legacy", {"format": "triprd_relation_branch_v1"}),
                            ("full", torch.load(self.fresh / "E/vdt/best.pt", weights_only=True))):
            path = self.work / f"{name}.pt"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "artifact"):
                load_structure_checkpoint(model, path, expected_protocol=protocol)
        original = self.fresh / "E/vdt/structure.pt"
        with self.assertRaisesRegex(ValueError, "source stage"):
            load_structure_checkpoint(model, original, expected_protocol=protocol, expected_stage="D/vdt")
        state = torch.load(original, weights_only=True)
        key = next(k for k, v in state["structure_branch"].items() if v.is_floating_point())
        state["structure_branch"][key] += 1
        path = self.work / "corrupt.pt"
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "digest"):
            load_structure_checkpoint(model, path, expected_protocol=protocol)

    def test_checkpoint_only_evaluation_does_not_read_training_data(self):
        output = self.work / "eval"
        index = MemoryCODIndex("COD10K", split="test", count=2)
        with patch.object(runner.shared, "_vdt_items", side_effect=AssertionError("VDT read")), \
             patch.object(runner.shared, "_cod_items", side_effect=AssertionError("COD train read")), \
             patch.object(runner.shared, "_index_for_dataset", side_effect=lambda *a: index), \
             patch.object(runner.shared, "COD10KIndex", side_effect=lambda *a: index), \
             redirect_stdout(io.StringIO()):
            runner.main(self.arguments(output) + ["--groups", "E", "--eval-only", "--no-skip-test",
                "--cod-checkpoint", f"E={self.fresh / 'E/cod/best.pt'}",
                "--test-datasets", "COD10K", "--max-test-samples", "2"])
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(result["test_status"], "complete")
        self.assertEqual(result["test"]["E"]["COD10K"]["official"]["num_samples"], 2)

    def test_failure_is_recorded_and_outputs_are_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            self.invoke(self.fresh, ["--groups", "A"])
        output = self.work / "failed"
        with patch.object(runner, "_fit_stage", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.invoke(output, ["--groups", "A"])
        result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "failed")


class StructureDataAndQueueTests(unittest.TestCase):
    def test_thermal_keeps_colors_that_have_identical_grayscale_intensity(self):
        pixels = np.array([[[255, 0, 0], [0, 130, 0]], [[255, 0, 0], [0, 130, 0]]], dtype=np.uint8)
        image = Image.fromarray(pixels)
        gray = np.asarray(image.convert("L"))
        self.assertEqual(int(gray[0, 0]), int(gray[0, 1]))
        value, valid, divisor = observation_tensor(image, 2)
        self.assertEqual(tuple(value.shape), (3, 2, 2))
        self.assertEqual(tuple(valid.shape), (1, 2, 2))
        self.assertTrue(valid.all())
        self.assertEqual(divisor, 255)
        self.assertFalse(torch.equal(value[:, 0, 0], value[:, 0, 1]))
        self.assertTrue(torch.equal(value[:, 0, 0], torch.tensor([1.0, 0.0, 0.0])))

    def test_gray_and_color_thermal_share_three_channel_batches_and_audit(self):
        data = MemoryVDTIndex(count=2)
        asset = data.entries[1]["thermal"]
        data.assets[asset.location] = data.assets[asset.location].convert("RGB")
        items = [runner.shared.Item(f"VDT/{s['rgb'].sample_id}", (s, data), "vdt") for s in data.entries]
        dataset = runner.StructurePairDataset(items, 32)
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(batch["thermal"].shape), (2, 3, 32, 32))
        self.assertEqual(tuple(batch["thermal_valid"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(batch["depth"].shape), (2, 1, 32, 32))
        records = list(dataset.sensor_records.values())
        self.assertEqual([r["thermal"]["original_mode"] for r in records], ["L", "RGB"])
        self.assertTrue(all(r["thermal"]["output_channels"] == 3 for r in records))
        self.assertTrue(all(r["thermal"]["validity_channels"] == 1 for r in records))

    def test_source_partition_digest_is_order_independent_and_requires_disjoint_members(self):
        args = runner.parse_args([])
        self.assertEqual(runner._source_contract(args, ["VDT/a", "VDT/b"], ["VDT/c"]),
                         runner._source_contract(args, ["VDT/b", "VDT/a"], ["VDT/c"]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner._source_contract(args, ["VDT/a"], ["VDT/a"])
        with self.assertRaisesRegex(ValueError, "no valid annotation"):
            runner._require_source_coverage({"structure_annotation_valid_fraction": 0}, "mask", stage="B/vdt/val")

    def test_uint16_scaling_preserves_original_mode_before_resize(self):
        image = Image.fromarray(np.full((2, 2), 32768, dtype=np.uint16))
        value, valid, divisor = observation_tensor(image, 4, depth=True)
        self.assertEqual(divisor, 65535)
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(value, torch.full_like(value, 32768 / 65535)))

    def test_invalid_depth_does_not_contaminate_resized_teacher_input(self):
        image = Image.fromarray(np.array([[0, 255], [255, 255]], dtype=np.uint8))
        value, valid, divisor = observation_tensor(image, 8, depth=True)
        self.assertEqual(divisor, 255)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.equal(value[valid], torch.ones_like(value[valid])))
        self.assertTrue(torch.equal(value[~valid], torch.zeros_like(value[~valid])))
        self.assertFalse(valid.all())
        thermal, thermal_valid, _ = observation_tensor(image, 8, depth=False)
        self.assertEqual(tuple(thermal.shape), (3, 8, 8))
        self.assertEqual(tuple(thermal_valid.shape), (1, 8, 8))
        self.assertTrue(thermal_valid.all())

    def test_nonfinite_sensor_input_is_sanitized_before_teacher(self):
        image = Image.fromarray(np.array([[float("nan"), 2], [float("inf"), -1]], dtype=np.float32))
        value, valid, divisor = observation_tensor(image, 4)
        self.assertEqual(divisor, 2)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((value >= 0).all() and (value <= 1).all())
        self.assertFalse(valid.all())

    def test_queue_dry_run_has_five_distinct_jobs_and_four_gpu_slots(self):
        with tempfile.TemporaryDirectory(prefix=".structure-queue-test-", dir=ROOT) as temporary:
            root = Path(temporary) / "absent"
            stream = io.StringIO()
            with redirect_stdout(stream):
                result = queue.main(["--output-root", str(root)])
            self.assertEqual(result, 0)
            plan = json.loads(stream.getvalue())
            self.assertFalse(root.exists())
            self.assertEqual([job["group"] for job in plan["jobs"]], list("ABCDE"))
            self.assertEqual([job["gpu"] for job in plan["jobs"]], ["0", "1", "2", "3", "0"])
            self.assertEqual(plan["jobs"][3]["teacher_inputs"], "RGB/RGB/RGB")
            self.assertEqual(plan["jobs"][4]["teacher_inputs"], "RGB/T/D")
            for job in plan["jobs"]:
                self.assertIn("--skip-test", job["command"])
                self.assertIn("structure_v2", job["output_dir"])


if __name__ == "__main__":
    unittest.main()
