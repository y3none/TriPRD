"""Checkpoint evaluation orchestration and a real tiny CPU inference round trip."""

from contextlib import ExitStack, redirect_stdout
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import evaluate_structure_v2 as evaluation


ROOT = Path(__file__).resolve().parents[1]
GROUP_DIRS = {
    "A": "A_no_source", "B": "B_mask_structure", "C": "C_direct_observed",
    "D": "D_rgb_teacher", "E": "E_multimodal_teacher",
}
DATASETS = ("NC4K", "COD10K", "CAMO", "CHAMELEON")
COUNTS = {"NC4K": 4121, "COD10K": 2026, "CAMO": 250, "CHAMELEON": 76}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class StructureEvaluationScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".structure-evaluation-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.run_dir = self.work / "source"
        self.out = self.work / "evaluated"
        self.records = {}
        self.checkpoints = {}

    def make_source(self, groups="ABCDE", *, checkpoints=True, protocol=None):
        source_protocol = {
            "method": "triprd_structure_v2", "model_variant": "tiny", "pretrained": False,
            "base_channels": 2, "image_size": 32, "relation_dim": 4, "relation_kernel": 3,
            "rgb_normalization": "[0,1]",
            "train_seed": 11, "split_seed": 7, "cod_train_mode": "split",
            "sample_counts": {"cod_train": 3232, "cod_val": 808, "vdt_train": 0, "vdt_val": 0},
        }
        if protocol:
            source_protocol.update(protocol)
        for group in groups:
            directory = self.run_dir / GROUP_DIRS[group]
            checkpoint = directory / group / "cod" / "best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            payload = f"synthetic-{group}-checkpoint".encode()
            if checkpoints:
                checkpoint.write_bytes(payload)
            record = {
                "status": "complete", "test_status": "skipped", "test": {},
                "protocol": copy.deepcopy(source_protocol),
                "groups": {group: {
                    "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                    "cod_finetune": {"best_epoch": 2, "selection_metric": "S_alpha"},
                }},
            }
            self.records[group] = record
            self.checkpoints[group] = checkpoint
            write_json(directory / "comparison.json", record)

    def save_source(self, group):
        write_json(self.run_dir / GROUP_DIRS[group] / "comparison.json", self.records[group])

    def arguments(self, groups="A,B,C,D,E", output=None):
        return ["--run-dir", str(self.run_dir), "--out", str(output or self.out),
                "--cod-root", str(self.work / "data"), "--device", "cpu", "--groups", groups]

    def result(self, groups="ABCDE", counts=None):
        counts = counts or COUNTS
        result = {"status": "complete", "test_status": "complete", "groups": {}, "test": {}}
        for group in groups:
            result["groups"][group] = {
                "checkpoint_sha256": self.records[group]["groups"][group]["checkpoint_sha256"]}
            result["test"][group] = {}
            for index, dataset in enumerate(DATASETS):
                result["test"][group][dataset] = {
                    "count": counts[dataset],
                    "official": {
                        "num_samples": counts[dataset], "S_alpha": 0.71 + index / 100,
                        "E_phi_adaptive": 0.81 + index / 100, "F_beta_w": 0.61 + index / 100,
                        "MAE": 0.04 + index / 100, "E_phi": 0.91 + index / 100,
                        "alpha_E": 0.92 + index / 100,
                    },
                    "coverage": {"verified_samples": counts[dataset],
                                 "expected_full_samples": counts[dataset], "limited": False},
                }
        return result

    def run_mock(self, result, *, arguments=None):
        def execute(command, **kwargs):
            self.assertIn("--eval-only", command)
            self.assertIn("--no-skip-test", command)
            self.assertEqual(command[command.index("--max-test-samples") + 1], "0")
            output = Path(command[command.index("--out") + 1])
            write_json(output / "comparison.json", result)
            return subprocess.CompletedProcess(command, 0)

        execute_mock = Mock(side_effect=execute)
        with patch.object(evaluation, "subprocess", SimpleNamespace(run=execute_mock)), redirect_stdout(io.StringIO()):
            status = evaluation.main(arguments or self.arguments())
        return status, execute_mock

    def test_dry_run_handles_missing_checkpoints_without_writes_or_execution(self):
        self.make_source(checkpoints=False)
        stream = io.StringIO()
        with patch.object(evaluation, "subprocess", SimpleNamespace(run=Mock())) as process, redirect_stdout(stream):
            status = evaluation.main(self.arguments() + ["--dry-run"])
        self.assertEqual(status, 0)
        process.run.assert_not_called()
        self.assertFalse(self.out.exists())
        self.assertIn("--eval-only", stream.getvalue())
        self.assertIn("--no-skip-test", stream.getvalue())
        self.assertTrue(all(not path.exists() for path in self.checkpoints.values()))

    def test_missing_checkpoint_fails_before_execution(self):
        self.make_source(checkpoints=False)
        with patch.object(evaluation, "subprocess", SimpleNamespace(run=Mock())) as process:
            with self.assertRaises(FileNotFoundError):
                evaluation.main(self.arguments())
        process.run.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_changed_checkpoint_is_rejected_before_execution(self):
        self.make_source()
        self.checkpoints["C"].write_bytes(b"changed checkpoint")
        with patch.object(evaluation, "subprocess", SimpleNamespace(run=Mock())) as process:
            with self.assertRaises(ValueError):
                evaluation.main(self.arguments())
        process.run.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_missing_selected_group_is_rejected(self):
        self.make_source(groups="ABCD")
        with patch.object(evaluation, "subprocess", SimpleNamespace(run=Mock())) as process:
            with self.assertRaises((FileNotFoundError, ValueError)):
                evaluation.main(self.arguments())
        process.run.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_existing_output_is_not_overwritten(self):
        self.make_source()
        self.out.mkdir()
        sentinel = self.out / "precious-result.txt"
        sentinel.write_text("keep me", encoding="utf-8")
        with patch.object(evaluation, "subprocess", SimpleNamespace(run=Mock())) as process:
            with self.assertRaises((FileExistsError, ValueError)):
                evaluation.main(self.arguments())
        process.run.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")

    def test_mixed_source_model_protocols_are_rejected(self):
        self.make_source()
        self.records["B"]["protocol"]["relation_dim"] = 8
        self.save_source("B")
        with patch.object(evaluation, "subprocess", SimpleNamespace(run=Mock())) as process:
            with self.assertRaises(ValueError):
                evaluation.main(self.arguments())
        process.run.assert_not_called()

    def test_complete_evaluation_exports_adaptive_e_in_table_order(self):
        self.make_source()
        status, process = self.run_mock(self.result())
        self.assertEqual(status, 0)
        process.assert_called_once()
        table = (self.out / "previous_performance_rows.tex").read_text(encoding="utf-8")
        expected = " & ".join(f"{value:.3f}" for i in range(4)
                              for value in (0.71 + i / 100, 0.81 + i / 100, 0.61 + i / 100, 0.04 + i / 100))
        rows = [line for line in table.splitlines() if " & " in line and not line.lstrip().startswith("%")]
        self.assertEqual(len(rows), 5)
        for group, row in zip("ABCDE", rows):
            self.assertIn(f"TriPRD-{group}", row)
            self.assertIn(expected, row)
        with (self.out / "metrics.csv").open(encoding="utf-8-sig", newline="") as source:
            records = list(csv.DictReader(source))
        self.assertEqual(len(records), 20)
        self.assertEqual([row["dataset"] for row in records[:4]], list(DATASETS))
        manifest = json.loads((self.out / "evaluation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")

    def test_incomplete_or_invalid_results_never_export_completed_tables(self):
        self.make_source()
        cases = {
            "missing_dataset": lambda r: r["test"]["A"].pop("CAMO"),
            "missing_group": lambda r: r["test"].pop("E"),
            "missing_adaptive_e": lambda r: r["test"]["A"]["CAMO"]["official"].pop("E_phi_adaptive"),
            "wrong_count": lambda r: r["test"]["A"]["CAMO"].update(count=249),
            "wrong_official_count": lambda r: r["test"]["A"]["CAMO"]["official"].update(num_samples=249),
            "limited": lambda r: r["test"]["A"]["CAMO"]["coverage"].update(limited=True),
            "wrong_coverage": lambda r: r["test"]["A"]["CAMO"]["coverage"].update(verified_samples=249),
            "nonfinite": lambda r: r["test"]["A"]["CAMO"]["official"].update(E_phi_adaptive=float("nan")),
            "wrong_checkpoint": lambda r: r["groups"]["A"].update(checkpoint_sha256="0" * 64),
            "incomplete": lambda r: r.update(test_status="running"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                result = self.result()
                mutate(result)
                output = self.work / name
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_mock(result, arguments=self.arguments(output=output))
                self.assertFalse((output / "previous_performance_rows.tex").exists())
                self.assertFalse((output / "metrics.csv").exists())
                manifest_path = output / "evaluation_manifest.json"
                if manifest_path.exists():
                    self.assertNotEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["status"], "complete")

    def test_real_tiny_checkpoint_inference_and_scoring_do_not_touch_training(self):
        import torch

        from test_runner_integration import MemoryCODIndex
        from models.triprd_structure_loss import STRUCTURE_TARGET_SPEC
        from tools import run_triprd_structure as runner
        from tools.triprd_structure_training import SENSOR_PREPROCESSING

        self.make_source(groups="A", protocol={
            "structure_target_spec": STRUCTURE_TARGET_SPEC,
            "reference_encoder_policy": "source_initial_frozen_complete_function_transfer",
            "sensor_preprocessing": SENSOR_PREPROCESSING,
        })
        protocol = self.records["A"]["protocol"]
        model = runner.StructureStudent(model_variant="tiny", base_channels=2, relation_dim=4)
        torch.save({"model": model.state_dict(), "stage": "A/cod", "epoch": 2,
                    "seed": 11, "protocol": protocol}, self.checkpoints["A"])
        digest = hashlib.sha256(self.checkpoints["A"].read_bytes()).hexdigest()
        self.records["A"]["groups"]["A"]["checkpoint_sha256"] = digest
        self.save_source("A")
        small_counts = {dataset: 2 for dataset in DATASETS}

        def index_for_dataset(root, dataset):
            return MemoryCODIndex(dataset, split="test", count=2)

        def execute(command, **kwargs):
            self.assertEqual(command[1], "-u")
            self.assertIn("--eval-only", command)
            runner.main(command[3:])
            return subprocess.CompletedProcess(command, 0)

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluation, "subprocess", SimpleNamespace(run=execute)))
            stack.enter_context(patch.object(evaluation, "DATASET_COUNTS", small_counts))
            stack.enter_context(patch.object(runner, "TEST_COUNTS", small_counts))
            stack.enter_context(patch.object(runner.shared, "_index_for_dataset", side_effect=index_for_dataset))
            for dataset in DATASETS:
                stack.enter_context(patch.object(runner.shared, dataset + "Index",
                    side_effect=lambda *args, name=dataset: index_for_dataset(None, name)))
            for target, name in ((runner.shared, "_cod_items"), (runner.shared, "_vdt_items"),
                                 (runner, "_fit_stage"), (runner, "_fit_fixed_cod")):
                stack.enter_context(patch.object(target, name,
                    side_effect=AssertionError(f"Evaluation called training function {name}")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(evaluation.main(self.arguments(groups="A")), 0)
        result = json.loads((self.out / "comparison.json").read_text(encoding="utf-8"))
        self.assertTrue(result["protocol"]["eval_only"])
        self.assertEqual(result["protocol"]["sample_counts"]["cod_train"], 0)
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                metrics = result["test"]["A"][dataset]["official"]
                self.assertEqual(metrics["num_samples"], 2)
                self.assertIn("E_phi_adaptive", metrics)
                self.assertEqual(len(list((self.out / "A/predictions" / dataset).glob("*.png"))), 2)
        self.assertTrue((self.out / "previous_performance_rows.tex").exists())
        self.assertEqual(hashlib.sha256(self.checkpoints["A"].read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
