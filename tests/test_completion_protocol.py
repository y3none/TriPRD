"""Completion-v6 benchmark export and stage-boundary guardrails."""
from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluation.task_protocol import DATASET_COUNTS, METRIC_FIELDS, PREDICTION_PROTOCOL
from tools import evaluate_triprd_completion as exporter
from tools import run_triprd_completion as runner

ROOT = Path(__file__).resolve().parents[1]


class CompletionExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix=".completion-export-", dir=ROOT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.report = {
            "method": runner.METHOD, "status": "complete", "formal": True,
            "prediction_protocol": PREDICTION_PROTOCOL,
            "checkpoint_sha256": "a" * 64, "test_status": "complete",
            "test": {
                name: {
                    "count": count,
                    "official": {"num_samples": count, "S_alpha": .8, "F_beta_w": .7,
                                 "E_phi_adaptive": .84, "E_phi_mean": .86,
                                 "E_phi_max": .9, "MAE": .03},
                    "coverage": {"verified_samples": count,
                                 "expected_full_samples": count,
                                 "limited": False, "exact_id_match": True},
                } for name, count in DATASET_COUNTS.items()
            },
        }

    def export(self):
        with patch.object(runner, "validate_completed_run", return_value=self.report) as validate:
            with redirect_stdout(io.StringIO()):
                rows = exporter.export_summary(self.root)
        validate.assert_called_once_with(self.root.resolve(), formal=True)
        return rows

    def test_one_complete_model_exports_all_datasets_and_separate_e_measures(self):
        self.export()
        with (self.root / "benchmark.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["dataset"] for row in rows], list(DATASET_COUNTS))
        self.assertEqual(len(rows), 4)
        self.assertEqual(float(rows[0]["E_phi_adaptive"]), .84)
        self.assertEqual(float(rows[0]["E_phi_mean"]), .86)
        self.assertEqual(float(rows[0]["E_phi_max"]), .9)
        text = (self.root / "benchmark.md").read_text(encoding="utf-8")
        self.assertIn(self.report["checkpoint_sha256"], text)
        self.assertIn("VDT", text)

    def test_partial_or_invalid_metrics_never_write_benchmark_files(self):
        mutations = (
            lambda r: r.update(test_status="running"),
            lambda r: r["test"].pop("CHAMELEON"),
            lambda r: r["test"]["CAMO"].update(count=249),
            lambda r: r["test"]["CAMO"]["official"].update(num_samples=249),
            lambda r: r["test"]["CAMO"]["coverage"].update(verified_samples=249),
            lambda r: r["test"]["CAMO"]["coverage"].update(limited=True),
            lambda r: r["test"]["CAMO"]["coverage"].update(exact_id_match=False),
            lambda r: r["test"]["CAMO"]["official"].update(MAE=float("nan")),
            lambda r: r["test"]["CAMO"]["official"].update(E_phi_mean=True),
            lambda r: r["test"]["CAMO"]["official"].pop("E_phi_adaptive"),
        )
        original = copy.deepcopy(self.report)
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                self.report = copy.deepcopy(original)
                mutate(self.report)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse((self.root / "benchmark.csv").exists())
                self.assertFalse((self.root / "benchmark.md").exists())

    def test_training_provenance_is_validated_before_export(self):
        with patch.object(runner, "validate_completed_run", side_effect=ValueError("not full4040")):
            with self.assertRaisesRegex(ValueError, "full4040"):
                exporter.export_summary(self.root)
        self.assertFalse((self.root / "benchmark.csv").exists())


class CompletionCheckpointBoundaryTests(unittest.TestCase):
    def test_prior_method_checkpoints_are_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory(prefix=".completion-method-", dir=ROOT) as directory:
            path = Path(directory) / "best.pt"
            for method in ("triprd_task_v4", "triprd_staged_v5", "triprd_correction_v3"):
                contract = {"method": method, "stage": "source"}
                state = {"method": method, "stage": "source", "contract": contract,
                         "contract_sha256": runner._digest(contract), "model": {"weight": torch.zeros(1)}}
                torch.save(state, path)
                with self.subTest(method=method), self.assertRaises(ValueError):
                    runner.read_checkpoint(path)

    def test_official_test_guard_precedes_any_data_access(self):
        common = ["--stage", "cod", "--out", str(ROOT / "results/never-created-completion-guard"),
                  "--device", "cpu", "--test-after-training"]
        cases = (
            ["--cod-train-mode", "split"],
            ["--cod-train-mode", "full", "--model-variant", "tiny", "--no-pretrained"],
            ["--cod-train-mode", "full", "--max-samples", "4"],
            ["--cod-train-mode", "full", "--no-pretrained"],
        )
        for extra in cases:
            with self.subTest(extra=extra):
                with patch.object(runner, "prepare_data", side_effect=AssertionError("test guard accessed data")):
                    with self.assertRaises(ValueError):
                        runner.main(common + extra)

    def test_source_and_teacher_cannot_request_cod_tests(self):
        for stage in ("teacher", "source"):
            with self.subTest(stage=stage):
                args = runner.parse_args(["--stage", stage, "--out", "unused", "--test-after-training"])
                with self.assertRaises(ValueError):
                    runner.validate_args(args)

    def test_cod_never_accepts_a_teacher_checkpoint(self):
        args = runner.parse_args(["--stage", "cod", "--out", "unused",
                                  "--source-checkpoint", "source/best.pt",
                                  "--teacher-checkpoint", "teacher/best.pt"])
        with self.assertRaises(ValueError):
            runner.validate_args(args)





class CompletionFormalEvidenceTests(unittest.TestCase):
    """Synthetic artifact evidence exercises export policy without training on tests."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix=".completion-formal-", dir=ROOT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        a = runner.parse_args(["--stage", "cod", "--out", str(self.root),
                               "--cod-train-mode", "full", "--epochs", "4",
                               "--teacher-epochs", "2", "--source-epochs", "2",
                               "--warmup-epochs", "1", "--locator-warmup-epochs", "1"])
        self.config = runner._json(a)
        manifest = {
            "vdt_train": [f"VDT/train_{i}" for i in range(838)],
            "vdt_val": [f"VDT/val_{i}" for i in range(210)],
            "cod_train": [f"COD10K/train_{i}" for i in range(3040)] +
                         [f"CAMO/train_{i}" for i in range(1000)],
            "cod_val": [],
        }
        resources = {"generic_weights_sha256": "1" * 64,
                     "generic_config_sha256": "2" * 64,
                     "dino_source_sha256": "3" * 64,
                     "training_device_count": 1, "compute_parallel": "single_device"}
        teacher_args = copy.copy(a)
        teacher_args.stage, teacher_args.cod_train_mode = "teacher", "split"
        teacher_contract = runner._contract(teacher_args, manifest, resources)
        teacher_proof = self.proof(teacher_contract, "4")
        source_args = copy.copy(teacher_args)
        source_args.stage = "source"
        source_contract = runner._contract(source_args, manifest, resources,
                                          teacher_provenance=teacher_proof)
        source_proof = {
            **self.proof(source_contract, "5"),
            "transfer_scope": runner.TRANSFER_SCOPE,
            "refiner_source_sha256": "6" * 64,
            "transferred_state_sha256": "7" * 64,
            "transferred_keys_sha256": "8" * 64,
            "transferred_tensors": 2,
        }
        contract = runner._contract(a, manifest, resources, source_provenance=source_proof)
        self.state = {
            "method": runner.METHOD, "stage": "cod", "contract": contract,
            "config": copy.deepcopy(self.config),
            "epoch": 4, "best_epoch": 4, "best_score": 4.,
            "history": [{"epoch": epoch, "phase": "joint",
                         "train": {"optimizer_steps": 1}} for epoch in range(1, 5)],
            "model": {"refiner.weight": torch.ones(1), "locators.cod.weight": torch.zeros(1)},
            "generic_initial_state_sha256": "9" * 64,
            "initial_state_sha256": "a" * 64,
            "cod_locator_before_sha256": "b" * 64,
            "cod_locator_after_sha256": "b" * 64,
            "refiner_before_sha256": "c" * 64,
            "refiner_after_sha256": source_proof["refiner_source_sha256"],
            "source_provenance": source_proof, "teacher_provenance": None,
            "stage_exposure": contract["stage_exposure"],
        }
        self.state["trained_refiner_sha256"] = runner._digest_state(
            runner._tensor_subset(self.state["model"], "refiner."))
        self.report = {
            "method": runner.METHOD, "stage": "cod", "status": "complete",
            "formal": True, "result_scope": "benchmark",
            "prediction_protocol": PREDICTION_PROTOCOL,
            "selection_metric": "last_epoch", "completed_epochs": 4,
            "best_epoch": 4, "best_score": 4., "config": copy.deepcopy(self.config),
            "sample_counts": {"vdt_train": 0, "vdt_val": 0, "cod_train": 4040, "cod_val": 0},
            "history": copy.deepcopy(self.state["history"]),
        }
        self.persist()

    @staticmethod
    def proof(contract, digit):
        return {"contract": contract, "contract_sha256": runner._digest(contract),
                "checkpoint_sha256": digit * 64, "completed_checkpoint_sha256": digit * 64,
                "epoch": 1, "completed_epochs": 2}

    @classmethod
    def rehash_chain(cls, contract):
        for key in ("source_provenance", "teacher_provenance"):
            proof = contract.get(key)
            if isinstance(proof, dict):
                cls.rehash_chain(proof["contract"])
                proof["contract_sha256"] = runner._digest(proof["contract"])

    def persist(self):
        contract = self.state["contract"]
        self.rehash_chain(contract)
        self.state["contract_sha256"] = runner._digest(contract)
        self.report["protocol"] = copy.deepcopy(contract)
        self.report["contract_sha256"] = self.state["contract_sha256"]
        for key in runner.METADATA_FIELDS:
            self.report[key] = copy.deepcopy(self.state.get(key))
        for name in ("best.pt", "last.pt"):
            torch.save(self.state, self.root / name)
        self.report["checkpoint_sha256"] = runner.sha256_file(self.root / "best.pt")
        (self.root / "comparison.json").write_text(json.dumps(self.report), encoding="utf-8")

    def assert_invalid(self):
        self.persist()
        with self.assertRaises(ValueError):
            runner.validate_completed_run(self.root, formal=True)

    def test_complete_full4040_evidence_passes_without_dataset_access(self):
        with patch.object(runner, "prepare_data", side_effect=AssertionError("validator accessed training")):
            self.assertEqual(runner.validate_completed_run(self.root, formal=True), self.report)

    def test_sample_counts_and_fixed_prediction_protocol_are_required(self):
        original = copy.deepcopy(self.report)
        for mutate in (lambda r: r["sample_counts"].update(cod_train=4039),
                       lambda r: r.update(prediction_protocol="legacy_uint8_resize"),
                       lambda r: r.update(selection_metric="macro_S_alpha"),
                       lambda r: r.update(formal=1)):
            with self.subTest(mutation=mutate):
                self.report = copy.deepcopy(original)
                mutate(self.report)
                self.assert_invalid()

    def test_full_training_rejects_duplicate_ids_casefold_and_empty_stems(self):
        original = copy.deepcopy(self.state["contract"]["cod_train"])
        for replacement in (original[1], "COD10K/" + original[1].split("/", 1)[1].upper(), "COD10K/"):
            with self.subTest(replacement=replacement):
                self.state["contract"]["cod_train"] = list(original)
                self.state["contract"]["cod_train"][0] = replacement
                self.assert_invalid()

    def test_nested_source_and_teacher_must_remain_isolated_from_cod(self):
        source = self.state["contract"]["source_provenance"]["contract"]
        teacher = source["teacher_provenance"]["contract"]
        for contract in (source, teacher):
            with self.subTest(stage=contract["stage"]):
                contract["cod_train"] = ["COD10K/leaked_training_image"]
                self.assert_invalid()
                contract["cod_train"] = []

    def test_nested_teacher_budget_resource_and_membership_must_match_source(self):
        teacher_proof = self.state["contract"]["source_provenance"]["contract"]["teacher_provenance"]
        original = copy.deepcopy(teacher_proof)
        mutations = (lambda p: p.update(completed_epochs=1),
                     lambda p: p["contract"].update(generic_weights_sha256="f" * 64),
                     lambda p: p["contract"]["vdt_val"].__setitem__(0, "VDT/train_0"))
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                teacher_proof.clear()
                teacher_proof.update(copy.deepcopy(original))
                mutate(teacher_proof)
                self.assert_invalid()

    def test_fresh_locator_and_complete_refiner_evidence_are_required(self):
        original = copy.deepcopy(self.state)
        for mutate in (
            lambda s: s.update(cod_locator_after_sha256="d" * 64),
            lambda s: s.update(refiner_after_sha256="d" * 64),
            lambda s: s.update(cod_locator_before_sha256=None, cod_locator_after_sha256=None),
            lambda s: s["source_provenance"].update(transfer_scope="shared_only_exclude_decoders"),
            lambda s: s["model"]["refiner.weight"].add_(1),
        ):
            with self.subTest(mutation=mutate):
                self.state = copy.deepcopy(original)
                mutate(self.state)
                self.assert_invalid()

    def test_final_stage_cannot_report_teacher_source_batches_or_selection(self):
        exposure = self.state["stage_exposure"]
        original = copy.deepcopy(exposure)
        for key, value in (("teacher_in_stage", True), ("source_batches_in_cod", 1),
                           ("selection", ["COD_val"]), ("oracle_cue", True)):
            with self.subTest(key=key):
                exposure.clear()
                exposure.update(copy.deepcopy(original))
                exposure[key] = value
                self.assert_invalid()

    def test_checkpoint_must_be_the_predefined_last_epoch(self):
        self.state["epoch"] = self.state["best_epoch"] = self.report["best_epoch"] = 3
        self.assert_invalid()


if __name__ == "__main__":
    unittest.main()
