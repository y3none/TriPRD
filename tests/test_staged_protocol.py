"""Final export must prove full COD training and valid staged initialization."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_task_protocol as v4_fixtures
from tools import evaluate_triprd_staged as exporter


class StagedFinalReportTests(unittest.TestCase):
    def setUp(self):
        v4_fixtures.FinalReportTests.setUp(self)
        for obj in (self.good, self.checkpoint):
            obj.update(method=exporter.METHOD, stage="cod", group="S0")
            obj["config"].update(stage="cod", group="S0", source_epochs=20)
            obj.update(generic_initial_state_sha256="a" * 64, initial_state_sha256="a" * 64,
                       cod_decoder_before_sha256="b" * 64, cod_decoder_after_sha256="b" * 64,
                       source_provenance=None)
        contract = self.good["protocol"]
        contract.update(method=exporter.METHOD, stage="cod", group="S0", source_epochs=20, source_provenance=None)
        self.good["stage_exposure"] = {"optimization": ["COD_train"], "selection": [],
                                       "source_batches_in_cod": 0, "teacher_in_stage": False}
        self.checkpoint["stage_exposure"] = copy.deepcopy(self.good["stage_exposure"])
        contract["stage_exposure"] = copy.deepcopy(self.good["stage_exposure"])
        self.persist()

    def write(self, report):
        self.path.write_text(json.dumps(report), encoding="utf-8")

    def write_checkpoint(self, checkpoint, report):
        path = self.root / "best.pt"
        torch.save(checkpoint, path)
        report["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def persist(self):
        self.checkpoint["contract"] = copy.deepcopy(self.good["protocol"])
        self.checkpoint["contract_sha256"] = exporter._digest(self.checkpoint["contract"])
        self.good["contract_sha256"] = self.checkpoint["contract_sha256"]
        self.write_checkpoint(self.checkpoint, self.good)
        self.write(self.good)

    def add_source(self):
        for obj in (self.good, self.checkpoint):
            obj["group"] = obj["config"]["group"] = "S1"
            obj["initial_state_sha256"] = "c" * 64
        self.good["protocol"]["group"] = "S1"
        source = {**self.good["protocol"], "stage": "source", "cod_train_mode": "split",
                  "cod_train": [], "cod_val": [], "vdt_train": [f"VDT/train{i}" for i in range(838)],
                  "vdt_val": [f"VDT/val{i}" for i in range(210)]}
        provenance = {"contract": source, "contract_sha256": exporter._digest(source), "group": "S1",
                      "epoch": 15, "completed_epochs": 20, "checkpoint_sha256": "d" * 64,
                      "completed_checkpoint_sha256": "e" * 64, "generic_initial_state_sha256": "a" * 64}
        self.good["source_provenance"] = self.checkpoint["source_provenance"] = provenance
        self.good["protocol"]["source_provenance"] = provenance
        self.persist()

    def test_full_cod_s0_without_source_passes(self):
        self.assertEqual(exporter.validate_run_report(self.path), self.good)

    def test_full_cod_with_complete_matching_source_provenance_passes(self):
        self.add_source()
        self.assertEqual(exporter.validate_run_report(self.path), self.good)

    def test_changed_cod_decoder_or_initialization_evidence_is_rejected(self):
        self.good["cod_decoder_after_sha256"] = self.checkpoint["cod_decoder_after_sha256"] = "c" * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, "decoder"):
            exporter.validate_run_report(self.path)

    def test_missing_source_or_mismatched_source_seed_is_rejected(self):
        self.add_source()
        provenance = self.good["source_provenance"]
        provenance["contract"]["train_seed"] += 1
        provenance["contract_sha256"] = exporter._digest(provenance["contract"])
        self.persist()
        with self.assertRaisesRegex(ValueError, "train_seed"):
            exporter.validate_run_report(self.path)

    def test_source_that_used_cod_data_is_rejected(self):
        self.add_source()
        provenance = self.good["source_provenance"]
        provenance["contract"]["cod_train"] = ["COD10K/leak"]
        provenance["contract_sha256"] = exporter._digest(provenance["contract"])
        self.persist()
        with self.assertRaisesRegex(ValueError, "no COD"):
            exporter.validate_run_report(self.path)

    def test_incomplete_source_budget_is_rejected(self):
        self.add_source()
        self.good["source_provenance"]["completed_epochs"] = 19
        self.persist()
        with self.assertRaisesRegex(ValueError, "budget"):
            exporter.validate_run_report(self.path)

    def test_source_batches_or_teacher_in_final_are_rejected(self):
        for key, value in (("source_batches_in_cod", 1), ("teacher_in_stage", True), ("selection", ["COD_val"])):
            with self.subTest(key=key):
                previous = self.good["stage_exposure"][key]
                self.good["stage_exposure"][key] = value
                self.write(self.good)
                with self.assertRaisesRegex(ValueError, "exclude"):
                    exporter.validate_run_report(self.path)
                self.good["stage_exposure"][key] = previous

    def test_source_resource_change_or_checkpoint_exposure_drift_is_rejected(self):
        self.add_source()
        provenance = self.good["source_provenance"]
        provenance["contract"]["generic_weights_sha256"] = "f" * 64
        provenance["contract_sha256"] = exporter._digest(provenance["contract"])
        self.persist()
        with self.assertRaisesRegex(ValueError, "generic_weights_sha256"):
            exporter.validate_run_report(self.path)
        self.setUp()
        self.checkpoint["stage_exposure"]["teacher_in_stage"] = True
        self.persist()
        with self.assertRaisesRegex(ValueError, "exposure differs"):
            exporter.validate_run_report(self.path)

    def test_development_wrong_method_or_nonfinal_checkpoint_cannot_export(self):
        mutations = [lambda r: r["config"].update(cod_train_mode="split"),
                     lambda r: r["config"].update(model_variant="tiny"),
                     lambda r: r.update(method="triprd_task_v4"),
                     lambda r: r.update(selection_metric="macro_S_alpha"),
                     lambda r: r.update(best_epoch=20), lambda r: r.update(formal=False),
                     lambda r: r["sample_counts"].update(cod_train=3232, cod_val=808)]
        for mutate in mutations:
            report = copy.deepcopy(self.good)
            mutate(report)
            self.write(report)
            with self.assertRaises(ValueError):
                exporter.validate_run_report(self.path)


if __name__ == "__main__":
    unittest.main()
