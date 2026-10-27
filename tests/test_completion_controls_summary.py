"""Reject misleading cross-group comparisons before writing any summary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import summarize_triprd_completion_controls as summary

ROOT = Path(__file__).resolve().parents[1]
H, L = "a" * 64, "b" * 64


class ControlsSummaryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix=".controls-summary-", dir=ROOT)
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        common = {key: H for key in summary.MATCHED}
        common.update(data_parallel=False, training_device_count=1, cod_train_mode="split",
                      train_seed=11, split_seed=7, epochs=2, source_epochs=2, teacher_epochs=2)
        self.reports = {}
        for index, group in enumerate(summary.GROUPS):
            metrics = dict(S_alpha=.85 + index * .01, F_beta_w=.8 + index * .01, E_phi=.9, MAE=.03 - index * .001)
            validation = {"num_samples": 2, "macro": metrics.copy(),
                          "sources": {name: {"num_samples": 1, **metrics} for name in ("COD10K", "CAMO")}}
            protocol = {**common, "group": group, "cod_train": ["COD10K/train", "CAMO/train"],
                        "cod_val": ["COD10K/val", "CAMO/val"], "vdt_train": [], "vdt_val": []}
            report = {"method": summary.METHOD, "stage": "cod", "config": {"group": group},
                      "status": "complete", "protocol": protocol,
                      "generic_initial_state_sha256": H, "initial_state_sha256": H,
                      "cod_locator_before_sha256": L, "cod_locator_after_sha256": L,
                      "parameters": {"total": 100, "trainable": 90}, "checkpoint_sha256": group * 64,
                      "completed_epochs": 2, "best_epoch": 2, "test_status": "skipped", "test": {},
                      "history": [{"epoch": e, "train": {"seconds": 1., "steps": 1, "optimizer_steps": 1},
                                   "validation": copy.deepcopy(validation)} for e in (1, 2)]}
            self.reports[group + "/cod"] = report
            if group != "A":
                source = copy.deepcopy(report)
                source["stage"] = "source"
                source["protocol"].update(cod_train=[], cod_val=[], vdt_train=["VDT/train"], vdt_val=["VDT/val"])
                source["checkpoint_sha256"] = ("s" + group) * 32
                self.reports[group + "/source"] = source
                report["source_provenance"] = {"checkpoint_sha256": source["checkpoint_sha256"], "contract": source["protocol"]}
            if group in ("C", "D"):
                teacher = copy.deepcopy(source)
                teacher["stage"] = "teacher"
                teacher["parameters"] = {"total": 110, "trainable": 100}
                teacher["checkpoint_sha256"] = ("t" + group) * 32
                self.reports[group + "/teacher"] = teacher
                source["teacher_provenance"] = {"checkpoint_sha256": teacher["checkpoint_sha256"], "contract": teacher["protocol"]}

    def run_summary(self, export=False):
        def validate(path, formal=False):
            key = Path(path).relative_to(self.root).as_posix()
            return self.reports[key]
        with patch("tools.run_triprd_completion_control.validate_completed_run", side_effect=validate):
            result = summary.summarize(self.root)
            if export:
                summary.export(self.root, result)
            return result

    def test_development_reports_selected_epoch_and_raw_mae_sign(self):
        result = self.run_summary(export=True)
        self.assertEqual(result["scope"], "internal_validation")
        self.assertAlmostEqual(result["deltas"]["D-C"]["macro"]["S_alpha"], .01)
        self.assertAlmostEqual(result["deltas"]["D-C"]["macro"]["MAE"], -.001)
        self.assertEqual(set(result["costs"]["A"]), {"cod"})
        self.assertEqual(set(result["costs"]["D"]), {"cod", "source", "teacher"})
        saved = json.loads((self.root / "controls_summary.json").read_text())
        self.assertTrue(saved["matched_artifacts_verified"])
        self.assertIn("not statistical significance", saved["interpretation"])

    def test_missing_group_and_changed_protocol_rejected(self):
        original = copy.deepcopy(self.reports)
        mutations = (
            lambda r: r.pop("C/cod"),
            lambda r: r["D/cod"]["protocol"].update(train_seed=12),
            lambda r: r["D/cod"]["protocol"].update(micro_batch_size=4),
            lambda r: r["D/cod"]["protocol"].update(cod_train=["COD10K/other", "CAMO/train"]),
            lambda r: r["D/cod"].update(generic_initial_state_sha256="c" * 64),
            lambda r: r["D/teacher"].update(parameters={"total": 111, "trainable": 100}),
            lambda r: r["D/teacher"].update(initial_state_sha256="c" * 64),
            lambda r: r["D/source"].update(initial_state_sha256="c" * 64),
            lambda r: r["D/cod"]["source_provenance"].update(checkpoint_sha256="c" * 64),
        )
        for mutate in mutations:
            self.reports = copy.deepcopy(original)
            mutate(self.reports)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, KeyError)):
                self.run_summary(export=True)
            self.assertFalse((self.root / "controls_summary.md").exists())

    def test_development_cannot_contain_official_tests_or_fake_macro(self):
        report = self.reports["D/cod"]
        report["test_status"] = "complete"
        with self.assertRaises(ValueError):
            self.run_summary()
        report["test_status"] = "skipped"
        report["history"][-1]["validation"]["macro"]["S_alpha"] = .99
        with self.assertRaisesRegex(ValueError, "macro"):
            self.run_summary()

    def test_full_coverage_and_all_e_variants_are_required(self):
        for group in summary.GROUPS:
            report = self.reports[group + "/cod"]
            report["protocol"]["cod_train_mode"] = "full"
            report["protocol"]["cod_val"] = []
            report["test_status"] = "complete"
            report["test"] = {name: {"count": count,
                "official": {"num_samples": count, **{key: .8 for key in summary.METRIC_FIELDS}},
                "coverage": {"verified_samples": count, "expected_full_samples": count,
                             "limited": False, "exact_id_match": True}}
                for name, count in summary.DATASET_COUNTS.items()}
        result = self.run_summary()
        self.assertEqual(result["scope"], "official_tests")
        self.assertEqual(len(result["scores"]["D"]), 4)
        report = self.reports["D/cod"]["test"]["COD10K"]
        report["coverage"]["verified_samples"] -= 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_summary()
        report["coverage"]["verified_samples"] += 1
        report["official"]["E_phi_max"] = float("nan")
        with self.assertRaisesRegex(ValueError, "metric"):
            self.run_summary()


if __name__ == "__main__":
    unittest.main()
