"""Comparison-table protocol: complete coverage and adaptive, not mean, E."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools import summarize_triprd_correction as summary


class CorrectionSummaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix=".correction-summary-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for group in "ABCD":
            out = self.root / f"{group}_fixture"
            out.mkdir()
            checkpoint = f"fixture weights {group}".encode()
            (out / "best.pt").write_bytes(checkpoint)
            report = {
                "method": "triprd_correction_v3", "config": {"group": group}, "status": "complete",
                "best_epoch": 100, "formal": True, "selection_metric": "last_epoch", "history": [],
                "protocol": {"epochs": 100, "cod_train_mode": "full", "teacher_sha256": "teacher" if group in "CD" else None},
                "initial_state_sha256": "matched", "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
                "sample_counts": {"cod_train": 4040, "cod_val": 0}, "test_status": "complete", "test": {},
            }
            for name, count in summary.DATASET_COUNTS.items():
                report["test"][name] = {
                    "coverage": {"limited": False, "verified_samples": count, "expected_full_samples": count},
                    "count": count,
                    "official": {"num_samples": count, "S_alpha": .8, "E_phi_adaptive": .9,
                                 "F_beta_w": .7, "MAE": .03, "E_phi": .85},
                }
            (out / "comparison.json").write_text(json.dumps(report), encoding="utf-8")

    def test_four_group_completeness_is_explicit(self):
        self.assertEqual(summary.summarize(self.root, "ABCD")["status"], "complete")
        legacy = summary.summarize(self.root)
        self.assertEqual(legacy["status"], "incomplete")
        self.assertEqual(legacy["missing_groups"], ["E", "F"])

    def test_table_uses_adaptive_e_in_paper_order(self):
        out = self.root / "table"
        summary.export_tests(summary.summarize(self.root, "ABCD"), out)
        with (out / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 16)
        self.assertEqual([r["dataset"] for r in records[:4]], list(summary.DATASET_COUNTS))
        self.assertEqual(float(records[0]["E_phi_adaptive"]), .9)
        self.assertEqual(float(records[0]["E_phi_mean"]), .85)
        lines = [r for r in (out / "previous_performance_rows.tex").read_text().splitlines() if not r.startswith("%")]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0].split(" & ")[1:5], ["0.800", "0.900", "0.700", "0.030"])

    def test_invalid_metrics_coverage_and_full_training_rejected_before_export(self):
        good = summary.summarize(self.root, "ABCD")
        mutations = [
            lambda r: r["test"]["NC4K"]["coverage"].update(limited=True),
            lambda r: r["test"]["NC4K"]["official"].pop("E_phi_adaptive"),
            lambda r: r["test"]["NC4K"]["official"].update(E_phi_adaptive=float("nan")),
            lambda r: r["sample_counts"].update(cod_train=3232),
            lambda r: r.update(epoch=80),
            lambda r: r.update(test_status="skipped"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                bad = copy.deepcopy(good)
                mutate(bad["records"][-1])
                out = self.root / f"bad-{index}"
                with self.assertRaises(ValueError):
                    summary.export_tests(bad, out)
                self.assertFalse(out.exists())

    def test_changed_checkpoint_rejected(self):
        result = summary.summarize(self.root, "ABCD")
        (self.root / "D_fixture/best.pt").write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            summary.export_tests(result, self.root / "table")


if __name__ == "__main__":
    unittest.main()
