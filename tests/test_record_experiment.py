"""Regression checks for experiment records; no training dependencies required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "record_experiment.py"


class RecordExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".record-test-", dir=ROOT)
        self.work = Path(self.temp.name).resolve()
        assert self.work.parent == ROOT
        self.addCleanup(self.temp.cleanup)
        self.run_dir = self.work / "run"
        self.run_dir.mkdir()
        self.log_dir = self.work / "records"

    def write_summary(self, results):
        (self.run_dir / "comparison.json").write_text(
            json.dumps({"protocol": {"seed": 7}, "test": {"A": results}}), encoding="utf-8"
        )

    def run_recorder(self, record_id="E001"):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--id", record_id,
             "--run-dir", str(self.run_dir), "--log-dir", str(self.log_dir),
             "--change", "check | metrics"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )

    def test_official_mae_is_separate_from_quick_statistics(self):
        self.write_summary({"CAMO": {
            "dice": 0.7, "mae": 0.15,
            "official": {"MAE": 0.05, "S_alpha": 0.8, "F_beta_w": 0.6,
                         "E_phi": 0.9, "Boundary_F": None},
        }})
        result = self.run_recorder()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = (self.log_dir / "E001.md").read_text(encoding="utf-8")
        self.assertIn("Quick Dice | Quick MAE | MAE (official)", record)
        self.assertIn("| A | CAMO | 0.700000 | 0.150000 | 0.050000 | 0.800000 | 0.600000 | 0.900000 |  | ok |", record)
        self.assertIn(r"check \| metrics", (self.log_dir / "index.md").read_text(encoding="utf-8"))

    def test_failed_missing_and_invalid_results_remain_visible(self):
        self.write_summary({
            "prediction_failed": {"status": "unavailable", "error": "image missing"},
            "evaluation_failed": {"dice": 0.7, "mae": 0.15,
                                  "official": {"status": "unavailable", "error": "bad | key\nretry"}},
            "missing_official": {"mae": 0.15},
            "invalid_result": None,
        })
        result = self.run_recorder()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = (self.log_dir / "E001.md").read_text(encoding="utf-8")
        self.assertIn("unavailable: image missing", record)
        self.assertIn(r"unavailable: bad \| key<br>retry", record)
        self.assertIn("missing/invalid official MAE", record)
        self.assertIn("invalid: test result must be an object", record)
        self.assertNotIn("| ok |", record)

    def test_duplicate_id_preserves_record_and_index(self):
        self.write_summary({})
        self.assertEqual(self.run_recorder().returncode, 0)
        before = {p.name: p.read_bytes() for p in self.log_dir.iterdir()}
        result = self.run_recorder()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Record already exists", result.stderr)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.log_dir.iterdir()})

    def test_path_and_reserved_ids_are_rejected_without_writes(self):
        self.write_summary({})
        for record_id in ("../escape", r"..\escape", "index", "README"):
            with self.subTest(record_id=record_id):
                result = self.run_recorder(record_id)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("filename-safe", result.stderr)
        self.assertFalse(self.log_dir.exists())
        self.assertFalse((self.work / "escape.md").exists())


if __name__ == "__main__":
    unittest.main()
