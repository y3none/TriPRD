"""Stdlib fixture tests for extracting saved boundary diagnostics without models."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools import summarize_triprd_boundary_diagnostics as summary


def record(mean, count):
    return {"mean": mean if count else None, "sum": mean * count,
            "count": count, "max": mean if count else None}


def add_records(rows):
    count = sum(row["count"] for row in rows)
    total = sum(row["sum"] for row in rows)
    return {"count": count, "sum": total, "mean": total / count if count else None,
            "max": max((row["max"] for row in rows if row["count"]), default=None)}


def fixture_statistics(*, empty_crossing=False, no_positive_crossing=False, subset_recovery=False):
    stats = {}
    for domain in ("VDT", "COD10K", "CAMO"):
        sections = ("relation", "recovery") if domain == "VDT" else ("relation",)
        for section in sections:
            fields = summary.RELATION_FIELDS if section == "relation" else summary.RECOVERY_FIELDS
            for scale, multiplier in (("stride8", 2), ("stride16", 1)):
                for stratum, units in (("fg", 2), ("bg", 8), ("crossing", 4)):
                    n = multiplier * units
                    if stratum == "crossing" and empty_crossing:
                        n = 0
                    if section == "recovery" and subset_recovery:
                        n //= 2
                    positive = n // 2
                    if stratum == "crossing" and no_positive_crossing:
                        positive = 0
                    # Different scale means make a naive mean-of-means fail.
                    mse = .03 if scale == "stride8" else .09
                    values = {"base_bce": .4, "corrected_bce": .3, "raw_abs": .1,
                        "clipped_abs": .08, "clip_fraction": .2,
                        "mse": mse, "zero_mse": .08, "constant_mse": .06,
                        "positive_coverage": positive / n if n else 0,
                        "positive_mse": mse, "positive_zero_mse": .1, "positive_constant_mse": .07,
                        "teacher_rgb_bce": .25, "teacher_full_bce": .15,
                        "teacher_rgb_plus_student_bce": .2, "student_base_plus_teacher_delta_bce": .18}
                    for field in fields:
                        count = positive if field in ("positive_mse", "positive_zero_mse", "positive_constant_mse") else n
                        stats[f"{domain}/{section}/{scale}/{stratum}/{field}"] = record(values[field], count)
                for field in fields:
                    stats[f"{domain}/{section}/{scale}/all/{field}"] = add_records([
                        stats[f"{domain}/{section}/{scale}/{stratum}/{field}"] for stratum in ("fg", "bg", "crossing")])
            for stratum in summary.STRATA:
                for field in fields:
                    stats[f"{domain}/{section}/pooled/{stratum}/{field}"] = add_records([
                        stats[f"{domain}/{section}/{scale}/{stratum}/{field}"] for scale in ("stride8", "stride16")])
    return stats


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def make_fixture(root, *, cod_per_source=2, vdt_count=3, **stats_options):
    root.mkdir()
    cod_keys = [f"{source}/image{i}" for source in ("COD10K", "CAMO") for i in range(cod_per_source)]
    contract = {"method": summary.METHOD, "groups": ["C", "D"], "scope": "internal_validation_only",
        "limit_cod_per_source": 0, "limit_vdt": 0, "reselect_checkpoint": False,
        "fit_constants": False, "optimize": False, "modes": list(summary.MODES),
        "training_recipe": {"official_test_access": False,
            "split_manifest": {"cod_train_mode": "split", "cod_val": cod_keys,
                               "vdt_val": [f"VDT/image{i}" for i in range(vdt_count)]}},
        "references": {group: {"directory": f"/original/{group}", "selected_epoch": 15,
            "checkpoint_sha256": "1" * 64, "report_sha256": "2" * 64,
            "initial_state_sha256": "3" * 64} for group in ("C", "D")}}
    write_json(root / "manifest.json", {"status": "complete", "contract": contract})
    for group in ("C", "D"):
        directory = root / group
        directory.mkdir()
        per_image = "".join(json.dumps({"key": key, "source": key.split("/", 1)[0],
            "S_alpha": {mode: .88 for mode in summary.MODES}}) + "\n" for key in cod_keys).encode()
        (directory / "cod_per_image.jsonl").write_bytes(per_image)
        report = {"method": summary.METHOD, "status": "complete", "group": group,
            "selected_epoch": 15, "reference": contract["references"][group], "student_unchanged": True,
            "contract_sha256": summary.digest(contract), "coverage": {"cod": len(cod_keys),
                "vdt": vdt_count, "partial_requested": False, "full_cod_validation": True,
                "full_vdt_validation": True}, "per_image_sha256": hashlib.sha256(per_image).hexdigest(),
            "path_statistics": fixture_statistics(**stats_options),
            "reproduction": {"macro_S_alpha_delta": 0., "positive_mse_delta": 1e-8}}
        write_json(directory / "diagnostics.json", report)


class BoundarySummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "input"

    def mutate_report(self, mutate):
        path = self.root / "C" / "diagnostics.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        write_json(path, value)

    def snapshot(self):
        return {str(path.relative_to(self.root)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.root.rglob("*") if path.is_file()}

    def test_weighted_numbers_gains_and_report_scope(self):
        make_fixture(self.root)
        result = summary.summarize(self.root)
        row = result["groups"]["D"]["crossing_relations"]["COD10K"]["pooled"]
        self.assertEqual(row["all_count"], 12)
        self.assertAlmostEqual(row["bce_gain"], .1)
        self.assertAlmostEqual(row["clipped_to_raw_ratio"], .8)
        recovery = result["groups"]["D"]["source_recovery"]["pooled"]["crossing"]
        self.assertEqual(recovery["positive_count"], 6)
        self.assertAlmostEqual(recovery["positive_coverage"], .5)
        self.assertAlmostEqual(recovery["positive_mse"], .05)
        cf = result["groups"]["D"]["counterfactuals"]["pooled"]["crossing"]
        self.assertAlmostEqual(cf["student_base_plus_teacher_delta_gain"], .22)
        self.assertAlmostEqual(cf["teacher_rgb_plus_student_gain"], .05)
        self.assertAlmostEqual(cf["teacher_full_gain"], .1)
        text = summary.format_summary(result)
        self.assertIn("not segmentation oracles", text)
        self.assertIn("no retraining or official-test access", text)
        self.assertIn("no automatic training decision", text)

    def test_empty_support_is_null_and_does_not_become_zero_error(self):
        make_fixture(self.root, empty_crossing=True)
        result = summary.summarize(self.root)["groups"]["C"]
        crossing = result["crossing_relations"]["VDT"]["pooled"]
        self.assertIsNone(crossing["bce_gain"])
        self.assertIsNone(crossing["clipped_to_raw_ratio"])
        self.assertIsNone(result["source_recovery"]["pooled"]["crossing"]["positive_mse"])
        self.assertIsNone(result["counterfactuals"]["pooled"]["crossing"]["teacher_full_gain"])

    def test_no_positive_crossing_keeps_all_edge_counterfactuals(self):
        make_fixture(self.root, no_positive_crossing=True)
        result = summary.summarize(self.root)["groups"]["C"]
        row = result["source_recovery"]["pooled"]["crossing"]
        self.assertEqual(row["positive_count"], 0)
        self.assertEqual(row["all_count"], 12)
        self.assertEqual(row["positive_coverage"], 0)
        self.assertIsNone(row["positive_mse"])
        self.assertIsNotNone(result["counterfactuals"]["pooled"]["crossing"]["teacher_full_gain"])

    def test_unequal_student_and_teacher_support_does_not_subtract_means(self):
        make_fixture(self.root, subset_recovery=True)
        row = summary.summarize(self.root)["groups"]["C"]["counterfactuals"]["pooled"]["crossing"]
        self.assertFalse(row["student_support_matches"])
        self.assertIsNone(row["student_base_bce"])
        self.assertIsNone(row["student_base_plus_teacher_delta_gain"])
        self.assertAlmostEqual(row["teacher_full_gain"], .1)

    def test_accepts_complete_splits_without_hardcoded_808_or_210(self):
        make_fixture(self.root, cod_per_source=5, vdt_count=7)
        result = summary.summarize(self.root, ("D",))
        self.assertEqual(list(result["groups"]), ["D"])
        self.assertEqual(result["groups"]["D"]["coverage"]["cod"], 10)

    def test_missing_statistic_is_rejected(self):
        make_fixture(self.root)
        self.mutate_report(lambda value: value["path_statistics"].pop("VDT/recovery/stride8/crossing/teacher_full_bce"))
        with self.assertRaisesRegex(ValueError, "Missing statistic"):
            summary.summarize(self.root)

    def test_tampered_per_image_file_is_rejected(self):
        make_fixture(self.root)
        path = self.root / "C" / "cod_per_image.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Per-image hash mismatch"):
            summary.summarize(self.root)

    def test_contract_reference_group_method_and_completion_are_checked(self):
        mutations = [
            (lambda r: r.update(contract_sha256="0" * 64), "Contract hash"),
            (lambda r: r["reference"].update(checkpoint_sha256="4" * 64), "Reference mismatch"),
            (lambda r: r.update(group="D"), "Incomplete or incompatible"),
            (lambda r: r.update(method="other"), "Incomplete or incompatible"),
            (lambda r: r.update(status="running"), "Incomplete or incompatible"),
            (lambda r: r["coverage"].update(full_cod_validation=False), "Incomplete internal"),
            (lambda r: r["coverage"].update(vdt=1), "Incomplete internal"),
        ]
        for index, (mutation, message) in enumerate(mutations):
            with self.subTest(message=message):
                self.root = self.base / f"input{index}"
                make_fixture(self.root)
                self.mutate_report(mutation)
                with self.assertRaisesRegex(ValueError, message):
                    summary.summarize(self.root)

    def test_running_manifest_is_rejected(self):
        make_fixture(self.root)
        path = self.root / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "running"
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "manifest is not complete"):
            summary.summarize(self.root)

    def test_pooled_statistics_and_coverage_are_checked(self):
        for index, field in enumerate(("positive_mse", "positive_coverage")):
            with self.subTest(field=field):
                self.root = self.base / f"stats{index}"
                make_fixture(self.root)
                self.mutate_report(lambda r: r["path_statistics"][f"VDT/recovery/pooled/crossing/{field}"].update(
                    mean=.75, sum=.75 * r["path_statistics"][f"VDT/recovery/pooled/crossing/{field}"]["count"], max=.75))
                with self.assertRaises(ValueError):
                    summary.summarize(self.root)

    def test_read_only_default_and_optional_disjoint_output(self):
        make_fixture(self.root)
        before = self.snapshot()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(summary.main(["--input-root", str(self.root)]), 0)
        self.assertIn("VDT counterfactual pooled crossing", stdout.getvalue())
        self.assertEqual(before, self.snapshot())
        output = self.base / "output"
        with redirect_stdout(io.StringIO()):
            summary.main(["--input-root", str(self.root), "--output-dir", str(output)])
        self.assertTrue((output / "boundary_summary.json").is_file())
        self.assertTrue((output / "boundary_summary.txt").is_file())
        self.assertEqual(before, self.snapshot())
        for bad in (self.root, self.root / "nested", self.base, output):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summary.main(["--input-root", str(self.root), "--output-dir", str(bad)])

    def test_cli_runs_without_site_packages_or_torch(self):
        make_fixture(self.root)
        process = subprocess.run([sys.executable, "-S", str(Path(summary.__file__).resolve()),
                                  "--input-root", str(self.root), "--groups", "D"],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("[D] selected_epoch=15", process.stdout)
        self.assertNotIn("[C] selected_epoch", process.stdout)


if __name__ == "__main__":
    unittest.main()
