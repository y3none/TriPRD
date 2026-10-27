"""Read-only plans, evidence gates, matched-stage reuse and failure propagation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import run_triprd_optimization_phases as launch


ROOT = Path(__file__).resolve().parents[1]


class OptimizationPhaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".optimization-phases-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.argv = ["--output-root", str(self.work / "runs"), "--model-variant", "tiny", "--no-pretrained",
                     "--device", "cpu", "--no-amp", "--num-workers", "0", "--base-channels", "4",
                     "--image-size", "32", "--epochs", "4", "--probe-epochs", "2", "--teacher-epochs", "3"]

    def plan(self, phase, *extra):
        return launch.build_plan(launch.parse_args(self.argv + ["--phase", phase, *extra]))

    def setup_root(self, plan):
        root = Path(plan["root"])
        launch._write(root / "optimization_recipe.json", {"recipe": plan["recipe"],
            "recipe_sha256": plan["recipe_sha256"], "data_input": plan["data_input"]})
        manifest = {"split_seed": 20261012, "cod_train_mode": "split", "vdt_train": ["s1", "s2"],
                    "vdt_val": ["s3"], "cod_train": ["c1", "c2"], "cod_val": ["c3"]}
        launch._write(Path(plan["manifest"]), manifest)
        launch._write(root / "data_manifest.json", {"sha256": launch._sha(plan["manifest"])})

    def report(self, plan, job, *, score=.7):
        manifest = launch._manifest(plan)
        protocol = {**plan["recipe"]["common"], **plan["recipe"]["resources"],
                    "source_sha256": plan["recipe"]["source_sha256"],
                    "stage": job["stage"], "group": job["group"], "epochs": job["config"]["epochs"],
                    "vdt_train": manifest["vdt_train"], "vdt_val": manifest["vdt_val"]}
        if job["stage"] == "student":
            full = job["config"]["cod_train_mode"] == "full"
            protocol.update(cod_train=manifest["cod_train"] + manifest["cod_val"] if full else manifest["cod_train"],
                            cod_val=[] if full else manifest["cod_val"], cod_train_mode=job["config"]["cod_train_mode"])
        if job["stage"] != "teacher" and job["group"] in "CD":
            protocol["teacher_sha256"] = launch._sha(Path(plan["root"]) / "teacher/best.pt")
        directory = Path(job["out"])
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / ("last.pt" if job["stage"] == "probe" else "best.pt")
        checkpoint.write_bytes((job["stage"] + job["group"]).encode())
        report = {"method": "triprd_correction_v3", "status": "complete", "config": job["config"],
                  "protocol": protocol, "contract_sha256": launch._digest(protocol),
                  "checkpoint_sha256": launch._sha(checkpoint), "checkpoint": str(checkpoint),
                  "initial_state_sha256": "shared-initial-state", "best_epoch": 4,
                  "selection_metric": "macro_S_alpha", "history": [{"epoch": 4,
                      "validation": {"macro": {"S_alpha": score}}}]}
        if job["stage"] == "teacher":
            report["teacher_probe"] = {"mean_relation_gain": .1, "positive_coverage": .5}
        if job["stage"] == "probe":
            report["source_recoverability"] = {"positive_mse": .01, "positive_zero_mse": .03,
                "positive_constant_mse": .02, "positive_count": 12., "count": 30.,
                "mse": .4, "zero_mse": .1, "constant_mse": .2}
        launch._write(directory / "comparison.json", report)
        return report

    def evidence(self, plan):
        self.setup_root(plan)
        self.report(plan, plan["teacher"])
        self.report(plan, plan["probe"])

    def test_baseline_and_probe_plans_keep_teacher_configuration_fixed(self):
        baseline, probe, screen, final = [self.plan(phase) for phase in ("baseline", "probe", "screen", "final")]
        self.assertEqual([job["group"] for job in baseline["jobs"]], ["A", "B"])
        self.assertEqual([job["stage"] for job in probe["jobs"]], ["teacher", "probe"])
        self.assertEqual([job["group"] for job in screen["jobs"]], list("ABCD"))
        self.assertEqual(baseline["teacher"], probe["teacher"])
        self.assertEqual(probe["teacher"], final["teacher"])
        for plan in (baseline, probe, screen, final):
            self.assertEqual(plan["recipe"], baseline["recipe"])
            for job in plan["jobs"]:
                self.assertNotIn("--cod-checkpoint", job["command"])
                self.assertEqual(job["config"]["warmup_epochs"], 5)
        self.assertEqual(probe["probe"]["config"]["epochs"], 2)
        for job in final["jobs"]:
            self.assertEqual(job["config"]["cod_train_mode"], "full")
            self.assertIn("--no-skip-test", job["command"])
            self.assertEqual(job["config"]["max_test_samples"], 0)
            self.assertEqual(job["config"]["test_datasets"], "NC4K,COD10K,CAMO,CHAMELEON")
        self.assertEqual([job["out"] for job in baseline["jobs"]], [job["out"] for job in screen["jobs"][:2]])

    def test_hf_config_changes_invalidate_recipe_and_match_runner_resources(self):
        from tools import run_triprd_correction as runner
        repo = self.work / "dinov3"
        repo.mkdir()
        (repo / "hubconf.py").write_text("# resource hash fixture", encoding="utf-8")
        snapshot = repo / "snapshots/master"
        snapshot.mkdir(parents=True)
        weights = snapshot / "model.safetensors"
        weights.write_bytes(b"hash-only fixture; never loaded")
        config = snapshot / "config.json"
        config.write_text('{"rope_theta": 100}', encoding="utf-8")
        extra = ["--model-variant", "dinov3_vitb16", "--pretrained",
                 "--dino-repo", str(repo), "--dino-weights", str(weights)]
        plan = self.plan("baseline", *extra)
        resources = runner._resources(runner.parse_args(["--out", str(self.work / "model"), *extra]))
        self.assertEqual(plan["recipe"]["resources"], resources)
        self.assertEqual(resources["generic_config_sha256"], launch._sha(config))
        self.assertIn("models/dinov3_weights.py", plan["recipe"]["source_sha256"])
        self.assertEqual(set(launch.RUNNER_SOURCES), set(runner.CODE_FILES))
        self.setup_root(plan)
        config.write_text('{"rope_theta": 200}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            self.plan("baseline", *extra)

    def test_hf_snapshot_weight_path_is_not_resolved_to_blob(self):
        # Exercise the path handed to the child runner without requiring Windows
        # symlink privileges. A real cache weight symlink would resolve to a blob.
        weights = self.work / "snapshot/model.safetensors"
        original = Path.resolve
        def resolve(path, *args, **kwargs):
            if path == weights:
                raise AssertionError("Do not dereference the snapshot weight path")
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve):
            plan = self.plan("baseline", "--model-variant", "dinov3_vitb16", "--pretrained",
                             "--dino-repo", str(self.work / "repo"), "--dino-weights", str(weights))
        self.assertEqual(plan["recipe"]["common"]["dino_weights"], str(weights.absolute()))

    def test_dry_run_never_writes_and_missing_screen_evidence_blocks(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(launch, "subprocess", SimpleNamespace(run=Mock())) as process:
            self.assertEqual(launch.main(self.argv + ["--phase", "baseline"]), 0)
            self.assertEqual(launch.main(self.argv + ["--phase", "screen"]), 2)
        process.run.assert_not_called()
        self.assertFalse((self.work / "runs").exists())

    def test_teacher_gate_rejects_negative_zero_and_nonfinite_evidence(self):
        for gain, coverage in ((0., .5), (-.1, .5), (.1, 0.), (.1, 1.1), (float("nan"), .5)):
            with self.subTest(gain=gain, coverage=coverage), self.assertRaises(launch.EvidenceBlocked):
                launch.teacher_gate({"teacher_probe": {"mean_relation_gain": gain, "positive_coverage": coverage}})

    def test_student_gate_matches_reliability_supervision_and_both_baselines(self):
        plan = self.plan("screen")
        self.evidence(plan)
        report = launch._read(Path(plan["probe"]["out"]) / "comparison.json")
        launch.student_gate(report)  # Poor all-edge MSE alone does not block positive-edge supervision.
        for key, value in (("positive_mse", .02), ("positive_zero_mse", .005),
                           ("positive_constant_mse", .005), ("positive_count", 0), ("count", 1),
                           ("positive_mse", float("inf"))):
            changed = copy.deepcopy(report)
            changed["source_recoverability"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(launch.EvidenceBlocked):
                launch.student_gate(changed)

    def test_changed_checkpoint_manifest_and_configuration_are_rejected(self):
        plan = self.plan("screen")
        self.evidence(plan)
        launch.check_gates(plan)
        checkpoint = Path(plan["probe"]["out"]) / "last.pt"
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(launch.EvidenceBlocked, "checkpoint"):
            launch.check_gates(plan)
        self.report(plan, plan["probe"])
        report_path = Path(plan["probe"]["out"]) / "comparison.json"
        report = launch._read(report_path)
        report["config"] = {**report["config"], "epochs": 5}
        launch._write(report_path, report)
        with self.assertRaisesRegex(launch.EvidenceBlocked, "configuration"):
            launch.check_gates(plan)
        launch._write(plan["manifest"], {})
        with self.assertRaisesRegex(launch.EvidenceBlocked, "manifest changed"):
            launch._manifest(plan)

    def test_screen_gate_failure_starts_no_jobs_and_records_reason(self):
        plan = self.plan("screen")
        self.evidence(plan)
        path = Path(plan["probe"]["out"]) / "comparison.json"
        report = launch._read(path)
        report["source_recoverability"]["positive_mse"] = .2
        launch._write(path, report)
        with patch.object(launch, "subprocess", SimpleNamespace(run=Mock())) as process, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.execute_plan(plan), 2)
        process.run.assert_not_called()
        state = launch._read(Path(plan["root"]) / "phases/screen.json")
        self.assertEqual(state["status"], "blocked")
        self.assertIn("BOTH zero", state["error"])

    def test_screen_revalidates_existing_baselines_and_initializes_new_students(self):
        plan = self.plan("screen")
        self.evidence(plan)
        for job in plan["screen"][:2]:
            self.report(plan, job)
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            job = next(job for job in plan["jobs"] if job["out"] == command[command.index("--out") + 1])
            self.report(plan, job)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(launch, "subprocess", SimpleNamespace(run=run)):
            self.assertEqual(launch.execute_plan(plan), 0)
        self.assertTrue(all("--resume" in command for command in calls[:2]))
        self.assertTrue(all("--resume" not in command and "--cod-checkpoint" not in command for command in calls[2:]))
        state = launch._read(Path(plan["root"]) / "phases/screen.json")
        self.assertEqual([job["reused"] for job in state["jobs"]], [True, True, False, False])

    def test_final_requires_completed_screen_and_d_beats_all_matched_controls(self):
        plan = self.plan("final")
        self.evidence(plan)
        with self.assertRaisesRegex(launch.EvidenceBlocked, "completed screen"):
            launch.check_gates(plan)
        launch._write(Path(plan["root"]) / "phases/screen.json", {"status": "complete"})
        for job in plan["screen"]:
            self.report(plan, job, score=.8 if job["group"] == "D" else .7)
        self.assertEqual(launch.check_gates(plan)["screen_S_alpha"]["D"], .8)
        self.report(plan, plan["screen"][2], score=.8)
        with self.assertRaisesRegex(launch.EvidenceBlocked, "D must beat"):
            launch.check_gates(plan)
        self.report(plan, plan["screen"][2], score=.7)
        path = Path(plan["screen"][1]["out"]) / "comparison.json"
        report = launch._read(path)
        report["initial_state_sha256"] = "unmatched"
        launch._write(path, report)
        with self.assertRaisesRegex(launch.EvidenceBlocked, "initialization"):
            launch.check_gates(plan)

    def test_subprocess_failure_propagates_without_marking_phase_complete(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        process = Mock(side_effect=subprocess.CalledProcessError(7, ["python"]))
        with patch.object(launch, "subprocess", SimpleNamespace(run=process)), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.execute_plan(plan), 1)
        state = launch._read(Path(plan["root"]) / "phases/baseline.json")
        self.assertEqual(state["status"], "failed")
        self.assertIn("7", state["error"])
        self.assertFalse((Path(plan["root"]) / "optimization_lock.json").exists())

    def test_changed_recipe_and_accidental_phase_overwrite_are_rejected(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            self.plan("probe", "--lr", ".001")
        receipt = Path(plan["root"]) / "phases/baseline.json"
        launch._write(receipt, {"status": "complete", "precious": True})
        before = receipt.read_bytes()
        with self.assertRaisesRegex(ValueError, "--resume"):
            launch.execute_plan(plan)
        self.assertEqual(receipt.read_bytes(), before)

    def test_interrupted_manifest_transaction_is_revalidated_before_receipt_repair(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        root = Path(plan["root"])
        (root / "data_manifest.json").unlink()
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if "--prepare-only" in command:
                self.assertIn("--resume", command)
                self.assertFalse((root / "data_manifest.json").exists())
            else:
                job = next(job for job in plan["jobs"] if job["out"] == command[command.index("--out") + 1])
                self.report(plan, job)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(launch, "subprocess", SimpleNamespace(run=run)):
            self.assertEqual(launch.execute_plan(plan, resume=True), 0)
        self.assertEqual(calls[0], plan["prepare_command"] + ["--resume"])
        launch._manifest(plan)

    def test_failed_manifest_revalidation_cannot_create_receipt(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        receipt = Path(plan["root"]) / "data_manifest.json"
        receipt.unlink()
        process = Mock(side_effect=subprocess.CalledProcessError(1, ["prepare"]))
        with patch.object(launch, "subprocess", SimpleNamespace(run=process)), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.execute_plan(plan, resume=True), 1)
        self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
