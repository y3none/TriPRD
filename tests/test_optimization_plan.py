"""Optimization plan and queue tests; never launch training or require a GPU."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_triprd_optimization as launcher


ROOT = Path(__file__).resolve().parents[1]


def option(job, name):
    return job["command"][job["command"].index(name) + 1]


class OptimizationPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".optimization-plan-test-", dir=ROOT)
        self.work = Path(self.temp.name).resolve()
        assert self.work.parent == ROOT
        self.addCleanup(self.temp.cleanup)
        self.argv = ["--source-run", str(self.work / "source"),
                     "--output-root", str(self.work / "output")]

    def plan(self, *args):
        return launcher.build_plan(launcher.parse_args(self.argv + list(args)))

    def checkpoints(self, plan):
        for name in {p for job in plan["jobs"] for p in job["required_checkpoints"]}:
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existence-check-only; never loaded by these tests")

    def test_cod_plan_is_isolated_and_uses_shared_validation(self):
        plan = self.plan()
        self.assertEqual([job["name"] for job in plan["jobs"]],
                         ["A_control", "A_schedule", "B_schedule", "C_schedule"])
        self.assertEqual([job["gpu"] for job in plan["jobs"]], ["0", "1", "2", "3"])
        self.assertEqual(len({job["output_dir"] for job in plan["jobs"]}), 4)
        for job in plan["jobs"]:
            self.assertIn("--skip-test", job["command"])
            self.assertIn("--validation-metrics", job["command"])
            self.assertEqual(option(job, "--selection-metric"), "S_alpha")
            self.assertEqual(option(job, "--split-seed"), "20261012")
            self.assertEqual(option(job, "--train-seed"), "20261012")
        self.assertEqual(option(plan["jobs"][0], "--cod-scheduler"), "none")
        self.assertEqual(option(plan["jobs"][0], "--cod-encoder-lr-scale"), "1")
        for job in plan["jobs"][1:]:
            self.assertEqual(option(job, "--cod-scheduler"), "cosine")
            self.assertEqual(option(job, "--cod-encoder-lr-scale"), "0.1")
        self.assertTrue(option(plan["jobs"][2], "--vdt-checkpoint").startswith("B="))
        self.assertTrue(option(plan["jobs"][3], "--vdt-checkpoint").startswith("C="))
        self.assertFalse(Path(plan["run_dir"]).exists())

    def test_distill_plan_adds_components_and_reuses_only_teacher(self):
        plan = self.plan("--phase", "distill", "--gpus", "2,3", "--seed", "8", "--split-seed", "7")
        self.assertEqual([job["name"] for job in plan["jobs"]],
                         ["C_logit", "D_relation", "D_boundary", "D_gate"])
        self.assertEqual([job["gpu"] for job in plan["jobs"]], ["2", "3", "2", "3"])
        for job in plan["jobs"]:
            self.assertIn("--teacher-checkpoint", job["command"])
            self.assertNotIn("--vdt-checkpoint", job["command"])
            self.assertEqual(option(job, "--train-seed"), "8")
            self.assertEqual(option(job, "--split-seed"), "7")
            self.assertIn("--skip-test", job["command"])
        self.assertEqual(option(plan["jobs"][0], "--groups"), "C")
        self.assertEqual(option(plan["jobs"][1], "--lambda-boundary-kd"), "0")
        for job in plan["jobs"][1:]:
            self.assertEqual(option(job, "--relation-sampling"), "uniform")
        self.assertEqual(option(plan["jobs"][2], "--boundary-kd-type"), "prob_mse")
        self.assertEqual(option(plan["jobs"][2], "--lambda-boundary-kd"), "0.1")
        self.assertEqual(option(plan["jobs"][2], "--gate-mode"), "none")
        self.assertEqual(option(plan["jobs"][3], "--gate-mode"), "disagreement")

    def test_transfer_plan_isolates_encoder_reuse_and_independent_repeats(self):
        plan = self.plan("--phase", "transfer")
        self.assertEqual([job["name"] for job in plan["jobs"]],
                         ["B_encoder", "C_encoder", "A_control_repeat", "A_schedule_repeat"])
        self.assertEqual([job["gpu"] for job in plan["jobs"]], ["0", "1", "2", "3"])
        self.assertEqual(Path(plan["run_dir"]),
                         self.work / "output" / "transfer" / "seed-20261012-repeat-20261013")
        self.assertEqual(len({job["output_dir"] for job in plan["jobs"]}), 4)
        self.assertEqual(plan["required_files"], [])
        for index, job in enumerate(plan["jobs"]):
            with self.subTest(job=job["name"]):
                seed = "20261012" if index < 2 else "20261013"
                self.assertEqual(option(job, "--seed"), seed)
                self.assertEqual(option(job, "--train-seed"), seed)
                self.assertEqual(option(job, "--split-seed"), "20261012")
                self.assertEqual(option(job, "--epochs"), "100")
                self.assertEqual(option(job, "--max-samples"), "0")
                self.assertEqual(option(job, "--selection-metric"), "S_alpha")
                self.assertEqual(option(job, "--val-metrics-every"), "5")
                self.assertIn("--validation-metrics", job["command"])
                self.assertIn("--skip-test", job["command"])
                self.assertNotIn("--teacher-checkpoint", job["command"])
                self.assertEqual(Path(job["output_dir"]).parent, Path(plan["run_dir"]))
                if index < 2:
                    group = "BC"[index]
                    checkpoint = self.work / "source" / group / "vdt" / "best.pt"
                    self.assertEqual(option(job, "--groups"), group)
                    self.assertEqual(option(job, "--transfer-scope"), "encoder")
                    self.assertEqual(option(job, "--vdt-checkpoint"), f"{group}={checkpoint}")
                    self.assertEqual(job["required_checkpoints"], [str(checkpoint)])
                else:
                    self.assertEqual(option(job, "--groups"), "A")
                    self.assertEqual(option(job, "--transfer-scope"), "full")
                    self.assertNotIn("--vdt-checkpoint", job["command"])
                    self.assertEqual(job["required_checkpoints"], [])
        self.assertFalse(Path(plan["run_dir"]).exists())

    def test_transfer_recipes_match_cod_without_duplicate_option_overrides(self):
        cod = self.plan()
        transfer = self.plan("--phase", "transfer")
        recipe_options = ("--cod-encoder-lr-scale", "--cod-scheduler",
                          "--cod-warmup-epochs", "--cod-min-lr-ratio",
                          "--learning-rate", "--weight-decay")
        for job, reference in zip(transfer["jobs"],
                                  [cod["jobs"][2], cod["jobs"][3], cod["jobs"][0], cod["jobs"][1]]):
            with self.subTest(job=job["name"]):
                for name in recipe_options:
                    self.assertEqual(option(job, name), option(reference, name))
        for plan in (cod, self.plan("--phase", "distill"), transfer):
            for job in plan["jobs"]:
                with self.subTest(phase=plan["phase"], job=job["name"]):
                    flags = [arg for arg in job["command"] if arg.startswith("--")]
                    self.assertEqual(len(flags), len(set(flags)), job["command"])

    def test_transfer_repeat_seed_changes_training_without_changing_split(self):
        for extra, expected in (([], "9"), (["--repeat-seed", "19"], "19")):
            with self.subTest(extra=extra):
                plan = self.plan("--phase", "transfer", "--seed", "8", "--split-seed", "7", *extra)
                self.assertEqual(Path(plan["run_dir"]).name, f"seed-8-repeat-{expected}")
                self.assertEqual([option(job, "--train-seed") for job in plan["jobs"]],
                                 ["8", "8", expected, expected])
                self.assertEqual([option(job, "--seed") for job in plan["jobs"]],
                                 ["8", "8", expected, expected])
                self.assertEqual([option(job, "--split-seed") for job in plan["jobs"]], ["7"] * 4)

    def test_repeat_seed_rejects_same_seed_and_nontransfer_phases(self):
        for args in (("--phase", "transfer", "--seed", "8", "--repeat-seed", "8"),
                     ("--phase", "cod", "--repeat-seed", "20261013"),
                     ("--phase", "distill", "--repeat-seed", "20261013")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(*args)
        self.assertFalse((self.work / "output").exists())

    def test_missing_split_manifest_is_forwarded_and_allowed_in_dry_run(self):
        manifest = self.work / "missing-manifest.json"
        for phase in ("cod", "distill", "transfer"):
            with self.subTest(phase=phase):
                stream = io.StringIO()
                args = self.argv + ["--phase", phase, "--split-manifest", str(manifest)]
                with patch.object(launcher, "execute_plan") as execute, contextlib.redirect_stdout(stream):
                    self.assertEqual(launcher.main(args), 0)
                execute.assert_not_called()
                plan = json.loads(stream.getvalue())
                self.assertEqual(plan["mode"], "dry-run")
                self.assertEqual(plan["required_files"], [str(manifest)])
                for job in plan["jobs"]:
                    self.assertEqual(option(job, "--split-manifest"), str(manifest))
                    self.assertEqual(job["command"].count("--split-manifest"), 1)
                    self.assertNotIn(str(manifest), job["required_checkpoints"])
                self.assertFalse(Path(plan["run_dir"]).exists())
        self.assertFalse((self.work / "output").exists())
        self.assertFalse(manifest.exists())

    def test_execute_rejects_missing_manifest_before_creating_or_starting_jobs(self):
        manifest = self.work / "missing-manifest.json"
        plan = self.plan("--phase", "transfer", "--split-manifest", str(manifest))
        self.checkpoints(plan)
        with patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaises(ValueError) as error:
                launcher.execute_plan(plan)
        self.assertIn(str(manifest), str(error.exception))
        popen.assert_not_called()
        self.assertFalse((self.work / "output").exists())

    def test_short_runs_use_a_valid_warmup(self):
        for epochs, warmup in ((1, "0"), (3, "2"), (100, "5")):
            with self.subTest(epochs=epochs):
                plan = self.plan("--epochs", str(epochs))
                for job in plan["jobs"][1:]:
                    self.assertEqual(option(job, "--cod-warmup-epochs"), warmup)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(self.argv + ["--epochs", "0"])

    def test_default_dry_run_never_starts_processes_or_writes_outputs(self):
        before = dict(os.environ)
        stream = io.StringIO()
        with patch.object(launcher, "execute_plan") as execute, contextlib.redirect_stdout(stream):
            self.assertEqual(launcher.main(self.argv), 0)
        execute.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue())["mode"], "dry-run")
        self.assertEqual(dict(os.environ), before)
        self.assertFalse((self.work / "output").exists())

    def test_invalid_gpu_lists_are_rejected(self):
        for value in ("", "0,0", "-1", "0,,1", "cpu", "01"):
            with self.subTest(gpus=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.parse_args(self.argv + ["--gpus", value])

    def test_existing_and_colliding_output_paths_are_rejected(self):
        plan = self.plan()
        collision = copy.deepcopy(plan)
        collision["jobs"][1]["output_dir"] = collision["jobs"][0]["output_dir"]
        with self.assertRaisesRegex(ValueError, "unique"):
            launcher.validate_plan(collision)
        Path(plan["run_dir"]).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.plan()

    def test_execute_checks_all_source_weights_before_starting(self):
        plan = self.plan()
        with patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "checkpoints are missing"):
                launcher.execute_plan(plan)
        popen.assert_not_called()
        self.assertFalse(Path(plan["run_dir"]).exists())

    def test_failure_stops_new_dispatch_and_waits_for_active_job(self):
        plan = self.plan("--gpus", "0,1")
        self.checkpoints(plan)
        calls = []
        polls = {"A_control": iter([None, 0]), "A_schedule": iter([1])}
        before = dict(os.environ)

        def popen(command, **kwargs):
            name = Path(command[command.index("--out") + 1]).name
            calls.append((name, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            self.assertEqual({p.name for p in Path(command[command.index("--out") + 1]).iterdir()},
                             {"launch.json", "run.log"})
            return SimpleNamespace(pid=100 + len(calls), poll=lambda: next(polls[name]))

        with contextlib.redirect_stdout(io.StringIO()):
            code = launcher.execute_plan(plan, popen_factory=popen, sleep=lambda _: None)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [("A_control", "0"), ("A_schedule", "1")])
        status = json.loads((Path(plan["run_dir"]) / "exit_status.json").read_text())["jobs"]
        self.assertEqual(status["A_control"]["status"], "completed")
        self.assertEqual(status["A_schedule"]["returncode"], 1)
        self.assertEqual(status["B_schedule"]["status"], "skipped")
        self.assertEqual(status["C_schedule"]["status"], "skipped")
        self.assertFalse(Path(plan["jobs"][2]["output_dir"]).exists())
        self.assertEqual(dict(os.environ), before)

    def test_single_gpu_queue_completes_all_jobs_without_overlap(self):
        plan = self.plan("--gpus", "3")
        self.checkpoints(plan)
        calls = []

        def popen(command, **kwargs):
            calls.append(command)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
            return SimpleNamespace(pid=100 + len(calls), poll=lambda: 0)

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.execute_plan(plan, popen_factory=popen, sleep=lambda _: None), 0)
        self.assertEqual(len(calls), 4)
        for job in plan["jobs"]:
            status = json.loads((Path(job["output_dir"]) / "exit_status.json").read_text())
            self.assertEqual(status["returncode"], 0)

    def test_process_start_failure_stops_dispatch_without_abandoning_active_job(self):
        plan = self.plan("--gpus", "0,1")
        self.checkpoints(plan)
        calls = []
        first_polls = iter([None, 0])

        def popen(command, **kwargs):
            name = Path(command[command.index("--out") + 1]).name
            calls.append(name)
            if len(calls) == 2:
                raise OSError("test executable unavailable")
            return SimpleNamespace(pid=123, poll=lambda: next(first_polls))

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.execute_plan(plan, popen_factory=popen, sleep=lambda _: None), 1)
        self.assertEqual(calls, ["A_control", "A_schedule"])
        status = json.loads((Path(plan["run_dir"]) / "exit_status.json").read_text())["jobs"]
        self.assertEqual(status["A_control"]["status"], "completed")
        self.assertEqual(status["A_schedule"]["returncode"], 127)
        self.assertIn("executable unavailable", status["A_schedule"]["error"])
        self.assertEqual(status["B_schedule"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
