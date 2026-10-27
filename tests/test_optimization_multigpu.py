"""Exercise GPU assignment with real, CPU-only subprocesses and valid evidence.

The children never import torch or access CUDA. They stand in for training by
writing fixed checkpoint bytes and reports; the real orchestrator still checks
all configuration, partition, checkpoint and gate contracts.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import test_optimization_phases as phase_tests
from tools import run_triprd_optimization_phases as launch


CHILD = r'''
import hashlib
import json
import os
from pathlib import Path
import sys
import time

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(spec["out"])
out.mkdir(parents=True, exist_ok=True)
resume = "--resume" in sys.argv
reused = resume and (out / "comparison.json").is_file()
def event(kind):
    row = {"event": kind, "time": time.monotonic_ns(), "pid": os.getpid(),
           "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "cwd": os.getcwd(),
           "out": str(out), "action": "verify" if reused else "train"}
    with (out / "fixture_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)

event("start")
(out / "fixture_started").write_text(str(os.getpid()), encoding="utf-8")
deadline = time.monotonic() + 5
while any(not Path(peer).is_file() for peer in spec.get("barrier", [])):
    if time.monotonic() > deadline:
        raise SystemExit("Parallel siblings never started")
    time.sleep(.01)
time.sleep(spec.get("delay", .15))
if spec.get("fail"):
    raise SystemExit(spec["fail"])
if reused:
    report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(Path(report["checkpoint"]).read_bytes()).hexdigest()
    if report["checkpoint_sha256"] != actual:
        raise SystemExit("Resume checkpoint changed")
elif not spec.get("missing_report"):
    Path(spec["report"]["checkpoint"]).write_bytes(spec["checkpoint_bytes"].encode())
    (out / "comparison.json").write_text(json.dumps(spec["report"]), encoding="utf-8")
event("end")
'''


class OptimizationMultiGPUTests(unittest.TestCase):
    def setUp(self):
        phase_tests.OptimizationPhaseTests.setUp(self)
        # A CUDA label exercises the production CLI; the fixture children use
        # Python's standard library exclusively and need no GPU or CUDA runtime.
        self.argv[self.argv.index("--device") + 1] = "cuda"
        self.script = self.work / "child.py"
        self.script.write_text(CHILD, encoding="utf-8")
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def plan(self, phase, gpus="0,1,2,3"):
        return phase_tests.OptimizationPhaseTests.plan(self, phase, "--gpus", gpus)

    def setup_root(self, plan):
        phase_tests.OptimizationPhaseTests.setup_root(self, plan)

    def report(self, plan, job, **kwargs):
        return phase_tests.OptimizationPhaseTests.report(self, plan, job, **kwargs)

    def evidence(self, plan):
        phase_tests.OptimizationPhaseTests.evidence(self, plan)

    def fixture(self, plan, job, *, keep=False, **options):
        report = self.report(plan, job)
        if job["config"]["cod_train_mode"] == "full":
            report.update(selection_metric="last_epoch", best_epoch=plan["recipe"]["epochs"],
                          test_status="complete", test={name: {"official": {
                              "num_samples": count, "S_alpha": .7, "E_phi_adaptive": .8,
                              "F_beta_w": .6, "MAE": .05}}
                              for name, count in launch.TEST_COUNTS.items()})
        directory = Path(job["out"])
        spec = {"out": job["out"], "report": report,
                "checkpoint_bytes": Path(report["checkpoint"]).read_text(encoding="utf-8"), **options}
        spec_path = directory / "fixture.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        job["command"] = [sys.executable, "-u", str(self.script), str(spec_path)]
        if keep:
            launch._write(directory / "comparison.json", report)
        else:
            (directory / "comparison.json").unlink()
            Path(report["checkpoint"]).unlink()
        return spec_path

    def events(self, job):
        path = Path(job["out"]) / "fixture_events.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def execute(self, plan, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return launch.execute_plan(plan, **kwargs)

    def receipt(self, plan):
        return launch._read(Path(plan["root"]) / "phases" / (plan["phase"] + ".json"))

    def parallel_fixtures(self, plan, jobs, **options):
        barrier = [str(Path(job["out"]) / "fixture_started") for job in jobs]
        for job in jobs:
            self.fixture(plan, job, barrier=barrier, **options)

    def assert_overlap_and_isolation(self, jobs, expected_tokens):
        events = [self.events(job) for job in jobs]
        self.assertTrue(all([row["event"] for row in attempt] == ["start", "end"] for attempt in events))
        self.assertLess(max(attempt[0]["time"] for attempt in events),
                        min(attempt[1]["time"] for attempt in events))
        self.assertEqual({attempt[0]["visible"] for attempt in events}, set(expected_tokens))
        self.assertEqual(len({attempt[0]["pid"] for attempt in events}), len(jobs))
        for job, attempt in zip(jobs, events):
            self.assertEqual(Path(attempt[0]["cwd"]).resolve(), launch.ROOT)
            self.assertEqual(attempt[0]["out"], job["out"])
            self.assertNotIn(",", attempt[0]["visible"])
            log = Path(job["out"]) / "run.log"
            self.assertTrue(log.is_file())
            self.assertIn('"event": "end"', log.read_text(encoding="utf-8"))

    def final_plan(self, gpus="0,1,2,3"):
        plan = self.plan("final", gpus)
        self.setup_root(plan)
        manifest = launch._read(plan["manifest"])
        manifest.update(cod_train=[f"c{i}" for i in range(3232)], cod_val=[f"c{i}" for i in range(3232, 4040)])
        launch._write(plan["manifest"], manifest)
        launch._write(Path(plan["root"]) / "data_manifest.json", {"sha256": launch._sha(plan["manifest"])})
        self.report(plan, plan["teacher"])
        self.report(plan, plan["probe"])
        for job in plan["screen"]:
            self.report(plan, job, score=.8 if job["group"] == "D" else .7)
        launch._write(Path(plan["root"]) / "phases/screen.json", {"status": "complete"})
        plan["summary_command"] = [sys.executable, "-c", "print('fixture summary complete')"]
        return plan

    def test_gpu_tokens_and_existing_visibility_are_validated(self):
        self.assertEqual(launch._parse_gpu_ids("0, 2,3", visible="0,2,3,5"), ["0", "2", "3"])
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        self.assertEqual(launch._parse_gpu_ids(uuid, visible=uuid), [uuid])
        for value in ("", " ", "0,", ",0", "0,,1", "0,0", "-1", "x", "0," + uuid):
            with self.subTest(value=value), self.assertRaises(ValueError):
                launch._parse_gpu_ids(value)
        for visible in ("", "-1", "0,1"):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                launch._parse_gpu_ids("2,3", visible=visible)

    def test_baseline_children_overlap_with_one_visible_gpu_each(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        plan = self.plan("baseline", "2,3")
        self.setup_root(plan)
        self.parallel_fixtures(plan, plan["jobs"])
        self.assertEqual(self.execute(plan), 0)
        self.assert_overlap_and_isolation(plan["jobs"], ["2", "3"])
        self.assertEqual(self.receipt(plan)["status"], "complete")
        for job in plan["jobs"]:
            launch.validate_result(plan, job)

    def test_gpu_schedule_can_change_without_changing_scientific_recipe(self):
        plan = self.plan("baseline", "0,1")
        self.setup_root(plan)
        other = self.plan("baseline", "2,3")
        self.assertEqual(other["recipe"], plan["recipe"])
        self.assertEqual(other["recipe_sha256"], plan["recipe_sha256"])

    def test_gpu_scheduler_requires_the_child_local_cuda_device(self):
        for device in ("cpu", "cuda:1"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "requires --device cuda"):
                phase_tests.OptimizationPhaseTests.plan(self, "baseline", "--gpus", "2,3", "--device", device)
        plan = phase_tests.OptimizationPhaseTests.plan(self, "baseline", "--gpus", "2,3", "--device", "cuda:0")
        self.assertEqual(plan["recipe"]["common"]["device"], "cuda")
        for job in plan["jobs"]:
            self.assertEqual(job["command"][job["command"].index("--device") + 1], "cuda")

    def test_probe_waits_for_teacher_and_applies_teacher_gate(self):
        plan = self.plan("probe")
        self.setup_root(plan)
        # Make both templates while the teacher checkpoint exists, then remove
        # the teacher's completed output so the actual child must produce it.
        self.fixture(plan, plan["teacher"], keep=True)
        self.fixture(plan, plan["probe"])
        Path(plan["teacher"]["out"], "comparison.json").unlink()
        Path(plan["teacher"]["out"], "best.pt").unlink()
        self.assertEqual(self.execute(plan), 0)
        self.assertLess(self.events(plan["teacher"])[-1]["time"], self.events(plan["probe"])[0]["time"])
        self.assertIn("source_probe", self.receipt(plan)["evidence"])

    def test_failed_teacher_gate_starts_no_probe_even_with_idle_gpus(self):
        plan = self.plan("probe")
        self.setup_root(plan)
        spec_path = self.fixture(plan, plan["teacher"], keep=True)
        self.fixture(plan, plan["probe"])
        spec = launch._read(spec_path)
        spec["report"]["teacher_probe"]["mean_relation_gain"] = -.1
        launch._write(spec_path, spec)
        Path(plan["teacher"]["out"], "comparison.json").unlink()
        Path(plan["teacher"]["out"], "best.pt").unlink()
        self.assertEqual(self.execute(plan), 2)
        self.assertEqual(self.events(plan["probe"]), [])
        self.assertEqual(self.receipt(plan)["status"], "blocked")

    def test_screen_revalidates_baselines_before_parallel_c_and_d(self):
        plan = self.plan("screen")
        self.evidence(plan)
        for job in plan["jobs"][:2]:
            self.fixture(plan, job, keep=True)
        self.parallel_fixtures(plan, plan["jobs"][2:])
        self.assertEqual(self.execute(plan), 0)
        verification = [self.events(job) for job in plan["jobs"][:2]]
        self.assertTrue(all(attempt[0]["action"] == "verify" for attempt in verification))
        self.assertLess(max(attempt[-1]["time"] for attempt in verification),
                        min(self.events(job)[0]["time"] for job in plan["jobs"][2:]))
        self.assert_overlap_and_isolation(plan["jobs"][2:], ["0", "1"])

    def test_screen_probe_gate_still_blocks_all_gpu_launches(self):
        plan = self.plan("screen")
        self.evidence(plan)
        probe = Path(plan["probe"]["out"]) / "comparison.json"
        report = launch._read(probe)
        report["source_recoverability"]["positive_mse"] = .4
        launch._write(probe, report)
        for job in plan["jobs"]:
            self.fixture(plan, job)
        self.assertEqual(self.execute(plan), 2)
        self.assertTrue(all(not self.events(job) for job in plan["jobs"]))

    def test_final_runs_all_four_fresh_students_in_parallel(self):
        plan = self.final_plan()
        self.parallel_fixtures(plan, plan["jobs"])
        self.assertEqual(self.execute(plan), 0)
        self.assert_overlap_and_isolation(plan["jobs"], ["0", "1", "2", "3"])
        self.assertTrue(all(self.events(job)[0]["action"] == "train" for job in plan["jobs"]))
        for job in plan["jobs"]:
            self.assertEqual(launch.validate_result(plan, job)["selection_metric"], "last_epoch")

    def test_final_gate_blocks_all_gpus_when_d_does_not_beat_controls(self):
        plan = self.final_plan()
        self.report(plan, plan["screen"][2], score=.8)
        for job in plan["jobs"]:
            self.fixture(plan, job)
        self.assertEqual(self.execute(plan), 2)
        self.assertTrue(all(not self.events(job) for job in plan["jobs"]))
        self.assertEqual(self.receipt(plan)["status"], "blocked")
        self.assertIn("D must beat", self.receipt(plan)["error"])

    def test_two_gpus_recycle_only_after_the_previous_job_finishes(self):
        plan = self.final_plan("1,3")
        first_pair = [str(Path(job["out"]) / "fixture_started") for job in plan["jobs"][:2]]
        for job in plan["jobs"]:
            self.fixture(plan, job, barrier=first_pair)
        self.assertEqual(self.execute(plan), 0)
        by_gpu = {"1": [], "3": []}
        for job in plan["jobs"]:
            events = self.events(job)
            by_gpu[events[0]["visible"]].append((events[0]["time"], events[-1]["time"]))
        self.assertEqual([len(intervals) for intervals in by_gpu.values()], [2, 2])
        for intervals in by_gpu.values():
            intervals.sort()
            self.assertLess(intervals[0][1], intervals[1][0])

    def test_changed_completed_checkpoint_blocks_before_any_process_starts(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        for job in plan["jobs"]:
            self.fixture(plan, job, keep=True)
        Path(plan["jobs"][0]["out"], "best.pt").write_bytes(b"tampered checkpoint")
        self.assertEqual(self.execute(plan), 2)
        self.assertTrue(all(not self.events(job) for job in plan["jobs"]))
        self.assertEqual(self.receipt(plan)["status"], "blocked")

    def test_resume_verifies_completed_jobs_without_another_training_run(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        for job in plan["jobs"]:
            self.fixture(plan, job)
        self.assertEqual(self.execute(plan), 0)
        self.assertEqual(self.execute(plan, resume=True), 0)
        for job in plan["jobs"]:
            starts = [row["action"] for row in self.events(job) if row["event"] == "start"]
            self.assertEqual(starts, ["train", "verify"])
        self.assertTrue(all(job["reused"] for job in self.receipt(plan)["jobs"]))

    def test_failed_child_stops_sibling_and_never_launches_pending_jobs(self):
        plan = self.final_plan("0,1")
        barrier = [str(Path(job["out"]) / "fixture_started") for job in plan["jobs"][:2]]
        self.fixture(plan, plan["jobs"][0], fail=7, barrier=barrier)
        self.fixture(plan, plan["jobs"][1], delay=10, barrier=barrier)
        for job in plan["jobs"][2:]:
            self.fixture(plan, job)
        processes, popen = [], subprocess.Popen
        def record_process(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(launch.subprocess, "Popen", side_effect=record_process):
            self.assertEqual(self.execute(plan), 1)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual([row["event"] for row in self.events(plan["jobs"][1])], ["start"])
        self.assertTrue(all(not self.events(job) for job in plan["jobs"][2:]))
        self.assertEqual(self.receipt(plan)["status"], "failed")
        self.assertFalse(Path(plan["root"], "optimization_lock.json").exists())

    def test_successful_exit_without_valid_report_blocks_pending_jobs(self):
        plan = self.final_plan("0,1")
        barrier = [str(Path(job["out"]) / "fixture_started") for job in plan["jobs"][:2]]
        self.fixture(plan, plan["jobs"][0], missing_report=True, barrier=barrier)
        self.fixture(plan, plan["jobs"][1], delay=10, barrier=barrier)
        for job in plan["jobs"][2:]:
            self.fixture(plan, job)
        self.assertEqual(self.execute(plan), 2)
        self.assertEqual(self.receipt(plan)["status"], "blocked")
        self.assertTrue(all(not self.events(job) for job in plan["jobs"][2:]))
        self.assertEqual([row["event"] for row in self.events(plan["jobs"][1])], ["start"])

    def test_keyboard_interrupt_reaps_children_and_releases_phase_lock(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        for job in plan["jobs"]:
            self.fixture(plan, job, delay=10)
        processes, popen, sleep = [], subprocess.Popen, time.sleep
        def record_process(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        interrupted = False
        def interrupt_after_start(seconds):
            nonlocal interrupted
            if not interrupted and all(self.events(job) for job in plan["jobs"]):
                interrupted = True
                raise KeyboardInterrupt()
            sleep(min(seconds, .02))
        with patch.object(launch.subprocess, "Popen", side_effect=record_process), \
                patch.object(launch.time, "sleep", side_effect=interrupt_after_start):
            self.assertEqual(self.execute(plan), 1)
        self.assertTrue(interrupted)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(self.receipt(plan)["status"], "interrupted")
        self.assertFalse(Path(plan["root"], "optimization_lock.json").exists())

    def test_sigterm_handler_cleans_children_and_restores_the_previous_handler(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        for job in plan["jobs"]:
            self.fixture(plan, job, delay=10)
        processes, popen, sleep = [], subprocess.Popen, time.sleep
        previous = signal.getsignal(signal.SIGTERM)
        def record_process(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        sent = False
        def send_term_after_start(seconds):
            nonlocal sent
            if not sent and all(self.events(job) for job in plan["jobs"]):
                sent = True
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler))
                self.assertIsNot(handler, previous)
                # Exercise the installed handler without sending a real signal
                # to pytest, which is not portable on Windows.
                handler(signal.SIGTERM, None)
            sleep(min(seconds, .02))
        with patch.object(launch.subprocess, "Popen", side_effect=record_process), \
                patch.object(launch.time, "sleep", side_effect=send_term_after_start):
            self.assertEqual(self.execute(plan), 1)
        self.assertTrue(sent)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(self.receipt(plan)["status"], "interrupted")
        self.assertFalse(Path(plan["root"], "optimization_lock.json").exists())

    def assert_control_sigterm_cleanup(self, plan, command_key):
        marker = self.work / (command_key + "_started")
        command = ("from pathlib import Path; import sys,time; "
                   "Path(sys.argv[1]).write_text('started', encoding='utf-8'); time.sleep(10)")
        plan[command_key] = [sys.executable, "-u", "-c", command, str(marker)]
        processes, popen, sleep = [], subprocess.Popen, time.sleep
        previous = signal.getsignal(signal.SIGTERM)
        def record_process(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        sent = False
        def send_term_during_control(seconds):
            nonlocal sent
            if not sent and marker.is_file():
                sent = True
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler))
                handler(signal.SIGTERM, None)
            sleep(min(seconds, .02))
        with patch.object(launch.subprocess, "Popen", side_effect=record_process), \
                patch.object(launch.time, "sleep", side_effect=send_term_during_control):
            self.assertEqual(self.execute(plan), 1)
        self.assertTrue(sent)
        self.assertTrue(processes)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertNotEqual(processes[-1].returncode, 0)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(self.receipt(plan)["status"], "interrupted")
        self.assertFalse(Path(plan["root"], "optimization_lock.json").exists())
        return processes

    def test_sigterm_during_shared_data_preparation_reaps_control_process(self):
        plan = self.plan("baseline")
        self.setup_root(plan)
        Path(plan["manifest"]).unlink()
        Path(plan["root"], "data_manifest.json").unlink()
        processes = self.assert_control_sigterm_cleanup(plan, "prepare_command")
        self.assertEqual(len(processes), 1)
        self.assertEqual(self.receipt(plan)["jobs"], [])
        self.assertFalse(Path(plan["root"], "data_manifest.json").exists())

    def test_sigterm_during_final_summary_reaps_control_process(self):
        plan = self.final_plan()
        self.parallel_fixtures(plan, plan["jobs"])
        processes = self.assert_control_sigterm_cleanup(plan, "summary_command")
        self.assertEqual(len(processes), 5)
        self.assertTrue(all(process.returncode == 0 for process in processes[:-1]))
        self.assertTrue(all(job["status"] == "completed" for job in self.receipt(plan)["jobs"]))


if __name__ == "__main__":
    unittest.main()
