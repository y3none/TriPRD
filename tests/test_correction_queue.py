"""Correction queue contracts and real subprocess dependency tests; no GPU needed."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tools import run_triprd_correction_queue as launcher


ROOT = Path(__file__).resolve().parents[1]
FAKE_CHILD = r'''
import argparse, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--stage')
p.add_argument('--prepare-only', action='store_true')
p.add_argument('--fixture-settings', type=Path, required=True)
a, _ = p.parse_known_args()
s = json.loads(a.fixture_settings.read_text())
a.out.mkdir(parents=True, exist_ok=True)
if a.prepare_only:
    (a.out/'split_manifest.json').write_text(json.dumps({
        'vdt_train':['s1','s2'], 'vdt_val':['s3'], 'cod_train':['c1','c2'], 'cod_val':['c3']}))
    raise SystemExit(s.get('prepare_returncode', 0))
job = json.loads((a.out/'launch.json').read_text(encoding='utf-8'))
name = job['name']
def event(kind):
    with Path(s['events']).open('a') as f:
        f.write(json.dumps({'name':name,'event':kind,'time':time.time(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\n')
event('start')
time.sleep(.2 if a.stage == 'teacher' else .03)
mode = s.get(name, 'ok')
if mode == 'fail':
    event('failed')
    raise SystemExit(7)
if mode == 'missing':
    event('end')
    raise SystemExit(0)
config = job['expected_config'].copy()
if mode == 'mismatch':
    config['epochs'] += 1
probe = {'status':'complete', 'contract_sha256':'a'*64,
         'mean_relation_gain': -.1 if mode == 'negative' else float('nan') if mode == 'nan' else .1,
         'positive_coverage': .5}
report = {'status':'complete','config':config,'contract_sha256':'a'*64}
if a.stage == 'teacher':
    report['teacher_probe'] = probe
    (a.out/'teacher_probe.json').write_text(json.dumps(probe))
(a.out/'best.pt').write_bytes(b'fixture checkpoint, not trained weights')
(a.out/'comparison.json').write_text(json.dumps(report))
event('end')
'''


def option(command, flag):
    return command[command.index(flag) + 1]


class CorrectionQueueTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory(prefix=".correction-queue-test-", dir=ROOT)
        self.addCleanup(work.cleanup)
        self.work = Path(work.name).resolve()
        self.argv = ["--output-root", str(self.work / "output")]

    def plan(self, *extra):
        return launcher.build_plan(launcher.parse_args(self.argv + list(extra)))

    def tiny(self, *extra):
        return self.plan("--model-variant", "tiny", "--no-pretrained", "--device", "cpu", "--no-amp",
                         "--epochs", "2", "--teacher-epochs", "2", "--base-channels", "8",
                         "--image-size", "32", "--num-workers", "0", "--max-samples", "8", *extra)

    def fixture(self, **modes):
        plan = self.tiny()
        child = self.work / "fake_child.py"
        child.write_text(FAKE_CHILD, encoding="utf-8")
        settings = self.work / "settings.json"
        settings.write_text(json.dumps({"events": str(self.work / "events.jsonl"), **modes}), encoding="utf-8")
        for command in [plan["prepare_command"]] + [job["command"] for job in plan["jobs"]]:
            command[:3] = [sys.executable, "-u", str(child)]
            command += ["--fixture-settings", str(settings)]
        plan["plan_sha256"] = launcher._digest({key: value for key, value in plan.items()
                                               if key not in ("plan_sha256", "resume", "allow_nonpositive_probe")})
        return plan

    def execute(self, plan):
        with contextlib.redirect_stdout(io.StringIO()):
            result = launcher.execute_plan(plan)
        summary = json.loads((Path(plan["run_dir"]) / "exit_status.json").read_text(encoding="utf-8"))
        return result, summary

    def events(self):
        return [json.loads(line) for line in (self.work / "events.jsonl").read_text().splitlines()]

    def test_formal_defaults_and_six_gpu_mapping(self):
        plan = self.plan("--dino-repo", str(self.work / "dino"), "--dino-weights", str(self.work / "weights.pt"))
        self.assertEqual(plan["gpus"], ["0", "1", "2", "3", "6", "7"])
        self.assertEqual([job["name"] for job in plan["jobs"]], ["teacher"] + [name for _, name in launcher.GROUPS])
        self.assertEqual([job["gpu"] for job in plan["jobs"]], ["7", "0", "1", "2", "3", "6", "7"])
        for job in plan["jobs"]:
            command = job["command"]
            for flag, value in (("--epochs", "100"), ("--teacher-epochs", "100"), ("--image-size", "384"),
                                ("--batch-size", "8"), ("--base-channels", "128"), ("--warmup-epochs", "5"),
                                ("--model-variant", "dinov3_vitb16"), ("--train-seed", "20261012")):
                self.assertEqual(option(command, flag), value)
            self.assertIn("--pretrained", command)
            self.assertIn("--skip-test", command)
            self.assertIn("--amp", command)
            self.assertEqual(option(command, "--split-manifest"), plan["split_manifest"])
            self.assertEqual(job["dependencies"], ["teacher"] if job["group"] in tuple("CDEF") else [])
            flags = [value for value in command if value.startswith("--")]
            self.assertEqual(len(flags), len(set(flags)))
        self.assertEqual(plan["selection_rule"], "macro_S_alpha")
        self.assertFalse(Path(plan["run_dir"]).exists())

    def test_dry_run_does_not_create_outputs_or_require_local_files(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = launcher.main(self.argv + ["--dino-repo", "absent-repo", "--dino-weights", "absent.pt"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stream.getvalue())["mode"], "dry-run")
        self.assertFalse((self.work / "output").exists())

    def test_invalid_protocol_is_rejected(self):
        for flags in ((), ("--model-variant", "tiny"),
                      ("--model-variant", "tiny", "--no-pretrained", "--groups", "A,A"),
                      ("--model-variant", "tiny", "--no-pretrained", "--lr", "nan"),
                      ("--model-variant", "tiny", "--no-pretrained", "--image-size", "33"),
                      ("--model-variant", "tiny", "--no-pretrained", "--seed", "-1")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.plan(*flags)
        with self.assertRaisesRegex(ValueError, "Full COD"):
            self.tiny("--cod-train-mode", "full")
        for flags in (("--gpus", "0,0"), ("--gpus", "-1"), ("--epochs", "0")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(self.argv + list(flags))

    def test_full_training_is_explicit_and_uses_last_epoch(self):
        plan = self.tiny("--cod-train-mode", "full", "--max-samples", "0", "--no-skip-test")
        self.assertEqual(plan["selection_rule"], "last_epoch")
        for job in plan["jobs"]:
            self.assertIn("--no-skip-test", job["command"])
            self.assertEqual(option(job["command"], "--max-samples"), "0")

    def test_real_subprocess_success_and_dependency_order(self):
        plan = self.fixture()
        code, summary = self.execute(plan)
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(all(state["status"] == "complete" for state in summary["jobs"].values()))
        events = self.events()
        end = next(event["time"] for event in events if event["name"] == "teacher" and event["event"] == "end")
        for event in events:
            if event["event"] == "start" and event["name"][0] in "CDEF":
                self.assertGreaterEqual(event["time"], end)
        self.assertEqual({event["gpu"] for event in events}, {"0", "1", "2", "3", "6", "7"})
        self.assertTrue(summary["manifest_sha256"])
        self.assertTrue(all(state["pid"] for state in summary["jobs"].values()))
        self.assertFalse((Path(plan["run_dir"]) / "queue_lock.json").exists())

    def test_failed_teacher_blocks_dependents_but_not_independent_jobs(self):
        plan = self.fixture(teacher="fail")
        code, summary = self.execute(plan)
        self.assertEqual(code, 1)
        self.assertEqual(summary["jobs"]["teacher"]["process_returncode"], 7)
        for group, name in launcher.GROUPS:
            self.assertEqual(summary["jobs"][name]["status"], "complete" if group in "AB" else "blocked")
        self.assertEqual({event["name"] for event in self.events()}, {"teacher", "A_cod_only", "B_rgb_joint"})

    def test_zero_exit_missing_report_is_failure(self):
        plan = self.fixture(teacher="missing")
        code, summary = self.execute(plan)
        self.assertEqual(code, 1)
        self.assertEqual(summary["jobs"]["teacher"]["process_returncode"], 0)
        self.assertEqual(summary["jobs"]["teacher"]["returncode"], 3)

    def test_nonfinite_teacher_is_never_allowed(self):
        plan = self.fixture(teacher="nan")
        plan["allow_nonpositive_probe"] = True
        code, summary = self.execute(plan)
        self.assertEqual(code, 1)
        self.assertIn("finite", summary["jobs"]["teacher"]["error"])

    def test_nonpositive_probe_requires_explicit_recorded_override(self):
        plan = self.fixture(teacher="negative")
        code, _ = self.execute(plan)
        self.assertEqual(code, 1)
        plan.update(resume=True, allow_nonpositive_probe=True)
        code, summary = self.execute(plan)
        self.assertEqual(code, 0)
        self.assertTrue(summary["allow_nonpositive_probe"])
        self.assertTrue(summary["jobs"]["teacher"]["warnings"])
        # Reuse is sent through the runner with --resume, never trusted from file presence.
        self.assertIn("--resume", summary["jobs"]["teacher"]["command"])

    def test_student_failure_does_not_cancel_other_students(self):
        plan = self.fixture(C_full_kd="mismatch")
        code, summary = self.execute(plan)
        self.assertEqual(code, 1)
        self.assertIn("epochs", summary["jobs"]["C_full_kd"]["error"])
        for name, state in summary["jobs"].items():
            self.assertEqual(state["status"], "failed" if name == "C_full_kd" else "complete")

    def test_existing_plan_and_lock_are_not_silently_reused(self):
        plan = self.fixture()
        self.execute(plan)
        with self.assertRaisesRegex(ValueError, "already exists"):
            launcher.validate_plan(plan)
        plan["resume"] = True
        launcher.validate_plan(plan)
        altered = {**plan, "plan_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "differs"):
            launcher.validate_plan(altered)
        (Path(plan["run_dir"]) / "queue_lock.json").write_text('{"pid":123}')
        with self.assertRaisesRegex(ValueError, "Queue lock"):
            self.execute(plan)

    def test_prepare_failure_starts_no_training(self):
        plan = self.fixture(prepare_returncode=5)
        with self.assertRaisesRegex(ValueError, "preparation failed"):
            self.execute(plan)
        self.assertFalse((self.work / "events.jsonl").exists())
        summary = json.loads((Path(plan["run_dir"]) / "exit_status.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(all(state["status"] == "blocked" for state in summary["jobs"].values()))


if __name__ == "__main__":
    unittest.main()
