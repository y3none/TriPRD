"""Real lightweight children verify queue isolation, recovery and failure cleanup."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_triprd_task_queue as queue


ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import argparse, hashlib, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--prepare-only', action='store_true')
p.add_argument('--fixture-settings', type=Path, required=True)
a, _ = p.parse_known_args()
settings = json.loads(a.fixture_settings.read_text())
a.out.mkdir(parents=True, exist_ok=True)
if a.prepare_only:
    if settings.get('prepare_fail'):
        raise SystemExit(9)
    (a.out/'split_manifest.json').write_text(json.dumps({
        'vdt_train':['v1'], 'vdt_val':['v2'], 'cod_train':['c1'], 'cod_val':['c2']}))
    (Path(settings['events']).parent/'prepare_environment.json').write_text(json.dumps({
        'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))
    raise SystemExit(0)
job = json.loads((a.out/'launch.json').read_text(encoding='utf-8'))
name = job['name']
def event(kind):
    with Path(settings['events']).open('a') as stream:
        stream.write(json.dumps({'name':name, 'event':kind, 'time':time.time(),
            'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid':os.getpid()})+'\n')
event('start')
mode = settings.get(name, 'ok')
time.sleep(.06 if mode == 'fail' else .8 if mode == 'slow' else .12)
if mode == 'fail':
    event('failed')
    raise SystemExit(7)
if mode == 'missing':
    event('end')
    raise SystemExit(0)
config = job['expected_config'].copy()
if mode == 'mismatch':
    config['batch_size'] += 1
checkpoint = ('fixture only '+name).encode()
(a.out/'best.pt').write_bytes(checkpoint)
report = {'method':'triprd_task_v4', 'status':'complete', 'stage':config['stage'],
    'group':config['group'], 'teacher_mode':config.get('teacher_mode', 'multimodal'),
    'config':config, 'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),
    'completed_epochs':config['teacher_epochs'] if config['stage']=='teacher' else config['epochs'],
    'best_epoch':config['epochs'], 'test_status':'skipped', 'test':{}}
(a.out/'comparison.json').write_text(json.dumps(report))
event('end')
'''


def option(command, flag):
    return command[command.index(flag) + 1]


class TaskQueueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix=".task-queue-", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name).resolve()
        self.root = self.work / "output"
        self.base = ["--output-root", str(self.root)]

    def plan(self, *options):
        return queue.build_plan(queue.parse_args(self.base + list(options)))

    def tiny(self, *options):
        return self.plan("--model-variant", "tiny", "--no-pretrained", "--device", "cpu",
                         "--epochs", "2", "--teacher-epochs", "2", "--image-size", "32",
                         "--base-channels", "8", "--num-workers", "0", "--max-samples", "4", *options)

    def fixture(self, *, phase="baseline", gpus="2,7", **modes):
        plan = self.tiny("--phase", phase, "--gpus", gpus)
        child = self.work / "child.py"
        child.write_text(CHILD, encoding="utf-8")
        settings = self.work / "settings.json"
        settings.write_text(json.dumps({"events": str(self.work / "events.jsonl"), **modes}), encoding="utf-8")
        for command in [plan["prepare_command"]] + [job["command"] for job in plan["jobs"]]:
            command[:3] = [sys.executable, "-u", str(child)]
            command.extend(("--fixture-settings", str(settings)))
        # The lightweight fixture does no CUDA work, but observes the exact
        # physical-device visibility passed to production CUDA children.
        plan["device"] = "cuda"
        plan["plan_sha256"] = queue._digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        return plan

    def run_plan(self, plan, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = queue.execute_plan(plan, poll_seconds=.01, **kwargs)
        state = json.loads((self.root / "_queue" / plan["phase"] / "status.json").read_text())
        return code, state

    def events(self):
        return [json.loads(line) for line in (self.work / "events.jsonl").read_text().splitlines()]

    def test_dry_run_has_no_writes_or_missing_local_weight_requirement(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = queue.main(self.base + ["--dino-repo", "missing-repo", "--dino-weights", "missing.pt"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["mode"], "dry-run")
        self.assertFalse(self.root.exists())

    def test_four_baselines_are_fixed_and_share_one_cpu_preparation(self):
        plan = self.tiny("--train-backbone-blocks", "2", "--no-detail-branch")
        self.assertEqual([job["name"] for job in plan["jobs"]], [row[0] for row in queue.BASELINES])
        self.assertEqual([(job["expected_config"]["train_backbone_blocks"], job["expected_config"]["detail_branch"])
                          for job in plan["jobs"]], [(0, False), (4, False), (0, True), (4, True)])
        self.assertEqual(option(plan["prepare_command"], "--device"), "cpu")
        self.assertIn("--prepare-only", plan["prepare_command"])
        for job in plan["jobs"]:
            self.assertEqual(job["expected_config"]["group"], "T0")
            self.assertEqual(option(job["command"], "--split-manifest"), plan["split_manifest"])
            flags = [token for token in job["command"] if token.startswith("--")]
            self.assertEqual(len(flags), len(set(flags)))
        self.assertFalse(self.root.exists())

    def test_teacher_screen_and_explicit_final_plans(self):
        teachers = self.tiny("--phase", "teachers", "--train-backbone-blocks", "0", "--no-detail-branch")
        self.assertEqual([job["name"] for job in teachers["jobs"]], ["rgb", "multimodal"])
        self.assertTrue(all(job["expected_config"]["train_backbone_blocks"] == 0 for job in teachers["jobs"]))
        screen = self.tiny("--phase", "screen")
        self.assertEqual([job["name"] for job in screen["jobs"]], [f"T{i}" for i in range(5)])
        for job in screen["jobs"]:
            self.assertNotIn("--test-after-training", job["command"])
            self.assertEqual(job["expected_config"]["cod_train_mode"], "split")
        self.assertTrue(screen["jobs"][2]["expected_config"]["teacher_checkpoint"].endswith("teachers\\rgb\\best.pt")
                        if os.name == "nt" else screen["jobs"][2]["expected_config"]["teacher_checkpoint"].endswith("teachers/rgb/best.pt"))
        self.assertEqual(screen["jobs"][3]["expected_config"]["teacher_checkpoint"],
                         screen["jobs"][4]["expected_config"]["teacher_checkpoint"])
        final = self.plan("--phase", "final", "--dino-repo", "missing", "--dino-weights", "missing.pt")
        for job in final["jobs"]:
            self.assertEqual(job["expected_config"]["cod_train_mode"], "full")
            self.assertIn("--test-after-training", job["command"])
        self.assertEqual(option(final["prepare_command"], "--cod-train-mode"), "split")
        with self.assertRaises(ValueError):
            self.tiny("--phase", "final")

    def test_managed_flags_and_abbreviations_are_not_forwarded(self):
        for flags in (("--stage", "teacher"), ("--sta", "teacher"), ("--group", "T4"),
                      ("--out", "elsewhere"), ("--cod-train-mode", "full"), ("--test-after-training",),
                      ("--teacher-checkpoint", "x"), ("--gpus", "0,0"), ("--gpus", "0,00"), ("--phase", "all")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                queue.parse_args(self.base + list(flags))

    def test_existing_gpu_visibility_cannot_be_broadened(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,7"}):
            plan = self.plan("--gpus", "2,7", "--dino-repo", "repo", "--dino-weights", "weights.pt")
            self.assertEqual(plan["gpus"], ["2", "7"])
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                self.plan("--gpus", "0,1", "--dino-repo", "repo", "--dino-weights", "weights.pt")

    def test_real_children_rotate_free_gpu_slots_and_finish(self):
        code, state = self.run_plan(self.fixture())
        self.assertEqual(code, 0)
        self.assertEqual(state["status"], "complete")
        self.assertTrue(all(row["status"] == "complete" for row in state["jobs"].values()))
        busy = {}
        for event in self.events():
            if event["event"] == "start":
                self.assertNotIn(event["gpu"], busy)
                busy[event["gpu"]] = event["name"]
            else:
                self.assertEqual(busy.pop(event["gpu"]), event["name"])
        self.assertFalse(busy)
        self.assertEqual({event["gpu"] for event in self.events()}, {"2", "7"})
        self.assertEqual(json.loads((self.work / "prepare_environment.json").read_text())["gpu"], "")
        self.assertFalse((self.root / "task_queue.lock").exists())
        self.assertTrue(all((Path(job["output_dir"]) / "run.log").exists() for job in self.fixture()["jobs"]))

    def test_failure_terminates_active_children_and_never_starts_remaining_jobs(self):
        plan = self.fixture(R0_frozen_no_detail="fail", R1_blocks4_no_detail="slow")
        code, state = self.run_plan(plan)
        self.assertEqual(code, 1)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["jobs"]["R0_frozen_no_detail"]["status"], "failed")
        self.assertEqual(state["jobs"]["R1_blocks4_no_detail"]["status"], "terminated")
        self.assertTrue(all(state["jobs"][name]["status"] == "not_started" for name in
                            ("R2_frozen_detail", "R3_blocks4_detail")))
        self.assertFalse(any(e["name"] in {"R2_frozen_detail", "R3_blocks4_detail"} for e in self.events()))
        self.assertFalse((self.root / "task_queue.lock").exists())

    def test_child_success_code_without_valid_report_is_failure(self):
        for mode in ("missing", "mismatch"):
            with self.subTest(mode=mode):
                # Use separate phase roots without deleting shared test state.
                self.root = self.work / mode
                self.base = ["--output-root", str(self.root)]
                plan = self.fixture(R0_frozen_no_detail=mode)
                code, state = self.run_plan(plan)
                self.assertEqual(code, 1)
                self.assertEqual(state["jobs"]["R0_frozen_no_detail"]["status"], "failed")

    def test_prepare_failure_starts_no_jobs(self):
        code, state = self.run_plan(self.fixture(prepare_fail=True))
        self.assertEqual(code, 1)
        self.assertTrue(all(row["status"] == "not_started" for row in state["jobs"].values()))
        self.assertFalse((self.work / "events.jsonl").exists())

    def test_resume_requires_identical_plan_and_verified_manifest(self):
        plan = self.fixture()
        self.assertEqual(self.run_plan(plan)[0], 0)
        with self.assertRaisesRegex(ValueError, "--resume"):
            self.run_plan(plan)
        changed = json.loads(json.dumps(plan))
        changed["jobs"][0]["expected_config"]["epochs"] += 1
        changed["plan_sha256"] = queue._digest({k: v for k, v in changed.items() if k != "plan_sha256"})
        with self.assertRaisesRegex(ValueError, "identical"):
            self.run_plan(changed, resume=True)
        self.assertEqual(self.run_plan(plan, resume=True)[0], 0)
        Path(plan["split_manifest"]).write_text("{}", encoding="utf-8")
        code, state = self.run_plan(plan, resume=True)
        self.assertEqual(code, 1)
        self.assertIn("manifest", state["error"])

    def test_shared_manifest_is_prepared_once_across_phases(self):
        baseline = self.fixture()
        self.assertEqual(self.run_plan(baseline)[0], 0)
        teacher = self.fixture(phase="teachers")
        with patch.object(queue, "_spawn", wraps=queue._spawn) as spawn:
            self.assertEqual(self.run_plan(teacher)[0], 0)
        self.assertEqual(spawn.call_count, 2)
        for call in spawn.call_args_list:
            self.assertNotIn("--prepare-only", call.args[0])

    def test_missing_teachers_fail_before_creating_screen_outputs(self):
        with self.assertRaisesRegex(ValueError, "teachers phase first"):
            queue.execute_plan(self.tiny("--phase", "screen"))
        self.assertFalse(self.root.exists())

    def test_keyboard_interrupt_reaps_started_children(self):
        plan = self.fixture(R0_frozen_no_detail="slow", R1_blocks4_no_detail="slow")
        with patch.object(queue.time, "sleep", side_effect=KeyboardInterrupt):
            code, state = self.run_plan(plan)
        self.assertEqual(code, 130)
        self.assertEqual(state["status"], "interrupted")
        self.assertTrue(all(row["status"] in {"terminated", "not_started"} for row in state["jobs"].values()))
        self.assertFalse((self.root / "task_queue.lock").exists())

    def test_shell_exposes_explicit_phases_and_safe_arrays(self):
        script = (ROOT / "tools/run_triprd_task.sh").read_text(encoding="utf-8")
        self.assertIn('phase="${PHASE:-baseline}"', script)
        self.assertIn('${DRY_RUN:-0}', script)
        for name in ("PYTHON", "GPUS", "DINO_REPO", "DINO_WEIGHTS", "OUTPUT_ROOT", "VDT_ROOT", "COD_ROOT",
                     "BATCH_SIZE", "NUM_WORKERS", "EPOCHS", "TEACHER_EPOCHS", "SEED", "SPLIT_SEED"):
            self.assertIn("${" + name, script)
        self.assertIn('"${extra[@]}" "$@"', script)

    def test_atomic_json_write_retries_transient_windows_sharing_denial(self):
        path = self.work / "status.json"
        original = Path.replace
        attempts = []
        def transient(source, destination):
            attempts.append(source)
            if len(attempts) < 3:
                raise PermissionError("fixture Windows sharing denial")
            return original(source, destination)
        with patch.object(Path, "replace", transient), patch.object(queue.time, "sleep") as sleep:
            queue._write(path, {"status": "complete"})
        self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_atomic_json_write_has_bounded_failure_without_truncating_existing_file(self):
        path = self.work / "status.json"
        path.write_text('{"status":"previous"}')
        with (patch.object(Path, "replace", side_effect=PermissionError("persistent denial")) as replace,
              patch.object(queue.time, "sleep") as sleep):
            with self.assertRaises(PermissionError):
                queue._write(path, {"status": "new"})
        self.assertEqual(replace.call_count, 8)
        self.assertEqual(sleep.call_count, 7)
        self.assertEqual(json.loads(path.read_text()), {"status": "previous"})
        self.assertEqual(json.loads(path.with_suffix(".json.tmp").read_text()), {"status": "new"})


if __name__ == "__main__":
    unittest.main()
