"""Scheduling tests use real tiny child processes, without importing CUDA."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_triprd_staged_queue as queue

ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import hashlib,json,os,sys
from pathlib import Path
out=Path(sys.argv[1])
launch=json.loads((out/'launch.json').read_text(encoding='utf-8'))
config=launch['expected_config']
payload=('fixture-'+config['group']).encode()
(out/'best.pt').write_bytes(payload)
protocol={'test_fixture':True}
digest=hashlib.sha256(json.dumps(protocol,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
report={'method':'triprd_staged_v5','status':'complete','config':config,'protocol':protocol,'contract_sha256':digest,
 'completed_epochs':config['source_epochs'] if config['stage']=='source' else config['epochs'],
 'checkpoint_sha256':hashlib.sha256(payload).hexdigest(),'best_epoch':1,'test':{},'test_status':'skipped'}
(out/'comparison.json').write_text(json.dumps(report),encoding='utf-8')
print('fixture-child-resume-count='+str(sys.argv.count('--resume')),flush=True)
'''


class StagedQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".staged-queue-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.manifest = self.work / "v4/_data/split_manifest.json"
        queue.old._write(self.manifest, {"split_seed": 20261012, "cod_train": ["COD10K/c1", "CAMO/c2"],
                                        "cod_val": ["COD10K/c3", "CAMO/c4"], "vdt_train": ["VDT/s1"], "vdt_val": ["VDT/s2"]})
        self.args = queue.parse_args(["--phase", "screen", "--output-root", str(self.work / "v5"),
                                     "--teacher-root", str(self.work / "v4/teachers"), "--split-manifest", str(self.manifest),
                                     "--device", "cpu", "--model-variant", "tiny", "--no-pretrained", "--max-samples", "4"])
        # The files are developed in parallel; only this queue's own source is
        # needed to check immutable-plan behavior in these scheduler fixtures.
        self.source_patch = patch.object(queue, "SOURCE_FILES", ("tools/run_triprd_staged_queue.py",))
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def test_source_and_cod_commands_separate_data_and_dependencies(self):
        source, screen = queue.build_plan(self.args, "source"), queue.build_plan(self.args, "screen")
        self.assertEqual([job["name"] for job in source["jobs"]], ["S1", "S2", "S3"])
        self.assertEqual([job["name"] for job in screen["jobs"]], ["S0", "S1", "S2", "S3"])
        for job in source["jobs"]:
            self.assertEqual(job["expected_config"]["stage"], "source")
            self.assertNotIn("--source-checkpoint", job["command"])
            self.assertNotIn("--test-after-training", job["command"])
        for job in screen["jobs"]:
            self.assertEqual(job["expected_config"]["stage"], "cod")
            self.assertNotIn("--teacher-checkpoint", job["command"])
            self.assertNotIn("--test-after-training", job["command"])
            self.assertEqual("--source-checkpoint" in job["command"], job["name"] != "S0")

    def test_reusing_v4_root_is_rejected(self):
        self.args.output_root = self.work / "v4"
        with self.assertRaisesRegex(ValueError, "new v5"):
            queue.build_plan(self.args)

    def test_overlapping_members_and_seed_drift_are_rejected(self):
        self.args.split_seed += 1
        with self.assertRaisesRegex(ValueError, "seed"):
            queue.build_plan(self.args)
        self.args.split_seed -= 1
        value = queue.old._read(self.manifest)
        value["cod_val"].append(value["cod_train"][0])
        queue.old._write(self.manifest, value)
        with self.assertRaisesRegex(ValueError, "disjointness"):
            queue.build_plan(self.args)

    def test_final_requires_real_unrestricted_model_and_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "full4040"):
            queue.build_plan(self.args, "final")
        self.args.model_variant, self.args.max_samples, self.args.pretrained = "dinov3_vitb16", 0, True
        final = queue.build_plan(self.args, "final")
        self.assertTrue(all("--test-after-training" in job["command"] for job in final["jobs"]))
        self.assertTrue(all(job["expected_config"]["cod_train_mode"] == "full" for job in final["jobs"]))

    def fixture_plan(self):
        plan = queue.build_plan(self.args, "screen")
        child = self.work / "child.py"
        child.write_text(CHILD, encoding="utf-8")
        plan["dependencies"] = {}
        for job in plan["jobs"]:
            job["command"] = [sys.executable, "-u", str(child), job["output_dir"]]
            directory = Path(job["output_dir"])
            directory.mkdir(parents=True)
            (directory / "run.log").write_text("prior log\n", encoding="utf-8")
        plan["plan_sha256"] = queue.old._digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        return plan

    def test_real_children_append_logs_resume_once_and_completed_phase_reuses(self):
        plan = self.fixture_plan()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(queue.execute_plan(plan, resume=True, poll_seconds=.01), 0)
        status = queue.old._read(Path(plan["output_root"]) / "_queue/screen/status.json")
        self.assertEqual(status["status"], "complete")
        self.assertEqual([status["jobs"][f"S{i}"]["gpu"] for i in range(4)], ["2", "3", "0", "1"])
        before = {}
        for job in plan["jobs"]:
            path = Path(job["output_dir"]) / "run.log"
            self.assertEqual(path.read_text(encoding="utf-8"), "prior log\nfixture-child-resume-count=1\n")
            before[path] = path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(queue.old, "_spawn", side_effect=AssertionError("Completed work spawned again")):
            self.assertEqual(queue.execute_plan(plan), 0)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_changed_plan_or_dependency_refuses_before_start(self):
        plan = self.fixture_plan()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(queue.execute_plan(plan, poll_seconds=.01), 0)
        plan["gpus"] = ["0"]
        plan["plan_sha256"] = queue.old._digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        with self.assertRaisesRegex(ValueError, "differs"):
            queue.execute_plan(plan, resume=True)
        plan = queue.build_plan(self.args, "source")
        with self.assertRaisesRegex(ValueError, "Missing completed"):
            queue.execute_plan(plan)

    def test_frozen_manifest_change_refuses_before_child_start(self):
        plan = self.fixture_plan()
        self.manifest.write_text("{}", encoding="utf-8")
        with patch.object(queue.old, "_spawn", side_effect=AssertionError("unexpected child")):
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                queue.execute_plan(plan)

    def test_failed_source_phase_never_starts_cod_or_final(self):
        argv = ["--phase", "screen", "--output-root", str(self.args.output_root), "--teacher-root", str(self.args.teacher_root),
                "--split-manifest", str(self.manifest), "--device", "cpu", "--execute"]
        phases = []
        def execute(plan, **kwargs):
            phases.append(plan["phase"])
            return 1
        with patch.object(queue, "execute_plan", side_effect=execute):
            self.assertEqual(queue.main(argv), 1)
        self.assertEqual(phases, ["source"])
        self.assertFalse((self.args.output_root / "staged_pipeline.lock").exists())


if __name__ == "__main__":
    unittest.main()
