"""Verify expansion preserves training contracts and refuses stale provenance."""
from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import run_triprd_task_queue as queue
from tools import expand_triprd_task_screen as expansion


ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import argparse, hashlib, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a, _ = p.parse_known_args()
if sys.argv.count('--resume') != 1:
    raise SystemExit(11)
launch = json.loads((a.out/'launch.json').read_text(encoding='utf-8'))
config = launch['expected_config']
checkpoint = ('fixture ' + launch['name']).encode()
(a.out/'best.pt').write_bytes(checkpoint)
report = {'method':'triprd_task_v4', 'status':'complete', 'stage':'student',
    'group':config['group'], 'config':config,
    'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),
    'completed_epochs':config['epochs'], 'best_epoch':config['epochs'],
    'test_status':'skipped', 'test':{}}
(a.out/'comparison.json').write_text(json.dumps(report))
print('resumed-on-' + os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
'''


class ScreenExpansionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".task-expand-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        self.output = self.work / "output"
        # Valid tiny files are sufficient: queue provenance verifies their bytes,
        # while the training runner owns checkpoint deserialization and contracts.
        for mode in ("rgb", "multimodal"):
            directory = self.output / "teachers" / mode
            directory.mkdir(parents=True)
            checkpoint = directory / "best.pt"
            checkpoint.write_bytes(("fixture teacher " + mode).encode())
            queue._write(directory / "comparison.json", {
                "method": queue.METHOD, "status": "complete", "stage": "teacher",
                "teacher_mode": mode, "checkpoint_sha256": queue._sha256(checkpoint),
            })
        args = queue.parse_args([
            "--phase", "screen", "--output-root", str(self.output), "--gpus", "2,3",
            "--dino-repo", str(self.work / "dinov3"),
            "--dino-weights", str(self.work / "pretrained.pt"),
        ])
        self.plan = queue.build_plan(args)
        self.manifest = Path(self.plan["split_manifest"])
        queue._write(self.manifest, {
            "vdt_train": ["v1"], "vdt_val": ["v2"],
            "cod_train": ["c1"], "cod_val": ["c2"],
        })
        self.manifest_hash = queue._sha256(self.manifest)
        queue._write(self.manifest.parent / "data_contract.json", {
            "contract": self.plan["data_contract"], "manifest_sha256": self.manifest_hash,
        })
        self.metadata = self.output / "_queue" / "screen"
        self.save_plan()
        queue._write(self.metadata / "status.json", {
            "phase": "screen", "status": "running", "plan_sha256": self.plan["plan_sha256"],
            "manifest_sha256": self.manifest_hash,
            "jobs": {
                "T0": {"status": "running", "pid": 857361, "gpu": "2"},
                "T1": {"status": "running", "pid": 857362, "gpu": "3"},
                **{f"T{i}": {"status": "pending"} for i in range(2, 5)},
            },
        })

    def save_plan(self, *, rehash=False):
        if rehash:
            self.plan["plan_sha256"] = queue._digest({
                key: value for key, value in self.plan.items() if key != "plan_sha256"
            })
        queue._write(self.metadata / "plan.json", self.plan)

    def test_saved_plan_validates_without_changing_existing_metadata(self):
        before = {path: path.read_bytes() for path in self.output.rglob("*.json")}
        self.assertEqual(expansion.validate_saved_plan(self.output), self.plan)
        self.assertEqual(before, {path: path.read_bytes() for path in self.output.rglob("*.json")})

    def test_expanded_plan_changes_scheduling_only_and_keeps_original_immutable(self):
        original = copy.deepcopy(self.plan)
        expanded = expansion.make_expansion_plan(self.plan, ["2", "3", "0", "1", "6"])
        self.assertEqual(self.plan, original)
        self.assertEqual(expanded["phase"], "screen_expanded")
        self.assertEqual(expanded["gpus"], ["2", "3", "0", "1", "6"])
        self.assertEqual(expanded["original_plan_sha256"], self.plan["plan_sha256"])
        self.assertEqual(expanded["jobs"], self.plan["jobs"])
        self.assertEqual(expanded["dependencies"], self.plan["dependencies"])
        self.assertEqual(expanded["source_sha256"], self.plan["source_sha256"])
        self.assertEqual(expanded["plan_sha256"], queue._digest({
            key: value for key, value in expanded.items() if key != "plan_sha256"
        }))
        for job in expanded["jobs"]:
            self.assertNotIn("--resume", job["command"])
            self.assertIn(str(self.output / "screen"), job["output_dir"])
        # Independence matters: subsequently persisted launch metadata must not
        # be able to mutate the old plan through shared nested containers.
        expanded["jobs"][0]["command"].append("--fixture-only")
        self.assertEqual(self.plan, original)

    def test_saved_digest_change_is_rejected(self):
        self.plan["gpus"] = ["0", "1"]
        self.save_plan()
        with self.assertRaises(ValueError):
            expansion.validate_saved_plan(self.output)

    def test_current_source_change_is_rejected(self):
        self.plan["source_sha256"]["tools/run_triprd_task.py"] = "0" * 64
        self.save_plan(rehash=True)
        with self.assertRaises(ValueError):
            expansion.validate_saved_plan(self.output)

    def test_omitting_a_hashed_source_is_rejected(self):
        del self.plan["source_sha256"]["tools/run_triprd_task.py"]
        self.save_plan(rehash=True)
        with self.assertRaises(ValueError):
            expansion.validate_saved_plan(self.output)

    def test_changed_shared_manifest_is_rejected(self):
        queue._write(self.manifest, {"cod_train": ["changed"]})
        with self.assertRaises(ValueError):
            expansion.validate_saved_plan(self.output)

    def test_changed_teacher_bytes_or_report_are_rejected(self):
        for changed in ("checkpoint", "report"):
            with self.subTest(changed=changed):
                checkpoint = self.output / "teachers" / "rgb" / "best.pt"
                report = checkpoint.parent / "comparison.json"
                path = checkpoint if changed == "checkpoint" else report
                prior = path.read_bytes()
                path.write_bytes(b"changed" if changed == "checkpoint" else b'{"status":"running"}')
                try:
                    with self.assertRaises(ValueError):
                        expansion.validate_saved_plan(self.output)
                finally:
                    path.write_bytes(prior)

    def test_command_configuration_disagreement_is_rejected(self):
        command = self.plan["jobs"][0]["command"]
        command[command.index("--epochs") + 1] = "999"
        self.save_plan(rehash=True)
        with self.assertRaises(ValueError):
            expansion.validate_saved_plan(self.output)

    def test_redirected_job_output_is_rejected(self):
        job = self.plan["jobs"][0]
        altered = str(self.work / "unrelated")
        job["output_dir"] = altered
        job["expected_config"]["out"] = altered
        job["command"][job["command"].index("--out") + 1] = altered
        self.save_plan(rehash=True)
        with self.assertRaises(ValueError):
            expansion.validate_saved_plan(self.output)

    def test_five_real_children_resume_once_append_logs_and_keep_old_metadata(self):
        child = self.work / "child.py"
        child.write_text(CHILD, encoding="utf-8")
        original = copy.deepcopy(self.plan)
        for job in original["jobs"]:
            job["command"][:3] = [sys.executable, "-u", str(child)]
            directory = Path(job["output_dir"])
            directory.mkdir(parents=True)
            (directory / "run.log").write_text("previous training progress\n", encoding="utf-8")
        original["plan_sha256"] = queue._digest({
            key: value for key, value in original.items() if key != "plan_sha256"
        })
        expanded = expansion.make_expansion_plan(original, ["2", "3", "0", "1", "6"])
        before = {path: path.read_bytes() for path in self.metadata.glob("*.json")}
        expanded_metadata = self.output / "_queue" / "screen_expanded"
        queue._write(expanded_metadata / "plan.json", expanded)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = queue.execute_plan(expanded, resume=True, poll_seconds=.01)
        self.assertEqual(code, 0)
        state = queue._read(expanded_metadata / "status.json")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(before, {path: path.read_bytes() for path in self.metadata.glob("*.json")})
        for job, gpu in zip(expanded["jobs"], expanded["gpus"]):
            directory = Path(job["output_dir"])
            log = (directory / "run.log").read_text(encoding="utf-8")
            self.assertTrue(log.startswith("previous training progress\n"))
            self.assertIn("resumed-on-" + gpu, log)
            launch = queue._read(directory / "launch.json")
            self.assertEqual(launch["executed_command"].count("--resume"), 1)
            self.assertEqual(launch["expected_config"], job["expected_config"])
            self.assertEqual(state["jobs"][job["name"]]["status"], "complete")

    def process(self, *, pid=900001, job=None):
        argv = ([sys.executable, "-u", str(ROOT / "tools/run_triprd_task_queue.py"),
                 "--phase", "screen", "--output-root", str(self.output)]
                if job is None else job["command"])
        return {"pid": pid, "ppid": 123 if job is None else 900001,
                "pgrp": pid, "uid": 1000, "start_ticks": 1234567,
                "state": "S", "cwd": str(ROOT), "argv": list(argv)}

    def test_process_owner_matches_exact_user_workspace_queue_and_job(self):
        with patch.object(expansion.os, "getuid", return_value=1000, create=True):
            expansion.assert_process_owner(self.process(), self.output)
            job = self.plan["jobs"][0]
            expansion.assert_process_owner(self.process(pid=857361, job=job), self.output, job=job)
            for key, value in (("uid", 1001), ("cwd", str(self.work)),
                               ("argv", ["python", str(self.work / "run_triprd_task_queue.py"),
                                         "--phase", "screen", "--output-root", str(self.output)]),
                               ("argv", ["python", str(ROOT / "tools/run_triprd_task_queue.py"),
                                         "--phase", "teachers", "--output-root", str(self.output)])):
                with self.subTest(key=key, value=value):
                    process = self.process()
                    process[key] = value
                    with self.assertRaises(ValueError):
                        expansion.assert_process_owner(process, self.output)
            process = self.process(pid=857361, job=job)
            process["pgrp"] += 1
            with self.assertRaises(ValueError):
                expansion.assert_process_owner(process, self.output, job=job)

    def test_wrong_queue_pid_is_never_signalled(self):
        queue._write(self.output / "task_queue.lock", {"pid": 900001})
        process = self.process()
        process["argv"] = ["python", "some_other_training.py"]
        with (patch.object(expansion.os, "getuid", return_value=1000, create=True),
              patch.object(expansion, "read_process", return_value=process),
              patch.object(expansion.os, "kill") as kill,
              patch.object(expansion.signal, "pidfd_send_signal", create=True) as pidfd_signal):
            with self.assertRaises(ValueError):
                expansion.safe_stop_queue(self.output, self.plan, {})
        kill.assert_not_called()
        pidfd_signal.assert_not_called()
        self.assertTrue((self.output / "task_queue.lock").exists())

    def test_recycled_queue_pid_is_never_signalled(self):
        queue._write(self.output / "task_queue.lock", {"pid": 900001})
        calls = []
        def read(pid):
            if pid == 900001:
                calls.append(pid)
                process = self.process()
                if len(calls) > 1:
                    process["start_ticks"] += 1
                return process
            return self.process(pid=pid, job=self.plan["jobs"][pid - 857361])
        with (patch.object(expansion.os, "getuid", return_value=1000, create=True),
              patch.object(expansion, "read_process", side_effect=read),
              patch.object(expansion.os, "kill") as kill,
              patch.object(expansion.signal, "pidfd_send_signal", create=True) as pidfd_signal):
            with self.assertRaisesRegex(ValueError, "recycled"):
                expansion.safe_stop_queue(self.output, self.plan, {})
        kill.assert_not_called()
        pidfd_signal.assert_not_called()

    def test_training_child_with_other_parent_aborts_before_any_signal(self):
        queue._write(self.output / "task_queue.lock", {"pid": 900001})
        def read(pid):
            if pid == 900001:
                return self.process()
            child = self.process(pid=pid, job=self.plan["jobs"][pid - 857361])
            child["ppid"] = 900009
            return child
        with (patch.object(expansion.os, "getuid", return_value=1000, create=True),
              patch.object(expansion, "read_process", side_effect=read),
              patch.object(expansion.os, "kill") as kill,
              patch.object(expansion.signal, "pidfd_send_signal", create=True) as pidfd_signal):
            with self.assertRaisesRegex(ValueError, "not a child"):
                expansion.safe_stop_queue(self.output, self.plan, {})
        kill.assert_not_called()
        pidfd_signal.assert_not_called()

    def make_run_locks(self):
        paths = []
        for i in range(2):
            path = self.output / "screen" / f"T{i}" / "task_run.lock"
            queue._write(path, {"pid": 857361 + i})
            paths.append(path)
        return paths

    def test_live_child_prevents_all_lock_recovery(self):
        paths = self.make_run_locks()
        with (patch.object(expansion, "active_job_processes", return_value=[]),
              patch.object(expansion, "read_process", return_value=self.process(pid=857361)),
              patch.object(expansion, "process_groups", return_value=[])):
            with self.assertRaisesRegex(RuntimeError, "live process"):
                expansion.recover_stale_locks(self.output, self.plan, {})
        self.assertTrue(all(path.exists() for path in paths))

    def test_surviving_dataloader_group_prevents_lock_recovery(self):
        paths = self.make_run_locks()
        with (patch.object(expansion, "active_job_processes", return_value=[]),
              patch.object(expansion, "read_process", return_value=None),
              patch.object(expansion, "process_groups", return_value=[999001])):
            with self.assertRaisesRegex(RuntimeError, "still alive"):
                expansion.recover_stale_locks(self.output, self.plan, {})
        self.assertTrue(all(path.exists() for path in paths))

    def test_mismatched_run_lock_pid_is_preserved(self):
        paths = self.make_run_locks()
        queue._write(paths[0], {"pid": 999001})
        with (patch.object(expansion, "active_job_processes", return_value=[]),
              patch.object(expansion, "read_process", return_value=None),
              patch.object(expansion, "process_groups", return_value=[])):
            with self.assertRaisesRegex(ValueError, "lock PID"):
                expansion.recover_stale_locks(self.output, self.plan, {})
        self.assertTrue(all(path.exists() for path in paths))

    def test_only_verified_dead_locks_are_removed_and_audited(self):
        paths = self.make_run_locks()
        before = {path: path.read_bytes() for path in self.metadata.glob("*.json")}
        audit = {}
        with (patch.object(expansion, "active_job_processes", return_value=[]),
              patch.object(expansion, "read_process", return_value=None),
              patch.object(expansion, "process_groups", return_value=[]) as groups):
            expansion.recover_stale_locks(self.output, self.plan, audit)
        self.assertTrue(all(not path.exists() for path in paths))
        groups.assert_called_once_with({857361, 857362})
        self.assertEqual(audit["recovered_run_locks"], {"T0": {"pid": 857361}, "T1": {"pid": 857362}})
        self.assertEqual(before, {path: path.read_bytes() for path in self.metadata.glob("*.json")})
        self.assertEqual(queue._read(self.output / "_queue/screen_expanded/handoff.json"), audit)

    def test_unrecorded_runner_prevents_original_and_expanded_recovery(self):
        paths = self.make_run_locks()
        state = queue._read(self.metadata / "status.json")
        for expanded in (False, True):
            with self.subTest(expanded=expanded):
                with (patch.object(expansion, "active_job_processes", return_value=[999123]),
                      patch.object(expansion, "read_process", return_value=None),
                      patch.object(expansion, "process_groups", return_value=[])):
                    with self.assertRaisesRegex(RuntimeError, "unrecorded launches"):
                        if expanded:
                            expansion.recover_expanded_locks(self.output, self.plan, state, {})
                        else:
                            expansion.recover_stale_locks(self.output, self.plan, {})
                self.assertTrue(all(path.exists() for path in paths))

    def test_active_process_scan_finds_same_output_without_status_or_lock(self):
        processes = {pid: self.process(pid=pid, job=self.plan["jobs"][2]) for pid in range(1001, 1007)}
        processes[1002]["cwd"] = str(self.work)
        processes[1003]["argv"] = ["python", "unrelated.py"]
        argv = processes[1004]["argv"]
        argv[argv.index("--out") + 1] = str(self.work / "another-output")
        processes[1005] = None
        entries = []
        for pid in processes:
            entry = Mock()
            entry.name = str(pid)
            entry.stat.return_value.st_uid = 1001 if pid == 1006 else 1000
            entries.append(entry)
        with (patch.object(Path, "iterdir", return_value=iter(entries)),
              patch.object(expansion.os, "getuid", return_value=1000, create=True),
              patch.object(expansion, "read_process", side_effect=lambda pid, **kwargs: processes[pid])):
            # An absolute output still conflicts even if the runner's cwd is
            # another workspace; reading cwd is unnecessary for detection.
            self.assertEqual(expansion.active_job_processes(self.plan), [1001, 1002])

    def test_interrupted_initialization_without_report_is_rejected_without_deleting_files(self):
        directory = Path(self.plan["jobs"][0]["output_dir"])
        directory.mkdir(parents=True)
        (directory / "run.log").write_text("initializing")
        (directory / "launch.json").write_text("{}")
        clean = expansion.verify_resume_checkpoints(self.plan)
        self.assertEqual(clean["T0"]["resume_epoch"], 0)
        for filename in ("split_manifest.json", "comparison.json.tmp", "last.tmp", "best.tmp"):
            with self.subTest(filename=filename):
                artifact = directory / filename
                artifact.write_text("interrupted write")
                try:
                    with self.assertRaisesRegex(ValueError, "interrupted initialization"):
                        expansion.verify_resume_checkpoints(self.plan)
                    self.assertEqual(artifact.read_text(), "interrupted write")
                finally:
                    artifact.unlink()

    def test_checkpoint_selection_can_only_recover_exact_saved_best_or_current_last(self):
        from tools import run_triprd_task as runner
        job = self.plan["jobs"][0]
        directory = Path(job["output_dir"])
        directory.mkdir(parents=True)
        contract = {"fixture": "contract"}
        queue._write(directory / "comparison.json", {
            "status": "running", "config": job["expected_config"],
            "protocol": contract, "contract_sha256": queue._digest(contract),
        })
        (directory / "last.pt").write_bytes(b"fixture last")
        (directory / "best.pt").write_bytes(b"fixture best")
        for best_epoch, valid in ((None, False), (4, False), (3, True), (5, True)):
            with self.subTest(best_epoch=best_epoch):
                def checkpoint(path, _contract):
                    return ({"epoch": 5, "best_epoch": best_epoch} if Path(path).name == "last.pt"
                            else {"epoch": 3, "best_epoch": 3})
                with patch.object(runner, "read_checkpoint", side_effect=checkpoint):
                    if valid:
                        result = expansion.verify_resume_checkpoints(self.plan)
                        self.assertEqual(result["T0"]["resume_epoch"], 5)
                    else:
                        with self.assertRaisesRegex(ValueError, "cannot be recovered"):
                            expansion.verify_resume_checkpoints(self.plan)

    def test_interruption_unwinds_migration_lock_and_restores_signal_handler(self):
        prior = expansion.signal.getsignal(expansion.signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            with expansion.interrupt_guard(), expansion.migration_guard(self.output):
                self.assertTrue((self.output / "task_expansion.lock").exists())
                handler = expansion.signal.getsignal(expansion.signal.SIGTERM)
                handler(expansion.signal.SIGTERM, None)
        self.assertFalse((self.output / "task_expansion.lock").exists())
        self.assertIs(expansion.signal.getsignal(expansion.signal.SIGTERM), prior)


if __name__ == "__main__":
    unittest.main()
