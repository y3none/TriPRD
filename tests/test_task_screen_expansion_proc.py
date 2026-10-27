"""Exercise Linux process inspection with controlled /proc files on any host."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import expand_triprd_task_screen as expansion
from tools import run_triprd_task_queue as queue


ROOT = Path(__file__).resolve().parents[1]


class ProcFiles:
    """Intercept only /proc paths; preserve normal filesystem behavior elsewhere."""
    def __init__(self, processes):
        self.processes = processes
        self.cwd_reads = []

    @staticmethod
    def identify(path):
        return re.fullmatch(r"/proc/(\d+)(?:/(stat|cmdline|cwd))?", path.as_posix())

    @contextmanager
    def installed(self):
        real_text, real_bytes = Path.read_text, Path.read_bytes
        real_stat, real_resolve = Path.stat, Path.resolve
        real_iterdir, real_is_dir = Path.iterdir, Path.is_dir

        def row(path):
            match = self.identify(path)
            if match is None:
                return None
            pid = int(match.group(1))
            if pid not in self.processes:
                raise FileNotFoundError(str(path))
            return pid, match.group(2), self.processes[pid]

        def read_text(path, *args, **kwargs):
            found = row(path)
            if found is None:
                return real_text(path, *args, **kwargs)
            pid, name, process = found
            if name != "stat":
                raise AssertionError(f"Unexpected proc text read: {path}")
            fields = ["0"] * 20
            fields[0] = process.get("state", "S")
            fields[1] = str(process.get("ppid", 1))
            fields[2] = str(process.get("pgrp", pid))
            fields[19] = str(process.get("start_ticks", 98765))
            return f"{pid} (fixture process) " + " ".join(fields)

        def read_bytes(path, *args, **kwargs):
            found = row(path)
            if found is None:
                return real_bytes(path, *args, **kwargs)
            pid, name, process = found
            if name != "cmdline":
                raise AssertionError(f"Unexpected proc byte read: {path}")
            if process.get("deny_cmdline"):
                raise PermissionError(f"fixture denied /proc/{pid}/cmdline")
            return b"\0".join(os.fsencode(value) for value in process["argv"]) + b"\0"

        def stat(path, *args, **kwargs):
            found = row(path)
            if found is None:
                return real_stat(path, *args, **kwargs)
            return SimpleNamespace(st_uid=found[2].get("uid", 1000))

        def resolve(path, *args, **kwargs):
            found = row(path)
            if found is None:
                return real_resolve(path, *args, **kwargs)
            pid, name, process = found
            if name != "cwd":
                raise AssertionError(f"Unexpected proc resolve: {path}")
            self.cwd_reads.append(pid)
            if process.get("deny_cwd"):
                raise PermissionError(f"fixture denied /proc/{pid}/cwd")
            return Path(process.get("cwd", ROOT))

        def iterdir(path):
            if path.as_posix() == "/proc":
                return iter(Path("/proc") / str(pid) for pid in self.processes)
            return real_iterdir(path)

        def is_dir(path):
            return True if path.as_posix() == "/proc" else real_is_dir(path)

        with ExitStack() as stack:
            for name, function in (("read_text", read_text), ("read_bytes", read_bytes),
                                   ("stat", stat), ("resolve", resolve),
                                   ("iterdir", iterdir), ("is_dir", is_dir)):
                stack.enter_context(patch.object(Path, name, function))
            stack.enter_context(patch.object(expansion.os, "getuid", return_value=1000, create=True))
            yield self


class ScreenExpansionProcTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".task-expand-proc-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "output"
        self.root.mkdir()
        self.plan = {
            "method": queue.METHOD, "phase": "screen", "output_root": str(self.root),
            "gpus": ["2", "3"], "device": "cuda", "dependencies": {},
            "jobs": [{"name": f"T{i}", "output_dir": str(self.root / "screen" / f"T{i}"),
                      "command": ["python", "-u", str(ROOT / "tools/run_triprd_task.py")],
                      "expected_config": {"group": f"T{i}"}} for i in range(5)],
        }
        self.plan["plan_sha256"] = queue._digest(self.plan)

    def runner(self, group="T0", *, relative_script=False, relative_output=False, **extra):
        output = self.root / "screen" / group
        script = "tools/run_triprd_task.py" if relative_script else str(ROOT / "tools/run_triprd_task.py")
        out = os.path.relpath(output, ROOT) if relative_output else str(output)
        return {"argv": ["python", "-u", script, "--out", out, "--group", group], **extra}

    def test_read_without_cwd_returns_identity_without_following_inaccessible_link(self):
        fixture = ProcFiles({1234: self.runner(deny_cwd=True)})
        with fixture.installed():
            process = expansion.read_process(1234, include_cwd=False)
        self.assertEqual(process["pid"], 1234)
        self.assertEqual(process["start_ticks"], 98765)
        self.assertEqual(process["argv"], fixture.processes[1234]["argv"])
        self.assertEqual(fixture.cwd_reads, [])

    def test_unrelated_same_user_unreadable_cwd_does_not_block_scan(self):
        fixture = ProcFiles({1234: {"argv": ["python", "unrelated_training.py"], "deny_cwd": True}})
        with fixture.installed():
            self.assertEqual(expansion.active_job_processes(self.plan), [])
        self.assertEqual(fixture.cwd_reads, [])

    def test_absolute_matching_runner_and_output_are_active_without_cwd_access(self):
        fixture = ProcFiles({1234: self.runner(deny_cwd=True)})
        with fixture.installed():
            self.assertEqual(expansion.active_job_processes(self.plan), [1234])
        self.assertEqual(fixture.cwd_reads, [])

    def test_same_runner_with_unrelated_absolute_output_ignores_inaccessible_cwd(self):
        process = self.runner(deny_cwd=True)
        process["argv"][process["argv"].index("--out") + 1] = str(self.root / "different")
        fixture = ProcFiles({1234: process})
        with fixture.installed():
            self.assertEqual(expansion.active_job_processes(self.plan), [])
        self.assertEqual(fixture.cwd_reads, [])

    def test_relative_runner_and_output_require_matching_workspace(self):
        fixture = ProcFiles({
            1234: self.runner(relative_script=True, relative_output=True, cwd=str(ROOT)),
            1235: self.runner(relative_script=True, relative_output=True, cwd=str(self.root)),
        })
        with fixture.installed():
            self.assertEqual(expansion.active_job_processes(self.plan), [1234])
        self.assertEqual(set(fixture.cwd_reads), {1234, 1235})

    def test_relative_candidate_with_unknown_cwd_fails_closed_with_pid(self):
        fixture = ProcFiles({1234: self.runner(relative_script=True, relative_output=True, deny_cwd=True)})
        with fixture.installed(), self.assertRaisesRegex(RuntimeError, "1234"):
            expansion.active_job_processes(self.plan)

    def test_runner_from_other_checkout_writing_same_absolute_output_blocks_recovery(self):
        process = self.runner(deny_cwd=True)
        process["argv"][2] = str(self.root / "another-checkout/tools/run_triprd_task.py")
        fixture = ProcFiles({1234: process})
        with fixture.installed():
            self.assertEqual(expansion.active_job_processes(self.plan), [1234])
        self.assertEqual(fixture.cwd_reads, [])

    def test_unreadable_cmdline_still_fails_closed_without_cwd_inspection(self):
        fixture = ProcFiles({1234: self.runner(deny_cmdline=True, deny_cwd=True)})
        with fixture.installed(), self.assertRaisesRegex(RuntimeError, "1234"):
            expansion.read_process(1234, include_cwd=False)
        self.assertEqual(fixture.cwd_reads, [])

    def test_tracked_permission_denial_is_not_reported_as_dead(self):
        fixture = ProcFiles({1234: self.runner(deny_cwd=True)})
        with fixture.installed(), self.assertRaisesRegex(RuntimeError, "1234"):
            expansion.read_process(1234)

    def test_unreadable_tracked_queue_is_not_signalled_and_lock_remains(self):
        lock = self.root / "task_queue.lock"
        queue._write(lock, {"pid": 1234})
        before = lock.read_bytes()
        fixture = ProcFiles({1234: {
            "argv": ["python", str(ROOT / "tools/run_triprd_task_queue.py"),
                     "--phase", "screen", "--output-root", str(self.root)], "deny_cwd": True,
        }})
        with (fixture.installed(), patch.object(expansion.os, "kill") as kill,
              patch.object(expansion.signal, "pidfd_send_signal", create=True) as pidfd_signal):
            with self.assertRaisesRegex(RuntimeError, "1234"):
                expansion.safe_stop_queue(self.root, self.plan, {})
        kill.assert_not_called()
        pidfd_signal.assert_not_called()
        self.assertEqual(lock.read_bytes(), before)

    def test_unreadable_tracked_run_is_not_cleared_as_stale(self):
        state = {"plan_sha256": self.plan["plan_sha256"], "jobs": {"T0": {"pid": 1234}}}
        queue._write(self.root / "_queue/screen/status.json", state)
        lock = self.root / "screen/T0/task_run.lock"
        queue._write(lock, {"pid": 1234})
        before = lock.read_bytes()
        # The process has changed argv, so broad scan skips it; the recorded PID
        # still must be inspected strictly before its training lock is removed.
        fixture = ProcFiles({1234: {"argv": ["python", "unrelated.py"], "deny_cwd": True}})
        with fixture.installed(), self.assertRaisesRegex(RuntimeError, "1234"):
            expansion.recover_stale_locks(self.root, self.plan, {})
        self.assertEqual(lock.read_bytes(), before)

    def test_resume_retries_handoff_after_old_queue_stopped_before_new_plan_written(self):
        metadata = self.root / "_queue/screen"
        queue._write(metadata / "plan.json", self.plan)
        queue._write(metadata / "status.json", {
            "plan_sha256": self.plan["plan_sha256"], "status": "interrupted",
            "jobs": {"T0": {"pid": 1234, "status": "terminated"}},
        })
        run_lock = self.root / "screen/T0/task_run.lock"
        queue._write(run_lock, {"pid": 1234})
        queue._write(self.root / "_queue/screen_expanded/handoff.json", {"previous_attempt": True})
        original_bytes = {path: path.read_bytes() for path in metadata.glob("*.json")}
        events = []
        def verify(plan):
            self.assertFalse(run_lock.exists())
            self.assertFalse((self.root / "_queue/screen_expanded/plan.json").exists())
            events.append("verified")
            return {"T0": {"resume_epoch": 6}}
        def execute(plan, *, resume):
            self.assertTrue(resume)
            self.assertEqual(plan["gpus"], ["2", "3", "0", "1", "6"])
            self.assertEqual(queue._read(self.root / "_queue/screen_expanded/plan.json"), plan)
            self.assertEqual(events, ["verified"])
            events.append("executed")
            return 0
        with (ProcFiles({}).installed(), patch.object(expansion.sys, "platform", "linux"),
              patch.object(expansion, "validate_saved_plan", return_value=self.plan),
              patch.object(expansion, "verify_resume_checkpoints", side_effect=verify),
              patch.object(queue, "execute_plan", side_effect=execute), redirect_stdout(io.StringIO())):
            code = expansion.main(["--output-root", str(self.root), "--gpus", "2,3,0,1,6",
                                   "--execute", "--resume"])
        self.assertEqual(code, 0)
        self.assertEqual(events, ["verified", "executed"])
        self.assertEqual(original_bytes, {path: path.read_bytes() for path in metadata.glob("*.json")})
        self.assertFalse((self.root / "task_expansion.lock").exists())


if __name__ == "__main__":
    unittest.main()
