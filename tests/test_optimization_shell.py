"""Exercise the Bash all-phase entry point using a harmless child command."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if BASH is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH = "C:/Program Files/Git/bin/bash.exe"

STUB = r'''#!/usr/bin/env bash
set -eu
phase=""; gpus=""; resume=0; execute=0; batch=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) phase="$2"; shift ;;
        --gpus) gpus="$2"; shift ;;
        --resume) resume=1 ;;
        --execute) execute=1 ;;
        --batch-size) batch="$2"; shift ;;
    esac
    shift
done
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$gpus" "$resume" "$execute" "$batch" "${CUDA_VISIBLE_DEVICES:-unset}" >> "$FIXTURE_EVENTS"
if [[ "$phase" == "${FAIL_PHASE:-}" ]]; then exit "${FAIL_CODE:-1}"; fi
'''


@unittest.skipUnless(BASH, "Bash is required to exercise the Linux entry point")
class OptimizationShellTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix=".optimization-shell-", dir=ROOT)
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.stub = self.folder / "fake_python.sh"
        self.stub.write_text(STUB, encoding="utf-8", newline="\n")
        self.stub.chmod(0o700)
        self.events = self.folder / "events.tsv"

    def invoke(self, phase="all", **settings):
        env = os.environ.copy()
        for name in ("GPUS", "GPU", "CUDA_VISIBLE_DEVICES", "RESUME", "DRY_RUN", "SPLIT_MANIFEST",
                     "FAIL_PHASE", "FAIL_CODE", "BATCH_SIZE"):
            env.pop(name, None)
        env.update(PYTHON=self.stub.as_posix(), FIXTURE_EVENTS=self.events.as_posix(),
                   DINO_REPO="fixture-repo", DINO_WEIGHTS="fixture.safetensors", GPUS="0,1,2,3")
        env.update(settings)
        result = subprocess.run([BASH, "tools/run_triprd_optimization.sh", phase], cwd=ROOT,
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        rows = [line.split("\t") for line in self.events.read_text().splitlines()] if self.events.exists() else []
        return result, rows

    def test_all_runs_four_phases_with_one_matched_batch_and_four_gpus(self):
        result, rows = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row[0] for row in rows], ["baseline", "probe", "screen", "final"])
        self.assertTrue(all(row[1:] == ["0,1,2,3", "0", "1", "8", "unset"] for row in rows))

    def test_all_stops_on_failed_or_blocked_phase(self):
        for code in ("1", "2"):
            with self.subTest(code=code):
                self.events.unlink(missing_ok=True)
                result, rows = self.invoke(FAIL_PHASE="probe", FAIL_CODE=code)
                self.assertEqual(result.returncode, int(code), result.stderr)
                self.assertEqual([row[0] for row in rows], ["baseline", "probe"])

    def test_dry_run_previews_later_blocked_phases_without_execute(self):
        result, rows = self.invoke(DRY_RUN="1", FAIL_PHASE="screen", FAIL_CODE="2")
        self.assertEqual(result.returncode, 2)
        self.assertEqual([row[0] for row in rows], ["baseline", "probe", "screen", "final"])
        self.assertTrue(all(row[3] == "0" for row in rows))

    def test_resume_forwarded_and_inherited_visibility_is_preserved(self):
        result, rows = self.invoke("screen", RESUME="1", CUDA_VISIBLE_DEVICES="0,1,2,3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows, [["screen", "0,1,2,3", "1", "1", "8", "0,1,2,3"]])

    def test_sigterm_to_shell_reaches_launcher_before_shell_exits(self):
        self.stub.write_text('''#!/usr/bin/env bash
trap 'echo stopped > "$FIXTURE_STOPPED"; exit 0' TERM
echo ready > "$FIXTURE_READY"
while true; do sleep 0.02; done
''', encoding="utf-8", newline="\n")
        env = os.environ.copy()
        env.update(PYTHON=self.stub.as_posix(), DINO_REPO="fixture-repo",
                   DINO_WEIGHTS="fixture.safetensors", GPUS="0,1,2,3",
                   FIXTURE_READY=(self.folder / "ready").as_posix(),
                   FIXTURE_STOPPED=(self.folder / "stopped").as_posix(),
                   FIXTURE_STDOUT=(self.folder / "signal.log").as_posix())
        driver = r'''
bash tools/run_triprd_optimization.sh all > "$FIXTURE_STDOUT" 2>&1 &
parent_pid=$!
trap 'kill -TERM "$parent_pid" 2>/dev/null || true' EXIT
for attempt in {1..100}; do
    [[ -f "$FIXTURE_READY" ]] && break
    sleep 0.02
done
[[ -f "$FIXTURE_READY" ]] || exit 90
kill -TERM "$parent_pid"
result=0
wait "$parent_pid" || result=$?
[[ "$result" == 143 ]] || exit 91
[[ -f "$FIXTURE_STOPPED" ]] || exit 92
'''
        result = subprocess.run([BASH, "-c", driver], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
