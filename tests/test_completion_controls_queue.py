"""Real subprocess tests for per-GPU controls, dependency order and cancellation."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import run_triprd_completion_controls_queue as queue


ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import argparse, json, os, subprocess, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--fixture-settings', type=Path, required=True)
p.add_argument('--fixture-group', required=True)
p.add_argument('--fixture-stage', required=True)
p.add_argument('--resume', action='store_true')
a, _ = p.parse_known_args()
settings = json.loads(a.fixture_settings.read_text())
events = Path(settings['events'])
events.mkdir(parents=True, exist_ok=True)
generation = settings.get('generation', 0)
key = a.fixture_group + '/' + a.fixture_stage
first = settings['first'].get(a.fixture_group) == a.fixture_stage
descendant = None
if first and settings.get('descendants'):
    descendant = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
row = {'group': a.fixture_group, 'stage': a.fixture_stage, 'pid': os.getpid(),
       'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'resume': a.resume,
       'descendant_pid': descendant.pid if descendant else None}
def event(kind):
    target = events / f'{generation}_{a.fixture_group}_{a.fixture_stage}_{kind}.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps({**row, 'event': kind, 'time': time.monotonic()}))
    temporary.replace(target)
event('start')
if first and settings.get('barrier', True):
    deadline = time.monotonic() + 12
    expected = [events / f'{generation}_{g}_{s}_start.json' for g, s in settings['first'].items()]
    while not all(path.exists() for path in expected):
        if time.monotonic() > deadline:
            event('barrier_timeout')
            raise SystemExit(11)
        time.sleep(.01)
time.sleep(settings.get('delay', .12))
if settings.get('fail') == key:
    event('failed')
    raise SystemExit(7)
event('end')
'''


def option(command, flag):
    return command[command.index(flag) + 1]


def alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f'/proc/{pid}/stat')
    return not stat.exists() or stat.read_text().split(') ', 1)[1][0] != 'Z'


class CompletionControlsQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='.completion-controls-queue-', dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.output = self.work / 'output'
        self.manifest = self.work / 'manifest.json'
        self.manifest.write_text('{}', encoding='utf-8')
        self.base = ['--output-root', str(self.output), '--split-manifest', str(self.manifest)]
        self.events = self.work / 'events'
        self.child = self.work / 'fixture.py'
        self.child.write_text(CHILD, encoding='utf-8')
        self.settings_path = self.work / 'settings.json'
        self.settings = {'events': str(self.events), 'first': {'A': 'cod', 'B': 'source', 'C': 'teacher', 'D': 'teacher'}}
        self.save_settings()
        self.code_patch = patch.object(queue, 'CODE_FILES', (self.child.relative_to(ROOT).as_posix(),))
        self.code_patch.start()
        self.addCleanup(self.code_patch.stop)
        environment_patch = patch.dict(os.environ)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    def save_settings(self):
        self.settings_path.write_text(json.dumps(self.settings), encoding='utf-8')

    def plan(self, *options):
        args = queue.parse_args([*self.base, *options])
        return args, queue.build_plan(args)

    def fixture(self, *options):
        args, plan = self.plan(*options)
        for group, details in plan['groups'].items():
            for job in details['jobs']:
                job['command'][:3] = [sys.executable, '-u', str(self.child)]
                job['command'] += ['--fixture-settings', str(self.settings_path),
                                   '--fixture-group', group, '--fixture-stage', job['stage']]
        if plan['summary']:
            plan['summary']['command'] = [sys.executable, '-u', str(self.child),
                '--fixture-settings', str(self.settings_path), '--fixture-group', '_summary', '--fixture-stage', 'summary']
        return args, plan

    def run_plan(self, args, plan, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            code = queue.execute_plan(plan, args, poll_seconds=.01, **kwargs)
        status = json.loads((self.output / 'queue_status.json').read_text())
        return code, status

    def records(self, generation=0):
        return [json.loads(path.read_text()) for path in self.events.glob(f'{generation}_*.json')]

    def test_defaults_route_dependencies_and_individual_gpu_visibility(self):
        args, plan = self.plan()
        self.assertEqual([details['gpu'] for details in plan['groups'].values()], ['0', '1', '2', '3'])
        self.assertEqual((plan['effective_batch_size'], plan['micro_batch_size']), (8, 2))
        self.assertEqual(args.locator_warmup_epochs, 0)
        for group, details in plan['groups'].items():
            self.assertEqual([job['stage'] for job in details['jobs']], [*queue.GROUP_STAGES[group], 'test'])
            for job in details['jobs'][:-1]:
                command = job['command']
                self.assertEqual(option(command, '--group'), group)
                self.assertIn('--no-data-parallel', command)
                self.assertNotIn('--data-parallel', command)
                self.assertEqual(option(command, '--batch-size'), '8')
                self.assertEqual(option(command, '--micro-batch-size'), '2')
                self.assertEqual('--teacher-checkpoint' in command, job['stage'] == 'source' and group in ('C', 'D'))
                self.assertEqual('--source-checkpoint' in command, job['stage'] == 'cod' and group != 'A')
                flags = [token for token in command if token.startswith('--')]
                self.assertEqual(len(flags), len(set(flags)))
        self.assertIsNotNone(plan['summary'])

    def test_dry_run_is_read_only_without_resources(self):
        self.manifest.unlink()
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = queue.main([*self.base, '--dino-repo', 'missing/repo', '--dino-weights', 'missing/weights'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(capture.getvalue())['mode'], 'dry-run')
        self.assertFalse(self.output.exists())

    def test_full_train_split_and_test_phase_plans(self):
        _, full = self.plan('--phase', 'train')
        self.assertIsNone(full['summary'])
        _, split = self.plan('--cod-train-mode', 'split')
        self.assertIsNotNone(split['summary'])
        for group, details in split['groups'].items():
            self.assertEqual([job['stage'] for job in details['jobs']], list(queue.GROUP_STAGES[group]))
        _, testing = self.plan('--phase', 'test')
        self.assertTrue(all([job['stage'] for job in details['jobs']] == ['test'] for details in testing['groups'].values()))
        with self.assertRaisesRegex(ValueError, 'full4040'):
            self.plan('--phase', 'test', '--cod-train-mode', 'split')

    def test_v6_project_exports_do_not_block_new_controls_or_write_during_dryrun(self):
        project = self.work / 'TriPRD_completion_v6'
        project.mkdir()
        (project / 'completion_v6_reports.tar.gz').write_bytes(b'project-level exported report')
        (project / 'completion_pipeline.json').write_text('{}')
        (self.work / 'completion_v6_reports.tar.gz').write_bytes(b'parent-level exported report')
        output = project / 'results' / 'completion_controls_v1_seed-20261012'
        before = {p.relative_to(self.work): p.read_bytes() for p in self.work.rglob('*') if p.is_file()}
        capture = io.StringIO()
        with patch.object(queue, 'ROOT', project), contextlib.redirect_stdout(capture):
            code = queue.main(['--output-root', str(output), '--split-manifest', 'missing.json'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(capture.getvalue())['output_root'], str(output.resolve()))
        self.assertFalse(output.exists())
        self.assertEqual(before, {p.relative_to(self.work): p.read_bytes() for p in self.work.rglob('*') if p.is_file()})
        old = project / 'results' / 'archived_run'
        old.mkdir(parents=True)
        (old / 'completion_pipeline.json').write_text('{}')
        with patch.object(queue, 'ROOT', project):
            for candidate in (old, old / 'nested_controls', old / 'cod' / 'nested_controls'):
                with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, 'existing v6'):
                    self.plan('--output-root', str(candidate))
            with self.assertRaisesRegex(ValueError, 'old v6'):
                self.plan('--output-root', str(project))

    def test_report_archive_above_external_output_does_not_mark_a_run(self):
        external = self.work / 'external'
        external.mkdir()
        (external / 'completion_v6_reports.tar.gz').write_bytes(b'exported report')
        output = external / 'controls'
        with patch.object(queue, 'ROOT', self.work / 'separate_project'):
            _, plan = self.plan('--output-root', str(output))
            self.assertEqual(plan['output_root'], str(output.resolve()))
            (external / 'completion_pipeline.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'existing v6'):
                self.plan('--output-root', str(output))

    def test_invalid_gpu_batches_and_old_v6_output_rejected(self):
        for ids in ('0,1', '0,1,2,02', '0,1,2,-1'):
            with self.assertRaises(ValueError):
                self.plan('--gpus', ids)
        with self.assertRaisesRegex(ValueError, 'micro-batch'):
            self.plan('--micro-batch-size', '9')
        with self.assertRaisesRegex(ValueError, 'old v6'):
            self.plan('--output-root', str(self.work / 'completion_v6_seed-20261012'))
        old = self.work / 'old_run'
        old.mkdir()
        (old / 'completion_pipeline.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'existing v6'):
            self.plan('--output-root', str(old / 'new'))

    def test_real_children_are_concurrent_ordered_isolated_and_summarized(self):
        args, plan = self.fixture('--gpus', '4,5,6,7')
        self.output.mkdir()
        (self.output / 'queue.log').write_text('launcher log created by nohup redirection\n')
        code, status = self.run_plan(args, plan)
        self.assertEqual(code, 0)
        self.assertEqual(status['status'], 'complete')
        rows = self.records()
        initial_starts = [row['time'] for row in rows if row['event'] == 'start' and self.settings['first'].get(row['group']) == row['stage']]
        initial_ends = [row['time'] for row in rows if row['event'] == 'end' and self.settings['first'].get(row['group']) == row['stage']]
        self.assertEqual(len(initial_starts), 4)
        self.assertLess(max(initial_starts), min(initial_ends))
        for group, gpu in zip(queue.GROUP_STAGES, ('4', '5', '6', '7')):
            group_rows = [row for row in rows if row['group'] == group]
            self.assertEqual({row['gpu'] for row in group_rows}, {gpu})
            ordered = sorted(group_rows, key=lambda row: row['time'])
            self.assertEqual([(row['stage'], row['event']) for row in ordered],
                             [(stage, event) for stage in (*queue.GROUP_STAGES[group], 'test') for event in ('start', 'end')])
        summary_start = next(row for row in rows if row['group'] == '_summary' and row['event'] == 'start')
        self.assertEqual(summary_start['gpu'], '')
        self.assertGreater(summary_start['time'], max(row['time'] for row in rows if row['group'] != '_summary'))
        self.assertEqual(status['summary']['status'], 'complete')
        self.assertFalse((self.output / 'completion_controls_queue.lock').exists())

    def test_failure_blocks_only_its_group_and_skips_summary(self):
        self.settings['fail'] = 'D/teacher'
        self.save_settings()
        args, plan = self.fixture()
        code, status = self.run_plan(args, plan)
        self.assertEqual(code, 1)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['groups']['D']['jobs']['teacher']['exit_code'], 7)
        self.assertEqual(status['groups']['D']['jobs']['source']['status'], 'skipped')
        self.assertEqual(status['groups']['D']['jobs']['cod']['status'], 'skipped')
        self.assertEqual(status['groups']['D']['jobs']['test']['status'], 'skipped')
        self.assertTrue(all(status['groups'][group]['status'] == 'complete' for group in 'ABC'))
        self.assertFalse(any(row['group'] == '_summary' for row in self.records()))
        self.assertEqual(status['summary']['status'], 'skipped')

    def test_resume_calls_completed_runners_and_contract_rejects_changes(self):
        args, plan = self.fixture('--phase', 'train')
        self.assertEqual(self.run_plan(args, plan)[0], 0)
        with self.assertRaisesRegex(ValueError, '--resume'):
            self.run_plan(args, plan)
        self.settings['generation'] = 1
        self.save_settings()
        args, plan = self.fixture('--phase', 'train', '--resume')
        self.assertEqual(self.run_plan(args, plan)[0], 0)
        starts = [row for row in self.records(1) if row['event'] == 'start']
        self.assertEqual(len(starts), sum(map(len, queue.GROUP_STAGES.values())))
        self.assertTrue(all(row['resume'] for row in starts))
        args.batch_size = 6
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.run_plan(args, plan)
        args.batch_size = 8
        self.manifest.write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.run_plan(args, plan)
        self.manifest.write_text('{}')
        self.child.write_text(CHILD + '\n# changed code\n')
        with self.assertRaisesRegex(ValueError, 'code'):
            self.run_plan(args, plan)

    def test_train_to_test_keeps_contract_and_runs_all_evaluators(self):
        args, plan = self.fixture('--phase', 'train')
        self.assertEqual(self.run_plan(args, plan)[0], 0)
        self.settings.update(generation=1, barrier=False)
        self.save_settings()
        args, plan = self.fixture('--phase', 'test')
        code, status = self.run_plan(args, plan)
        self.assertEqual(code, 0)
        self.assertEqual(status['summary']['status'], 'complete')
        starts = [row for row in self.records(1) if row['event'] == 'start' and row['group'] != '_summary']
        self.assertEqual({(row['group'], row['stage']) for row in starts}, {(group, 'test') for group in 'ABCD'})

    def test_cancel_cleans_every_child_and_descendant_and_unlocks(self):
        self.settings.update(delay=30, descendants=True)
        self.save_settings()
        args, plan = self.fixture()
        deadline = time.monotonic() + 15
        def stop_requested():
            starts = list(self.events.glob('*_start.json'))
            if len(starts) == 4:
                return True
            if time.monotonic() > deadline:
                self.fail('Concurrent children did not start')
            return False
        code, status = self.run_plan(args, plan, stop_requested=stop_requested)
        self.assertEqual(code, 130)
        self.assertEqual(status['status'], 'interrupted')
        starts = [row for row in self.records() if row['event'] == 'start']
        self.assertEqual(len(starts), 4)
        self.assertTrue(all(row['descendant_pid'] for row in starts))
        for row in starts:
            self.assertFalse(alive(row['pid']))
            self.assertFalse(alive(row['descendant_pid']))
        self.assertFalse((self.output / 'completion_controls_queue.lock').exists())
        self.assertFalse(any(row['event'] == 'end' for row in self.records()))

    def test_existing_lock_is_not_removed_and_unmarked_output_is_protected(self):
        args, plan = self.fixture()
        self.output.mkdir()
        lock = self.output / 'completion_controls_queue.lock'
        lock.write_text('{"pid": 12345}')
        with self.assertRaisesRegex(RuntimeError, 'lock exists'):
            self.run_plan(args, plan)
        self.assertTrue(lock.exists())
        lock.unlink()
        (self.output / 'cod').mkdir()
        with self.assertRaisesRegex(ValueError, 'Nonempty'):
            self.run_plan(args, plan)
        self.assertFalse(lock.exists())


if __name__ == '__main__':
    unittest.main()
