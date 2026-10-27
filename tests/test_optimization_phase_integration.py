"""Real tiny-CPU baseline -> teacher/probe -> screen interface integration.

Only data I/O and the subprocess boundary are replaced. Scientific success
gates record their actual result instead of requiring random toy data to prove
transfer gains; report, checkpoint, split and recipe validation remains real.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_runner_integration import MemoryCODIndex, MemoryVDTIndex
from tools import run_triprd_ab as shared
from tools import run_triprd_correction as runner
from tools import run_triprd_optimization_phases as launch
from tools.triprd_structure_training import state_digest


ROOT = Path(__file__).resolve().parents[1]


class OptimizationPhaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)
        temporary = tempfile.TemporaryDirectory(prefix=".phase-integration-", dir=ROOT)
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name).resolve()
        vdt_index = MemoryVDTIndex()
        vdt_items = [shared.Item(f"VDT/{sample['rgb'].sample_id}", (sample, vdt_index), "vdt")
                     for sample in vdt_index.entries]
        cod_indexes = [MemoryCODIndex(name) for name in ("COD10K", "CAMO")]
        cod_items = [shared.Item(f"{index.dataset}/{sample.sample_id}", (sample, index), "cod")
                     for index in cod_indexes for sample in index.samples()]
        cls.argv = ["--output-root", str(cls.work / "runs"), "--model-variant", "tiny",
                    "--no-pretrained", "--device", "cpu", "--no-amp", "--num-workers", "0",
                    "--base-channels", "4", "--image-size", "32", "--batch-size", "2",
                    "--epochs", "2", "--teacher-epochs", "2", "--probe-epochs", "2",
                    "--val-metrics-every", "2", "--val-fraction", "0.25", "--max-samples", "4",
                    "--warmup-epochs", "1", "--seed", "11", "--split-seed", "7",
                    "--vdt-root", str(cls.work / "vdt"), "--cod-root", str(cls.work / "cod")]
        cls.commands, cls.training, cls.checkpoint_reads, cls.gates = [], [], [], []
        cls.probe_loaders, cls.constant_fit_keys, cls.validations = [], [], []
        active = {"phase": None, "stage": None, "group": None}
        real_train, real_read = runner._train_epoch, runner._read_checkpoint
        real_loaders, real_constant = runner._loaders, runner._constant_probe
        real_validate = runner._validate_cod

        def read_cod(*args, **kwargs):
            if active["stage"] == "probe":
                raise AssertionError("Source probe must not even index COD")
            return cod_items

        def guard_cod_read(original):
            def read(*args, **kwargs):
                if active["stage"] == "probe":
                    raise AssertionError("Source probe read a COD image or ground truth")
                return original(*args, **kwargs)
            return read

        def train(*args, **kwargs):
            a, epoch = args[6], args[7]
            cls.training.append((active["phase"], a.stage, a.group, epoch))
            return real_train(*args, **kwargs)

        def read_checkpoint(path, contract=None):
            cls.checkpoint_reads.append((active["phase"], active["stage"], active["group"], Path(path).resolve()))
            return real_read(path, contract)

        def loaders(a, splits, device):
            result = real_loaders(a, splits, device)
            if a.stage == "probe":
                if set(result) != {"vdt_train", "vdt_val"}:
                    raise AssertionError("Source probe loaders contain a COD partition")
                cls.probe_loaders.append({key: [item.key for item in loader.dataset.items]
                                         for key, loader in result.items()})
            return result

        def constant(teacher, loader, device, a):
            keys = [item.key for item in loader.dataset.items]
            manifest = json.loads(a.split_manifest.read_text(encoding="utf-8"))
            if keys != manifest["vdt_train"]:
                raise AssertionError("Constant correction was not fitted on source TRAIN only")
            cls.constant_fit_keys.append(keys)
            return real_constant(teacher, loader, device, a)

        def validate(model, items, device, a, intervention="none"):
            if a.stage == "probe":
                raise AssertionError("Source probe must not evaluate COD validation")
            cls.validations.append((active["phase"], a.stage, a.group))
            return real_validate(model, items, device, a, intervention)

        def record_gate(name, gate):
            def record(report):
                try:
                    evidence = gate(report)
                except launch.EvidenceBlocked as error:
                    evidence = {"toy_evidence_blocked": str(error)}
                cls.gates.append((active["phase"], name, evidence))
                return evidence
            return record

        def subprocess_run(command, *, cwd, check):
            if Path(cwd).resolve() != ROOT or not check or command[:3] != [launch.sys.executable, "-u", str(launch.RUNNER)]:
                raise AssertionError("Unexpected process invocation")
            parsed = runner.parse_args(command[3:])
            active.update(stage=parsed.stage, group=parsed.group)
            cls.commands.append((active["phase"], list(command)))
            result = runner.main(command[3:])
            if result:
                raise subprocess.CalledProcessError(result, command)
            return subprocess.CompletedProcess(command, result)

        cls.plans, cls.reports, cls.receipts = {}, {}, {}
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(output))
            stack.enter_context(patch.object(shared, "_vdt_items", return_value=vdt_items))
            stack.enter_context(patch.object(shared, "_cod_items", side_effect=read_cod))
            for index in cod_indexes:
                for name in ("read_image", "read_pair"):
                    stack.enter_context(patch.object(index, name, side_effect=guard_cod_read(getattr(index, name))))
            stack.enter_context(patch.object(runner, "_train_epoch", side_effect=train))
            stack.enter_context(patch.object(runner, "_read_checkpoint", side_effect=read_checkpoint))
            stack.enter_context(patch.object(runner, "_loaders", side_effect=loaders))
            stack.enter_context(patch.object(runner, "_constant_probe", side_effect=constant))
            stack.enter_context(patch.object(runner, "_validate_cod", side_effect=validate))
            for owner, name in ((runner, "_formal_tests"), (shared, "_index_for_dataset")):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("No formal test access in development phases")))
            stack.enter_context(patch.object(launch.subprocess, "run", side_effect=subprocess_run))
            stack.enter_context(patch.object(launch, "teacher_gate", side_effect=record_gate("teacher", launch.teacher_gate)))
            stack.enter_context(patch.object(launch, "student_gate", side_effect=record_gate("student", launch.student_gate)))
            for phase in ("baseline", "probe", "screen"):
                active.update(phase=phase, stage=None, group=None)
                plan = launch.build_plan(launch.parse_args(cls.argv + ["--phase", phase, "--execute"]))
                cls.plans[phase] = plan
                result = launch.execute_plan(plan)
                if result != 0:
                    raise AssertionError(f"Real {phase} phase failed with {result}:\n{output.getvalue()}")
                cls.receipts[phase] = launch._read(Path(plan["root"]) / "phases" / f"{phase}.json")
                for job in plan["jobs"]:
                    cls.reports[(job["stage"], job["group"])] = launch.validate_result(plan, job)
        cls.root = Path(cls.plans["baseline"]["root"])

    def test_real_reports_and_checkpoint_hashes_satisfy_phase_contracts(self):
        for phase, plan in self.plans.items():
            with self.subTest(phase=phase):
                receipt = self.receipts[phase]
                self.assertEqual(receipt["status"], "complete")
                self.assertEqual(receipt["recipe_sha256"], plan["recipe_sha256"])
                for job, saved in zip(plan["jobs"], receipt["jobs"]):
                    report = launch.validate_result(plan, job)
                    self.assertEqual(saved["comparison_sha256"], launch._sha(Path(job["out"]) / "comparison.json"))
                    self.assertEqual(saved["checkpoint_sha256"], report["checkpoint_sha256"])
                    self.assertEqual(report["contract_sha256"], runner._digest(report["protocol"]))
                    self.assertFalse(report["formal"])
                    self.assertEqual(report["test"], {})
                    self.assertEqual(report["test_status"], "skipped")
        self.assertEqual(len(self.gates), 4)

    def test_probe_uses_only_source_train_and_val_and_fixed_final_checkpoint(self):
        report = self.reports[("probe", "D")]
        self.assertFalse(report["cod_access"])
        self.assertFalse(report["validation_used_for_selection"])
        self.assertEqual(report["sample_counts"], {"vdt_train": 3, "vdt_val": 1, "cod_train": 0, "cod_val": 0})
        self.assertEqual(report["selected_epoch"], 2)
        self.assertEqual(report["selection_metric"], "fixed_last_epoch")
        self.assertEqual([row["epoch"] for row in report["history"]], [1, 2])
        self.assertIn("source_recoverability", report["history"][-1])
        self.assertFalse((self.root / "probe/best.pt").exists())
        self.assertTrue((self.root / "probe/last.pt").is_file())
        manifest = launch._manifest(self.plans["probe"])
        self.assertEqual(self.probe_loaders, [{key: manifest[key] for key in ("vdt_train", "vdt_val")}])
        self.assertEqual(self.constant_fit_keys, [manifest["vdt_train"]])
        self.assertTrue(self.validations)
        self.assertTrue(all(stage == "student" for _, stage, _ in self.validations))

    def test_screen_reuses_baselines_but_new_students_do_not_warm_start_from_probe(self):
        self.assertEqual([item["reused"] for item in self.receipts["screen"]["jobs"]], [True, True, False, False])
        screen_commands = [command for phase, command in self.commands if phase == "screen"]
        self.assertEqual(len(screen_commands), 4)
        self.assertTrue(all("--resume" in command for command in screen_commands[:2]))
        self.assertTrue(all("--resume" not in command and "--cod-checkpoint" not in command
                            for command in screen_commands[2:]))
        self.assertEqual([entry for entry in self.training if entry[0] == "screen"],
                         [("screen", "student", group, epoch) for group in "CD" for epoch in (1, 2)])
        initial = self.reports[("probe", "D")]["initial_state_sha256"]
        self.assertEqual({self.reports[("student", group)]["initial_state_sha256"] for group in "ABCD"}, {initial})
        from models.triprd_correction import CorrectionStudent
        probe_model = CorrectionStudent(model_variant="tiny", base_channels=4, pretrained=False,
                                        relation_kernel=3, freeze_backbone=True)
        probe_model.load_state_dict(runner._read_checkpoint(self.root / "probe/last.pt")["model"])
        self.assertNotEqual(state_digest(probe_model), initial)
        reads = [path for phase, stage, group, path in self.checkpoint_reads
                 if phase == "screen" and stage == "student" and group in "CD"]
        self.assertIn(self.root / "teacher/best.pt", reads)
        self.assertNotIn(self.root / "probe/last.pt", reads)
        teacher_digest = launch._sha(self.root / "teacher/best.pt")
        for key in (("probe", "D"), ("student", "C"), ("student", "D")):
            self.assertEqual(self.reports[key]["protocol"]["teacher_sha256"], teacher_digest)


if __name__ == "__main__":
    unittest.main()
