"""Matched relation experiment queue contracts; no training or GPU required."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_triprd_relation_queue as launcher


ROOT = Path(__file__).resolve().parents[1]


def option(job, flag):
    return job["command"][job["command"].index(flag) + 1]


class RelationQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".relation-queue-test-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        self.argv = ["--output-root", str(self.work / "output")]

    def plan(self, *extra):
        return launcher.build_plan(launcher.parse_args(self.argv + list(extra)))

    def test_matched_four_gpu_plan_has_no_legacy_dependencies(self):
        plan = self.plan("--gpus", "2,3,6,7")
        self.assertEqual([job["name"] for job in plan["jobs"]],
                         ["A_scratch", "B_rgb_source", "C_rgb_relation", "D_multimodal_relation"])
        self.assertEqual([job["gpu"] for job in plan["jobs"]], ["2", "3", "6", "7"])
        self.assertEqual(plan["required_files"], [])
        self.assertEqual(plan["selection_rule"], "S_alpha")
        recipes = []
        for group, job in zip("ABCD", plan["jobs"]):
            self.assertEqual(option(job, "--groups"), group)
            self.assertEqual(job["required_checkpoints"], [])
            self.assertIn("--pretrained", job["command"])
            self.assertIn("--skip-test", job["command"])
            self.assertEqual(option(job, "--model-variant"), "resnet50")
            self.assertEqual(option(job, "--image-size"), "352")
            self.assertEqual(option(job, "--base-channels"), "256")
            self.assertEqual(option(job, "--batch-size"), "8")
            self.assertEqual(option(job, "--epochs"), "100")
            self.assertEqual(option(job, "--vdt-epochs"), "100")
            self.assertEqual(option(job, "--selection-metric"), "S_alpha")
            self.assertEqual(option(job, "--val-metrics-every"), "5")
            self.assertEqual(option(job, "--cod-warmup-epochs"), "5")
            self.assertEqual(option(job, "--cod-encoder-lr-scale"), "0.1")
            self.assertEqual(option(job, "--cod-scheduler"), "cosine")
            self.assertNotIn("--teacher-checkpoint", job["command"])
            self.assertNotIn("--vdt-checkpoint", job["command"])
            self.assertEqual(Path(job["output_dir"]).parent, Path(plan["run_dir"]))
            flags = [value for value in job["command"] if value.startswith("--")]
            self.assertEqual(len(flags), len(set(flags)))
            recipes.append(job["command"][:-4])
        self.assertTrue(all(recipe == recipes[0] for recipe in recipes))
        self.assertEqual(plan["jobs"][2]["teacher_inputs"], "RGB/RGB/RGB")
        self.assertEqual(plan["jobs"][3]["teacher_inputs"], "RGB/T/D")
        self.assertFalse(Path(plan["run_dir"]).exists())

    def test_smoke_and_independent_train_seed_keep_fixed_split(self):
        plan = self.plan("--gpus", "2,3", "--epochs", "2", "--vdt-epochs", "2",
                         "--max-samples", "32", "--val-metrics-every", "1", "--train-seed", "19",
                         "--split-seed", "7")
        self.assertEqual(Path(plan["run_dir"]).name, "seed-19")
        self.assertEqual([job["gpu"] for job in plan["jobs"]], ["2", "3", "2", "3"])
        for job in plan["jobs"]:
            self.assertEqual(option(job, "--train-seed"), "19")
            self.assertEqual(option(job, "--split-seed"), "7")
            self.assertEqual(option(job, "--cod-warmup-epochs"), "1")
            self.assertEqual(option(job, "--max-samples"), "32")
            self.assertEqual(option(job, "--val-metrics-every"), "1")
        for job in self.plan("--epochs", "1")["jobs"]:
            self.assertEqual(option(job, "--cod-warmup-epochs"), "0")

    def test_dry_run_is_read_only_and_missing_manifest_blocks_execution(self):
        manifest = self.work / "absent.json"
        stream = io.StringIO()
        with patch.object(launcher, "execute_plan") as execute, contextlib.redirect_stdout(stream):
            self.assertEqual(launcher.main(self.argv + ["--split-manifest", str(manifest)]), 0)
        execute.assert_not_called()
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["mode"], "dry-run")
        self.assertEqual(plan["required_files"], [str(manifest)])
        for job in plan["jobs"]:
            self.assertEqual(option(job, "--split-manifest"), str(manifest))
        with patch("tools.run_triprd_optimization.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "Required input files"):
                launcher.execute_plan(plan)
        popen.assert_not_called()
        self.assertFalse((self.work / "output").exists())

    def test_invalid_protocols_fail_before_outputs_exist(self):
        for flags in (("--seed", "-1"), ("--split-seed", str(2 ** 32)),
                      ("--relation-kernel", "2"), ("--relation-kernel", "1"),
                      ("--relation-weight", "0"), ("--relation-weight", "nan"),
                      ("--relation-temperature", "inf"), ("--relation-temperature", "0"),
                      ("--image-size", "16"), ("--model-variant", "tiny"),
                      ("--cod-train-mode", "full", "--max-samples", "32")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.plan(*flags)
        for flags in (("--gpus", "2,2"), ("--epochs", "0"), ("--max-samples", "-1")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(self.argv + list(flags))
        self.assertFalse((self.work / "output").exists())

    def test_generated_commands_are_accepted_by_relation_runner(self):
        from tools import run_triprd_relation as runner

        for extra in ((), ("--epochs", "2", "--vdt-epochs", "2", "--max-samples", "32"),
                      ("--cod-train-mode", "full"),
                      ("--model-variant", "tiny", "--no-pretrained", "--base-channels", "8")):
            for job in self.plan(*extra)["jobs"]:
                with self.subTest(extra=extra, group=job["group"]):
                    parsed = runner.parse_args(job["command"][3:])
                    groups, _, _, _, _ = runner._validate_args(parsed)
                    self.assertEqual(groups, [job["group"]])

    def test_full_train_retains_budget_and_disables_test(self):
        plan = self.plan("--cod-train-mode", "full", "--epochs", "65")
        self.assertEqual(plan["selection_rule"], "last_epoch")
        for job in plan["jobs"]:
            self.assertEqual(option(job, "--cod-train-mode"), "full")
            self.assertEqual(option(job, "--epochs"), "65")
            self.assertEqual(option(job, "--max-samples"), "0")
            self.assertIn("--skip-test", job["command"])

    def test_existing_destination_is_never_overwritten(self):
        plan = self.plan()
        Path(plan["run_dir"]).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
