"""Checkpoint isolation and provenance tests for the new relation experiments."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tools.triprd_relation_training import (
    BRANCH_FORMAT,
    export_relation_checkpoint,
    load_full_checkpoint,
    load_relation_checkpoint,
    state_digest,
)
from tools.triprd_training import sha256_file


class RelationStudentStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
        self.relation_branch = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        self.decoder = nn.Linear(3, 2)
        self.head = nn.Linear(2, 1)
        self.register_buffer("temporary_marker", torch.zeros(1), persistent=False)


class RelationTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="triprd_relation_transfer_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_path = self.root / "source.pt"
        self.artifact_path = self.root / "transfer.pt"
        self.protocol = {
            "method": "triprd_relation_v1", "model_variant": "tiny",
            "base_channels": 8, "image_size": 64, "rgb_normalization": "[0,1]",
            "relation_dim": 4, "relation_kernel": 3,
        }
        torch.manual_seed(100)
        self.source = RelationStudentStub()
        torch.manual_seed(101)
        self.target = RelationStudentStub()

    def save_source(self, stage="C/vdt", protocol=None, model=None):
        state = {
            "model": self.source.state_dict() if model is None else model,
            "stage": stage, "epoch": 7, "seed": 42,
            "protocol": dict(self.protocol if protocol is None else protocol),
            "optimizer": {"state": {"momentum": torch.ones(20)}},
        }
        torch.save(state, self.source_path)
        return state

    def export(self, stage="C/vdt"):
        self.save_source(stage)
        return export_relation_checkpoint(self.source_path, self.artifact_path,
                                          expected_protocol=self.protocol)

    def load(self, stage="C/vdt"):
        return load_relation_checkpoint(self.target, self.artifact_path,
                                        expected_protocol=self.protocol, expected_stage=stage)

    def assert_unchanged(self, before):
        self.assertEqual(state_digest(self.target), before)

    def test_artifact_contains_only_branch_tensors_and_json_provenance(self):
        metadata = self.export()
        artifact = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        self.assertEqual(artifact["format"], BRANCH_FORMAT)
        self.assertEqual(set(artifact["relation_branch"]), set(self.source.relation_branch.state_dict()))
        self.assertFalse(any(key.startswith(("encoder.", "decoder.", "head."))
                             for key in artifact["relation_branch"]))
        self.assertNotIn("model", artifact)
        self.assertNotIn("optimizer", artifact)
        self.assertNotIn("relation_branch", metadata)
        self.assertEqual(metadata["source_stage"], "C/vdt")
        self.assertEqual(metadata["source_epoch"], 7)
        self.assertEqual(metadata["source_seed"], 42)
        self.assertEqual(metadata["source_checkpoint_sha256"], sha256_file(self.source_path))
        json.dumps(metadata, allow_nan=False)

    def test_transfer_changes_only_relation_branch_including_its_buffers(self):
        self.source.relation_branch[1].running_mean.fill_(4.0)
        self.source.relation_branch[1].num_batches_tracked.fill_(7)
        self.export()
        before = {key: value.clone() for key, value in self.target.state_dict().items()}
        branch_before = state_digest(self.target.relation_branch)
        metadata = self.load()
        self.assertNotEqual(state_digest(self.target.relation_branch), branch_before)
        self.assertEqual(state_digest(self.target.relation_branch), state_digest(self.source.relation_branch))
        for key, value in self.target.state_dict().items():
            expected = self.source.state_dict()[key] if key.startswith("relation_branch.") else before[key]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertEqual(metadata["transfer_scope"], "relation_branch")
        self.assertTrue(metadata["preprocessing_verified"])
        self.assertEqual(metadata["artifact_sha256"], sha256_file(self.artifact_path))
        json.dumps(metadata, allow_nan=False)

    def test_export_accepts_only_matching_new_source_stages(self):
        for stage in ("A/vdt", "C/cod", "teacher", None):
            with self.subTest(stage=stage):
                self.save_source(stage)
                with self.assertRaisesRegex(ValueError, "Relation export requires"):
                    export_relation_checkpoint(self.source_path, self.artifact_path,
                                               expected_protocol=self.protocol)
        for stage in ("B/vdt", "C/vdt", "D/vdt"):
            with self.subTest(stage=stage):
                self.export(stage)
                self.load(stage)

    def test_legacy_and_full_checkpoints_cannot_cross_branch_boundary(self):
        self.save_source()
        before = state_digest(self.target)
        with self.assertRaisesRegex(ValueError, "full and legacy checkpoints"):
            load_relation_checkpoint(self.target, self.source_path, expected_protocol=self.protocol)
        self.assert_unchanged(before)
        self.save_source(protocol={"method": "mean_fusion"})
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            export_relation_checkpoint(self.source_path, self.artifact_path,
                                       expected_protocol=self.protocol)
        raw = {key: value for key, value in self.source.state_dict().items()
               if not key.startswith("relation_branch.")}
        self.save_source(model=raw)
        with self.assertRaisesRegex(ValueError, "Relation branch must contain"):
            export_relation_checkpoint(self.source_path, self.artifact_path,
                                       expected_protocol=self.protocol)

    def test_protocol_requires_every_expected_key_and_strict_values(self):
        self.export()
        original = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        before = state_digest(self.target)
        for key in self.protocol:
            for mutation in ("missing", "mismatch"):
                with self.subTest(key=key, mutation=mutation):
                    artifact = copy.deepcopy(original)
                    if mutation == "missing":
                        artifact["protocol"].pop(key)
                    else:
                        artifact["protocol"][key] = "wrong-value"
                    torch.save(artifact, self.artifact_path)
                    with self.assertRaisesRegex(ValueError, "protocol"):
                        self.load()
                    self.assert_unchanged(before)
        with self.assertRaisesRegex(ValueError, "Expected protocol must include"):
            load_relation_checkpoint(self.target, self.artifact_path, expected_protocol={})

    def test_branch_keys_shapes_dtypes_and_values_are_checked_before_loading(self):
        self.export()
        original = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        before = state_digest(self.target)
        for mutation, error in (("missing", "keys mismatch"), ("unexpected", "keys mismatch"),
                                ("shape", "shape mismatch"), ("dtype", "dtype mismatch"),
                                ("nan", "non-finite"), ("inf", "non-finite"),
                                ("digest", "tensor digest"), ("key-list", "keys differ")):
            with self.subTest(mutation=mutation):
                artifact = copy.deepcopy(original)
                raw = artifact["relation_branch"]
                if mutation == "missing":
                    raw.pop("0.bias")
                elif mutation == "unexpected":
                    raw["encoder.weight"] = torch.ones(3)
                elif mutation == "shape":
                    raw["0.weight"] = torch.ones(10, 10)
                elif mutation == "dtype":
                    raw["0.weight"] = raw["0.weight"].double()
                elif mutation in {"nan", "inf"}:
                    raw["0.weight"][0, 0] = float(mutation)
                elif mutation == "digest":
                    raw["0.weight"][0, 0] += 1.0
                else:
                    artifact["relation_keys"] = []
                torch.save(artifact, self.artifact_path)
                with self.assertRaisesRegex(ValueError, error):
                    self.load()
                self.assert_unchanged(before)

    def test_stage_and_extra_nonbranch_fields_are_rejected(self):
        self.export()
        before = state_digest(self.target)
        with self.assertRaisesRegex(ValueError, "source stage mismatch"):
            self.load(stage="D/vdt")
        self.assert_unchanged(before)
        artifact = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        artifact["encoder"] = self.source.encoder.state_dict()
        torch.save(artifact, self.artifact_path)
        with self.assertRaisesRegex(ValueError, "missing or unexpected fields"):
            self.load()
        self.assert_unchanged(before)

    def test_export_does_not_overwrite_full_source_checkpoint(self):
        self.save_source()
        before = sha256_file(self.source_path)
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            export_relation_checkpoint(self.source_path, self.source_path,
                                       expected_protocol=self.protocol)
        self.assertEqual(sha256_file(self.source_path), before)

    def test_full_load_uses_cpu_and_returns_small_tensor_free_metadata(self):
        source_protocol = {**self.protocol, "train_seed": 42,
                           "split_manifest_sha256": "a" * 64,
                           "source_sha256": {"models/triprd_relation.py": "b" * 64},
                           "cod_scheduler": "cosine"}
        self.save_source(stage="A/cod", protocol=source_protocol)
        with mock.patch("tools.triprd_relation_training.torch.load", wraps=torch.load) as load:
            metadata = load_full_checkpoint(self.target, self.source_path,
                                            expected_stage="A/cod", expected_protocol=self.protocol)
        self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertEqual(state_digest(self.target), state_digest(self.source))
        self.assertNotIn("optimizer", metadata)
        self.assertNotIn("model", metadata)
        self.assertNotIn("relation_branch", metadata)
        self.assertEqual(metadata["protocol"], source_protocol)
        json.dumps(metadata, allow_nan=False)

    def test_branch_export_and_reuse_preserve_complete_json_source_provenance(self):
        source_protocol = {**self.protocol, "train_seed": 42, "split_seed": 20261012,
                           "split_manifest_sha256": "a" * 64,
                           "source_sha256": {"models/triprd_relation.py": "b" * 64},
                           "source_learning_rate": 0.0001}
        self.save_source(protocol=source_protocol)
        exported = export_relation_checkpoint(self.source_path, self.artifact_path,
                                              expected_protocol=self.protocol)
        self.assertEqual(exported["protocol"], source_protocol)
        self.assertEqual(self.load()["protocol"], source_protocol)

    def test_full_load_checks_teacher_kind_and_stage_before_mutation(self):
        expected = {**self.protocol, "teacher_kind": "multimodal"}
        self.save_source(stage="teacher", protocol={**self.protocol, "teacher_kind": "rgb"})
        before = state_digest(self.target)
        with self.assertRaisesRegex(ValueError, "protocol mismatch for teacher_kind"):
            load_full_checkpoint(self.target, self.source_path,
                                 expected_stage="teacher", expected_protocol=expected)
        self.assert_unchanged(before)
        self.save_source(stage="teacher", protocol=expected)
        with self.assertRaisesRegex(ValueError, "Expected a A/cod checkpoint"):
            load_full_checkpoint(self.target, self.source_path,
                                 expected_stage="A/cod", expected_protocol=expected)
        self.assert_unchanged(before)

    def test_full_load_validates_all_state_before_any_mutation(self):
        before = state_digest(self.target)
        for mutation in ("missing", "shape", "dtype", "nan"):
            with self.subTest(mutation=mutation):
                raw = copy.deepcopy(self.source.state_dict())
                if mutation == "missing":
                    raw.pop("head.bias")
                elif mutation == "shape":
                    raw["head.weight"] = torch.ones(9, 9)
                elif mutation == "dtype":
                    raw["head.weight"] = raw["head.weight"].double()
                else:
                    raw["head.weight"][0, 0] = float("nan")
                self.save_source(model=raw)
                with self.assertRaises(ValueError):
                    load_full_checkpoint(self.target, self.source_path,
                                         expected_stage="C/vdt", expected_protocol=self.protocol)
                self.assert_unchanged(before)

    def test_state_digest_includes_parameter_buffer_and_nonpersistent_values(self):
        before = state_digest(self.target)
        with torch.no_grad():
            self.target.encoder[0].weight[0, 0] += 1.0
        self.assertNotEqual(before, state_digest(self.target))
        before = state_digest(self.target)
        self.target.encoder[1].num_batches_tracked += 1
        self.assertNotEqual(before, state_digest(self.target))
        before = state_digest(self.target)
        self.target.temporary_marker += 1
        self.assertNotEqual(before, state_digest(self.target))
        tensor = nn.Module()
        tensor.register_buffer("bfloat16_buffer", torch.ones(1, dtype=torch.bfloat16))
        self.assertEqual(len(state_digest(tensor)), 64)


if __name__ == "__main__":
    unittest.main()
