"""Behavioral checks for the task-oriented model and source teacher controls."""

import inspect
import unittest

import torch

from models.triprd_task import TaskStudent, TaskTeacher


def _model(kind=TaskStudent, **options):
    return kind(model_variant="tiny", base_channels=8, pretrained=False, **options)


def _loss(output):
    return (output["logits"].square().mean() + output["boundary_logits"].square().mean()
            + sum(value.square().mean() for value in output["aux_logits"]))


def _has_gradient(module):
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in module.parameters())


class TaskModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)

    def setUp(self):
        torch.manual_seed(21)
        self.rgb = torch.randn(2, 3, 35, 51)
        self.thermal = torch.rand(2, 3, 35, 51)
        self.depth = torch.rand(2, 1, 35, 51) * .9 + .05

    def test_rgb_only_interface_and_rectangular_output_shapes(self):
        model = _model()
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["rgb", "task"])
        for task in ("sod", "cod"):
            with self.subTest(task=task):
                output = model(self.rgb, task=task)
                self.assertEqual(set(output), {"logits", "aux_logits", "boundary_logits", "task_features"})
                self.assertEqual(output["logits"].shape, (2, 1, 35, 51))
                self.assertEqual(output["boundary_logits"].shape, (2, 1, 35, 51))
                self.assertEqual(tuple(value.shape for value in output["aux_logits"]),
                                 ((2, 1, 5, 7), (2, 1, 3, 4)))
                self.assertEqual(tuple(value.shape for value in output["task_features"]),
                                 ((2, 8, 5, 7), (2, 8, 3, 4)))
                self.assertTrue(torch.isfinite(output["logits"]).all())
        with self.assertRaises(TypeError):
            model(self.rgb, thermal=self.thermal)

    def test_selected_encoder_tail_gets_gradients_and_early_stages_stay_frozen(self):
        model = _model(train_backbone_blocks=2).train()
        stages = model.encoder.backbone.stages
        self.assertFalse(model.encoder.frozen)
        for stage in stages[:2]:
            self.assertFalse(stage.training)
            self.assertFalse(any(parameter.requires_grad for parameter in stage.parameters()))
        for stage in stages[2:]:
            self.assertTrue(stage.training)
            self.assertTrue(all(parameter.requires_grad for parameter in stage.parameters()))
        _loss(model(self.rgb)).backward()
        self.assertTrue(all(parameter.grad is None for stage in stages[:2] for parameter in stage.parameters()))
        self.assertTrue(all(_has_gradient(stage) for stage in stages[2:]))
        self.assertTrue(_has_gradient(model.detail))
        model.eval().train()
        self.assertFalse(any(stage.training for stage in stages[:2]))
        self.assertTrue(all(stage.training for stage in stages[2:]))

    def test_zero_blocks_is_strict_freeze_but_adapter_and_detail_train(self):
        model = _model(train_backbone_blocks=0).train()
        self.assertTrue(model.encoder.frozen)
        self.assertFalse(model.encoder.backbone.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.parameters()))
        _loss(model(self.rgb)).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))
        self.assertTrue(_has_gradient(model.adapters))
        self.assertTrue(_has_gradient(model.detail))

    def test_train_mode_cannot_unfreeze_a_teacher(self):
        teacher = _model(TaskTeacher, train_backbone_blocks=2)
        teacher.requires_grad_(False).eval().train()
        self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))

    def test_decoders_are_disjoint_and_source_does_not_update_cod_decoder(self):
        model = _model(train_backbone_blocks=0)
        sod = {id(parameter) for parameter in model.decoders["sod"].parameters()}
        cod = {id(parameter) for parameter in model.decoders["cod"].parameters()}
        self.assertFalse(sod & cod)
        _loss(model(self.rgb, task="sod")).backward()
        self.assertTrue(_has_gradient(model.decoders["sod"]))
        self.assertTrue(all(parameter.grad is None for parameter in model.decoders["cod"].parameters()))
        self.assertTrue(_has_gradient(model.shared_features))
        model.zero_grad(set_to_none=True)
        _loss(model(self.rgb, task="cod")).backward()
        self.assertTrue(_has_gradient(model.decoders["cod"]))
        self.assertTrue(all(parameter.grad is None for parameter in model.decoders["sod"].parameters()))

    def test_task_features_and_real_pixel_detail_affect_the_final_prediction(self):
        model = _model(train_backbone_blocks=0).eval()
        output = model(self.rgb)
        for value in output["task_features"]:
            value.retain_grad()
        output["logits"].square().mean().backward()
        self.assertTrue(all(value.grad is not None and value.grad.abs().sum() > 0
                            for value in output["task_features"]))
        self.assertTrue(_has_gradient(model.detail.stride2))
        self.assertTrue(_has_gradient(model.detail.stride4))
        self.assertTrue(_has_gradient(model.detail.stride8))
        # An intervention on the actual pixel branch must change the decoder
        # output, even with identical semantic encoder/adapters and input RGB.
        hook = model.detail.stride4.register_forward_hook(lambda module, inputs, value: torch.zeros_like(value))
        try:
            changed = model(self.rgb)["logits"]
        finally:
            hook.remove()
        self.assertFalse(torch.allclose(changed, output["logits"]))

    def test_detail_ablation_has_no_hidden_detail_parameters(self):
        model = _model(detail_branch=False)
        self.assertIsNone(model.detail)
        self.assertFalse(any(name.startswith("detail.") for name, _ in model.named_parameters()))
        output = model(self.rgb)
        self.assertEqual(output["logits"].shape, (2, 1, 35, 51))
        _loss(output).backward()
        self.assertTrue(_has_gradient(model.adapters))

    def test_real_auxiliary_changes_task_values_and_receives_gradients(self):
        teacher = _model(TaskTeacher, train_backbone_blocks=0).eval()
        output = teacher(self.rgb, self.thermal, self.depth)
        changed = teacher(self.rgb, self.thermal.flip(-1), self.depth.flip(-2))
        self.assertFalse(torch.allclose(output["logits"], changed["logits"]))
        self.assertTrue(all(not torch.allclose(left, right) for left, right in
                            zip(output["task_features"], changed["task_features"])))
        _loss(output).backward()
        self.assertTrue(_has_gradient(teacher.modal_adapters["T"]))
        self.assertTrue(_has_gradient(teacher.modal_adapters["D"]))
        self.assertTrue(all(_has_gradient(fusion) for fusion in teacher.fusion))

    def test_rgb_capacity_control_uses_all_adapters_but_ignores_auxiliary(self):
        teacher = _model(TaskTeacher).eval()
        state_keys = tuple(teacher.state_dict())
        output = teacher(self.rgb, input_mode="rgb")
        changed = teacher(self.rgb, torch.full((1,), float("nan")), torch.zeros(1), input_mode="rgb")
        torch.testing.assert_close(output["logits"], changed["logits"], rtol=0, atol=0)
        self.assertEqual(tuple(teacher.state_dict()), state_keys)
        _loss(output).backward()
        self.assertTrue(_has_gradient(teacher.modal_adapters["T"]))
        self.assertTrue(_has_gradient(teacher.modal_adapters["D"]))

    def test_rgb_coalition_is_auxiliary_independent_and_missing_tokens_train(self):
        teacher = _model(TaskTeacher).eval()
        output = teacher(self.rgb, coalition="R")
        malformed = teacher(self.rgb, torch.zeros(1), torch.full((1,), float("nan")), coalition="R")
        torch.testing.assert_close(output["logits"], malformed["logits"], rtol=0, atol=0)
        _loss(output).backward()
        self.assertTrue(all(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in teacher.missing_tokens.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.modal_adapters.parameters()))

    def test_coalitions_cache_encoder_and_match_individual_forwards(self):
        teacher = _model(TaskTeacher).eval()
        for mode, expected_calls in (("multimodal", 3), ("rgb", 1)):
            with self.subTest(input_mode=mode):
                calls = []
                hook = teacher.encoder.register_forward_hook(lambda *arguments: calls.append(1))
                try:
                    outputs = teacher.forward_coalitions(self.rgb, self.thermal, self.depth, input_mode=mode)
                finally:
                    hook.remove()
                self.assertEqual(set(outputs), {"R", "RTD"})
                self.assertEqual(len(calls), expected_calls)
                for coalition in outputs:
                    individual = teacher(self.rgb, self.thermal, self.depth, coalition=coalition, input_mode=mode)
                    torch.testing.assert_close(outputs[coalition]["logits"], individual["logits"], rtol=0, atol=0)
                teacher.zero_grad(set_to_none=True)
                sum(_loss(output) for output in outputs.values()).backward()
                missing = [name for name, parameter in teacher.named_parameters()
                           if parameter.requires_grad and parameter.grad is None]
                self.assertEqual(missing, [])

    def test_invalid_configuration_and_inputs_fail_explicitly(self):
        for count in (-1, 5, 1.5, True):
            with self.subTest(train_backbone_blocks=count), self.assertRaises(ValueError):
                _model(train_backbone_blocks=count)
        with self.assertRaises(ValueError):
            _model(detail_branch="false")
        student = _model()
        for rgb in (torch.zeros(2, 1, 35, 51), torch.zeros(0, 3, 35, 51),
                    torch.zeros(2, 3, 35, 51, dtype=torch.uint8), torch.full((2, 3, 35, 51), float("nan"))):
            with self.assertRaises(ValueError):
                student(rgb)
        with self.assertRaises(ValueError):
            student(self.rgb, task="other")
        teacher = _model(TaskTeacher)
        for options in ({}, {"thermal": self.thermal}, {"input_mode": "other"}, {"coalition": "other"},
                        {"thermal": self.thermal[:1], "depth": self.depth}):
            with self.subTest(options=tuple(options)), self.assertRaises(ValueError):
                teacher(self.rgb, **options)


if __name__ == "__main__":
    unittest.main()
