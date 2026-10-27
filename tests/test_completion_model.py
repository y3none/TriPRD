"""Behavioral checks for the actual completion forward and transfer semantics."""
import inspect
import unittest
import torch
from models.triprd_completion import CompletionStudent, CompletionTeacher, _object_context
from models.triprd_completion_loss import supervised_loss


def model(kind=CompletionStudent, **options):
    return kind(model_variant="tiny", base_channels=8, pretrained=False, **options)


def has_gradient(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


class CompletionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        torch.manual_seed(37)
        self.rgb = torch.randn(2, 3, 35, 51)
        self.thermal = torch.rand(2, 3, 35, 51)
        self.depth = torch.rand(2, 1, 35, 51) * .9 + .05
        self.mask = (torch.rand(2, 1, 35, 51) > .6).float()

    def test_rgb_only_interface_shapes_and_no_unused_old_decoders(self):
        network = model()
        self.assertEqual(list(inspect.signature(network.forward).parameters),
                         ["rgb", "task", "cue", "perturb_cue"])
        output = network(self.rgb)
        self.assertEqual(output["logits"].shape, (2, 1, 35, 51))
        self.assertEqual(output["boundary_logits"].shape, (2, 1, 35, 51))
        self.assertEqual([tuple(x.shape[-2:]) for x in output["stage_logits"]], [(3, 4), (5, 7), (9, 13)])
        self.assertTrue(output["cue"].requires_grad)
        self.assertFalse(any("decoder" in name for name, _ in network.named_parameters()))
        with self.assertRaises(TypeError):
            network(self.rgb, thermal=self.thermal)

    def test_both_tasks_train_same_complete_refiner_with_only_selected_locator(self):
        network = model(train_backbone_blocks=2)
        identity = [id(p) for p in network.refiner.parameters()]
        for task, other in (("sod", "cod"), ("cod", "sod")):
            network.zero_grad(set_to_none=True)
            supervised_loss(network(self.rgb, task=task), self.mask)["total"].backward()
            self.assertEqual([id(p) for p in network.refiner.parameters()], identity)
            self.assertTrue(has_gradient(network.locators[task]))
            self.assertTrue(all(p.grad is None for p in network.locators[other].parameters()))
            self.assertTrue(all(has_gradient(stage) for stage in network.refiner.stages))
            self.assertTrue(has_gradient(network.refiner.stages[-1].mask))
            self.assertTrue(has_gradient(network.refiner.stages[-1].boundary))
            self.assertTrue(has_gradient(network.detail))
            self.assertTrue(has_gradient(network.encoder.backbone.stages[-1]))

    def test_cod_final_mask_trains_locator_but_source_cue_stays_detached(self):
        network = model()
        for task in ("sod", "cod"):
            network.zero_grad(set_to_none=True)
            output = network(self.rgb, task=task)
            output["logits"].square().mean().backward()
            self.assertEqual(output["cue"].requires_grad, task == "cod")
            self.assertEqual(has_gradient(network.locators[task]), task == "cod")
        # During locator warmup, mask supervision still differentiates through
        # the fixed refiner's operations to the only trainable COD locator.
        network.requires_grad_(False)
        network.locators["cod"].requires_grad_(True)
        network.zero_grad(set_to_none=True)
        network(self.rgb, task="cod")["logits"].square().mean().backward()
        self.assertTrue(has_gradient(network.locators["cod"]))
        self.assertTrue(all(p.grad is None for p in network.refiner.parameters()))
        external = self.mask.clone().requires_grad_()
        output = network(self.rgb, task="cod", cue=external)
        self.assertFalse(output["cue"].requires_grad)

    def test_predicted_boundary_numerically_changes_later_mask_and_receives_mask_gradient(self):
        network = model().eval()
        output = network(self.rgb)
        first_edge = output["stage_boundaries"][0]
        second_edge = output["stage_boundaries"][1]
        first_edge.retain_grad()
        second_edge.retain_grad()
        output["logits"].square().mean().backward()
        self.assertGreater(first_edge.grad.abs().sum().item(), 0)
        self.assertGreater(second_edge.grad.abs().sum().item(), 0)
        hook = network.refiner.stages[0].boundary.register_forward_hook(
            lambda module, inputs, value: torch.full_like(value, 12))
        try:
            changed = network(self.rgb)["logits"]
        finally:
            hook.remove()
        self.assertFalse(torch.allclose(changed, output["logits"]))

    def test_full_image_correction_and_pixel_detail_survive_empty_cue(self):
        network = model().eval()
        rgb = self.rgb.detach().requires_grad_()
        cue = torch.zeros_like(self.mask)
        output = network(rgb, cue=cue)
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertGreater(output["logits"].sigmoid().min().item(), 0)
        output["logits"][..., -5:, -5:].sum().backward()
        self.assertGreater(rgb.grad[..., -10:, -10:].abs().sum().item(), 0)
        self.assertTrue(has_gradient(network.detail.stride4))
        self.assertTrue(has_gradient(network.refiner.stages[-1].global_context))
        hook = network.detail.stride4.register_forward_hook(lambda module, inputs, value: value * 0)
        try:
            changed = network(self.rgb, cue=cue)["logits"]
        finally:
            hook.remove()
        self.assertFalse(torch.allclose(changed, output["logits"]))

    def test_auxiliary_changes_teacher_prediction_but_never_rgb_auto_cue(self):
        teacher = model(CompletionTeacher).eval()
        output = teacher(self.rgb, self.thermal, self.depth)
        changed = teacher(self.rgb, self.thermal.flip(-1), self.depth.flip(-2))
        torch.testing.assert_close(output["cue"], changed["cue"], rtol=0, atol=0)
        torch.testing.assert_close(output["coarse_logits"], changed["coarse_logits"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(output["logits"], changed["logits"]))
        supervised_loss(output, self.mask)["total"].backward()
        self.assertTrue(has_gradient(teacher.modal_adapters["T"]))
        self.assertTrue(has_gradient(teacher.modal_adapters["D"]))
        self.assertTrue(all(has_gradient(fusion) for fusion in teacher.fusion))
        self.assertTrue(has_gradient(teacher.locator))

    def test_same_explicit_cue_is_used_detached_by_teacher_and_student(self):
        student, teacher = model().train(), model(CompletionTeacher).eval()
        source = student(self.rgb, task="sod", perturb_cue=True)
        cue = source["cue"].clone().requires_grad_()
        target = teacher(self.rgb, self.thermal, self.depth, cue=cue)
        torch.testing.assert_close(source["cue"], target["cue"], rtol=0, atol=0)
        self.assertFalse(target["cue"].requires_grad)
        target["logits"].mean().backward()
        self.assertIsNone(cue.grad)
        changed = teacher(self.rgb, self.thermal, self.depth, cue=1 - cue)
        self.assertFalse(torch.allclose(target["logits"], changed["logits"]))

    def test_rgb_control_uses_all_adapters_without_reading_auxiliary(self):
        teacher = model(CompletionTeacher).eval()
        output = teacher(self.rgb, input_mode="rgb")
        changed = teacher(self.rgb, torch.tensor(float("nan")), torch.zeros(1), input_mode="rgb")
        torch.testing.assert_close(output["logits"], changed["logits"], rtol=0, atol=0)
        supervised_loss(output, self.mask)["total"].backward()
        self.assertTrue(has_gradient(teacher.modal_adapters["T"]))
        self.assertTrue(has_gradient(teacher.modal_adapters["D"]))

    def test_empty_full_and_tiny_mass_prototypes_are_fp32_finite_under_autocast(self):
        for value in (0., 1., 1e-12, 1 - 1e-7):
            network = model()
            cue = torch.full_like(self.mask, value)
            feature = torch.randn(2, 8, 3, 4, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                context = _object_context(feature.to(torch.bfloat16), cue)
                output = network(self.rgb, cue=cue)
                loss = supervised_loss(output, self.mask)["total"]
            self.assertTrue(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in context))
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters()))
            if value in (0., 1.):
                global_mean = feature.to(torch.bfloat16).float().mean((-2, -1), keepdim=True)
                torch.testing.assert_close(context[0 if value == 0 else 1], global_mean)

    def test_prediction_perturbation_replays_with_torch_rng_and_eval_disallows_it(self):
        network = model().train()
        torch.manual_seed(123)
        first = network(self.rgb, task="sod", perturb_cue=True)["cue"]
        torch.manual_seed(123)
        second = network(self.rgb, task="sod", perturb_cue=True)["cue"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        with self.assertRaises(ValueError):
            network(self.rgb, task="cod", perturb_cue=True)
        with self.assertRaises(ValueError):
            network.eval()(self.rgb, task="sod", perturb_cue=True)

    def test_invalid_cue_and_auxiliary_are_rejected(self):
        network = model()
        for cue in (torch.ones(2, 1, 2, 2), self.mask + 1, self.mask - 1,
                    torch.full_like(self.mask, float("nan")), self.mask.long()):
            with self.assertRaises(ValueError):
                network(self.rgb, cue=cue)
        with self.assertRaises(ValueError):
            model(CompletionTeacher)(self.rgb)
        with self.assertRaises(ValueError):
            network(self.rgb, task="unknown")


if __name__ == "__main__":
    unittest.main()
