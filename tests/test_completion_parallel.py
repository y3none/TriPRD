"""Batch alignment and gradient-preserving balanced GPU partitioning."""
from collections import namedtuple
import unittest
from unittest.mock import patch

import torch

from tools.completion_parallel import (BalancedDataParallel, balanced_chunk_sizes,
                                       balanced_scatter_kwargs)


class BalancedScatterTests(unittest.TestCase):
    @staticmethod
    def cpu_scatter(devices, chunks, dim, value):
        # The public chunk contract is tested independently from CUDA transport.
        return value.split(chunks, dim=dim)

    def test_global_eight_uses_five_nonempty_chunks_without_padding(self):
        self.assertEqual(balanced_chunk_sizes(8, 5), (2, 2, 2, 1, 1))
        self.assertEqual(balanced_chunk_sizes(10, 5), (2, 2, 2, 2, 2))
        self.assertEqual(balanced_chunk_sizes(4, 5), (1, 1, 1, 1))
        self.assertEqual(balanced_chunk_sizes(1, 5), (1,))
        for total in range(1, 30):
            for count in range(1, 8):
                sizes = balanced_chunk_sizes(total, count)
                self.assertEqual(sum(sizes), total)
                self.assertEqual(len(sizes), min(total, count))
                self.assertLessEqual(max(sizes) - min(sizes), 1)
                self.assertGreater(min(sizes), 0)

    def test_rgb_sensors_and_cue_have_identical_order_and_boundaries(self):
        rgb = torch.arange(8).view(8, 1, 1, 1).float()
        thermal, depth, cue = rgb + 20, rgb + 40, rgb + 60
        with patch("tools.completion_parallel.Scatter.apply", side_effect=self.cpu_scatter) as transport:
            args, kwargs = balanced_scatter_kwargs(
                (rgb,), {"thermal": thermal, "depth": depth, "cue": cue,
                         "task": "source", "optional": None}, [0, 1, 2, 3, 4])
        self.assertEqual(len(args), 5)
        self.assertEqual(len(kwargs), 5)
        self.assertEqual([item[0].shape[0] for item in args], [2, 2, 2, 1, 1])
        for part, metadata in zip(args, kwargs):
            self.assertTrue(torch.equal(metadata["thermal"] - 20, part[0]))
            self.assertTrue(torch.equal(metadata["depth"] - 40, part[0]))
            self.assertTrue(torch.equal(metadata["cue"] - 60, part[0]))
            self.assertEqual(metadata["task"], "source")
            self.assertIsNone(metadata["optional"])
        self.assertTrue(torch.equal(torch.cat([item[0] for item in args]), rgb))
        for call in transport.call_args_list:
            self.assertEqual(call.args[:3], ((0, 1, 2, 3, 4), (2, 2, 2, 1, 1), 0))

    def test_student_and_teacher_share_the_same_cue_partition(self):
        rgb = torch.randn(8, 3, 4, 4)
        cue = torch.rand(8, 1, 4, 4)
        with patch("tools.completion_parallel.Scatter.apply", side_effect=self.cpu_scatter):
            student = balanced_scatter_kwargs((rgb,), {"cue": cue}, range(5))
            teacher = balanced_scatter_kwargs((rgb, rgb + 1, rgb + 2), {"cue": cue}, range(5))
        for student_kwargs, teacher_kwargs in zip(student[1], teacher[1]):
            self.assertTrue(torch.equal(student_kwargs["cue"], teacher_kwargs["cue"]))

    def test_nested_containers_and_short_batches_are_preserved(self):
        Pair = namedtuple("Pair", ("first", "second"))
        tensor = torch.arange(3).reshape(3, 1)
        inputs = ({"pair": Pair(tensor, tensor + 10), "list": [tensor + 20, ()]},)
        with patch("tools.completion_parallel.Scatter.apply", side_effect=self.cpu_scatter) as transport:
            args, kwargs = balanced_scatter_kwargs(inputs, {}, range(5))
        self.assertEqual(len(args), 3)
        self.assertEqual(kwargs, ({}, {}, {}))
        for index, part in enumerate(args):
            self.assertIsInstance(part[0]["pair"], Pair)
            self.assertEqual(part[0]["pair"].first.item(), index)
            self.assertEqual(part[0]["pair"].second.item(), index + 10)
            self.assertEqual(part[0]["list"][0].item(), index + 20)
            self.assertEqual(part[0]["list"][1], ())
        self.assertTrue(all(call.args[0] == (0, 1, 2) for call in transport.call_args_list))

    def test_no_samples_are_dropped_or_duplicated_in_backward(self):
        tensor = torch.arange(8, dtype=torch.float).reshape(8, 1).requires_grad_()
        cue = torch.zeros_like(tensor)
        with patch("tools.completion_parallel.Scatter.apply", side_effect=self.cpu_scatter):
            args, kwargs = balanced_scatter_kwargs((tensor,), {"cue": cue}, range(5))
        # A global mean, as used after gathering logits, preserves per-image weight.
        gathered = torch.cat([(arg[0] + kw["cue"]) ** 2 for arg, kw in zip(args, kwargs)])
        gathered.mean().backward()
        self.assertTrue(torch.equal(tensor.grad, 2 * tensor.detach() / 8))

    def test_invalid_or_misaligned_tensors_fail_before_any_transport(self):
        cases = [
            ((torch.zeros(8, 1),), {"cue": torch.zeros(7, 1)}, range(5)),
            ((torch.tensor(1.),), {}, range(5)),
            ((torch.empty(0, 1),), {}, range(5)),
            ((torch.zeros(8, 1),), {}, []),
            (("no tensor",), {}, range(5)),
        ]
        for args, kwargs, devices in cases:
            with self.subTest(shapes=[getattr(x, "shape", None) for x in args]):
                with patch("tools.completion_parallel.Scatter.apply") as transport:
                    with self.assertRaises(ValueError):
                        balanced_scatter_kwargs(args, kwargs, devices)
                    transport.assert_not_called()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Needs at least two physical CUDA devices")
    def test_real_cuda_transport_and_parameter_gradient(self):
        count = min(torch.cuda.device_count(), 5)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
            def forward(self, x, *, cue):
                return {"logits": x * self.scale + cue,
                        "devices": torch.full((x.shape[0],), x.device.index,
                                              dtype=torch.long, device=x.device)}
        base = Toy().to("cuda:0")
        model = BalancedDataParallel(base, device_ids=list(range(count)))
        x = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        cue = torch.full_like(x, 2.)
        output = model(x, cue=cue)
        self.assertTrue(torch.equal(output["logits"].cpu(), x + cue))
        self.assertEqual(output["devices"].unique().numel(), count)
        output["logits"].sum().backward()
        self.assertEqual(base.scale.grad.item(), x.sum().item())


if __name__ == "__main__":
    unittest.main()

