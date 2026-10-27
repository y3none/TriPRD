"""Balanced DataParallel batching without changing the experiment's global batch.

PyTorch's default scatter leaves chunk sizes implicit. Explicitly distributing the
remainder guarantees five nonempty replicas for a global batch of eight:
[2, 2, 2, 1, 1]. A short final batch uses only its nonempty replicas. Every RGB,
sensor and cue tensor follows the same partition, so source KD stays aligned.

Scatter.apply is PyTorch's autograd-aware scatter used by DataParallel itself;
it preserves backward gathering and CUDA stream synchronization. This adapter
targets the project's PyTorch 2.x runtime.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn.parallel._functions import Scatter


def balanced_chunk_sizes(batch_size: int, num_devices: int) -> tuple[int, ...]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(num_devices) is not int or num_devices < 1:
        raise ValueError("num_devices must be a positive integer")
    active = min(batch_size, num_devices)
    quotient, remainder = divmod(batch_size, active)
    return tuple(quotient + (index < remainder) for index in range(active))


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def balanced_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter nested arguments with one shared, nonempty batch partition.

    Model inputs are batched tensors and immutable metadata such as task names.
    Scalar tensors and tensors with a different batch length are rejected rather
    than silently misaligning a multimodal input or its RGB-generated cue.
    """
    devices = tuple(target_gpus)
    if not devices:
        raise ValueError("At least one target GPU is required")
    tensors = list(_tensors((inputs, kwargs)))
    if not tensors:
        raise ValueError("BalancedDataParallel requires a batched tensor input")
    if type(dim) is not int or dim < 0:
        raise ValueError("The batch dimension must be a nonnegative integer")
    if tensors[0].ndim <= dim:
        raise ValueError("Scalar/unbatched tensors cannot be scattered")
    batch_size = tensors[0].shape[dim]
    chunks = balanced_chunk_sizes(batch_size, len(devices))
    devices = devices[:len(chunks)]
    if any(tensor.ndim <= dim or tensor.shape[dim] != batch_size for tensor in tensors):
        raise ValueError("All input, sensor and cue tensors must share the batch length")

    def scatter(value):
        if isinstance(value, torch.Tensor):
            return Scatter.apply(devices, chunks, dim, value)
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            parts = [scatter(item) for item in value]
            return tuple(type(value)(*(part[i] for part in parts)) for i in range(len(devices)))
        if isinstance(value, tuple):
            parts = [scatter(item) for item in value]
            return tuple(tuple(part[i] for part in parts) for i in range(len(devices)))
        if isinstance(value, list):
            parts = [scatter(item) for item in value]
            return tuple([part[i] for part in parts] for i in range(len(devices)))
        if isinstance(value, Mapping):
            parts = {key: scatter(item) for key, item in value.items()}
            return tuple(type(value)((key, part[i]) for key, part in parts.items())
                         for i in range(len(devices)))
        return tuple(value for _ in devices)

    return scatter(inputs), scatter(kwargs)


class BalancedDataParallel(torch.nn.DataParallel):
    """DataParallel with all available nonempty replicas and an unchanged batch."""

    def scatter(self, inputs, kwargs, device_ids):
        return balanced_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

