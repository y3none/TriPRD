"""Task-output and decoder-feature supervision for RGB-only COD students.

No teacher residual is a regression target. Task KD supervises the student's
actual mask logits and the shared features consumed by its task decoders.
Teacher GT-BCE advantage supplies detached reliability weights on source TRAIN.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from models.triprd_correction_loss import segmentation_loss


TASK_LOSS_SPEC = "task_output_feature_kd_v1"


def _weight(value: float, name: str, *, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _mask(mask: Tensor) -> Tensor:
    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a tensor")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1 or min(mask.shape) < 1:
        raise ValueError("mask must be nonempty Bx1xHxW")
    mask = mask.detach().float()
    divisor = torch.where(mask.amax() > 1, mask.new_tensor(255.), mask.new_tensor(1.))
    return (mask / divisor).clamp(0, 1)


def _logits(output: Mapping[str, object], key: str, target: Tensor) -> Tensor:
    value = output.get(key)
    if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[:2] != target.shape[:2]:
        raise ValueError(f"{key} must be Bx1xHxW with the mask batch size")
    if min(value.shape[-2:]) < 1 or value.device != target.device:
        raise ValueError(f"{key} must have a nonempty grid on the mask device")
    value = value.float()
    if value.shape[-2:] != target.shape[-2:]:
        value = F.interpolate(value, target.shape[-2:], mode="bilinear", align_corners=False)
    return value


def boundary_target(mask: Tensor) -> Tensor:
    """Binary 3x3 morphological gradient, without artificial image-border edges."""
    target = (_mask(mask) >= .5).float()
    dilation = F.max_pool2d(target, 3, stride=1, padding=1)
    erosion = -F.max_pool2d(-target, 3, stride=1, padding=1)
    return (dilation > erosion).float()


def _balanced_mean(values: Tensor, masses: Sequence[Tensor]) -> Tensor:
    """Equal nonempty regions per image, then equal nonempty images.

    Each region uses sum(error * reliable mass) / sum(reliable mass). A region
    or image without reliable support is omitted; an entirely empty batch
    returns a differentiable zero. No batch-wide majority class can dominate.
    """
    means, active = [], []
    for mass in masses:
        denominator = mass.sum((1, 2, 3))
        numerator = (values * mass).sum((1, 2, 3))
        means.append(numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny))
        active.append(denominator > 0)
    present = torch.stack(active, dim=1)
    per_image = (torch.stack(means, dim=1) * present).sum(1) / present.sum(1).clamp_min(1)
    valid_images = present.any(1)
    return (per_image * valid_images).sum() / valid_images.sum().clamp_min(1)


def supervised_loss(output: Mapping[str, object], mask: Tensor,
                    boundary_weight: float = .2, aux_weight: float = .5) -> dict[str, Tensor]:
    """Weighted BCE+IoU masks, auxiliary masks, and class-balanced edge BCE."""
    _weight(boundary_weight, "boundary_weight")
    _weight(aux_weight, "aux_weight")
    target = _mask(mask)
    with torch.autocast(device_type=target.device.type, enabled=False):
        logits = _logits(output, "logits", target)
        segmentation = segmentation_loss(logits, target)
        aux = output.get("aux_logits", ())
        if not isinstance(aux, (tuple, list)):
            raise ValueError("aux_logits must be a tuple or list of mask logits")
        auxiliary = torch.stack([
            segmentation_loss(_logits({"logits": value}, "logits", target), target)
            for value in aux
        ]).mean() if aux else segmentation * 0
        edges = boundary_target(target)
        edge_errors = F.binary_cross_entropy_with_logits(
            _logits(output, "boundary_logits", target), edges, reduction="none")
        boundary = _balanced_mean(edge_errors, (edges, 1 - edges))
        total = segmentation + aux_weight * auxiliary + boundary_weight * boundary
    return {"total": total, "segmentation": segmentation, "boundary": boundary, "auxiliary": auxiliary}


def teacher_loss(coalitions: Mapping[str, Mapping[str, object]], mask: Tensor,
                 boundary_weight: float = .2, aux_weight: float = .5) -> dict[str, Tensor]:
    """Supervise R and RTD equally, including for a capacity-matched RGB teacher."""
    if set(coalitions) != {"R", "RTD"}:
        raise ValueError("teacher_loss requires exactly R and RTD coalitions")
    entries = [supervised_loss(coalitions[name], mask, boundary_weight, aux_weight)
               for name in ("R", "RTD")]
    return {key: torch.stack([entry[key] for entry in entries]).mean() for key in entries[0]}


def _bernoulli_kl(student: Tensor, teacher: Tensor, temperature: float) -> Tensor:
    """Float32 KL(teacher || student), with T^2 and roundoff-negative clipping."""
    s, t = student.float() / temperature, teacher.detach().float() / temperature
    probability = t.sigmoid()
    divergence = probability * (F.logsigmoid(t) - F.logsigmoid(s))
    divergence = divergence + (1 - probability) * (F.logsigmoid(-t) - F.logsigmoid(-s))
    return divergence.clamp_min(0) * (temperature * temperature)


def _region_masses(target: Tensor, reliability: Tensor) -> tuple[Tensor, ...]:
    foreground = (target >= .5).float()
    edge = boundary_target(target)
    # Disjoint source-pixel regions. Pool their joint reliable mass, never
    # multiply separately pooled reliability and labels (which invents support).
    return (reliability * foreground * (1 - edge),
            reliability * (1 - foreground) * (1 - edge), reliability * edge)


def _features(output: Mapping[str, object], target: Tensor) -> tuple[Tensor, ...]:
    values = output.get("task_features")
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("task KD requires a nonempty task_features tuple or list")
    for value in values:
        if (not isinstance(value, Tensor) or value.ndim != 4 or value.shape[0] != target.shape[0]
                or min(value.shape) < 1 or value.device != target.device):
            raise ValueError("task_features must be nonempty BxCxHxW tensors on the mask device")
    return tuple(values)


def transfer_loss(student: Mapping[str, object], teacher_full: Mapping[str, object], mask: Tensor,
                  teacher_rgb: Mapping[str, object] | None = None, mode: str = "ordinary",
                  temperature: float = 2., feature_weight: float = 1.) -> dict[str, Tensor]:
    """Ordinary output KD or reliability/region-balanced output and feature KD.

    For ``task``, reliability = max(BCE(R, GT) - BCE(RTD, GT), 0), evaluated
    without temperature and detached from both teachers. With no RGB reference
    all pixels have unit mass; this is an explicit unfiltered control, not a
    measured teacher advantage. Feature vectors are normalized over channels;
    squared Euclidean distance is averaged equally over feature scales.
    ``ordinary`` uses only spatially averaged mask KL and ignores features and
    RGB reference. Teacher tensors never receive gradients.
    """
    if mode not in {"ordinary", "task"}:
        raise ValueError("mode must be 'ordinary' or 'task'")
    _weight(temperature, "temperature", positive=True)
    _weight(feature_weight, "feature_weight")
    target = _mask(mask)
    with torch.autocast(device_type=target.device.type, enabled=False):
        s = _logits(student, "logits", target)
        t = _logits(teacher_full, "logits", target).detach()
        errors = _bernoulli_kl(s, t, temperature)
        if mode == "ordinary":
            kd_output = errors.mean()
            kd_feature = s.sum() * 0
            coverage = target.new_ones(())
        else:
            with torch.no_grad():
                if teacher_rgb is None:
                    reliability = torch.ones_like(target)
                else:
                    r = _logits(teacher_rgb, "logits", target)
                    reliability = (F.binary_cross_entropy_with_logits(r, target, reduction="none")
                                   - F.binary_cross_entropy_with_logits(t, target, reduction="none")).clamp_min(0)
                masses = _region_masses(target, reliability)
                coverage = (reliability > 0).float().mean()
            kd_output = _balanced_mean(errors, masses)
            student_features, teacher_features = _features(student, target), _features(teacher_full, target)
            if len(student_features) != len(teacher_features):
                raise ValueError("student and teacher task-feature scale counts must match")
            feature_losses = []
            for sf, tf in zip(student_features, teacher_features):
                if sf.shape[:2] != tf.shape[:2]:
                    raise ValueError("corresponding student and teacher feature channels must match")
                sf, tf = sf.float(), tf.detach().float()
                if sf.shape[-2:] != tf.shape[-2:]:
                    tf = F.interpolate(tf, sf.shape[-2:], mode="bilinear", align_corners=False)
                distance = (F.normalize(sf, dim=1, eps=1e-6)
                            - F.normalize(tf, dim=1, eps=1e-6)).square().sum(1, keepdim=True)
                scale_masses = tuple(F.interpolate(mass, sf.shape[-2:], mode="area")
                                     for mass in masses)
                feature_losses.append(_balanced_mean(distance, scale_masses))
            kd_feature = torch.stack(feature_losses).mean()
        total = kd_output + feature_weight * kd_feature
    return {"total": total, "kd_output": kd_output, "kd_feature": kd_feature,
            "reliable_coverage": coverage}
