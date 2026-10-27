"""GT supervision and fixed-denominator spatial output KD for completion v6."""
from __future__ import annotations
from collections.abc import Mapping
import torch
from torch import Tensor
from torch.nn import functional as F
from models.triprd_correction_loss import segmentation_loss
from models.triprd_task_loss import _balanced_mean, _bernoulli_kl, _logits, _mask, _weight, boundary_target

COMPLETION_LOSS_SPEC = "completion_stage_gt_and_quality_weighted_output_kl_fixed_pixel_mean_v1"


def _stages(output: Mapping[str, object], key: str, target: Tensor) -> tuple[Tensor, ...]:
    values = output.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{key} must contain three stage logits")
    return tuple(_logits({"value": value}, "value", target) for value in values)


def supervised_loss(output: Mapping[str, object], mask: Tensor,
                    boundary_weight: float = .2, aux_weight: float = .5) -> dict[str, Tensor]:
    """Final + aux_weight*(mean(early masks)+coarse) + weighted all-stage edges."""
    _weight(boundary_weight, "boundary_weight")
    _weight(aux_weight, "aux_weight")
    target = _mask(mask)
    with torch.autocast(device_type=target.device.type, enabled=False):
        stages = _stages(output, "stage_logits", target)
        final = segmentation_loss(_logits(output, "logits", target), target)
        auxiliary = torch.stack([segmentation_loss(value, target) for value in stages[:-1]]).mean()
        coarse = segmentation_loss(_logits(output, "coarse_logits", target), target)
        edges = boundary_target(target)
        boundaries = _stages(output, "stage_boundaries", target)
        boundary = torch.stack([_balanced_mean(F.binary_cross_entropy_with_logits(
            value, edges, reduction="none"), (edges, 1 - edges)) for value in boundaries]).mean()
        total = final + aux_weight * (auxiliary + coarse) + boundary_weight * boundary
    return {"total": total, "mask": final, "aux": auxiliary, "coarse": coarse,
            "boundary": boundary, "segmentation": final, "auxiliary": auxiliary}


def output_distillation(student: Mapping[str, object], teacher: Mapping[str, object], mask: Tensor,
                        temperature: float = 2., boundary_weight: float = .2) -> dict[str, Tensor]:
    """Three-stage KL with detached teacher correctness weights in [0,1].

    weight = 1 - |sigmoid(teacher) - source_GT|. The denominator is the full
    pixel count, not reliable mass; poor teacher support reduces KD strength.
    Boundary KD uses the corresponding source edge label. No feature matching.
    """
    _weight(temperature, "temperature", positive=True)
    _weight(boundary_weight, "boundary_weight")
    target = _mask(mask)
    with torch.autocast(device_type=target.device.type, enabled=False):
        losses = {}
        for key, label, result_key in (("stage_logits", target, "mask_kd"),
                                       ("stage_boundaries", boundary_target(target), "boundary_kd")):
            student_stages, teacher_stages = _stages(student, key, target), _stages(teacher, key, target)
            entries = []
            for source, privileged in zip(student_stages, teacher_stages):
                privileged = privileged.detach()
                quality = (1 - (privileged.sigmoid() - label).abs()).clamp(0, 1).detach()
                entries.append((_bernoulli_kl(source, privileged, temperature) * quality).mean())
            losses[result_key] = torch.stack(entries).mean()
        losses["total"] = losses["mask_kd"] + boundary_weight * losses["boundary_kd"]
    return losses


__all__ = ["supervised_loss", "output_distillation", "COMPLETION_LOSS_SPEC"]
