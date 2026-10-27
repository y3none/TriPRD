"""Typed edge supervision for TriPRD structure v2.

The three source candidates mean thermal-image appearance continuity, relative depth
continuity, and source annotation compatibility, respectively.  They are
independent Bernoulli edges, never a categorical distribution over neighbours.
COD labels supervise its separate semantic edge predictor; they do not rename
the frozen source candidates.  These objectives are research hypotheses, not
evidence of improved COD accuracy.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .triprd import _prepare_mask, _soft_iou_loss
from .triprd_relation import local_valid_mask


OBSERVATION_SCALE = 0.1
MIN_VALID_COVERAGE = 0.99
STRUCTURE_TYPES = ("thermal", "depth", "annotation")
STRUCTURE_TARGET_SPEC = {
    "version": "typed_observation_edges_v2",
    "types": list(STRUCTURE_TYPES),
    "continuity": "exp(-channel_RMS(observation_i-observation_j)/(relative_range*scale))",
    "relative_range": "per-image channel-RMS of valid q95-q05; fallback full range, then 1 for constant images",
    "range_sampling": "deterministic valid-pixel stride, at most 4096 values per channel",
    "thermal_meaning": "pseudo-color thermal image appearance, not radiometric temperature",
    "observation_scale": OBSERVATION_SCALE,
    "observation_range": [0.0, 1.0],
    "zero_depth_invalid": True,
    "zero_thermal_invalid": False,
    "minimum_valid_coverage": MIN_VALID_COVERAGE,
    "pooling": "validity_weighted_area",
    "self_edges_supervised": False,
    "annotation": "nearest_binary_same_label_not_instance",
    "edge_balance": "per_sample_nonempty_strata",
    "training_type_weights": {name: 1.0 / 3.0 for name in STRUCTURE_TYPES},
    "selection": "mean_of_present_supervised_types",
}


def _kernel(neighbours: int) -> int:
    kernel = math.isqrt(neighbours)
    if kernel * kernel != neighbours or kernel < 3 or kernel % 2 == 0:
        raise ValueError("Edge neighbourhood must be an odd square with kernel >= 3")
    return kernel


def _neighbours(x: Tensor, kernel: int) -> Tensor:
    b, c, h, w = x.shape
    return F.unfold(x, kernel, padding=kernel // 2).reshape(b, c, kernel * kernel, h, w)


def _valid_edges(logits: Tensor, supplied: Tensor) -> Tensor:
    if logits.ndim != 5:
        raise ValueError("Expected BxCxKxHxW edge logits")
    _, _, k, h, w = logits.shape
    if supplied.shape != (1, 1, k, h, w) or supplied.dtype != torch.bool:
        raise ValueError("Expected boolean 1x1xKxHxW edge validity")
    valid = local_valid_mask(h, w, _kernel(k), device=logits.device) & supplied
    valid[:, :, k // 2] = False
    return valid


@torch.no_grad()
def annotation_targets(mask: Tensor, size: tuple[int, int], kernel: int) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Same binary annotation class, plus FG/FG, BG/BG and crossing strata.

    Background compatibility is not instance identity or a complete objectness
    label. Nearest resizing keeps the target binary at each prediction scale.
    """
    mask = (F.interpolate(_prepare_mask(mask).float(), size=size, mode="nearest") >= 0.5)
    query = mask.unsqueeze(2)
    other = _neighbours(mask.float(), kernel).bool()
    foreground = query & other
    background = ~query & ~other
    crossing = query != other
    return (~crossing).float(), (foreground, background, crossing)


@torch.no_grad()
def observed_targets(observation: Tensor, size: tuple[int, int], kernel: int, *, depth: bool = False,
                     scale: float = OBSERVATION_SCALE, validity: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Relative-contrast targets from aligned observations, not sensor full range.

    Nonfinite/out-of-range readings and zero depth are excluded before pooling.
    Thermal zero is a valid response. Invalid readings cannot contaminate
    pooled measurements; edge validity also requires valid query and neighbour.
    Depth can occupy a tiny fraction of a 16-bit container's range. A fixed
    temperature on that container scale gives nearly constant positive edges.
    Normalize differences by each image's robust valid contrast instead. For
    pseudo-color thermal images retain RGB and use channel-RMS differences.
    This definition applies to the dataset loader's resized observations.
    """
    if observation.ndim != 4 or observation.shape[1] not in (1, 3) or (depth and observation.shape[1] != 1):
        raise ValueError("Depth must be Bx1xHxW; thermal appearance can be Bx1/3xHxW")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Observation scale must be finite and positive")
    value = observation.detach().float()
    valid = (torch.isfinite(value) & (value >= 0) & (value <= 1)).all(dim=1, keepdim=True)
    if depth:
        valid &= value > 0
    if validity is not None:
        if validity.shape not in (value.shape, (value.shape[0], 1, *value.shape[-2:])):
            raise ValueError("Observation validity must match the batch/grid and have one or all channels")
        valid &= (torch.isfinite(validity) & (validity >= MIN_VALID_COVERAGE)).all(dim=1, keepdim=True)
    ranges = []
    for sample, accepted in zip(value, valid):
        pixels = sample[:, accepted[0]]
        if pixels.shape[1] == 0:
            ranges.append(value.new_tensor(1.0))
            continue
        sampled = pixels[:, ::max(1, math.ceil(pixels.shape[1] / 4096))]
        quantiles = torch.quantile(sampled, value.new_tensor([0.05, 0.95]), dim=1)
        spread = (quantiles[1] - quantiles[0]).square().mean().sqrt()
        full_range = (pixels.amax(dim=1) - pixels.amin(dim=1)).square().mean().sqrt()
        eps = torch.finfo(value.dtype).eps
        spread = torch.where(spread > eps, spread, full_range)
        ranges.append(torch.where(spread > eps, spread, torch.ones_like(spread)))
    relative_range = torch.stack(ranges)[:, None, None, None, None]
    coverage = F.interpolate(valid.float(), size=size, mode="area")
    pooled = F.interpolate(torch.where(valid, value, 0), size=size, mode="area")
    pooled = pooled / coverage.clamp_min(1e-6)
    point_valid = coverage >= MIN_VALID_COVERAGE
    difference = (pooled.unsqueeze(2) - _neighbours(pooled, kernel)).square().mean(dim=1, keepdim=True).sqrt()
    target = torch.exp(-difference / (relative_range * scale))
    edge_valid = point_valid.unsqueeze(2) & _neighbours(point_valid.float(), kernel).bool()
    grid_valid = local_valid_mask(*size, kernel, device=value.device)
    grid_valid[:, :, kernel * kernel // 2] = False
    return target, edge_valid & grid_valid


def balanced_edge_mean(values: Tensor, valid: Tensor, strata: tuple[Tensor, ...]) -> Tensor:
    """Average nonempty strata per sample/channel, then present samples.

    Empty neighbourhoods contribute a differentiable zero, not NaN. A sea of
    background pairs cannot swamp the rarer foreground/cross-boundary pairs.
    """
    valid = valid.expand_as(values)
    means, present = [], []
    for stratum in strata:
        selected = valid & stratum.expand_as(values)
        count = selected.sum((2, 3, 4))
        means.append(torch.where(selected, values, 0).sum((2, 3, 4)) / count.clamp_min(1))
        present.append(count > 0)
    means = torch.stack(means, dim=-1)
    present = torch.stack(present, dim=-1)
    sample_mean = (means * present).sum(-1) / present.sum(-1).clamp_min(1)
    sample_present = present.any(-1)
    return (sample_mean * sample_present).sum() / sample_present.sum().clamp_min(1)


def _edge_bce(logits: Tensor, target: Tensor, valid: Tensor,
              strata: tuple[Tensor, ...] | None = None) -> Tensor:
    if strata is None:
        strata = (target >= 0.5, target < 0.5)
    values = F.binary_cross_entropy_with_logits(logits.float(), target.expand_as(logits), reduction="none")
    return balanced_edge_mean(values, valid, strata)


def _segmentation(logits: Tensor, target: Tensor) -> Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits.float(), target.shape[-2:], mode="bilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(logits.float(), target) + _soft_iou_loss(logits.float(), target)


class StructureLoss(nn.Module):
    """Explicit source edges, detached Bernoulli KD, and target semantic edges.

    ``none`` disables source candidate supervision, not the trainable semantic
    graph. ``mask`` labels only source annotation edges. ``observed`` adds
    thermal/depth targets and requires real source modalities in the batch.
    """

    requires_modalities = True

    def __init__(self, supervision: str = "none", distillation_weight: float = 0.0,
                 aux_weight: float = 0.5, semantic_weight: float = 1.0,
                 structure_weight: float = 1.0) -> None:
        super().__init__()
        if supervision not in {"none", "mask", "observed"}:
            raise ValueError("supervision must be none, mask or observed")
        for name, value in {"distillation_weight": distillation_weight, "aux_weight": aux_weight,
                            "semantic_weight": semantic_weight, "structure_weight": structure_weight}.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.supervision = supervision
        self.distillation_weight = float(distillation_weight)
        self.aux_weight = float(aux_weight)
        self.semantic_weight = float(semantic_weight)
        self.structure_weight = float(structure_weight)

    def forward(self, output: Mapping[str, object], mask: Tensor,
                teacher: Mapping[str, object] | None = None, *,
                modalities: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:
        target = _prepare_mask(mask).float()
        segmentation = _segmentation(output["logits"], target)
        zero = segmentation * 0
        source = tuple(output["structure_logits"])
        masks = tuple(output["structure_valid_masks"])
        semantic = tuple(output["semantic_logits"])
        if not source or len(source) != len(masks) or len(source) != len(semantic):
            raise ValueError("Structure, semantic and valid-mask scales must match and be nonempty")
        if self.supervision == "observed" and (modalities is None or any(k not in modalities for k in ("thermal", "depth"))):
            raise ValueError("Observed supervision requires real thermal and depth source observations")
        if self.distillation_weight > 0:
            if teacher is None:
                raise ValueError("Positive distillation_weight requires teacher structure targets")
            if len(teacher["structure_logits"]) != len(source) or len(teacher["structure_valid_masks"]) != len(source):
                raise ValueError("Teacher and student structure scales must match")

        components = {name: [] for name in STRUCTURE_TYPES}
        component_present = {name: [] for name in STRUCTURE_TYPES}
        edge_errors = {name: [] for name in STRUCTURE_TYPES}
        coverage = {name: [] for name in STRUCTURE_TYPES}
        semantic_losses, kd_losses, accuracies, spreads = [], [], [], []
        for index, (logits, valid_mask, semantic_logits) in enumerate(zip(source, masks, semantic)):
            if logits.ndim != 5 or logits.shape[1] != 3:
                raise ValueError("Source structure logits must have three types: thermal, depth, annotation")
            if semantic_logits.shape != (logits.shape[0], 1, *logits.shape[2:]):
                raise ValueError("Semantic and source edge grids must match")
            valid = _valid_edges(logits, valid_mask)
            kernel = _kernel(logits.shape[2])
            annotation, strata = annotation_targets(target, tuple(logits.shape[-2:]), kernel)
            semantic_losses.append(_edge_bce(semantic_logits, annotation, valid, strata))
            with torch.no_grad():
                correct = ((semantic_logits >= 0) == (annotation >= 0.5)).float()
                accuracies.append(balanced_edge_mean(correct, valid, strata))
                # Within-type variation across valid edges, not categorical entropy.
                probability = logits.detach().float().sigmoid()
                expanded = valid.expand_as(probability)
                counts = expanded.sum((2, 3, 4)).clamp_min(1)
                mean = (probability * expanded).sum((2, 3, 4)) / counts
                variance = (((probability - mean[:, :, None, None, None]) ** 2) * expanded).sum((2, 3, 4)) / counts
                spreads.append(variance.sqrt().mean())
            if self.supervision != "none":
                components["annotation"].append(_edge_bce(logits[:, 2:3], annotation, valid, strata))
                component_present["annotation"].append(valid.any().float())
                edge_errors["annotation"].append(balanced_edge_mean(
                    (logits[:, 2:3].detach().float().sigmoid() - annotation).abs(), valid, strata))
                coverage["annotation"].append(logits.new_tensor(1. if valid.any() else 0.))
            target_valid = [valid, valid, valid]
            if self.supervision == "observed":
                for channel, name in enumerate(("thermal", "depth")):
                    observed, observation_valid = observed_targets(
                        modalities[name], tuple(logits.shape[-2:]), kernel, depth=name == "depth",
                        validity=modalities.get(f"{name}_valid"))
                    if observed.shape[0] != logits.shape[0]:
                        raise ValueError("Observation and prediction batches must match")
                    target_valid[channel] = valid & observation_valid
                    components[name].append(_edge_bce(logits[:, channel:channel + 1], observed, target_valid[channel]))
                    component_present[name].append(target_valid[channel].any().float())
                    edge_errors[name].append(balanced_edge_mean(
                        (logits[:, channel:channel + 1].detach().float().sigmoid() - observed).abs(),
                        target_valid[channel], (observed >= 0.5, observed < 0.5)))
                    coverage[name].append(target_valid[channel].sum().float() /
                        valid.expand_as(observed).sum().clamp_min(1))
            if self.distillation_weight > 0:
                teacher_logits = teacher["structure_logits"][index].detach().float()
                teacher_valid = teacher["structure_valid_masks"][index]
                if teacher_logits.shape != logits.shape or not torch.equal(teacher_valid, valid_mask):
                    raise ValueError("Teacher and student edge types, grids and masks must match")
                teacher_probability = teacher_logits.sigmoid()
                cross_entropy = F.binary_cross_entropy_with_logits(logits.float(), teacher_probability, reduction="none")
                entropy = F.binary_cross_entropy_with_logits(teacher_logits, teacher_probability, reduction="none")
                kl = (cross_entropy - entropy).clamp_min(0)
                kd_losses.append(torch.stack([
                    balanced_edge_mean(kl[:, c:c + 1], target_valid[c], strata) for c in range(3)
                ]).mean())

        mean = lambda values: torch.stack(values).mean() if values else zero
        structure_components, active_components = {}, []
        for key, values in components.items():
            if values:
                present = torch.stack(component_present[key])
                structure_components[key] = (torch.stack(values) * present).sum() / present.sum().clamp_min(1)
                active_components.append(present.any().float())
            else:
                structure_components[key] = zero
                active_components.append(zero.detach())
        active = torch.stack(active_components)
        structure = (torch.stack(list(structure_components.values())) * active).sum() / active.sum().clamp_min(1)
        # Fixed coefficients preserve the annotation objective in B vs C/D/E.
        # Missing modalities are not silently compensated by upweighting masks.
        # Validation selection separately averages actually supervised types.
        structure_objective = torch.stack(list(structure_components.values())).sum() / len(STRUCTURE_TYPES)
        affinity = mean(semantic_losses)
        kd = mean(kd_losses)
        auxiliary = mean([_segmentation(logits, target) for logits in output.get("aux_logits", ())])
        supervised = segmentation + self.aux_weight * auxiliary + self.semantic_weight * affinity + self.structure_weight * structure_objective
        return {"total": supervised + self.distillation_weight * kd,
                "supervised": supervised, "segmentation": segmentation, "auxiliary": auxiliary,
                "semantic_affinity": affinity, "structure_supervision": structure,
                "structure_training_objective": structure_objective,
                **{f"structure_{key}": value for key, value in structure_components.items()},
                **{f"structure_{key}_mae": mean(values) for key, values in edge_errors.items()},
                **{f"structure_{key}_valid_fraction": mean(values) for key, values in coverage.items()},
                "relation_kd": kd, "weighted_relation_kd": self.distillation_weight * kd,
                "semantic_balanced_accuracy": mean(accuracies), "structure_prediction_std": mean(spreads)}


__all__ = ["StructureLoss", "annotation_targets", "observed_targets", "balanced_edge_mean",
           "OBSERVATION_SCALE", "MIN_VALID_COVERAGE", "STRUCTURE_TYPES", "STRUCTURE_TARGET_SPEC"]
