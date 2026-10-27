"""Explicit losses and held-out probes for RGB structural correction transfer.

Edges describe binary annotation agreement, not instance identity. Teachers
provide q_RTD-q_R in probability space. Reliability is improvement in GT BCE;
it is not a confidence score or an identifiable causal effect. Probe constants
must be fitted on source TRAIN data and then fixed during validation.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


COALITIONS = ("R", "RT", "RD", "RTD")
_EPS = 1e-6


def _mask(mask: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be Bx1xHxW")
    mask = mask.detach().float()
    # No host synchronization for the common 0/1 versus uint8 mask convention.
    divisor = torch.where(mask.amax() > 1, mask.new_tensor(255.), mask.new_tensor(1.))
    return (mask / divisor).clamp(0, 1)


def _kernel(kernel: int) -> None:
    if kernel < 3 or kernel % 2 != 1:
        raise ValueError("relation_kernel must be an odd integer >= 3")


def _edges(value: Tensor, kernel: int) -> Tensor:
    if value.ndim == 4:
        value = value.unsqueeze(1)
    if value.ndim != 5 or value.shape[1] != 1 or value.shape[2] != kernel * kernel:
        raise ValueError("relations must be Bx1xKxHxW (or BxKxHxW), K=kernel squared")
    return value.float()


def _sequence(values: Sequence[Tensor] | Tensor) -> tuple[Tensor, ...]:
    result = (values,) if isinstance(values, Tensor) else tuple(values)
    if not result:
        raise ValueError("at least one relation scale is required")
    return result


def _neighbours(value: Tensor, kernel: int) -> Tensor:
    b, c, h, w = value.shape
    return F.unfold(value, kernel, padding=kernel // 2).reshape(b, c, kernel * kernel, h, w)


@torch.no_grad()
def annotation_relations(mask: Tensor, size: tuple[int, int], relation_kernel: int = 3,
                         supplied_valid: Tensor | None = None) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    """Return agreement, valid off-diagonal edges, and FG/BG/crossing strata."""
    _kernel(relation_kernel)
    label = F.interpolate(_mask(mask), size=size, mode="nearest") >= 0.5
    centre = label.unsqueeze(2)
    nearby = _neighbours(label.float(), relation_kernel).bool()
    valid = _neighbours(torch.ones(1, 1, *size, device=mask.device), relation_kernel).bool()
    valid[:, :, relation_kernel * relation_kernel // 2] = False
    if supplied_valid is not None:
        if supplied_valid.ndim == 4:
            supplied_valid = supplied_valid.unsqueeze(1)
        if supplied_valid.shape != valid.shape or supplied_valid.dtype != torch.bool:
            raise ValueError("validity must be boolean 1x1xKxHxW")
        valid = valid & supplied_valid
    strata = (centre & nearby, ~centre & ~nearby, centre != nearby)
    return (centre == nearby).float(), valid, strata


def _balanced(values: Tensor, valid: Tensor, strata: tuple[Tensor, ...],
              weights: Tensor | None = None) -> Tensor:
    """Equal nonempty strata per image, then equal nonempty images.

    Reliability-weighted losses normalize by the reliability mass within each
    stratum. Empty or entirely nonpositive-reliability strata contribute zero.
    """
    valid = valid.expand_as(values)
    mass = valid.to(values.dtype) if weights is None else weights.detach().to(values.dtype).masked_fill(~valid, 0)
    means, active = [], []
    for stratum in strata:
        selected = mass * stratum.to(values.dtype)
        denominator = selected.sum((2, 3, 4))
        numerator = (values * selected).sum((2, 3, 4))
        means.append(numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny))
        active.append(denominator > 0)
    present = torch.stack(active, -1)
    sample = (torch.stack(means, -1) * present).sum(-1) / present.sum(-1).clamp_min(1)
    valid_samples = present.any(-1)
    return (sample * valid_samples).sum() / valid_samples.sum().clamp_min(1)


def segmentation_loss(logits: Tensor, mask: Tensor, boundary_weight: float = 5.,
                      boundary_kernel: int = 31) -> Tensor:
    """Boundary-weighted BCE + weighted soft IoU, averaged per image."""
    if not math.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("boundary_weight must be finite and nonnegative")
    if boundary_kernel < 1 or boundary_kernel % 2 != 1:
        raise ValueError("boundary_kernel must be positive odd")
    target = _mask(mask)
    if logits.ndim != 4 or logits.shape[:2] != target.shape[:2]:
        raise ValueError("segmentation logits and mask must share Bx1 dimensions")
    logits = logits.float()
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    local = F.avg_pool2d(target, boundary_kernel, stride=1, padding=boundary_kernel // 2, count_include_pad=False)
    weight = 1 + boundary_weight * (local - target).abs()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (weight * bce).sum((1, 2, 3)) / weight.sum((1, 2, 3))
    probability = logits.sigmoid()
    intersection = (weight * probability * target).sum((1, 2, 3))
    union = (weight * (probability + target - probability * target)).sum((1, 2, 3))
    return (bce + 1 - (intersection + 1) / (union + 1)).mean()


def relation_supervision(logits_seq: Sequence[Tensor] | Tensor, mask: Tensor,
                         relation_kernel: int = 3, valid_masks: Sequence[Tensor] | None = None) -> Tensor:
    _kernel(relation_kernel)
    logits_seq = _sequence(logits_seq)
    if valid_masks is not None and len(valid_masks) != len(logits_seq):
        raise ValueError("valid-mask and relation scale counts must match")
    losses = []
    for index, raw in enumerate(logits_seq):
        logits = _edges(raw, relation_kernel)
        supplied = None if valid_masks is None else valid_masks[index]
        target, valid, strata = annotation_relations(mask, tuple(logits.shape[-2:]), relation_kernel, supplied)
        if logits.shape != target.shape:
            raise ValueError("relation batch/grid must match annotation targets")
        losses.append(_balanced(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid, strata))
    return torch.stack(losses).mean()


def _supervised(output: Mapping[str, object], mask: Tensor, relation_kernel: int,
                relation_weight: float, aux_weight: float, *, supervise_base: bool = True) -> dict[str, Tensor]:
    if not all(math.isfinite(weight) and weight >= 0 for weight in (relation_weight, aux_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    segmentation = segmentation_loss(output["logits"], mask)
    auxiliary = tuple(output.get("aux_logits", ()))
    auxiliary = torch.stack([segmentation_loss(x, mask) for x in auxiliary]).mean() if auxiliary else segmentation * 0
    valid = output.get("relation_valid_masks")
    relation = relation_supervision(output["relation_logits"], mask, relation_kernel, valid)
    base = output.get("base_relation_logits")
    base_relation = relation_supervision(base, mask, relation_kernel, valid) if base is not None else relation
    # Students supervise both predictions with a fixed total coefficient. The
    # teacher's RGB base is a reference shared by every coalition; repeating its
    # loss here would give R five times each other coalition's relation weight.
    relation_objective = (relation + base_relation) * 0.5 if supervise_base and base is not None else relation
    supervised = segmentation + aux_weight * auxiliary + relation_weight * relation_objective
    return {"supervised": supervised, "segmentation": segmentation, "auxiliary": auxiliary,
            "relation": relation, "base_relation": base_relation}


def teacher_loss(coalitions: Mapping[str, Mapping[str, object]], mask: Tensor,
                 relation_kernel: int = 3, relation_weight: float = 1., aux_weight: float = 0.5) -> dict[str, Tensor]:
    """Four equally supervised coalitions, each with its own relation objective.

    The shared RGB base is diagnostic only here. It already receives its full
    one-quarter relation weight from coalition R.
    """
    if set(coalitions) != set(COALITIONS):
        raise ValueError("teacher_loss requires exactly R, RT, RD and RTD coalitions")
    losses = [_supervised(coalitions[name], mask, relation_kernel, relation_weight,
                          aux_weight, supervise_base=False) for name in COALITIONS]
    result = {key: torch.stack([entry[key] for entry in losses]).mean() for key in losses[0]}
    result["total"] = result["supervised"]
    return result


@torch.no_grad()
def correction_targets(teacher_rgb: Mapping[str, object], teacher_full: Mapping[str, object], mask: Tensor,
                       relation_kernel: int = 3) -> dict[str, tuple[Tensor, ...]]:
    """Detached q_RTD-q_R and positive GT-BCE advantage, excluding invalid edges."""
    rgb, full = _sequence(teacher_rgb["relation_logits"]), _sequence(teacher_full["relation_logits"])
    if len(rgb) != len(full):
        raise ValueError("teacher coalition scale counts must match")
    fields = {name: [] for name in ("delta", "reliability", "valid", "target", "base_probability", "full_probability")}
    for index, (r, f) in enumerate(zip(rgb, full)):
        r, f = _edges(r, relation_kernel), _edges(f, relation_kernel)
        if r.shape != f.shape:
            raise ValueError("teacher coalition relation grids must match")
        target, valid, _ = annotation_relations(mask, tuple(r.shape[-2:]), relation_kernel)
        if target.shape != r.shape:
            raise ValueError("teacher batch must match masks")
        for teacher in (teacher_rgb, teacher_full):
            supplied = teacher.get("relation_valid_masks")
            if supplied is not None:
                if len(supplied) != len(rgb):
                    raise ValueError("teacher valid-mask scale count must match")
                _, available, _ = annotation_relations(mask, tuple(r.shape[-2:]), relation_kernel, supplied[index])
                valid = valid & available
        qr, qf = r.sigmoid(), f.sigmoid()
        advantage = (F.binary_cross_entropy_with_logits(r, target, reduction="none") -
                     F.binary_cross_entropy_with_logits(f, target, reduction="none")).clamp_min(0)
        values = {"delta": (qf - qr).masked_fill(~valid, 0), "reliability": advantage.masked_fill(~valid, 0),
                  "valid": valid, "target": target, "base_probability": qr, "full_probability": qf}
        for key, value in values.items():
            fields[key].append(value.detach())
    return {key: tuple(value) for key, value in fields.items()}


def source_loss(output: Mapping[str, object], mask: Tensor,
                teacher_rgb: Mapping[str, object] | None = None, teacher_full: Mapping[str, object] | None = None,
                mode: str = "none", relation_kernel: int = 3, relation_weight: float = 1.,
                kd_weight: float = 1., alignment: Tensor | float | None = None,
                aux_weight: float = 0.5) -> dict[str, Tensor]:
    """GT supervision plus ordinary Bernoulli KD or reliable probability-residual KD.

    ``transfer`` is the unscaled loss for gradient probing. ``weighted_transfer``
    applies kd_weight and detached alignment. Source GT supervision is never gated.
    Ordinary KD uses final relation_logits, i.e. the same correction route, not
    a separate bypass head. Reliable correction uses stratum-balanced SmoothL1.
    """
    if mode not in ("none", "ordinary", "correction"):
        raise ValueError("mode must be none, ordinary or correction")
    if not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("kd_weight must be finite and nonnegative")
    result = _supervised(output, mask, relation_kernel, relation_weight, aux_weight)
    zero = result["supervised"] * 0
    transfer, coverage, reliability_mean = zero, zero.detach(), zero.detach()
    if mode != "none":
        if teacher_rgb is None or teacher_full is None:
            raise ValueError("distillation requires RGB and full teacher coalitions")
        targets = correction_targets(teacher_rgb, teacher_full, mask, relation_kernel)
        predicted = _sequence(output["relation_logits"] if mode == "ordinary" else output["corrections"])
        if len(predicted) != len(targets["delta"]):
            raise ValueError("student/teacher scale counts must match")
        losses, positive, valid_count, reliability_sums = [], [], [], []
        for index, raw in enumerate(predicted):
            prediction = _edges(raw, relation_kernel)
            target, valid = targets["target"][index], targets["valid"][index]
            if prediction.shape != target.shape:
                raise ValueError("student/teacher relation grids must match")
            supplied = output.get("relation_valid_masks")
            if supplied is not None and len(supplied) != len(predicted):
                raise ValueError("student valid-mask scale count must match")
            _, student_valid, strata = annotation_relations(
                mask, tuple(prediction.shape[-2:]), relation_kernel, None if supplied is None else supplied[index])
            valid = valid & student_valid
            reliability = targets["reliability"][index]
            if mode == "ordinary":
                probability = targets["full_probability"][index]
                entropy = -(probability * probability.clamp_min(_EPS).log() +
                            (1 - probability) * (1 - probability).clamp_min(_EPS).log())
                kl = (F.binary_cross_entropy_with_logits(prediction, probability, reduction="none") - entropy).clamp_min(0)
                losses.append(_balanced(kl, valid, strata))
            else:
                error = F.smooth_l1_loss(prediction, targets["delta"][index], reduction="none", beta=0.1)
                losses.append(_balanced(error, valid, strata, reliability))
            valid_expanded = valid.expand_as(target)
            positive.append(((reliability > 0) & valid_expanded).sum().float())
            valid_count.append(valid_expanded.sum().float())
            reliability_sums.append(reliability.masked_fill(~valid, 0).sum())
        transfer = torch.stack(losses).mean()
        count = torch.stack(valid_count).sum().clamp_min(1)
        coverage = torch.stack(positive).sum() / count
        reliability_mean = torch.stack(reliability_sums).sum() / count
    gate = result["supervised"].new_tensor(1.) if alignment is None else torch.as_tensor(
        alignment, device=result["supervised"].device, dtype=torch.float32).detach()
    if gate.numel() != 1:
        raise ValueError("alignment must be scalar")
    gate = gate.reshape(()).clamp(0, 1)
    weighted = kd_weight * gate * transfer
    result.update(total=result["supervised"] + weighted, transfer=transfer, weighted_transfer=weighted,
                  ordinary_kd=transfer if mode == "ordinary" else zero,
                  correction_kd=transfer if mode == "correction" else zero,
                  positive_coverage=coverage.detach(), reliability_mean=reliability_mean.detach(), alignment=gate)
    return result


def positive_gradient_alignment(source_loss: Tensor, cod_train_loss: Tensor,
                                adapter_parameters: Iterable[Tensor], retain_graph: bool = True,
                                gradient_scale: float = 1.) -> Tensor:
    """Detached positive cosine on shared trainable parameters, without higher-order AD.

    The caller MUST supply a COD training-batch loss, never validation/test loss.
    This function cannot infer data provenance. It leaves parameter .grad intact.
    AMP callers pass their GradScaler scale: scale before AD, then unscale each
    gradient in float32. Nonfinite probes raise, rather than pretending rejection.
    """
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be finite and positive")
    parameters = tuple(parameter for parameter in adapter_parameters if parameter.requires_grad)
    if source_loss.numel() != 1 or cod_train_loss.numel() != 1:
        raise ValueError("gradient alignment needs scalar losses")
    zero = source_loss.detach().float().new_zeros(())
    if not parameters or not source_loss.requires_grad or not cod_train_loss.requires_grad:
        return zero
    source = torch.autograd.grad(source_loss * gradient_scale, parameters, retain_graph=retain_graph, create_graph=False, allow_unused=True)
    target = torch.autograd.grad(cod_train_loss * gradient_scale, parameters, retain_graph=retain_graph, create_graph=False, allow_unused=True)
    probes = [gradient for gradient in (*source, *target) if gradient is not None]
    # One aggregate device-to-host check per probe, not one synchronization per parameter.
    if probes and not torch.stack([torch.isfinite(gradient).all() for gradient in probes]).all():
        raise FloatingPointError("Nonfinite gradient-alignment probe; check AMP gradient scale and source/COD losses")
    dots, source_squares, target_squares = [], [], []
    for first, second in zip(source, target):
        first = None if first is None else first.detach().float() / gradient_scale
        second = None if second is None else second.detach().float() / gradient_scale
        if first is not None:
            source_squares.append(first.square().sum())
        if second is not None:
            target_squares.append(second.square().sum())
        if first is not None and second is not None:
            dots.append((first * second).sum())
    if not dots or not source_squares or not target_squares:
        return zero
    dot = torch.stack(dots).sum()
    norm = (torch.stack(source_squares).sum() * torch.stack(target_squares).sum()).sqrt()
    return torch.where(norm > 0, dot / norm.clamp_min(torch.finfo(dot.dtype).tiny), zero).clamp(0, 1).detach()


@torch.no_grad()
def constant_correction_statistics(targets: Mapping[str, Sequence[Tensor]]) -> dict[str, tuple[Tensor, ...]]:
    """TRAIN-only constants on all edges and separately on reliable edges.

    The positive-support mean is the constant minimizer of the unweighted
    positive-edge MSE used by the learnability gate. An all-edge mean is not
    a sufficient baseline for that restricted metric.
    """
    sums, counts, positive_sums, positive_counts = [], [], [], []
    for delta, valid, reliability in zip(targets["delta"], targets["valid"], targets["reliability"]):
        selected = valid.expand_as(delta)
        positive = selected & (reliability > 0)
        sums.append(delta.masked_fill(~selected, 0).sum((0, 3, 4), keepdim=True))
        counts.append(selected.sum((0, 3, 4), keepdim=True).to(delta.dtype))
        positive_sums.append(delta.masked_fill(~positive, 0).sum((0, 3, 4), keepdim=True))
        positive_counts.append(positive.sum((0, 3, 4), keepdim=True).to(delta.dtype))
    return {"sum": tuple(sums), "count": tuple(counts),
            "positive_sum": tuple(positive_sums), "positive_count": tuple(positive_counts)}


@torch.no_grad()
def finalize_constant_correction(statistics: Mapping[str, Sequence[Tensor]], *, positive_only: bool = False) -> tuple[Tensor, ...]:
    prefix = "positive_" if positive_only else ""
    sums, counts = statistics[prefix + "sum"], statistics[prefix + "count"]
    if len(sums) != len(counts) or not sums:
        raise ValueError("constant statistics must have matching nonempty scales")
    return tuple(total / count.clamp_min(1) for total, count in zip(sums, counts))


@torch.no_grad()
def correction_probe(output: Mapping[str, object], targets: Mapping[str, Sequence[Tensor]],
                     constant_correction: Sequence[Tensor], relation_kernel: int = 3, *,
                     positive_constant_correction: Sequence[Tensor]) -> dict[str, Tensor]:
    """Additive raw-edge statistics; task improvement isolates q_R + predicted delta.

    Both TRAIN-fitted constant baselines are mandatory inputs. This never fits
    constants on probe data. ``student_task_error`` measures the actual student relation
    prediction; q_R+delta is a diagnostic, not a deployable student prediction.
    """
    predicted = _sequence(output["corrections"])
    actual_relations = _sequence(output["relation_logits"])
    if not (len(predicted) == len(targets["delta"]) == len(constant_correction) ==
            len(positive_constant_correction) == len(actual_relations)):
        raise ValueError("probe scale counts must match, including train-fitted constants")
    records = []
    for index, raw in enumerate(predicted):
        prediction = _edges(raw.detach(), relation_kernel)
        delta, valid = targets["delta"][index], targets["valid"][index]
        if prediction.shape != delta.shape:
            raise ValueError("probe prediction/target grids must match")
        constant = torch.as_tensor(constant_correction[index], device=prediction.device, dtype=prediction.dtype)
        positive_constant = torch.as_tensor(positive_constant_correction[index], device=prediction.device, dtype=prediction.dtype)
        if any(value.shape not in ((), (1, 1, relation_kernel * relation_kernel, 1, 1))
               for value in (constant, positive_constant)):
            raise ValueError("constant must be scalar or 1x1xKx1x1, fixed across images/pixels")
        selected = valid.expand_as(delta)
        positive = selected & (targets["reliability"][index] > 0)
        squared, zero_squared, constant_squared = (prediction - delta).square(), delta.square(), (constant - delta).square()
        sum_valid = lambda value: value.masked_fill(~selected, 0).sum()
        sum_positive = lambda value: value.masked_fill(~positive, 0).sum()
        label = targets["target"][index]
        base = targets["base_probability"][index]
        bce = lambda probability: F.binary_cross_entropy(probability.clamp(_EPS, 1 - _EPS), label, reduction="none")
        base_error = bce(base)
        corrected_error = bce(base + prediction)
        actual = _edges(actual_relations[index], relation_kernel)
        if actual.shape != label.shape:
            raise ValueError("student task-relation shape must match targets")
        records.append({"count": selected.sum().float(), "squared_error_sum": sum_valid(squared),
                        "zero_squared_error_sum": sum_valid(zero_squared), "constant_squared_error_sum": sum_valid(constant_squared),
                        "positive_count": positive.sum().float(), "positive_squared_error_sum": sum_positive(squared),
                        "positive_zero_squared_error_sum": sum_positive(zero_squared),
                        "positive_constant_squared_error_sum": sum_positive((positive_constant - delta).square()),
                        "positive_all_edge_constant_squared_error_sum": sum_positive(constant_squared),
                        "base_task_error_sum": sum_valid(base_error), "corrected_task_error_sum": sum_valid(corrected_error),
                        "constant_task_error_sum": sum_valid(bce(base + constant)),
                        "teacher_task_error_sum": sum_valid(bce(targets["full_probability"][index])),
                        "student_task_error_sum": sum_valid(F.binary_cross_entropy_with_logits(actual, label, reduction="none")),
                        "task_improvement_sum": sum_valid(base_error - corrected_error)})
    return {key: torch.stack([record[key] for record in records]).sum().detach() for key in records[0]}


@torch.no_grad()
def finalize_correction_probe(statistics: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Finalize aggregated statistics once, after every source-validation batch."""
    count, positive = statistics["count"].clamp_min(1), statistics["positive_count"].clamp_min(1)
    fields = {"mse": "squared_error_sum", "zero_mse": "zero_squared_error_sum", "constant_mse": "constant_squared_error_sum",
              "base_task_error": "base_task_error_sum", "corrected_task_error": "corrected_task_error_sum",
              "constant_task_error": "constant_task_error_sum", "teacher_task_error": "teacher_task_error_sum",
              "student_task_error": "student_task_error_sum", "improved_task_error": "task_improvement_sum"}
    result = {key: statistics[value] / count for key, value in fields.items()}
    for name in ("mse", "zero_mse", "constant_mse"):
        result[f"positive_{name}"] = statistics["positive_" + fields[name]] / positive
    result["positive_all_edge_constant_mse"] = statistics["positive_all_edge_constant_squared_error_sum"] / positive
    result.update(positive_coverage=statistics["positive_count"] / count,
                  count=statistics["count"], positive_count=statistics["positive_count"])
    return result


__all__ = ["COALITIONS", "annotation_relations", "segmentation_loss", "relation_supervision", "teacher_loss", "source_loss",
           "correction_targets", "positive_gradient_alignment", "constant_correction_statistics", "finalize_constant_correction",
           "correction_probe", "finalize_correction_probe"]
