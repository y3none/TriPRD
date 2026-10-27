"""Source-only regional MSE control and read-only source validation.

The positive-edge criterion, squared error and scale/image pooling match the
existing positive MSE control. Only the final FG/BG/crossing weights change.
No production loss, teacher target, checkpoint selection or gate is modified.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from models.triprd_correction_loss import (
    _edges, _sequence, annotation_relations, correction_probe,
    correction_targets, finalize_correction_probe,
)
from tools.correction_path_diagnostics import DiagnosticAccumulator, _inspect_relations


STRATA = ("fg", "bg", "crossing")
POSITIVE_BCE_FIELDS = (
    "positive_base_bce", "positive_corrected_bce", "positive_teacher_rgb_bce",
    "positive_teacher_full_bce", "positive_teacher_rgb_plus_student_bce",
    "positive_student_base_plus_teacher_delta_bce",
)


def positive_stratified_mse_loss(output, targets, mask, relation_kernel=3):
    """Mean of nonempty FG/BG/crossing positive-edge MSEs in this minibatch.

    Each stratum pools SSE and edge count across all images and scales before
    division. Reliability selects strictly positive edges; its magnitude never
    weights them. Geometry, supplied student validity and detached teacher
    validity are intersected exactly as in ``positive_mse_loss``. Empty strata
    are omitted, and an entirely empty minibatch returns a differentiable zero.
    """
    predicted = _sequence(output["corrections"])
    supplied = output.get("relation_valid_masks")
    if any(len(targets[name]) != len(predicted) for name in ("delta", "valid", "reliability")) or (
            supplied is not None and len(supplied) != len(predicted)):
        raise ValueError("student/teacher validity and scale counts must match")
    squared_by_scale, count_by_scale = [], []
    for index, raw in enumerate(predicted):
        value = _edges(raw, relation_kernel)
        delta = targets["delta"][index].detach()
        if value.shape != delta.shape:
            raise ValueError("student/teacher correction grids must match")
        _, valid, strata = annotation_relations(
            mask, tuple(value.shape[-2:]), relation_kernel,
            None if supplied is None else supplied[index])
        positive = (valid & targets["valid"][index].detach().bool()
                    & (targets["reliability"][index].detach() > 0))
        squared = (value - delta).square()
        selections = tuple(positive & stratum for stratum in strata)
        squared_by_scale.append(torch.stack([
            squared.masked_fill(~selected, 0).sum() for selected in selections]))
        count_by_scale.append(torch.stack([
            selected.expand_as(value).sum() for selected in selections]))
    squared = torch.stack(squared_by_scale).sum(0)
    count = torch.stack(count_by_scale).sum(0)
    present = count > 0
    return (squared / count.clamp_min(1)).sum() / present.sum().clamp_min(1)


@torch.no_grad()
def _inspect_positive_relation_errors(output, targets, mask, kernel, acc):
    """Compare bases and counterfactuals on exactly the positive-MSE support.

    Called inside disabled autocast, reusing the same normal output and teacher
    targets as the existing full-edge inspector. No extra forward or data pass.
    """
    for index, raw in enumerate(output["corrections"]):
        raw = raw.detach().float()
        target, valid, strata = annotation_relations(
            mask, tuple(raw.shape[-2:]), kernel, output["relation_valid_masks"][index])
        selected = (valid.expand_as(raw) & targets["valid"][index].detach().bool()
                    & (targets["reliability"][index].detach() > 0))
        base_logits = output["base_relation_logits"][index].detach().float()
        corrected_logits = output["relation_logits"][index].detach().float()
        base = base_logits.sigmoid()
        teacher_base = targets["base_probability"][index].detach().float()
        teacher_full = targets["full_probability"][index].detach().float()
        delta = targets["delta"][index].detach().float()
        bce = lambda probability: F.binary_cross_entropy(
            probability.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        values = {
            "positive_base_bce": F.binary_cross_entropy_with_logits(base_logits, target, reduction="none"),
            "positive_corrected_bce": F.binary_cross_entropy_with_logits(corrected_logits, target, reduction="none"),
            "positive_teacher_rgb_bce": bce(teacher_base),
            "positive_teacher_full_bce": bce(teacher_full),
            "positive_teacher_rgb_plus_student_bce": bce(teacher_base + raw),
            "positive_student_base_plus_teacher_delta_bce": bce(base + delta),
        }
        memberships = {"all": torch.ones_like(selected), **dict(zip(STRATA, strata))}
        for stratum, membership in memberships.items():
            keep = selected & membership
            for scale in (f"stride{8 * 2 ** index}", "pooled"):
                for name, value in values.items():
                    acc.add(f"VDT/recovery/{scale}/{stratum}/{name}", value, keep)


@torch.no_grad()
def evaluate_source_detailed(model, teacher, loader, constants, device, a):
    """Evaluate normal source outputs once per batch, without interventions.

    Returns the original globally pooled ``source_recoverability`` metrics and
    the flat-key ``path_statistics`` from the existing relation inspector.
    For each scale and pooled stratum these include full/positive edge support,
    positive coverage, recovery baselines, actual relation GT BCE and the
    student-base-plus-teacher-delta counterfactual, on both full and positive
    recovery support. Constants must already have been fitted on TRAIN. All
    module training flags are restored on exit.
    """
    from tools import run_triprd_correction as runner

    device = torch.device(device)
    constants = {
        name: tuple(torch.as_tensor(value, device=device).detach() for value in values)
        for name, values in constants.items()
    }
    flags = [(module, module.training) for root in (model, teacher) for module in root.modules()]
    stats, acc = {}, DiagnosticAccumulator()
    try:
        model.eval()
        teacher.eval()
        for batch in loader:
            batch = runner.shared._move(batch, device)
            with runner._autocast(a, device):
                coals = teacher.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
                prediction = model(batch["rgb"], task="sod")
            # Both existing probes contain probability-space BCE. Disable any
            # caller's enclosing autocast as well as this function's context.
            with torch.autocast(device_type=device.type, enabled=False):
                targets = correction_targets(coals["R"], coals["RTD"], batch["mask"], a.relation_kernel)
                runner._add_stats(stats, correction_probe(
                    prediction, targets, constants["all"], a.relation_kernel,
                    positive_constant_correction=constants["positive"]))
                _inspect_relations(prediction, batch["mask"], a.relation_kernel,
                                   acc, "VDT", targets, constants)
                _inspect_positive_relation_errors(prediction, targets, batch["mask"], a.relation_kernel, acc)
        if not stats:
            raise ValueError("source validation requires at least one batch")
        result = {name: float(value) for name, value in finalize_correction_probe(stats).items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Nonfinite source recoverability probe")
        return {"source_recoverability": result, "path_statistics": acc.result()}
    finally:
        for module, training in flags:
            module.training = training


def source_summary(validation):
    """Compact pooled source rows named global/fg/bg/crossing.

    ``count`` and ``positive_count`` use teacher-valid source recovery support;
    shares use the corresponding global support. Relation BCE uses the normal
    student's valid GT edges, as specified by the existing path inspector.
    Positive-prefixed BCE fields instead use precisely the positive-MSE support.
    Missing support is represented by ``None`` means/shares, never a false zero.
    """
    statistics = validation["path_statistics"]

    def metric(section, stratum, name, field="mean"):
        return statistics[f"VDT/{section}/pooled/{stratum}/{name}"][field]

    full_count = metric("recovery", "all", "mse", "count")
    positive_count = metric("recovery", "all", "positive_mse", "count")
    rows = {}
    for stratum in ("all", *STRATA):
        count = metric("recovery", stratum, "mse", "count")
        positive = metric("recovery", stratum, "positive_mse", "count")
        row = {"count": count, "positive_count": positive,
               "full_share": count / full_count if full_count else None,
               "positive_share": positive / positive_count if positive_count else None}
        for name in ("mse", "zero_mse", "constant_mse", "positive_coverage", "positive_mse",
                     "positive_zero_mse", "positive_constant_mse", "teacher_rgb_bce",
                     "teacher_full_bce", "teacher_rgb_plus_student_bce",
                     "student_base_plus_teacher_delta_bce", *POSITIVE_BCE_FIELDS):
            row[name] = metric("recovery", stratum, name)
        for name in ("base_bce", "corrected_bce", "bce_improvement", "raw_abs", "clipped_abs", "clip_fraction"):
            row[name] = metric("relation", stratum, name)
        rows["global" if stratum == "all" else stratum] = row
    return rows


__all__ = ["STRATA", "POSITIVE_BCE_FIELDS", "positive_stratified_mse_loss", "evaluate_source_detailed", "source_summary"]
