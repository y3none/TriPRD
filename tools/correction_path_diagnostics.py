"""Read-only, element-pooled diagnostics for a trained correction student.

All segmentation differences use floating point network outputs, before resize,
min-max normalization or uint8 conversion. Relation strata use the same nearest
annotation downsampling and valid off-diagonal edges as the training loss.
These statistics are descriptive; they do not select or modify checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from models.triprd_correction import CORRECTION_INTERVENTIONS
from models.triprd_correction_loss import annotation_relations, correction_targets


class DiagnosticAccumulator:
    """Accumulate scalar sums/counts/maxima, never minibatch-mean averages.

    Values and a broadcastable boolean mask may live on CPU or CUDA. Only small
    detached scalar tensors survive each call. Empty selections are represented
    by count=0, sum=0, mean=None and max=None in :meth:`result`.
    """

    def __init__(self) -> None:
        self._statistics: dict[str, dict[str, Tensor]] = {}

    @torch.no_grad()
    def add(self, name: str, values: Tensor, mask: Tensor | None = None) -> None:
        if not isinstance(values, Tensor):
            raise TypeError("diagnostic values must be a Tensor")
        values = values.detach()
        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError("diagnostic mask must be boolean")
            selected = values.masked_select(torch.broadcast_to(mask, values.shape))
        else:
            selected = values.reshape(-1)
        if not selected.is_floating_point():
            selected = selected.float()
        count = selected.new_tensor(selected.numel(), dtype=torch.int64)
        total = selected.sum(dtype=torch.float64)
        maximum = selected.max() if selected.numel() else selected.new_tensor(float("-inf"))
        record = self._statistics.get(name)
        if record is None:
            self._statistics[name] = {"sum": total, "count": count, "max": maximum}
        else:
            if total.device != record["sum"].device:
                raise ValueError("a diagnostic metric must stay on one device")
            record["sum"].add_(total)
            record["count"].add_(count)
            record["max"] = torch.maximum(record["max"], maximum)

    def result(self) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for name, record in sorted(self._statistics.items()):
            count = int(record["count"].item())
            total = float(record["sum"].item())
            maximum = float(record["max"].item()) if count else None
            if not torch.isfinite(record["sum"]).item() or (count and not torch.isfinite(record["max"]).item()):
                raise FloatingPointError(f"Nonfinite diagnostic values for {name}")
            result[name] = {"mean": total / count if count else None,
                            "sum": total, "count": count, "max": maximum}
        return result


def _constant(constants: Mapping[str, Sequence[Any]], name: str, index: int,
              reference: Tensor, kernel: int) -> Tensor:
    value = torch.as_tensor(constants[name][index], device=reference.device, dtype=torch.float32)
    if value.shape not in ((), (1, 1, kernel * kernel, 1, 1)):
        raise ValueError("TRAIN constant must be scalar or 1x1xKx1x1, fixed across images/pixels")
    return value


def _pooled_add(acc: DiagnosticAccumulator, prefix: str, section: str, scale: str,
                stratum: str, name: str, values: Tensor, mask: Tensor) -> None:
    for level in (scale, "pooled"):
        acc.add(f"{prefix}/{section}/{level}/{stratum}/{name}", values, mask)


def _inspect_relations(output: Mapping[str, Any], mask: Tensor, kernel: int,
                       acc: DiagnosticAccumulator, prefix: str,
                       targets: Mapping[str, Sequence[Tensor]] | None,
                       constants: Mapping[str, Sequence[Any]] | None) -> None:
    corrections = output["corrections"]
    if targets is not None:
        if constants is None or not all(name in constants for name in ("all", "positive")):
            raise ValueError("Source diagnostics require both fixed TRAIN constant sets")
        expected = len(corrections)
        if any(len(values) != expected for values in targets.values()) or any(
                len(constants[name]) != expected for name in ("all", "positive")):
            raise ValueError("Student, teacher and TRAIN constants must have matching scale counts")
    for index, raw in enumerate(corrections):
        raw = raw.float()
        scale = f"stride{8 * 2 ** index}"
        base_logits = output["base_relation_logits"][index].float()
        base = base_logits.sigmoid()
        corrected_logits = output["relation_logits"][index].float()
        clipped = output["clipped_corrections"][index].float()
        target, valid, strata = annotation_relations(
            mask, tuple(raw.shape[-2:]), kernel, output["relation_valid_masks"][index])
        if raw.shape != target.shape:
            raise ValueError("Student relation grids must match annotation batch/grids")
        selected = valid.expand_as(raw)
        stratified = {"all": torch.ones_like(selected),
                      **dict(zip(("fg", "bg", "crossing"), strata))}
        base_bce = F.binary_cross_entropy_with_logits(base_logits, target, reduction="none")
        corrected_bce = F.binary_cross_entropy_with_logits(corrected_logits, target, reduction="none")
        before_clip = base + raw
        relation_values = {
            "raw_abs": raw.abs(), "clipped_abs": clipped.abs(),
            "clip_fraction": ((before_clip < 1e-6) | (before_clip > 1 - 1e-6)).float(),
            "base_bce": base_bce, "corrected_bce": corrected_bce,
            "bce_improvement": base_bce - corrected_bce,
        }
        for stratum, membership in stratified.items():
            keep = selected & membership
            for name, value in relation_values.items():
                _pooled_add(acc, prefix, "relation", scale, stratum, name, value, keep)
        if targets is None:
            continue
        delta = targets["delta"][index].float()
        if delta.shape != raw.shape:
            raise ValueError("Student and teacher correction grids must match")
        selected = selected & targets["valid"][index]
        reliability = targets["reliability"][index]
        positive = reliability > 0
        constant = _constant(constants, "all", index, raw, kernel)
        positive_constant = _constant(constants, "positive", index, raw, kernel)
        teacher_base = targets["base_probability"][index].float()
        teacher_full = targets["full_probability"][index].float()
        bce = lambda probability: F.binary_cross_entropy(
            probability.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        squared = (raw - delta).square()
        zero_squared = delta.square()
        constant_squared = (constant - delta).square()
        recovery_values = {
            "mse": squared, "zero_mse": zero_squared, "constant_mse": constant_squared,
            "positive_coverage": positive.float(),
            "student_base_teacher_abs": (base - teacher_base).abs(),
            "student_base_teacher_mse": (base - teacher_base).square(),
            "teacher_rgb_bce": bce(teacher_base), "teacher_full_bce": bce(teacher_full),
            "teacher_rgb_plus_student_bce": bce(teacher_base + raw),
            "student_base_plus_teacher_delta_bce": bce(base + delta),
        }
        positive_values = {
            "positive_mse": squared, "positive_zero_mse": zero_squared,
            "positive_constant_mse": (positive_constant - delta).square(),
            "positive_all_edge_constant_mse": constant_squared,
        }
        for stratum, membership in stratified.items():
            keep = selected & membership
            for name, value in recovery_values.items():
                _pooled_add(acc, prefix, "recovery", scale, stratum, name, value, keep)
            for name, value in positive_values.items():
                _pooled_add(acc, prefix, "recovery", scale, stratum, name, value, keep & positive)


@torch.no_grad()
def inspect_batch(model, rgb: Tensor, mask: Tensor, task: str,
                  acc: DiagnosticAccumulator, prefix: str, *, teacher=None,
                  constants: Mapping[str, Sequence[Any]] | None = None,
                  thermal: Tensor | None = None, depth: Tensor | None = None,
                  targets: Mapping[str, Sequence[Tensor]] | None = None) -> dict[str, dict]:
    """Inspect all four interventions without training or changing model state.

    ``targets`` may be supplied from ``correction_targets``; otherwise passing a
    teacher requires thermal/depth and computes its coalitions once. In either
    source case, ``constants`` contains ``all`` and ``positive`` TRAIN-fitted
    tuples. No teacher means source recovery statistics are omitted.

    Returns the ordinary output dict for each intervention, suitable for the
    caller's existing segmentation evaluator. The student RGB feature extraction
    is cached for this call only. Hooks, an instance's prior ``_features``
    override, and every module's training flag are restored even on failure.
    """
    if teacher is not None and targets is not None:
        raise ValueError("Supply either teacher or precomputed targets, not both")
    if teacher is not None and (thermal is None or depth is None):
        raise ValueError("Source teacher evaluation requires thermal and depth")
    prefix = prefix.rstrip("/")
    modules = list(model.modules()) + (list(teacher.modules()) if teacher is not None else [])
    training_flags = [(module, module.training) for module in modules]
    missing = object()
    prior_features = model.__dict__.get("_features", missing)
    hooks = []
    captured: dict[str, Any] = {}
    outputs = {}
    try:
        model.eval()
        if teacher is not None:
            teacher.eval()
            coalitions = teacher.forward_coalitions(rgb, thermal, depth)
            targets = correction_targets(coalitions["R"], coalitions["RTD"], mask, model.kernel)
            del coalitions
        cached_features = model._features(rgb)
        model._features = lambda _rgb: cached_features

        def propagation_hook(index):
            def capture(_module, arguments, output):
                captured[f"stride{8 * 2 ** index}"] = (arguments[0].detach(), output.detach())
            return capture

        def decoder_hook(_module, _arguments, output):
            captured["decoder"] = output[0].detach()

        for index, propagation in enumerate(model.decoder.propagation):
            hooks.append(propagation.register_forward_hook(propagation_hook(index)))
        hooks.append(model.decoder.register_forward_hook(decoder_hook))
        reference_features = None
        for mode in CORRECTION_INTERVENTIONS:
            captured.clear()
            output = model(rgb, task=task, intervention=mode)
            outputs[mode] = output
            for scale, (incoming, outgoing) in ((key, value) for key, value in captured.items() if key != "decoder"):
                for name, value in (("input_abs", incoming.float().abs()),
                                    ("output_abs", outgoing.float().abs()),
                                    ("update_abs", (outgoing.float() - incoming.float()).abs())):
                    acc.add(f"{prefix}/propagation/{mode}/{scale}/{name}", value)
            if mode == "none":
                reference_features = dict(captured)
                # CUDA autocast forbids probability-space BCE; all relation
                # arithmetic above the accumulator is explicitly float32.
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    _inspect_relations(output, mask, model.kernel, acc, prefix, targets, constants)
                continue
            reference = outputs["none"]
            acc.add(f"{prefix}/intervention/{mode}/logits_abs_diff",
                    (output["logits"].float() - reference["logits"].float()).abs())
            acc.add(f"{prefix}/intervention/{mode}/prob_abs_diff",
                    (output["logits"].float().sigmoid() - reference["logits"].float().sigmoid()).abs())
            acc.add(f"{prefix}/intervention/{mode}/decoder_abs_diff",
                    (captured["decoder"].float() - reference_features["decoder"].float()).abs())
            for scale in (key for key in captured if key != "decoder"):
                incoming, outgoing = captured[scale]
                reference_in, reference_out = reference_features[scale]
                acc.add(f"{prefix}/intervention/{mode}/propagation_{scale}_abs_diff",
                        (outgoing.float() - reference_out.float()).abs())
                acc.add(f"{prefix}/intervention/{mode}/propagation_{scale}_input_abs_diff",
                        (incoming.float() - reference_in.float()).abs())
        return outputs
    finally:
        for hook in hooks:
            hook.remove()
        if prior_features is missing:
            if "_features" in model.__dict__:
                delattr(model, "_features")
        else:
            model._features = prior_features
        for module, training in training_flags:
            module.training = training


__all__ = ["DiagnosticAccumulator", "inspect_batch"]
