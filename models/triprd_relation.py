"""Explicit local relation transfer for the next TriPRD experiment.

The RGB-only student predicts two neighbourhood distributions and uses them
in its segmentation forward path.  A capacity-matched teacher predicts the
same distributions from RGB/RGB/RGB or aligned RGB/thermal/depth features.
Only ``relation_branch`` is intended to cross the source-to-COD boundary.
These are established attention/KL building blocks, not a claim of novelty
or measured improvement over the archived TriPRD model.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .triprd import OutputHeads, TinyEncoder, UNetDecoder, _prepare_mask, _soft_iou_loss
from .triprd_full import ResNet50Pyramid, ResUNetDecoder


def _check_kernel(kernel: int) -> None:
    if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1 or kernel % 2 != 1:
        raise ValueError("relation_kernel must be a positive odd integer")


def local_valid_mask(height: int, width: int, kernel: int, *, device: torch.device) -> Tensor:
    """Return 1x1xKxHxW validity in ``unfold`` row-major neighbourhood order."""
    _check_kernel(kernel)
    if height < 1 or width < 1:
        raise ValueError("relation grids must have positive height and width")
    grid = torch.ones((1, 1, height, width), device=device)
    return F.unfold(grid, kernel, padding=kernel // 2).reshape(1, 1, kernel * kernel, height, width).bool()


def masked_local_distribution(logits: Tensor, valid: Tensor, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    """Stable float32 probabilities/log-probabilities; invalid entries are zero.

    Each query always has its in-bounds centre, so no row is entirely masked.
    Zero log-probabilities outside the grid avoid undefined ``0 * -inf`` in
    KL without assigning any probability mass to padded neighbours.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("relation temperature must be finite and positive")
    if logits.ndim != 5 or valid.shape != (1, 1, *logits.shape[2:]) or valid.dtype != torch.bool:
        raise ValueError("expected Bx2xKxHxW logits and a boolean 1x1xKxHxW valid mask")
    if logits.shape[1] != 2:
        raise ValueError("exactly two relation distributions are required")
    if not valid.any(dim=2).all():
        raise ValueError("every query needs at least one valid neighbour")
    log_probs = F.log_softmax((logits.float() / temperature).masked_fill(~valid, -torch.inf), dim=2)
    probs = log_probs.exp().masked_fill(~valid, 0)
    return probs, log_probs.masked_fill(~valid, 0)


@torch.no_grad()
def normalized_relation_entropy(logits: Tensor, valid: Tensor) -> Tensor:
    """Detached entropy in [0, 1], using each query's valid neighbour count.

    Diagnostics use the forward distribution (temperature 1), independently
    of KD temperature. A singleton neighbourhood has zero entropy by
    convention. Uniform distributions have value 1 even at padded borders.
    """
    probabilities, logs = masked_local_distribution(logits.detach(), valid)
    counts = valid.sum(dim=2)
    entropy = -(probabilities * logs).sum(dim=2)
    normalized = entropy / counts.float().log().clamp_min(1e-12)
    return torch.where(counts > 1, normalized, 0).clamp(0, 1).mean()


class LocalRelationBranch(nn.Module):
    """Two local relation predictors plus all parameters needed for aggregation.

    The shared input projection defines a jointly trained feature space for
    RGB and each context.  Separate target heads retain the identities of
    the two RGB-to-context relations.  Parameter layout is identical in
    student and teachers; neither decoder nor encoder parameters live here.
    """

    def __init__(self, channels: int, relation_dim: int = 32, relation_kernel: int = 3) -> None:
        super().__init__()
        _check_kernel(relation_kernel)
        if isinstance(relation_dim, bool) or not isinstance(relation_dim, int) or relation_dim < 1:
            raise ValueError("relation_dim must be a positive integer")
        self.relation_dim = relation_dim
        self.relation_kernel = relation_kernel
        self.shared_projection = nn.Conv2d(channels, relation_dim, 1, bias=False)
        self.queries = nn.ModuleList([nn.Conv2d(relation_dim, relation_dim, 1, bias=False) for _ in range(2)])
        self.keys = nn.ModuleList([nn.Conv2d(relation_dim, relation_dim, 1, bias=False) for _ in range(2)])
        self.values = nn.ModuleList([nn.Conv2d(relation_dim, relation_dim, 1, bias=False) for _ in range(2)])
        self.output_projection = nn.Conv2d(relation_dim * 2, channels, 1, bias=False)

    def _neighbours(self, features: Tensor) -> Tensor:
        b, c, h, w = features.shape
        k = self.relation_kernel
        return F.unfold(features, k, padding=k // 2).reshape(b, c, k * k, h, w)

    def forward(self, rgb: Tensor, contexts: tuple[Tensor, Tensor] | None = None) -> dict[str, Tensor]:
        if contexts is None:
            contexts = (rgb, rgb)
        if len(contexts) != 2 or any(context.shape != rgb.shape for context in contexts):
            raise ValueError("both relation contexts must match the RGB feature grid and channels")
        rgb_projected = self.shared_projection(rgb)
        context_projected = [self.shared_projection(context) for context in contexts]
        logits = torch.stack([
            (query(rgb_projected).unsqueeze(2) * self._neighbours(key(context))).sum(dim=1)
            / math.sqrt(self.relation_dim)
            for query, key, context in zip(self.queries, self.keys, context_projected)
        ], dim=1)
        valid = local_valid_mask(*rgb.shape[-2:], self.relation_kernel, device=rgb.device)
        # Finite sentinels also make JSON diagnostics and all-finite checks safe.
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        probs, _ = masked_local_distribution(logits, valid)
        aggregates = [
            (self._neighbours(value(context)) * probs[:, index].unsqueeze(1).to(rgb.dtype)).sum(dim=2)
            for index, (value, context) in enumerate(zip(self.values, context_projected))
        ]
        residual = self.output_projection(torch.cat(aggregates, dim=1))
        return {"residual": residual, "relation_logits": logits,
                "relation_probs": probs, "relation_valid_mask": valid}


def _make_encoder(model_variant: str, base_channels: int, pretrained: bool) -> nn.Module:
    if base_channels < 1:
        raise ValueError("base_channels must be positive")
    if model_variant == "tiny":
        if pretrained:
            raise ValueError("the tiny encoder has no pretrained weights; use pretrained=False")
        return TinyEncoder(3, base_channels)
    if model_variant == "resnet50":
        return ResNet50Pyramid(3, pretrained)
    raise ValueError("model_variant must be 'tiny' or 'resnet50'")


def _make_decoder(model_variant: str, channels: tuple[int, int, int, int], base_channels: int) -> nn.Module:
    return UNetDecoder(channels) if model_variant == "tiny" else ResUNetDecoder(channels, base_channels)


def _check_rgb(rgb: Tensor) -> None:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape Bx3xHxW")


class RelationStudent(nn.Module):
    """RGB-only relation-aware student, with a source-stage backbone freeze."""

    def __init__(self, model_variant: str = "tiny", base_channels: int = 24, pretrained: bool = False,
                 relation_dim: int = 32, relation_kernel: int = 3) -> None:
        super().__init__()
        self.encoder = _make_encoder(model_variant, base_channels, pretrained)
        self.decoder = _make_decoder(model_variant, self.encoder.channels, base_channels)
        self.heads = OutputHeads(self.decoder.out_channels)
        self.relation_branch = LocalRelationBranch(self.encoder.channels[-2], relation_dim, relation_kernel)
        self._backbone_frozen = False

    def freeze_backbone(self) -> "RelationStudent":
        self._backbone_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        return self

    def unfreeze_backbone(self) -> "RelationStudent":
        self._backbone_frozen = False
        self.encoder.requires_grad_(True)
        self.encoder.train(self.training)
        return self

    def train(self, mode: bool = True) -> "RelationStudent":
        super().train(mode)
        if self._backbone_frozen:
            self.encoder.eval()
        return self

    def forward(self, rgb: Tensor) -> dict[str, Tensor]:
        _check_rgb(rgb)
        features = list(self.encoder(rgb))
        relation = self.relation_branch(features[-2])
        features[-2] = features[-2] + relation.pop("residual")
        decoded = self.decoder(tuple(features))
        output = self.heads(decoded, tuple(rgb.shape[-2:]))
        output.update(relation)
        output["features"] = decoded
        return output


class RelationTeacher(nn.Module):
    """Matched three-encoder RGB or RGB/T/D relation teacher.

    All three stems take three channels, including the RGB control.  Thermal
    and depth are provided as one channel repeated three times; this does
    not synthesize missing modalities.  Their normalization stays the data
    loader's responsibility and must be disclosed in experiment metadata.
    """

    def __init__(self, model_variant: str = "tiny", base_channels: int = 24, pretrained: bool = False,
                 relation_dim: int = 32, relation_kernel: int = 3, teacher_kind: str = "multimodal") -> None:
        super().__init__()
        if teacher_kind not in {"rgb", "multimodal"}:
            raise ValueError("teacher_kind must be 'rgb' or 'multimodal'")
        self.teacher_kind = teacher_kind
        self.rgb_encoder = _make_encoder(model_variant, base_channels, pretrained)
        self.thermal_encoder = _make_encoder(model_variant, base_channels, pretrained)
        self.depth_encoder = _make_encoder(model_variant, base_channels, pretrained)
        self.decoder = _make_decoder(model_variant, self.rgb_encoder.channels, base_channels)
        self.heads = OutputHeads(self.decoder.out_channels)
        self.relation_branch = LocalRelationBranch(self.rgb_encoder.channels[-2], relation_dim, relation_kernel)

    @staticmethod
    def _modality(x: Tensor | None, rgb: Tensor, name: str) -> Tensor:
        if x is None or x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"multimodal teacher requires real {name} with shape Bx1xHxW")
        if x.shape[0] != rgb.shape[0] or x.shape[-2:] != rgb.shape[-2:]:
            raise ValueError(f"{name} must be paired on the RGB batch and spatial grid")
        return x.repeat(1, 3, 1, 1)

    def forward(self, rgb: Tensor, thermal: Tensor | None = None, depth: Tensor | None = None) -> dict[str, Tensor]:
        _check_rgb(rgb)
        if self.teacher_kind == "rgb":
            thermal_rgb, depth_rgb = rgb, rgb
        else:
            thermal_rgb = self._modality(thermal, rgb, "thermal")
            depth_rgb = self._modality(depth, rgb, "depth")
        rf = self.rgb_encoder(rgb)
        tf = self.thermal_encoder(thermal_rgb)
        df = self.depth_encoder(depth_rgb)
        fused = [(r + t + d) / 3.0 for r, t, d in zip(rf, tf, df)]
        relation = self.relation_branch(rf[-2], (tf[-2], df[-2]))
        fused[-2] = fused[-2] + relation.pop("residual")
        decoded = self.decoder(tuple(fused))
        output = self.heads(decoded, tuple(rgb.shape[-2:]))
        output.update(relation)
        output["features"] = decoded
        return output


class RelationLoss(nn.Module):
    """BCE+IoU supervision and optional detached categorical relation KL.

    KL is averaged across valid query locations and the two relation types;
    each local neighbourhood is one categorical distribution.  No boundary,
    uncertainty or segmentation-logit KD is introduced in this mechanism.
    """

    def __init__(self, relation_weight: float = 0.0, relation_temperature: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(relation_weight) or relation_weight < 0:
            raise ValueError("relation_weight must be finite and nonnegative")
        if not math.isfinite(relation_temperature) or relation_temperature <= 0:
            raise ValueError("relation_temperature must be finite and positive")
        self.relation_weight = float(relation_weight)
        self.relation_temperature = float(relation_temperature)

    def forward(self, output: Mapping[str, Tensor], mask: Tensor,
                teacher: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:
        target = _prepare_mask(mask)
        bce = F.binary_cross_entropy_with_logits(output["logits"], target)
        iou = _soft_iou_loss(output["logits"], target)
        segmentation = bce + iou
        kd = segmentation.new_zeros(())
        if self.relation_weight > 0:
            if teacher is None:
                raise ValueError("positive relation_weight requires teacher relation targets")
            student_logits, teacher_logits = output["relation_logits"], teacher["relation_logits"].detach()
            student_valid = output["relation_valid_mask"]
            teacher_valid = teacher["relation_valid_mask"]
            if student_logits.shape != teacher_logits.shape or not torch.equal(student_valid, teacher_valid):
                raise ValueError("student and teacher relation grids/neighbourhoods must match exactly")
            _, student_logs = masked_local_distribution(student_logits, student_valid, self.relation_temperature)
            teacher_probs, teacher_logs = masked_local_distribution(teacher_logits, teacher_valid, self.relation_temperature)
            kd = (teacher_probs * (teacher_logs - student_logs)).sum(dim=2).mean().clamp_min(0)
            kd = kd * self.relation_temperature ** 2
        weighted = self.relation_weight * kd
        losses = {"total": segmentation + weighted, "segmentation": segmentation,
                  "supervised": segmentation, "bce": bce, "iou": iou,
                  "relation_kd": kd, "weighted_relation_kd": weighted}
        if "relation_logits" in output:
            losses["relation_entropy"] = normalized_relation_entropy(
                output["relation_logits"], output["relation_valid_mask"])
        if teacher is not None and "relation_logits" in teacher:
            losses["teacher_relation_entropy"] = normalized_relation_entropy(
                teacher["relation_logits"], teacher["relation_valid_mask"])
        return losses


__all__ = ["RelationStudent", "RelationTeacher", "RelationLoss", "LocalRelationBranch",
           "local_valid_mask", "masked_local_distribution", "normalized_relation_entropy"]
