"""Compact, experimentally faithful TriPRD teacher/student models.

The teacher has separate RGB/thermal/depth encoders and a fused decoder; the
student is strictly RGB only.  Distillation uses decoder logits, boundary
logits, and fixed target-defined feature relations.  There are no random
projection, learned uncertainty, or teacher-injection paths.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _as_one_channel(x: Tensor, *, name: str) -> Tensor:
    if x.ndim != 4 or x.shape[1] not in (1, 3):
        raise ValueError(f"{name} must have shape Bx1xHxW (or Bx3xHxW), got {tuple(x.shape)}")
    return x if x.shape[1] == 1 else x.mean(dim=1, keepdim=True)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class TinyEncoder(nn.Module):
    """Four-stage encoder, with no external or pretrained weights."""

    def __init__(self, in_channels: int = 3, base: int = 24) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, base)
        self.stage2 = ConvBlock(base, base * 2, stride=2)
        self.stage3 = ConvBlock(base * 2, base * 4, stride=2)
        self.stage4 = ConvBlock(base * 4, base * 8, stride=2)
        self.channels = (base, base * 2, base * 4, base * 8)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        f1 = self.stem(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4


class UNetDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up3 = ConvBlock(c4 + c3, c3)
        self.up2 = ConvBlock(c3 + c2, c2)
        self.up1 = ConvBlock(c2 + c1, c1)
        self.out_channels = c1

    @staticmethod
    def _up(x: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f1, f2, f3, f4 = features
        x = self.up3(torch.cat([self._up(f4, f3), f3], dim=1))
        x = self.up2(torch.cat([self._up(x, f2), f2], dim=1))
        return self.up1(torch.cat([self._up(x, f1), f1], dim=1))


class OutputHeads(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mask = nn.Conv2d(channels, 1, 1)
        self.boundary = nn.Conv2d(channels, 1, 1)

    def forward(self, features: Tensor, output_size: tuple[int, int]) -> dict[str, Tensor]:
        return {
            "logits": F.interpolate(self.mask(features), size=output_size, mode="bilinear", align_corners=False),
            "boundary_logits": F.interpolate(self.boundary(features), size=output_size, mode="bilinear", align_corners=False),
        }


class TriPRDTeacher(nn.Module):
    """Tri-modal training teacher with separately supervised modality heads."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        self.rgb_encoder = TinyEncoder(3, base_channels)
        self.thermal_encoder = TinyEncoder(1, base_channels)
        self.depth_encoder = TinyEncoder(1, base_channels)
        channels = self.rgb_encoder.channels
        self.decoder = UNetDecoder(channels)
        self.heads = OutputHeads(self.decoder.out_channels)
        self.modality_heads = nn.ModuleList([nn.Conv2d(channels[-1], 1, 1) for _ in range(3)])

    def forward(self, rgb: Tensor, thermal: Tensor, depth: Tensor) -> dict[str, Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape Bx3xHxW, got {tuple(rgb.shape)}")
        thermal = _as_one_channel(thermal, name="thermal")
        depth = _as_one_channel(depth, name="depth")
        output_size = tuple(rgb.shape[-2:])
        rgb_features = self.rgb_encoder(rgb)
        thermal_features = self.thermal_encoder(thermal)
        depth_features = self.depth_encoder(depth)
        fused = tuple((r + t + d) / 3.0 for r, t, d in zip(rgb_features, thermal_features, depth_features))
        decoded = self.decoder(fused)
        output = self.heads(decoded, output_size)
        modality_features = (rgb_features[-1], thermal_features[-1], depth_features[-1])
        output["modality_logits"] = torch.cat(
            [F.interpolate(head(feat), size=output_size, mode="bilinear", align_corners=False)
             for head, feat in zip(self.modality_heads, modality_features)], dim=1)
        output["features"] = decoded
        return output


class TriPRDStudent(nn.Module):
    """RGB-only student; forward has no teacher/prior argument by design."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        self.encoder = TinyEncoder(3, base_channels)
        self.decoder = UNetDecoder(self.encoder.channels)
        self.heads = OutputHeads(self.decoder.out_channels)

    def forward(self, rgb: Tensor) -> dict[str, Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape Bx3xHxW, got {tuple(rgb.shape)}")
        decoded = self.decoder(self.encoder(rgb))
        output = self.heads(decoded, tuple(rgb.shape[-2:]))
        output["features"] = decoded
        return output


def _prepare_mask(mask: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape Bx1xHxW, got {tuple(mask.shape)}")
    mask = mask.float()
    if mask.detach().max() > 1:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def _soft_iou_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    pred = torch.sigmoid(logits)
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims) - intersection
    return (1.0 - (intersection + eps) / (union + eps)).mean()


def _boundary_target(mask: Tensor) -> Tensor:
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def teacher_gate(output: Mapping[str, Tensor], temperature: float = 1.0) -> Tensor:
    """Detached Bx1xHxW confidence from variance of three modality predictions."""
    if "modality_logits" not in output:
        raise KeyError("teacher output must contain modality_logits")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = output["modality_logits"]
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("modality_logits must have shape Bx3xHxW")
    probs = torch.sigmoid(logits / float(temperature))
    variance = probs.var(dim=1, keepdim=True, unbiased=False)
    return (1.0 - variance / 0.25).clamp(0.1, 1.0).detach()


def _binary_kl(student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
    t = float(temperature)
    q = torch.sigmoid(teacher_logits.detach() / t)
    bce = F.binary_cross_entropy_with_logits(student_logits / t, q, reduction="none")
    entropy = -(q * torch.log(q.clamp_min(1e-7)) + (1 - q) * torch.log((1 - q).clamp_min(1e-7)))
    return (bce - entropy).clamp_min(0.0) * (t * t)


def _category_indices(mask: Tensor, feature_size: tuple[int, int], k: int) -> list[list[Tensor]]:
    target = F.interpolate(mask, size=feature_size, mode="nearest") > 0.5
    boundary = _boundary_target(target.float()) > 0.0
    categories = (
        target & ~boundary,  # interior
        (F.max_pool2d(target.float(), 3, stride=1, padding=1) > 0.5) & ~target,  # nearby bg
        boundary & target,  # boundary inside
        boundary & ~target,  # boundary outside
    )
    b, _, h, w = target.shape
    result: list[list[Tensor]] = []
    for batch in range(b):
        sample: list[Tensor] = []
        for category in categories:
            indices = category[batch, 0].flatten().nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                indices = torch.arange(h * w, device=mask.device)
            sample.append(indices[torch.arange(k, device=mask.device) % indices.numel()])
        result.append(sample)
    return result


def _relation_cosine_gram(student_features: Tensor, teacher_features: Tensor, mask: Tensor, gate: Tensor, tokens_per_category: int) -> Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError("student and teacher feature maps must have identical shapes")
    b, c, h, w = student_features.shape
    index_sets = _category_indices(mask, (h, w), tokens_per_category)
    gate = F.interpolate(gate, size=(h, w), mode="bilinear", align_corners=False)
    terms: list[Tensor] = []
    for batch in range(b):
        for indices in index_sets[batch]:
            s = student_features[batch].flatten(1).transpose(0, 1)[indices]
            t = teacher_features[batch].flatten(1).transpose(0, 1)[indices].detach()
            s = F.normalize(s, dim=1, eps=1e-6)
            t = F.normalize(t, dim=1, eps=1e-6)
            sg = s @ s.transpose(0, 1)
            tg = t @ t.transpose(0, 1)
            point = gate[batch, 0].flatten()[indices].detach()
            pair = point[:, None] * point[None, :]
            terms.append(((sg - tg).pow(2) * pair).sum() / pair.sum().clamp_min(1e-6))
    return torch.stack(terms).mean() if terms else student_features.new_zeros(())


class TriPRDLoss:
    """Supervised objective and selectable KD components.

    ``none`` is the RGB control; the other modes add one or all KD terms.
    ``full`` applies the detached disagreement gate, whereas ``fixed`` is the
    gate ablation with identical components and weights.
    """

    def __init__(self, mode: str = "full", temperature: float = 4.0, calibration_temperature: float = 1.0,
                 relation_tokens: int = 8, lambda_iou: float = 1.0, lambda_boundary: float = 0.5,
                 lambda_logit: float = 1.0, lambda_relation: float = 1.0, lambda_boundary_kd: float = 0.5) -> None:
        allowed = {"none", "logit", "relation", "boundary", "full", "fixed"}
        if mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}, got {mode!r}")
        if temperature <= 0 or calibration_temperature <= 0 or relation_tokens <= 0:
            raise ValueError("temperature, calibration_temperature and relation_tokens must be positive")
        self.mode, self.temperature, self.calibration_temperature = mode, float(temperature), float(calibration_temperature)
        self.relation_tokens = int(relation_tokens)
        self.lambda_iou, self.lambda_boundary = float(lambda_iou), float(lambda_boundary)
        self.lambda_logit, self.lambda_relation, self.lambda_boundary_kd = float(lambda_logit), float(lambda_relation), float(lambda_boundary_kd)

    def __call__(self, output: Mapping[str, Tensor], mask: Tensor, teacher: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:
        target = _prepare_mask(mask)
        logits = output["logits"]
        bce = F.binary_cross_entropy_with_logits(logits, target)
        iou = _soft_iou_loss(logits, target)
        boundary = F.binary_cross_entropy_with_logits(output["boundary_logits"], _boundary_target(target))
        total = bce + self.lambda_iou * iou + self.lambda_boundary * boundary
        zero = total.new_zeros(())
        losses: dict[str, Tensor] = {"bce": bce, "iou": iou, "boundary": boundary,
                                      "logit_kd": zero, "relation_kd": zero, "boundary_kd": zero,
                                      "modality_bce": zero, "modality_iou": zero, "gate_mean": total.new_ones(())}
        if "modality_logits" in output:
            modality_target = target.expand(-1, 3, -1, -1)
            modality_bce = F.binary_cross_entropy_with_logits(output["modality_logits"], modality_target)
            modality_iou = _soft_iou_loss(output["modality_logits"], modality_target)
            losses["modality_bce"], losses["modality_iou"] = modality_bce, modality_iou
            total = total + modality_bce + self.lambda_iou * modality_iou
        if self.mode != "none" and teacher is None:
            raise ValueError(f"mode={self.mode!r} requires teacher output")
        if teacher is not None and self.mode != "none":
            gate = teacher_gate(teacher, self.calibration_temperature) if self.mode == "full" else torch.ones_like(teacher["logits"][:, :1]).detach()
            losses["gate_mean"] = gate.mean()
            if self.mode in {"logit", "full", "fixed"}:
                value = (_binary_kl(logits, teacher["logits"], self.temperature) * (gate if self.mode == "full" else 1.0)).mean()
                losses["logit_kd"] = value
                total = total + self.lambda_logit * value
            if self.mode in {"relation", "full", "fixed"}:
                value = _relation_cosine_gram(output["features"], teacher["features"].detach(), target, gate, self.relation_tokens)
                losses["relation_kd"] = value
                total = total + self.lambda_relation * value
            if self.mode in {"boundary", "full", "fixed"}:
                value = ((output["boundary_logits"] - teacher["boundary_logits"].detach()).pow(2) * (gate if self.mode == "full" else 1.0)).mean()
                losses["boundary_kd"] = value
                total = total + self.lambda_boundary_kd * value
        losses["total"] = total
        return losses


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


__all__ = ["TriPRDTeacher", "TriPRDStudent", "TriPRDLoss", "teacher_gate", "count_parameters"]
