"""ResNet-50 TriPRD implementation for paper-comparable experiments.

The feasibility model in :mod:`models.triprd` is intentionally tiny.  This
module keeps the same teacher/student/loss contract while replacing the
encoder with ImageNet-compatible ResNet-50 branches and a 256-channel
multi-scale decoder.  The student remains RGB-only at inference.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .triprd import OutputHeads, TriPRDLoss, _as_one_channel, count_parameters, teacher_gate


def _make_resnet50(in_channels: int, pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("The ResNet-50 implementation requires torchvision") from exc

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    try:
        backbone = resnet50(weights=weights)
    except TypeError:  # torchvision < 0.13 compatibility
        backbone = resnet50(pretrained=pretrained)
    if in_channels != 3:
        old = backbone.conv1
        replacement = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                                stride=old.stride, padding=old.padding,
                                bias=False)
        with torch.no_grad():
            if pretrained:
                replacement.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
        backbone.conv1 = replacement
    return backbone


class ResNet50Pyramid(nn.Module):
    """Return feature maps at strides 4, 8, 16 and 32."""

    channels = (256, 512, 1024, 2048)

    def __init__(self, in_channels: int, pretrained: bool) -> None:
        super().__init__()
        backbone = _make_resnet50(in_channels, pretrained)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


class ResUNetDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], width: int = 256) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, width, 1, bias=False) for c in channels])
        self.norms = nn.ModuleList([nn.BatchNorm2d(width) for _ in channels])
        self.up3 = nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.out_channels = width

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        p1, p2, p3, p4 = [F.relu(norm(proj(feat)), inplace=True)
                           for feat, proj, norm in zip(features, self.projections, self.norms)]
        x = self.up3(torch.cat([F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False), p3], dim=1))
        x = self.up2(torch.cat([F.interpolate(x, size=p2.shape[-2:], mode="bilinear", align_corners=False), p2], dim=1))
        return self.up1(torch.cat([F.interpolate(x, size=p1.shape[-2:], mode="bilinear", align_corners=False), p1], dim=1))


class TriPRDTeacherFull(nn.Module):
    def __init__(self, base_channels: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        self.rgb_encoder = ResNet50Pyramid(3, pretrained)
        self.thermal_encoder = ResNet50Pyramid(1, pretrained)
        self.depth_encoder = ResNet50Pyramid(1, pretrained)
        self.decoder = ResUNetDecoder(ResNet50Pyramid.channels, width=base_channels)
        self.heads = OutputHeads(self.decoder.out_channels)
        self.modality_heads = nn.ModuleList([nn.Conv2d(2048, 1, 1) for _ in range(3)])

    def forward(self, rgb: Tensor, thermal: Tensor, depth: Tensor) -> dict[str, Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape Bx3xHxW, got {tuple(rgb.shape)}")
        thermal, depth = _as_one_channel(thermal, name="thermal"), _as_one_channel(depth, name="depth")
        output_size = tuple(rgb.shape[-2:])
        rf, tf, df = self.rgb_encoder(rgb), self.thermal_encoder(thermal), self.depth_encoder(depth)
        fused = tuple((r + t + d) / 3.0 for r, t, d in zip(rf, tf, df))
        decoded = self.decoder(fused)
        output = self.heads(decoded, output_size)
        output["modality_logits"] = torch.cat([
            F.interpolate(head(feat[-1]), size=output_size, mode="bilinear", align_corners=False)
            for head, feat in zip(self.modality_heads, (rf, tf, df))
        ], dim=1)
        output["features"] = decoded
        return output


class TriPRDStudentFull(nn.Module):
    def __init__(self, base_channels: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = ResNet50Pyramid(3, pretrained)
        self.decoder = ResUNetDecoder(ResNet50Pyramid.channels, width=base_channels)
        self.heads = OutputHeads(self.decoder.out_channels)

    def forward(self, rgb: Tensor) -> dict[str, Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape Bx3xHxW, got {tuple(rgb.shape)}")
        decoded = self.decoder(self.encoder(rgb))
        output = self.heads(decoded, tuple(rgb.shape[-2:]))
        output["features"] = decoded
        return output


__all__ = [
    "TriPRDTeacherFull", "TriPRDStudentFull", "TriPRDLoss",
    "teacher_gate", "count_parameters",
]
