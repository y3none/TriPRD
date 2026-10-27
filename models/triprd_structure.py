"""Two-scale structural transfer with an independently frozen RGB reference.

The transferred object is the complete ``structure_branch``: reference encoder
and edge heads. The task encoder and coarse-to-fine readout remain trainable.
Edge channels are thermal continuity, depth continuity, and annotation agreement;
their sigmoid predictions are *not* categorical neighbourhood distributions.
Feature indices 1/2 mean strides 8/16 for ResNet-50, but 2/4 for TinyEncoder.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .triprd import OutputHeads, TinyEncoder
from .triprd_full import ResNet50Pyramid


EDGE_TYPES = ("thermal", "depth", "annotation")
STRUCTURE_FEATURE_INDICES = (1, 2)
INTERVENTIONS = ("none", "identity", "uniform", "shuffle", "no_source", "uniform_source", "shuffle_source")


def _validate_options(base_channels: int, relation_dim: int, relation_kernel: int) -> None:
    if base_channels < 1 or relation_dim < 1:
        raise ValueError("base_channels and relation_dim must be positive")
    if relation_kernel < 3 or relation_kernel % 2 != 1:
        raise ValueError("relation_kernel must be an odd integer >= 3")


def _make_encoder(model_variant: str, base_channels: int, pretrained: bool) -> nn.Module:
    if model_variant == "tiny":
        if pretrained:
            raise ValueError("The tiny encoder has no pretrained weights")
        return TinyEncoder(3, base_channels)
    if model_variant == "resnet50":
        return ResNet50Pyramid(3, pretrained)
    raise ValueError(f"Unsupported model_variant: {model_variant!r}")


def _truncate_structure_encoder(encoder: nn.Module) -> nn.Module:
    """Remove the unused final stage while preserving graph-input features.

    Structure heads consume only pyramid indices 1 and 2. Replacing stage 4
    also avoids its forward compute; the unused fourth tuple entry aliases
    index 2. The task RGB encoder must never be passed to this function.
    """
    if isinstance(encoder, ResNet50Pyramid):
        encoder.layer4 = nn.Identity()
    elif isinstance(encoder, TinyEncoder):
        encoder.stage4 = nn.Identity()
    else:
        raise TypeError("Unsupported structure reference encoder")
    return encoder


def _check_rgb(rgb: Tensor) -> None:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape Bx3xHxW")


def _neighbours(features: Tensor, kernel: int) -> Tensor:
    b, c, h, w = features.shape
    return F.unfold(features, kernel, padding=kernel // 2).reshape(b, c, kernel * kernel, h, w)


def structure_valid_mask(height: int, width: int, kernel: int, *, device: torch.device) -> Tensor:
    """Valid *off-diagonal* local edges, shape 1x1xKxHxW.

    Self edges have fixed weight one in the propagation operator and must not
    enter the learned-edge loss. A 1x1 image therefore has no valid edge.
    """
    if height < 1 or width < 1 or kernel < 1 or kernel % 2 != 1:
        raise ValueError("spatial dimensions must be positive and kernel positive odd")
    valid = _neighbours(torch.ones(1, 1, height, width, device=device), kernel).bool()
    valid[:, :, kernel * kernel // 2] = False
    return valid


def normalize_structure_edges(weights: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Return off-diagonal and self probabilities with a fixed unit self-loop."""
    if weights.ndim != 5 or valid.shape != (1, 1, *weights.shape[2:]):
        raise ValueError("weights must be BxMxKxHxW and valid 1x1xKxHxW")
    # Normalization is float32 under AMP; masked logits never contribute.
    off = weights.float().masked_fill(~valid, 0.0)
    denominator = 1.0 + off.sum(dim=2, keepdim=True)
    return off / denominator, denominator.reciprocal()


def propagate_structure(values: Tensor, weights: Tensor, valid: Tensor, kernel: int) -> tuple[Tensor, Tensor]:
    """Convex local RGB-value aggregation, returning BxMxCxHxW and self mass."""
    probabilities, self_mass = normalize_structure_edges(weights, valid)
    neighbours = _neighbours(values, kernel)
    aggregated = torch.einsum("bmkij,bckij->bmcij", probabilities.to(values.dtype), neighbours)
    aggregated = aggregated + self_mass.squeeze(2).unsqueeze(2).to(values.dtype) * values.unsqueeze(1)
    return aggregated, self_mass.squeeze(2)


def _shuffle_valid_edges(weights: Tensor, valid: Tensor) -> Tensor:
    """Deterministically permute query locations per offset, preserving padding.

    Unlike flatten-and-roll, no invalid boundary edge can become valid. The
    intervention preserves the marginal edge values for every local offset.
    """
    b, m, k, h, w = weights.shape
    flat = weights.reshape(b, m, k, h * w)
    indices = torch.arange(h * w, device=weights.device).expand(k, -1).clone()
    for offset in range(k):
        positions = valid[0, 0, offset].flatten().nonzero(as_tuple=False).flatten()
        indices[offset, positions] = positions.flip(0)
    return flat.gather(3, indices[None, None].expand(b, m, -1, -1)).reshape_as(weights)


class PairwiseEdgeHead(nn.Module):
    """Learn symmetric pair scores from normalized endpoint descriptors."""

    def __init__(self, channels: int, relation_dim: int, edge_types: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.projection = nn.Conv2d(channels, relation_dim, 1, bias=False)
        self.pair_mlp = nn.Sequential(
            nn.Conv3d(relation_dim * 3, relation_dim, 1), nn.SiLU(),
            nn.Conv3d(relation_dim, edge_types, 1),
        )

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        projected = F.normalize(self.projection(features), dim=1, eps=1e-6)
        centre = projected.unsqueeze(2)
        nearby = _neighbours(projected, self.kernel)
        descriptors = torch.cat((torch.abs(centre - nearby), centre * nearby, centre + nearby), dim=1)
        # Finite invalid entries simplify losses; callers still must use valid.
        return self.pair_mlp(descriptors).masked_fill(~valid, 0.0)


class RGBStructureBranch(nn.Module):
    """Complete source knowledge function, independent of the task backbone."""

    def __init__(self, encoder: nn.Module, relation_dim: int, kernel: int) -> None:
        super().__init__()
        self.reference_encoder = _truncate_structure_encoder(deepcopy(encoder))
        self.reference_encoder.requires_grad_(False)
        self.reference_encoder.eval()
        self.kernel = kernel
        self.heads = nn.ModuleList([
            PairwiseEdgeHead(encoder.channels[index], relation_dim, len(EDGE_TYPES), kernel)
            for index in STRUCTURE_FEATURE_INDICES
        ])
        self._frozen = False

    def train(self, mode: bool = True) -> RGBStructureBranch:
        super().train(mode and not self._frozen)
        self.reference_encoder.eval()
        return self

    def freeze(self) -> None:
        self._frozen = True
        self.requires_grad_(False)
        self.eval()

    def forward(self, rgb: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        with torch.no_grad():
            features = self.reference_encoder(rgb)
        logits, masks = [], []
        for index, head in zip(STRUCTURE_FEATURE_INDICES, self.heads):
            feature = features[index]
            valid = structure_valid_mask(*feature.shape[-2:], self.kernel, device=feature.device)
            logits.append(head(feature, valid))
            masks.append(valid)
        return tuple(logits), tuple(masks)


class TeacherStructureBranch(nn.Module):
    """Modalities are allowed to determine edges, never decoder values."""

    def __init__(self, channels: tuple[int, ...], relation_dim: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.heads = nn.ModuleList([
            PairwiseEdgeHead(3 * channels[index], relation_dim, len(EDGE_TYPES), kernel)
            for index in STRUCTURE_FEATURE_INDICES
        ])

    def forward(self, rgb_features: tuple[Tensor, ...], thermal_features: tuple[Tensor, ...],
                depth_features: tuple[Tensor, ...]) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        logits, masks = [], []
        for index, head in zip(STRUCTURE_FEATURE_INDICES, self.heads):
            feature = torch.cat((rgb_features[index], thermal_features[index], depth_features[index]), dim=1)
            valid = structure_valid_mask(*feature.shape[-2:], self.kernel, device=feature.device)
            logits.append(head(feature, valid))
            masks.append(valid)
        return tuple(logits), tuple(masks)


class TaskConditionedPropagation(nn.Module):
    """RGB context selects identity, three compatible structure graphs, or GT-trained semantic edges."""

    def __init__(self, channels: int, relation_dim: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.semantic_head = PairwiseEdgeHead(channels, relation_dim, 1, kernel)
        # All inputs to the gate and value projection belong to the RGB path.
        self.gate = nn.Sequential(nn.Conv2d(channels + 1, channels, 1), nn.SiLU(), nn.Conv2d(channels, 5, 1))
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, features: Tensor, guidance: Tensor, structures: Tensor, valid: Tensor,
                intervention: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        semantic_logits = self.semantic_head(features, valid)
        semantic = semantic_logits.float().sigmoid()
        source = structures.float().sigmoid()
        if intervention == "no_source":
            source = torch.zeros_like(source)
        elif intervention == "uniform_source":
            # Preserve source confidence per image/type while deleting its spatial pattern.
            mean = source.masked_fill(~valid, 0.0).sum(dim=(2, 3, 4), keepdim=True)
            mean = mean / valid.sum().clamp_min(1)
            source = mean.expand_as(source)
        elif intervention == "shuffle_source":
            # Semantic compatibility remains attached to its original query pair.
            source = _shuffle_valid_edges(source, valid)
        weights = torch.cat((source * semantic, semantic), dim=1)
        if intervention == "identity":
            weights = torch.zeros_like(weights)
        elif intervention == "uniform":
            weights = torch.ones_like(weights)
        elif intervention == "shuffle":
            weights = _shuffle_valid_edges(weights, valid)
        gates = self.gate(torch.cat((features, guidance.sigmoid()), dim=1)).float().softmax(dim=1)
        values = self.value(features)
        aggregated, self_mass = propagate_structure(values, weights, valid, self.kernel)
        # Identity is the omitted zeroth candidate: its update is exactly zero.
        differences = aggregated - values.unsqueeze(1)
        update = (differences * gates[:, 1:].unsqueeze(2).to(values.dtype)).sum(dim=1)
        return features + self.output(update), semantic_logits, gates, self_mass


def _refine(in_channels: int, out_channels: int) -> nn.Sequential:
    # No batch statistics in the graph readout: works for 1x1 feature maps too.
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU())


class StructureDecoder(nn.Module):
    """Coarse-to-fine readout that propagates at both interior pyramid scales."""

    def __init__(self, channels: tuple[int, ...], width: int, relation_dim: int, kernel: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.coarse_context = _refine(width * 2, width)
        self.fine_context = _refine(width * 2 + 1, width)
        self.output_context = _refine(width * 2 + 1, width)
        self.graphs = nn.ModuleList([TaskConditionedPropagation(width, relation_dim, kernel) for _ in range(2)])
        self.guidance_head = nn.Conv2d(width, 1, 1)
        self.aux_heads = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in range(2)])
        self.out_channels = width

    @staticmethod
    def _up(value: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features: tuple[Tensor, ...], structures: tuple[Tensor, ...],
                masks: tuple[Tensor, ...], intervention: str) -> dict[str, object]:
        p1, p2, p3, p4 = [F.silu(layer(feature)) for layer, feature in zip(self.projections, features)]
        coarse = self.coarse_context(torch.cat((p3, self._up(p4, p3)), dim=1))
        coarse, sem3, gate3, self3 = self.graphs[1](
            coarse, self._up(self.guidance_head(p4), p3), structures[1], masks[1], intervention)
        mask3 = self.aux_heads[1](coarse)
        fine = self.fine_context(torch.cat((p2, self._up(coarse, p2), self._up(mask3, p2).sigmoid()), dim=1))
        fine, sem2, gate2, self2 = self.graphs[0](fine, self._up(mask3, p2), structures[0], masks[0], intervention)
        mask2 = self.aux_heads[0](fine)
        decoded = self.output_context(torch.cat((p1, self._up(fine, p1), self._up(mask2, p1).sigmoid()), dim=1))
        return {"features": decoded, "semantic_logits": (sem2, sem3), "aux_logits": (mask2, mask3),
                "propagation_gates": (gate2, gate3), "propagation_self_mass": (self2, self3)}


class StructureStudent(nn.Module):
    """RGB-only task encoder plus a separately transferable structure function."""

    def __init__(self, model_variant: str = "tiny", base_channels: int = 24,
                 pretrained: bool = False, relation_dim: int = 32, relation_kernel: int = 3) -> None:
        super().__init__()
        _validate_options(base_channels, relation_dim, relation_kernel)
        self.encoder = _make_encoder(model_variant, base_channels, pretrained)
        self.structure_branch = RGBStructureBranch(self.encoder, relation_dim, relation_kernel)
        self.decoder = StructureDecoder(self.encoder.channels, base_channels, relation_dim, relation_kernel)
        self.heads = OutputHeads(self.decoder.out_channels)
        self._backbone_frozen = False

    def freeze_backbone(self) -> None:
        self._backbone_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def unfreeze_backbone(self) -> None:
        self._backbone_frozen = False
        self.encoder.requires_grad_(True)
        self.encoder.train(self.training)

    def freeze_structure(self) -> None:
        self.structure_branch.freeze()

    def train(self, mode: bool = True) -> StructureStudent:
        super().train(mode)
        if self._backbone_frozen:
            self.encoder.eval()
        return self

    def forward(self, rgb: Tensor, *, intervention: str = "none") -> dict[str, object]:
        _check_rgb(rgb)
        if intervention not in INTERVENTIONS:
            raise ValueError(f"intervention must be one of {INTERVENTIONS}")
        features = self.encoder(rgb)
        structures, masks = self.structure_branch(rgb)
        decoded = self.decoder(features, structures, masks, intervention)
        output = self.heads(decoded["features"], tuple(rgb.shape[-2:]))
        output.update(decoded)
        output.update(structure_logits=structures, structure_valid_masks=masks)
        return output


class StructureTeacher(nn.Module):
    """Capacity-matched three-encoder teacher, with RGB-only value readout."""

    def __init__(self, model_variant: str = "tiny", base_channels: int = 24,
                 pretrained: bool = False, relation_dim: int = 32, relation_kernel: int = 3,
                 teacher_kind: str = "multimodal") -> None:
        super().__init__()
        _validate_options(base_channels, relation_dim, relation_kernel)
        if teacher_kind not in ("rgb", "multimodal"):
            raise ValueError("teacher_kind must be 'rgb' or 'multimodal'")
        self.teacher_kind = teacher_kind
        self.rgb_encoder = _make_encoder(model_variant, base_channels, pretrained)
        self.thermal_encoder = _truncate_structure_encoder(_make_encoder(model_variant, base_channels, pretrained))
        self.depth_encoder = _truncate_structure_encoder(_make_encoder(model_variant, base_channels, pretrained))
        self.structure_branch = TeacherStructureBranch(self.rgb_encoder.channels, relation_dim, relation_kernel)
        self.decoder = StructureDecoder(self.rgb_encoder.channels, base_channels, relation_dim, relation_kernel)
        self.heads = OutputHeads(self.decoder.out_channels)

    @staticmethod
    def _modality(value: Tensor | None, rgb: Tensor, name: str) -> Tensor:
        if value is None or value.ndim != 4 or value.shape[1] not in (1, 3):
            raise ValueError(f"{name} must have shape Bx1xHxW or Bx3xHxW")
        if value.shape[0] != rgb.shape[0] or value.shape[-2:] != rgb.shape[-2:]:
            raise ValueError(f"{name} must be registered to rgb at the same batch/spatial size")
        return value.repeat(1, 3, 1, 1) if value.shape[1] == 1 else value

    def forward(self, rgb: Tensor, thermal: Tensor | None = None, depth: Tensor | None = None,
                *, intervention: str = "none") -> dict[str, object]:
        _check_rgb(rgb)
        if intervention not in INTERVENTIONS:
            raise ValueError(f"intervention must be one of {INTERVENTIONS}")
        if self.teacher_kind == "rgb":
            thermal_input, depth_input = rgb, rgb
        else:
            thermal_input = self._modality(thermal, rgb, "thermal")
            depth_input = self._modality(depth, rgb, "depth")
        rgb_features = self.rgb_encoder(rgb)
        structures, masks = self.structure_branch(
            rgb_features, self.thermal_encoder(thermal_input), self.depth_encoder(depth_input))
        # The only modal-to-mask route is through the two learned graphs.
        decoded = self.decoder(rgb_features, structures, masks, intervention)
        output = self.heads(decoded["features"], tuple(rgb.shape[-2:]))
        output.update(decoded)
        output.update(structure_logits=structures, structure_valid_masks=masks)
        return output
