"""Independent RGB correction student and one four-coalition source teacher.

The default encoder is an official, *generic pretrained* DINOv3 ViT-B/16
loaded strictly from a local repository and checkpoint. No COD model is reused.
RGB inputs are ImageNet-normalized; auxiliary observations are in [0, 1].
Only adapters/readout/edge predictors are trained with the default configuration.
Teacher modalities share one backbone, but have distinct trainable projections.
There is no second RGB reference encoder and no auxiliary input in the student.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.dinov3_weights import load_dinov3_weights


COALITIONS = ("R", "RT", "RD", "RTD")
TASKS = ("sod", "cod")
CORRECTION_INTERVENTIONS = ("none", "no_correction", "shuffle_correction", "identity")
CORRECTION_MODEL_SPEC = {
    "version": "coalition_probability_correction_v2",
    "pyramid_strides": [4, 8, 16, 32],
    "relation_strides": [8, 16],
    "relation_semantics": "annotation_compatibility_off_diagonal",
    "correction": "signed_tanh_prediction_then_clip_base_plus_delta_for_propagation",
    "teacher": "one_shared_backbone_distinct_modal_adapters_missing_tokens",
    "teacher_relation_supervision": "equal_four_coalition_relations_without_repeated_rgb_base",
    "student_correction_initialization": "zero_last_projection",
    "rgb_input": "imagenet_normalized",
    "thermal_input": "three_channel_pseudocolor_[0,1]_imagenet_normalized_in_model",
    "depth_input": "positive_valid_[0,1]_per_image_q05_q95_to_rgb_imagenet_normalized",
    "depth_constant": "valid_pixels_0.5_invalid_pixels_0",
}


def _resize(x: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _neighbours(x: Tensor, kernel: int) -> Tensor:
    b, c, h, w = x.shape
    return F.unfold(x, kernel, padding=kernel // 2).reshape(b, c, kernel * kernel, h, w)


def relation_valid_mask(height: int, width: int, kernel: int, *, device: torch.device) -> Tensor:
    """1x1xKxHxW; excludes padding and the fixed, non-learned self loop."""
    if height < 1 or width < 1 or kernel < 3 or kernel % 2 != 1:
        raise ValueError("Positive feature dimensions and an odd kernel >= 3 are required")
    mask = _neighbours(torch.ones(1, 1, height, width, device=device), kernel).bool()
    mask[:, :, kernel * kernel // 2] = False
    return mask


def shuffle_correction(correction: Tensor, valid: Tensor) -> Tensor:
    """Fixed spatial permutation, independently for every valid offset.

    It preserves each offset's signed residual histogram and never mixes padded
    and valid edges. Applying a shuffled residual may require clipping to the
    receiving RGB probability's headroom; the raw permutation itself is exact.
    """
    b, c, k, h, w = correction.shape
    indices = torch.arange(h * w, device=correction.device).expand(k, -1).clone()
    for offset in range(k):
        positions = valid[0, 0, offset].flatten().nonzero(as_tuple=False).flatten()
        indices[offset, positions] = positions.flip(0)
    return correction.flatten(3).gather(3, indices[None, None].expand(b, c, -1, -1)).reshape_as(correction)


def normalize_auxiliary(image: Tensor, *, depth: bool) -> Tensor:
    """Finite ImageNet-normalized RGB input for the shared generic encoder.

    Depth zeros/nonfinite/out-of-range values are invalid. The deterministic
    robust per-image range preserves relative geometry despite uint16 export
    scaling; this is preprocessing, not a metric-depth prediction objective.
    Thermal color channels are retained, not converted to grayscale.
    """
    if image.ndim != 4 or image.shape[1] not in ((1,) if depth else (1, 3)):
        raise ValueError("depth must be Bx1xHxW; thermal must be Bx1/3xHxW")
    image = image.float()
    valid = (torch.isfinite(image) & (image >= 0) & (image <= 1)).all(1, keepdim=True)
    if depth:
        valid = valid & (image > 0)
    clean = torch.where(valid, image, torch.zeros_like(image))
    if depth:
        normalized = []
        for sample, keep in zip(clean, valid):
            values = sample[keep]
            if values.numel() == 0:
                normalized.append(torch.zeros_like(sample))
                continue
            # Bound quantile work without random sampling or GPU-to-CPU copies.
            if values.numel() > 4096:
                pick = torch.linspace(0, values.numel() - 1, 4096, device=values.device).long()
                values = values[pick]
            low, high = torch.quantile(values, torch.tensor([0.05, 0.95], device=values.device))
            if (high - low).item() < 1e-8:
                low, high = values.min(), values.max()
            if (high - low).item() < 1e-8:
                unit = torch.full_like(sample, 0.5)
            else:
                unit = ((sample - low) / (high - low)).clamp(0, 1)
            normalized.append(torch.where(keep, unit, torch.zeros_like(unit)))
        clean = torch.stack(normalized)
    if clean.shape[1] == 1:
        clean = clean.expand(-1, 3, -1, -1)
    mean = clean.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = clean.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (clean - mean) / std


class _TinyBackbone(nn.Module):
    """Explicit test fixture. Never selected as a pretrained fallback."""

    def __init__(self, width: int):
        super().__init__()
        self.channels = (width, width * 2, width * 4, width * 8)
        incoming = 3
        self.stages = nn.ModuleList()
        for i, outgoing in enumerate(self.channels):
            self.stages.append(nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, stride=4 if i == 0 else 2, padding=1),
                nn.GroupNorm(1, outgoing), nn.GELU(),
            ))
            incoming = outgoing

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features)


class GenericFeatureEncoder(nn.Module):
    """Strict local official DINO loading; frozen state survives model.train()."""

    def __init__(self, model_variant: str, width: int, pretrained: bool,
                 dino_repo: str | Path | None, dino_weights: str | Path | None,
                 frozen: bool):
        super().__init__()
        self.model_variant = model_variant
        self.is_dino = model_variant != "tiny"
        self.weight_load_metadata = None
        if model_variant == "tiny":
            if pretrained:
                raise ValueError("tiny is test-only and requires pretrained=False")
            self.backbone = _TinyBackbone(width)
            self.channels = self.backbone.channels
        elif model_variant in ("dinov3_vitb16", "dinov3_vits16", "dinov3_vitl16"):
            if dino_repo is None:
                raise ValueError("DINOv3 requires local dino_repo even for checkpoint-only evaluation")
            if pretrained and dino_weights is None:
                raise ValueError("Pretrained DINOv3 requires local generic dino_weights")
            repo = Path(dino_repo).expanduser().resolve()
            # Keep snapshot paths: resolving a HF cache symlink loses the adjacent config.json.
            weights = Path(dino_weights).expanduser().absolute() if dino_weights is not None else None
            if not (repo / "hubconf.py").is_file():
                raise FileNotFoundError(f"Official DINOv3 local repository lacks hubconf.py: {repo}")
            if pretrained and not weights.is_file():
                raise FileNotFoundError(f"Generic DINOv3 weights not found: {weights}")
            # pretrained=False prevents the hub function from choosing/downloading
            # any default weights; the provided local generic state is strict.
            self.backbone = torch.hub.load(str(repo), model_variant, source="local", pretrained=False)
            if pretrained:
                self.weight_load_metadata = load_dinov3_weights(self.backbone, weights, model_variant)
            if not callable(getattr(self.backbone, "get_intermediate_layers", None)):
                raise TypeError("Official DINOv3 get_intermediate_layers API is required")
            dimension = int(self.backbone.embed_dim)
            self.channels = (dimension,) * 4
            count = len(self.backbone.blocks)
            self.layer_indices = tuple((i + 1) * count // 4 - 1 for i in range(4))
            self.local_repo = str(repo)
            self.local_weights = str(weights) if pretrained else None
        else:
            raise ValueError(f"Unsupported model_variant: {model_variant!r}")
        self.set_frozen(frozen)

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        self.backbone.requires_grad_(not self.frozen)
        self.backbone.train(self.training and not self.frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode and not self.frozen)
        return self

    def forward(self, rgb: Tensor) -> tuple[Tensor, ...]:
        with torch.no_grad() if self.frozen else nullcontext():
            if not self.is_dino:
                return self.backbone(rgb)
            # Pad, rather than silently crop the last image row/column in ViT.
            h, w = rgb.shape[-2:]
            padded = F.pad(rgb, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
            features = self.backbone.get_intermediate_layers(
                padded, n=self.layer_indices, reshape=True, norm=True,
            )
            if len(features) != 4 or any(x.ndim != 4 for x in features):
                raise RuntimeError("DINOv3 must return four normalized BxCxHxW features")
            return tuple(features)


class _PyramidAdapters(nn.Module):
    def __init__(self, channels: tuple[int, ...], width: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, width, 1), nn.GroupNorm(1, width), nn.GELU())
            for ch in channels
        ])
        bottleneck = max(4, width // 4)
        self.residuals = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, bottleneck, 1), nn.GELU(),
                          nn.Conv2d(bottleneck, width, 3, padding=1)) for _ in channels
        ])

    def forward(self, features: tuple[Tensor, ...], image_size: tuple[int, int]) -> tuple[Tensor, ...]:
        outputs = []
        for feature, projection, residual, stride in zip(features, self.projections, self.residuals, (4, 8, 16, 32)):
            projected = projection(feature)
            adapted = projected + residual(projected)
            size = tuple(max(1, (dimension + stride - 1) // stride) for dimension in image_size)
            outputs.append(_resize(adapted, size))
        return tuple(outputs)


class _PairHead(nn.Module):
    def __init__(self, width: int, kernel: int, *, zero_init: bool = False):
        super().__init__()
        self.kernel = kernel
        hidden = max(4, width // 4)
        self.projection = nn.Conv2d(width, hidden, 1)
        self.edge = nn.Sequential(nn.Conv3d(hidden * 2, hidden, 1), nn.GELU(), nn.Conv3d(hidden, 1, 1))
        if zero_init:
            # Start at the RGB base exactly. The final layer learns immediately;
            # its preceding feature projections receive gradients after it moves.
            nn.init.zeros_(self.edge[-1].weight)
            nn.init.zeros_(self.edge[-1].bias)

    def forward(self, feature: Tensor, valid: Tensor) -> Tensor:
        projected = F.normalize(self.projection(feature), dim=1, eps=1e-6)
        nearby = _neighbours(projected, self.kernel)
        centre = projected.unsqueeze(2)
        pair = torch.cat(((centre - nearby).abs(), centre * nearby), dim=1)
        return self.edge(pair).masked_fill(~valid, 0)


class _Propagation(nn.Module):
    def __init__(self, width: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.value = nn.Conv2d(width, width, 1)
        self.readout = nn.Conv2d(width, width, 1, bias=False)

    def forward(self, feature: Tensor, probability: Tensor, valid: Tensor) -> Tensor:
        value = self.value(feature)
        weights = probability.float().masked_fill(~valid, 0)
        mass = 1 + weights.sum(2)
        neighbours = _neighbours(value, self.kernel)
        aggregate = (neighbours * weights.to(value.dtype)).sum(2) + value
        aggregate = aggregate / mass.to(value.dtype)
        return feature + self.readout(aggregate - value)


def _refine(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(1, width), nn.GELU(),
                         nn.Conv2d(width, width, 3, padding=1), nn.GELU())


class _Decoder(nn.Module):
    def __init__(self, width: int, kernel: int):
        super().__init__()
        self.refine = nn.ModuleList([_refine(width) for _ in range(4)])
        self.propagation = nn.ModuleList([_Propagation(width, kernel) for _ in range(2)])

    def forward(self, features: tuple[Tensor, ...], probabilities: tuple[Tensor, ...],
                valid: tuple[Tensor, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
        x = self.refine[3](features[3])
        relation_features = {}
        for level in (2, 1, 0):
            x = self.refine[level](features[level] + _resize(x, features[level].shape[-2:]))
            if level in (1, 2):
                index = level - 1
                x = self.propagation[index](x, probabilities[index], valid[index])
                relation_features[index] = x
        return x, tuple(relation_features[i] for i in (0, 1))


class _TaskHeads(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mask = nn.Conv2d(width, 1, 1)
        self.auxiliary = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in range(2)])

    def forward(self, decoded: Tensor, intermediate: tuple[Tensor, ...], size: tuple[int, int]) -> dict:
        return {"logits": _resize(self.mask(decoded), size),
                "aux_logits": tuple(head(feature) for head, feature in zip(self.auxiliary, intermediate))}


class _BaseCorrection(nn.Module):
    def __init__(self, model_variant: str, base_channels: int, pretrained: bool,
                 dino_repo: str | Path | None, dino_weights: str | Path | None,
                 relation_kernel: int, freeze_backbone: bool):
        super().__init__()
        if base_channels < 4 or relation_kernel < 3 or relation_kernel % 2 != 1:
            raise ValueError("base_channels >= 4 and odd relation_kernel >= 3 are required")
        self.kernel = relation_kernel
        self.encoder = GenericFeatureEncoder(model_variant, base_channels, pretrained,
                                             dino_repo, dino_weights, freeze_backbone)
        self.adapters = _PyramidAdapters(self.encoder.channels, base_channels)
        self.decoder = _Decoder(base_channels, relation_kernel)

    def alignment_parameters(self) -> Iterator[nn.Parameter]:
        """Small shared RGB adaptation parameters for source/target gradients."""
        return (parameter for parameter in self.adapters.parameters() if parameter.requires_grad)

    def freeze_backbone(self) -> None:
        self.encoder.set_frozen(True)

    def unfreeze_backbone(self) -> None:
        self.encoder.set_frozen(False)

    def _features(self, rgb: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 1:
            raise ValueError("ImageNet-normalized RGB must have shape Bx3xHxW")
        if not torch.isfinite(rgb).all():
            raise ValueError("RGB input contains nonfinite values")
        features = self.adapters(self.encoder(rgb), tuple(rgb.shape[-2:]))
        valid = tuple(relation_valid_mask(*features[i].shape[-2:], self.kernel, device=rgb.device) for i in (1, 2))
        return features, valid


class CorrectionStudent(_BaseCorrection):
    """Strict RGB inference, shared adaptation, separate SOD/COD mask heads."""

    def __init__(self, model_variant: str = "dinov3_vitb16", base_channels: int = 128,
                 pretrained: bool = True, dino_repo: str | Path | None = None,
                 dino_weights: str | Path | None = None, relation_kernel: int = 3,
                 freeze_backbone: bool = True):
        super().__init__(model_variant, base_channels, pretrained, dino_repo, dino_weights,
                         relation_kernel, freeze_backbone)
        self.base_heads = nn.ModuleList([_PairHead(base_channels, self.kernel) for _ in range(2)])
        self.correction_heads = nn.ModuleList([_PairHead(base_channels, self.kernel, zero_init=True) for _ in range(2)])
        self.task_heads = nn.ModuleDict({task: _TaskHeads(base_channels) for task in TASKS})

    def forward(self, rgb: Tensor, task: str = "cod", *, intervention: str = "none") -> dict:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if intervention not in CORRECTION_INTERVENTIONS:
            raise ValueError(f"intervention must be one of {CORRECTION_INTERVENTIONS}")
        features, valid = self._features(rgb)
        base_logits, logits, corrections, probabilities, applied, clipped = [], [], [], [], [], []
        for index, (base_head, correction_head, mask) in enumerate(zip(self.base_heads, self.correction_heads, valid)):
            feature = features[index + 1]
            base_logit = base_head(feature, mask)
            base = base_logit.float().sigmoid()
            signed = correction_head(feature, mask).float().tanh()
            # Supervise the signed correction independently of the student's
            # own base calibration. Only the deployed probability is clipped;
            # otherwise a saturated base could make a teacher delta unreachable.
            residual = signed.masked_fill(~mask, 0)
            probability = (base + residual).clamp(1e-6, 1 - 1e-6)
            clipped.append((probability - base).masked_fill(~mask, 0))
            effective_residual = residual
            if intervention == "no_correction":
                effective_residual = torch.zeros_like(residual)
            elif intervention == "shuffle_correction":
                effective_residual = shuffle_correction(residual, mask)
            effective_probability = (base + effective_residual).clamp(1e-6, 1 - 1e-6)
            if intervention == "identity":
                effective_probability = torch.zeros_like(probability)
            base_logits.append(base_logit)
            logits.append(torch.logit(probability).masked_fill(~mask, 0))
            corrections.append(residual)
            probabilities.append(probability)
            applied.append(effective_probability)
        decoded, intermediates = self.decoder(features, tuple(applied), valid)
        output = self.task_heads[task](decoded, intermediates, tuple(rgb.shape[-2:]))
        output.update(relation_logits=tuple(logits), base_relation_logits=tuple(base_logits),
                      corrections=tuple(corrections), relation_probs=tuple(probabilities),
                      clipped_corrections=tuple(clipped), propagation_probs=tuple(applied), relation_valid_masks=valid)
        return output


class CorrectionTeacher(_BaseCorrection):
    """One teacher, four trained input coalitions, explicit missing tokens.

    Auxiliary information can change relations, but the propagation values and
    entire decoder are RGB. forward_coalitions caches each modality extraction
    once, so all q_R/q_RT/q_RD/q_RTD use identical weights and RGB features.
    """

    def __init__(self, model_variant: str = "dinov3_vitb16", base_channels: int = 128,
                 pretrained: bool = True, dino_repo: str | Path | None = None,
                 dino_weights: str | Path | None = None, relation_kernel: int = 3,
                 freeze_backbone: bool = True):
        super().__init__(model_variant, base_channels, pretrained, dino_repo, dino_weights,
                         relation_kernel, freeze_backbone)
        self.thermal_adapters = _PyramidAdapters(self.encoder.channels, base_channels)
        self.depth_adapters = _PyramidAdapters(self.encoder.channels, base_channels)
        self.missing_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.zeros(2, base_channels, 1, 1)) for modality in ("T", "D")
        })
        self.fusion = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * base_channels, base_channels, 1), nn.GELU(),
                          nn.Conv2d(base_channels, base_channels, 3, padding=1)) for _ in range(2)
        ])
        self.relation_heads = nn.ModuleList([_PairHead(base_channels, self.kernel) for _ in range(2)])
        self.heads = _TaskHeads(base_channels)

    def _aux_features(self, value: Tensor | None, rgb: Tensor, *, depth: bool) -> tuple[Tensor, ...]:
        if value is None:
            raise ValueError(f"Selected coalition requires {'depth' if depth else 'thermal'} input")
        if value.shape[0] != rgb.shape[0] or value.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("Auxiliary inputs must match RGB batch and spatial dimensions")
        normalized = normalize_auxiliary(value, depth=depth)
        adapter = self.depth_adapters if depth else self.thermal_adapters
        return adapter(self.encoder(normalized), tuple(rgb.shape[-2:]))

    def _edges(self, features: tuple[Tensor, ...], thermal: tuple[Tensor, ...] | None,
               depth: tuple[Tensor, ...] | None, valid: tuple[Tensor, ...], coalition: str) -> tuple[Tensor, ...]:
        outputs = []
        for index in (0, 1):
            rgb = features[index + 1]
            inputs = [rgb]
            for modality, auxiliary in (("T", thermal), ("D", depth)):
                inputs.append(auxiliary[index + 1] if modality in coalition else
                              self.missing_tokens[modality][index:index + 1].expand_as(rgb))
            fused = rgb + self.fusion[index](torch.cat(inputs, dim=1))
            outputs.append(self.relation_heads[index](fused, valid[index]))
        return tuple(outputs)

    def _readout(self, features: tuple[Tensor, ...], logits: tuple[Tensor, ...],
                 base: tuple[Tensor, ...], valid: tuple[Tensor, ...], size: tuple[int, int]) -> dict:
        probabilities = tuple(x.float().sigmoid() for x in logits)
        decoded, intermediates = self.decoder(features, probabilities, valid)
        output = self.heads(decoded, intermediates, size)
        output.update(relation_logits=logits, base_relation_logits=base, relation_probs=probabilities,
                      corrections=tuple((q - r.float().sigmoid()).masked_fill(~mask, 0)
                                        for q, r, mask in zip(probabilities, base, valid)),
                      relation_valid_masks=valid, propagation_probs=probabilities)
        return output

    def forward(self, rgb: Tensor, thermal: Tensor | None = None, depth: Tensor | None = None,
                coalition: str = "RTD") -> dict:
        if coalition not in COALITIONS:
            raise ValueError(f"coalition must be one of {COALITIONS}")
        features, valid = self._features(rgb)
        # A missing coalition does not validate, normalize, encode, or use that
        # modality. R therefore remains independent even of malformed aux data.
        t = self._aux_features(thermal, rgb, depth=False) if "T" in coalition else None
        d = self._aux_features(depth, rgb, depth=True) if "D" in coalition else None
        base = self._edges(features, None, None, valid, "R")
        logits = base if coalition == "R" else self._edges(features, t, d, valid, coalition)
        return self._readout(features, logits, base, valid, tuple(rgb.shape[-2:]))

    def forward_coalitions(self, rgb: Tensor, thermal: Tensor, depth: Tensor) -> dict[str, dict]:
        features, valid = self._features(rgb)
        t = self._aux_features(thermal, rgb, depth=False)
        d = self._aux_features(depth, rgb, depth=True)
        base = self._edges(features, None, None, valid, "R")
        return {coalition: self._readout(
            features, base if coalition == "R" else self._edges(features, t, d, valid, coalition),
            base, valid, tuple(rgb.shape[-2:]),
        ) for coalition in COALITIONS}
