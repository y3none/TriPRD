"""Task-oriented RGB COD student and capacity-matched source teachers.

This model deliberately has no relation graph or teacher-relative residual.
The pretrained encoder can adapt in its final blocks, a real RGB convolutional
path supplies stride-4/8 detail, and shared task features enter separate SOD
and COD decoders. Thermal/depth teacher features change the decoder values.

RGB is ImageNet-normalized. Teacher auxiliary observations use the existing
documented [0, 1] thermal/depth preprocessing. ``tiny`` is a test-only encoder.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.triprd_correction import GenericFeatureEncoder, normalize_auxiliary


TASKS = ("sod", "cod")
COALITIONS = ("R", "RTD")
TASK_MODEL_SPEC = {
    "version": "task_feature_transfer_v1",
    "pyramid_strides": [4, 8, 16, 32],
    "task_feature_strides": [8, 16],
    "encoder_adaptation": "last_N_blocks_and_final_norm_N_zero_strictly_frozen",
    "detail": "real_rgb_stride4_stride8_convolutions_optional",
    "student_decoder": "independent_sod_and_cod_top_down_decoders",
    "teacher": "shared_encoder_three_modal_adapters_feature_value_fusion",
    "teacher_rgb_control": "three_rgb_slots_same_parameters_as_multimodal",
    "teacher_coalitions": list(COALITIONS),
    "task_features": "shared_decoder_inputs_unnormalized",
    "student_inference": "rgb_only",
}


def _resize(value: Tensor, size: tuple[int, int]) -> Tensor:
    if value.shape[-2:] == size:
        return value
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


def _grid(size: tuple[int, int], stride: int) -> tuple[int, int]:
    return tuple((dimension + stride - 1) // stride for dimension in size)


def _conv(incoming: int, outgoing: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(incoming, outgoing, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.GroupNorm(1, outgoing),
        nn.GELU(),
    )


class _Refine(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(_conv(width, width), _conv(width, width))

    def forward(self, value: Tensor) -> Tensor:
        return value + self.layers(value)


class TaskFeatureEncoder(GenericFeatureEncoder):
    """Official local weight loading with an explicitly trainable encoder tail.

    Frozen blocks remain in eval mode after ``model.train()``. Changing training
    mode never changes parameter trainability, including on a frozen teacher.
    The parent's no-grad forward is enabled only when the entire encoder is
    frozen, so gradients really reach a selected DINO/tiny tail.
    """

    def __init__(self, model_variant: str, width: int, pretrained: bool,
                 dino_repo: str | Path | None, dino_weights: str | Path | None,
                 train_backbone_blocks: int):
        super().__init__(model_variant, width, pretrained, dino_repo, dino_weights, frozen=True)
        blocks = self.backbone.blocks if self.is_dino else self.backbone.stages
        if (isinstance(train_backbone_blocks, bool) or not isinstance(train_backbone_blocks, int)
                or not 0 <= train_backbone_blocks <= len(blocks)):
            raise ValueError(f"train_backbone_blocks must be an integer in [0, {len(blocks)}]")
        self.train_backbone_blocks = train_backbone_blocks
        self.frozen = train_backbone_blocks == 0
        self.backbone.requires_grad_(False)
        if train_backbone_blocks:
            for block in list(blocks)[-train_backbone_blocks:]:
                block.requires_grad_(True)
            if self.is_dino:
                # Official get_intermediate_layers applies this final norm to
                # the patch tokens used by all four readouts.
                self.backbone.norm.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        # GenericFeatureEncoder.__init__ calls no train override before these
        # fields exist, but keep this method robust during normal construction.
        super().train(mode)
        self.backbone.eval()
        count = getattr(self, "train_backbone_blocks", 0)
        if mode and count:
            blocks = self.backbone.blocks if self.is_dino else self.backbone.stages
            for block in list(blocks)[-count:]:
                block.train(True)
            if self.is_dino:
                self.backbone.norm.train(True)
        return self


class _Pyramid(nn.Module):
    def __init__(self, channels: tuple[int, ...], width: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(_conv(channel, width, kernel=1), _Refine(width))
            for channel in channels
        ])

    def forward(self, encoded: tuple[Tensor, ...], size: tuple[int, int]) -> tuple[Tensor, ...]:
        return tuple(_resize(projection(feature), _grid(size, stride))
                     for feature, projection, stride in zip(encoded, self.projections, (4, 8, 16, 32)))


class _RGBDetail(nn.Module):
    """Learned high-resolution features extracted from pixels, not ViT resize."""

    def __init__(self, width: int):
        super().__init__()
        self.stride2 = _conv(3, max(4, width // 2), stride=2)
        self.stride4 = nn.Sequential(_conv(max(4, width // 2), width, stride=2), _Refine(width))
        self.stride8 = nn.Sequential(_conv(width, width, stride=2), _Refine(width))
        self.fusion = nn.ModuleList([_conv(width * 2, width, kernel=1) for _ in range(2)])

    def forward(self, rgb: Tensor, pyramid: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        detail4 = self.stride4(self.stride2(rgb))
        detail8 = self.stride8(detail4)
        fused = tuple(fusion(torch.cat((semantic, _resize(detail, semantic.shape[-2:])), 1))
                      for semantic, detail, fusion in zip(pyramid[:2], (detail4, detail8), self.fusion))
        return (*fused, *pyramid[2:])


class _SharedTaskFeatures(nn.Module):
    """Contextual shared representation consumed by either task decoder."""

    def __init__(self, width: int):
        super().__init__()
        self.stride16 = _Refine(width)
        self.stride8 = _Refine(width)

    def forward(self, pyramid: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        fine, middle, coarse, deepest = pyramid
        coarse = self.stride16(coarse + _resize(deepest, coarse.shape[-2:]))
        middle = self.stride8(middle + _resize(coarse, middle.shape[-2:]))
        return fine, middle, coarse, deepest


class _TaskDecoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.refine = nn.ModuleList([_Refine(width) for _ in range(4)])
        self.mask = nn.Conv2d(width, 1, 1)
        self.auxiliary = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in range(2)])
        self.boundary = nn.Sequential(_conv(width, width), nn.Conv2d(width, 1, 1))

    def forward(self, features: tuple[Tensor, ...], size: tuple[int, int]) -> dict[str, object]:
        decoded = [None] * 4
        decoded[3] = self.refine[3](features[3])
        for level in (2, 1, 0):
            decoded[level] = self.refine[level](
                features[level] + _resize(decoded[level + 1], features[level].shape[-2:]))
        return {
            "logits": _resize(self.mask(decoded[0]), size),
            "aux_logits": tuple(head(feature) for head, feature in zip(self.auxiliary, decoded[1:3])),
            "boundary_logits": _resize(self.boundary(decoded[0]), size),
            "task_features": (features[1], features[2]),
        }


class _TaskBase(nn.Module):
    def __init__(self, model_variant: str, base_channels: int, pretrained: bool,
                 dino_repo: str | Path | None, dino_weights: str | Path | None,
                 train_backbone_blocks: int, detail_branch: bool):
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 4:
            raise ValueError("base_channels must be an integer >= 4")
        if not isinstance(detail_branch, bool):
            raise ValueError("detail_branch must be boolean")
        self.base_channels = base_channels
        self.detail_branch = detail_branch
        self.encoder = TaskFeatureEncoder(model_variant, base_channels, pretrained,
                                          dino_repo, dino_weights, train_backbone_blocks)
        self.adapters = _Pyramid(self.encoder.channels, base_channels)
        self.detail = _RGBDetail(base_channels) if detail_branch else None
        self.shared_features = _SharedTaskFeatures(base_channels)

    @staticmethod
    def _check_rgb(rgb: Tensor) -> tuple[int, int]:
        if not isinstance(rgb, Tensor) or rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape) < 1:
            raise ValueError("ImageNet-normalized RGB must have nonempty shape Bx3xHxW")
        if not rgb.is_floating_point() or not torch.isfinite(rgb).all():
            raise ValueError("RGB must be finite and floating point")
        return tuple(rgb.shape[-2:])

    def _rgb_features(self, rgb: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        size = self._check_rgb(rgb)
        encoded = self.encoder(rgb)
        pyramid = self.adapters(encoded, size)
        if self.detail is not None:
            pyramid = self.detail(rgb, pyramid)
        return encoded, pyramid


class TaskStudent(_TaskBase):
    """RGB-only inference; SOD and COD have separate decoders and mask heads."""

    def __init__(self, model_variant: str = "dinov3_vitb16", base_channels: int = 128,
                 pretrained: bool = True, dino_repo: str | Path | None = None,
                 dino_weights: str | Path | None = None, train_backbone_blocks: int = 4,
                 detail_branch: bool = True):
        super().__init__(model_variant, base_channels, pretrained, dino_repo, dino_weights,
                         train_backbone_blocks, detail_branch)
        self.decoders = nn.ModuleDict({task: _TaskDecoder(base_channels) for task in TASKS})

    def forward(self, rgb: Tensor, task: str = "cod") -> dict[str, object]:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        _, pyramid = self._rgb_features(rgb)
        features = self.shared_features(pyramid)
        return self.decoders[task](features, tuple(rgb.shape[-2:]))


class TaskTeacher(_TaskBase):
    """One source SOD teacher with RGB or RGB/T/D input in identical slots.

    ``input_mode='rgb'`` runs all three modality adapters on the same cached
    RGB encoding, preserving the multimodal teacher's parameter capacity.
    ``coalition='R'`` substitutes learned missing tokens and ignores auxiliary
    inputs, including their contents and shape. Features, not pairwise edge
    weights, carry the additional information into the decoder.
    """

    def __init__(self, model_variant: str = "dinov3_vitb16", base_channels: int = 128,
                 pretrained: bool = True, dino_repo: str | Path | None = None,
                 dino_weights: str | Path | None = None, train_backbone_blocks: int = 4,
                 detail_branch: bool = True):
        super().__init__(model_variant, base_channels, pretrained, dino_repo, dino_weights,
                         train_backbone_blocks, detail_branch)
        self.modal_adapters = nn.ModuleDict({
            modality: _Pyramid(self.encoder.channels, base_channels) for modality in ("T", "D")
        })
        self.missing_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.zeros(4, base_channels, 1, 1)) for modality in ("T", "D")
        })
        self.fusion = nn.ModuleList([
            nn.Sequential(_conv(base_channels * 3, base_channels, kernel=1), _Refine(base_channels))
            for _ in range(4)
        ])
        self.decoder = _TaskDecoder(base_channels)

    @staticmethod
    def _check_mode(input_mode: str) -> None:
        if input_mode not in ("multimodal", "rgb"):
            raise ValueError("input_mode must be 'multimodal' or 'rgb'")

    def _auxiliary_features(self, value: Tensor | None, rgb: Tensor, modality: str) -> tuple[Tensor, ...]:
        if (not isinstance(value, Tensor) or value.ndim != 4 or value.shape[0] != rgb.shape[0]
                or value.shape[-2:] != rgb.shape[-2:]):
            raise ValueError(f"RTD multimodal input requires aligned {modality} observations")
        encoded = self.encoder(normalize_auxiliary(value, depth=modality == "D"))
        return self.modal_adapters[modality](encoded, tuple(rgb.shape[-2:]))

    def _prepare(self, rgb: Tensor, thermal: Tensor | None, depth: Tensor | None,
                 input_mode: str, full: bool) -> tuple[tuple[Tensor, ...], dict[str, tuple[Tensor, ...]]]:
        self._check_mode(input_mode)
        encoded, pyramid = self._rgb_features(rgb)
        auxiliary = {}
        if full:
            if input_mode == "rgb":
                auxiliary = {modality: adapter(encoded, tuple(rgb.shape[-2:]))
                             for modality, adapter in self.modal_adapters.items()}
            else:
                auxiliary = {
                    "T": self._auxiliary_features(thermal, rgb, "T"),
                    "D": self._auxiliary_features(depth, rgb, "D"),
                }
        return pyramid, auxiliary

    def _decode(self, rgb_features: tuple[Tensor, ...], auxiliary: dict[str, tuple[Tensor, ...]],
                coalition: str, size: tuple[int, int]) -> dict[str, object]:
        fused = []
        for level, (rgb, fusion) in enumerate(zip(rgb_features, self.fusion)):
            values = [rgb]
            for modality in ("T", "D"):
                values.append(auxiliary[modality][level] if coalition == "RTD" else
                              self.missing_tokens[modality][level:level + 1].expand_as(rgb))
            fused.append(rgb + fusion(torch.cat(values, dim=1)))
        features = self.shared_features(tuple(fused))
        return self.decoder(features, size)

    def forward(self, rgb: Tensor, thermal: Tensor | None = None, depth: Tensor | None = None,
                coalition: str = "RTD", input_mode: str = "multimodal") -> dict[str, object]:
        if coalition not in COALITIONS:
            raise ValueError(f"coalition must be one of {COALITIONS}")
        pyramid, auxiliary = self._prepare(rgb, thermal, depth, input_mode, full=coalition == "RTD")
        return self._decode(pyramid, auxiliary, coalition, tuple(rgb.shape[-2:]))

    def forward_coalitions(self, rgb: Tensor, thermal: Tensor | None = None, depth: Tensor | None = None,
                           input_mode: str = "multimodal") -> dict[str, dict[str, object]]:
        pyramid, auxiliary = self._prepare(rgb, thermal, depth, input_mode, full=True)
        return {coalition: self._decode(pyramid, auxiliary, coalition, tuple(rgb.shape[-2:]))
                for coalition in COALITIONS}


__all__ = ["TaskStudent", "TaskTeacher", "TaskFeatureEncoder", "TASK_MODEL_SPEC", "TASKS", "COALITIONS"]
