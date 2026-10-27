"""Shared RGB object-conditioned completion and privileged source teacher.

Only coarse locators are task-specific. The whole three-stage mask/boundary
refiner is inherited from source to COD. No GT cue or hard ROI is used.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from models.triprd_correction import normalize_auxiliary
from models.triprd_task import TASKS, _TaskBase, _Pyramid, _Refine, _conv, _resize

COMPLETION_MODEL_SPEC = {
    "version": "object_conditioned_completion_v6",
    "stage_strides": [16, 8, 4],
    "locator": "rgb_only_source_cue_detached_cod_generated_cue_differentiable",
    "refiner": "shared_full_image_foreground_background_global_context",
    "boundary": "predicted_boundary_enters_next_mask_stage",
    "source_transfer": "all_student_parameters_except_cod_locator",
    "teacher": "real_rgb_thermal_depth_values_same_rgb_cue_as_student",
    "student_inference": "rgb_only_single_pass_three_stages",
    "cue_perturbation": "source_train_prediction_only_local_holes_false_positives_dilation",
}


def _prepare_cue(cue: Tensor | None, coarse: Tensor, size: tuple[int, int], perturb: bool,
                 detach_generated: bool = True) -> Tensor:
    """Source/external cues detach; a generated COD cue can train its locator."""
    if not isinstance(perturb, bool):
        raise ValueError("perturb_cue must be boolean")
    generated = cue is None
    if generated:
        logits = coarse.detach() if detach_generated else coarse
        cue = _resize(logits.float().sigmoid(), size)
    elif (not isinstance(cue, Tensor) or cue.ndim != 4 or cue.shape[:2] != coarse.shape[:2]
          or cue.shape[-2:] != size or cue.device != coarse.device or not cue.is_floating_point()
          or not torch.isfinite(cue).all() or bool((cue < 0).any()) or bool((cue > 1).any())):
        raise ValueError("cue must be finite Bx1xHxW probabilities aligned to RGB")
    cue = cue.float() if generated and not detach_generated else cue.detach().float()
    if not perturb:
        return cue
    # The caller restricts perturbation to source TRAIN. In KD, it passes this
    # exact returned cue to the teacher without perturbing a second time.
    b, _, h, w = cue.shape
    changed = cue.clone()
    for index in range(b):
        operation = int(torch.randint(4, (), device=cue.device))
        if operation == 0:
            continue
        if operation == 3:
            changed[index:index + 1] = F.max_pool2d(cue[index:index + 1], 5, 1, 2)
            continue
        rh, rw = max(1, h // 5), max(1, w // 5)
        top = int(torch.randint(h - rh + 1, (), device=cue.device))
        left = int(torch.randint(w - rw + 1, (), device=cue.device))
        region = changed[index:index + 1, :, top:top + rh, left:left + rw]
        if operation == 1:
            region.mul_(.2)
        else:
            region.mul_(.5).add_(.5)
    return changed


def _object_context(feature: Tensor, cue: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """FP32 soft prototypes; empty classes fall back to global context."""
    with torch.autocast(device_type=feature.device.type, enabled=False):
        value, probability = feature.float(), _resize(cue.float(), feature.shape[-2:])
        global_context = value.mean((-2, -1), keepdim=True)
        prototypes = []
        for mass in (probability, 1 - probability):
            denominator = mass.sum((-2, -1), keepdim=True)
            prototype = (value * mass).sum((-2, -1), keepdim=True) / denominator.clamp_min(1e-6)
            prototypes.append(torch.where(denominator > 1e-6, prototype, global_context))
        normalized = F.normalize(value, dim=1, eps=1e-6)
        similarities = [(normalized * F.normalize(p, dim=1, eps=1e-6)).sum(1, keepdim=True)
                        for p in prototypes]
        descriptors = torch.cat((probability, 4 * probability * (1 - probability),
                                 similarities[0], similarities[1], similarities[0] - similarities[1]), 1)
    return prototypes[0], prototypes[1], global_context, descriptors


class _CompletionStage(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.foreground = nn.Conv2d(width, width, 1, bias=False)
        self.background = nn.Conv2d(width, width, 1, bias=False)
        self.global_context = nn.Conv2d(width, width, 1, bias=False)
        self.previous = nn.Conv2d(width, width, 1, bias=False)
        self.condition = _conv(6, width, kernel=1)
        self.refine = nn.Sequential(_conv(width * 2, width, kernel=1), _Refine(width))
        self.mask = nn.Conv2d(width, 1, 1)
        self.boundary = nn.Sequential(_conv(width, width), nn.Conv2d(width, 1, 1))

    def forward(self, feature: Tensor, cue: Tensor, previous_boundary: Tensor,
                previous: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        fg, bg, global_context, descriptors = _object_context(feature, cue)
        context = (feature + self.foreground(fg) + self.background(bg)
                   + self.global_context(global_context)
                   + self.previous(_resize(previous, feature.shape[-2:])))
        condition = self.condition(torch.cat((descriptors,
            _resize(previous_boundary.float(), feature.shape[-2:])), dim=1))
        state = self.refine(torch.cat((context, condition), dim=1))
        # Absolute full-image logits allow correction outside the initial cue.
        return state, self.mask(state), self.boundary(state)


class _CompletionRefiner(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.stages = nn.ModuleList([_CompletionStage(width) for _ in range(3)])

    def forward(self, features: tuple[Tensor, ...], cue: Tensor,
                size: tuple[int, int]) -> dict[str, object]:
        state = features[2]
        prior, edge = cue, torch.zeros_like(cue)
        masks, boundaries = [], []
        for stage, feature in zip(self.stages, (features[2], features[1], features[0])):
            state, mask, boundary = stage(feature, prior, edge, state)
            masks.append(mask)
            boundaries.append(boundary)
            prior, edge = mask.sigmoid(), boundary.sigmoid()
        return {"logits": _resize(masks[-1], size), "stage_logits": tuple(masks),
                "stage_boundaries": tuple(boundaries),
                "boundary_logits": _resize(boundaries[-1], size), "cue": cue}


def _locator(width: int) -> nn.Module:
    return nn.Sequential(_Refine(width), nn.Conv2d(width, 1, 1))


class CompletionStudent(_TaskBase):
    """Source/COD share a complete RGB mask-repair function, including heads."""
    def __init__(self, model_variant: str = "dinov3_vitb16", base_channels: int = 128,
                 pretrained: bool = True, dino_repo: str | Path | None = None,
                 dino_weights: str | Path | None = None, train_backbone_blocks: int = 4,
                 detail_branch: bool = True):
        super().__init__(model_variant, base_channels, pretrained, dino_repo, dino_weights,
                         train_backbone_blocks, detail_branch)
        self.locators = nn.ModuleDict({task: _locator(base_channels) for task in TASKS})
        self.refiner = _CompletionRefiner(base_channels)

    def forward(self, rgb: Tensor, task: str = "cod", cue: Tensor | None = None,
                perturb_cue: bool = False) -> dict[str, object]:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if perturb_cue and (task != "sod" or not self.training):
            raise ValueError("cue perturbation is restricted to source training")
        _, pyramid = self._rgb_features(rgb)
        features = self.shared_features(pyramid)
        coarse = self.locators[task](features[2])
        cue = _prepare_cue(cue, coarse, tuple(rgb.shape[-2:]), perturb_cue,
                           detach_generated=task == "sod")
        output = self.refiner(features, cue, tuple(rgb.shape[-2:]))
        output["coarse_logits"] = coarse
        return output


class CompletionTeacher(_TaskBase):
    """T/D demonstrate repair; initial localization remains strictly RGB-only.

    In distillation pass the student's cue. RGB control mode feeds the same
    cached RGB encoding through all modal adapters at identical capacity.
    """
    def __init__(self, model_variant: str = "dinov3_vitb16", base_channels: int = 128,
                 pretrained: bool = True, dino_repo: str | Path | None = None,
                 dino_weights: str | Path | None = None, train_backbone_blocks: int = 4,
                 detail_branch: bool = True):
        super().__init__(model_variant, base_channels, pretrained, dino_repo, dino_weights,
                         train_backbone_blocks, detail_branch)
        self.locator = _locator(base_channels)
        self.modal_adapters = nn.ModuleDict({
            modality: _Pyramid(self.encoder.channels, base_channels) for modality in ("T", "D")})
        self.fusion = nn.ModuleList([
            nn.Sequential(_conv(base_channels * 3, base_channels, kernel=1), _Refine(base_channels))
            for _ in range(4)])
        self.refiner = _CompletionRefiner(base_channels)

    def _auxiliary_features(self, value: Tensor | None, rgb: Tensor, modality: str) -> tuple[Tensor, ...]:
        if (not isinstance(value, Tensor) or value.ndim != 4 or value.shape[0] != rgb.shape[0]
                or value.shape[-2:] != rgb.shape[-2:]):
            raise ValueError(f"multimodal input requires aligned {modality} observations")
        encoded = self.encoder(normalize_auxiliary(value, depth=modality == "D"))
        return self.modal_adapters[modality](encoded, tuple(rgb.shape[-2:]))

    def forward(self, rgb: Tensor, thermal: Tensor | None = None, depth: Tensor | None = None,
                cue: Tensor | None = None, input_mode: str = "multimodal",
                perturb_cue: bool = False) -> dict[str, object]:
        if input_mode not in ("multimodal", "rgb"):
            raise ValueError("input_mode must be 'multimodal' or 'rgb'")
        if perturb_cue and not self.training:
            raise ValueError("cue perturbation is restricted to source training")
        encoded, pyramid = self._rgb_features(rgb)
        size = tuple(rgb.shape[-2:])
        rgb_features = self.shared_features(pyramid)
        coarse = self.locator(rgb_features[2])
        cue = _prepare_cue(cue, coarse, size, perturb_cue)
        if input_mode == "rgb":
            auxiliary = {name: adapter(encoded, size) for name, adapter in self.modal_adapters.items()}
        else:
            auxiliary = {"T": self._auxiliary_features(thermal, rgb, "T"),
                         "D": self._auxiliary_features(depth, rgb, "D")}
        fused = tuple(feature + fusion(torch.cat((feature, auxiliary["T"][level],
                                                  auxiliary["D"][level]), dim=1))
                      for level, (feature, fusion) in enumerate(zip(pyramid, self.fusion)))
        output = self.refiner(self.shared_features(fused), cue, size)
        output["coarse_logits"] = coarse
        return output


__all__ = ["CompletionStudent", "CompletionTeacher", "COMPLETION_MODEL_SPEC"]
