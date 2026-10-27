"""Capacity/encoder-compute matched teachers for the frozen v6 controls.

The original v6 implementation remains untouched so historical checkpoint
source hashes stay reproducible. Student architecture and multimodal behavior
are exactly v6; RGB control executes the shared encoder in all three slots.
"""
from __future__ import annotations
import copy
from torch import Tensor
import torch
from models.triprd_completion import (
    CompletionStudent, CompletionTeacher as V6CompletionTeacher,
    COMPLETION_MODEL_SPEC as V6_MODEL_SPEC, _prepare_cue,
)

COMPLETION_MODEL_SPEC = copy.deepcopy(V6_MODEL_SPEC)
COMPLETION_MODEL_SPEC.update(
    control_version="completion_controls_v1",
    teacher_rgb_control="three_shared_encoder_calls_same_normalized_rgb_same_adapters",
)


class CompletionTeacher(V6CompletionTeacher):
    """Same initialized parameters and three encoder calls in RGB/MM modes."""

    def forward(self, rgb: Tensor, thermal: Tensor | None = None,
                depth: Tensor | None = None, cue: Tensor | None = None,
                input_mode: str = "multimodal", perturb_cue: bool = False):
        if input_mode != "rgb":
            return super().forward(rgb, thermal, depth, cue, input_mode, perturb_cue)
        if perturb_cue and not self.training:
            raise ValueError("cue perturbation is restricted to source training")
        _, pyramid = self._rgb_features(rgb)
        size = tuple(rgb.shape[-2:])
        coarse = self.locator(self.shared_features(pyramid)[2])
        cue = _prepare_cue(cue, coarse, size, perturb_cue)
        # rgb is already ImageNet-normalized: auxiliary sensor normalization
        # would corrupt these values. Do not detach either encoder pass.
        auxiliary = {name: adapter(self.encoder(rgb), size)
                     for name, adapter in self.modal_adapters.items()}
        fused = tuple(feature + fusion(torch.cat((feature, auxiliary["T"][level],
                                                  auxiliary["D"][level]), dim=1))
                      for level, (feature, fusion) in enumerate(zip(pyramid, self.fusion)))
        output = self.refiner(self.shared_features(fused), cue, size)
        output["coarse_logits"] = coarse
        return output


__all__ = ["CompletionStudent", "CompletionTeacher", "COMPLETION_MODEL_SPEC"]
