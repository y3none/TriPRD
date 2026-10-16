"""Source-balanced COD validation metrics without a torch dependency."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure


_METRIC_NAMES = ("S_alpha", "F_beta_w", "E_phi", "MAE")


@dataclass
class _SourceMetrics:
    sm: Smeasure = field(default_factory=Smeasure)
    wfm: WeightedFmeasure = field(default_factory=WeightedFmeasure)
    em: Emeasure = field(default_factory=Emeasure)
    mae: MAE = field(default_factory=MAE)
    num_samples: int = 0

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        for metric in (self.sm, self.wfm, self.em, self.mae):
            metric.step(pred=prediction, gt=ground_truth)
        self.num_samples += 1

    def compute(self) -> dict[str, float | int]:
        return {
            "S_alpha": float(self.sm.get_results()["sm"]),
            "F_beta_w": float(self.wfm.get_results()["wfm"]),
            "E_phi": float(self.em.get_results()["em"]["curve"].mean()),
            "MAE": float(self.mae.get_results()["mae"]),
            "num_samples": self.num_samples,
        }


class CODValidationMetrics:
    """Accumulate official COD metrics separately for each validation source.

    The caller quantizes the probability map to uint8 and resizes it to the
    original GT dimensions using the same procedure as test prediction export.
    This accumulator neither resizes nor thresholds either input. Each source
    averages its images, then ``macro`` averages sources with equal weight.
    Boundary metrics are deliberately omitted from per-epoch validation.
    """

    def __init__(self) -> None:
        self._sources: dict[str, _SourceMetrics] = {}

    def update(self, source: str, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        prediction = np.asarray(prediction)
        ground_truth = np.asarray(ground_truth)
        if prediction.ndim != 2 or ground_truth.ndim != 2:
            raise ValueError("prediction and ground_truth must be 2-D grayscale arrays")
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                "prediction must already match the original ground_truth shape: "
                f"{prediction.shape} != {ground_truth.shape}"
            )
        if not prediction.size:
            raise ValueError("prediction and ground_truth must not be empty")
        if prediction.dtype != np.uint8 or ground_truth.dtype != np.uint8:
            raise TypeError("prediction and ground_truth must both have dtype uint8")

        metrics = self._sources.get(source)
        if metrics is None:
            metrics = self._sources[source] = _SourceMetrics()
        metrics.update(prediction, ground_truth)

    def compute(self) -> dict:
        if not self._sources:
            raise ValueError("cannot compute COD validation metrics without samples")
        sources = {}
        for source, metrics in self._sources.items():
            result = metrics.compute()
            for name in _METRIC_NAMES:
                if not np.isfinite(result[name]):
                    raise ValueError(f"non-finite validation metric {name} for source {source!r}")
            sources[source] = result
        macro = {
            name: float(np.mean([result[name] for result in sources.values()]))
            for name in _METRIC_NAMES
        }
        if not all(np.isfinite(value) for value in macro.values()):
            raise ValueError("non-finite source-balanced validation metrics")
        return {
            "sources": sources,
            "macro": macro,
            "num_samples": sum(result["num_samples"] for result in sources.values()),
        }
