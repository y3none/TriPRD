"""Unified RGB/mask readers and split checks for the COD benchmarks.

The public SINet-style release stores the standard 4040-image training pool
(3040 COD10K-train + 1000 CAMO-train) in ``TrainDataset.zip``.  The two
datasets are separated below by their official sample-name prefixes.  The
COD10K/CAMO/CHAMELEON test data share ``TestDataset.zip`` and NC4K is a
separate archive.  Both archives and extracted directories are supported.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MODALITY_DIRS = {
    "image": {"imgs", "images", "image", "rgb"},
    "mask": {"gt", "groundtruth", "mask", "masks", "label", "labels"},
    "edge": {"edge", "edges", "boundary", "boundaries"},
    "instance": {"instance", "instances"},
}


DATASET_CONFIG: Mapping[str, Mapping[str, Any]] = {
    "COD10K": {
        "splits": {"train", "test"},
        "archives": {"train": "TrainDataset.zip", "test": "TestDataset.zip"},
        "prefixes": {
            "train": ("TrainDataset",),
            "test": ("TestDataset/COD10K", "COD10K"),
        },
        "sample_prefixes": {"train": ("COD10K-",)},
        "expected_counts": {"train": 3040, "test": 2026},
    },
    "CAMO": {
        "splits": {"train", "test"},
        "archives": {"train": "TrainDataset.zip", "test": "TestDataset.zip"},
        "prefixes": {
            "train": ("TrainDataset",),
            "test": ("TestDataset/CAMO", "CAMO"),
        },
        "sample_prefixes": {"train": ("camourflage_",)},
        "expected_counts": {"train": 1000, "test": 250},
    },
    "CHAMELEON": {
        "splits": {"test"},
        "archives": {"test": "TestDataset.zip"},
        "prefixes": {"test": ("TestDataset/CHAMELEON", "CHAMELEON")},
        "expected_counts": {"test": 76},
        # The official CODToolbox keeps GT as the reference and resizes each
        # prediction to GT size before computing metrics.
        "evaluation_size_policy": "resize_prediction_to_gt",
    },
    "NC4K": {
        "splits": {"test"},
        "archives": {"test": "NC4K.zip"},
        "prefixes": {"test": ("", "NC4K")},
        "expected_counts": {"test": 4121},
    },
}


def _normalise(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip("/")


def _stem_key(name: str) -> str:
    return PurePosixPath(name).stem.casefold()


def _modality(directory: str) -> str | None:
    return next(
        (canonical for canonical, aliases in MODALITY_DIRS.items() if directory.casefold() in aliases),
        None,
    )


class _Source:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def exists(self) -> bool:
        return self.root.exists()

    def locations(self) -> Iterator[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def read_bytes(self, location: str) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None


class _DirectorySource(_Source):
    def locations(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for path in self.root.rglob("*"):
            if path.is_file():
                yield _normalise(path.relative_to(self.root))

    def read_bytes(self, location: str) -> bytes:
        return (self.root / Path(location)).read_bytes()


class _ZipSource(_Source):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.archive = zipfile.ZipFile(root)

    def locations(self) -> Iterator[str]:
        yield from (_normalise(name) for name in self.archive.namelist() if not name.endswith("/"))

    def read_bytes(self, location: str) -> bytes:
        return self.archive.read(location)

    def close(self) -> None:
        self.archive.close()


@dataclass(frozen=True)
class CODAsset:
    """One RGB, mask, edge, or instance file."""

    sample_id: str
    modality: str
    location: str


@dataclass(frozen=True)
class CODSample:
    """A paired RGB image and ground-truth mask."""

    dataset: str
    split: str
    sample_id: str
    image: CODAsset | None
    mask: CODAsset | None
    edge: CODAsset | None = None
    instance: CODAsset | None = None


def _choose_archive(root: Path, dataset: str, split: str) -> Path:
    """Use the dataset-specific archive when a parent directory is provided."""

    archive_name = str(DATASET_CONFIG[dataset]["archives"][split])
    candidate = root / archive_name
    return candidate if candidate.is_file() else root


def _discover_groups(source: _Source) -> dict[str, dict[str, list[str]]]:
    """Group image paths by the parent of Imgs/GT/Edge/Instance."""

    groups: dict[str, dict[str, list[str]]] = {}
    for location in source.locations():
        path = PurePosixPath(location)
        if path.suffix.casefold() not in IMAGE_EXTENSIONS or len(path.parts) < 2:
            continue
        modality = _modality(path.parent.name)
        if modality is None:
            continue
        prefix = "/".join(path.parent.parent.parts)
        groups.setdefault(prefix, {}).setdefault(modality, []).append(location)
    return groups


def _select_prefix(groups: Mapping[str, Mapping[str, list[str]]], dataset: str, split: str) -> str | None:
    config = DATASET_CONFIG[dataset]
    expected = {_normalise(item).casefold() for item in config["prefixes"][split]}
    complete = {prefix: modalities for prefix, modalities in groups.items() if "image" in modalities and "mask" in modalities}
    if not complete:
        return None

    for prefix in complete:
        if prefix.casefold() in expected:
            return prefix

    def score(prefix: str) -> tuple[int, int, int]:
        value = prefix.casefold()
        dataset_match = int(dataset.casefold() in value)
        split_match = int(split.casefold() in value or (dataset == "NC4K" and not value))
        direct = int(not prefix)
        return dataset_match, split_match, direct

    candidates = sorted(complete, key=lambda item: (score(item), item.casefold()), reverse=True)
    best = candidates[0]
    best_score = score(best)
    # A root containing several unrelated datasets must not silently choose
    # one when the path gives no dataset identity.
    if len(candidates) > 1 and best_score == score(candidates[1]) and best_score == (0, 0, 0):
        return None
    # A caller may point directly at one extracted dataset directory, in
    # which case its relative prefix is empty and the parent path carries the
    # dataset identity.  Only reject an unknown identity when several layouts
    # compete for selection.
    if len(complete) > 1 and dataset != "NC4K" and best_score[0] == 0:
        return None
    return best


def _assert_split_identity(requested_root: Path, prefix: str | None, dataset: str, split: str) -> str:
    """Return passed/unknown without treating an ambiguous COD10K root as safe."""

    if prefix is None:
        return "failed"
    if len(DATASET_CONFIG[dataset]["splits"]) == 1:
        return "passed"
    identity = f"{prefix}/{requested_root}".casefold()
    return "passed" if split.casefold() in identity else "unknown"


class CODDatasetIndex:
    """Index one official COD dataset split from a directory or ZIP archive."""

    def __init__(self, root: str | Path, dataset: str, split: str) -> None:
        dataset = dataset.upper()
        split = split.lower()
        if dataset not in DATASET_CONFIG:
            raise ValueError(f"Unsupported COD dataset: {dataset}")
        if split not in DATASET_CONFIG[dataset]["splits"]:
            allowed = ", ".join(sorted(DATASET_CONFIG[dataset]["splits"]))
            raise ValueError(f"{dataset} only supports split(s): {allowed}; got {split}")

        self.dataset = dataset
        self.split = split
        self.requested_root = Path(root)
        source_root = _choose_archive(self.requested_root, dataset, split)
        self.source: _Source = _ZipSource(source_root) if source_root.is_file() and source_root.suffix.casefold() == ".zip" else _DirectorySource(source_root)
        self.source_root = source_root
        self.groups = _discover_groups(self.source) if self.source.exists else {}
        self.prefix = _select_prefix(self.groups, dataset, split)
        self.split_assertion = _assert_split_identity(self.requested_root, self.prefix, dataset, split)
        self.layout_error: str | None = None
        if self.source.exists and self.prefix is None:
            self.layout_error = f"Could not find an unambiguous Imgs/GT layout for {dataset} {split} under {source_root}"
        elif self.source.exists and self.split_assertion == "unknown":
            self.layout_error = f"Could not prove the official {dataset} {split} split from path {source_root}"
        self.assets: dict[str, dict[str, CODAsset]] = {}
        if self.prefix is not None:
            self._index(self.groups[self.prefix])

    @property
    def ready(self) -> bool:
        return self.source.exists and self.prefix is not None and bool(self.assets)

    def _index(self, modalities: Mapping[str, list[str]]) -> None:
        for modality, locations in modalities.items():
            by_id: dict[str, str] = {}
            for location in sorted(locations):
                sample_id = PurePosixPath(location).stem
                allowed_prefixes = DATASET_CONFIG[self.dataset].get("sample_prefixes", {}).get(self.split, ())
                if allowed_prefixes and not any(
                    sample_id.casefold().startswith(prefix.casefold()) for prefix in allowed_prefixes
                ):
                    continue
                by_id.setdefault(_stem_key(location), location)
            for key, location in by_id.items():
                self.assets.setdefault(key, {})[modality] = CODAsset(
                    sample_id=PurePosixPath(location).stem,
                    modality=modality,
                    location=location,
                )

    def sample_ids(self) -> list[str]:
        return sorted(self.assets)

    def samples(self, max_samples: int | None = None) -> Iterator[CODSample]:
        ids = self.sample_ids()
        if max_samples is not None and max_samples > 0:
            ids = ids[:max_samples]
        for key in ids:
            assets = self.assets[key]
            yield CODSample(
                dataset=self.dataset,
                split=self.split,
                sample_id=next(iter(assets.values())).sample_id,
                image=assets.get("image"),
                mask=assets.get("mask"),
                edge=assets.get("edge"),
                instance=assets.get("instance"),
            )

    def __len__(self) -> int:
        return len(self.assets)

    def read_image(self, asset: CODAsset):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Pillow is required for COD image validation") from exc
        with Image.open(io.BytesIO(self.source.read_bytes(asset.location))) as image:
            return image.copy()

    def read_pair(self, sample: CODSample, *, strict_size: bool = True):
        """Read an RGB/mask pair, optionally enforcing equal source sizes."""

        if sample.image is None or sample.mask is None:
            missing = [name for name, asset in (("image", sample.image), ("mask", sample.mask)) if asset is None]
            raise ValueError(f"Sample {sample.sample_id} is missing: {', '.join(missing)}")
        image = self.read_image(sample.image)
        mask = self.read_image(sample.mask)
        if strict_size and image.size != mask.size:
            raise ValueError(f"Sample {sample.sample_id} has image size {image.size} and mask size {mask.size}")
        return image, mask

    def validate(self, max_samples: int | None = None) -> dict[str, Any]:
        report: dict[str, Any] = {
            "dataset": self.dataset,
            "split": self.split,
            "root": str(self.requested_root),
            "source": str(self.source_root),
            "resolved_prefix": self.prefix,
            "split_assertion": self.split_assertion,
            "expected_samples": DATASET_CONFIG[self.dataset].get("expected_counts", {}).get(self.split),
            "sample_count_mismatch": False,
            "evaluation_size_policy": DATASET_CONFIG[self.dataset].get("evaluation_size_policy", "require_equal"),
            "evaluation_ready": False,
            "status": "ok",
            "num_samples": len(self.assets),
            "checked_samples": 0,
            "complete_samples": 0,
            "missing_image": 0,
            "missing_mask": 0,
            "size_mismatches": 0,
            "empty_masks": 0,
            "read_errors": 0,
            "optional_edge": sum("edge" in assets for assets in self.assets.values()),
            "optional_instance": sum("instance" in assets for assets in self.assets.values()),
            "errors": [],
        }
        if not self.source.exists:
            report.update(status="not_ready", split_assertion="failed")
            report["errors"].append(f"Dataset path does not exist: {self.source_root}")
            return report
        if self.layout_error:
            report.update(status="not_ready")
            report["errors"].append(self.layout_error)
            return report
        if not self.assets:
            report.update(status="not_ready")
            report["errors"].append("No RGB/mask image files were indexed")
            return report

        expected_samples = report["expected_samples"]
        if expected_samples is not None and len(self.assets) != expected_samples:
            report["sample_count_mismatch"] = True
            report["errors"].append(
                f"Expected {expected_samples} samples for the official {self.dataset} {self.split} split, "
                f"indexed {len(self.assets)}"
            )

        for sample in self.samples(max_samples):
            report["checked_samples"] += 1
            if sample.image is None:
                report["missing_image"] += 1
            if sample.mask is None:
                report["missing_mask"] += 1
            if sample.image is None or sample.mask is None:
                if len(report["errors"]) < 100:
                    report["errors"].append({"sample_id": sample.sample_id, "missing": [name for name, asset in (("image", sample.image), ("mask", sample.mask)) if asset is None]})
                continue
            try:
                image = self.read_image(sample.image)
                mask = self.read_image(sample.mask)
                if image.size != mask.size:
                    report["size_mismatches"] += 1
                    if len(report["errors"]) < 100:
                        report["errors"].append({"sample_id": sample.sample_id, "image_size": image.size, "mask_size": mask.size})
                    continue
                if mask.convert("L").getbbox() is None:
                    report["empty_masks"] += 1
                    if len(report["errors"]) < 100:
                        report["errors"].append({"sample_id": sample.sample_id, "error": "empty_mask"})
                    continue
                report["complete_samples"] += 1
            except Exception as exc:
                report["read_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append({"sample_id": sample.sample_id, "read_error": str(exc)})

        hard_errors = (
            report["missing_image"]
            or report["missing_mask"]
            or report["empty_masks"]
            or report["read_errors"]
            or report["sample_count_mismatch"]
            or report["split_assertion"] != "passed"
        )
        report["evaluation_ready"] = not hard_errors and (
            report["size_mismatches"] == 0
            or report["evaluation_size_policy"] == "resize_prediction_to_gt"
        )
        if report["evaluation_ready"]:
            report["status"] = "ok_with_alignment" if report["size_mismatches"] else "ok"
        else:
            report["status"] = "failed"
        return report

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> "CODDatasetIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class COD10KIndex(CODDatasetIndex):
    def __init__(self, root: str | Path, split: str) -> None:
        super().__init__(root, "COD10K", split)


class CAMOIndex(CODDatasetIndex):
    def __init__(self, root: str | Path, split: str = "test") -> None:
        super().__init__(root, "CAMO", split)


class CHAMELEONIndex(CODDatasetIndex):
    def __init__(self, root: str | Path, split: str = "test") -> None:
        super().__init__(root, "CHAMELEON", split)


class NC4KIndex(CODDatasetIndex):
    def __init__(self, root: str | Path, split: str = "test") -> None:
        super().__init__(root, "NC4K", split)


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "CAMOIndex",
    "COD10KIndex",
    "CODAsset",
    "CODDatasetIndex",
    "CODSample",
    "CHAMELEONIndex",
    "NC4KIndex",
    "write_report",
]
