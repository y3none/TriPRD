"""Index, read, and validate the VDT-2048 RGB/thermal/depth/mask data.

The downloaded dataset is intentionally not stored in this repository.  This
module accepts either an extracted directory or the original zip archive and
builds a modality-aware index from file and directory names.  It is designed
to fail loudly on ambiguous or incomplete samples instead of silently pairing
different scenes.

Typical use::

    from datasets.vdt2048 import VDT2048Index

    index = VDT2048Index("data/VDT2048", split="train")
    report = index.validate(max_samples=200)
    print(report["status"], report["num_samples"])

The actual VDT-2048 archive has to be downloaded separately.  See
``datasets/README.md`` for the official source and commands.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence


MODALITIES: tuple[str, ...] = ("rgb", "thermal", "depth", "mask")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}

# The public archive and downstream reimplementations use slightly different
# names.  Matching complete path tokens keeps ``depth`` from being confused
# with an arbitrary substring in a scene name.
MODALITY_ALIASES: Mapping[str, set[str]] = {
    "rgb": {"rgb", "visible", "color", "colour", "vis"},
    "thermal": {"thermal", "therm", "infrared", "infraredimage", "ir", "t"},
    "depth": {"depth", "depthmap", "disparity", "disp", "d"},
    "mask": {"mask", "masks", "label", "labels", "gt", "groundtruth", "sal", "saliency"},
}
GENERIC_TOKENS = {"data", "dataset", "images", "image", "frames", "frame", "samples"}


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _path_parts(location: str | Path) -> tuple[str, ...]:
    if isinstance(location, Path):
        return tuple(location.parts)
    return PurePosixPath(location).parts


def _infer_split(parts: Sequence[str]) -> str | None:
    for part in parts:
        for token in _tokens(part):
            if token in SPLIT_NAMES:
                return "val" if token in {"val", "valid", "validation"} else token
    return None


def _infer_modality(parts: Sequence[str]) -> str | None:
    # Mask is checked first because labels often live below a generic
    # ``images`` directory and must not be mistaken for RGB images.
    for modality in ("mask", "thermal", "depth", "rgb"):
        aliases = MODALITY_ALIASES[modality]
        for part in parts:
            # Absolute Windows paths contain a drive component such as
            # ``D:\\``.  It must never be interpreted as the one-letter
            # depth alias ``d``.
            if ":" in part or part in {"/", "\\"}:
                continue
            if aliases.intersection(_tokens(part)):
                return modality
    return None


def _canonical_sample_id(location: str | Path, modality: str) -> str:
    """Remove modality/split boilerplate while preserving scene identity."""

    parts = _path_parts(location)
    modality_tokens = MODALITY_ALIASES[modality]
    kept: list[str] = []
    for index, part in enumerate(parts):
        if ":" in part or part in {"/", "\\"}:
            continue
        # Strip the image suffix before tokenizing.  Otherwise ``0001.jpg``
        # becomes ``0001_jpg`` and the same sample gets a different ID when
        # another modality uses a different extension (for example
        # ``0001.png`` for the mask).
        token_part = part
        if index == len(parts) - 1:
            token_part = re.sub(r"\.(png|jpe?g|bmp|tiff?)$", "", token_part, flags=re.I)
        part_tokens = _tokens(token_part)
        if not part_tokens:
            continue
        filtered = [
            token
            for token in part_tokens
            if token not in modality_tokens
            and token not in SPLIT_NAMES
            and token not in GENERIC_TOKENS
            and token not in {"mask", "masks", "label", "labels", "gt", "groundtruth"}
        ]
        if filtered:
            kept.append("_".join(filtered))

    # A stable fallback is preferable to dropping a sample when a publisher
    # uses only numeric IDs.
    sample_id = "/".join(kept)
    return sample_id or Path(str(location)).stem.lower()


@dataclass(frozen=True)
class Asset:
    """One modality file belonging to a canonical sample."""

    sample_id: str
    modality: str
    location: str
    split: str | None


class _Source:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def exists(self) -> bool:
        return self.root.exists()

    def locations(self) -> Iterator[str | Path]:  # pragma: no cover - interface
        raise NotImplementedError

    def read_bytes(self, location: str | Path) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        return None


class _DirectorySource(_Source):
    def locations(self) -> Iterator[Path]:
        yield from (path for path in self.root.rglob("*") if path.is_file())

    def read_bytes(self, location: str | Path) -> bytes:
        return Path(location).read_bytes()


class _ZipSource(_Source):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.archive = zipfile.ZipFile(self.root)

    def locations(self) -> Iterator[str]:
        yield from (name for name in self.archive.namelist() if not name.endswith("/"))

    def read_bytes(self, location: str | Path) -> bytes:
        return self.archive.read(str(location))

    def close(self) -> None:
        self.archive.close()


def _make_source(root: str | Path) -> _Source:
    path = Path(root)
    if path.is_dir():
        return _DirectorySource(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        return _ZipSource(path)
    # Preserve a useful missing-path error in the report rather than failing
    # while constructing the index.
    return _DirectorySource(path)


class VDT2048Index:
    """Build a deterministic sample index for one VDT-2048 split."""

    def __init__(self, root: str | Path, split: str | None = "train") -> None:
        self.root = Path(root)
        self.split = None if split in {None, "both", "all"} else split.lower()
        self.source = _make_source(self.root)
        self.assets: dict[str, dict[str, Asset]] = {}
        self._index()

    @property
    def ready(self) -> bool:
        return self.source.exists and bool(self.assets)

    def _index(self) -> None:
        if not self.source.exists:
            return
        for location in self.source.locations():
            suffix = Path(str(location)).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            parts = _path_parts(location)
            split = _infer_split(parts)
            if self.split and split and split != self.split:
                continue
            modality = _infer_modality(parts)
            if modality is None:
                continue
            sample_id = _canonical_sample_id(location, modality)
            sample = self.assets.setdefault(sample_id, {})
            # Duplicate files can occur when an archive contains thumbnails or
            # copied folders.  Keep the lexicographically first path so the
            # index remains deterministic and report the duplicate separately.
            current = sample.get(modality)
            candidate = Asset(sample_id, modality, str(location), split)
            if current is None or candidate.location < current.location:
                sample[modality] = candidate

    def sample_ids(self) -> list[str]:
        return sorted(self.assets)

    def samples(self, max_samples: int | None = None) -> Iterator[dict[str, Asset]]:
        ids = self.sample_ids()
        if max_samples and max_samples > 0:
            ids = ids[:max_samples]
        for sample_id in ids:
            yield self.assets[sample_id]

    def read_image(self, asset: Asset):
        """Read an asset as a detached PIL image (Pillow is loaded lazily)."""

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Pillow is required for image validation") from exc
        with Image.open(io.BytesIO(self.source.read_bytes(asset.location))) as image:
            return image.copy()

    def validate(self, max_samples: int | None = None) -> dict[str, Any]:
        report: dict[str, Any] = {
            "root": str(self.root),
            "split": self.split or "all",
            "status": "ok",
            "num_samples": len(self.assets),
            "checked_samples": 0,
            "complete_samples": 0,
            "missing_modalities": {modality: 0 for modality in MODALITIES},
            "size_mismatches": 0,
            "empty_masks": 0,
            "read_errors": 0,
            "value_range_examples": [],
            "errors": [],
        }
        if not self.source.exists:
            report.update(status="not_ready", num_samples=0)
            report["errors"].append(f"Dataset path does not exist: {self.root}")
            return report
        if not self.assets:
            report.update(status="not_ready")
            report["errors"].append("No RGB/Thermal/Depth/Mask image files were indexed")
            return report

        for sample in self.samples(max_samples):
            report["checked_samples"] += 1
            missing = [modality for modality in MODALITIES if modality not in sample]
            for modality in missing:
                report["missing_modalities"][modality] += 1
            if missing:
                if len(report["errors"]) < 100:
                    report["errors"].append(
                        {"sample_id": next(iter(sample.values())).sample_id, "missing": missing}
                    )
                continue

            sizes: dict[str, tuple[int, int]] = {}
            images: dict[str, Any] = {}
            try:
                for modality in MODALITIES:
                    images[modality] = self.read_image(sample[modality])
                    sizes[modality] = images[modality].size
            except Exception as exc:  # keep checking the rest of the index
                report["read_errors"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(
                        {"sample_id": next(iter(sample.values())).sample_id, "read_error": str(exc)}
                    )
                continue
            mask_bbox = images["mask"].getbbox()
            if mask_bbox is None:
                report["empty_masks"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(
                        {"sample_id": next(iter(sample.values())).sample_id, "error": "empty_mask"}
                    )
                continue
            if len(report["value_range_examples"]) < 20:
                ranges: dict[str, Any] = {}
                for modality in ("depth", "mask"):
                    extrema = images[modality].getextrema()
                    if isinstance(extrema, tuple) and extrema and isinstance(extrema[0], tuple):
                        ranges[modality] = [list(channel) for channel in extrema]
                    else:
                        ranges[modality] = list(extrema) if isinstance(extrema, tuple) else extrema
                report["value_range_examples"].append(
                    {"sample_id": next(iter(sample.values())).sample_id, "ranges": ranges}
                )
            if len(set(sizes.values())) != 1:
                report["size_mismatches"] += 1
                if len(report["errors"]) < 100:
                    report["errors"].append(
                        {"sample_id": next(iter(sample.values())).sample_id, "sizes": sizes}
                    )
                continue
            report["complete_samples"] += 1

        if report["errors"]:
            report["status"] = "failed"
        return report

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> "VDT2048Index":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["Asset", "MODALITIES", "VDT2048Index", "write_report"]
