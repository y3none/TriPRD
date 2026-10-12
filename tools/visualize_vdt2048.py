#!/usr/bin/env python3
"""Create RGB | Thermal | Depth | Mask panels for VDT2048 samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.vdt2048 import MODALITIES, VDT2048Index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extracted VDT2048 directory or .zip archive")
    parser.add_argument("--split", default="train", choices=["train", "test", "val", "both", "all"])
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--output", default="visualizations/vdt2048_train")
    return parser.parse_args()


def _panel_image(image, modality: str):
    from PIL import Image, ImageDraw, ImageOps

    # Keep RGB natural; normalize auxiliary modalities only for display.  The
    # source file remains untouched and all normalization is local to the PNG.
    if modality == "rgb":
        display = image.convert("RGB")
    elif modality == "mask":
        display = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    else:
        display = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    draw = ImageDraw.Draw(display)
    draw.rectangle((0, 0, display.width - 1, display.height - 1), outline=(255, 255, 0), width=2)
    draw.rectangle((0, 0, min(display.width, 180), 28), fill=(0, 0, 0))
    draw.text((8, 6), modality.upper(), fill=(255, 255, 255))
    return display


def _make_panel(images: dict[str, object]):
    from PIL import Image, ImageDraw

    prepared = [_panel_image(images[modality], modality) for modality in MODALITIES]
    width = max(image.width for image in prepared)
    height = max(image.height for image in prepared)
    gap = 8
    canvas = Image.new("RGB", (width * 2 + gap * 3, height * 2 + gap * 3), "white")
    positions = [(gap, gap), (width + gap * 2, gap), (gap, height + gap * 2), (width + gap * 2, height + gap * 2)]
    for image, (x, y) in zip(prepared, positions):
        canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, canvas.height - gap - 16), "VDT2048 four-panel preview", fill=(0, 0, 0))
    return canvas


def main() -> int:
    args = parse_args()
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("需要 Pillow：pip install pillow", file=sys.stderr)
        return 2

    with VDT2048Index(args.root, split=args.split) as index:
        if not index.ready:
            print(f"[INFO] 数据不存在或没有可识别样本：{args.root}", file=sys.stderr)
            print("请先下载 VDT2048，再重新运行本脚本。", file=sys.stderr)
            return 2
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        written = 0
        for sample in index.samples(max_samples=args.max_images):
            if any(modality not in sample for modality in MODALITIES):
                continue
            try:
                images = {modality: index.read_image(sample[modality]) for modality in MODALITIES}
                panel = _make_panel(images)
                safe_id = sample["rgb"].sample_id.replace("/", "__").replace("\\", "__")
                panel.save(output / f"{safe_id}.jpg", quality=92)
                written += 1
            except Exception as exc:
                print(f"[WARN] {sample}: {exc}", file=sys.stderr)
        print(f"Saved {written} four-panel visualizations to {output.resolve()}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
