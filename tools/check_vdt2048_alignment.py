#!/usr/bin/env python3
"""Run the VDT2048 file-pairing and image-size preflight check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.vdt2048 import VDT2048Index, write_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extracted VDT2048 directory or .zip archive")
    parser.add_argument("--split", default="train", choices=["train", "test", "val", "both", "all"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--output",
        default="results/vdt2048_alignment_report.json",
        help="JSON report path, relative to the current directory",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Return a non-zero exit code when the dataset has not been downloaded",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with VDT2048Index(args.root, split=args.split) as index:
        report = index.validate(max_samples=args.max_samples)
    write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["status"] == "not_ready":
        print("[INFO] VDT2048 尚未就绪；已生成 preflight 报告，没有伪造样本结果。", file=sys.stderr)
        return 2 if args.require_data else 0
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
