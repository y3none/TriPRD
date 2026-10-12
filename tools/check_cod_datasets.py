#!/usr/bin/env python3
"""Validate COD10K, CAMO, NC4K, and CHAMELEON RGB/mask splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cod_dataset import CAMOIndex, COD10KIndex, CHAMELEONIndex, NC4KIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="datasets/COD_benchmark",
        help="Directory containing TrainDataset.zip, TestDataset.zip, and NC4K.zip",
    )
    parser.add_argument("--max-samples", type=int, default=200, help="Samples to read per dataset; 0 means all")
    parser.add_argument("--output", default="results/cod_dataset_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    limit = None if args.max_samples == 0 else args.max_samples
    indexes = [
        COD10KIndex(root, "train"),
        COD10KIndex(root, "test"),
        CAMOIndex(root),
        NC4KIndex(root),
        CHAMELEONIndex(root),
    ]
    reports = []
    try:
        for index in indexes:
            with index:
                reports.append(index.validate(max_samples=limit))
    finally:
        # The context managers above close each archive.  This finally block
        # keeps the command safe if validation raises before one is entered.
        for index in indexes:
            try:
                index.close()
            except Exception:
                pass

    payload = {"root": str(root), "datasets": reports}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(report["status"] == "ok" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
