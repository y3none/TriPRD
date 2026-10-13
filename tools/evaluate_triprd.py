#!/usr/bin/env python3
"""Evaluate TriPRD prediction folders with the project COD metric protocol.

This thin entry point reuses ``evaluate_cod.py`` so the feasibility run and
the planned paper tables use the same S-measure, weighted F-measure,
E-measure, MAE and boundary metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    from evaluate_cod import _index_for_dataset, evaluate_dataset
except ImportError as exc:  # pragma: no cover - depends on experiment environment
    raise SystemExit("COD evaluation requires numpy, scipy and opencv-python") from exc


DATASETS = ("NC4K", "COD10K", "CAMO", "CHAMELEON")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default="datasets/COD_benchmark")
    parser.add_argument("--pred-root", default="results/predictions/triprd")
    parser.add_argument("--dataset", action="append", choices=DATASETS, dest="datasets")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model", default="TriPRD-feasibility")
    parser.add_argument("--boundary", choices=("on", "off"), default="on")
    parser.add_argument("--boundary-rate", type=int, default=15)
    parser.add_argument("--output-json", default="results/triprd_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for dataset in args.datasets or list(DATASETS):
        index = _index_for_dataset(Path(args.data_root), dataset)
        try:
            result = evaluate_dataset(
                index=index,
                prediction_root=Path(args.pred_root),
                dataset=dataset,
                model=args.model,
                boundary=args.boundary == "on",
                boundary_rate=args.boundary_rate,
                max_samples=args.max_samples or None,
            )
            results.append(result)
        finally:
            index.close()
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
