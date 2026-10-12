#!/usr/bin/env python3
"""Evaluate COD predictions with CamoFormer's original metric classes.

The metric implementation remains the vendored CamoFormer ``metrics.py``.
This project entry point only adapts the data source and experiment registry:
ground-truth masks can stay in the official COD ZIP files, predictions live in
``<pred-root>/<dataset>/``, and resized predictions are kept in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the experiment environment
    raise SystemExit("COD evaluation requires numpy and opencv-python") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(items, **_: Any):
        return items

try:
    from datasets.cod_dataset import CAMOIndex, COD10KIndex, CHAMELEONIndex, NC4KIndex
    from evaluation.camoformer_metrics import (
        Emeasure,
        Fmeasure_and_FNR,
        MAE,
        Smeasure,
        WeightedFmeasure,
    )
except ImportError as exc:  # pragma: no cover - depends on the experiment environment
    raise SystemExit("COD evaluation requires the project dependencies; install numpy scipy opencv-python") from exc


DATASETS = ("NC4K", "COD10K", "CAMO", "CHAMELEON")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CSV_FIELDS = [
    "experiment_id",
    "dataset",
    "split",
    "backbone",
    "input_size",
    "seed",
    "teacher_type",
    "student_type",
    "distillation_type",
    "lambda_logit",
    "lambda_relation",
    "lambda_boundary",
    "lambda_uncertainty",
    "lambda_prior",
    "best_epoch",
    "S_alpha",
    "F_beta_w",
    "E_phi",
    "MAE",
    "Boundary_F",
    "params",
    "FLOPs",
    "FPS",
    "checkpoint",
    "notes",
]


def borders_capture(gt, pred, dksize: int = 15):
    """The boundary procedure copied from CamoFormer ``eval.py``."""

    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = gt.copy()
    image[:] = 0
    cv2.drawContours(image, contours, -1, (255, 255, 255), 3)
    kernel = np.ones((dksize, dksize), np.uint8)
    image_dilate = cv2.dilate(image, kernel)

    response = cv2.bitwise_and(image_dilate, gt)
    b, g, r = cv2.split(response)
    alpha = np.rollaxis(image_dilate, 2, 0)[0]
    merge = cv2.merge((b, g, r, alpha))

    response_pred = cv2.bitwise_and(image_dilate, pred)
    b, g, r = cv2.split(response_pred)
    alpha = np.rollaxis(image_dilate, 2, 0)[0]
    merge_pred = cv2.merge((b, g, r, alpha))

    merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
    merge_pred = cv2.cvtColor(merge_pred, cv2.COLOR_RGB2GRAY)
    return merge, merge_pred, np.sum(image_dilate) / 255


def _index_for_dataset(root: Path, dataset: str):
    if dataset == "COD10K":
        return COD10KIndex(root, "test")
    if dataset == "CAMO":
        return CAMOIndex(root, "test")
    if dataset == "NC4K":
        return NC4KIndex(root, "test")
    if dataset == "CHAMELEON":
        return CHAMELEONIndex(root, "test")
    raise ValueError(f"Unsupported dataset: {dataset}")


def _prediction_paths(root: Path, dataset: str) -> list[Path]:
    directory = root / dataset
    if not directory.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _sample_map(index) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for sample in index.samples():
        mapping[sample.sample_id.casefold()] = sample
        if sample.mask is not None:
            mapping[Path(sample.mask.location).stem.casefold()] = sample
    return mapping


def evaluate_dataset(
    *,
    index,
    prediction_root: Path,
    dataset: str,
    model: str,
    boundary: bool,
    boundary_rate: int,
    max_samples: int | None,
) -> dict[str, Any]:
    predictions = _prediction_paths(prediction_root, dataset)
    if max_samples is not None and max_samples > 0:
        predictions = predictions[:max_samples]
    if not predictions:
        raise ValueError(f"No prediction images found for {dataset}")

    samples = _sample_map(index)
    fm = Fmeasure_and_FNR()
    wfm = WeightedFmeasure()
    sm = Smeasure()
    em = Emeasure()
    mae = MAE()
    boundary_mae = MAE() if boundary else None
    boundary_f = WeightedFmeasure() if boundary else None

    for prediction_path in tqdm(predictions, total=len(predictions), desc=dataset):
        sample = samples.get(prediction_path.stem.casefold())
        if sample is None or sample.mask is None:
            raise FileNotFoundError(f"No GT mask indexed for prediction: {prediction_path.name}")
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
        if prediction is None:
            raise ValueError(f"Unable to read prediction: {prediction_path}")
        ground_truth = np.asarray(index.read_image(sample.mask).convert("L"), dtype=np.uint8)
        if prediction.shape != ground_truth.shape:
            # Match CamoFormer eval.py: cv2.resize without an interpolation
            # argument uses INTER_LINEAR. Keep the resized image in RAM.
            prediction = cv2.resize(prediction, ground_truth.shape[::-1])

        fm.step(pred=prediction, gt=ground_truth)
        wfm.step(pred=prediction, gt=ground_truth)
        sm.step(pred=prediction, gt=ground_truth)
        em.step(pred=prediction, gt=ground_truth)
        mae.step(pred=prediction, gt=ground_truth)

        if boundary:
            gt_color = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)
            pred_color = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
            boundary_gt, boundary_pred, area = borders_capture(gt_color, pred_color, boundary_rate)
            boundary_mae.step(pred=boundary_pred, gt=boundary_gt, area=area)
            boundary_f.step(pred=boundary_pred, gt=boundary_gt)

    fm_result, fnr = fm.get_results()
    wfm_result = wfm.get_results()["wfm"]
    sm_result = sm.get_results()["sm"]
    em_result = em.get_results()["em"]
    mae_result = mae.get_results()["mae"]
    return {
        "model": model,
        "dataset": dataset,
        "split": "test",
        "num_samples": len(predictions),
        "S_alpha": float(sm_result),
        "F_beta_w": float(wfm_result),
        "E_phi": float(em_result["curve"].mean()),
        "E_phi_adaptive": float(em_result["adp"]),
        "MAE": float(mae_result),
        "Boundary_F": None if boundary_f is None else float(boundary_f.get_results()["wfm"]),
        "Boundary_MAE": None if boundary_mae is None else float(boundary_mae.get_results()["mae"]),
        "FNR": float(fnr),
        "F_beta_adaptive": float(fm_result["adp"]),
    }


def _append_records(path: Path, results: Iterable[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_fields: list[str] | None = None
    if path.is_file() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle), None)
    fields = existing_fields or CSV_FIELDS
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not existing_fields:
            writer.writeheader()
        for result in results:
            row = {field: "" for field in fields}
            row.update(
                {
                    "experiment_id": args.experiment_id,
                    "dataset": result["dataset"],
                    "split": result["split"],
                    "backbone": args.backbone,
                    "input_size": args.input_size,
                    "seed": args.seed,
                    "teacher_type": args.teacher_type,
                    "student_type": args.student_type,
                    "distillation_type": args.distillation_type,
                    "lambda_logit": args.lambda_logit,
                    "lambda_relation": args.lambda_relation,
                    "lambda_boundary": args.lambda_boundary,
                    "lambda_uncertainty": args.lambda_uncertainty,
                    "lambda_prior": args.lambda_prior,
                    "best_epoch": args.best_epoch,
                    "S_alpha": f"{result['S_alpha']:.6f}",
                    "F_beta_w": f"{result['F_beta_w']:.6f}",
                    "E_phi": f"{result['E_phi']:.6f}",
                    "MAE": f"{result['MAE']:.6f}",
                    "Boundary_F": "" if result["Boundary_F"] is None else f"{result['Boundary_F']:.6f}",
                    "params": args.params,
                    "FLOPs": args.flops,
                    "FPS": args.fps,
                    "checkpoint": args.checkpoint,
                    "notes": args.notes,
                }
            )
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets/COD_benchmark")
    parser.add_argument("--pred-root", default="Prediction/CamoFormer")
    parser.add_argument("--dataset", dest="datasets", action="append", choices=DATASETS)
    parser.add_argument("--model", default="CamoFormer")
    parser.add_argument("--BR", choices=("on", "off"), default="on")
    parser.add_argument("--br-rate", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all prediction images")
    parser.add_argument("--experiment-id", default="E05-CamoFormer")
    parser.add_argument("--record-csv", default="results/experiment_records.csv")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--backbone", default="")
    parser.add_argument("--input-size", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--teacher-type", default="")
    parser.add_argument("--student-type", default="RGB")
    parser.add_argument("--distillation-type", default="None")
    parser.add_argument("--lambda-logit", default="")
    parser.add_argument("--lambda-relation", default="")
    parser.add_argument("--lambda-boundary", default="")
    parser.add_argument("--lambda-uncertainty", default="")
    parser.add_argument("--lambda-prior", default="")
    parser.add_argument("--best-epoch", default="")
    parser.add_argument("--params", default="")
    parser.add_argument("--flops", default="")
    parser.add_argument("--fps", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--notes", default="CamoFormer metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    prediction_root = Path(args.pred_root)
    datasets = args.datasets or DATASETS
    limit = None if args.max_samples == 0 else args.max_samples
    results = []

    for dataset in datasets:
        with _index_for_dataset(data_root, dataset) as index:
            results.append(
                evaluate_dataset(
                    index=index,
                    prediction_root=prediction_root,
                    dataset=dataset,
                    model=args.model,
                    boundary=args.BR == "on",
                    boundary_rate=args.br_rate,
                    max_samples=limit,
                )
            )

    payload = {"source": "CamoFormer evaltools/metrics.py", "results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.no_record:
        _append_records(Path(args.record_csv), results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
