#!/usr/bin/env python3
"""Read-only checkpoint diagnosis of the completed positive-MSE COD screen.

Default: verify evidence and print the plan. --execute evaluates existing best
checkpoints on internal validation only, in a new independent output directory.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.validation import CODValidationMetrics
from evaluation.cod_metrics import Smeasure
from models.triprd_correction import CorrectionStudent, CorrectionTeacher
from models.triprd_correction_loss import correction_targets, correction_probe, finalize_correction_probe
from tools import run_triprd_correction as runner
from tools import run_triprd_mse_screen as screen
from tools import triprd_mse_screen_training as training
from tools.triprd_structure_training import state_digest
from tools.triprd_training import sha256_file
from tools.correction_path_diagnostics import DiagnosticAccumulator, inspect_batch

METHOD = "triprd_correction_path_diagnostic_v1"
MODES = ("none", "no_correction", "shuffle_correction", "identity")
FILES = ("tools/diagnose_triprd_correction_path.py", "tools/correction_path_diagnostics.py")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "diagnostics-root", "screen-root", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--groups", default="A,B,C,D")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-cod-per-source", type=int, default=0, help="0=all; positive is a labelled partial smoke diagnosis")
    p.add_argument("--limit-vdt", type=int, default=0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse complete groups only under the identical diagnostic contract")
    return p.parse_args(argv)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _disjoint(paths):
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("Input and output directories must be disjoint, including their parents")


def prepare(args):
    for key in ("run_root", "diagnostics_root", "screen_root", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    _disjoint([args.run_root, args.diagnostics_root, args.screen_root, args.output_root])
    groups = tuple(value.strip().upper() for value in args.groups.split(","))
    if not groups or len(set(groups)) != len(groups) or set(groups) - set("ABCD"):
        raise ValueError("groups must be distinct A/B/C/D letters separated by commas")
    if min(args.num_workers, args.limit_cod_per_source, args.limit_vdt) < 0:
        raise ValueError("workers and sample limits must be nonnegative")
    args.groups = ",".join(groups)
    # This reads/validates the existing baseline, teacher and passed probe. It
    # neither starts jobs nor writes to these evidence directories.
    context = screen.build_context(argparse.Namespace(run_root=args.run_root,
        diagnostics_root=args.diagnostics_root, output_root=args.screen_root, resume=False))
    reports = dict(context["baselines"])
    for group in "CD":
        directory = args.screen_root / screen.GROUP_DIRS[group]
        reports[group] = training.verify_completed(directory, screen.student_contract(context, group))
    references = {}
    for group in "ABCD":
        directory = (args.run_root if group in "AB" else args.screen_root) / screen.GROUP_DIRS[group]
        report = reports[group]
        references[group] = {"directory": str(directory), "checkpoint_sha256": report["checkpoint_sha256"],
            "report_sha256": sha256_file(directory / "comparison.json"), "selected_epoch": report["best_epoch"],
            "initial_state_sha256": report["initial_state_sha256"]}
    context["reports"] = reports
    contract = {"method": METHOD, "groups": list(groups), "references": references,
        "training_recipe": context["recipe"], "source_sha256": {p: sha256_file(ROOT / p) for p in FILES},
        "device": args.device, "amp": context["a"].amp, "num_workers": args.num_workers,
        "limit_cod_per_source": args.limit_cod_per_source, "limit_vdt": args.limit_vdt,
        "scope": "internal_validation_only", "modes": list(MODES),
        "reselect_checkpoint": False, "fit_constants": False, "optimize": False}
    context["contract"] = contract
    return context


def _cod_items(items, limit):
    if not limit:
        return items
    counts, selected = {}, []
    for item in items:
        source = item.key.split("/", 1)[0]
        if counts.get(source, 0) < limit:
            selected.append(item)
            counts[source] = counts.get(source, 0) + 1
    return selected


def _sm(prediction, gt):
    metric = Smeasure()
    metric.step(pred=prediction, gt=gt)
    return float(metric.get_results()["sm"])


def paired_summary(rows):
    """Per-source paired image differences, then source-balanced macro means.

    These are descriptive differences of fixed selected checkpoints, not
    independent training replicates or a test of method-level significance.
    """
    result = {}
    for mode in MODES[1:]:
        sources = {}
        for source in sorted({row["source"] for row in rows}):
            values = np.array([row["S_alpha"]["none"] - row["S_alpha"][mode]
                               for row in rows if row["source"] == source])
            sources[source] = {"count": len(values), "mean_none_minus_intervention": float(values.mean()),
                "median": float(np.median(values)), "p05": float(np.quantile(values, .05)),
                "p95": float(np.quantile(values, .95)), "positive_count": int((values > 0).sum()),
                "negative_count": int((values < 0).sum()), "zero_count": int((values == 0).sum())}
        result[mode] = {"sources": sources, "macro_none_minus_intervention": float(np.mean([
            value["mean_none_minus_intervention"] for value in sources.values()]))}
    return result


@torch.no_grad()
def evaluate_cod(model, items, a, device, accumulator, per_image_path):
    metrics = {mode: CODValidationMetrics() for mode in MODES}
    rows = []
    runner.shared.NORMALIZE_RGB = a.pretrained
    with per_image_path.open("w", encoding="utf-8") as stream:
        for number, item in enumerate(items, 1):
            sample, index = item.source
            rgb, mask = index.read_pair(sample, strict_size=False)
            image = runner.shared._tensor(rgb, a.image_size, rgb=True).unsqueeze(0).to(device)
            label = runner.shared._tensor(mask, a.image_size, mask=True).unsqueeze(0).to(device)
            source = item.key.split("/", 1)[0]
            with runner._autocast(a, device):
                outputs = inspect_batch(model, image, label, "cod", accumulator, source)
            gt = np.asarray(mask.convert("L"), dtype=np.uint8)
            row = {"key": item.key, "source": source, "S_alpha": {}, "float_probability_abs_difference": {}}
            baseline = outputs["none"]["logits"].float().sigmoid()
            for mode, output in outputs.items():
                pred = np.asarray(runner.shared._prediction_image(output["logits"].float(), mask.size), dtype=np.uint8)
                metrics[mode].update(source, pred, gt)
                row["S_alpha"][mode] = _sm(pred, gt)
                difference = (output["logits"].float().sigmoid() - baseline).abs()
                row["float_probability_abs_difference"][mode] = {
                    "mean": float(difference.mean()), "max": float(difference.max())}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rows.append(row)
            if number % 50 == 0 or number == len(items):
                print(f"COD validation {number}/{len(items)}", flush=True)
    return {mode: metric.compute() for mode, metric in metrics.items()}, paired_summary(rows)


@torch.no_grad()
def evaluate_source(model, teacher, items, constants, a, device, accumulator):
    dataset = runner.CorrectionDataset(items, a.image_size, normalize=a.pretrained, augment=False)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(a.train_seed))
    totals = {}
    for number, batch in enumerate(loader, 1):
        batch = runner.shared._move(batch, device)
        with runner._autocast(a, device):
            coals = teacher.forward_coalitions(batch["rgb"], batch["thermal"], batch["depth"])
        targets = correction_targets(coals["R"], coals["RTD"], batch["mask"], a.relation_kernel)
        del coals
        with runner._autocast(a, device):
            outputs = inspect_batch(model, batch["rgb"], batch["mask"], "sod", accumulator, "VDT",
                targets=targets, constants=constants)
        runner._add_stats(totals, correction_probe(outputs["none"], targets, constants["all"],
            a.relation_kernel, positive_constant_correction=constants["positive"]))
        if number % 10 == 0 or number == len(loader):
            print(f"VDT validation batches {number}/{len(loader)}", flush=True)
    return {key: float(value) for key, value in finalize_correction_probe(totals).items()}


def run_group(args, context, group, reference, teacher):
    device = torch.device(args.device)
    a = copy.deepcopy(context["a"])
    a.device, a.num_workers = args.device, args.num_workers
    report = context["reports"][group]
    path = Path(reference["directory"]) / "best.pt"
    if sha256_file(path) != reference["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed since preflight")
    state = (runner._read_checkpoint if group in "AB" else training.read_checkpoint)(path, report["protocol"])
    if state["epoch"] != reference["selected_epoch"]:
        raise ValueError("Selected checkpoint epoch differs")
    model = CorrectionStudent(**runner._model_options(a, pretrained=False)).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).eval()
    del state
    before = state_digest(model)
    accumulator = DiagnosticAccumulator()
    directory = args.output_root / group
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "cod_per_image.jsonl"
    cod = _cod_items(context["splits"]["cod_val"], args.limit_cod_per_source)
    source = context["splits"]["vdt_val"]
    if args.limit_vdt:
        source = source[:args.limit_vdt]
    print(f"[{group}] best epoch={reference['selected_epoch']}; COD={len(cod)}, VDT={len(source)}", flush=True)
    cod_metrics, paired = evaluate_cod(model, cod, a, device, accumulator, image_path)
    constants = {name: tuple(value.to(device) for value in context["teacher_state"][field])
        for name, field in (("all", "constant_correction"), ("positive", "positive_constant_correction"))}
    source_probe = evaluate_source(model, teacher, source, constants, a, device, accumulator)
    if state_digest(model) != before:
        raise RuntimeError("Diagnostic changed student parameters or buffers")
    selected = screen.selected_validation(report)
    full_cod = args.limit_cod_per_source == 0 and len(cod) == len(context["splits"]["cod_val"])
    full_source = args.limit_vdt == 0 and len(source) == len(context["splits"]["vdt_val"])
    result = {"method": METHOD, "status": "complete", "group": group,
        "contract_sha256": runner._digest(context["contract"]), "reference": reference,
        "selected_epoch": reference["selected_epoch"], "student_unchanged": True,
        "coverage": {"cod": len(cod), "vdt": len(source),
                     "partial_requested": bool(args.limit_cod_per_source or args.limit_vdt), "full_cod_validation": full_cod,
                     "full_vdt_validation": full_source},
        "metrics": cod_metrics, "paired_S_alpha": paired, "path_statistics": accumulator.result(),
        "source_recoverability": source_probe, "original_selected_validation": selected,
        "reproduction": {"macro_S_alpha_delta": cod_metrics["none"]["macro"]["S_alpha"] - selected["macro"]["S_alpha"] if full_cod else None,
            "positive_mse_delta": source_probe["positive_mse"] - report["source_recoverability"]["positive_mse"]
                if full_source and report.get("source_recoverability") else None},
        "per_image_sha256": sha256_file(image_path), "torch_version": str(torch.__version__),
        "device": str(device), "amp": a.amp and device.type == "cuda",
        "notes": ["E_phi is mean E; COD macro weights each source equally.",
            "Relation diagnostics use the resized label grid; COD metrics use original GT dimensions.",
            "Feature differences are paired traces; upstream changes also affect later layers.",
            "Empty support is null. Image differences do not establish across-seed significance."]}
    runner._save_json(directory / "diagnostics.json", result)
    return result


def _completed(args, context, group):
    path = args.output_root / group / "diagnostics.json"
    if not path.exists():
        return None
    report = _read(path)
    if (report.get("status") != "complete" or report.get("method") != METHOD
            or report.get("group") != group or report.get("contract_sha256") != runner._digest(context["contract"])
            or report.get("reference") != context["contract"]["references"][group]
            or report.get("per_image_sha256") != sha256_file(path.parent / "cod_per_image.jsonl")):
        raise ValueError("Existing group diagnostics differ from this run contract or image output")
    return report


def brief_report(reports):
    """A compact, copyable report; retain full precision/statistics in JSON."""
    def number(value):
        return "null" if value is None else f"{value:.8g}"

    lines = ["Internal validation; fixed best checkpoints; Sm differences = none - intervention.",
             "No method-significance threshold is inferred from this report."]
    for group in sorted(reports):
        report = reports[group]
        stats = report["path_statistics"]
        get = lambda key: stats.get(key, {}).get("mean")
        lines += ["", f"[{group}] selected_epoch={report['selected_epoch']} coverage={report['coverage']}",
                  "Sm " + " ".join(f"{mode}={number(report['metrics'][mode]['macro']['S_alpha'])}" for mode in MODES),
                  "Sm_gain " + " ".join(f"{mode}={number(report['paired_S_alpha'][mode]['macro_none_minus_intervention'])}" for mode in MODES[1:]),
                  "Reproduction " + " ".join(f"{key}={number(value)}" for key, value in report["reproduction"].items()),
                  "VDT_probe " + " ".join(f"{key}={number(report['source_recoverability'][key])}" for key in
                      ("positive_mse", "positive_zero_mse", "positive_constant_mse"))]
        for domain in ("COD10K", "CAMO", "VDT"):
            for scale in ("stride8", "stride16"):
                prefix = f"{domain}/relation/{scale}/all"
                if prefix + "/raw_abs" not in stats:
                    continue
                fields = {name: get(prefix + "/" + name) for name in
                          ("raw_abs", "clipped_abs", "clip_fraction", "base_bce", "corrected_bce")}
                lines.append(f"{domain} {scale} " + " ".join(f"{key}={number(value)}" for key, value in fields.items()))
            for mode in ("no_correction", "identity"):
                fields = {name: get(f"{domain}/intervention/{mode}/{name}") for name in
                          ("prob_abs_diff", "decoder_abs_diff")}
                lines.append(f"{domain} {mode} " + " ".join(f"{key}={number(value)}" for key, value in fields.items()))
        for stratum in ("fg", "bg", "crossing"):
            prefix = f"VDT/recovery/pooled/{stratum}"
            fields = {name: get(prefix + "/" + name) for name in
                      ("positive_mse", "positive_zero_mse", "positive_constant_mse")}
            count = stats.get(prefix + "/positive_mse", {}).get("count", 0)
            lines.append(f"VDT {stratum} positive_edges={count} " + " ".join(f"{key}={number(value)}" for key, value in fields.items()))
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    context = prepare(args)
    contract = context["contract"]
    print(json.dumps({"method": METHOD, "groups": contract["groups"], "references": contract["references"],
        "modes": MODES, "output": str(args.output_root), "execute": args.execute}, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose a supported device")
    manifest_path = args.output_root / "manifest.json"
    if args.output_root.exists():
        if not args.resume or not manifest_path.exists() or _read(manifest_path).get("contract") != contract:
            raise ValueError("Output exists: use a new directory or --resume with its identical contract")
    else:
        args.output_root.mkdir(parents=True)
    lock_path = args.output_root / "diagnostic.lock"
    with lock_path.open("x", encoding="utf-8") as lock:
        import os
        lock.write(str(os.getpid()))
    try:
        runner._save_json(manifest_path, {"status": "running", "contract": contract})
        reports = {}
        pending = []
        for group in contract["groups"]:
            existing = _completed(args, context, group) if args.resume else None
            if existing is None:
                pending.append(group)
            else:
                reports[group] = existing
                print(f"[{group}] reused complete diagnostic", flush=True)
        if pending:
            teacher = CorrectionTeacher(**runner._model_options(context["a"], pretrained=False)).to(device)
            teacher.load_state_dict(context["teacher_state"]["model"], strict=True)
            teacher.requires_grad_(False).eval()
            teacher_before = state_digest(teacher)
            for group in pending:
                reports[group] = run_group(args, context, group, contract["references"][group], teacher)
            if state_digest(teacher) != teacher_before:
                raise RuntimeError("Diagnostic changed teacher parameters or buffers")
        summary = {"method": METHOD, "status": "complete", "contract_sha256": runner._digest(contract),
            "scope": "internal_validation_only", "groups": {group: {
                "selected_epoch": reports[group]["selected_epoch"], "coverage": reports[group]["coverage"],
                "metrics": reports[group]["metrics"], "reproduction": reports[group]["reproduction"],
                "paired_S_alpha": reports[group]["paired_S_alpha"],
                "source_recoverability": reports[group]["source_recoverability"],
                "diagnostics_file": f"{group}/diagnostics.json"} for group in contract["groups"]}}
        runner._save_json(args.output_root / "summary.json", summary)
        (args.output_root / "diagnosis.txt").write_text(brief_report(reports), encoding="utf-8")
        runner._save_json(manifest_path, {"status": "complete", "contract": contract})
        print("group epoch normal-Sm no-correction-Sm shuffle-Sm identity-Sm", flush=True)
        for group in contract["groups"]:
            report = reports[group]
            print(group, report["selected_epoch"], *[f"{report['metrics'][mode]['macro']['S_alpha']:.8f}" for mode in MODES], flush=True)
    except BaseException as error:
        runner._save_json(manifest_path, {"status": "failed", "contract": contract, "error": str(error)})
        raise
    finally:
        lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
