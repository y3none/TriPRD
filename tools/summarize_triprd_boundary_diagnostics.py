#!/usr/bin/env python3
"""Summarize existing full internal correction-path diagnostics using stdlib only.

Reads manifest.json, GROUP/diagnostics.json and GROUP/cod_per_image.jsonl.
No model, checkpoint, dataset or training code is loaded. By default writes only
to stdout; --output-dir must be a new directory disjoint from --input-root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

METHOD = "triprd_correction_path_diagnostic_v1"
SUMMARY_METHOD = "triprd_boundary_summary_v1"
SCALES = ("stride8", "stride16", "pooled")
STRATA = ("all", "fg", "bg", "crossing")
DOMAINS = ("VDT", "COD10K", "CAMO")
MODES = ("none", "no_correction", "shuffle_correction", "identity")
RELATION_FIELDS = ("base_bce", "corrected_bce", "raw_abs", "clipped_abs", "clip_fraction")
RECOVERY_FIELDS = ("mse", "zero_mse", "constant_mse", "positive_coverage",
    "positive_mse", "positive_zero_mse", "positive_constant_mse", "teacher_rgb_bce",
    "teacher_full_bce", "teacher_rgb_plus_student_bce", "student_base_plus_teacher_delta_bce")
NOTES = [
    "Internal validation only; existing fixed best checkpoints; no retraining or official-test access.",
    "GT-assisted counterfactuals diagnose relations, not segmentation oracles; no model was rerun.",
    "BCE gain = baseline BCE - alternative BCE (positive is better).",
    "clipped_to_raw_ratio = mean(abs(clipped)) / mean(abs(raw)); null if raw mean is zero.",
    "Saved counterfactual BCE includes non-positive-reliability edges; positive MSE uses reliability > 0.",
    "Relation/BCE use all valid edges (counterfactuals also intersect teacher validity); coverage = positive_count / all_count.",
    "An all-edge counterfactual failure cannot reject positive-reliability corrections or balanced positive MSE; positive-only BCE is needed.",
    "Pooled statistics weight edges across scales, not scale means. Empty support is null, never zero error.",
    "Evidence consistency is checked against the saved manifest; upstream checkpoints are not re-hashed.",
    "These descriptive results establish neither causality nor significance; there is no automatic advancement.",
]
CANDIDATES = [
    "Inspect crossing GT BCE, local clipping and positive coverage together; validate TRAIN stratum support separately.",
    "If teacher delta improves student-base crossing relations, consider a paired stratum-balanced positive-MSE experiment.",
    "If all-edge teacher delta fails on the student base, inspect compatibility and add positive-only BCE before judging the supervised subset.",
    "If relations improve but segmentation barely changes, inspect propagation/decoder sensitivity using existing traces.",
    "Compare source boundary recovery and final relation quality before planning COD experiments; no numeric release threshold is applied.",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _load(path):
    try:
        data = path.read_bytes()
        return json.loads(data), hashlib.sha256(data).hexdigest()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read valid JSON: {path}: {exc}") from exc


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _record(stats, key):
    record = stats.get(key)
    _require(isinstance(record, dict) and all(k in record for k in ("count", "sum", "mean", "max")),
             f"Missing statistic or fields: {key}")
    n, total, mean, maximum = (record[k] for k in ("count", "sum", "mean", "max"))
    _require(_count(n) and _number(total), f"Invalid statistic: {key}")
    if n == 0:
        _require(total == 0 and mean is None and maximum is None, f"Invalid empty support: {key}")
    else:
        _require(_number(mean) and _number(maximum) and mean >= 0 and maximum >= mean - 1e-12,
                 f"Invalid finite/nonnegative statistic: {key}")
        _require(math.isclose(mean, total / n, rel_tol=1e-9, abs_tol=1e-12),
                 f"Mean/sum/count mismatch: {key}")
    return record


def _validate_statistics(stats):
    _require(isinstance(stats, dict), "Missing path_statistics")
    for domain in DOMAINS:
        for section, fields in (("relation", RELATION_FIELDS), ("recovery", RECOVERY_FIELDS)):
            if section == "recovery" and domain != "VDT":
                continue
            for stratum in STRATA:
                for scale in SCALES:
                    prefix = f"{domain}/{section}/{scale}/{stratum}"
                    records = {field: _record(stats, f"{prefix}/{field}") for field in fields}
                    normal = [record["count"] for field, record in records.items() if not field.startswith("positive_")]
                    if section == "relation":
                        normal = [record["count"] for record in records.values()]
                        fraction = records["clip_fraction"]["mean"]
                        _require(fraction is None or 0 <= fraction <= 1, f"Invalid clip fraction: {prefix}")
                    _require(len(set(normal)) == 1, f"Different all-edge supports: {prefix}")
                    if section == "recovery":
                        positive = [records[field]["count"] for field in
                                    ("positive_mse", "positive_zero_mse", "positive_constant_mse")]
                        coverage = records["positive_coverage"]
                        _require(len(set(positive)) == 1 and positive[0] <= normal[0]
                                 and coverage["count"] == normal[0], f"Different positive supports: {prefix}")
                        _require(math.isclose(coverage["sum"], positive[0], abs_tol=1e-8),
                                 f"Positive coverage/count mismatch: {prefix}")
                for field in fields:
                    rows = [stats[f"{domain}/{section}/{scale}/{stratum}/{field}"] for scale in SCALES]
                    _require(rows[2]["count"] == rows[0]["count"] + rows[1]["count"]
                             and math.isclose(rows[2]["sum"], rows[0]["sum"] + rows[1]["sum"],
                                              rel_tol=1e-9, abs_tol=1e-9),
                             f"Pooled support/sum mismatch: {domain}/{section}/{stratum}/{field}")
            for scale in SCALES:
                for field in fields:
                    rows = [stats[f"{domain}/{section}/{scale}/{stratum}/{field}"] for stratum in STRATA]
                    _require(rows[0]["count"] == sum(r["count"] for r in rows[1:])
                             and math.isclose(rows[0]["sum"], sum(r["sum"] for r in rows[1:]),
                                              rel_tol=1e-9, abs_tol=1e-9),
                             f"Stratum support/sum mismatch: {domain}/{section}/{scale}/{field}")


def _difference(base, alternative):
    return None if base is None or alternative is None else base - alternative


def _summary_group(report):
    stats = report["path_statistics"]
    relation, recovery, counterfactual = {}, {}, {}
    for domain in DOMAINS:
        relation[domain] = {}
        for scale in SCALES:
            prefix = f"{domain}/relation/{scale}/crossing"
            row = {field: stats[f"{prefix}/{field}"]["mean"] for field in RELATION_FIELDS}
            row["all_count"] = stats[f"{prefix}/base_bce"]["count"]
            row["bce_gain"] = _difference(row["base_bce"], row["corrected_bce"])
            row["clipped_to_raw_ratio"] = (row["clipped_abs"] / row["raw_abs"]
                                           if row["raw_abs"] else None)
            relation[domain][scale] = row
    for scale in SCALES:
        recovery[scale], counterfactual[scale] = {}, {}
        for stratum in STRATA:
            prefix = f"VDT/recovery/{scale}/{stratum}"
            mean = lambda field: stats[f"{prefix}/{field}"]["mean"]
            n = stats[f"{prefix}/mse"]["count"]
            recovery[scale][stratum] = {"all_count": n,
                "positive_count": stats[f"{prefix}/positive_mse"]["count"],
                **{field: mean(field) for field in ("positive_coverage", "mse", "zero_mse", "constant_mse",
                                                    "positive_mse", "positive_zero_mse", "positive_constant_mse")}}
            relation_prefix = f"VDT/relation/{scale}/{stratum}"
            matching = stats[f"{relation_prefix}/base_bce"]["count"] == n
            base = stats[f"{relation_prefix}/base_bce"]["mean"] if matching else None
            corrected = stats[f"{relation_prefix}/corrected_bce"]["mean"] if matching else None
            rgb = mean("teacher_rgb_bce")
            counterfactual[scale][stratum] = {"all_count": n, "student_support_matches": matching,
                "student_base_bce": base, "student_corrected_bce": corrected,
                "student_correction_gain": _difference(base, corrected),
                **{field: mean(field) for field in ("teacher_rgb_bce", "teacher_full_bce",
                    "teacher_rgb_plus_student_bce", "student_base_plus_teacher_delta_bce")},
                "student_base_plus_teacher_delta_gain": _difference(base, mean("student_base_plus_teacher_delta_bce")),
                "teacher_rgb_plus_student_gain": _difference(rgb, mean("teacher_rgb_plus_student_bce")),
                "teacher_full_gain": _difference(rgb, mean("teacher_full_bce"))}
    return {"selected_epoch": report["selected_epoch"], "coverage": report["coverage"],
            "reference": report["reference"], "reproduction": report.get("reproduction"),
            "crossing_relations": relation, "source_recovery": recovery, "counterfactuals": counterfactual}


def summarize(input_root, groups=("C", "D")):
    root = Path(input_root).expanduser().resolve()
    groups = tuple(groups)
    _require(groups and len(set(groups)) == len(groups) and not set(groups) - set("ABCD"),
             "groups must be distinct A/B/C/D letters")
    manifest, manifest_sha = _load(root / "manifest.json")
    _require(manifest.get("status") == "complete", "Diagnostic manifest is not complete")
    contract = manifest.get("contract", {})
    _require(contract.get("method") == METHOD, "Unexpected diagnostic method")
    _require(contract.get("scope") == "internal_validation_only"
             and contract.get("limit_cod_per_source") == 0 and contract.get("limit_vdt") == 0
             and all(contract.get(k) is False for k in ("reselect_checkpoint", "fit_constants", "optimize")),
             "Contract must describe complete fixed-checkpoint internal validation only")
    _require(contract.get("modes") == list(MODES), "Unexpected intervention modes")
    _require(isinstance(contract.get("groups"), list) and set(groups) <= set(contract["groups"]),
             "Requested group not present in manifest")
    recipe = contract.get("training_recipe", {})
    split = recipe.get("split_manifest", {})
    _require(recipe.get("official_test_access") is False and split.get("cod_train_mode") == "split",
             "Missing internal-split/no-official-test evidence")
    for key in ("cod_val", "vdt_val"):
        keys = split.get(key)
        _require(isinstance(keys, list) and keys and all(isinstance(k, str) for k in keys)
                 and len(set(keys)) == len(keys), f"Missing or invalid split_manifest/{key}")
    _require({key.split("/", 1)[0] for key in split["cod_val"]} == {"COD10K", "CAMO"},
             "COD split must contain exactly COD10K and CAMO")
    contract_sha = digest(contract)
    result = {"method": SUMMARY_METHOD, "status": "complete", "input_root": str(root),
        "manifest_sha256": manifest_sha, "contract_sha256": contract_sha,
        "notes": NOTES, "planned_candidates": CANDIDATES, "groups": {}}
    for group in groups:
        directory = (root / group).resolve()
        _require(root in directory.parents, f"Group directory escapes input root: {group}")
        report_path = directory / "diagnostics.json"
        report, report_sha = _load(report_path)
        _require(report.get("method") == METHOD and report.get("status") == "complete"
                 and report.get("group") == group and report.get("student_unchanged") is True,
                 f"Incomplete or incompatible diagnostic group: {group}")
        reference = contract.get("references", {}).get(group)
        _require(isinstance(reference, dict) and report.get("reference") == reference
                 and all(_sha(reference.get(k)) for k in ("checkpoint_sha256", "report_sha256", "initial_state_sha256"))
                 and isinstance(reference.get("directory"), str)
                 and _count(reference.get("selected_epoch"))
                 and report.get("selected_epoch") == reference["selected_epoch"],
                 f"Reference mismatch or missing reference hashes: {group}")
        _require(report.get("contract_sha256") == contract_sha, f"Contract hash mismatch: {group}")
        coverage = report.get("coverage", {})
        _require(coverage.get("partial_requested") is False
                 and coverage.get("full_cod_validation") is True and coverage.get("full_vdt_validation") is True
                 and coverage.get("cod") == len(split["cod_val"]) and coverage.get("vdt") == len(split["vdt_val"]),
                 f"Incomplete internal validation coverage: {group}")
        image_path = directory / "cod_per_image.jsonl"
        try:
            raw = image_path.read_bytes()
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid per-image evidence: {group}: {exc}") from exc
        _require(hashlib.sha256(raw).hexdigest() == report.get("per_image_sha256"),
                 f"Per-image hash mismatch: {group}")
        _require(len(rows) == len(split["cod_val"]) and all(isinstance(r, dict) for r in rows),
                 f"Per-image coverage mismatch: {group}")
        keys = [row.get("key") for row in rows]
        _require(all(isinstance(k, str) for k in keys) and len(set(keys)) == len(keys)
                 and set(keys) == set(split["cod_val"]), f"Per-image split keys mismatch: {group}")
        for row in rows:
            _require(row.get("source") == row["key"].split("/", 1)[0]
                     and all(_number(row.get("S_alpha", {}).get(mode)) for mode in MODES),
                     f"Per-image source/metrics mismatch: {group}/{row['key']}")
        _validate_statistics(report.get("path_statistics"))
        result["groups"][group] = {**_summary_group(report), "diagnostics_sha256": report_sha,
                                  "per_image_sha256": report["per_image_sha256"]}
    return result


def format_summary(summary):
    def fmt(value):
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        return str(value) if isinstance(value, int) else f"{value:.8g}"

    def fields(row, names):
        return " ".join(f"{name}={fmt(row[name])}" for name in names)

    lines = ["Boundary relation diagnosis from saved reports", *summary["notes"]]
    for group, report in summary["groups"].items():
        lines += ["", f"[{group}] selected_epoch={report['selected_epoch']} coverage={report['coverage']}",
                  "Reproduction " + json.dumps(report["reproduction"], ensure_ascii=False)]
        for domain, scales in report["crossing_relations"].items():
            for scale, row in scales.items():
                lines.append(f"{domain} {scale} crossing " + fields(row, ("all_count", "base_bce", "corrected_bce",
                    "bce_gain", "raw_abs", "clipped_abs", "clipped_to_raw_ratio", "clip_fraction")))
        for scale, strata in report["source_recovery"].items():
            for stratum, row in strata.items():
                lines.append(f"VDT recovery {scale} {stratum} " + fields(row, ("positive_count", "all_count",
                    "positive_coverage", "positive_mse", "positive_zero_mse", "positive_constant_mse",
                    "mse", "zero_mse", "constant_mse")))
        for scale, strata in report["counterfactuals"].items():
            for stratum, row in strata.items():
                lines.append(f"VDT counterfactual {scale} {stratum} " + fields(row, ("all_count",
                    "student_support_matches", "student_base_bce", "student_corrected_bce", "student_correction_gain",
                    "student_base_plus_teacher_delta_bce", "student_base_plus_teacher_delta_gain",
                    "teacher_rgb_bce", "teacher_full_bce", "teacher_full_gain",
                    "teacher_rgb_plus_student_bce", "teacher_rgb_plus_student_gain")))
    lines += ["", "Planned candidates (human interpretation required; no automatic training decision):"]
    lines += [f"- {candidate}" for candidate in summary["planned_candidates"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--groups", default="C,D")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.input_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else None
    if output is not None:
        _require(output != root and output not in root.parents and root not in output.parents,
                 "Output must be disjoint from input, including parents and children")
        _require(not output.exists(), f"Output directory already exists: {output}")
    summary = summarize(root, tuple(g.strip().upper() for g in args.groups.split(",")))
    text = format_summary(summary)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "boundary_summary.json").write_text(json.dumps(summary, ensure_ascii=False,
             allow_nan=False, indent=2) + "\n", encoding="utf-8")
        (output / "boundary_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
