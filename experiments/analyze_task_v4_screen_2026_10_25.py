"""Read the uploaded reports without extracting or modifying training artifacts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/TriPRD_task_v4/screen_reports.tar.gz"
OUT = ROOT / "experiments/task_v4_screen_analysis_2026-10-25.json"
METRICS = ("S_alpha", "F_beta_w", "E_phi", "MAE")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected(report):
    return next(row for row in report["history"] if row["epoch"] == report["best_epoch"])


def delta(a, b):
    return {key: a[key] - b[key] for key in METRICS}


def main():
    names = ([f"screen/T{i}/comparison.json" for i in range(5)]
             + [f"teachers/{mode}/comparison.json" for mode in ("rgb", "multimodal")]
             + ["_queue/screen/plan.json", "_queue/screen_expanded/status.json"])
    with tarfile.open(ARCHIVE) as archive:
        assert Counter(m.name for m in archive.getmembers()) == Counter(names)
        assert all(m.isfile() and m.size < 5_000_000 for m in archive.getmembers())
        payloads = {name: archive.extractfile(name).read() for name in names}
    records = {name: json.loads(data) for name, data in payloads.items()}
    students = {f"T{i}": records[f"screen/T{i}/comparison.json"] for i in range(5)}
    teachers = {mode: records[f"teachers/{mode}/comparison.json"] for mode in ("rgb", "multimodal")}
    plan = records["_queue/screen/plan.json"]
    status = records["_queue/screen_expanded/status.json"]
    assert plan["plan_sha256"] == digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    expanded = {**plan, "phase": "screen_expanded", "gpus": ["2", "3", "0", "1", "6"],
                "original_plan_sha256": plan["plan_sha256"]}
    assert status["status"] == "complete"
    assert status["plan_sha256"] == digest({k: v for k, v in expanded.items() if k != "plan_sha256"})
    assert len({d["initial_state_sha256"] for d in students.values()}) == 1
    assert len({d["initial_state_sha256"] for d in teachers.values()}) == 1
    for name, expected in plan["source_sha256"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    with zipfile.ZipFile(ROOT / "deliverables/triprd_task_v4_2026-10-22.zip") as original:
        for report in [*students.values(), *teachers.values()]:
            assert report["status"] == "complete" and report["completed_epochs"] == 100
            assert report["test_status"] == "skipped" and report["test"] == {} and not report["formal"]
            assert report["contract_sha256"] == digest(report["protocol"])
            assert [row["epoch"] for row in report["history"]] == list(range(1, 101))
            for name, expected in report["protocol"]["source_sha256"].items():
                assert sha(original.read(name)) == expected == sha((ROOT / name).read_bytes())
            for prefix in ("cod", "vdt"):
                train, val = [report["protocol"].get(prefix + suffix, []) for suffix in ("_train", "_val")]
                assert len(train) == len(set(train)) and len(val) == len(set(val))
                assert not set(train).intersection(val)
            for row in report["history"]:
                tr = row["train"]
                assert tr["optimizer_steps"] + tr["amp_skipped_steps"] == tr["steps"]

    runs, curves = {}, {}
    for job in plan["jobs"]:
        name = job["name"]
        report = students[name]
        assert all(report["config"].get(k) == v for k, v in job["expected_config"].items())
        assert status["jobs"][name]["status"] == "complete" and status["jobs"][name]["returncode"] == 0
        assert status["jobs"][name]["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert status["jobs"][name]["selected_epoch"] == report["best_epoch"]
        assert report["selection_metric"] == "macro_S_alpha"
        for key in ("cod_train", "cod_val"):
            assert report["protocol"][key] == students["T0"]["protocol"][key]
        if name != "T0":
            for key in ("vdt_train", "vdt_val"):
                assert report["protocol"][key] == teachers["rgb"]["protocol"][key] == teachers["multimodal"]["protocol"][key]
        rows = {row["epoch"]: row for row in report["history"] if "validation" in row}
        assert list(rows) == list(range(5, 101, 5))
        assert report["best_epoch"] == max(rows, key=lambda e: rows[e]["validation"]["macro"]["S_alpha"])
        assert report["best_score"] == selected(report)["validation"]["macro"]["S_alpha"]
        for row in rows.values():
            val = row["validation"]
            assert val["num_samples"] == 808
            assert {k: v["num_samples"] for k, v in val["sources"].items()} == {"COD10K": 608, "CAMO": 200}
            for key in METRICS:
                assert abs(val["macro"][key] - statistics.mean(s[key] for s in val["sources"].values())) < 1e-12
        train = [row["train"] for row in report["history"]]
        aggregate = {key: sum(row[key] for row in train) for key in
                     ("steps", "optimizer_steps", "amp_skipped_steps", "primary_images", "source_images", "seconds")}
        aggregate["median_epoch_train_seconds"] = statistics.median(row["seconds"] for row in train)
        aggregate["peak_cuda_allocated_bytes"] = max(row["peak_cuda_allocated_bytes"] for row in train)
        if name in ("T2", "T4"):
            aggregate["mean_reliable_pixel_fraction"] = statistics.mean(row["reliable_coverage"] for row in train)
        runs[name] = {"best_epoch": report["best_epoch"], "selected_validation": selected(report)["validation"],
                      "last_validation": rows[100]["validation"], "training": aggregate,
                      "config": report["config"], "sample_counts": report["sample_counts"],
                      "initial_state_sha256": report["initial_state_sha256"], "parameters": report["parameters"],
                      "split_member_sha256": {key: digest(report["protocol"].get(key, [])) for key in
                                              ("cod_train", "cod_val", "vdt_train", "vdt_val")},
                      "checkpoint_reported_sha256": report["checkpoint_sha256"]}
        curves[name] = {str(e): row["validation"] for e, row in rows.items()}

    pairs = {}
    for a, b in (("T1", "T0"), ("T2", "T1"), ("T3", "T1"), ("T4", "T0"), ("T4", "T2"), ("T4", "T3")):
        diffs = {e: delta(curves[a][e]["macro"], curves[b][e]["macro"]) for e in curves[a]}
        pairs[f"{a}_minus_{b}"] = {
            "selected_epoch_delta": delta(runs[a]["selected_validation"]["macro"], runs[b]["selected_validation"]["macro"]),
            "same_epoch_deltas": diffs,
            "same_epoch_better_count_out_of_20": {key: sum((d[key] < 0 if key == "MAE" else d[key] > 0) for d in diffs.values()) for key in METRICS},
            "last_epoch_delta": diffs["100"],
            "mean_delta_epoch_50_through_100": {key: statistics.mean(d[key] for e, d in diffs.items() if int(e) >= 50) for key in METRICS},
        }
    teacher_summary = {}
    for mode, report in teachers.items():
        assert report["selection_metric"] == "source_validation_loss"
        assert report["best_epoch"] == min(report["history"], key=lambda r: r["validation"]["loss"])["epoch"]
        assert plan["dependencies"][mode]["sha256"] == report["checkpoint_sha256"]
        for group in (("T2",) if mode == "rgb" else ("T3", "T4")):
            assert students[group]["protocol"]["teacher_sha256"] == report["checkpoint_sha256"]
        teacher_summary[mode] = {"best_epoch": report["best_epoch"], "selected_validation": selected(report)["validation"],
                                 "last_validation": report["history"][-1]["validation"],
                                 "parameters": report["parameters"], "initial_state_sha256": report["initial_state_sha256"]}
    baseline = json.loads((ROOT / "results/TriPRD_task_v4/baseline/R3_blocks4_detail/comparison.json").read_text())
    result = {
        "date": "2026-10-25", "scope": "one paired training seed; internal COD development only",
        "archive": str(ARCHIVE.relative_to(ROOT)), "archive_sha256": sha(ARCHIVE.read_bytes()),
        "member_sha256": {name: sha(data) for name, data in payloads.items()},
        "audit": {"all_report_contracts_and_source_hashes_valid": True, "all_jobs_complete": True,
                  "all_tests_skipped": True, "expanded_plan_reconstruction_matches": True,
                  "original_plan_sha256": plan["plan_sha256"], "expanded_plan_sha256": status["plan_sha256"],
                  "training_source_files": len(students["T0"]["protocol"]["source_sha256"]),
                  "queue_source_files": len(plan["source_sha256"])},
        "metric_semantics": {"selection": "equal-source macro S_alpha", "E_phi": "mean E, not adaptive E",
                             "accompanying_metrics": "same selected checkpoint; never independent metric maxima"},
        "runs": runs, "validation_curves": curves, "pairwise": pairs, "teachers": teacher_summary,
        "baseline_R3_reproduction": {"same_protocol": baseline["protocol"] == students["T0"]["protocol"],
                                     "same_initialization": baseline["initial_state_sha256"] == students["T0"]["initial_state_sha256"],
                                     "screen_T0_minus_baseline_R3": delta(runs["T0"]["selected_validation"]["macro"], selected(baseline)["validation"]["macro"])},
        "decision": "Do not promote current T4 to final; preserve RGB baseline and redesign transfer after targeted mechanism checks.",
        "limitations": ["No actual checkpoints, split_manifest file, source images or predictions in the bundle: reported hashes are cross-checked, not recomputed from those missing artifacts.",
                        "Twenty validation epochs are repeated observations of one training trajectory, not twenty independent seeds.",
                        "Aggregate teacher loss averages R and RTD plus segmentation, auxiliary and boundary terms; it does not establish per-coalition COD transferability.",
                        "Training seconds exclude validation, teacher pretraining and interrupted partial epochs; physical GPU equivalence is not established.",
                        "The chosen epochs differ by group. Same-epoch comparisons are diagnostics and do not replace the prespecified selection rule."],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "audit": result["audit"],
                      "selected": {name: {"epoch": run["best_epoch"], **run["selected_validation"]["macro"]} for name, run in runs.items()}}, indent=2))


if __name__ == "__main__":
    main()
