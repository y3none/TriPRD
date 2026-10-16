#!/usr/bin/env python3
"""Create a human-readable experiment record from a completed run directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def experiment_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value.lower() in {"index", "readme"}:
        raise argparse.ArgumentTypeError("ID must be a filename-safe name, excluding index and README")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--id", required=True, type=experiment_id, help="Stable ID, for example E001 or E001_lr1e-4")
    parser.add_argument("--run-dir", required=True, help="Directory containing comparison.json")
    parser.add_argument("--change", required=True, help="What changed from the previous experiment")
    parser.add_argument("--hypothesis", default="", help="Expected effect of the change")
    parser.add_argument("--analysis", default="待填写", help="Initial analysis; edit the generated record later")
    parser.add_argument("--decision", default="待填写", help="Keep, reject, or investigate")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--log-dir", default="experiments", help="Experiment record directory")
    return parser.parse_args()


def fmt(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return "" if value is None else str(value)


def cell(value: object) -> str:
    return fmt(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def metric_row(group: str, dataset: str, result: object) -> str:
    if not isinstance(result, dict):
        result = {"status": "invalid", "error": "test result must be an object"}
    official = result.get("official")
    if not isinstance(official, dict):
        official = {}
    required = ("MAE", "S_alpha", "F_beta_w", "E_phi")
    invalid = [key for key in required if not isinstance(official.get(key), (int, float))
               or not math.isfinite(official[key])]
    status = official.get("status", result.get("status"))
    error = official.get("error") or result.get("error")
    if error or status not in (None, "ok", "success"):
        status = f"{status or 'unavailable'}: {error or 'official evaluation unavailable'}"
    elif invalid:
        status = "unavailable: missing/invalid official " + ", ".join(invalid)
    else:
        status = "ok"
    values = [group, dataset, result.get("dice"), result.get("mae"),
              *(official.get(key) for key in required), official.get("Boundary_F"), status]
    return "| " + " | ".join(cell(value) for value in values) + " |"


def validation_lines(summary: dict) -> list[str]:
    """Report the explicitly selected epoch, never infer it from legacy Dice."""
    metric_names = ("S_alpha", "F_beta_w", "E_phi", "MAE")
    lines = [
        "## Validation selection", "",
        "Official validation metrics below belong to the selected epoch; macro averages weight each source equally.", "",
        "| Group | Best epoch | Selection metric | Selection score | S_alpha | F_beta_w | E_phi mean | MAE | Status / error |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    groups = summary.get("groups")
    source_rows = []
    if not isinstance(groups, dict) or not groups:
        lines.append("|  |  |  |  |  |  |  |  | unavailable: group validation summaries not recorded |")
        return lines
    for group, group_summary in groups.items():
        fit = group_summary.get("cod_finetune") if isinstance(group_summary, dict) else None
        fit = fit if isinstance(fit, dict) else {}
        epoch, selection, score = fit.get("best_epoch"), fit.get("selection_metric"), fit.get("best_selection_score")
        valid_epoch = isinstance(epoch, int) and not isinstance(epoch, bool) and epoch > 0
        valid_score = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
        history = fit.get("history")
        selected = next((row for row in history if isinstance(row, dict) and row.get("epoch") == epoch), None) \
            if valid_epoch and isinstance(history, list) else None
        official = selected.get("val_official") if isinstance(selected, dict) else None
        official = official if isinstance(official, dict) else {}
        macro = official.get("macro")
        macro = macro if isinstance(macro, dict) else {}
        invalid = [name for name in metric_names if not isinstance(macro.get(name), (int, float))
                   or isinstance(macro.get(name), bool) or not math.isfinite(macro[name])]
        if not valid_epoch or not isinstance(selection, str) or not selection or not valid_score:
            status = "unavailable: selection metadata not recorded"
        elif selected is None:
            status = "unavailable: selected epoch history not recorded"
        elif not official:
            status = "unavailable: official validation not recorded at selected epoch"
        elif invalid:
            status = "unavailable: missing/invalid official validation " + ", ".join(invalid)
        else:
            status = "ok"
        values = [group, epoch if valid_epoch else None, selection, score if valid_score else None,
                  *(None if name in invalid else macro[name] for name in metric_names), status]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        sources = official.get("sources")
        if isinstance(sources, dict):
            for source, metrics in sources.items():
                if not isinstance(metrics, dict):
                    continue
                values = [group, source, epoch, metrics.get("num_samples"), *(metrics.get(name) for name in metric_names)]
                source_rows.append("| " + " | ".join(cell(value) for value in values) + " |")
    if source_rows:
        lines += ["", "### Selected-epoch validation by source", "",
                  "| Group | Source | Epoch | Samples | S_alpha | F_beta_w | E_phi mean | MAE |",
                  "|---|---|---:|---:|---:|---:|---:|---:|", *source_rows]
    return lines


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "comparison.json"
    if not summary_path.is_file():
        raise SystemExit(f"comparison.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary.get("protocol", {})
    output_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / f"{args.id}.md"
    if record_path.exists():
        raise SystemExit(f"Record already exists: {record_path}; choose another --id")

    recorded_at = datetime.now(timezone.utc)
    lines = [
        f"# {args.id}", "", f"- Recorded at: {recorded_at.isoformat()}",
        f"- Run directory: `{run_dir}`", f"- Tags: {args.tags or 'none'}", "",
        "## Change", "", args.change, "", "## Hypothesis", "", args.hypothesis or "待填写", "",
        "## Protocol", "", "```json", json.dumps(protocol, ensure_ascii=False, indent=2), "```", "",
        *validation_lines(summary), "",
        "## Results", "",
        "Test status: skipped — 未进行测试（validation-only run）." if summary.get("test_status") == "skipped"
        else f"Test status: {summary.get('test_status', 'not recorded')}.", "",
        "Quick Dice/MAE are prediction statistics; official metrics use evaluation/. E_phi is the threshold mean.", "",
        "| Group | Dataset | Quick Dice | Quick MAE | MAE (official) | S_alpha | F_beta_w | E_phi mean | Boundary_F | Status / error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group, datasets in summary.get("test", {}).items():
        for dataset, result in datasets.items():
            lines.append(metric_row(group, dataset, result))
    lines += [
        "", "## Analysis", "", args.analysis, "", "### Comparison with previous run", "", "待填写", "",
        "### Failure cases and observations", "", "待填写", "", "## Decision", "", args.decision, "",
        "## Artifacts", "", f"- Raw summary: `{summary_path}`", f"- Full log: `{run_dir / 'run.log'}`", "",
    ]
    record_path.write_text("\n".join(lines), encoding="utf-8")
    index_path = output_dir / "index.md"
    entry = f"| {args.id} | {recorded_at.date().isoformat()} | {cell(args.change)} | [{record_path.name}]({record_path.name}) | {cell(args.decision)} |\n"
    if not index_path.exists():
        index_path.write_text("# Experiment index\n\n| ID | Date | Change | Record | Decision |\n|---|---|---|---|---|\n", encoding="utf-8")
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    print(f"Created {record_path}")
    print(f"Updated {index_path}")


if __name__ == "__main__":
    main()
