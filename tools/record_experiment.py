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
        "## Results", "", "Quick Dice/MAE are prediction statistics; official metrics use evaluation/. E_phi is the threshold mean.", "",
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
