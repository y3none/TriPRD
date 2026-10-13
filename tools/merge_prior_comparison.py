#!/usr/bin/env python3
"""Merge TriPRD metrics with the prior-performance LaTeX table.

The script does not alter the prior values.  It extracts the four published
columns per dataset from ``previous_performance.tex`` and appends the selected
TriPRD group from ``comparison.json`` using the same metric names.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DATASETS = ("NC4K", "COD10K", "CAMO", "CHAMELEON")
TABLE_NAMES = {"NC4K": "NC4K", "COD10K": "COD10K-Test", "CAMO": "CAMO-Test", "CHAMELEON": "CHAMELEON"}


def _clean_method(value: str) -> str:
    value = re.sub(r"\\citep\{[^}]*\}", "", value)
    value = re.sub(r"\\text(?:bf|color)\{(?:red\})?", "", value)
    value = value.replace("\\textit{", "").replace("}", "").replace("{", "")
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = value.replace("$", "").replace("~", " ").replace("\\", "").strip()
    return re.sub(r"\s+", " ", value)


def _clean_value(value: str) -> str:
    match = re.search(r"(?:\d+\.\d+|\.\d+)", value)
    return match.group(0) if match else value.strip()


def _read_prior(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    text = path.read_text(encoding="utf-8")
    # Ordinary prior rows keep all 16 values on one line: four metrics for
    # each of NC4K, COD10K, CAMO and CHAMELEON.
    for line in text.splitlines():
        if "multicolumn" in line or "midrule" in line or "toprule" in line or "bottomrule" in line:
            continue
        parts = [part.strip() for part in line.split("&")]
        if len(parts) < 17 or "\\\\" not in parts[-1]:
            continue
        method = _clean_method(parts[0])
        values = [_clean_value(part) for part in parts[1:17]]
        if not method or not all(re.fullmatch(r"(?:\d+\.\d+|\.\d+)", item) for item in values):
            continue
        for offset, dataset in enumerate(DATASETS):
            metric = values[offset * 4 : offset * 4 + 4]
            rows.append({"method": method, "dataset": TABLE_NAMES[dataset], "S_m": metric[0], "alpha_E": metric[1], "F_m_w": metric[2], "M": metric[3], "source": "previous_performance.tex"})
    # E2Net is formatted across four lines in the source table.  Its 16
    # numeric cells are still unambiguous, so parse that block separately.
    marker = text.find("E2Net")
    end = text.find("\\bottomrule", marker)
    if marker >= 0 and end > marker:
        block = text[marker:end]
        numeric = re.findall(r"(?:\d+\.\d+|\.\d+)", block)
        if len(numeric) >= 16:
            values = numeric[-16:]
            for offset, dataset in enumerate(DATASETS):
                metric = values[offset * 4 : offset * 4 + 4]
                rows.append({"method": "E2Net", "dataset": TABLE_NAMES[dataset], "S_m": metric[0], "alpha_E": metric[1], "F_m_w": metric[2], "M": metric[3], "source": "previous_performance.tex"})
    return rows


def _read_triprd(path: Path, group: str) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: list[dict[str, str]] = []
    for dataset in DATASETS:
        item = payload.get("test", {}).get(group, {}).get(dataset, {})
        official = item.get("official", item)
        output.append({
            "method": f"TriPRD-{group}", "dataset": TABLE_NAMES[dataset],
            "S_m": str(official.get("S_m", official.get("S_alpha", "NA"))),
            "alpha_E": str(official.get("alpha_E", official.get("E_phi", "NA"))),
            "F_m_w": str(official.get("F_m_w", official.get("F_beta_w", "NA"))),
            "M": str(official.get("M", official.get("MAE", "NA"))),
            "source": str(path),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", default="previous_performance.tex")
    parser.add_argument("--triprd", required=True, help="comparison.json from run_triprd_ab.py")
    parser.add_argument("--group", default="D")
    parser.add_argument("--output", default="results/prior_comparison.csv")
    args = parser.parse_args()
    rows = _read_prior(Path(args.prior)) + _read_triprd(Path(args.triprd), args.group.upper())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "dataset", "S_m", "alpha_E", "F_m_w", "M", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "rows": len(rows), "datasets": list(DATASETS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
