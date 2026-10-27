#!/usr/bin/env python3
"""Build a source-only v4 bundle for a new server directory, with SHA256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_triprd_task_queue import SOURCE_FILES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "deliverables/triprd_task_v4_2026-10-22.zip")
    args = parser.parse_args(argv)
    files = sorted(set(SOURCE_FILES) | {
        "models/__init__.py", "models/triprd_full.py", "datasets/__init__.py", "evaluation/__init__.py",
        "requirements-task-v4.txt", "requirements-correction.txt", "requirements-triprd.txt",
        "README_TASK_V4.md", "tools/TRIPRD_TASK_V4.md", "tools/package_triprd_task.py",
        "tests/check_task_dinov3.py", "tests/test_runner_integration.py",
        "tests/test_task_model.py", "tests/test_task_loss.py", "tests/test_task_runner.py",
        "tests/test_task_protocol.py", "tests/test_task_queue.py",
        "evaluation/README.md", "datasets/README.md",
    })
    payloads = {name: (ROOT / name).read_bytes() for name in files}
    manifest = {"method": "triprd_task_v4", "files": {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in payloads.items()}, "excluded": ["datasets/images", "weights", "dinov3 checkout", "results"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            info = zipfile.ZipInfo(name, (2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, data)
        info = zipfile.ZipInfo("MANIFEST_TASK_V4.json", (2026, 10, 22, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Invalid output archive")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Archive hash mismatch: {name}")
    print(json.dumps({"archive": str(args.out.resolve()), "files": len(files) + 1,
                      "bytes": args.out.stat().st_size,
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
