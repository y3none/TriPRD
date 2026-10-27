#!/usr/bin/env python3
"""Package staged v5 plus the byte-identical v4 dependencies, without weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ADDITIONS = ("README_STAGED_V5.md", "tools/run_triprd_staged.py", "tools/run_triprd_staged_queue.py",
             "tools/run_triprd_staged.sh", "tools/run_triprd_staged_server.sh", "tools/evaluate_triprd_staged.py", "tools/summarize_triprd_staged.py",
             "tools/package_triprd_staged.py", "tests/test_staged_runner.py", "tests/test_staged_queue.py",
             "tests/test_staged_protocol.py", "tests/test_staged_summary.py")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "deliverables/triprd_staged_v5_2026-10-25.zip")
    a = p.parse_args(argv)
    with zipfile.ZipFile(ROOT / "deliverables/triprd_task_v4_2026-10-22.zip") as old:
        manifest = json.loads(old.read("MANIFEST_TASK_V4.json"))
        payloads = {name: old.read(name) for name in old.namelist()}
        for name, expected in manifest["files"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"Original v4 file changed: {name}; restore it before packaging")
    payloads.update({name: (ROOT / name).read_bytes() for name in ADDITIONS})
    staged = {"method": "triprd_staged_v5", "original_v4_files_unchanged": len(manifest["files"]),
              "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                        for name, data in payloads.items()},
              "excluded": ["datasets", "weights", "teacher checkpoints", "results", "dinov3 checkout"],
              "evidence": "Functionality checks do not establish COD accuracy improvement; run paired screen first."}
    payloads["MANIFEST_STAGED_V5.json"] = (json.dumps(staged, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w") as archive:
        for name, data in payloads.items():
            info = zipfile.ZipInfo(name, (2026, 10, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(a.out) as archive:
        assert archive.testzip() is None
        assert all(archive.read(name) == data for name, data in payloads.items())
    print(json.dumps({"archive": str(a.out.resolve()), "entries": len(payloads), "bytes": a.out.stat().st_size,
                      "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
