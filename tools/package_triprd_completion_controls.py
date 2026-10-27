#!/usr/bin/env python3
"""Build a self-contained controls source archive without Git or older ZIPs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "MANIFEST_COMPLETION_CONTROLS.json"
# Deliberately shallow, extension-restricted globs: datasets contains reader
# source as well as large local datasets, and must never be included recursively.
SOURCE_GLOBS = (
    "models/*.py", "tools/*.py", "tools/*.sh", "evaluation/*.py",
    "datasets/*.py", "tests/*.py", "configs/*.yaml", "configs/*.yml",
)
REQUIRED_FILES = (
    "README_COMPLETION_CONTROLS.md", "README_COMPLETION_V6.md",
    "requirements-completion-v6.txt", "requirements-task-v4.txt",
    "requirements-correction.txt", "requirements-triprd.txt",
    "models/triprd_completion_control.py", "models/triprd_completion.py",
    "models/triprd_completion_loss.py", "tools/run_triprd_completion_control.py",
    "tools/evaluate_triprd_completion_control.py",
    "tools/run_triprd_completion_controls_queue.py",
    "tools/run_triprd_completion_controls.sh",
    "tools/summarize_triprd_completion_controls.py",
    "tools/package_triprd_completion_controls.py",
)
LOCAL_PACKAGES = frozenset(("models", "tools", "evaluation", "datasets"))
EXCLUDED = (
    ".git and hidden environment directories", "dinov3 external checkout",
    "dataset images/archives (dataset reader Python files are included)",
    "results and logs", "checkpoints and pretrained weights",
    "deliverables and all previous archives", "virtual environments and caches",
)


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (bool(name) and not path.is_absolute() and "\\" not in name
            and ":" not in name and all(part not in (".", "..", "") for part in path.parts))


def collect_sources(root: Path) -> dict[str, bytes]:
    """Use current on-disk source, including newly added untracked controls."""
    root = Path(root).resolve()
    names = set(REQUIRED_FILES)
    for pattern in SOURCE_GLOBS:
        names.update(path.relative_to(root).as_posix() for path in root.glob(pattern)
                     if path.is_file())
    payloads = {}
    for name in sorted(names):
        path = root / name
        if not _safe_name(name) or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe source path: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Required controls source missing: {name}")
        payloads[name] = path.read_bytes()
    validate_local_imports(payloads)
    return payloads


def validate_local_imports(payloads: dict[str, bytes]) -> None:
    """Catch omitted first-party modules without importing PyTorch or the runner.

    Third-party packages and the externally supplied DINO checkout are handled
    by the requirements/environment. Every local Python source directory is
    collected, including files used by runtime CODE_FILES provenance hashes.
    """
    names = set(payloads)
    namespaces = {str(PurePosixPath(name).parent) for name in names}

    def exists(module: str) -> bool:
        stem = module.replace(".", "/")
        return stem + ".py" in names or stem + "/__init__.py" in names or stem in namespaces

    for name, data in payloads.items():
        if not name.endswith(".py"):
            continue
        tree = ast.parse(data.decode("utf-8-sig"), filename=name)
        package = list(PurePosixPath(name).parent.parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - node.level + 1]
                    modules = [".".join(base + ([node.module] if node.module else []))]
                else:
                    modules = [node.module or ""]
                # Namespace packages have no __init__ exports; these imports
                # identify actual sibling modules, e.g. from tools import v4.
                if modules[0] in LOCAL_PACKAGES and modules[0] + "/__init__.py" not in names:
                    modules += [modules[0] + "." + alias.name for alias in node.names if alias.name != "*"]
            else:
                continue
            for module in modules:
                if module.split(".")[0] in LOCAL_PACKAGES and not exists(module):
                    raise ValueError(f"Missing local import {module!r} required by {name}")


def verify_archive(path: Path) -> dict:
    """Validate the exact entry set, safe paths, CRC, byte counts and SHA256."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not _safe_name(name) for name in names):
            raise ValueError("Archive contains duplicate or unsafe paths")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        expected = manifest["files"]
        if set(names) != set(expected) | {MANIFEST_NAME}:
            raise ValueError("Archive entries do not match the manifest")
        for name, record in expected.items():
            data = archive.read(name)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Archive content hash mismatch: {name}")
        validate_local_imports({name: archive.read(name) for name in expected})
    return manifest


def build_archive(root: Path, output: Path) -> dict:
    payloads = collect_sources(root)
    manifest = {
        "method": "triprd_completion_controls_v1",
        "source_collection": "current source allowlist; no git command or prior archive required",
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in payloads.items()},
        "manifest_note": "files covers every payload entry; this manifest does not hash itself",
        "excluded": list(EXCLUDED),
        "external_resources": ["CUDA-compatible Python/PyTorch environment", "DINOv3 source and weights",
                               "VDT/COD datasets", "fixed split manifest"],
        "groups": {"A": "COD only", "B": "RGB/GT source -> COD",
                   "C": "RGB teacher -> RGB student -> COD", "D": "RGB/T/D teacher -> RGB student -> COD"},
        "accuracy_evidence": "Source package only; no real-data accuracy gain or CUDA throughput claim.",
    }
    payloads[MANIFEST_NAME] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, data)
    verify_archive(output)
    return {"archive": str(output.resolve()), "entries": len(payloads),
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "deliverables/triprd_completion_controls_2026-10-27.zip")
    parser.add_argument("--verify", type=Path, help="Verify an existing archive instead of packaging")
    args = parser.parse_args(argv)
    if args.verify:
        manifest = verify_archive(args.verify)
        result = {"archive": str(args.verify.resolve()), "verified": True,
                  "payload_files": len(manifest["files"])}
    else:
        result = build_archive(ROOT, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
