"""Source archives must work without old bundles and must not ship run data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import package_triprd_completion_controls as package


class CompletionControlsPackageTests(unittest.TestCase):
    def make_source(self, root):
        # A fresh source tree: deliberately no .git metadata and no old ZIPs.
        for name in package.REQUIRED_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("pass\n" if name.endswith(".py") else "source\n", encoding="utf-8")
        (root / "datasets").mkdir()
        (root / "datasets/reader.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "models/dependency.py").write_text("VALUE = 2\n", encoding="utf-8")
        (root / "models/triprd_completion_control.py").write_text(
            "from models.dependency import VALUE\n", encoding="utf-8")

    def test_fresh_source_package_excludes_run_data_and_verifies_every_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root)
            excluded = (".git/config", ".venv/lib/secret.py", "dinov3/hubconf.py",
                        "datasets/images/a.png", "datasets/archive.zip", "datasets/nested/private.py",
                        "results/run/comparison.json", "checkpoints/best.pt",
                        "tools/__pycache__/thing.pyc", "models/model.safetensors",
                        "deliverables/old.zip", "logs/train.log")
            for name in excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"must not ship")
            result = package.build_archive(root, root / "deliverables/new.zip")
            manifest = package.verify_archive(Path(result["archive"]))
            self.assertIn("datasets/reader.py", manifest["files"])
            self.assertIn("models/dependency.py", manifest["files"])
            self.assertTrue(set(excluded).isdisjoint(manifest["files"]))
            with zipfile.ZipFile(result["archive"]) as archive:
                self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {package.MANIFEST_NAME})
                for name, record in manifest["files"].items():
                    data = archive.read(name)
                    self.assertEqual(record["bytes"], len(data))
                    self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
            first = Path(result["archive"]).read_bytes()
            package.build_archive(root, root / "deliverables/new.zip")
            self.assertEqual(first, (root / "deliverables/new.zip").read_bytes())

    def test_missing_first_party_dependency_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root)
            (root / "models/dependency.py").unlink()
            with self.assertRaisesRegex(ValueError, "Missing local import"):
                package.build_archive(root, root / "new.zip")
            self.assertFalse((root / "new.zip").exists())

    def test_tampered_payload_is_rejected_even_with_valid_zip_crc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_source(root)
            package.build_archive(root, root / "good.zip")
            with zipfile.ZipFile(root / "good.zip") as original:
                entries = {name: original.read(name) for name in original.namelist()}
            entries["models/dependency.py"] = b"VALUE = 999\n"
            with zipfile.ZipFile(root / "bad.zip", "w") as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                package.verify_archive(root / "bad.zip")

    def test_namespace_sibling_imports_are_checked(self):
        with self.assertRaisesRegex(ValueError, "tools.missing"):
            package.validate_local_imports({"tools/main.py": b"from tools import missing\n"})

    def test_relative_imports_are_checked(self):
        with self.assertRaisesRegex(ValueError, "models.missing"):
            package.validate_local_imports({"models/__init__.py": b"from .missing import Value\n"})


if __name__ == "__main__":
    unittest.main()
