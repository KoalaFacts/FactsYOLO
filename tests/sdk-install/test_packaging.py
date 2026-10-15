# SPDX-License-Identifier: Apache-2.0
"""Reject untrustworthy local package inputs before packing any binaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_assets", ROOT / "sdk/dotnet/native_assets.py")


class Packaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(SPEC.origin).is_file():
            raise AssertionError("Native asset validation is not implemented")
        cls.api = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(cls.api)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit = "a" * 40
        for rid, name in [("linux-x64", "libfacts_yolo.so"), ("win-x64", "facts_yolo.dll")]:
            directory = self.root / rid
            directory.mkdir()
            binary = bytearray(512)
            if rid == "linux-x64":
                binary[:6] = b"\x7fELF\x02\x01"
                struct.pack_into("<H", binary, 18, 62)
            else:
                binary[:2] = b"MZ"
                struct.pack_into("<I", binary, 60, 128)
                binary[128:132] = b"PE\x00\x00"
                struct.pack_into("<H", binary, 132, 0x8664)
            (directory / name).write_bytes(binary)
            (directory / "LICENSE.txt").write_text("test-only license content")
            (directory / "sbom.spdx.json").write_text('{"spdxVersion":"SPDX-2.3"}')
            manifest = {"schema": 1, "rid": rid, "source_commit": self.commit,
                        "library": name, "exports": self.api.PUBLIC_EXPORTS,
                        "licenses": ["LICENSE.txt"], "files": {}}
            for file in directory.iterdir():
                manifest["files"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
            (directory / "manifest.json").write_text(json.dumps(manifest))

    def mutate(self, rid, transform):
        path = self.root / rid / "manifest.json"
        data = json.loads(path.read_text())
        transform(data)
        path.write_text(json.dumps(data))

    def test_two_matching_rids_are_required(self):
        self.assertEqual(set(self.api.verify_assets(self.root, self.commit)), {"linux-x64", "win-x64"})
        (self.root / "win-x64/manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.api.verify_assets(self.root, self.commit)

    def test_wrong_source_rejected(self):
        self.mutate("win-x64", lambda m: m.update(source_commit="b" * 40))
        with self.assertRaisesRegex(ValueError, "source"):
            self.api.verify_assets(self.root, self.commit)

    def test_modified_binary_rejected(self):
        with (self.root / "linux-x64/libfacts_yolo.so").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.api.verify_assets(self.root, self.commit)

    def test_wrong_architecture_rejected_even_with_matching_hash(self):
        file = self.root / "win-x64/facts_yolo.dll"
        data = bytearray(file.read_bytes()); struct.pack_into("<H", data, 132, 0xaa64); file.write_bytes(data)
        self.mutate("win-x64", lambda m: m["files"].update({file.name: hashlib.sha256(data).hexdigest()}))
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.api.verify_assets(self.root, self.commit)

    def test_fixture_export_rejected(self):
        self.mutate("linux-x64", lambda m: m["exports"].append("fy_test_model_create"))
        with self.assertRaisesRegex(ValueError, "exports"):
            self.api.verify_assets(self.root, self.commit)

    def test_missing_license_inventory_rejected(self):
        self.mutate("linux-x64", lambda m: m.update(licenses=[]))
        with self.assertRaisesRegex(ValueError, "license"):
            self.api.verify_assets(self.root, self.commit)

    def test_unlisted_extra_file_rejected(self):
        (self.root / "linux-x64/model.fytd").write_bytes(b"never package weights")
        with self.assertRaisesRegex(ValueError, "file set"):
            self.api.verify_assets(self.root, self.commit)

    def test_traversal_rejected(self):
        self.mutate("linux-x64", lambda m: m["files"].update({"../outside": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "path"):
            self.api.verify_assets(self.root, self.commit)


if __name__ == "__main__":
    unittest.main()
