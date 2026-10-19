#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile every negative loader fixture with the actual host C compiler."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk/dotnet"))
import native_assets


class LoaderFixtureCompilation(unittest.TestCase):
    def test_host_compiled_fixture_cannot_hide_extra_exports_in_manifest(self):
        # Compile genuine host libraries; these stubs are not inference models.
        rid = "win-x64" if os.name == "nt" else "linux-x64"
        with tempfile.TemporaryDirectory(prefix="factsyolo-export-regression-") as temp:
            directory = Path(temp)
            source = directory / "fixture.c"
            library = directory / native_assets.RIDS[rid][1]
            for extra in [False, True]:
                symbols = native_assets.PUBLIC_EXPORTS + (["fy_test_export_probe"] if extra else [])
                prefix = "__declspec(dllexport) " if os.name == "nt" else ""
                source.write_text("\n".join(f"{prefix}int {name}(void) {{ return 0; }}" for name in symbols))
                command = (["cl", "/nologo", "/LD", "/W4", "/WX", str(source),
                            "/Fe:" + str(library), "/Fo:" + str(directory / "fixture.obj")]
                           if os.name == "nt" else
                           ["cc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(library)])
                result = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout)
                # The manifest deliberately claims only the public exports in
                # both runs. Hashes are accurate; actual exports must decide.
                files = {p.name: native_assets.sha256(p) for p in directory.iterdir()
                         if p.is_file() and p.name != "manifest.json"}
                (directory / "manifest.json").write_text(json.dumps({"schema": 1, "rid": rid,
                    "source_commit": "test-only", "library": library.name,
                    "exports": native_assets.PUBLIC_EXPORTS, "licenses": [source.name], "files": files}))
                # Minimal test inventory; no assertion of redistribution clearance.
                (directory / "sbom.spdx.json").write_text("{}")
                manifest = json.loads((directory / "manifest.json").read_text())
                manifest["files"]["sbom.spdx.json"] = native_assets.sha256(directory / "sbom.spdx.json")
                (directory / "manifest.json").write_text(json.dumps(manifest))
                if extra:
                    with self.assertRaisesRegex(ValueError, "actual native exports"):
                        native_assets.verify_bundle(directory, rid, "test-only")
                else:
                    native_assets.verify_bundle(directory, rid, "test-only")
                print(f"HOST_EXPORT_AUDIT rid={rid} extra_fixture_export={extra} correctly_{'rejected' if extra else 'accepted'}", flush=True)

    def test_all_variants_compile_with_strict_host_warnings(self):
        # Compile only: WITH_DEPENDENCY intentionally needs a separate DLL at
        # link time. The installed-consumer gate still links and runs every case.
        with tempfile.TemporaryDirectory(prefix="factsyolo-loader-compile-") as temp:
            for variant in ["MISSING_EXPORT", "WRONG_ABI", "DIAGNOSTIC_BOUNDARY", "WITH_DEPENDENCY"]:
                with self.subTest(variant=variant):
                    source = ROOT / "tests/sdk-install/loader_fixture.c"
                    output = Path(temp) / (variant + (".obj" if os.name == "nt" else ".o"))
                    if os.name == "nt":
                        command = ["cl", "/nologo", "/c", "/W4", "/WX", "/TC", "/D" + variant,
                                   "/I" + str(ROOT / "include"), str(source), "/Fo:" + str(output)]
                    else:
                        command = ["cc", "-c", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror",
                                   "-D" + variant, "-I", str(ROOT / "include"), str(source), "-o", str(output)]
                    result = subprocess.run(command, cwd=temp, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, timeout=120)
                    print(f"strict host fixture compilation: {command!r}\n{result.stdout}", flush=True)
                    self.assertEqual(result.returncode, 0, f"{command!r}\n{result.stdout}")
                    self.assertTrue(output.is_file())
                    self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
