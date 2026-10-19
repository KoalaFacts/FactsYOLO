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
import ctypes
import struct

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk/dotnet"))
import native_assets


class LoaderFixtureCompilation(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "ELF hash lookup requires the Linux loader")
    def test_elf_hash_lookup_with_actual_host_libraries(self):
        with tempfile.TemporaryDirectory(prefix="factsyolo-hash-regression-") as temp:
            directory = Path(temp)
            source = directory / "fixture.c"
            source.write_text("\n".join(f"int {name}(void) {{ return 0; }}"
                                        for name in native_assets.PUBLIC_EXPORTS))
            for style in ["gnu", "sysv", "both"]:
                library = directory / (style + ".so")
                command = ["cc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", str(source),
                           "-Wl,--hash-style=" + style, "-o", str(library)]
                result = subprocess.run(command, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(native_assets.exported_names(library, "linux-x64"), native_assets.PUBLIC_EXPORTS)
                lookup = [sys.executable, "-c", "import ctypes,sys; ctypes.CDLL(sys.argv[1]).fy_detect", str(library)]
                self.assertEqual(subprocess.run(lookup, capture_output=True, timeout=10).returncode, 0)
                original = library.read_bytes()
                data = bytearray(original)
                start = struct.unpack_from("<Q", data, 40)[0]
                size, count = struct.unpack_from("<HH", data, 58)
                sections = [struct.unpack_from("<IIQQQQIIQQ", data, start + i * size) for i in range(count)]
                kind = 5 if style == "sysv" else 0x6ffffff6
                table = next(s for s in sections if s[1] == kind)
                if style == "sysv":
                    buckets, entries = struct.unpack_from("<II", data, table[4])
                    offset, length = table[4] + 8, buckets * 4
                else:
                    words = struct.unpack_from("<I", data, table[4] + 8)[0]
                    offset, length = table[4] + 16, words * 8
                data[offset:offset + length] = bytes(length)
                library.write_bytes(data)
                result = subprocess.run(lookup, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("undefined symbol: fy_detect", result.stderr)
                with self.assertRaisesRegex(ValueError, "unreachable through runtime hash"):
                    native_assets.exported_names(library, "linux-x64")
                print(f"HOST_HASH_EXPORT_AUDIT style={style} ordinary_lookup_failed correctly_rejected", flush=True)
                if style == "sysv":
                    for value in [1, entries + 1]:
                        data = bytearray(original)
                        for entry in range(buckets):
                            struct.pack_into("<I", data, table[4] + 8 + entry * 4, value)
                        struct.pack_into("<I", data, table[4] + 8 + buckets * 4 + 4, 1)
                        library.write_bytes(data)
                        with self.assertRaisesRegex(ValueError, "invalid ELF export SysV hash chain"):
                            native_assets.exported_names(library, "linux-x64")

    def test_host_compiled_fixture_cannot_hide_extra_exports_in_manifest(self):
        # Compile genuine host libraries; these stubs are not inference models.
        rid = "win-x64" if os.name == "nt" else "linux-x64"
        with tempfile.TemporaryDirectory(prefix="factsyolo-export-regression-") as temp:
            directory = Path(temp)
            source = directory / "fixture.c"
            library = directory / native_assets.RIDS[rid][1]
            variants = ["ordinary", "extra"] + (["versioned"] if os.name != "nt" else [])
            for variant in variants:
                extra = variant == "extra"
                symbols = native_assets.PUBLIC_EXPORTS + (["fy_test_export_probe"] if extra else [])
                prefix = "__declspec(dllexport) " if os.name == "nt" else ""
                source.write_text("\n".join(f"{prefix}int {name}(void) {{ return 0; }}" for name in symbols
                                            if variant != "versioned" or name != "fy_detect"))
                if variant == "versioned":
                    with source.open("a") as stream:
                        stream.write('\nint implementation(void) { return 0; }\n'
                                     '__asm__(".symver implementation,fy_detect@VER_1,remove");\n')
                    (directory / "versions.map").write_text("BASE {}; VER_1 {} BASE;\n")
                command = (["cl", "/nologo", "/LD", "/W4", "/WX", str(source),
                            "/Fe:" + str(library), "/Fo:" + str(directory / "fixture.obj")]
                           if os.name == "nt" else
                           ["cc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(library)])
                if variant == "versioned":
                    command.append("-Wl,--version-script=" + str(directory / "versions.map"))
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
                if variant == "versioned":
                    # The real loader cannot resolve this non-default version
                    # by the ordinary ABI name despite its unchanged st_name.
                    loaded = ctypes.CDLL(str(library))
                    with self.assertRaises(AttributeError):
                        loaded.fy_detect
                    with self.assertRaisesRegex(ValueError, "versioned FactsYOLO ELF export"):
                        native_assets.verify_bundle(directory, rid, "test-only")
                    print("HOST_VERSION_EXPORT_AUDIT ordinary_lookup_failed correctly_rejected", flush=True)
                elif extra:
                    with self.assertRaisesRegex(ValueError, "actual native exports"):
                        native_assets.verify_bundle(directory, rid, "test-only")
                else:
                    native_assets.verify_bundle(directory, rid, "test-only")
                if variant != "versioned":
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
