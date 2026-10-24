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
    @staticmethod
    def elf_sections(data):
        start = struct.unpack_from("<Q", data, 40)[0]
        size, count = struct.unpack_from("<HH", data, 58)
        return [(start + i * size, struct.unpack_from("<IIQQQQIIQQ", data, start + i * size))
                for i in range(count)]

    def elf_symbol_offset(self, data, name):
        sections = self.elf_sections(data)
        _, symbols = next(s for s in sections if s[1][1] == 11)
        strings = sections[symbols[6]][1]
        expected = name.encode("ascii") + b"\0"
        for offset in range(symbols[4], symbols[4] + symbols[5], 24):
            start = strings[4] + struct.unpack_from("<I", data, offset)[0]
            if data[start:start + len(expected)] == expected:
                return offset
        self.fail(f"host fixture has no dynamic symbol {name}")

    def compile_elf_fixture(self, directory, body):
        source = directory / "fixture.c"
        library = directory / native_assets.RIDS["linux-x64"][1]
        source.write_text("/* SPDX-License-Identifier: Apache-2.0 */\n" + body)
        command = ["cc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", str(source),
                   "-Wl,--hash-style=gnu", "-o", str(library)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        return library

    def compile_elf_export_fixture(self, directory, extra):
        library = self.compile_elf_fixture(directory, "\n".join(
            f"int {name}(void) {{ return 0; }}" for name in native_assets.PUBLIC_EXPORTS + [extra]))
        self.assertEqual(native_assets.exported_names(library, "linux-x64"),
                         sorted(native_assets.PUBLIC_EXPORTS + [extra]))
        return library

    def assert_loader_resolves(self, library, extra):
        # Load only the stubs compiled in this test. Each lookup gets a fresh
        # process so patching a library never changes a live memory mapping.
        lookup = [sys.executable, "-c",
                  "import ctypes,sys; library=ctypes.CDLL(sys.argv[1]); "
                  "assert all(getattr(library, name)() == 0 for name in sys.argv[2:])",
                  str(library), *native_assets.PUBLIC_EXPORTS, extra]
        result = subprocess.run(lookup, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        print(f"HOST_ELF_EXPORT_LOOKUP symbol={extra} ordinary_lookup_succeeded", flush=True)

    @staticmethod
    def write_fixture_manifest(library):
        directory = library.parent
        # Accurate hashes with a claimed public ABI and minimal test inventory;
        # this is not an assertion of redistribution clearance.
        (directory / "sbom.spdx.json").write_text("{}")
        files = {p.name: native_assets.sha256(p) for p in directory.iterdir()
                 if p.is_file() and p.name != "manifest.json"}
        (directory / "manifest.json").write_text(json.dumps({
            "schema": 1, "rid": "linux-x64", "source_commit": "test-only",
            "library": library.name, "exports": native_assets.PUBLIC_EXPORTS,
            "licenses": ["fixture.c"], "files": files}))

    def assert_extra_export_rejected(self, library, extra):
        self.write_fixture_manifest(library)
        # Either reject inconsistent/unsupported ELF metadata or include the
        # runtime-visible extra export; silently losing it is never acceptable.
        try:
            names = native_assets.exported_names(library, "linux-x64")
        except ValueError:
            pass
        else:
            self.assertEqual(names, sorted(native_assets.PUBLIC_EXPORTS + [extra]),
                             f"audit omitted loader-visible extra export {extra}")
        with self.assertRaises(ValueError):
            native_assets.verify_bundle(library.parent, "linux-x64", "test-only")

    @unittest.skipUnless(sys.platform == "linux", "ELF fixture requires the Linux loader")
    def test_elf_empty_gnu_hash_with_unhashed_imports_has_no_exports(self):
        with tempfile.TemporaryDirectory(prefix="factsyolo-empty-gnu-hash-") as temp:
            library = self.compile_elf_fixture(Path(temp),
                "static void __attribute__((constructor)) initialize(void) {}\n")
            data = library.read_bytes()
            sections = self.elf_sections(data)
            _, symbols = next(s for s in sections if s[1][1] == 11)
            _, table = next(s for s in sections if s[1][1] == 0x6ffffff6)
            buckets, first, words, _ = struct.unpack_from("<IIII", data, table[4])
            bucket = table[4] + 16 + words * 8
            self.assertGreater(symbols[5] // 24, first)
            self.assertEqual(data[bucket:bucket + buckets * 4], bytes(buckets * 4))
            self.assertEqual(table[5], 16 + words * 8 + buckets * 4)
            lookup = [sys.executable, "-c",
                      "import ctypes,sys; library=ctypes.CDLL(sys.argv[1]); "
                      "assert not any(hasattr(library, name) for name in sys.argv[2:])",
                      str(library), *native_assets.PUBLIC_EXPORTS]
            result = subprocess.run(lookup, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            print(f"HOST_EMPTY_GNU_HASH dynsym_count={symbols[5] // 24} first={first} "
                  f"hash_size={table[5]} chain_bytes=0 ordinary_load_succeeded", flush=True)
            try:
                names = native_assets.exported_names(library, "linux-x64")
            except ValueError as error:
                self.fail(f"ordinary host library with empty GNU hash was rejected: {error}")
            self.assertEqual(names, [])
            self.write_fixture_manifest(library)
            with self.assertRaisesRegex(ValueError, "actual native exports"):
                native_assets.verify_bundle(library.parent, "linux-x64", "test-only")

    @unittest.skipUnless(sys.platform == "linux", "ELF fixture requires the Linux loader")
    def test_elf_gnu_hash_count_rejects_invalid_buckets_and_unterminated_chain(self):
        with tempfile.TemporaryDirectory(prefix="factsyolo-gnu-hash-bounds-") as temp:
            library = self.compile_elf_export_fixture(Path(temp), "fy_test_export_probe")
            original = library.read_bytes()
            _, table = next(s for s in self.elf_sections(original) if s[1][1] == 0x6ffffff6)
            buckets, first, words, _ = struct.unpack_from("<IIII", original, table[4])
            bucket = table[4] + 16 + words * 8
            chain = bucket + buckets * 4
            self.assertGreater(first, 1)
            for mutation in ["bucket below first", "oversized bucket", "unterminated chain"]:
                with self.subTest(mutation=mutation):
                    data = bytearray(original)
                    if mutation == "unterminated chain":
                        for offset in range(chain, table[4] + table[5], 4):
                            value = struct.unpack_from("<I", data, offset)[0]
                            struct.pack_into("<I", data, offset, value & ~1)
                    else:
                        value = first - 1 if mutation == "bucket below first" else 0xffffffff
                        struct.pack_into("<I", data, bucket, value)
                    library.write_bytes(data)
                    # Damaged tables are inspected offline, never loaded.
                    with self.assertRaises(ValueError):
                        native_assets.exported_names(library, "linux-x64")

    @unittest.skipUnless(sys.platform == "linux", "ELF fixture requires the Linux loader")
    def test_elf_gnu_hash_cannot_hide_trailing_export_with_short_dynsym(self):
        with tempfile.TemporaryDirectory(prefix="factsyolo-short-dynsym-") as temp:
            extra = "fy_hidden_extra"
            library = self.compile_elf_export_fixture(Path(temp), extra)
            data = bytearray(library.read_bytes())
            sections = self.elf_sections(data)
            self.assertIn(0x6ffffff6, [section[1] for _, section in sections])
            self.assertNotIn(5, [section[1] for _, section in sections])
            header, symbols = next(s for s in sections if s[1][1] == 11)
            # Check the actual linker order: removing this last entry must leave
            # every public ABI symbol in the section's claimed inventory.
            self.assertEqual(self.elf_symbol_offset(data, extra), symbols[4] + symbols[5] - 24)
            struct.pack_into("<Q", data, header + 32, symbols[5] - 24)
            library.write_bytes(data)
            self.assert_loader_resolves(library, extra)
            self.assert_extra_export_rejected(library, extra)

    @unittest.skipUnless(sys.platform == "linux", "ELF fixture requires the Linux loader")
    def test_elf_gnu_unique_binding_cannot_hide_loader_visible_export(self):
        with tempfile.TemporaryDirectory(prefix="factsyolo-unique-export-") as temp:
            extra = "fy_test_export_probe"
            library = self.compile_elf_export_fixture(Path(temp), extra)
            data = bytearray(library.read_bytes())
            offset = self.elf_symbol_offset(data, extra)
            data[offset + 4] = (10 << 4) | (data[offset + 4] & 15)  # STB_GNU_UNIQUE
            library.write_bytes(data)
            self.assert_loader_resolves(library, extra)
            self.assert_extra_export_rejected(library, extra)

    @unittest.skipUnless(sys.platform == "linux", "ELF fixture requires the Linux loader")
    def test_elf_st_other_high_bit_cannot_hide_default_visibility_export(self):
        with tempfile.TemporaryDirectory(prefix="factsyolo-visible-export-") as temp:
            extra = "fy_test_export_probe"
            library = self.compile_elf_export_fixture(Path(temp), extra)
            data = bytearray(library.read_bytes())
            offset = self.elf_symbol_offset(data, extra)
            self.assertEqual(data[offset + 5] & 3, 0)  # STV_DEFAULT
            data[offset + 5] |= 0x04
            library.write_bytes(data)
            self.assert_loader_resolves(library, extra)
            self.assert_extra_export_rejected(library, extra)

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

    @unittest.skipUnless(os.name == "nt", "requires MSVC and the Windows PE loader")
    def test_pe_ordinal_only_export_cannot_hide_in_named_manifest(self):
        # A sparse EAT is valid: unoccupied ordinal slots are zero, and every
        # occupied slot must have a name. Then add a real NONAME function,
        # callable by ordinal but missing from AddressOfNames. These are
        # host-compiled stubs, not detector models.
        with tempfile.TemporaryDirectory(prefix="factsyolo-pe-ordinal-") as temp:
            directory = Path(temp)
            source = directory / "fixture.c"
            definition = directory / "fixture.def"
            library = directory / native_assets.RIDS["win-x64"][1]
            source.write_text("/* SPDX-License-Identifier: Apache-2.0 */\n" +
                              "\n".join(f"int {name}(void) {{ return 0; }}"
                                        for name in native_assets.PUBLIC_EXPORTS) +
                              "\nint ordinal_probe(void) { return 0; }\n")
            # Non-consecutive named ordinals guarantee real zero-filled holes.
            named = "\n".join(f"{name} @{3 * i + 1}"
                              for i, name in enumerate(native_assets.PUBLIC_EXPORTS))
            for extra in (False, True):
                with self.subTest(ordinal_only=extra):
                    definition.write_text("LIBRARY facts_yolo\nEXPORTS\n" + named +
                                          ("\nordinal_probe @123 NONAME" if extra else "") + "\n")
                    command = ["cl", "/nologo", "/LD", "/W4", "/WX", str(source),
                               "/Fe:" + str(library), "/Fo:" + str(directory / "fixture.obj"),
                               "/link", "/DEF:" + str(definition)]
                    compiled = subprocess.run(command, cwd=directory, capture_output=True,
                                              text=True, timeout=120)
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertTrue(library.is_file())
                    lookup = [sys.executable, "-c",
                              "import ctypes,sys; lib=ctypes.WinDLL(sys.argv[1]); "
                              "assert all(getattr(lib,name)() == 0 for name in sys.argv[2:]); " +
                              ("assert lib[123]() == 0" if extra else "pass"),
                              str(library), *native_assets.PUBLIC_EXPORTS]
                    loaded = subprocess.run(lookup, cwd=directory, capture_output=True,
                                            text=True, timeout=10)
                    self.assertEqual(loaded.returncode, 0, loaded.stdout + loaded.stderr)
                    (directory / "sbom.spdx.json").write_text("{}")
                    files = {p.name: native_assets.sha256(p) for p in directory.iterdir()
                             if p.is_file() and p.name != "manifest.json"}
                    (directory / "manifest.json").write_text(json.dumps({
                        "schema": 1, "rid": "win-x64", "source_commit": "test-only",
                        "library": library.name, "exports": native_assets.PUBLIC_EXPORTS,
                        "licenses": [source.name], "files": files}))
                    if extra:
                        with self.assertRaisesRegex(ValueError, "ordinal-only"):
                            native_assets.verify_bundle(directory, "win-x64", "test-only")
                    else:
                        native_assets.verify_bundle(directory, "win-x64", "test-only")
                    print(f"HOST_PE_ORDINAL_EXPORT sparse_holes_accepted "
                          f"extra_ordinal={'rejected' if extra else 'absent'} "
                          f"loader_lookup_succeeded", flush=True)

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
