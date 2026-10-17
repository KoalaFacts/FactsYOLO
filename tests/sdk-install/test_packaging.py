# SPDX-License-Identifier: Apache-2.0
"""Reject untrustworthy local package inputs before packing any binaries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_assets", ROOT / "sdk/dotnet/native_assets.py")
sys.path.insert(0, str(ROOT / "sdk/dotnet"))
from build_package import verify_package_entries
from pack_linux_probe import pack_linux_probe
import native_assets


class RustRuntimeNotices(unittest.TestCase):
    def test_pinned_runtime_notices_cover_observed_missing_components(self):
        toolchain = subprocess.check_output(["rustc", "-vV"], cwd=ROOT, text=True)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "supplement"
            native_assets.stage_runtime_notices(destination, toolchain)
            for name in ("addr2line-0.27.1", "adler2-2.0.1", "object-0.39.1",
                         "miniz_oxide-0.9.1", "memchr-2.8.3", "compiler_builtins-0.1.160"):
                with self.subTest(component=name):
                    texts = [p for p in (destination / name).rglob("*") if p.is_file()]
                    self.assertTrue(texts, name)
                    self.assertTrue(any("Permission is hereby granted" in p.read_text() for p in texts), name)
            self.assertIn("LLVM Exceptions", (destination / "compiler_builtins-0.1.160/LICENSE.txt").read_text())
            self.assertIn("Copyright 2013-2014 RAD Game Tools", (destination / "miniz_oxide-0.9.1/LICENSE").read_text())

    def test_memchr_includes_both_declared_license_alternatives(self):
        toolchain = subprocess.check_output(["rustc", "-vV"], cwd=ROOT, text=True)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "supplement"
            native_assets.stage_runtime_notices(destination, toolchain)
            unlicense = destination / "memchr-2.8.3/UNLICENSE"
            self.assertTrue(unlicense.is_file(), "the versioned memchr notices must include its declared Unlicense alternative")
            self.assertIn("This is free and unencumbered software released into the public domain", unlicense.read_text())

    def test_other_toolchain_requires_new_notice_review(self):
        with tempfile.TemporaryDirectory() as temp:
            for version in ("1.100.0", "1.99.0"):
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "runtime notice.*toolchain"):
                    native_assets.stage_runtime_notices(Path(temp) / "supplement", f"release: {version}\ncommit-hash: unknown\n")

    def test_modified_upstream_notice_is_rejected(self):
        toolchain = subprocess.check_output(["rustc", "-vV"], cwd=ROOT, text=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            copied = root / "sdk/dotnet/runtime-notices"
            shutil.copytree(ROOT / "sdk/dotnet/runtime-notices", copied)
            (copied / "miniz_oxide-0.9.1/LICENSE").write_text("missing upstream attribution")
            with patch.object(native_assets, "ROOT", root), self.assertRaisesRegex(ValueError, "runtime notice hash"):
                native_assets.stage_runtime_notices(root / "output", toolchain)
            self.assertFalse((root / "output").exists())

    def test_git_autocrlf_checkout_preserves_upstream_notice_bytes(self):
        # Exercise Git conversion on Linux too; this is not Windows execution.
        toolchain = subprocess.check_output(["rustc", "-vV"], cwd=ROOT, text=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = root / "repository"
            repository.mkdir()
            shutil.copytree(ROOT / "sdk/dotnet/runtime-notices", repository / "sdk/dotnet/runtime-notices")
            if (ROOT / ".gitattributes").exists():
                shutil.copyfile(ROOT / ".gitattributes", repository / ".gitattributes")
            for command in (["git", "init", "-q"], ["git", "-c", "core.autocrlf=false", "add", "--all"]):
                subprocess.run(command, cwd=repository, check=True, capture_output=True)
            checkout = root / "checkout"
            subprocess.run(["git", "-c", "core.autocrlf=true", "checkout-index", "--all",
                            f"--prefix={checkout.as_posix()}/"], cwd=repository, check=True, capture_output=True)
            with patch.object(native_assets, "ROOT", checkout):
                try:
                    native_assets.stage_runtime_notices(root / "output", toolchain)
                except ValueError as error:
                    self.fail(f"Git line-ending conversion changed an upstream notice: {error}")


class PackageEntries(unittest.TestCase):
    required = {"lib/net10.0/PackProbe.dll", "README.md", "LICENSE", "NOTICE"}
    metadata = {"[Content_Types].xml", "_rels/.rels", "FactsYOLO.LocalTest.nuspec"}
    prefix = "package/services/metadata/core-properties/"

    def entries(self, *properties):
        return sorted(self.required | self.metadata) + [self.prefix + name for name in properties]

    def test_fixed_nuget_properties_name_accepted(self):
        verify_package_entries(self.entries("nuget.psmdcp"), self.required)

    def test_legacy_hex_properties_name_accepted(self):
        verify_package_entries(self.entries("0123456789abcdef.psmdcp"), self.required)

    def test_exactly_one_known_properties_entry_required(self):
        for names in [(), ("nuget.psmdcp", "abcd.psmdcp"), ("abcd.psmdcp", "0123.psmdcp"),
                      ("other.psmdcp",), ("nuget.psmdcp.extra",), ("NUGET.psmdcp",)]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                verify_package_entries(self.entries(*names), self.required)

    def test_duplicate_entries_rejected(self):
        for name in ["LICENSE", self.prefix + "nuget.psmdcp"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "duplicate"):
                verify_package_entries(self.entries("nuget.psmdcp") + [name], self.required)

    def test_missing_or_extra_entries_rejected(self):
        for name in ["model.fytd", "runtimes/win-x64/native/facts_yolo.dll", self.prefix + "extra.txt"]:
            with self.subTest(extra=name), self.assertRaises(ValueError):
                verify_package_entries(self.entries("nuget.psmdcp") + [name], self.required)
        for name in self.required | self.metadata:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                verify_package_entries([n for n in self.entries("nuget.psmdcp") if n != name], self.required)

    def test_unsafe_paths_rejected_even_if_allowlisted(self):
        for name in ["", "/absolute", "../outside", "path/../outside", "path/./file",
                     "path//file", "path/", "C:/outside", "path\\file"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "path"):
                verify_package_entries(self.entries("abcd.psmdcp") + [name], self.required | {name})

    def test_normal_dotnet_pack_metadata_is_accepted(self):
        # Exercise the actual pinned SDK, not a fabricated ZIP or Windows asset.
        with tempfile.TemporaryDirectory(prefix="factsyolo-pack-regression-") as temp:
            root = Path(temp)
            shutil.copyfile(ROOT / "sdk/dotnet/global.json", root / "global.json")
            for name in ["LICENSE", "NOTICE"]:
                shutil.copyfile(ROOT / name, root / name)
            (root / "README.md").write_text("Local metadata regression only; never publish.\n")
            (root / "NuGet.Config").write_text("<configuration><packageSources><clear/></packageSources></configuration>\n")
            project = root / "PackProbe.csproj"
            project.write_text('''<Project Sdk="Microsoft.NET.Sdk">
<PropertyGroup><TargetFramework>net10.0</TargetFramework>
<PackageId>FactsYOLO.LocalTest</PackageId><PackageVersion>0.0.0-pack-regression</PackageVersion>
<PackageReadmeFile>README.md</PackageReadmeFile><PackageLicenseExpression>Apache-2.0</PackageLicenseExpression>
<Description>Local metadata validation regression only.</Description></PropertyGroup>
<ItemGroup><None Include="README.md;LICENSE;NOTICE" Pack="true" PackagePath="/" /></ItemGroup>
</Project>''')
            version = subprocess.check_output(["dotnet", "--version"], cwd=root, text=True).strip()
            self.assertEqual(version, "10.0.401")
            command = ["dotnet", "pack", str(project), "-c", "Release", "-o", str(root / "feed"),
                       "--configfile", str(root / "NuGet.Config")]
            result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=180)
            print(f"ordinary dotnet pack regression: {command!r}\n{result.stdout}", flush=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            with zipfile.ZipFile(root / "feed/FactsYOLO.LocalTest.0.0.0-pack-regression.nupkg") as archive:
                names = archive.namelist()
            self.assertIn(self.prefix + "nuget.psmdcp", names)
            verify_package_entries(names, self.required)


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

    def test_unreviewed_bundle_root_file_rejected(self):
        (self.root / "unreviewed.bin").write_bytes(b"not covered by either RID manifest")
        with self.assertRaisesRegex(ValueError, "root"):
            self.api.verify_assets(self.root, self.commit)

    def test_unreviewed_bundle_root_directory_rejected(self):
        (self.root / "unreviewed").mkdir()
        with self.assertRaisesRegex(ValueError, "root"):
            self.api.verify_assets(self.root, self.commit)


class WindowsFixtureExports(unittest.TestCase):
    @unittest.skipIf(__import__('os').name == 'nt', 'Linux preprocessor check; Windows runs actual cl/link gate')
    def test_missing_export_is_not_declared_dllexport(self):
        import re
        # Preprocess the real Windows declaration branch without compiling a
        # pretend Windows DLL. MSVC forbids undefined dllexport declarations.
        output = subprocess.check_output(['cc', '-E', '-P', '-D_WIN32', '-DMISSING_EXPORT',
                                          '-I', str(ROOT / 'include'), str(ROOT / 'tests/sdk-install/loader_fixture.c')], text=True)
        exports = set(re.findall(r'__declspec\s*\(\s*dllexport\s*\)\s+int32_t\s+(?:__cdecl\s+)?(fy_\w+)', output))
        expected = set((ROOT / 'tests/c/public-symbols.txt').read_text().splitlines()) - {'fy_result_release'}
        self.assertEqual(exports, expected)


class RealSdkLayout(unittest.TestCase):
    def test_production_nuspec_preserves_exact_two_rid_inventory_paths(self):
        # Generate the production nuspec from both real bundles without packing
        # a second archive: the workflow still creates one shared package.
        assets = Path(os.environ.get("FACTSYOLO_SDK_TEST_ASSETS", ROOT / "target/sdk-assets")).resolve()
        commit = native_assets.capture(["git", "rev-parse", "HEAD"]).strip()
        manifests = native_assets.verify_assets(assets, commit)
        with tempfile.TemporaryDirectory(prefix="factsyolo-production-nuspec-") as temp:
            root = Path(temp)
            command = ["dotnet", "msbuild", str(ROOT / "sdk/dotnet/FactsYOLO/FactsYOLO.csproj"),
                       "-restore", "-target:_GetRestoreProjectStyle;GenerateNuspec", "-p:Configuration=Release",
                       "-p:IsPackable=true", "-p:PackageVersion=0.0.0-nuspec-regression",
                       f"-p:NativeAssetsRoot={assets}", f"-p:RepositoryCommit={commit}",
                       f"-p:ArtifactsPath={root / 'artifacts'}", "-p:UseArtifactsOutput=true",
                       f"-p:NuspecOutputPath={root / 'nuspec'}", f"-p:PackageOutputPath={root / 'feed'}",
                       "-p:ContinuePackingAfterGeneratingNuspec=false"]
            result = subprocess.run(command, cwd=ROOT / "sdk/dotnet", text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            print(f"production nuspec regression: {command!r}\n{result.stdout}", flush=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            specs = list((root / "nuspec").glob("*.nuspec"))
            self.assertEqual(len(specs), 1)
            files = [element.attrib for element in ET.parse(specs[0]).getroot().iter()
                     if element.tag.rsplit("}", 1)[-1] == "file"]
            targets = [entry["target"].replace("\\", "/") for entry in files]
            expected = {"lib/net10.0/FactsYOLO.dll", "/LICENSE", "/NOTICE", "/README.md"}
            for rid, manifest in manifests.items():
                expected.add(f"runtimes/{rid}/native/{manifest['library']}")
                expected.update(f"buildinfo/{rid}/{name}" for name in manifest["files"]
                                if name != manifest["library"])
                expected.add(f"buildinfo/{rid}/manifest.json")
            self.assertEqual(len(targets), len(set(targets)), "duplicate production nuspec paths")
            self.assertEqual(set(targets), expected)
            self.assertFalse(list(root.rglob("*.nupkg")), "the regression must not create a second package")

    def test_real_sdk_linux_layout_packs_with_warnings_as_errors(self):
        # CI has staged real bundles before this test. Local runs can select a
        # verified Linux bundle explicitly; missing inputs fail rather than skip.
        assets = Path(os.environ.get("FACTSYOLO_SDK_TEST_LINUX_ASSETS", ROOT / "target/sdk-assets/linux-x64"))
        with tempfile.TemporaryDirectory(prefix="factsyolo-real-sdk-pack-") as temp:
            try:
                result = pack_linux_probe(assets, Path(temp) / "probe")
            except RuntimeError as error:
                self.fail(str(error))
            names = set(result["payload_sha256"])
            self.assertIn("lib/net10.0/FactsYOLO.dll", names)
            self.assertIn("runtimes/linux-x64/native/libfacts_yolo.so", names)
            for component, license_file in [("addr2line-0.27.1", "LICENSE-MIT"),
                                            ("adler2-2.0.1", "LICENSE-MIT"),
                                            ("object-0.39.1", "LICENSE-MIT"),
                                            ("miniz_oxide-0.9.1", "LICENSE"),
                                            ("memchr-2.8.3", "LICENSE-MIT"),
                                            ("compiler_builtins-0.1.160", "LICENSE.txt")]:
                self.assertIn(f"buildinfo/linux-x64/licenses/rust-runtime-supplement/{component}/{license_file}", names)
            self.assertFalse(any(n.startswith(("ref/", "runtimes/any/", "runtimes/win-x64/")) for n in names))


if __name__ == "__main__":
    unittest.main()
