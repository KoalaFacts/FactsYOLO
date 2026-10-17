#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict Linux layout regression using real SDK output and a verified bundle.

This diagnostic projects the production project's evaluated managed pack items
into a Linux-only nuspec. It does not execute or replace the dual-RID pack gate.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk/dotnet"))
from native_assets import capture, sha256, verify_bundle
from build_package import verify_package_entries


def pack_linux_probe(assets, output):
    commit = capture(["git", "rev-parse", "HEAD"]).strip()
    manifest = verify_bundle(assets, "linux-x64", commit)
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, args):
        args = list(map(str, args))
        result = subprocess.run(args, cwd=ROOT / "sdk/dotnet", text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        (output / f"{label}.log").write_text(result.stdout)
        commands.append(dict(label=label, argv=args, exit=result.returncode))
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label}: exit {result.returncode}\n{result.stdout}")
        return result.stdout

    if run("sdk-version", ["dotnet", "--version"]).strip() != "10.0.401":
        raise ValueError(".NET SDK 10.0.401 required")
    project = ROOT / "sdk/dotnet/FactsYOLO/FactsYOLO.csproj"
    artifacts = output / "artifacts"
    run("build-real-sdk", ["dotnet", "build", project, "-c", "Release", "--artifacts-path", artifacts])
    evaluation = json.loads(run("evaluated-production-items", ["dotnet", "msbuild", project,
        "-p:Configuration=Release", f"-p:ArtifactsPath={artifacts}", "-p:UseArtifactsOutput=true", "-p:IsPackable=true",
        "-getProperty:TargetPath,TargetFramework,BuildOutputTargetFolder,TreatWarningsAsErrors,NoWarn,WarningsNotAsErrors,NoPackageAnalysis",
        "-getItem:None"]))
    props = evaluation["Properties"]
    if props["TreatWarningsAsErrors"] != "true" or "NU5131" in props["NoWarn"] or props["WarningsNotAsErrors"] or props["NoPackageAnalysis"]:
        raise ValueError("strict production pack warning policy changed")
    assembly = Path(props["TargetPath"])
    payload = output / "payload"
    entries = {}

    def add(source, name):
        target = payload / name
        if name in entries:
            raise ValueError(f"duplicate probe payload: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries[name] = sha256(target)

    framework = props["TargetFramework"]
    add(assembly, f"{props['BuildOutputTargetFolder']}/{framework}/{assembly.name}")
    for item in evaluation["Items"]["None"]:
        if item.get("Pack") == "true" and item.get("FullPath") == str(assembly):
            add(assembly, item["PackagePath"].rstrip("/") + "/" + assembly.name)
    for name in ["LICENSE", "NOTICE"]:
        add(ROOT / name, name)
    add(ROOT / "sdk/dotnet/README.md", "README.md")
    add(assets / manifest["library"], "runtimes/linux-x64/native/" + manifest["library"])
    for name in [*manifest["files"], "manifest.json"]:
        if name != manifest["library"]:
            add(assets / name, "buildinfo/linux-x64/" + name)

    version = f"0.0.0-linux-layout.{commit[:12]}"
    pack = output / "pack"; pack.mkdir()
    (pack / "Probe.csproj").write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>
<TargetFramework>net10.0</TargetFramework><IsPackable>true</IsPackable>
<IncludeBuildOutput>false</IncludeBuildOutput><EnableDefaultItems>false</EnableDefaultItems>
<TreatWarningsAsErrors>true</TreatWarningsAsErrors></PropertyGroup></Project>\n''')
    spec = ET.Element("package")
    metadata = ET.SubElement(spec, "metadata")
    for name, value in dict(id="FactsYOLO.LocalTest", version=version, authors="FactsYOLO contributors",
                           description="Linux-only local SDK layout regression; not a dual-platform release.", readme="README.md").items():
        ET.SubElement(metadata, name).text = value
    ET.SubElement(metadata, "license", {"type": "expression"}).text = "Apache-2.0"
    ET.SubElement(metadata, "repository", {"type": "git", "url": "https://github.com/KoalaFacts/FactsYOLO", "commit": commit})
    ET.SubElement(ET.SubElement(metadata, "dependencies"), "group", {"targetFramework": framework})
    files = ET.SubElement(spec, "files")
    for name in sorted(entries):
        ET.SubElement(files, "file", {"src": name, "target": name})
    nuspec = pack / "Probe.nuspec"
    ET.ElementTree(spec).write(nuspec, encoding="utf-8", xml_declaration=True)
    run("restore-probe", ["dotnet", "restore", pack / "Probe.csproj"])
    run("strict-pack", ["dotnet", "pack", pack / "Probe.csproj", "--no-build", "--no-restore",
        f"-p:NuspecFile={nuspec}", f"-p:NuspecBasePath={payload}", "-o", output / "feed"])
    package = output / f"feed/FactsYOLO.LocalTest.{version}.nupkg"
    with zipfile.ZipFile(package) as archive:
        verify_package_entries(archive.namelist(), set(entries))
        for name, digest in entries.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"payload hash changed: {name}")
    summary = dict(source_commit=commit, source_dirty=bool(capture(["git", "status", "--porcelain"]).strip()),
                   package=str(package), version=version, package_sha256=sha256(package), payload_sha256=entries,
                   native_sha256=manifest["files"][manifest["library"]], managed_sha256=sha256(assembly),
                   scope="Linux-only production-layout projection", formal_dual_rid_pack=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: pack_linux_probe.py <verified-linux-bundle> <new-output-directory>")
    print(json.dumps(pack_linux_probe(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()), indent=2))
