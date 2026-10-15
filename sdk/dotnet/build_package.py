#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack exactly one temporary dual-RID package; never upload/publish it."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from native_assets import ROOT, RIDS, capture, clean_commit, sha256, verify_assets


def main(assets, output):
    commit = clean_commit()
    manifests = verify_assets(assets, commit)
    version = f"0.0.0-local.{commit[:12]}"
    if capture(["dotnet", "--version"], ROOT / "sdk/dotnet").strip() != "10.0.401":
        raise ValueError(".NET SDK 10.0.401 required")
    output.mkdir(parents=True, exist_ok=False)
    command = ["dotnet", "pack", ROOT / "sdk/dotnet/FactsYOLO/FactsYOLO.csproj", "-c", "Release", "-o", output,
               "-p:IsPackable=true", f"-p:PackageVersion={version}", f"-p:NativeAssetsRoot={assets}", f"-p:RepositoryCommit={commit}"]
    result = subprocess.run(list(map(str, command)), cwd=ROOT / "sdk/dotnet", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    (output / "pack.log").write_text(result.stdout)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError("dotnet pack failed")
    package = output / f"FactsYOLO.LocalTest.{version}.nupkg"
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate package entries")
        required = {"ref/net10.0/FactsYOLO.dll", "runtimes/any/lib/net10.0/FactsYOLO.dll", "LICENSE", "NOTICE", "README.md"}
        for rid, manifest in manifests.items():
            entry = f"runtimes/{rid}/native/{RIDS[rid][1]}"
            required.add(entry)
            import hashlib
            if hashlib.sha256(archive.read(entry)).hexdigest() != manifest["files"][manifest["library"]]:
                raise ValueError("packaged native hash differs")
            for name, digest in manifest["files"].items():
                if name == manifest["library"]:
                    continue
                entry = f"buildinfo/{rid}/{name}"
                required.add(entry)
                if hashlib.sha256(archive.read(entry)).hexdigest() != digest:
                    raise ValueError(f"packaged inventory hash differs: {entry}")
            required.add(f"buildinfo/{rid}/manifest.json")
        if not required.issubset(names):
            raise ValueError(f"package entries missing: {required - set(names)}")
        if any(n.endswith((".fytd", ".rgb", ".a", ".lib")) for n in names):
            raise ValueError("unexpected model/pixel/static/import asset in package")
    summary = {"source_commit": commit, "package": package.name, "version": version,
               "sha256": sha256(package), "rids": list(RIDS), "local_test_only": True}
    (output / "package.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise ValueError("usage: build_package.py <two-rid-assets-root> <new-local-feed>")
        main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr); sys.exit(1)
