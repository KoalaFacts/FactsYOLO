#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage actual host native assets and audit the two inputs to a local test package."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from native_exports import exported_names

ROOT = Path(__file__).resolve().parents[2]
RIDS = {"linux-x64": ("x86_64-unknown-linux-gnu", "libfacts_yolo.so"),
        "win-x64": ("x86_64-pc-windows-msvc", "facts_yolo.dll")}
PUBLIC_EXPORTS = sorted((ROOT / "tests/c/public-symbols.txt").read_text().splitlines())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_architecture(path, rid):
    data = path.read_bytes()
    valid = False
    if rid == "linux-x64":
        valid = len(data) >= 64 and data[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<H", data, 18)[0] == 62
    elif rid == "win-x64" and len(data) >= 64 and data[:2] == b"MZ":
        pe = struct.unpack_from("<I", data, 60)[0]
        valid = pe + 6 <= len(data) and data[pe:pe+4] == b"PE\x00\x00" and struct.unpack_from("<H", data, pe+4)[0] == 0x8664
    if not valid:
        raise ValueError(f"wrong architecture for {rid}: {path}")


def verify_bundle(directory, rid, commit):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1 or manifest.get("rid") != rid:
        raise ValueError("invalid manifest schema or RID")
    if manifest.get("source_commit") != commit:
        raise ValueError(f"native source differs: {rid}")
    if manifest.get("library") != RIDS[rid][1] or manifest.get("exports") != PUBLIC_EXPORTS:
        raise ValueError(f"native library name or exports differ: {rid}")
    files = manifest.get("files", {})
    licenses = manifest.get("licenses", [])
    if not licenses or not all(name in files for name in licenses):
        raise ValueError("missing license inventory")
    if "sbom.spdx.json" not in files or manifest["library"] not in files:
        raise ValueError("native binary or SBOM is missing")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"unsafe manifest path: {name}")
    actual = {file.relative_to(directory).as_posix() for file in directory.rglob("*") if file.is_file() and file != manifest_path}
    if set(files) != actual:
        raise ValueError(f"manifest file set differs: {rid}")
    for name, digest in files.items():
        file = directory / name
        if file.is_symlink() or not file.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"unsafe manifest path: {name}")
        if sha256(file) != digest:
            raise ValueError(f"asset hash differs: {rid}/{name}")
    check_architecture(directory / manifest["library"], rid)
    if exported_names(directory / manifest["library"], rid) != PUBLIC_EXPORTS:
        raise ValueError(f"actual native exports differ: {rid}")
    return manifest


def verify_assets(directory, commit):
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(RIDS):
        raise ValueError("asset root must contain exactly linux-x64 and win-x64 directories")
    if any(not p.is_dir() or p.is_symlink() for p in directory.iterdir()):
        raise ValueError("asset root contains a non-directory or symbolic link")
    return {rid: verify_bundle(directory / rid, rid, commit) for rid in RIDS}


def capture(args, cwd=ROOT, env=None):
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    if result.returncode:
        raise RuntimeError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def clean_commit():
    if capture(["git", "status", "--porcelain"]).strip():
        raise ValueError("native/package staging requires a clean source worktree")
    # Git status trusts assume-unchanged and skip-worktree index flags.
    # Reject hidden tracked sources, as the measurement provenance guard does.
    entries = capture(["git", "ls-files", "-v", "-z"]).split("\0")
    if any(not entry.startswith("H ") for entry in entries if entry):
        raise ValueError("native/package staging source index hides tracked files")
    return capture(["git", "rev-parse", "HEAD"]).strip()


def selected_inventory(target):
    metadata = json.loads(capture(["cargo", "metadata", "--format-version", "1", "--locked", "--offline",
                                   "--no-default-features", "--features", "td1", "--filter-platform", target]))
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    root = next(p["id"] for p in metadata["packages"] if p["name"] == "facts-yolo")
    selected = {}
    pending = [(root, "native-source")]
    seen = set()
    while pending:
        identifier, role = pending.pop()
        if (identifier, role) in seen:
            continue
        seen.add((identifier, role))
        selected.setdefault(identifier, set()).add(role)
        for dep in nodes[identifier]["deps"]:
            for kind in dep["dep_kinds"]:
                if kind["kind"] in (None, "build"):
                    pending.append((dep["pkg"], "build-only" if role == "build-only" or kind["kind"] == "build" else "native-source"))
    return [(packages[key], sorted(roles)) for key, roles in sorted(selected.items())]


def stage_runtime_notices(destination, toolchain):
    """Supplement omissions in the pinned toolchain's upstream std report."""
    source = ROOT / "sdk/dotnet/runtime-notices"
    inventory = json.loads((source / "inventory.json").read_text())
    version = dict(line.split(": ", 1) for line in toolchain.splitlines() if ": " in line)
    if (version.get("release"), version.get("commit-hash")) != (inventory["rust_release"], inventory["rust_commit"]):
        raise ValueError("runtime notice inventory requires review for this Rust toolchain")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual != set(inventory["files"]) | {"inventory.json"}:
        raise ValueError("runtime notice file set differs")
    for name, digest in inventory["files"].items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or sha256(path) != digest:
            raise ValueError(f"runtime notice hash or path differs: {name}")
    shutil.copytree(source, destination)


def stage(output):
    commit = clean_commit()
    if platform.machine().lower() not in ("x86_64", "amd64") or platform.system() not in ("Linux", "Windows"):
        raise ValueError("native staging requires actual Linux/Windows x64 host")
    rid = "win-x64" if os.name == "nt" else "linux-x64"
    target, library_name = RIDS[rid]
    directory = output / rid
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env["FACTSYOLO_BUILD_ID"] = commit
    target_dir = ROOT / "target/sdk-native"
    args = ["cargo", "build", "-p", "facts-yolo", "--release", "--locked", "--offline", "--no-default-features",
            "--features", "td1", "--target-dir", target_dir]
    build = subprocess.run(list(map(str, args)), cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    (directory / "build.log.txt").write_text(build.stdout)
    if build.returncode:
        raise RuntimeError(build.stdout)
    library = directory / library_name
    shutil.copyfile(target_dir / "release" / library_name, library)
    check_architecture(library, rid)
    if rid == "win-x64":
        exports = capture(["dumpbin", "/nologo", "/exports", library])
        names = sorted(m.group(1) for line in exports.splitlines() if (m := re.fullmatch(r"\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(fy_\w+)(?:\s+=\s+\S+)?\s*", line)))
        dependencies = capture(["dumpbin", "/nologo", "/dependents", library])
    else:
        exports = capture(["nm", "-D", "--defined-only", library])
        names = sorted(parts[-1] for line in exports.splitlines() if len(parts := line.split()) == 3 and parts[-2] == "T" and parts[-1].startswith("fy_"))
        dependencies = capture(["readelf", "-d", library]) + capture(["readelf", "--version-info", library])
    if names != PUBLIC_EXPORTS:
        raise ValueError(f"unexpected native exports: {names}")
    (directory / "exports.txt").write_text(exports)
    (directory / "system-dependencies.txt").write_text(dependencies)
    toolchain = capture(["rustc", "-vV"])
    (directory / "toolchain.txt").write_text(toolchain)
    inventory = []
    for package, roles in selected_inventory(target):
        name, version = package["name"], package["version"]
        source = Path(package["manifest_path"]).parent
        license_files = [p for p in source.iterdir() if p.is_file() and p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))]
        if package.get("license_file"):
            license_files.append(source / package["license_file"])
        if package["source"] is None:
            license_files = [ROOT / "LICENSE", ROOT / "NOTICE"]
        if not license_files:
            raise ValueError(f"no license text for {name} {version}")
        license_destination = directory / "licenses" / f"{name}-{version}"
        license_destination.mkdir(parents=True)
        for file in set(license_files):
            shutil.copyfile(file, license_destination / file.name)
        inventory.append({"name": name, "version": version, "license": package["license"],
                          "source": package["source"] or f"https://github.com/KoalaFacts/FactsYOLO/tree/{commit}",
                          "roles": roles})
    sysroot = Path(capture(["rustc", "--print", "sysroot"]).strip())
    docs = sysroot / "share/doc/rust"
    std_notices = docs / "COPYRIGHT-library.html"
    if not std_notices.is_file():
        raise ValueError("Rust standard-library copyright report missing; install official rust-docs component")
    shutil.copytree(docs / "licenses", directory / "licenses/rust-standard-library/licenses")
    shutil.copyfile(std_notices, directory / "licenses/rust-standard-library/COPYRIGHT-library.html")
    stage_runtime_notices(directory / "licenses/rust-runtime-supplement", toolchain)
    (directory / "cargo-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    packages = [{"SPDXID": f"SPDXRef-Cargo-{i}", "name": p["name"], "versionInfo": p["version"],
                 "downloadLocation": p["source"] if p["source"].startswith("https:") else "NOASSERTION",
                 "filesAnalyzed": False, "licenseDeclared": (p["license"] or "NOASSERTION").replace("MIT/Apache-2.0", "MIT OR Apache-2.0"), "licenseConcluded": "NOASSERTION",
                 "copyrightText": "NOASSERTION", "comment": ", ".join(p["roles"])} for i, p in enumerate(inventory)]
    sbom = {"spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"FactsYOLO-local-{rid}", "documentNamespace": f"https://spdx.org/spdxdocs/factsyolo-{commit}-{rid}-{sha256(library)}",
            "creationInfo": {"creators": ["Tool: FactsYOLO-local-native-assets"], "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
            "comment": "Source/build inventory, not a complete linked-object SBOM. Cargo native-source entries may include dead-stripped code; build-only entries are tools, not shipped dependencies. Rust standard-library source notices and the pinned rust-runtime-supplement/inventory.json are included conservatively; target-specific linked subset is not resolved. External OS libraries remain prerequisites, see system-dependencies.txt. This local test package is not release-cleared.",
            "packages": packages,
            "files": [{"SPDXID": "SPDXRef-NativeFile", "fileName": library_name,
                       "checksums": [{"algorithm": "SHA256", "checksumValue": sha256(library)}],
                       "licenseConcluded": "NOASSERTION", "copyrightText": "NOASSERTION"}],
            "relationships": [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": "SPDXRef-NativeFile"}]}
    (directory / "sbom.spdx.json").write_text(json.dumps(sbom, indent=2) + "\n")
    files = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*") if p.is_file()}
    manifest = {"schema": 1, "rid": rid, "source_commit": commit, "library": library_name,
                "exports": names, "licenses": sorted(k for k in files if k.startswith("licenses/")), "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_bundle(directory, rid, commit)
    print(json.dumps({"rid": rid, "source_commit": commit, "native_sha256": sha256(library), "directory": str(directory)}))


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: native_assets.py <new-asset-root>")
        stage(Path(sys.argv[1]).resolve())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr); sys.exit(1)
