#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the same local package into a fresh consumer outside the repository."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk/dotnet"))
from native_assets import clean_commit, sha256, verify_assets  # noqa: E402


def main(feed, training, evidence):
    commit = clean_commit()
    rid = "win-x64" if os.name == "nt" else "linux-x64"
    if platform.system() not in ("Linux", "Windows") or platform.machine().lower() not in ("x86_64", "amd64"):
        raise ValueError("installation gate requires actual Linux/Windows x64")
    metadata = json.loads((feed / "package.json").read_text())
    if metadata["source_commit"] != commit or metadata["local_test_only"] is not True:
        raise ValueError("package source differs")
    package = feed / metadata["package"]
    if package.parent != feed or sha256(package) != metadata["sha256"]:
        raise ValueError("package SHA differs")
    evidence.mkdir(parents=True, exist_ok=False)
    detached = Path(tempfile.mkdtemp(prefix="factsyolo-sdk-install-")).resolve()
    if detached.is_relative_to(ROOT):
        raise ValueError("consumer must be outside repository")
    project = detached / "consumer"
    shutil.copytree(ROOT / "tests/sdk-install/Consumer", project,
                    ignore=shutil.ignore_patterns("bin", "obj"))
    shutil.copyfile(ROOT / "sdk/dotnet/global.json", detached / "global.json")
    local_feed = detached / "feed"; local_feed.mkdir()
    shutil.copyfile(package, local_feed / package.name)
    runtime_library = "facts_yolo.dll" if os.name == "nt" else "libfacts_yolo.so"
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts or "\\" in n or ":" in n for n in names):
            raise ValueError("unsafe or duplicate package path")
        # Rebuild complete asset bundles from package entries, then run the same
        # provenance/hash/architecture/license checks used before packing.
        assets = detached / "unpacked-assets"
        for name in names:
            if name.startswith("buildinfo/") and not name.endswith("/"):
                target = assets / name.removeprefix("buildinfo/")
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(archive.read(name))
        for target_rid, library in [("linux-x64", "libfacts_yolo.so"), ("win-x64", "facts_yolo.dll")]:
            (assets / target_rid / library).write_bytes(archive.read(f"runtimes/{target_rid}/native/{library}"))
    manifests = verify_assets(assets, commit)
    native_hash = manifests[rid]["files"][runtime_library]
    config = detached / "NuGet.Config"
    config.write_text(f'''<configuration>
<packageSources><clear/><add key="local" value="{escape(str(local_feed), {'"': '&quot;'})}"/><add key="official" value="https://api.nuget.org/v3/index.json"/></packageSources>
<packageSourceMapping><clear/><packageSource key="local"><package pattern="FactsYOLO.LocalTest"/></packageSource><packageSource key="official"><package pattern="*"/></packageSource></packageSourceMapping>
</configuration>''')
    env = os.environ.copy()
    env["NUGET_PACKAGES"] = str(detached / "packages")
    env["NUGET_HTTP_CACHE_PATH"] = str(detached / "http-cache")
    env["DOTNET_CLI_HOME"] = str(detached / "dotnet-home")
    env.pop("LD_PRELOAD", None)
    unrelated = detached / "unrelated-working-directory"; unrelated.mkdir()
    commands = []

    def run(label, args, cwd=project, expected=0, contains=()):
        started = time.monotonic()
        result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        (evidence / f"{label}.log").write_text(result.stdout)
        commands.append({"label": label, "argv": list(map(str, args)), "cwd": str(cwd),
                         "exit": result.returncode, "expected_exit": expected, "seconds": time.monotonic() - started})
        (evidence / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(f"{label}: exit={result.returncode} expected={expected}", flush=True)
        if result.returncode != expected or any(text not in result.stdout for text in contains):
            print(result.stdout[-12000:], flush=True)
            raise RuntimeError(f"{label} failed; see {evidence / (label + '.log')}")
        return result.stdout

    version = run("sdk-version", ["dotnet", "--version"]).strip()
    if version != "10.0.401":
        raise ValueError(".NET SDK version differs")
    csproj = project / "Consumer.csproj"
    if "ProjectReference" in csproj.read_text() or "<Compile" in csproj.read_text():
        raise ValueError("consumer must reference packages only")
    if Path(env["NUGET_PACKAGES"]).exists() or Path(env["NUGET_HTTP_CACHE_PATH"]).exists():
        raise ValueError("consumer NuGet caches are not fresh")
    models = []
    for backend in ["burn", "candle"]:
        source = training / backend
        report = json.loads((source / "metrics.json").read_text())
        if report.get("passed") is not True or report.get("source_commit") != commit or report.get("steps") != 2000 or report.get("test_seed") != 909:
            raise ValueError("require passed current-source original training evidence")
        cases = detached / backend; cases.mkdir()
        artifact = source / "model.fytd"
        shutil.copyfile(artifact, cases / "model.fytd")
        for i, prediction in enumerate(report["predictions"][:6]):
            if prediction["index"] != i:
                raise ValueError("reference ordering differs")
            value = prediction["prediction"]
            text = "0\n" if value is None else "1\n" + " ".join(map(str, [value["class"], value["score"], *value["corners"]])) + "\n"
            (cases / f"expected-{i}.txt").write_text(text)
            shutil.copyfile(source / f"sample-{i}.rgb", cases / f"sample-{i}.rgb")
        models.append((backend, cases, sha256(artifact)))
    if any(len(list(cases.glob("sample-*.rgb"))) != 6 for _, cases, _ in models):
        raise ValueError("six actual held-out samples required per backend")
    for mode in ["jit", "aot"]:
        publish = detached / f"publish-{mode}"
        args = ["dotnet", "publish", csproj, "-c", "Release", "-r", rid, "-o", publish,
                "--self-contained", "true" if mode == "aot" else "false", f"-p:PublishAot={'true' if mode == 'aot' else 'false'}",
                f"-p:SdkTestVersion={metadata['version']}", "--configfile", config]
        run(f"publish-{mode}", args)
        restored = json.loads((project / "obj/project.assets.json").read_text())
        sdk_key = f"FactsYOLO.LocalTest/{metadata['version']}"
        if restored["libraries"][sdk_key]["type"] != "package" or any(v["type"] == "project" for v in restored["libraries"].values()):
            raise ValueError("restore included a project instead of an installed SDK package")
        if {Path(p).resolve() for p in restored["packageFolders"]} != {Path(env["NUGET_PACKAGES"]).resolve()}:
            raise ValueError("restore used a non-isolated package directory")
        cached = Path(env["NUGET_PACKAGES"]) / "factsyolo.localtest" / metadata["version"] / package.name.lower()
        if sha256(cached) != metadata["sha256"] or sha256(publish / runtime_library) != native_hash:
            raise ValueError("installed package/native hash differs")
        execute = ["dotnet", publish / "Consumer.dll"] if mode == "jit" else [publish / ("Consumer.exe" if os.name == "nt" else "Consumer")]
        for backend, cases, _ in models:
            run(f"run-{mode}-{backend}", [*execute, cases / "model.fytd", cases, commit, native_hash], cwd=unrelated,
                contains=[f"MODE {mode}", "PASS public SDK sample 5", "PASS public SDK: 32"])
        # Separate ownership-source regression: the detached public consumer
        # above remains package-only. No test hooks ship in the SDK assembly.
        ownership = evidence / f"ownership-{mode}"
        run(f"ownership-publish-{mode}", ["dotnet", "publish", ROOT / "tests/sdk-install/Ownership/Ownership.csproj",
            "-c", "Release", "-r", rid, "-o", ownership, "--self-contained", "true" if mode == "aot" else "false",
            f"-p:PublishAot={'true' if mode == 'aot' else 'false'}", "--configfile", config])
        ownership_command = ["dotnet", ownership / "Ownership.dll"] if mode == "jit" else [ownership / ("Ownership.exe" if os.name == "nt" else "Ownership")]
        run(f"ownership-run-{mode}", [*ownership_command, publish / runtime_library, models[0][1] / "model.fytd", mode],
            cwd=unrelated, contains=[f"PASS {mode} admitted SDK lease survives Dispose", "exactly one model release"])
        loader_negatives(mode, publish, runtime_library, unrelated, evidence, run)
    summary = {"passed": True, "source_commit": commit, "source_dirty": False, "rid": rid,
               "package_sha256": metadata["sha256"], "native_sha256": native_hash,
               "models": {backend: digest for backend, _, digest in models}, "heldout_samples_per_model": 6,
               "modes": ["jit", "aot"], "consumer_directory": str(detached), "fresh_nuget_cache": True,
               "platform": platform.platform(), "local_test_only": True}
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def loader_negatives(mode, publish, library_name, cwd, evidence, run):
    native = publish / library_name
    original = native.read_bytes()
    command = ["dotnet", publish / "Consumer.dll", "probe"] if mode == "jit" else [publish / ("Consumer.exe" if os.name == "nt" else "Consumer"), "probe"]
    fixture = ROOT / "tests/sdk-install/loader_fixture.c"
    output = evidence / f"loader-{mode}"; output.mkdir()
    try:
        native.unlink()
        # A valid library in CWD is deliberately not an accepted fallback.
        (cwd / library_name).write_bytes(original)
        run(f"{mode}-reject-missing", command, cwd=cwd, expected=1, contains=["Native package asset is absent", "attempted:", "process=X64"])
        native.write_bytes(b"not a shared library")
        run(f"{mode}-reject-malformed", command, cwd=cwd, expected=1, contains=["Loading '", "failed:"])
        wrong_arch = bytearray(original)
        offset = struct.unpack_from("<I", wrong_arch, 60)[0] + 4 if os.name == "nt" else 18
        struct.pack_into("<H", wrong_arch, offset, 0xaa64 if os.name == "nt" else 183)
        native.write_bytes(wrong_arch)
        run(f"{mode}-reject-architecture", command, cwd=cwd, expected=1, contains=["Loading '", "process=X64"])
        for variant, macro, expected in [("exports", "MISSING_EXPORT", "fy_result_release"),
                                          ("abi", "WRONG_ABI", "Incompatible ABI major 2"),
                                          ("diagnostic", "DIAGNOSTIC_BOUNDARY", "required=1024 truncated=True chars=1023")]:
            built = output / f"{variant}-{library_name}"
            if os.name == "nt":
                args = ["cl", "/nologo", "/LD", "/W4", "/WX", "/TC", "/D" + macro, "/I" + str(ROOT / "include"), fixture,
                        "/Fe:" + str(built), "/Fo:" + str(output / f"{variant}.obj")]
            else:
                args = ["cc", "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror", "-D" + macro,
                        "-I", ROOT / "include", fixture, "-o", built]
            run(f"{mode}-build-{variant}", args, cwd=output)
            shutil.copyfile(built, native)
            run(f"{mode}-reject-{variant}", command, cwd=cwd, expected=1, contains=[expected])
        dependency = output / "dependency.c"
        dependency.write_text(("__declspec(dllexport) " if os.name == "nt" else "") + "int fy_dependency_probe(void) { return 2; }\n")
        dep_lib = output / ("fy_dependency_probe.dll" if os.name == "nt" else "libfy_dependency_probe.so")
        linked = output / ("dependent-" + library_name)
        if os.name == "nt":
            run(f"{mode}-build-dependency", ["cl", "/nologo", "/LD", dependency, "/Fe:" + str(dep_lib), "/Fo:" + str(output / "dependency.obj")], cwd=output)
            args = ["cl", "/nologo", "/LD", "/DWITH_DEPENDENCY", "/I" + str(ROOT / "include"), fixture,
                    "/Fe:" + str(linked), "/Fo:" + str(output / "dependent.obj"), "/link", output / "fy_dependency_probe.lib"]
        else:
            run(f"{mode}-build-dependency", ["cc", "-shared", "-fPIC", dependency, "-o", dep_lib], cwd=output)
            args = ["cc", "-shared", "-fPIC", "-DWITH_DEPENDENCY", "-I", ROOT / "include", fixture,
                    "-L" + str(output), "-lfy_dependency_probe", "-Wl,-rpath,$ORIGIN", "-o", linked]
        run(f"{mode}-build-dependent", args, cwd=output)
        shutil.copyfile(linked, native)
        run(f"{mode}-reject-dependency", command, cwd=cwd, expected=1, contains=["Loading '", "failed:"])
    finally:
        native.write_bytes(original)
        (cwd / library_name).unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            raise ValueError("usage: verify.py <local-feed> <passed-training-root> <new-evidence-directory>")
        main(*(Path(arg).resolve() for arg in sys.argv[1:]))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr); sys.exit(1)
