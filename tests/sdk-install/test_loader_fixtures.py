#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile every negative loader fixture with the actual host C compiler."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LoaderFixtureCompilation(unittest.TestCase):
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
