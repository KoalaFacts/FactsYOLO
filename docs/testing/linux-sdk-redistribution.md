# Linux SDK binary audit: 07e1cc2

This record concerns the actual Linux-only local installation probe built from
`07e1cc2da8e79d50208e1698c83dacf4a95d6ef9`. It does not certify a Windows artifact,
a formal dual-RID package, or a public release.

| Audited artifact | SHA-256 |
| --- | --- |
| Original Linux nupkg | `63395634531f5b18d97f836e058699a395b73d2ab05593e93cfc3e8060362eba` |
| libfacts_yolo.so | `1e4a6fbe9a1d46f4bf511c49619bc32fd886057819f72d46e6d21cbf048eee48` |
| FactsYOLO.dll | `50adc01bf1b8c93e42fc420164e7f769b671c3ed0a14486d20691dc83db26750` |

The original archive has 52 entries and exactly these two binaries. Its 11-entry
Cargo source/build inventory and license copies match the locked local sources.
The managed assembly has six .NET 10 System.* assembly references and 11 native
imports, all from facts_yolo. Its ILOnly/AnyCPU metadata does not imply native x86
support. No .NET runtime, OS shared library, training framework, model or pixels
are bundled in this nupkg.

## Actual native requirements

`readelf -dW`, `readelf -VW`, `readelf --dyn-syms -W`, `nm -C`, and `ldd` were
run against the hashed native file. Its DT_NEEDED entries are exactly:

- libgcc_s.so.1: symbol versions through GCC_4.2.0.
- libm.so.6: expf at GLIBC_2.27.
- libc.so.6: strong imports through GLIBC_2.34.
- ld-linux-x86-64.so.2: __tls_get_addr at GLIBC_2.3.

The required GLIBC_2.34 imports are pthread_key_create, pthread_key_delete and
pthread_setspecific. fstat64/stat64 require GLIBC_2.33. Thus **GLIBC 2.34 is a
necessary symbol-version floor for this binary**. This is an ELF measurement,
not a claim that execution on glibc 2.34 or every newer distribution was tested.
It cannot be replaced by Rust's general target minimum. There is no RPATH or
RUNPATH in this library.

Actual load and prior package JIT/NativeAOT inference succeeded on Debian 13
x86_64 with glibc 2.41-12+deb13u4 and libgcc-s1 14.2.0-19. The prior JIT host also
depends on libstdc++, libdl and libpthread and requires the installed .NET runtime
(10.0.12 in the acceptance host). The prior AOT consumer executable itself has
a GLIBC_2.34 floor; it still loads libfacts_yolo.so. Consumer executable/runtime
dependencies are separate from the nupkg's four native dependencies.

The evidence establishes GNU/Linux x64 on this host only. musl, older glibc,
other distributions, minimum CPU variants and a minimum kernel were not executed
in this audit. Absence of an ELF ISA note does not prove a minimum CPU level.

## Static code and notices

ELF FILE symbols identify facts_yolo, fy_tiny_model, sha2, std, panic_unwind,
alloc, core, object, memchr, addr2line, gimli, rustc_demangle, miniz_oxide,
adler2 and compiler_builtins compilation units, plus GCC crtstuff.c startup
glue. This is positive evidence of those units; absence of a crate name cannot
rule out inlined code. Cargo's graph remains a source/build inventory, and
version_check is build-only. No complete object-to-source linker map is claimed.

The original Rust copyright report matches the installed toolchain byte for
byte, but omits addr2line 0.27.1, adler2 2.0.1, object 0.39.1, miniz_oxide 0.9.1
and compiler_builtins 0.1.160, and records memchr 2.8.0 instead of 2.8.3. These
versions were resolved from the official
[pinned Rust library lockfile](https://github.com/rust-lang/rust/blob/b940084d7eb6a299eb4bfeb8e34901bc051e7ac4/library/Cargo.lock)
and matching rust-src archive; the addr2line/miniz_oxide versions also appear in
embedded source paths. Registry archives match the lockfile checksums.

The [supplement inventory](../../sdk/dotnet/runtime-notices/inventory.json)
preserves original license alternatives and full upstream notices, including
compiler_builtins' MIT AND Apache-2.0 WITH LLVM-exception AND (MIT OR Apache-2.0)
declaration and bundled libm notices. It is staged alongside the unchanged
official report. The original nupkg remains evidence of the omission; it is not
silently relabeled as corrected.

The dynamically loaded glibc and libgcc files are OS prerequisites, not bundled
copies. glibc's LGPL terms and GCC's GPL plus Runtime Library Exception apply to
those upstream components. GCC's startup glue is distinct from those external
libraries; the installed GCC copyright record explicitly identifies crtstuff.c
under GPL with the Runtime Library Exception. The Rust notice bundle already
contains the GPLv3 and GCC exception texts. The presence of generic license files
does not by itself assign that license to all SDK code or establish release
clearance. The host linker also selects glibc crti.o/crtn.o; upstream
[crti.S](https://github.com/bminor/glibc/blob/glibc-2.41/sysdeps/x86_64/crti.S)
and [crtn.S](https://github.com/bminor/glibc/blob/glibc-2.41/sysdeps/x86_64/crtn.S)
carry a specific permission to link compiled versions into other programs and
distribute those programs without restrictions arising from those files. This
does not change the terms of separately distributed glibc shared libraries.
Redistributing a separate .NET runtime or consumer executable also
requires reviewing its own notices, outside this two-binary nupkg.

## Recheck commands

Use the exact archived binaries, not files rebuilt with different flags:

```sh
sha256sum FactsYOLO.LocalTest.0.0.0-linux-layout.07e1cc2da8e7.nupkg
unzip -l FactsYOLO.LocalTest.0.0.0-linux-layout.07e1cc2da8e7.nupkg
readelf -dW libfacts_yolo.so
readelf -VW libfacts_yolo.so
readelf --dyn-syms -W libfacts_yolo.so
readelf -Ws libfacts_yolo.so
nm -C --defined-only libfacts_yolo.so
ldd libfacts_yolo.so
getconf GNU_LIBC_VERSION
rustc -vV
```

Full command output, exact shared-library provider hashes, managed metadata,
official source checksums, notice-gap reproduction and preservation checks are
in the separately delivered `sdk-redistribution-audit-20261016T080136Z` evidence.
