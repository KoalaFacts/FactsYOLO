# FactsYOLO local SDK test package

This unregistered `FactsYOLO.LocalTest` package is only an installation acceptance
input. Its temporary version is tied to a source commit. Do not publish it.
It wraps experimental C ABI 1.0; no stable managed API is promised.

The package selects Linux x64 or Windows x64 native assets for a net10.0 process.
Use an explicit `linux-x64` or `win-x64` runtime identifier when publishing.
Managed AnyCPU does not add ARM, x86, macOS, musl or older-Linux support.
Linux dynamic dependencies and GLIBC symbol requirements, or Windows PE import
dependencies, are recorded per target under `buildinfo/`. NativeAOT consumers
still need the selected external native library and its OS dependencies.

```csharp
using FactsYOLO;
using Model model = Model.Open(File.ReadAllBytes(modelPath));
using Session session = model.CreateSession();
using Result result = session.Detect(rgb, 64, 64, 192, PixelFormat.Rgb8);
Detection[] detections = result.Copy();
```

The only current real backend is experimental TD1 CPU/F32 with one CPU thread,
64x64 input and existing synthetic-rectangle models. Shape/format conversion,
inference and output interpretation stay in Rust. No model, pixels or downloader
are included. Supply a separately licensed compatible artifact. Model, session
and result may be disposed independently; input bytes are borrowed synchronously.
Dispose is idempotent; later operations throw ObjectDisposedException. Concurrent
Detect calls retain the native BUSY behavior. FactsYoloException preserves the
numeric primary status and a bounded per-call diagnostic; failures are not retried.

The selected native module stays loaded for process lifetime. Loader diagnostics
show attempted installed paths, process architecture and underlying OS errors.
No library is selected from the current directory, repository or PATH fallback.

Original SDK/native code is Apache-2.0. LICENSE/NOTICE and selected dependency
license texts accompany the package. SPDX files are **source/build inventories**,
not complete linked-object SBOMs or release clearance. Build-only dependencies,
conservatively included Rust standard-library source notices and external OS
requirements are identified separately. No Burn/Candle training framework ships
in the selected native dependency graph. A public binary release still needs a
complete redistribution and supported-OS audit.
