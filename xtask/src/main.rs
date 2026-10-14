// SPDX-License-Identifier: Apache-2.0
//! Development checks; never a production CLI or package publisher.
mod dotnet;
mod evidence;
mod gates;
mod native;
mod resources;
mod safety;

fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [command] if command == "help" => {
            println!(
                "cargo run -p xtask -- c|exports|resources|manifest|verify|miri|layout32|dotnet --mode jit|aot [--rid RID]"
            );
            Ok(())
        }
        [command] if command == "c" => native::run(false),
        [command] if command == "exports" => native::run(true),
        [command] if command == "resources" => resources::run(),
        [command] if command == "miri" => safety::miri(),
        [command] if command == "layout32" => safety::layout32(),
        [command] if command == "manifest" => evidence::start("manifest").and_then(|directory| {
            evidence::manifest(&directory)?;
            println!("manifest: {}", directory.display());
            Ok(())
        }),
        [command] if command == "verify" => gates::verify(false),
        [command] if command == "verify-worker" => gates::verify(true),
        [command, flag] if command == "verify" && flag == "--preflight" => gates::preflight(),
        [command, rest @ ..] if command == "dotnet" => dotnet::run(rest),
        _ => Err("expected c, exports, dotnet, resources, manifest, verify, or help".to_owned()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("conformance failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
