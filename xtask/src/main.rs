// SPDX-License-Identifier: Apache-2.0
//! Development checks; never a production CLI or package publisher.
mod dotnet;
mod native;

fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [command] if command == "help" => {
            println!("cargo run -p xtask -- c|exports|dotnet --mode jit|aot [--rid RID]");
            Ok(())
        }
        [command] if command == "c" => native::run(false),
        [command] if command == "exports" => native::run(true),
        [command, rest @ ..] if command == "dotnet" => dotnet::run(rest),
        _ => Err("expected c, exports, dotnet, or help".to_owned()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("conformance failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
