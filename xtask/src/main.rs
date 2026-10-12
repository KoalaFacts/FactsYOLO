// SPDX-License-Identifier: Apache-2.0
//! Development runner scaffold; never treats unimplemented checks as success.
fn main() -> std::process::ExitCode {
    eprintln!("M1 task 1: use cargo test --workspace and cargo build -p facts-yolo");
    if std::env::args().nth(1).as_deref() == Some("help") {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}
