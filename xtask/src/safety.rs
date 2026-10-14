// SPDX-License-Identifier: Apache-2.0
use super::evidence;
use std::process::Command;

pub(super) const MIRI_TOOLCHAIN: &str = "nightly-2026-10-13";

pub(super) fn miri() -> Result<(), String> {
    let directory = evidence::start("miri")?;
    evidence::manifest(&directory)?;
    let command = |arguments: &[&str]| {
        let mut command = Command::new("rustup");
        command
            .current_dir(evidence::root())
            .args(["run", MIRI_TOOLCHAIN, "cargo", "miri"])
            .args(arguments)
            .env_remove("MIRIFLAGS");
        command
    };
    evidence::run(&directory, "version", &mut command(&["--version"]))?;
    evidence::run(&directory, "sysroot", &mut command(&["setup"]))?;
    let result = evidence::run(
        &directory,
        "tests",
        &mut command(&[
            "test",
            "-p",
            "facts-yolo",
            "--locked",
            "--offline",
            "--no-default-features",
            "--features",
            "contract-fixtures",
            "--target-dir",
            "target/miri",
            "--test",
            "abi",
            "--test",
            "images",
            "--test",
            "lifetimes",
            "--test",
            "results_ffi",
            "--test",
            "descriptor_corpus",
            "--",
            "--test-threads=1",
            "--skip",
            "diagnostics_are_per_call",
            "--skip",
            "immutable_view_can_be_shared_between_threads",
            "--skip",
            "immutable_session_queries_can_overlap_while_owner_is_live",
            "--skip",
            "ten_thousand_create_release_cycles_return_counts_to_baseline",
            "--skip",
            "ten_thousand_backed_descriptors_match_u128_reference",
        ]),
    );
    std::fs::write(
        directory.join("outcome.txt"),
        if result.is_ok() { "PASS\n" } else { "FAIL\n" },
    )
    .map_err(|error| error.to_string())?;
    print!("{}", result?);
    println!(
        "Miri descriptor/ownership coverage passed; evidence: {}",
        directory.display()
    );
    Ok(())
}

pub(super) fn layout32() -> Result<(), String> {
    if !cfg!(target_os = "linux") {
        return Err("the i686 GNU layout gate requires Linux multilib".to_owned());
    }
    let directory = evidence::start("i686")?;
    evidence::manifest(&directory)?;
    let rust_object = directory.join("rust-layout.o");
    evidence::run(
        &directory,
        "rust-compile",
        Command::new("rustc")
            .current_dir(evidence::root())
            .args([
                "--edition=2024",
                "--crate-type=lib",
                "--target",
                "i686-unknown-linux-gnu",
                "--emit=obj",
                "tests/layout32/probe.rs",
                "-o",
            ])
            .arg(&rust_object),
    )?;
    let extract = |label: &str, object: &std::path::Path| -> Result<Vec<u8>, String> {
        let bytes = std::fs::read(object).map_err(|error| error.to_string())?;
        if bytes.get(..7) != Some(b"\x7fELF\x01\x01\x01") {
            return Err("layout probe must be an ELF32 little-endian object".to_owned());
        }
        let binary = directory.join(format!("{label}.bin"));
        evidence::run(
            &directory,
            &format!("extract-{label}"),
            Command::new("objcopy")
                .arg("--dump-section")
                .arg(format!(".fy_layout={}", binary.display()))
                .arg(object),
        )?;
        let values = std::fs::read(binary).map_err(|error| error.to_string())?;
        if values.len() != 123 * 4 || values[..4] != 4_u32.to_le_bytes() {
            return Err("expected 123 i686 layout scalars and four-byte pointers".to_owned());
        }
        Ok(values)
    };
    let rust = extract("rust", &rust_object)?;
    for (label, include) in [("current", "include"), ("frozen", "tests/c/frozen")] {
        let object = directory.join(format!("c-layout-{label}.o"));
        let compiler = std::env::var_os("CC_I686").unwrap_or_else(|| "cc".into());
        evidence::run(
            &directory,
            &format!("c-compile-{label}"),
            Command::new(compiler)
                .current_dir(evidence::root())
                .args(["-m32", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I"])
                .arg(include)
                .args(["-c", "tests/layout32/probe.c", "-o"])
                .arg(&object),
        )?;
        let c = extract(label, &object)?;
        if c != rust {
            return Err(format!("i686 {label} C/Rust compiler layouts disagree"));
        }
    }
    std::fs::write(directory.join("outcome.txt"), "PASS\n").map_err(|error| error.to_string())?;
    println!(
        "i686 compiler-only probe: 123 scalars (113 records) agree for current/frozen C and Rust; evidence: {}",
        directory.display()
    );
    Ok(())
}
