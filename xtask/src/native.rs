// SPDX-License-Identifier: Apache-2.0
//! Builds isolated libraries and executes real native consumers.
mod verify;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use verify::{compare_layout, exported_names, validate_symbols};

fn checked(command: &mut Command) -> Result<(), String> {
    eprintln!("+ {command:?}");
    let status = command
        .status()
        .map_err(|e| format!("cannot execute {command:?}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{command:?} exited {status}"))
    }
}

fn output(command: &mut Command) -> Result<Output, String> {
    eprintln!("+ {command:?}");
    command
        .output()
        .map_err(|e| format!("cannot execute {command:?}: {e}"))
}

fn captured(command: &mut Command) -> Result<String, String> {
    let result = output(command)?;
    if !result.status.success() {
        return Err(format!(
            "{command:?} exited {}\n{}\n{}",
            result.status,
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    String::from_utf8(result.stdout).map_err(|e| e.to_string())
}

fn cargo(root: &Path) -> Command {
    let executable = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(executable);
    command.current_dir(root);
    command
}

fn ensure_file(path: &Path) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("required file is absent: {}", path.display()))
    }
}

fn library_file(directory: &Path) -> PathBuf {
    directory.join(if cfg!(windows) {
        "facts_yolo.dll"
    } else {
        "libfacts_yolo.so"
    })
}

fn compile(
    root: &Path,
    directory: &Path,
    source: &str,
    library: Option<(&Path, bool, bool)>,
    frozen: bool,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let source = root.join(source);
    ensure_file(&source)?;
    let name = source.file_stem().ok_or("source has no filename")?;
    let mut executable = directory.join(name);
    if cfg!(windows) {
        executable.set_extension("exe");
    }
    let compiler = std::env::var_os("CC").unwrap_or_else(|| {
        if cfg!(windows) {
            "cl".into()
        } else {
            "cc".into()
        }
    });
    let mut command = Command::new(compiler);
    command.current_dir(directory);
    let includes = [
        root.join(if frozen { "tests/c/frozen" } else { "include" }),
        root.join("tests/fixtures"),
    ];
    if cfg!(windows) {
        command.args(["/nologo", "/std:c11", "/W4", "/WX", "/TC"]);
        for include in &includes {
            command.arg(format!("/I{}", include.display()));
        }
        if let Some((_, fixture, static_link)) = library {
            if fixture {
                command.arg("/DFY_TEST_FIXTURES");
            }
            if static_link {
                return Err("Windows static C is not an M1 task 6 target".to_owned());
            }
        }
        command
            .arg(&source)
            .arg(format!("/Fe:{}", executable.display()))
            .arg(format!(
                "/Fo:{}",
                directory.join(name).with_extension("obj").display()
            ));
        command.arg("/link").arg("kernel32.lib");
        if let Some((release, _, _)) = library {
            let import = release.join("facts_yolo.dll.lib");
            ensure_file(&import)?;
            command.arg(import);
            let dll = library_file(release);
            ensure_file(&dll)?;
            std::fs::copy(dll, directory.join("facts_yolo.dll")).map_err(|e| e.to_string())?;
        }
    } else {
        command.args(["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread"]);
        for include in &includes {
            command.arg("-I").arg(include);
        }
        if let Some((_, fixture, static_link)) = library {
            if fixture {
                command.arg("-DFY_TEST_FIXTURES");
            }
            if static_link {
                command.arg("-DFY_STATIC");
            }
        }
        command.arg(&source).arg("-o").arg(&executable);
        if let Some((release, _, static_link)) = library {
            let link = if static_link {
                release.join("libfacts_yolo.a")
            } else {
                library_file(release)
            };
            ensure_file(&link)?;
            command.arg(link);
            command
                .args(["-Xlinker", "-rpath", "-Xlinker"])
                .arg(release);
        }
        command.args(["-ldl", "-lm", "-lrt", "-lutil"]);
    }
    checked(&mut command)?;
    ensure_file(&executable)?;
    Ok(executable)
}

fn run_program(executable: &Path, release: Option<&Path>) -> Result<String, String> {
    let mut command = Command::new(executable);
    command.current_dir(executable.parent().ok_or("executable has no directory")?);
    if cfg!(target_os = "linux") {
        command.env_remove("LD_PRELOAD");
        if let Some(release) = release {
            command.env("LD_LIBRARY_PATH", release);
        }
    }
    captured(&mut command)
}

fn check_exports(root: &Path, release: &Path, fixture: bool) -> Result<(), String> {
    let file = library_file(release);
    ensure_file(&file)?;
    let mut command = Command::new(if cfg!(windows) { "dumpbin" } else { "nm" });
    if cfg!(windows) {
        command.args(["/nologo", "/exports"]);
    } else {
        command.args(["-D", "--defined-only"]);
    }
    command.arg(&file);
    let names = exported_names(&captured(&mut command)?);
    let mut expected = std::fs::read_to_string(root.join("tests/c/public-symbols.txt"))
        .map_err(|e| e.to_string())?;
    if !expected.ends_with('\n') {
        expected.push('\n');
    }
    if fixture {
        expected.push_str(
            "fy_test_model_create\nfy_test_session_resume\nfy_test_session_wait_running\n",
        );
    }
    validate_symbols(&expected, &names)?;
    std::fs::write(release.join("verified-exports.txt"), &names).map_err(|e| e.to_string())?;
    println!(
        "{}: {} exact exports verified",
        file.display(),
        names.lines().count()
    );
    Ok(())
}

pub(super) fn run(exports_only: bool) -> Result<(), String> {
    if !cfg!(any(target_os = "linux", windows)) || std::env::consts::ARCH != "x86_64" {
        return Err("this runner currently requires Linux x64 or Windows x64".to_owned());
    }
    // Cargo supplies an absolute path without depending on the caller's cwd.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("workspace root is absent")?;
    if !root.is_absolute() {
        return Err("workspace root must be absolute".to_owned());
    }
    let normal = root.join("target/normal");
    let fixture = root.join("target/contract-fixtures");
    for (target, fixtures) in [(&normal, false), (&fixture, true)] {
        let mut build = cargo(root);
        build
            .args([
                "build",
                "-p",
                "facts-yolo",
                "--locked",
                "--offline",
                "--release",
                "--no-default-features",
                "--target-dir",
            ])
            .arg(target);
        if fixtures {
            build.args(["--features", "contract-fixtures"]);
        }
        checked(&mut build)?;
        check_exports(root, &target.join("release"), fixtures)?;
    }
    if exports_only {
        return Ok(());
    }
    let evidence = root.join("target/c-conformance");
    std::fs::create_dir_all(&evidence).map_err(|e| e.to_string())?;
    let mut layout_command = cargo(root);
    layout_command
        .args([
            "run",
            "-p",
            "facts-yolo",
            "--example",
            "abi_layout",
            "--locked",
            "--offline",
            "--release",
            "--quiet",
            "--target-dir",
        ])
        .arg(&normal);
    let rust_layout = captured(&mut layout_command)?;
    if rust_layout.lines().count() != 113 {
        return Err("expected 10 types, 75 fields, 27 constants and pointer width".to_owned());
    }
    std::fs::write(evidence.join("rust-layout.txt"), &rust_layout).map_err(|e| e.to_string())?;
    for frozen in [false, true] {
        let header = if frozen { "frozen" } else { "current" };
        let directory = evidence.join(format!("{header}-layout"));
        let executable = compile(root, &directory, "tests/c/abi.c", None, frozen)?;
        let c_layout = run_program(&executable, None)?;
        compare_layout(&rust_layout, &c_layout)?;
        std::fs::write(directory.join("c-layout.txt"), &c_layout).map_err(|e| e.to_string())?;
        println!("{header} header: all 113 compiler-produced layout records agree");
        for (target, fixtures) in [(&normal, false), (&fixture, true)] {
            let release = target.join("release");
            for static_link in [false, true] {
                if cfg!(windows) && static_link {
                    continue;
                }
                let mode = if fixtures { "fixture" } else { "normal" };
                let link = if static_link { "static" } else { "dynamic" };
                let directory = evidence.join(format!("{header}-{mode}-{link}"));
                let library = Some((release.as_path(), fixtures, static_link));
                let executable = compile(root, &directory, "tests/c/contract.c", library, frozen)?;
                print!("{}", run_program(&executable, Some(&release))?);
                if fixtures {
                    let executable =
                        compile(root, &directory, "tests/c/threads.c", library, frozen)?;
                    print!("{}", run_program(&executable, Some(&release))?);
                }
            }
        }
    }
    let probe = compile(
        root,
        &evidence.join("loader"),
        "tests/c/load_path.c",
        None,
        false,
    )?;
    for (target, expected) in [(&normal, "normal"), (&fixture, "fixture")] {
        let path = library_file(&target.join("release"));
        print!(
            "{}",
            captured(Command::new(&probe).arg(&path).arg(expected))?
        );
    }
    let missing = evidence
        .join("does-not-exist")
        .join(library_file(Path::new("")));
    if missing.exists() {
        return Err("negative-control path unexpectedly exists".to_owned());
    }
    let result = output(Command::new(&probe).arg(&missing).arg("fixture"))?;
    if result.status.code() != Some(2)
        || String::from_utf8_lossy(&result.stdout).trim() != "LOAD_FAILED"
    {
        return Err(
            "missing absolute library did not fail through the expected loader path".to_owned(),
        );
    }
    println!("missing-library negative control rejected; C conformance passed");
    Ok(())
}
