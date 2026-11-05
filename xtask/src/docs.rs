// SPDX-License-Identifier: Apache-2.0
//! Markdown consistency gate: relative links and anchors resolve, every
//! project document is indexed from the README, and repository paths named in
//! inline code exist. It reads files only; it never edits documentation.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

/// Directories never scanned: build output and package caches. Hidden
/// directories (VCS data, local worktrees, tool state) are skipped too, except
/// `.github`.
const SKIPPED_DIRECTORIES: [&str; 4] = ["target", "node_modules", "bin", "obj"];
/// Third-party licence texts are notices, not project documentation, so they
/// are link-checked but need no index entry.
const UNINDEXED_DIRECTORY: &str = "runtime-notices";
/// Top-level repository directories whose paths, when named in inline code,
/// must exist.
const PATH_ROOTS: [&str; 9] = [
    ".github/",
    "crates/",
    "docs/",
    "experiments/",
    "include/",
    "sdk/",
    "tests/",
    "xtask/",
    "target/",
];

pub(super) fn run() -> Result<(), String> {
    let root = super::evidence::root();
    let problems = check(root)?;
    if problems.is_empty() {
        println!("docs check passed");
        return Ok(());
    }
    for problem in &problems {
        eprintln!("{problem}");
    }
    Err(format!("{} documentation problem(s)", problems.len()))
}

/// Returns every problem found under `root`, sorted, one line each.
pub(super) fn check(root: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    collect(root, root, &mut files)?;
    files.sort();
    let mut texts = BTreeMap::new();
    for file in &files {
        let text = std::fs::read_to_string(root.join(file))
            .map_err(|error| format!("{}: {error}", shown(file)))?;
        texts.insert(file.clone(), text);
    }
    let mut problems = BTreeSet::new();
    let readme = PathBuf::from("README.md");
    let mut indexed = BTreeSet::new();
    for (file, text) in &texts {
        let prose = strip_fenced_code(text);
        for target in links(&prose) {
            match resolve(root, file, &target, &texts) {
                Ok(Some(path)) if *file == readme => {
                    indexed.insert(path);
                }
                Ok(_) => {}
                Err(problem) => {
                    problems.insert(format!("{}: {problem}", shown(file)));
                }
            }
        }
        for path in code_paths(&prose) {
            if !path_exists(root, path.trim_end_matches('/'))? {
                problems.insert(format!(
                    "{}: named path `{path}` does not exist",
                    shown(file)
                ));
            }
        }
    }
    if !texts.contains_key(&readme) {
        problems.insert("README.md: missing; it is the documentation index".to_owned());
    }
    for file in texts.keys() {
        let notice = file
            .components()
            .any(|part| part.as_os_str() == UNINDEXED_DIRECTORY);
        if *file != readme && !notice && !indexed.contains(file) {
            problems.insert(format!(
                "{}: not linked from the README documentation index",
                shown(file)
            ));
        }
    }
    Ok(problems.into_iter().collect())
}

/// Repository-relative path with `/` separators on every host.
fn shown(path: &Path) -> String {
    let parts: Vec<_> = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect();
    parts.join("/")
}

/// A named path exists from the repository root, or relative to a crate
/// directory (`crates/<name>/`), as crate plans name `tests/abi.rs`.
fn path_exists(root: &Path, path: &str) -> Result<bool, String> {
    if root.join(path).exists() {
        return Ok(true);
    }
    let crates = root.join("crates");
    if !crates.is_dir() {
        return Ok(false);
    }
    for entry in std::fs::read_dir(&crates).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        if entry.path().join(path).exists() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        let name = entry.file_name();
        if kind.is_dir() {
            let hidden = name.to_string_lossy().starts_with('.') && name != ".github";
            if !hidden && !SKIPPED_DIRECTORIES.iter().any(|skip| name == *skip) {
                collect(root, &path, files)?;
            }
        } else if path.extension().is_some_and(|extension| extension == "md") {
            let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
            files.push(relative.to_path_buf());
        }
    }
    Ok(())
}

/// Replaces fenced code blocks with blank lines so their content is ignored.
fn strip_fenced_code(text: &str) -> String {
    let mut fenced = false;
    let mut output = String::with_capacity(text.len());
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        } else if !fenced {
            output.push_str(line);
        }
        output.push('\n');
    }
    output
}

/// Inline `[text](target)` and reference `[label]: target` link targets.
fn links(text: &str) -> Vec<String> {
    let mut targets = Vec::new();
    for line in text.lines() {
        let line = strip_code_spans(line);
        let mut rest = line.as_str();
        while let Some(start) = rest.find("](") {
            rest = &rest[start + 2..];
            let Some(end) = rest.find(')') else { break };
            let target = rest[..end].split_whitespace().next().unwrap_or("");
            targets.push(target.trim_matches(['<', '>']).to_owned());
            rest = &rest[end..];
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') {
            if let Some(index) = trimmed.find("]: ") {
                if let Some(target) = trimmed[index + 3..].split_whitespace().next() {
                    targets.push(target.to_owned());
                }
            }
        }
    }
    targets
}

fn strip_code_spans(line: &str) -> String {
    let mut output = String::with_capacity(line.len());
    let mut inside = false;
    for character in line.chars() {
        if character == '`' {
            inside = !inside;
        } else if !inside {
            output.push(character);
        }
    }
    output
}

/// Resolves one link target. `Ok(Some(path))` names a local Markdown file.
fn resolve(
    root: &Path,
    file: &Path,
    target: &str,
    texts: &BTreeMap<PathBuf, String>,
) -> Result<Option<PathBuf>, String> {
    if target.is_empty() {
        return Err("empty link target".to_owned());
    }
    if target.contains("://") || target.starts_with("mailto:") {
        return Ok(None);
    }
    let (path, anchor) = match target.split_once('#') {
        Some((path, anchor)) => (path, Some(anchor)),
        None => (target, None),
    };
    let resolved = if path.is_empty() {
        file.to_path_buf()
    } else {
        let joined = file.parent().unwrap_or(Path::new("")).join(path);
        normalize(&joined).ok_or_else(|| format!("link `{target}` leaves the repository"))?
    };
    if !root.join(&resolved).exists() {
        return Err(format!("broken link `{target}`"));
    }
    if let Some(anchor) = anchor {
        match texts.get(&resolved) {
            Some(text) if !anchors(text).contains(anchor) => {
                return Err(format!("missing anchor `{target}`"));
            }
            Some(_) => {}
            None => return Err(format!("anchor on non-Markdown link `{target}`")),
        }
    }
    Ok(texts.contains_key(&resolved).then_some(resolved))
}

fn normalize(path: &Path) -> Option<PathBuf> {
    let mut parts: Vec<&std::ffi::OsStr> = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part),
            Component::ParentDir => {
                parts.pop()?;
            }
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(parts.iter().collect())
}

/// GitHub-style heading anchors, with `-1`, `-2` suffixes for duplicates.
fn anchors(text: &str) -> BTreeSet<String> {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut result = BTreeSet::new();
    for line in strip_fenced_code(text).lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('#') {
            continue;
        }
        let heading = trimmed.trim_start_matches('#');
        if !heading.starts_with(' ') {
            continue;
        }
        let slug: String = heading
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
            .map(|c| if c == ' ' { '-' } else { c })
            .collect();
        let count = seen.entry(slug.clone()).or_insert(0);
        result.insert(if *count == 0 {
            slug
        } else {
            format!("{slug}-{count}")
        });
        *count += 1;
    }
    result
}

/// Repository-relative paths named in inline code, such as `docs/a.md`.
fn code_paths(text: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in text.lines() {
        let mut spans = line.split('`');
        spans.next();
        while let Some(span) = spans.next() {
            let candidate = span.trim();
            let plain = !candidate.is_empty()
                && !candidate.contains(char::is_whitespace)
                && !candidate.contains(['*', '<', '>', '{', '}', '$', '?']);
            if plain
                && PATH_ROOTS
                    .iter()
                    .any(|prefix| candidate.starts_with(prefix))
            {
                // Build output is produced on demand; only its naming is prose.
                if !candidate.starts_with("target/") {
                    paths.push(candidate.to_owned());
                }
            }
            spans.next();
        }
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tree(PathBuf);

    impl Tree {
        fn new(name: &str, files: &[(&str, &str)]) -> Self {
            let root =
                std::env::temp_dir().join(format!("factsyolo-docs-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for (path, text) in files {
                let path = root.join(path);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, text).unwrap();
            }
            Self(root)
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn consistent_tree_passes() {
        let tree = Tree::new(
            "ok",
            &[
                (
                    "README.md",
                    "# Index\n\n- [Guide](docs/guide.md#how-it-works)\n",
                ),
                (
                    "docs/guide.md",
                    "# Guide\n\n## How it works\n\nSee `docs/guide.md` and [top](#guide).\n",
                ),
                ("sdk/runtime-notices/LICENSE.md", "licence text\n"),
                (".worktrees/copy/README.md", "[broken](missing.md)\n"),
                ("target/doc/x.md", "[broken](missing.md)\n"),
            ],
        );
        assert_eq!(check(&tree.0).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn rejects_broken_link_anchor_unindexed_doc_and_missing_path() {
        let tree = Tree::new(
            "bad",
            &[
                (
                    "README.md",
                    "# Index\n\n[a](docs/a.md#absent) [gone](docs/gone.md)\n",
                ),
                (
                    "docs/a.md",
                    "# A\n\nRun `xtask/src/absent.rs`; `tests/t.rs` is crate-relative.\n",
                ),
                ("crates/core/tests/t.rs", ""),
                ("docs/orphan.md", "# Orphan\n"),
            ],
        );
        assert_eq!(
            check(&tree.0).unwrap(),
            [
                "README.md: broken link `docs/gone.md`",
                "README.md: missing anchor `docs/a.md#absent`",
                "docs/a.md: named path `xtask/src/absent.rs` does not exist",
                "docs/a.md: not linked from the README documentation index",
                "docs/orphan.md: not linked from the README documentation index",
            ]
        );
    }

    #[test]
    fn ignores_fenced_code_external_links_and_build_output() {
        let tree = Tree::new(
            "code",
            &[(
                "README.md",
                "# I\n\n```sh\n[x](missing.md) `docs/missing.md`\n```\n\n[w](https://example.com/x.md) `target/release/x` `[y](z.md)`\n",
            )],
        );
        assert_eq!(check(&tree.0).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn rejects_links_that_escape_the_repository_and_duplicate_anchor_suffixes_work() {
        let tree = Tree::new(
            "escape",
            &[(
                "README.md",
                "# I\n\n## Step\n\n## Step\n\n[ok](#step-1) [bad](#step-2) [out](../../x.md)\n",
            )],
        );
        assert_eq!(
            check(&tree.0).unwrap(),
            [
                "README.md: link `../../x.md` leaves the repository",
                "README.md: missing anchor `#step-2`",
            ]
        );
    }

    #[test]
    fn repository_documentation_is_consistent() {
        assert_eq!(
            check(super::super::evidence::root()).unwrap(),
            Vec::<String>::new()
        );
    }
}
