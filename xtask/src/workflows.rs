// SPDX-License-Identifier: Apache-2.0
//! Repository rule for GitHub Actions workflows: every workflow triggered by
//! `pull_request` cancels the superseded run of the same pull request, so a
//! new push does not leave the previous head's matrix occupying the free
//! runner pool. Runs on `main` are keyed by commit and never cancelled.

/// The concurrency group every pull-request workflow must use: one group per
/// pull request, and one per commit for `main` pushes.
const GROUP: &str = "${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}";

/// Accepted `cancel-in-progress` values (always, or only for pull requests).
const CANCEL: [&str; 2] = ["true", "${{ github.event_name == 'pull_request' }}"];

/// Returns why `text` (a workflow file) breaks the rule, or `None`.
pub(super) fn violation(text: &str) -> Option<&'static str> {
    let lines: Vec<&str> = text.lines().map(str::trim_end).collect();
    // Fails closed: any `pull_request` / `pull_request_target` key or inline
    // trigger list counts, whatever its indentation or trailing comment.
    let triggered_by_pull_request = lines.iter().any(|line| {
        let line = line.split(" #").next().unwrap_or_default().trim();
        line.starts_with("pull_request:")
            || line.starts_with("pull_request_target:")
            || (line.starts_with("on:") && line.contains("pull_request"))
    });
    if !triggered_by_pull_request {
        return None;
    }
    let Some(start) = lines.iter().position(|line| *line == "concurrency:") else {
        return Some("pull_request workflow has no top-level concurrency block");
    };
    let mut group = None;
    let mut cancel = None;
    for line in lines[start + 1..]
        .iter()
        .take_while(|line| line.is_empty() || line.starts_with(' '))
    {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("group:") {
            group = Some(value.trim());
        } else if let Some(value) = line.strip_prefix("cancel-in-progress:") {
            cancel = Some(value.trim());
        }
    }
    if group != Some(GROUP) {
        return Some("concurrency group is not per pull request / per main commit");
    }
    match cancel {
        Some(value) if CANCEL.contains(&value) => None,
        _ => Some("concurrency block does not cancel superseded pull-request runs"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pull_request_workflow_without_concurrency_is_rejected() {
        let text = "name: x\non:\n  pull_request:\n  push:\n    branches: [main]\njobs: {}\n";
        assert!(violation(text).is_some());
    }

    #[test]
    fn pull_request_workflow_with_a_weaker_block_is_rejected() {
        let text = "name: x\non:\n  pull_request:\nconcurrency:\n  group: ${{ github.workflow }}\n  cancel-in-progress: false\njobs: {}\n";
        assert!(violation(text).is_some());
    }

    #[test]
    fn pull_request_workflow_with_the_block_passes_with_lf_and_crlf() {
        for cancel in CANCEL {
            let text = format!(
                "name: x\non:\n  pull_request:\nconcurrency:\n  # note\n  group: {GROUP}\n  cancel-in-progress: {cancel}\njobs: {{}}\n"
            );
            assert_eq!(violation(&text), None);
            assert_eq!(violation(&text.replace('\n', "\r\n")), None);
        }
    }

    #[test]
    fn job_level_or_misnamed_groups_are_rejected() {
        let job_level = format!(
            "on:\n  pull_request:\njobs:\n  a:\n    concurrency:\n      group: {GROUP}\n      cancel-in-progress: true\n"
        );
        assert!(violation(&job_level).is_some());
        let shared = "on:\n  pull_request:\nconcurrency:\n  group: ${{ github.workflow }}\n  cancel-in-progress: true\n";
        assert!(violation(shared).is_some());
    }

    #[test]
    fn every_pull_request_trigger_form_requires_the_block() {
        for on in [
            "on:\n  pull_request:\n",
            "on:\n    pull_request:\n",
            "on:\n  pull_request: # gated\n",
            "on:\n  pull_request_target:\n",
            "on: pull_request\n",
            "on: [pull_request, push]\n",
        ] {
            assert!(violation(&format!("{on}jobs: {{}}\n")).is_some(), "{on}");
        }
    }

    #[test]
    fn non_pull_request_workflows_are_exempt() {
        assert_eq!(violation("on:\n  workflow_call:\njobs: {}\n"), None);
        assert_eq!(violation("on:\n  workflow_dispatch:\njobs: {}\n"), None);
    }

    #[test]
    fn every_repository_pull_request_workflow_cancels_superseded_runs() {
        let directory = crate::evidence::root().join(".github/workflows");
        let mut checked = 0;
        let mut failures = Vec::new();
        for entry in std::fs::read_dir(&directory).expect("workflow directory") {
            let path = entry.expect("workflow entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("workflow text");
            checked += 1;
            if let Some(reason) = violation(&text) {
                failures.push(format!("{}: {reason}", path.display()));
            }
        }
        assert!(
            checked >= 8,
            "expected the repository workflows, found {checked}"
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
