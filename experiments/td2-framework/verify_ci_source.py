"""Verify source provenance for GitHub Actions test-merge or push checkouts.

For pull_request, GITHUB_SHA denotes GitHub's synthetic merge commit, not the
source PR head. Check the raw second parent, even on depth-1 Git fetches.
This does not attest the built Rust binary or establish product accuracy.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

_SHA = re.compile(r"[0-9a-f]{40}")


def git(repo, *args):
    run = subprocess.run(["git", "-C", str(repo), *args], text=True,
                         capture_output=True, check=False)
    if run.returncode:
        raise ValueError(f"git {args[0]} failed: {run.stderr.strip()[:200]}")
    return run.stdout.strip()


def inspect(repo, event, checkout_sha, source_sha):
    if not all(isinstance(sha, str) and _SHA.fullmatch(sha)
               for sha in (checkout_sha, source_sha)):
        raise ValueError("invalid SHA supplied to CI identity check")
    actual = git(repo, "rev-parse", "HEAD")
    if actual != checkout_sha:
        raise ValueError(f"checked out commit {actual} != GITHUB_SHA {checkout_sha}")
    if event == "pull_request":
        # Parent headers end at the first blank line. Commit-message text
        # may also begin with "parent " and must never affect provenance.
        headers = git(repo, "cat-file", "-p", "HEAD").split("\n\n", 1)[0]
        parents = [line.split(" ", 1)[1] for line in
                   headers.splitlines()
                   if line.startswith("parent ")]
        if len(parents) != 2:
            raise ValueError("PR test merge must have exactly two parents")
        if parents[1] != source_sha:
            raise ValueError("PR head is not second parent of CI test merge")
        kind = "pr-test-merge"
    elif event == "push":
        if checkout_sha != source_sha:
            raise ValueError("push source must be the checked out commit")
        kind = "push-commit"
    else:
        raise ValueError(f"unsupported GitHub event: {event}")
    return {"checkout_kind": kind, "checked_out_sha": checkout_sha,
            "source_head_sha": source_sha, "binary_source_attested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(Path.cwd(), args.event, args.checkout, args.source)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"verified {result['checkout_kind']} checkout {args.checkout}, source {args.source}")


if __name__ == "__main__":
    main()
