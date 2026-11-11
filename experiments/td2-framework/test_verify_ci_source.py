"""Tests for the provenance check, using local commits and no network."""
import pathlib
import subprocess
import tempfile
import unittest

import verify_ci_source as ci


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "config", "user.email", "test@example.invalid")
        git(self.root, "config", "user.name", "CI test")
        (self.root / "a.txt").write_text("base\n")
        git(self.root, "add", "a.txt")
        git(self.root, "commit", "-qm", "base")
        self.base = git(self.root, "rev-parse", "HEAD")
        git(self.root, "checkout", "-qb", "topic")
        (self.root / "b.txt").write_text("feature\n")
        git(self.root, "add", "b.txt")
        git(self.root, "commit", "-qm", "topic")
        self.topic = git(self.root, "rev-parse", "HEAD")
        git(self.root, "checkout", "-q", "main")
        git(self.root, "merge", "-q", "--no-ff", "topic", "-m", "CI synthetic test merge")
        self.merge = git(self.root, "rev-parse", "HEAD")

    def test_shallow_fetch_keeps_raw_parent_identity(self):
        clone = self.root / "shallow"
        clone.mkdir()
        subprocess.run(["git", "clone", "-q", "--depth=1",
                        self.root.as_uri(), str(clone / "repo")], check=True)
        self.assertEqual(ci.inspect(clone / "repo", "pull_request",
                                    self.merge, self.topic, self.base)["checkout_kind"], "pr-test-merge")

    def test_pr_checkout_is_merge_and_records_head(self):
        r = ci.inspect(self.root, "pull_request", self.merge, self.topic, self.base)
        self.assertEqual(r["checkout_kind"], "pr-test-merge")
        self.assertEqual(r["checked_out_sha"], self.merge)
        self.assertEqual(r["source_head_sha"], self.topic)
        self.assertNotEqual(r["checked_out_sha"], r["source_head_sha"])

    def test_rejects_incorrect_pr_source_and_checkout(self):
        with self.assertRaisesRegex(ValueError, "PR head is not second parent"):
            ci.inspect(self.root, "pull_request", self.merge, self.base, self.base)
        with self.assertRaisesRegex(ValueError, "checked out commit"):
            ci.inspect(self.root, "pull_request", self.topic, self.topic, self.base)

    def test_pr_requires_merge_not_single_parent(self):
        git(self.root, "reset", "--hard", "-q", self.topic)
        with self.assertRaisesRegex(ValueError, "two parents"):
            ci.inspect(self.root, "pull_request", self.topic, self.topic, self.base)

    def test_wrong_first_parent_rejected_even_when_head_matches(self):
        # Rebuild the synthetic test merge on an unexpected base commit.
        # The original source-head-only checker would accept this as CI
        # evidence for the expected base, even though it tested another tree.
        git(self.root, "reset", "--hard", "-q", self.base)
        (self.root / "a.txt").write_text("different base\n")
        git(self.root, "commit", "-qam", "unexpected base")
        wrong_base = git(self.root, "rev-parse", "HEAD")
        self.assertNotEqual(wrong_base, self.base)
        git(self.root, "merge", "-q", "--no-ff", "topic", "-m", "wrong-base test merge")
        wrong_merge = git(self.root, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "PR base is not first parent"):
            ci.inspect(self.root, "pull_request", wrong_merge, self.topic, self.base)
        # Identifying the *actual* base still works when explicitly supplied.
        accepted = ci.inspect(self.root, "pull_request", wrong_merge, self.topic, wrong_base)
        self.assertEqual(accepted["base_sha"], wrong_base)

    def test_pr_requires_declared_valid_base_sha(self):
        with self.assertRaisesRegex(ValueError, "invalid SHA"):
            ci.inspect(self.root, "pull_request", self.merge, self.topic)
        with self.assertRaisesRegex(ValueError, "invalid SHA"):
            ci.inspect(self.root, "pull_request", self.merge, self.topic, "invalid")

    def test_push_checkout_matching_source(self):
        r = ci.inspect(self.root, "push", self.merge, self.merge)
        self.assertEqual(r["checkout_kind"], "push-commit")
        with self.assertRaisesRegex(ValueError, "push source"):
            ci.inspect(self.root, "push", self.merge, self.topic)

    def test_rejects_unsupported_events_and_malformed_sha(self):
        with self.assertRaisesRegex(ValueError, "unsupported GitHub event"):
            ci.inspect(self.root, "workflow_dispatch", self.merge, self.merge)
        with self.assertRaisesRegex(ValueError, "invalid SHA"):
            ci.inspect(self.root, "push", "oops", self.merge)


    def test_single_parent_message_cannot_forge_merge_parent(self):
        git(self.root, "reset", "--hard", "-q", self.topic)
        git(self.root, "commit", "--amend", "-qm", "ordinary commit\n\nparent " + self.base)
        single = git(self.root, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "two parents"):
            ci.inspect(self.root, "pull_request", single, self.base, self.base)

    def test_merge_message_parent_line_does_not_change_real_parents(self):
        git(self.root, "commit", "--amend", "-qm", "real merge\n\nparent " + self.base)
        merge = git(self.root, "rev-parse", "HEAD")
        result = ci.inspect(self.root, "pull_request", merge, self.topic, self.base)
        self.assertEqual(result["checkout_kind"], "pr-test-merge")
        self.assertEqual(result["source_head_sha"], self.topic)


if __name__ == "__main__":
    unittest.main()
