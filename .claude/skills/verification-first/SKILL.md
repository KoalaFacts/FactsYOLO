---
name: verification-first
description: Use before claiming any change works. Write observable done criteria, exercise the real product behaviour the change targets, record evidence, and get an independent verifier's verdict before merging.
---

# Verification first

A successful build or a green unit run is not proof. A change is done only when the behaviour that motivated it has been observed working in the real product (or the closest real boundary the repo can run), the evidence is written down, and someone who did not write the change agrees.

## 1. Brief before code
In the PR body, before changing code, write:
- Problem: what is wrong or missing, for whom.
- Done criteria: observable pass/fail checks — a command and its expected output, a page and its expected state, a request and its expected response, a measured number and its threshold.
- Out of scope.
Do not prescribe the method; prescribe what must be observably true.

## 2. Observe the real behaviour
Open `.claude/skills/verify-<repo>/SKILL.md` and find the feature-map rows the change touches. Exercise them for real: run the CLI on a real input, drive the page in a headless browser, send the request to the local worker, run the two processes, compare against the reference oracle. Do it before the change (see the gap or the failure) and after (see it pass). Also run the neighbours the feature map marks as at risk.

## 3. Evidence
Add a "Verification evidence" section to the PR body. For each done criterion: the exact command or steps, the relevant output (trimmed, not paraphrased), counts or measurements, and paths of artefacts (logs, screenshots) kept in the session. List what was not verified and why. No evidence means not done.

## 4. Independent verdict
Spawn a fresh verifier agent that did not write the change. Give it only the brief, the diff, the verify skill and repo access. It re-runs the checks itself and returns PASS or FAIL with its own evidence and findings. The author never issues its own verdict. A new commit after a PASS needs a new verdict. Post the verdict as a PR comment.

## 5. Iterate
FAIL means fix, re-verify, and ask for a new verdict. At most three rounds per item; then shrink the item to the part that passes and split out the rest, or mark it blocked with the evidence.

## 6. Make it stick
When verification catches a class of bug, add the test, lint rule or CI check that would have caught it automatically. When a manual step repeats, script it. Keep the verify skill's feature map current.

## 7. Keep moving
A missing tool, a slow environment or an unclear doc is a problem to solve or route around, not a reason to stop. Stop only for what the task's rules reserve for the owner.

## 8. Decision log
One line per judgement call in the PR body: decision — reason — evidence.

## Traps
- A test that only checks that nothing threw is not evidence of behaviour.
- Mocks prove the mock; reach the real boundary when the repo can.
- "It builds", "CI is green" and "the type check passes" are prerequisites, not verification.
- Never weaken a check to get a PASS.
