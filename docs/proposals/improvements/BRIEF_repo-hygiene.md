# Run brief: repo-hygiene — artefacts out of git, and CI (proposal 15, item 6, the cheap half)

Run id `repo-hygiene`. Repository `~/work/repo-hygiene` (branch `dan_repo-hygiene` from the fork's
`origin/dan`, made by `newrun`; starts after `results-registry` is merged). Role: executor.
`AGENT_POLICY.md` governs. Budget **$2**, ceiling **4 pod-hours**. Authorised by Dan 2026-09-28
(proposal 15's queue). Read first: proposal 15 §6.

## Why

4.1 of the 4.4 GB tracked on the fork's `dan` is committed experiment output (`artifacts/`), which is why
run worktrees were 4 GB until `newrun` made them sparse. There is one unit-test file
(`tests/test_lean_only_judge.py`) and no CI, so bugs have been found by reviewers after runs rather than
before them.

## What to do

1. **Artefacts out of the tip of `dan`, losing nothing.** For every tracked file under `artifacts/`,
   confirm a copy exists in the bucket (same path under its run's directory and same size; the registry
   helps), upload any that do not, **then** `git rm -r --cached artifacts/`, add it to `.gitignore`, and
   commit. History keeps everything; the bucket is the store going forward. Report counts: files
   verified, files uploaded first, bytes removed from the tip. Zero files may be lost.
2. **CI on the fork** (GitHub Actions, on pushes to `dan` and on pull requests), fast (< 10 min) and on
   CPU: tokenizer and `lean_seq` render ↔ parse round-trips; the Lean judge on a fixed sample including the
   canonical `Not.elim` example and negatives (install Lean with elan; cache it); the `lean-prefilter`
   soundness test on a small fixed sample; a 50-step CPU training smoke test; a sampler smoke test; the
   registry smoke test; `state-env`'s round-trip and replay tests if merged.
3. **Configs**: every script writes its full resolved configuration next to its outputs (many already
   write `args.json`; make it universal and record it in the registry). No new framework.

Out of scope: moving code into a package (proposal 15 keeps that incremental).

## Acceptance (pre-register)

The CI run is green on `dan` after the merge, and red on a deliberately broken commit on a scratch branch
(show it); a fresh `newrun` worktree without sparse patterns is < 300 MB; the artefact inventory shows 0
lost files.

## Deliverables

Merged into the fork's `dan` after acceptance (the reviewer follows); `CONTRIBUTING.md` (how to run the
tests, where outputs go); `preregistration/repo-hygiene.md`; `run_repo_hygiene.md` (≤ 250 words);
`log.md`; `STATUS.md` line `REPO-HYGIENE DONE <UTC>`; pods deleted; `touch ~/runs/repo-hygiene/executor.done`.
