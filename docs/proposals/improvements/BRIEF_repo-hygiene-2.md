# Run brief: repo-hygiene-2 — small files back, manifest and fetch helper, size guard

Run id `repo-hygiene-2`. Repository `~/work/repo-hygiene-2` (branch `dan_repo-hygiene-2` from the fork's
`origin/dan`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Budget **$2**, ceiling **4 pod-hours** (no pods are expected).
Authorised by Dan 2026-09-29.

## Why

Dan chose to take bulk files out of git, keep small files, and add a manifest and a fetch helper
(2026-09-28). The `repo-hygiene` brief was amended to say so while that run was already under way. The
amendment (`~/runs/repo-hygiene/BRIEF_AMENDMENT.md`) did not reach the executor before it finished. So
the fork's `dan` (from `3cb0a0f8`) untracks **all** of `artifacts/` (5,992 files, 4.16 GB), including
the small summaries. It has no manifest and no way to fetch a moved file other than
`git show <old sha>:…` or a manual bucket lookup. `repo-hygiene`'s inventories are in
`~/runs/repo-hygiene/` (`inv_*.tsv`, `index_*.tsv`, `hash_*.log`). Reuse them; don't redo the work.

**Update, 2026-09-29 15:30 UTC.**
- `repo-hygiene`'s reviewer **quarantined** it. Its review is `review_repo-hygiene.md` on
  `origin/dan_repo-hygiene`; read it first. The hard-constraint violation: untracking
  `artifacts/TEST_RUN_DONE` switched off `test_run_once.sh`'s refuse-to-run-twice guard in every fresh
  checkout.
- The orchestrator has restored `artifacts/TEST_RUN_DONE` and `artifacts/test_scores.txt` on `dan`
  (cf5924e2; the marker's sha256 is unchanged at `aa1745827596a2cc`). It also narrowed CI's
  `no_tracked_artifacts` step to bulk kinds or files over 5 MB.
- **Never run `test_run_once.sh`**, not even to test the guard. `repo-hygiene` will not be recorded on
  its own; this run clears its quarantine (see Deliverables).

## What to do

1. **Restore the small files to tracking.** Under `artifacts/`, bring back from the last pre-hygiene
   commit **every file that is not a bulk kind and is ≤ 5 MB, whatever its extension**: `.json`, `.md`,
   `.png`, `.log`, `.txt`, `.done`, and markers with no extension. Track them again. Only `.jsonl`,
   `.jsonl.gz`, `.pt`, and any file over 5 MB stay out. Apply the same rule to `ckpts/*.pt` and to
   files over 5 MB in `data/`, unless a test or CI needs them (then say which, and why they stay).
2. **`artifacts/MANIFEST.jsonl`**: one row per untracked file, with repo path, bucket URI (under
   `hf://buckets/dan-pandori/nd-rl/`), bytes, sha256, and the producing run where known. Verify every
   row by sha256 against the bucket copy, not only by size.
3. **`fetch_artifacts.py <path-or-prefix>...`**: downloads from the manifest to the same repo path,
   checks sha256, and is idempotent. The scripts that read `artifacts/` keep their paths. Demonstrate
   it: fetch one run's prefix into a fresh worktree and re-run that run's analysis script to the same
   numbers.
4. **`publish_artifacts.py <run-id> <dir>`** for runs going forward: uploads bulk files and appends
   manifest rows. Draft the `AGENT_POLICY.md` text (runs publish instead of committing bulk files; the
   librarian converts any bulk file a run branch still adds, at merge time) in
   `run_repo_hygiene_2.md`. The orchestrator applies it in nd-rl.
5. **Guard**: `.gitignore` patterns for the moved kinds only (no longer all of `/artifacts/`), plus a
   pre-commit and CI check that fails on any tracked file over 5 MB. Show it red on a scratch commit
   that adds a 6 MB file.
6. **Public bucket**: scan everything `repo-hygiene` uploaded, and anything you upload, for
   credentials: `hf_…` tokens, RunPod and Anthropic keys, `~/.config` contents, SSH keys. On a hit,
   delete that object from the bucket, record it in `QUESTIONS.md`, and continue.
7. **Links**: re-point relative links in the fork's `.md` files to moved files at the last
   pre-hygiene commit (`blob/<sha>/…`).
8. **Housekeeping**: remove the orchestrator's `**AMENDMENT …**` line from the top of `STATUS.md`.
   Update `CONTRIBUTING.md` (fetch, publish, the sparse-checkout alternative). The token now has the
   `workflow` scope (Dan, 2026-09-29): run `ci/install_workflow.sh` and show one green GitHub Actions
   run on `dan`, plus a red one on a scratch branch.
9. **Do not rewrite history.** Before merging into `dan`, check that no running run reads tracked
   artefacts from `dan` mid-run (`runqueue`, `STATUS.md`).

## Acceptance (pre-register)

- 0 lost files: every manifest row's sha256 matches the bucket copy, and every file untracked since
  `3cb0a0f8` is either tracked again or has a manifest row.
- The fetch helper reproduces one run's analysis numbers in a fresh worktree.
- A fresh `newrun` worktree without sparse patterns is < 300 MB.
- The size guard is red on the 6 MB scratch commit.
- CI (`ci/run_ci.sh`) is green on `dan` after the merge.

## Deliverables

Merged into the fork's `dan` after acceptance (the reviewer follows); `preregistration/repo-hygiene-2.md`;
`run_repo_hygiene_2.md` (≤ 250 words); `log.md`; `STATUS.md` line `REPO-HYGIENE-2 DONE <UTC>`;
`touch ~/runs/repo-hygiene-2/executor.done`.

**Clearing `repo-hygiene`'s quarantine.** In `run_repo_hygiene_2.md`, answer each item under "What would
clear the quarantine" in `review_repo-hygiene.md`. This run's reviewer checks them as well. This run's
librarian records both runs: `repo-hygiene` as quarantined, then cleared here.
