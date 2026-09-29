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

1. **Bulk files out of the tip of `dan`, losing nothing** (Dan chose this over gzip-in-place or dropping
   per-round files, 2026-09-28; measured then: `artifacts/` 4.16 of the 4.39 GB checkout, 4.28 GB of it
   `.jsonl`; the EI `found_*_<round>.jsonl` files alone 2.69 GB).
   - **What moves:** every tracked `.jsonl`, `.jsonl.gz` and `.pt` under `artifacts/` and `ckpts/`, and
     any other tracked file over 5 MB anywhere (e.g. `data/r2/nec_impi_ore.jsonl`, 50 MB), unless a test
     or CI needs it (then say which, and why it stays). **What stays tracked:** the small files a reader
     wants without downloading anything: `.json` summaries, `.md`, `.png`, `.log`, `.done` markers.
   - **Manifest:** `artifacts/MANIFEST.jsonl`, one row per moved file: repo path, bucket URI (under
     `hf://buckets/dan-pandori/nd-rl/`), bytes, sha256, and the run that produced it where known.
   - **Order:** for each file to move, confirm a copy exists in the bucket (same sha256; the registry
     helps), upload any that do not, write the manifest row, **then** `git rm --cached` it and commit.
     Zero files may be lost. Report counts: files verified, files uploaded first, bytes removed from
     the tip.
   - **Public bucket:** before uploading anything, scan it for credentials (`hf_…` tokens, RunPod and
     Anthropic keys, `~/.config` contents, SSH keys). On any hit, upload nothing, record it in
     `QUESTIONS.md` and continue with the rest.
   - **Fetch helper:** `fetch_artifacts.py <path-or-prefix>...` downloads from the manifest to the same
     repo path and checks sha256. It is idempotent: it skips files already present and matching. The
     177 scripts that read `artifacts/` keep their paths; nothing else about them changes. Show that it
     works: fetch one run's prefix into a fresh worktree and re-run that run's analysis script to the
     same numbers.
   - **Guard:** `.gitignore` patterns for the moved kinds, plus a check (pre-commit and CI) that fails
     if any tracked file is over 5 MB.
   - **Going forward:** `publish_artifacts.py <run-id> <dir>` uploads a run's bulk files to the bucket
     and appends their manifest rows. Draft the change to `AGENT_POLICY.md` (runs publish instead of
     committing bulk files, and the librarian converts any bulk file a run branch still adds, at merge
     time) in `run_repo_hygiene.md`. The orchestrator applies it to the canonical copy in nd-rl.
   - **Links:** relative links in the fork's `.md` files to moved files are re-pointed at the last
     pre-hygiene commit (`blob/<sha>/…`), so they keep working.
   - **Do not rewrite history.** The clone download is already small (≈ 0.46 GB packed). A rewrite would
     force every clone to be redone.
   - **In-flight runs:** a worktree that later merges this `dan` loses the moved files from disk. Before
     merging, check with `runqueue` and `STATUS.md` that no running run reads tracked artefacts from
     `dan` mid-run. If one does, wait for it to finish.
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
lost files (every manifest row's sha256 matches the bucket copy); the fetch helper reproduces one run's
analysis numbers in a fresh worktree; the size guard is red on a scratch commit that adds a 6 MB file.

## Deliverables

Merged into the fork's `dan` after acceptance (the reviewer follows); `CONTRIBUTING.md` (how to run the
tests, where outputs go, how to fetch and publish artefacts, the sparse-checkout alternative); `preregistration/repo-hygiene.md`; `run_repo_hygiene.md` (≤ 250 words);
`log.md`; `STATUS.md` line `REPO-HYGIENE DONE <UTC>`; pods deleted; `touch ~/runs/repo-hygiene/executor.done`.
