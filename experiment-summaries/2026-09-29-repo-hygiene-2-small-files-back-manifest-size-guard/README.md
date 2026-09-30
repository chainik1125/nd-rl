---
written_on: 2026-09-30
written_by: agent:claude
---

# repo-hygiene-2: small files back in git, bulk files behind a sha256 manifest, and a 5 MB guard (clears repo-hygiene's quarantine)

Status: completed and reviewed 2026-09-30. No hard-constraint violation, nothing quarantined. Every
quantitative claim reproduces from the reviewer's own code, with three rewordings (applied below). This
run also **clears the quarantine of the earlier run `repo-hygiene`** (2026-09-29), which is recorded here
first. Infrastructure only: no model is trained or sampled and no proof is counted, so no checker, seed
count or noise floor applies. The only model mention is CI's smoke models (114 k parameters, `lean_seq`,
from scratch on the 150-proof fixture), which only show that code runs.

Question: can the fork's `dan` hold every small file in git again, move only bulk kinds (`.jsonl`,
`.jsonl.gz`, `.pt`, or > 5 MB) out, and make each moved file recoverable by path from a sha256-checked
manifest, with nothing lost, nothing secret in the public bucket, and a guard that stops a > 5 MB file
from being committed?

## The earlier run: repo-hygiene (quarantined 2026-09-29)

`repo-hygiene` (proposal 15 §6, brief
[`BRIEF_repo-hygiene.md`](../../docs/proposals/improvements/BRIEF_repo-hygiene.md); executor 01:30–02:02
UTC, no pods, $0) moved all of `artifacts/` out of the tip of the fork's `dan` in `e844a8ea` (5,992 files,
4,159,751,936 bytes, verified in the bucket first) and added CI. Its reviewer
**quarantined** it ([`review_repo-hygiene.md`](https://github.com/dan-pandori/nd-takehome/blob/de22e988/review_repo-hygiene.md),
commit `de22e988`, 02:01 UTC) and stopped at the hard-constraint checks:

- `artifacts/TEST_RUN_DONE` was no longer tracked. `test_run_once.sh` line 4 refuses a second test-set
  run only if that file exists on disk, so in any fresh checkout of `dan` the one-test-run rule was
  unenforced. `artifacts/test_scores.txt` was removed the same way. The content itself was never lost
  (bucket copy, sha256 `aa174582…` unchanged).
- The brief amendment (small files stay in git, sha256 manifest, fetch/publish helpers, 5 MB guard,
  credential scan) was delivered by stopping the executor, and the resumed session never read it.
- The merge into `dan` happened before any review.

None of `repo-hygiene`'s own numbers was reviewed, so none is recorded here as a result. `cf5924e2`
(Dan's session) restored `TEST_RUN_DONE` and `test_scores.txt` before this run started.

## Configuration

- **Brief:** [`BRIEF_repo-hygiene-2.md`](../../docs/proposals/improvements/BRIEF_repo-hygiene-2.md).
  Budget $2 / 4 pod-hours registered as a ceiling; no pods were used.
- **Fork:** worktree `~/work/repo-hygiene-2`, branch `dan_repo-hygiene-2` of `dan-pandori/nd-takehome`,
  from `cf5924e2`. The executor fast-forwarded the fork's `dan` `cf5924e2..0c34c534` at 15:56 UTC
  (no history rewrite); executor tip `6b7b2f5e` (= fork `dan` at review time).
- **Pre-registration:** fork `preregistration/repo-hygiene-2.md`, commit `c07a4822` (15:36:27 UTC),
  before the first work commit `8fe52a5c` (15:43:59).
- **Rule:** bulk = path ends `.jsonl`, `.jsonl.gz`, `.pt`, or size > 5,242,880 bytes. Everything else
  that `repo-hygiene` untracked is tracked again from the original blobs. The same rule applied to the
  tracked `ckpts/` and `data/` moves `ckpts/stage1_abs.pt`, `ckpts/final.pt` and 7 `data/` files out.
- **Tooling added on the fork's `dan`:** `artifacts/MANIFEST.jsonl` (path, uri, bytes, sha256, run per
  bulk file); `fetch_artifacts.py <path prefix>` (stdlib only, public HTTPS, sha256-checked, idempotent);
  `publish_artifacts.py <run-id> <dir>…` (scans for credentials, uploads, appends manifest rows);
  `ci/check_sizes.sh` in a pre-commit hook (`ci/hooks/`, `ci/install_hooks.sh`) and in `ci/run_ci.sh`,
  replacing the old "no tracked artifacts" step; `tests/test_manifest.py`; `CONTRIBUTING.md` (fetch,
  publish, sparse checkout); the GitHub Actions workflow at `.github/workflows/ci.yml`.
- **Run write-up:** `run_repo_hygiene_2.md`, `numbers.md` § repo-hygiene-2, `log.md` on the fork branch.
- **Review:** [`review_repo-hygiene-2.md`](https://github.com/dan-pandori/nd-takehome/blob/5cc0338d/review_repo-hygiene-2.md),
  §Recount in [`799bf43b`](https://github.com/dan-pandori/nd-takehome/commit/799bf43b), §Comparison and
  §Verdict in [`5cc0338d`](https://github.com/dan-pandori/nd-takehome/commit/5cc0338d). The first
  reviewer session was killed at ≈ 16:05 UTC on 2026-09-29 when another run's 160 parallel downloads
  exhausted the VPS's memory; it resumed at 23:59 UTC and redid everything except a partial bucket scan.
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/repo-hygiene-2/` (`git_tip/` holds the 9 moved `ckpts/`
  and `data/` files; `artifacts/repo-hygiene-2/` holds the verification and scan tables). `repo-hygiene`'s
  uploads are under `hf://buckets/dan-pandori/nd-rl/repo-hygiene/`.
- **Compute:** no pods (`podbudget`: 0.00 h, $0.00). Hashing, bucket downloads and the credential scan
  ran on the VPS.

## Observations

All values below were reproduced by the reviewer's own scripts.

- **0 files lost.** Of the 6,006 files untracked by either run (5,992 `artifacts/` at `6123570e`, 5
  `artifacts/lpool/` at `470d39e0`, 9 `ckpts/`/`data/` at `cf5924e2`), 3,953 are tracked again with the
  same blob and 2,053 are manifest rows whose sha256 and bytes equal the last git blob. The reviewer's
  union over all 18 commits `6123570e..HEAD` gives the same result.
- **Manifest = bucket.** 2,054 rows (2,053 + the run's own published bucket listing), 4,252,105,864
  bytes; **2,054 / 2,054** match in sha256 and bytes on full downloads by both executor and reviewer; 0
  objects missing.
- **Credential scan: 0 hits.** The executor scanned the bucket as of 15:56 UTC (6,023 objects under
  `repo-hygiene/` and `repo-hygiene-2/`). The 22 files uploaded at DONE (3,064,505 bytes) were scanned by
  the review, not by the run: 0 hits over all 6,045 objects (4,332,285,282 bytes). The reviewer's scan
  also matched the literal values of the VPS's own 7 tokens; planted fakes and a real token split across
  a chunk boundary were found.
- **Fetch reproduces a run's numbers.** The pre-registered demo (`lp_analysis.py` after fetching
  `artifacts/lp/`) could not test this, because its C1 input `artifacts/lp/corpus/*.dump.jsonl*` was never
  in git or the manifest; the run disclosed this as a deviation. The substitute: `nf_analysis.py` differs
  from the tracked `artifacts/nf/summary.json` before `fetch_artifacts.py artifacts/nf/` (365 files) and
  is **byte-identical** after it. A second fetch downloads nothing; a one-byte corruption is re-fetched.
- **Fresh non-sparse worktree: 147.4 / 148.1 MiB (154.6 / 155.3 MB)** at `8fe52a5c` / `0c34c534`;
  the reviewer measured 160 MiB on disk at `6b7b2f5e` (155,261,304 tracked bytes, 4,538 files), inside the
  pre-registered 120–170 MB and below 300 MB.
- **Size guard.** Red on a 6 MB scratch commit in the pre-commit hook, `check_sizes.sh` and GitHub Actions
  (run 36592832438); green at exactly 5,242,880 bytes; red on a 1 kB `artifacts/x.jsonl` (bulk kind).
  `publish_artifacts.py` refuses a directory holding a planted `hf_…` token and uploads nothing.
- **CI green on `dan`.** GitHub Actions runs 36594025125 (`0c34c534`) and 36594323823 (`6b7b2f5e`)
  succeeded. `ci/run_ci.sh` was not re-run locally by the reviewer (no torch on the VPS).
- **Found on the way:** `cf5924e2`'s message says it narrowed CI's guard, but `ci/run_ci.sh` there still
  failed on any tracked `artifacts/` file. `check_sizes.sh` replaces that step.
- **Quarantine cleared.** `TEST_RUN_DONE` (blob `1d5cf064`, sha256 `aa174582…`), `test_scores.txt` and
  `test_run_once.sh` are unchanged and tracked; the CI guard follows "> 5 MB or bulk kind"; the rest of
  the amendment is done; this review is the re-review. `nd_verify` (tree `9437bb72`) is unchanged, no
  training or evaluation code changed, and `test_run_once.sh` was not run.
- **Rewordings applied here** (the fork files keep the executor's text): "147.4 / 148.1 MB" is MiB; the
  scan claim covers the bucket as of 15:56, with the 22 DONE files scanned by the review; the log line
  "16:13" for the scan should be ≤ 15:56 (cosmetic). The write-up is 348 words against the brief's 250.

## Limitations and next step

- **Take-home checkpoints.** A plain clone no longer contains `ckpts/stage1_abs.pt`, `ckpts/final.pt` or
  `data/train.jsonl.gz`, so `README.md`'s reproduction commands fail until the user runs
  `python3 fetch_artifacts.py ckpts/ data/`. Dan answered on 2026-09-30 (fork `683d1513`): keep them out of
  git. The reviewer recommends a one-line pointer to the manifest in the fork's `README.md`; not done yet.
- **The pre-commit hook** only runs where `core.hooksPath=ci/hooks` is set (set in this clone's shared
  config; a fresh clone must run `ci/install_hooks.sh`). CI is the backstop.
- **`merge=union` on `MANIFEST.jsonl`** lets concurrent runs append cleanly but can leave two rows for a
  re-published path; `tests/test_manifest.py` asserts unique paths, so CI catches it.
- Four relative links in `artifacts/fu/{,phase2_}followup_draft.md` were already broken before this run.
- **Policy draft** for `AGENT_POLICY.md` (Artifacts), from `run_repo_hygiene_2.md`, for Dan's session:
  runs publish bulk files with `publish_artifacts.py` instead of committing them, commit the small files
  and `artifacts/MANIFEST.jsonl`, and the librarian publishes any bulk file a run branch still adds at
  merge time.
- **Next measurement** (reviewer): repeat the fetch-to-reproduce check on a run whose analysis reads
  `ckpts/` or `data/` manifest rows, to exercise the non-`artifacts/` rows end to end.
