# Run brief: shared-code-port — the proof-state tools, ported to `main`'s `code/shared_code`

Run id `shared-code-port`. Working tree `~/work/shared-code-port`: a worktree of nd-rl on branch `dan_shared_proof_state`,
made from `origin/main`. Role: executor. Authorised by Dan 2026-10-13. The result is a pull request to `main` that
Robbie has asked to review, so the bar is code a careful colleague can review in one sitting.

- **This is an engineering run, not an experiment.** No pods, no pre-registration, no bucket uploads, and no
  `STATUS.md` or run files in the repository. Ignore those parts of the executor role brief.
- **Notes:** keep them in `~/runs/shared-code-port/notes.md`.
- **Effort and time:** `--effort max`; hard stop 48 h.
- **Host:** the VPS. At most 4 concurrent processes.

## Goal

Port the model-agnostic tools behind the proof-state results from the fork `dan-pandori/nd-takehome` (branch `dan`;
checkout at `~/nd-takehome`) into nd-rl `main`'s `code/shared_code/`, with tests. They are the parts other
workstreams could reuse.

1. **The proof-state environment** (`state_env.py`). Include:
   - `decompose` (a proof into (state, action) pairs);
   - the step environment;
   - environment-assigned names (the `lean_staten` / SN-v2 mode);
   - rendering of the state and action text.
2. **The Lean-sound step checker** (`step_check.py` and the parts of `lean_prefilter.py` it needs). It has 0 false
   rejects against Lean on 314,541 steps; keep that property and say how it was established.
3. **The proof generator** (`gen.py`, `prune.py`): forward and goal modes, dependency pruning, renaming-class keys,
   per-length flattening, and the K12 filters (box depth ≤ 2, disjoint by renaming class from given pools). Optionally
   the split builder.
4. **Lean term size**, if `code/shared_code/lean_check.py` does not already report it.
5. **Estimators** (a new module), as pure functions with docstrings that say when each is valid:
   - the unbiased pass@k;
   - the k-to-solve interval (Clopper–Pearson);
   - the known-proof estimate of p from teacher-forced log-probs (log-sum-exp over known proofs; a lower bound);
   - a Good–Turing estimate of the unseen proof mass;
   - the zero-success sample size that certifies p < q/K.

   Take the definitions from `capability-defs` and `create-or-elicit` (fork branches `dan_capability-defs`,
   `dan_create-or-elicit`: `cd_*.py`, `coe_*.py`, and their reports). The minimal-length search (`minlen.py`) is
   optional, at your judgement.

**Out of scope:**
- **Model, training and sampling code.** Explain in the README how the step checker plugs into a sampler, e.g. the
  fork's `guided_sample.py`.
- **Orchestration and infrastructure:** pods, hosts, buckets, budgets.

## Requirements

- **Match `code/shared_code`'s style.** Build on `lean_seq.py`'s types and parsers instead of duplicating them: typed,
  passing `make -C code check` (lint, format, type checks) and `make -C code test-all`, as CI runs them
  (`.github/workflows/checks.yaml`; `uv sync --project code --locked --extra dev`).
- **Prove equivalence with the fork, byte for byte, using golden fixtures produced by running the fork's own code.**
  Commit the fixtures, with the script that made them and the fork commit it ran.
  - `decompose` and the environment on ≥ 1,000 K12 records (pairs and rendered text identical);
  - the step checker's verdicts on ≥ 5,000 (state, step) pairs, accepted and rejected;
  - the generator's output for fixed seeds and settings (the first N records identical);
  - the estimators on worked cases reproduced from the two runs' reports.
- **Lean-dependent tests skip cleanly** when no Lean binary is present, following PR #19's tests.
- **The ND verifier.** `main` has none yet; Leon's open PR #22 adds `code/shared_code/nd_verify` with a soundness fix.
  Do not make the shared code depend on an unmerged PR without saying so. Prefer Lean (the shared judge) or no
  verification where the fork only used the ND verifier as a sanity filter, and show the outputs still match the
  fork's. If you do stack on #22, say so prominently.
- **Other open PRs:** Robbie's #20 (stacked on #19) touches `code/shared_code`. Avoid conflicts with it; check before
  opening.
- **Commits** follow `AGENTS.md`: an `Edited-by: agent:claude` trailer, plus `Co-Authored-By: Claude Opus 5.5
  <noreply@anthropic.com>`.

## The pull request

- **Scope.** One coherent draft PR to `main`. If it would exceed ≈ 2,500 changed lines including fixtures, split it
  into two stacked PRs (e.g. environment + checker, then generator + estimators).
- **No reviewers.** Dan adds them after reading.
- **Description:**
  - what each module is and where it came from (fork commit);
  - the equivalence evidence;
  - how to use each piece, with a short example;
  - what is deliberately left out;
  - known limitations;
  - the line "🤖 Generated with [Claude Code](https://claude.com/claude-code)" at the end.
- **Before you finish:** run an independent review subagent that has not seen your reasoning. It re-runs the checks,
  verifies the equivalence claims from the fixtures, and reviews the code for clarity. Fix what it finds and record it
  in the PR description under "Pre-review".

## Done

Push the branch, open the draft PR(s), and write the PR URL(s) and a ≤ 150-word summary in
`~/runs/shared-code-port/notes.md`. Then create `~/runs/shared-code-port/executor.done`.

**Rules:**
- Do not contact anyone, and do not request reviewers.
- Do not modify other people's open PRs or branches.
- Never commit sealed evaluation data (textbook72) or anything containing a token.
