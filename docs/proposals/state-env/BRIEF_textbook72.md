# Run brief: textbook72 — the proof-state models on the group's 72 textbook problems

Run id `textbook72`. Repository `~/work/textbook72` (branch `dan_textbook72` from the fork's `origin/dan`, made by `newrun`).
Role: executor. `AGENT_POLICY.md` governs; record compute as it asks. Budget **$3**, ceiling **6 pod-hours**.
Authorised by Dan 2026-09-30. Sampling read-outs of existing checkpoints only: no training.

## Why

Robbie's "combined model" post (2026-09-29, https://robbiewmthompson.com/blog/combined-model/) scores the group's 72
textbook problems at pass@256. His naive pipeline solves 13 / 72 and his combined model **31 / 72**. Our textbook
numbers are on a different pool: 760 generated schema instances. Dan wants the apples-to-apples number for our
proof-state models on Robbie's set.

## The problems

In nd-rl, branch `origin/robbie-experiments`, directory
`code/experiments/current/factorial_20260929/textbook72/`: `textbook_dev.jsonl` (58), `textbook_train.jsonl` (14),
and `MANIFEST.json`. Robbie copied them from Dmitry's `origin/dmitry/cap-comparison-textbook-eval-data`.

- Fetch them with `git -C ~/nd-rl fetch origin robbie-experiments`. Check both files' sha256 against the manifest
  before use.
- The prompts are ND prompts (`THM … SEQ … PRF`). Each record has `reference_lines`.
- **The manifest's rule applies:** never train or tune on these problems or on a renaming of them. Report dev58
  and train14 separately and combined.
- Keep a copy in the run under `data/eval_only/textbook72/`, with the manifest and a README that states the rule.
- **Check for contamination first.** By renaming class, compare the 72 against every set the scored models trained
  on: the cap-6 control set, K12's cap-12 set, `rl_targets.jsonl`, and the EI replay sets. Report the overlap. A
  problem that overlaps is reported separately, never silently dropped.

## Models (proof-state only; Dan does not want the whole-proof models)

- **SN-cap12 T1** (RL) and **SN-cap12 frozen** (Stage-1, no RL), seeds s0–s3
  (`hf://buckets/dan-pandori/nd-rl/state-cap12/ckpts/`).
- **SN-v2 cap-6 T1** and **SN-v2 cap-6 frozen**, seeds s0 / s1 (`hf://buckets/dan-pandori/nd-rl/state-env/ckpts/se/`).

## Read-out

- k 256 per problem, T 0.8 (Robbie's protocol), `max_steps` 96 (reference proofs run to 33 lines),
  `max_action` 512. Report the step-cap and action-cap hit rates.
- **Primary: Lean alone** (`lean_judge`). Also report which accepted proofs would fail `nd_verify` (Robbie's reward
  is Lean ∧ `nd_verify`), as a second count labelled as such, only to line up with his number. Nothing is judged by
  it.
- **Report:**
  - solved per checkpoint on dev58, train14 and all 72;
  - solved by `reference_lines` bin (1–5, 6–10, 11–15, 16+);
  - the list of problems solved by no checkpoint;
  - for the 72, which ones Robbie's post says his combined model solves, if the post or `robbie-experiments` names
    them (otherwise say it cannot be matched per problem).
- Show two or three of the longest accepted proofs in Lean.

## Pre-registration (before the first pod)

Expected solves on all 72 for SN-cap12 T1 and frozen, and for SN-v2 T1, with a one-line reason each.

## Deliverables

- `preregistration/textbook72.md`, `run_textbook72.md` (≤ 300 words plus the per-checkpoint table), `numbers.md`
  § textbook72, `log.md`, and `STATUS.md` line `TEXTBOOK72 DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/textbook72/artifacts`.
- Pods deleted; `touch ~/runs/textbook72/executor.done`.

Nothing here is sent to Robbie or anyone else. Dan shares results himself. Never run `test_run_once.sh`. At most
4 concurrent processes on the VPS.
