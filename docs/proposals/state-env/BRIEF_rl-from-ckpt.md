# Run brief: rl-from-ckpt — the T1 ladder from earlier pretraining checkpoints of the model organism

Run id `rl-from-ckpt`. Repository `~/work/rl-from-ckpt` (branch `dan_rl-from-ckpt` from the fork's `origin/dan`, made by
`newrun`). Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$90**, ceiling **180
pod-hours**. Authorised by Dan 2026-10-02 with the step-1,600 arm added. Proposal: nd-rl
`docs/proposals/2026-10-02-rl-from-pretraining-checkpoints.md` (proposal 19), which is the question.

Read first:
- The proposal.
- The `trajectory` summary (`~/nd-rl/experiment-summaries/2026-10-01-trajectory-…`) and its write-up and
  `numbers.md` § trajectory on `origin/dan_trajectory`.
- The `best-state` summary.

**Code not on `dan`:** `trajectory`'s scoring and analysis code is only on `origin/dan_trajectory`. Take it with
`git checkout origin/dan_trajectory -- tj_score.py tj_targets.py tj_analysis.py tj_compute.py tj_registry.py pod/tj/`
and merge it with this run. Reuse it; don't rewrite it, so the scores stay comparable.

## Starting checkpoints

`trajectory`'s three cap-12 seeds: Robbie's recipe (ALiBiGPT, 9,560,832 parameters), `lean_staten`, K12, at
`hf://buckets/dan-pandori/nd-rl/trajectory/ckpts/tj/stage1_best12_s{0,1,2}_b1200_step{0,1600,5000,12000,16000}.pt`.
That is 5 starts × 3 seeds = **15 new ladders**.

The end-of-pretraining ladders already exist (`trajectory`'s `la_T1_best12_s{0,1,2}_r{1..8}`, with read-outs) and
are reused as the 100 % arm. Don't re-run them.

## RL

`trajectory`'s T1 ladder exactly: `state_ladder_ei.py`, 8 rounds, k 32, T 0.8, `rl_targets.jsonl`, replay from K12,
and the same flags and batch. Keep every round's checkpoint.
- **Stop rule, step-0 arm only:** if rounds 1 and 2 accept no proof on any target, stop that ladder and report.
  Expert iteration has nothing to train on without a success. Report the round-1 and round-2 accept counts either
  way.
- **Pods:** run the ladders in parallel as A40 / A6000 stock allows (one ladder per pod).

## Read-outs

Same theorems (textbook72 + holdout250, 322), same code and settings as `trajectory`.
- **For each ladder, at r0 (its start), r2, r4 and r8:** pass@k up to 256 with both sampling seeds, plus
  teacher-forced log p (per step, sum, mean, worst) of each theorem's eventual and reference proofs.
- `trajectory` has already scored every starting checkpoint. Re-score r0 for one seed as a reproducibility check,
  and otherwise reuse its records (`hf://buckets/dan-pandori/nd-rl/trajectory/artifacts/`).

## Analysis (pre-register the details)

1. **The threshold test.**
   - For each (theorem, seed, start), take x = the worst-step log p of the reference proof under the starting
     checkpoint (from `trajectory`), and y = solved by that ladder's r8 at k 256 (sample seed 1).
   - Fit P(y | x) per start, with logistic fits and seed as a grouping factor.
   - Elicitation predicts a single curve across starts. Creation predicts early starts solving theorems from well
     below the end-of-pretraining threshold.
   - Report: the curves; the per-start threshold where P = 0.5; and **the number of theorems solved from
     x < −12 nats**, with examples in Lean.
   - Also repeat it with x from the eventual proof, where one exists.
2. **RL as a substitute for pretraining:** r8 performance by start (textbook72, holdout250, per `trajectory`
   group), against the pretraining compute skipped and the RL compute spent (compute rows).
3. **Trajectories:** `trajectory`'s figures (eventual and reference worst step, pass@k by group) per start, on
   shared axes.

## Pre-registration (before the first pod)

- An expectation for each start's r8 textbook72 and holdout250.
- The predicted shape of P(y | x) under each hypothesis, and the falsifier for "RL only finishes what pretraining
  started". For example, ≥ N theorems solved from x < −12 nats across seeds, with N chosen and justified.
- The MDDs, from `trajectory`'s seed spread.

## Deliverables

- `preregistration/rl-from-ckpt.md`, `run_rl_from_ckpt.md` (≤ 450 words + figures), `numbers.md` § rl-from-ckpt,
  `log.md`, and `STATUS.md` line `RL-FROM-CKPT DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/rl-from-ckpt/{ckpts,artifacts}`.
- Post intermediate results in `STATUS.md`, marked unreviewed.
- Pods deleted; `touch ~/runs/rl-from-ckpt/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
