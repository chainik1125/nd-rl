# Run brief: trajectory — log-likelihood and pass@k of the eventual proofs, through pretraining and RL

Run id `trajectory`. Repository `~/work/trajectory` (branch `dan_trajectory` from the fork's `origin/dan`, made by
`newrun`). Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$30**, ceiling **60
pod-hours**. Proposal: nd-rl `docs/proposals/2026-10-01-learning-trajectory.md`, which is the question.

Read first:
- The proposal.
- The `best-state`, `support-followups` (part D: per-step surprisal), `lit-measures` (M1) and `textbook72`
  summaries in `~/nd-rl/experiment-summaries/`.
- On `origin/dan_best-state`: `state_train_best.py` / `best_model.py` (the recipe), the T1 ladder invocation, and
  `bs_analysis`.
- `state_env.py` (`decompose`: an ND proof to state-env actions), `state_sample.py`.

**Code not on `dan`:** the recipe port is on `origin/dan_best-state` (or on `dan` if merged by then), and
`lpool_reread.py` / `pod/sc12/` are on `origin/dan_state-cap12`. textbook72's eval-only copy and manifest rule are
on `origin/dan_textbook72`. Never train or tune on textbook72 or holdout250.

## Models

3 fresh seeds of `best-state`'s **best-cap12** recipe: Stage-1 for 1,200 s on an A40 with the same data and flags,
then the T1 ladder with the same protocol. Two differences:
- Stage-1 saves about 14 checkpoints, log-spaced from step 0 (initialisation) to the last step (for example 0,
  100, 200, 400, 800, 1.6k, 3k, 5k, 8k, 12k, 16k, 20k, the end).
- The ladder keeps every round's checkpoint, r0–r8.

Upload every checkpoint as it is saved; `save_ckpt` does this.

## Theorems

textbook72 (all 72) plus holdout250, both read exactly as in `best-state`.

## Targets per theorem

- **Eventual proof:** the final RL checkpoint's highest-likelihood accepted proof among its seed-0 samples at
  k 256 (theorems it solves).
- **Reference proof:** the textbook72 reference proof where one exists; otherwise the shortest known ND proof
  from the pool record (`minlen_proof`, `ub_proof` or `gen_proof`, in that order). Convert it with `decompose`.
  Report any that fail to convert.

## Scoring, per checkpoint (pretraining and RL alike)

- **Teacher-forced log p** of each target: per step log p(action | state) and its sum, mean and minimum, and the
  index of the worst step with its rule or tactic. Score in nats, with the environment's own name assignment.
- **Samples:** k 256 at T 0.8 (`max_steps` 96), twice: sampling seed 0 and seed 1.
  - Seed 0 fixes group membership at the final checkpoints. **A:** the final Stage-1 checkpoint solves it.
    **B:** the final RL checkpoint does and the final Stage-1 does not. **C:** the final RL checkpoint does not.
  - Seed 1 gives the reported pass@1, @8, @64 and @256 (unbiased estimator), so the endpoints are not inflated by
    selection.
  - If the budget is tight, sample seed 1 at every checkpoint and seed 0 only at the two final checkpoints.
- Score each RL round as soon as its checkpoint is uploaded, on separate reader pods.

## Deliverables

- **Figures (one set per pool, plus the pools combined):**
  1. Log-likelihood of the eventual proof (total, per-step mean, worst step) by group across checkpoints: median
     and IQR, x-axis pretraining (log steps) then RL rounds, with a divider. Then the same for the reference proof.
  2. pass@k (1, 8, 64, 256) by group across checkpoints.
  3. Step-by-step heatmaps for 12 example theorems (4 per group, chosen by a stated rule, not by eye).
- **A table at key checkpoints:** init, mid-pretraining, end of pretraining, RL r1, r4 and r8.
- **Per-checkpoint, per-theorem records** (step log-probs, pass counts): to the bucket and the registry.
- **Sanity check:** the final checkpoints against `best-state`'s best-cap12 per-seed numbers.
- `preregistration/trajectory.md`, written before the first pod. It must include expected group sizes and a
  prediction for B's worst step (climbs mainly in pretraining, or in RL), with a falsifier.
- `run_trajectory.md` (≤ 450 words + figures), `numbers.md` § trajectory, `log.md`, and `STATUS.md` line
  `TRAJECTORY DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/trajectory/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/trajectory/executor.done`.

Post intermediate figures in `STATUS.md`, marked unreviewed: the pretraining half as soon as it exists. Never run
`test_run_once.sh`. At most 4 concurrent processes on the VPS.
