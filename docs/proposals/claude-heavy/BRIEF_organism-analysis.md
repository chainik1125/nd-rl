# Run brief: organism-analysis — what predicts RL success, and what RL learns, from data already on disk

Run id `organism-analysis`. Repository `~/work/organism-analysis` (branch `dan_organism-analysis` from the fork's
`origin/dan`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Pod budget **$8** / 16 h, for forward passes on
stored checkpoints only. **No training.** Claude time is expected to be heavy. Authorised by Dan 2026-10-02 (proposal 21).

Read first:
- The `trajectory`, `trajectory-cap6`, `rl-from-ckpt`, `best-state`, `support-followups` (part D) and `lit-measures`
  (M1) summaries.
- `grpo-best`, if it is recorded by the time you reach the analysis. Otherwise say it was not included.

**Code not on `dan`:** `trajectory`'s `tj_*.py` and `pod/tj/` are on `origin/dan_trajectory`; reuse them. The data
(per-checkpoint, per-theorem log-probs and pass counts, accepted-proof dumps) is in
`hf://buckets/dan-pandori/nd-rl/{trajectory,trajectory-cap6,rl-from-ckpt}/artifacts/`.

## Questions

1. **What predicts RL success?**
   - **Features of each (theorem, seed, start),** taken at the start of RL: worst, second-worst and third-worst
     step log p; the number of steps below −4 and below −8 nats; total log p; proof length in ND lines and in Lean
     term size; the rule or tactic of the worst step; and the slope of the worst step over the last pretraining
     checkpoints.
   - **Target:** solved at r8.
   - **Models:** logistic regression and a small gradient-boosted model, cross-validated **across seeds, runs and
     caps**.
   - **Report:** which features matter; whether one worst-step threshold suffices; and whether the same model
     transfers from cap 12 to cap 6 and to the `rl-from-ckpt` starts.
2. **What RL learns, step by step.**
   - A taxonomy of the hard steps (box openers by kind, ∧ projections, …) and how each class's log p moves per
     round.
   - **Transfer:** do theorems whose hard step shares a class gain together, even when RL never sampled a proof of
     some of them? Compare gains with matched controls.
   - A gallery of 12–20 before/after examples rendered in Lean, chosen by a stated rule.
3. **Entropy and diversity.**
   - Per-round policy entropy at the hard steps, and distinct accepted proofs per theorem.
   - Fit performance against entropy (Cui et al., R = −a·e^H + b) for the EI ladders.
   - Does diversity collapse before group C stalls?

   Compute entropies with forward passes on the stored checkpoints. This is the only GPU use.

## Deliverables

- `organism/ANALYSIS.md` (≤ 2,500 words plus figures) and `organism/figures/`, with a static gallery.
  Pre-register the analysis plan in `preregistration/organism-analysis.md` before any pod.
- `run_organism_analysis.md` (≤ 300 words), `numbers.md` § organism-analysis, `log.md`, `STATUS.md` line
  `ORGANISM-ANALYSIS DONE <UTC>`, and `touch ~/runs/organism-analysis/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
