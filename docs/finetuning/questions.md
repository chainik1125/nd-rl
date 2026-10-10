# Open questions for Charles (finetuning)

Each entry: the question, the default being followed until answered, and the date asked.

1. **Primary subject.** Dan's best-cap12 family (9.56M, state format, per-round EI and GRPO
   checkpoints for 3 seeds already in his bucket) or the pinned 3.2M ws3 model (whole-proof,
   CPU, RL we ran ourselves but only exp-00c's single seed)? Default: Dan's family is primary
   because every checkpoint and k = 256 dump exists at zero cost; the 3.2M model is the F5
   replica. (2026-10-10)

2. **Fine-tuning on textbook seed-split proofs.** The `textbook-problem` plan kept its sets
   evaluation-only; `pretrain-data` § 5 then split them into seeds (candidate_v0 + Lean 47) and
   test. M1 / M2 / M7 fine-tune on seed-split reference proofs as a measurement, with the models
   discarded and nothing selected on the test split. Default: allowed under the same split; the
   test split and the 72 are never trained on. (2026-10-10)

3. **k for the pass@k area.** Charles said "pass@64 or whatever". Dan's dumps allow K = 256;
   our own sampling gives K = 32. Default: report A_256 and A_64 from Dan's dumps and A_32 from
   ours, never mixed in one table; the headline plot uses A_256 on the test split. (2026-10-10)

4. **Budget and pod.** Phase A (F0) is laptop CPU, $0. F1 / F2 / F4 need ≈ 30–40 A40-hours
   (≈ $20) on one pod we start and stop. Default: no pod until confirmed; the CPU work proceeds. (2026-10-10)

5. **Using Dan's dumps and checkpoints directly.** They are public in his bucket; we only read
   them, re-judge with the pinned verifier, and never write to his bucket or branch. Default:
   use them, cite the file and his commit, tell Dan at the Saturday meeting. (2026-10-10)

6. **Masks as coordinates or subspaces.** Charles described random parameters (coordinates). Li
   et al.'s random dense subspace is the literature's version. Default: coordinates first, at six
   sizes and three draws; the dense subspace only if coordinates at d ≤ 10,000 recover nothing. (2026-10-10)

7. **Fixed algorithm A for EDL.** Dan's ladder fine-tune (lr 3e-4 → 3e-5, warmup 50, 128
   proofs per step) with steps scaled to the prefix length (floor 20, cap 600), chosen on one
   pilot seed, then frozen for every learner and stream. Default as stated; if population
   monotonicity fails (held-out loss rising within a block), halve the lr once and record it. (2026-10-10)

8. **Masked RL arms for H1.** The most informative hypothesis in `literature_hypotheses.md` (RL
   needs orders of magnitude fewer trainable parameters than SFT) needs small GRPO runs with
   parameter masks: ≈ 36 GPU-hours, ≈ $18. The plan's § 1 puts new RL runs out of scope. Default:
   run only the SFT half (H2) until Charles says otherwise. (2026-10-10)

9. **Overtraining ladder variant for H7.** Three EI ladders with the fine-tune mix restricted to
   problems not yet solved, ≈ $5. Default: not run; it is a method, not a measurement. (2026-10-10)
