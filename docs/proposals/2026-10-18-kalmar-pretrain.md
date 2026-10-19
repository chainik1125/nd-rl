---
written_on: 2026-10-18
written_by: agent:claude
status: approved by Dan 2026-10-18, $50 (queued as `kalmar-pretrain` after `proof-strategies`)
---

# Proposal 29: pretrain on Kalmár-construction proofs

**Question (Dan):** if we pretrain on the proofs `code/tools/tt_prover.py` writes (Kalmár's completeness construction)
for the same theorems as K12, does the model learn the construction, and does it generalise?

**Why it is interesting.** The construction is a uniform algorithm that proves every valid sequent over our four atoms.
Every step is a simple function of the visible proof state. So the experiment separates two things our earlier runs
could not:
- **learning a procedure:** high per-step agreement with the algorithm;
- **executing it reliably over long horizons:** full-proof success.

It also tests generalisation to shapes K12 never shows, where an algorithm-follower should be indifferent and a
pattern-matcher should not.

**The main obstacle is length.** Kalmár proofs grow about 2× per atom (≈ 10 / 28 / 70 / 200 constructor steps for
1–4 atoms), while K12 proofs are 2–12 lines. The data is ~10–20× more tokens, and the step and length caps must rise.
The run measures this first and pre-registers a data-matched or a compute-matched training length.

| step | what |
|---|---|
| data | K12-Kalmár: Kalmár ND proofs for K12's 155,000 theorems, Lean-checked, replayed in the proof-state environment |
| arms | KAL (Stage 1 on K12-Kalmár, 3 seeds) against the existing K12 bases at matched caps |
| learned? | teacher-forced per-step agreement by step type, depth and position; free-generation adherence |
| solve rates | holdout250, long pools, Charles's release, textbook72 (scoring only); plain and guided; by atom count and length |
| generalises? | more premises, larger or deeper formulas, patterns K12 lacks, against the control |
| optional | a short EI ladder from KAL: does RL keep the algorithm or find shortcuts? |

**Budget** $50 / 90 pod-hours. **Run** `kalmar-pretrain` on agent2, `--effort max`, 96 h hard stop, queued after
`proof-strategies`. **Brief:** [state-env/BRIEF_kalmar-pretrain.md](state-env/BRIEF_kalmar-pretrain.md).
