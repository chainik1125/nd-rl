---
written_on: 2026-09-20
written_by: agent:codex
---

# Proof novelty, difficulty, and self-training selection

Status: completed June 10–11, 2026 experiment series, imported from the earlier
natural-deduction repository.

This sprint asked whether proof interest can be separated from proof difficulty and whether that
distinction helps verifier-guided self-training.

## Main observations

- Across 3,000 theorems, lemma-value was nearly orthogonal to minimum proof length
  (Spearman 0.008), while the proposed genus measure mostly tracked difficulty (0.798 with length).
- On a 135-target, 40-shot evaluation, frontier-model success correlated with proof difficulty but
  not lemma-value. The specialized 3.3M prover reached 36% greedy success and 44% pass@8, though
  the cross-model comparison is confounded by format compliance and unequal sample counts.
- Difficulty-matched high-lemma-value targets produced roughly 60–70% more harvestable verified
  prefixes and a modest held-out transfer advantage in two seeds: +5.4 and +4.6 percentage points.
- A same-theorem authorship control did not support the hypothesis that generator writing style
  caused the earlier external-proof collapse at approximately ten lines. It could not test the
  original 17–30-line regime.
- Flat, zero-nesting training did not break the 12-line ceiling, including at a matched 450k-proof
  data budget. This falsified the proposed nesting-specific explanation of the ceiling.

## Key figures

![Difficulty and lemma-value form distinct empirical axes](figures/fig_w4_axes.png)

![Few-shot natural-deduction capability ladder](figures/fig_ladder.png)

![Novelty-selected hindsight expert iteration across two seeds](figures/fig_w5.png)

## Imported evidence

- [Original full writeup](summary.md)
- [Original numbers ledger](numbers.md)
- [Theory and preregistered predictions](theory.md)
- [Original literature note](literature.md)
- [Original sprint plan](plan.md)
- [Natural Deduction Pack extraction report](ndpack_report.md)
- [Archived implementation and runbooks](../../code/experiments/archived/nd_sprints_20260609_11/IMPORT.md)

Source: `../earlier-iterations/Natural Deduction LLM/nd_sprint`, source commit `e8766d2`. Raw
results, API transcripts, generated pools, and checkpoints were not duplicated.
