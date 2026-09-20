---
written_on: 2026-09-20
written_by: agent:codex
---

# Proof-fragment selection and backward-reset curriculum

Status: completed remote experiment, imported from `simplex3` for local review.

Question: Can a supervised fragment-transfer score select useful practice families, and can
verifier RL with backward resets turn that selection into better complete-proof coverage?

## Observations

The frozen family ranking predicted the supervised intervention it measured (Spearman 0.958), but
the selected mixture had no resolved coverage advantage over its matched uniform control. All four
long-RL conditions reduced coverage from the adapted starting model: endpoints solved 269–279 of
768 targets versus 351/768 initially. A separately fitted reset diagnostic increased mixed-reward
groups, but it was not tested as a training curriculum.

## Imported evidence

- [Detailed final summary and four key figures](summary.md)
- [Archived implementation, tools, and configurations](../../code/experiments/archived/proof_fragments_20260910/IMPORT.md)

The summary and figures were copied from
`simplex3:/data/users/dmitry/nd-rl/proof-fragments-20260910-1800`. Raw results and checkpoints
remain remote.
