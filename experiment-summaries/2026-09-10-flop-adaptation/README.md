---
written_on: 2026-09-20
written_by: agent:codex
---

# FLoP adaptation to natural deduction

Status: completed remote experiment, imported from `simplex3` for local review.

Question: Which parts of FLoP-style backward-start training transfer to small symbolic policies and
token-generating natural-deduction language models?

## Observations

The symbolic curriculum enabled narrow depth extrapolation, including 129-line proofs on 1/16 and
2/16 later-added challenge statements, but the deepest successes covered only two atomic-projection
statements. On the token-LM experiment, RL did not improve observed held-out coverage and all 21 LM
endpoints solved zero of 384 structured statements, even with four samples.

## Imported evidence

- [Final summary and key depth-challenge figure](summary.md)
- [Archived implementation, tools, and configurations](../../code/experiments/archived/flop_nd_20260910/IMPORT.md)

The summary and figure were copied from
`simplex3:/data/users/dmitry/nd-rl/flop-nd-20260910-1934`. Raw results and checkpoints remain
remote.
