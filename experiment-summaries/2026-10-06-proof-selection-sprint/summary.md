---
written_on: 2026-10-06
written_by: agent:claude
status: running
---

# Can the proof model pick its own winning attempt and know when to stop?

<!-- Executive summary: 2–5 findings, each with one self-explanatory figure. Filled in during the writing block. -->

## Executive summary

(pending: results due 09:30 UTC)

## What problem this addresses

Dan's r16 proof-state model solves far more textbook theorems at pass@256 than at pass@1, but every one of those
extra solves is found by Lean, not by the model. Test-time scaling only pays off if something decides which attempt
to keep and when to stop sampling. This sprint asks whether the model itself can do both: recognise a valid proof
among its own attempts (selection) and sample only as long as needed (stopping), with the generator frozen.

## Map of what was done

- Working folder: `code/experiments/current/proof_selection_20261006/` (kickoff, schema, briefs, logs, code, reviews).
- Heavy artifacts: `~/runs/proof-selection/` on this machine (pools with every attempt and separate Lean labels,
  scores, verifier checkpoints, stopping curves).
- Branch `dmitry/selection-sprint` (worktree `/data/users/dmitry/ndrl-selection-sprint`).

## Findings

(pending)

## Method details

(pending)

## Limitations and what to do next

(pending)
