---
written_on: 2026-09-08
written_by: agent:codex
---

# Project state

The latest experiment code and artifacts are on simplex1 in `/data/users/dmitry/nd-rl/take-home/`, especially `window_sprint_20260906`, `transfer_sprint_20260907`, and `current_state_20260909`. The local earlier-iterations directory predates these experiments.

The [R1-style reasoning sprint](../experiment-summaries/2026-09-08-r1-cot/README.md) completed its first independent pair audit. No endpoint expanded the observed proof-length window beyond six reference lines; the 73 final-only problems remained unopened in the local record. Next step: reconcile the remaining remote run state on simplex1 and decide whether to audit another frozen pair or close the sprint without opening the final-only split.

A [literature review for learning beyond initial training data](../astra_suggestions/learning-beyond-training-data.md) now recommends five core readings and five additional readings, with experiments and a critique of natural deduction as the setting. Its proposed next step is to specify a fresh structural-transfer benchmark and information/compute accounting before choosing the next RL recipe; this does not alter the current sprint's evaluation restrictions.

An [automated research orchestration proposal](proposals/2026-09-16-automated-research-orchestration.md) (2026-09-16, design only) describes four agent roles (proposer, executor, reviewer, librarian), the gates between them, the hard constraints the reviewer enforces, and the infrastructure Dan would need to approve; its worked example is the ignition study on the `nd-takehome` fork.
