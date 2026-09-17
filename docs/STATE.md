---
written_on: 2026-09-08
written_by: agent:codex
---

# Project state

The latest experiment code and artifacts are on simplex1 in `/data/users/dmitry/nd-rl/take-home/`, especially `window_sprint_20260906`, `transfer_sprint_20260907`, and `current_state_20260909`. The local earlier-iterations directory predates these experiments.

The [R1-style reasoning sprint](../experiment-summaries/2026-09-08-r1-cot/README.md) completed its first independent pair audit. No endpoint expanded the observed proof-length window beyond six reference lines; the 73 final-only problems remained unopened in the local record. Next step: reconcile the remaining remote run state on simplex1 and decide whether to audit another frozen pair or close the sprint without opening the final-only split.

A [literature review for learning beyond initial training data](../astra_suggestions/learning-beyond-training-data.md) now recommends five core readings and five additional readings, with experiments and a critique of natural deduction as the setting. Its proposed next step is to specify a fresh structural-transfer benchmark and information/compute accounting before choosing the next RL recipe; this does not alter the current sprint's evaluation restrictions.

[Five proposals](proposals/2026-09-17-proposals-round-2.md) follow the novelty campaign: cast ND into Lean and measure novelty by model scale (Dmitry's idea, with a Qwen3-Coder in-context v0), test the structural-vs-rule-sequence rule of thumb on new patterns, quantify the precursor effect, GRPO vs expert iteration at f = 0, and target pools that require a pattern.

Proposal 6 ([RL technique ladder](proposals/2026-09-17-rl-technique-ladder.md)) returns to the original L − P objective: true proof length from the bounded prover, a fixed transfer pool, equal sample budgets, and a leaderboard of RL techniques (expert-iteration variants, GRPO, preference training, search) with pre-registration per rung.
