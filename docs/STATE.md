---
written_on: 2026-09-08
written_by: agent:codex
---

# Project state

The latest experiment code and artifacts are on simplex1 in `/data/users/dmitry/nd-rl/take-home/`, especially `window_sprint_20260906`, `transfer_sprint_20260907`, and `current_state_20260909`. The local earlier-iterations directory predates these experiments.

The [R1-style reasoning sprint](../experiment-summaries/2026-09-08-r1-cot/README.md) completed its first independent pair audit. No endpoint expanded the observed proof-length window beyond six reference lines; the 73 final-only problems remained unopened in the local record. Next step: reconcile the remaining remote run state on simplex1 and decide whether to audit another frozen pair or close the sprint without opening the final-only split.

A [literature review for learning beyond initial training data](../astra_suggestions/learning-beyond-training-data.md) now recommends five core readings and five additional readings, with experiments and a critique of natural deduction as the setting. Its proposed next step is to specify a fresh structural-transfer benchmark and information/compute accounting before choosing the next RL recipe; this does not alter the current sprint's evaluation restrictions.

[Round-3 proposals](proposals/2026-09-18-proposals-round-3.md) (2026-09-18, executor briefs in `proposals/round3/`) attack the rule of thumb where it is weakest: the ignition asymmetry between depth-3 and reductio coincides with whether the pool rewarded pattern-free neighbours (zero-rate reductio arms never trained; zero-rate depth-3 igniters trained on 99–203 optional targets first), so run 1 tests both patterns on required-only, mixed and neighbours-only pools; run 2 measures the 8-line reductio stratum's ignition probability and the derived-ORE base-rate dependence; run 3 fixes n for the base generalisation rate by class (24 seeds each, no RL); run 4 (second host, two executors) asks whether "or nothing" survives 25M and 85M parameters.
