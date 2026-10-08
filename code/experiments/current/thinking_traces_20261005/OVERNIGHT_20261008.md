---
written_on: 2026-10-08
written_by: agent:claude
---

# Overnight log, 2026-10-08 (Leon asleep; decisions delegated)

Leon (about 1:15 am PDT): "you have my permission to start experiments and make decisions on your own … when I get
back tell me what you chose in a tldr and a summary of what worked and what didn't."

Times are PDT. Each entry is a decision or a result, with its reason.

## Decisions

- **1:20 am.** Gap analysis done ([gap_analysis/README.md](gap_analysis/README.md)). One unit-test theorem was a
  renaming of a textbook72 problem, so I replaced it. textbook72 must never be trained or tuned on, even in a test
  string.
- **1:25 am.** Started building the proof-first hard 7–12-line generator (`hard712/`) from gap-analysis targets
  T1–T8.
  - The gates: the shortest proof must be 7–12 lines and within 1 line of the constructed proof; the targeted
    structure must be necessary; Lean; exclusion against every set.
  - The sealed test split is an unfiltered random sample, drawn before any model-based filtering.

## Results

- **21–30-line sealed set (2,767, k 8), interim.** Everything is at the floor.
  - Best is graded thinking GRPO it78: 0.8% pass@1, 3.9% solved within 8.
  - Others:
    - effort-levels seed 1: 0.5%;
    - token cost: 0.3%;
    - no-think GRPO: 0.2%;
    - effort seed 0: 0.2%;
    - EI no-think: 0.2%.
