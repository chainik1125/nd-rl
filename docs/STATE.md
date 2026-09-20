---
written_on: 2026-09-08
written_by: agent:codex
---

# Project state

Four completed September 10 experiment records have now been imported from `simplex3` alongside
archived code snapshots. Large raw runs and checkpoints remain remote.

- The [OLMo-32B pilot](../experiment-summaries/2026-09-10-olmo-32b-pilot/README.md) found no
  longer-proof gain beyond ICL on its fixed 24-target panel.
- The [proof-cap comparison](../experiment-summaries/2026-09-10-proof-cap-comparison/README.md)
  completed all six configurations but did not extend verified textbook reach beyond the supplied
  supervision cap.
- The [proof-fragment experiment](../experiment-summaries/2026-09-10-proof-fragment-selection/README.md)
  reproduced its supervised transfer ranking, while all four tested long-RL procedures reduced
  complete-proof coverage from the adapted start.
- The [FLoP adaptation](../experiment-summaries/2026-09-10-flop-adaptation/README.md) showed narrow
  symbolic depth extrapolation without a held-out coverage gain for the token LM.

The earlier [R1-style reasoning sprint](../experiment-summaries/2026-09-08-r1-cot/README.md)
completed its first independent pair audit. No endpoint expanded the observed proof-length window
beyond six reference lines; the 73 final-only problems remained unopened in the local record.

A [literature review for learning beyond initial training data](../astra_suggestions/learning-beyond-training-data.md) now recommends five core readings and five additional readings, with experiments and a critique of natural deduction as the setting. Its proposed next step is to specify a fresh structural-transfer benchmark and information/compute accounting before choosing the next RL recipe; this does not alter the current sprint's evaluation restrictions.
