---
written_on: 2026-10-06
written_by: agent:codex
---

# Actual eight-run DEV causal aggregate

Accepted the numerical aggregation of `context/causal_main_dev_aggregate_v1`. The independent [audit script](audit_causal_dev_aggregate.py) and [results](causal_dev_aggregate_independent_audit.json) reconstruct the saved per-state and branch arithmetic without importing the production aggregator. No TEST data were accessed and no generation was run.

All 3,840 branch records satisfy the old/new mass and continuation-minus-accept decompositions. Their averages match per-state records exactly. All 130 checked metric means and 95% intervals match exactly across the four complete condition panels, their common-first16 subsets, and the two history-removal contrasts. Resampling preserves family composition and keeps each theorem's two count states and two fitted checkpoints together. All 192 repeated common-panel starting-state comparisons match exactly, including controller probabilities, incumbent identities, raw/effective/transition content, name bases and generation seeds. All saved starting states are eligible.

The primary one-extra result on 64 DEV theorems is −0.01578364 selected-correctness probability, 95% interval [−0.01989683, −0.01183291]. It decomposes into −0.04197329 old-correct mass and +0.02618965 new-correct mass. Oracle availability rises from 0.9140625 to 0.9184570. This supports a mechanism of categorical selection dilution under these fitted models; it is not evidence that additional candidates reduce oracle availability or inevitably harm greedy deployment. The intervals condition on two fitted checkpoints and do not estimate variability over a population of training seeds.

The common-first16 figure is visually readable and its points/intervals are numerically correct:

| Intervention | Correctness change | 95% interval |
|---|---:|---:|
| One extra, then select | −0.012370 | [−0.020166, −0.004846] |
| Force one, then resume policy | −0.022871 | [−0.037147, −0.010175] |
| Force two, then resume policy | −0.042830 | [−0.061946, −0.023558] |
| Force four, then resume policy | −0.092510 | [−0.125526, −0.058357] |

For the final report, replace internal row names such as `one extra64` and `policy forced1 32` with the descriptive labels above: every point shown uses the same 16 theorems, regardless of the original run's full panel size. Supply a caption stating DEV, native categorical terminal selection, equal weighting of counts 1/2 and checkpoints 101/102, and family-stratified paired theorem-bootstrap intervals. The primary 64-theorem estimate is separate from this figure. These horizon comparisons remain descriptive; no paired horizon-difference significance test was registered or performed. No locked plotting/analysis source was changed in this review.
