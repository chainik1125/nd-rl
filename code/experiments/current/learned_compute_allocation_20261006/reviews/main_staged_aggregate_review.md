---
written_on: 2026-10-06
written_by: agent:codex
---

# MAIN staged-policy aggregate review

Accepted `analysis/main_test_staged_aggregate_v1`. The [independent reconstruction](audit_main_staged_aggregate.py) and [results](main_staged_aggregate_independent_audit.json) verify all 51,200 normalized rows against their originating per-theorem results and frozen DEV policy mappings, then independently average BCE seeds within theorem and reproduce all paired family-stratified bootstrap intervals across both scorer groups and all four costs. Maximum discrepancy is exactly zero. Registered aggregate signature, inventory, consumed-output hashes and frozen policy hashes match the sealed MAIN registry. No fitting or policy revision occurred.

At the primary evaluation cost 0.03:

| Scorer | Adaptive accuracy | Mean attempts | Logical input-token work | Accuracy gain versus own DEV-work mixture | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| Public features | 0.849485119 | 2.307160714 | 14,012.7423 | +0.029751601 | [+0.017592842, +0.042531803] |
| BCE, three-seed theorem mean | 0.837386905 | 2.387098214 | 8,408.4207 | +0.030918158 | [+0.019544269, +0.043242974] |

The respective fixed mixtures achieve accuracies 0.819733518 and 0.806468746. These are each scorer's own policies, whose mixture weights were frozen from DEV. TEST work is not refitted: adaptive-minus-mixture work is −782.7410 for features and −44.7317 for BCE. Their work intervals are [−2258.7903, +762.5235] and [−858.5007, +837.3632], respectively. Thus positive paired accuracy gains are supported, but an uncertainty-qualified work-reduction claim is not established by these intervals. Use “compared with a DEV-work-matched fixed mixture,” with the TEST residual visible, rather than “exactly equal TEST compute.”

Bootstrap draws retain the paired theorem, average the three fixed BCE checkpoints within each theorem, and preserve the 200/200 family composition. These intervals condition on the saved candidate pools, fitted models and development choices. They do not treat the three seeds as 1,200 independent theorems or establish a population-of-training-seeds interval.

The comparison does not establish a globally optimal fixed frontier pooled across scorers, a comparison against every external selection method, or joint contextual-RL stopping skill. Feature replay is charged, BCE scoring has its own forwards, and the shared plotted unit remains an unpadded logical lower bound rather than measured latency/FLOPs. Historical outcomes were not read. The audit helper passes Ruff.
