---
written_on: 2026-10-06
written_by: agent:codex
status: development_review
---

# Full feature baseline, staged thresholds, and binary critic

## Independent feature and threshold reproduction

An independent implementation reconstructed all ten public features from **12,800 raw training attempts**. Training feature means/scales match the saved 1,600-theorem model within `1.4e-13` and `2.3e-14`, respectively; training IDs match exactly. The replay deliberately used raw sampled actions rather than the analysis implementation's effective actions.

On the 320-theorem development panel, independent scoring reproduces feature-selector accuracy **0.871875**. The completion table contains 1,813 valid completed attempts, 273 invalid completed attempts, and 474 incomplete attempts. Completion is therefore not a validity oracle.

At threshold 0.85, independent finite-pool expectation reproduces accuracy **0.8412909226**, mean attempts **2.2150669643**, and reward **0.8048389137** for cost 0.03. Expected generated-token and state-prefill proxies also match the saved report.

No private outcome enters the feature extraction function. It uses sampling likelihood/entropy/length statistics and deterministic environment replay. The latter is more informative than ordinary string syntax: it checks grammar, scope, declared-goal equality, and binder compatibility, but not full proof validity. Call this **public grammar/scope completion**. Training its weights against private validity labels is intentional supervised learning; inference never reads those labels.

## Staged-policy assumptions and compute

Exact threshold integration is valid because each candidate score depends only on that candidate, its theorem, and frozen training statistics. The candidate must be fully generated and scored before stopping. The independent generator does not condition future attempts on previously sampled proofs. Do not reuse this finite-pool expectation as if it described a history-conditioned live generator.

On development, the cost-0.03 threshold uses 10.75% more attempts than fixed two, but 33.48% more generated tokens and 33.70% more state-prefix tokens. Thus matching mean attempts does not match even these additive compute proxies. The development fixed-2/fixed-4 upper-envelope mixtures give:

| Matched quantity | Fixed-mixture accuracy | Threshold accuracy |
|---|---:|---:|
| Mean attempts | 0.7984234 | 0.8412909 |
| Generated-token proxy | 0.8087701 | 0.8412909 |
| State-prefill-token proxy | 0.8088708 | 0.8412909 |

These are development calculations for baseline design, not final claims. Freeze mixture weights before test; report residual test mismatch. Expected sparse inclusion costs cannot reconstruct batched sampler wall time or padding. A staged neural critic additionally requires a forward for each generated candidate, including the first; fixed-budget candidate scoring may batch them. Report these costs separately.

## Secondary BCE critic audit

The frozen secondary method uses the same independent original r16 checkpoint, per-candidate `n1 - n2` logit, and unweighted binary cross-entropy across all candidate labels. All-invalid and all-valid pools contribute, unlike the marginal-valid-mass selector. The sigmoid probabilities, raw logits, and pool-softmax probabilities are stored separately. Oversize-candidate fallback depends only on that candidate. The earliest checkpoint with minimum development binary logloss is chosen, as the freeze specifies. I found no load-bearing loss, label-alignment, or selection-criterion defect in the inspected implementation.

This method was added during development after the initial marginal-selector result; it is a legitimate frozen secondary experiment, not a replacement primary. Report pooled primary and marginal control regardless of which method wins. For primary success, use the strongest comparator selected on development and paired theorem-level uncertainty, rather than treating improvement over random as sufficient. Calibration and threshold tuning on development must remain frozen before final outcomes.

## What the controls can establish

A uniform index head plus worsening future generation can make first-attempt stopping locally advantageous even with zero explicit cost. An independently trained fixed-four control can establish whether multi-attempt generation/selection competence is achievable under a matched initialization. If it outperforms the stopping-trained model forced to four attempts, that supports an optimization or training-distribution explanation. It does not prove a unique local optimum.

A staged BCE policy on a frozen independent generator tests recognition and allocation under direct validity supervision. It does not demonstrate joint outcome-only RL or useful history conditioning. Applying the frozen BCE selector to the **same saved live candidate prefixes** would be a useful development diagnostic separating candidate availability from the joint selector's mistakes; any live-domain calibration change must be explicit and development-only.

## Evidence

[Independent feature audit](feature_baseline_independent_audit.json), [reproduction script](audit_feature_baseline.py), `analysis/features1600_main_dev`, `analysis/threshold1600_main_dev`, `selection/method_freeze.md`, `selection/bce_method_freeze.md`, and `selection/train_bce.py`. No final-test outcomes were accessed.
