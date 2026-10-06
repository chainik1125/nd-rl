---
written_on: 2026-10-06
written_by: agent:codex
status: frozen
---

# Secondary binary-validity critic

The coordinator approved this secondary method before final test sampling, after the first five-epoch factorized marginal run reached 0.859375 development correctness. The pooled primary and factorized marginal control remain frozen and must both be reported. This new method tests whether direct candidate supervision improves recognition and calibration relative to marginal-any-valid credit assignment. The public-feature classifier also uses binary validity supervision.

Start independently from the original seed-0 r16 checkpoint for optimizer seeds 0, 1 and 2. Use exactly the same main 1600 training theorem pools, public transition rendering, 8192 per-forward cap, five epochs, batch size two pools, AdamW learning rate 0.00003, weight decay 0.01, gradient clipping at 1 and BF16 model forwards. Each candidate's existing vocabulary logit difference `n1` minus `n2` is a binary validity logit. Optimize unweighted binary cross-entropy on all 12800 labels, including all-valid and all-invalid pools; do not inherit any marginal selector or controller weights.

Select the checkpoint with minimum development binary logloss, breaking exact epoch ties by earliest epoch. Report accuracy, logloss, Brier score and raw sigmoid validity probabilities at that checkpoint. Accuracy is not the checkpoint selection criterion. Store the eight raw logits and sigmoid probabilities in original candidate ID order, together with pool-softmax scores for comparison. Ranking uses raw-logit argmax with seeded uniform exact ties and exact uniform-tie expected correctness. Use primary permutation seed 77106 and report the two previously declared supplementary seeds 77107/77108 without selecting a favorable order.

An oversized candidate receives a neutral validity logit 0 and probability 0.5, determined independently from its own visible length; other candidates remain scored. This avoids future-candidate length information affecting a staged threshold baseline. No text is truncated. Training skips an oversized pool; the observed main factorized inputs have no such cases. Root may fit development-only confidence thresholds for a staged sampling baseline, then freeze them before test. This staged use is separate from independently initialized joint-RL proposal experiments and does not supply their model initialization.
