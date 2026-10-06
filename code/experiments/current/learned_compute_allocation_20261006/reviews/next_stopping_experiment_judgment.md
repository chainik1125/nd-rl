---
written_on: 2026-10-06
written_by: agent:codex
status: recommendation_defer_from_current_test_registry
---

# Next controlled learned-stopping experiment

Defer another trained stopping method from this sprint's final-test registry. Freezing the generator/scorer would cleanly isolate stopping optimization from the observed history-generation degradation, but the existing staged confidence thresholds already provide an adaptive-allocation control under independent generation. An additional DEV-selected head now adds a relatively narrow claim and substantial evaluation/provenance/reporting obligations before the primary results are settled. The coordinator agreed not to add it in this sprint.

A later experiment should reserve independent stopping-training pools or use out-of-fold scorer predictions. Training a stopping rule on the same candidate records and validity labels used to fit its scorer exposes it to in-sample score distributions; this is not held-out-label leakage, but it weakens the claim that the stopping rule learned transferable allocation rather than exploiting overly optimistic training confidence. Keep scorer fitting, stopping fitting, development choice, and final evaluation distinct.

The simplest informative design freezes the generator and scorer, fixes argmax proof selection, and learns a small count-dependent confidence threshold schedule from only attempt count and incumbent maximum score. Optimize expected correctness minus `.03 * (attempts − 1)` over sampled training permutations, with terminal stopping enforced at the fixed cap. Monotonic acceptance in incumbent confidence gives an interpretable low-capacity policy; a general MLP is unnecessary initially. Compare with the existing constant confidence threshold, a count-only policy, and development-frozen fixed-budget mixtures matched on required scoring plus generation work.

Use a held-out theorem panel, fixed seed aggregation, and paired uncertainty. The result would test **learned stopping under a frozen generator and selector**, not joint generation adaptation or meta-learning by itself. Only a clear incremental benefit over the constant threshold would justify expanding to richer public-prefix representations.
