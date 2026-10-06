---
written_on: 2026-10-06
written_by: agent:codex
status: prespecification_review_no_test_outcomes
---

# Held-out causal plan interpretation

The plan fixes a meaningful bounded primary estimand: the change in exact selected-proof correctness under one forced extra attempt followed by acceptance, relative to immediate acceptance from the same state. Keeping resumed-policy and forced-two/four outcomes secondary avoids redefining the primary effect after seeing whether extra search helps. The paired memoryless-generator intervention preserves the selectable proof pool, which separates generator-history effects from deletion of evidence available to the selector. The report must retain the old/new selection-mass decomposition.

The two-checkpoint mean should average **effects**, not model logits or selection distributions. Average branch futures within state, declared states within theorem, and the two checkpoint-specific theorem effects equally. Bootstrap theorem IDs jointly across both checkpoints, stratified by family; do not resample checkpoints independently as if two runs characterized a training-seed population. Show both checkpoint results. Because each checkpoint generates its own incumbent states, their mean is the average effect under two checkpoint-specific forced-state distributions, not an intervention on one common state distribution.

Clarify hard-cap eligibility before registration. If a state after attempt two cannot be collected, or an already terminal state cannot continue, silently dropping it changes the intended count weights and possibly the cross-seed weights. Prespecify how unavailable terms are represented; report planned, reached, eligible, and excluded states by seed and attempt count. If the endpoint is conditional on eligibility, label that conditioning and preserve separate count-specific results. Do not turn an unavailable continuation into an observed zero effect without explicitly defining such a policy intervention. Forced-two/four branches that stop at a cap likewise require realized extra counts and shortfall rates.

The existing forced-state warning is essential: the primary mean weights states after attempts one and two by the measurement design, including states a near-always-stop-first policy would almost never visit naturally. It therefore cannot be read as the expected gain from modifying the deployed policy. Report first-count and second-count effects beside the pooled primary value. Naturally reached or reach-weighted descriptive summaries may supplement the result only if separately declared; they must not replace the frozen primary after outcomes are known.

A negative average effect can arise from selection dilution despite increased oracle availability. A positive effect does not alone show that the stopping head recognizes where continuation is useful. Acceptance/effect associations and residual variation are relevant secondary diagnostics, with the near-constant-policy suppression already specified. Neither outcome establishes a general meta-learning mechanism. The current plan acknowledges these distinctions and does not authorize test generation.

Evidence: read `context/causal_test_plan.md` and the previously audited development protocol. No final-test outcomes or protected benchmarks were opened; no model or generation run was launched.
