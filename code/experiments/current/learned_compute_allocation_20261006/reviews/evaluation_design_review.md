---
written_on: 2026-10-06
written_by: agent:codex
status: development_design_review_not_test_authorization
---

# Minimal decisive evaluation of correctness-only continuation

## Assessment of the draft

The development evaluation draft correctly distinguishes selection, joint cost-aware control, and correctness-only continuation; retains all-negative pools; requires independent initialization; separates test-selected envelopes from deployable baselines; and does not authorize test access. Its remaining main gap is distinguishing the value of continuing under a particular selector from a causal explanation involving accumulated context.

## Proposed bounded panel

Before outcomes, select 64 development theorems and retain a state after attempts one and two for every theorem. Continue collection to the second state even when the natural policy would accept, and label this explicitly as an intervention-state panel, not the natural policy's state distribution. This avoids considering only late states that the policy already chose to reach.

For each of the 128 states, sample eight independent one-additional-attempt branches: 1,024 new proof attempts. Use the same trained checkpoint, precise serialization, sampling precision, token bounds, and original theorem. Require space for a full declared attempt; report any state exclusion and all branch truncations. Freeze the panel and analysis before inspecting continuation outcomes.

Compute accept-now success exactly as `A(s) = sum_i q(i|s, ACCEPT) * V(i)` over the existing pool. For each continuation branch, force acceptance after its extra attempt and compute the analogous exact selection expectation over the expanded pool. This removes unnecessary index-sampling noise. Private checker labels are used for measurement only after recording actions, never for generation or stopping. The primary value difference is the branch-averaged expanded-pool expectation minus `A(s)`.

On 32 prespecified states, four additional independent branches can estimate force-one-continue-then-follow-policy value. A small prespecified two-additional-attempt subset detects delayed benefits. These remain separate estimands; neither may be substituted for the other after seeing results. Start with one complete trained checkpoint, then repeat the core measurement on a second independently trained seed if feasible. Additional theorem count does not replace training-seed replication.

## Interpretation and uncertainty

Bootstrap whole theorems, keeping both retained states and all branch replicates inside each sampled theorem. Report state counts, theorem counts, branch counts, and the continuation Monte Carlo variance. Do not classify individual states as harmful merely because their noisy point estimate is below zero. Show acceptance probability against continuation value within attempt count and incumbent-validity strata. An aggregate association may otherwise be explained by counting or already possessing a valid proof.

For each branch, decompose selected success into probability mass assigned to valid old candidates and probability mass assigned to the new candidate times its validity. Also report new-attempt validity and monotone oracle availability. A valid incumbent can remain available while the chooser assigns it less probability; this produces negative continuation value without loss of generation capability.

Force-CONTINUE versus ACCEPT estimates the value of the intervention under this model and environment. It does **not** identify a causal effect of history content or length. To support that mechanism, compare branches with prior-proof attention masked/removed during generation while preserving the authoritative state and selectable candidate pool, and keep chooser representation fixed. To isolate chooser interference, keep selectable candidates fixed while manipulating explicitly nonselectable context. The unadapted boundary-marker collapse is already sufficient reason to avoid equating generic prefix sensitivity with reasoning-context interference.

## Fair baselines and reporting details

- Freeze stochastic versus argmax selection. Exact expected index correctness should be used consistently for live fixed-count baselines where feasible; do not compare integrated outcomes on one arm to one noisy sample on another without saying so.
- Compare same-checkpoint forced budgets, format-only initialization, and independently trained fixed-count controls. Same-checkpoint controls alone can be weak if joint training made forced continuation unusual.
- Distinguish attempt-matched mixtures from token/work-matched mixtures. Development-frozen mixture weights may not exactly match realized test compute; report the difference. Keep generator forward work, history reprocessing, chooser work, and diagnostics distinct.
- The feature-threshold policy remains a strong baseline on independent frozen pools. Its scores may not be calibrated for newly sampled history-conditioned proofs; that domain change prevents treating it as an automatically matched live comparator.
- Headline selection differences must use the strongest comparator selected on development, while reporting all predeclared baselines and all theorem families. State that uncertainty is conditional on the completed training seeds.

## Scope

This is a bounded recommendation for the existing ten-hour sprint, not a new launch or final-test authorization. If only the one-step panel completes, report that policy-relative result and leave full-policy continuation/context causality unestablished.
