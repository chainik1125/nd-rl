---
written_on: 2026-10-06
written_by: agent:codex
---

# Independent final causal TEST audit

Accepted the numerical aggregate, within the frozen estimand. CPU-only independent reconstruction checked eight registered runs, 512 planned eligible states, 3,840 branches, 192 exact shared starting-state comparisons, and 338 aggregate means and confidence intervals. All discrepancies were zero. Every launch completed with the original MAIN registry binding. No model inference, retuning, historical outcome access, or scientific-source edits were performed.

Evidence: [arithmetic replay](audit_causal_test_aggregate.py), [arithmetic results](causal_test_aggregate_independent_audit.json), [association replay](audit_causal_test_associations.py), and [association results](causal_test_associations_independent_audit.json). The authoritative output is `artifacts/learned_compute_allocation_20261006/context/causal_main_test_aggregate_v1/metrics.json`.

## What was independently checked

Raw categorical probabilities and recorded attempt verdicts reconstruct acceptance, old and new correct mass, continuation and oracle values. Future identities, decision-path lengths, repeat coverage, count caps, and forced-count shortfall logic agree. Each condition contains its complete planned theorem × count inventory. Incumbent actions, public transitions, action seeds and controller state match exactly on the common first16 across conditions. The aggregate averages futures, then the two fixed counts and two fixed checkpoints within theorem; paired family-stratified 5,000-draw bootstrap intervals reproduce exactly. By-count, by-checkpoint and paired history-removal aggregates also reproduce. This is an arithmetic audit of recorded verdicts, not a fresh Lean recheck or controller forward pass.

## Findings and interpretation

On the primary 64-theorem panel, one additional attempt changes native categorical selected-proof correctness by −0.00866723, 95% CI [−0.01143360, −0.00603915]. Acceptance correctness is .850832 and continuation correctness .842165. Old correct selection mass falls by .03942879 while new correct mass contributes .03076156. Oracle availability increases from .871094 to .891113; the retained candidate pool does not lose valid proofs.

Both checkpoints have negative effects: seed101 −.00714831 and seed102 −.01018614. The count1 effect is −.00295354 [−.00503815, −.00090383], versus count2 −.01438091 [−.01841011, −.01054121]. All planned states are available; the primary has no forced-count shortfall or terminal hard-limit cases.

On the identical first16 theorem panel, one-extra, resumed-policy minimum-one, minimum-two and minimum-four effects are respectively −.00802625, −.02322326, −.04318328 and −.10230994. These are separate intervention endpoints; this ordered descriptive pattern is not a prespecified paired horizon-difference significance test. Removing generator history while retaining the chooser's candidates improves one-extra new-proof validity by .234375 and continuation effect by .00776389 [.00396496, .01260634]. The resumed-policy minimum-one history contrast is +.02419616 [.01358316, .03596587]. This supports a generator-history contribution under the tested interface, not a universal claim about semantic history interference.

All eight runs' descriptive acceptance-association point estimates reproduce. Both primary checkpoint association interval calculations also reproduce. After conditioning on count and incumbent oracle availability, correlation of acceptance probability with continuation gain is +.10276 for seed101 [−.09159, .30524] and +.25120 for seed102 [.03442, .45793]. The useful direction would be negative. Unconditional and count-only primary intervals include zero. These exploratory associations do not establish useful state-sensitive stopping; their multiple comparisons and noisy future estimates preclude a broad individual-state prediction claim.

## Claim boundary and figure

Supported: under these two frozen zero-cost checkpoints and the prescribed forced-state distribution, additional generation reduces expected correctness with the native categorical selector, despite increasing oracle availability. Selection probability redistribution and degraded contextual generation both matter.

Not supported: that more available proofs inherently harm greedy deployment, that the model learned when it was done, or that this demonstrates meta-learning. Count2 states are experimentally forced and are not weighted by natural policy visitation. The intervals condition on the two trained checkpoints and realized generation protocol, rather than estimating population variability over training seeds. Greedy selection can behave differently, including selecting the first proof throughout.

The saved figure is legible. Its condition names retain original panel sizes (64/32/16), although all plotted estimates use the common16. A reader-facing caption should state that distinction, define the minimum forced future count and resumed-policy endpoint, and identify native categorical selection, equal checkpoint/count weights, and conditional theorem-bootstrap intervals. No frozen plotting source was changed.

Both replay helpers passed Ruff and Python syntax compilation. Formatting is not additional numerical evidence; the recorded arithmetic and association runs above were actually executed.

## Revised presentation review

Reviewed the separate `report_figures/causal_test_v2/continuation.png`, its provenance, and the summary's “Held-out continuation intervention” section. Accepted without required corrections. The revised labels distinguish one-extra selection from minimum-forced resumed policy, and the titles separate the primary64 from the common16 secondary panel. The figure and prose explicitly identify categorical selection, retained candidates, forced count weighting, fixed checkpoint uncertainty, and the absence of horizon-difference tests. Values agree with the previously audited results; no new numerical analysis was run.

The prose correctly limits the history-removal inference to changing generator inputs and generated proofs, and distinguishes it from fixed-pool recognition interference. It also states that harmful average continuation does not demonstrate useful acceptance discrimination or inevitable greedy-selection harm. The small embedded caption is readable at full resolution and is supplemented by the Markdown caption and surrounding prose. The original registered plot and scientific sources remain unchanged.
