---
written_on: 2026-10-06
written_by: agent:codex
status: prespecified_panel_and_estimands_not_test_authorization
---

# Held-out continuation-value measurements

Before any main intervention or final model-test outcomes, fix a held-out version of the [development causal protocol](causal_method_freeze.md). The final evaluation registry must still name and hash every model, runner, analysis and invocation before generation. This note alone does not authorize test access.

Use the zero-cost seeds101 and102 selected by the original DEV32 checkpoint rule. The public panel is `artifacts/learned_compute_allocation_20261006/context/causal_test_panel_v1/public.jsonl`, SHA256 `56b8e2cc5f79202a423fe90f107c33c9fb9afbd96dd3d58f9f182b75c4919e55`. Choose the first32 rows of each family in the already-frozen main test metadata, alternating forward then ORE2. Its manifest lists every ID and source-data hash. Only theorem ID, original prompt, split, family and structural key enter this panel; no reference proof or generated outcome is included.

Repeat the complete bounded development measurement matrix: one-extra-and-accept on64 theorems with eight futures at states after attempts1/2; one-forced-extra-then-policy on32 theorems with four futures; forced-two and forced-four on16 theorems with four futures. Pair generator-history removal on the first16 theorems for the first two measurements only. Use a distinct prespecified generation seed81006 for all conditions. Hard caps and all saved candidates remain unchanged.

The primary causal endpoint is mean change in exact categorical selected-proof correctness from one extra attempt followed by acceptance, averaged over futures, then the two states, then the two checkpoints within each theorem, then theorems. Whole-policy, forced-two/four, selection-mass decomposition, and history removal are prespecified secondary mechanisms. Report each checkpoint separately and their equal-weight theorem-level mean; average correctness effects, never model logits. Uncertainty conditions on these two trained checkpoints and their own generated state distributions. Bootstrap theorem IDs jointly across both checkpoints within each family,5000 draws with seed81006. Use acceptance/continuation-value diagnostics, including within-count and within-count-plus-incumbent-validity residual variation. Suppress association estimates when stopping probabilities are effectively constant, as declared in the analysis implementation.

Report planned, reached, hard-limited and eligible state counts for each checkpoint and attempt count. If any planned n1/n2 state is unreachable or ineligible for continuation, mark the fixed-weight primary aggregate unavailable; do not reweight the remaining states or checkpoints silently. Eligible-state per-checkpoint and count-specific summaries remain descriptive and must name their changed population. Compare the four continuation conditions additionally on their common first16 theorems, retaining paired checkpoint/state identities.

States after attempt two are forced even when the learned policy would already have stopped. They therefore describe a forced-state distribution, not solely naturally visited states. An interval for an average continuation effect does not label individual states as truly harmful. Retaining every previous candidate makes declining oracle availability an implementation error; selected accuracy can still fall because the model gives probability to an invalid candidate. These effects must remain distinct.

No condition, seed, family, or unfavorable outcome will be dropped after observing the final results. Any incomplete run will be reported as incomplete rather than silently replaced with a smaller or more favorable panel.
