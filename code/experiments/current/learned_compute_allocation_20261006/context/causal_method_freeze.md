---
written_on: 2026-10-06
written_by: agent:codex
status: development_protocol_frozen_before_main_interventions
---

# Continuation-value interventions

This protocol names development measurements before their main outcomes are generated. It is not authorization to sample the final test set. It extends the [evaluation plan](../evaluation_plan_draft.md) and follows the independent [whole-policy review](../reviews/whole_policy_extension_review.md).

Use the independently trained zero-cost seeds 101 and 102, each with its checkpoint chosen at completed update 100 or 200 by the already declared DEV32 rule. Do not select the checkpoint using its continuation-value outcomes. The one-extra and whole-policy measurements answer different questions; both must remain visible regardless of their signs.

The panel is `artifacts/learned_compute_allocation_20261006/context/causal_dev_panel_v1/public.jsonl`, SHA256 `3eb46542e969b616e57dbb05bf681c44d2fa129d842b6c4e3b687c5e906be4f8`. It contains the same 64 previously prespecified development theorems, now alternating the two families while retaining their within-family source order. Thus its first 32 and first 16 cases are also balanced. The manifest records every theorem ID. No theorem was included or excluded using an intervention outcome. Two cases have appeared in a runtime smoke; this remains an exploratory development panel.

| Measurement | Theorems | States per theorem | Futures per state | Generator-history intervention |
|---|---:|---|---:|---|
| One extra attempt, then accept | First 64 | After attempts 1 and 2 | 8 | First 16 theorems |
| One forced extra attempt, then original stop policy | First 32 | After attempts 1 and 2 | 4 | First 16 theorems |
| Two forced extra attempts, then original stop policy | First 16 | After attempts 1 and 2 | 4 | None |
| Four forced extra attempts, then original stop policy | First 16 | After attempts 1 and 2 | 4 | None |

Use seed71006 and the existing keyed theorem/state/replicate seed rule. Force initial state collection through attempt two even when the learned policy would accept sooner. Exclude only states already at a hard limit; count and report these exclusions. Preserve the original theorem and every selectable proof. No checker call occurs until all states and futures have been generated. Final selection probabilities are integrated exactly; whole-policy future stop decisions are sampled with an independent keyed uniform stream.

The history intervention makes the future generator memoryless at **every** resumed step. The chooser still sees the original theorem and the full retained candidate set. Match branch generation seeds and stop uniforms between the actual-history and memoryless conditions. Different stop decisions may produce different branch lengths; this is part of the intervention, not missing data.

Report immediate-accept value, continuation value, their difference, mass assigned to valid old/new proofs, first-future validity, any-new-proof validity, realized extra attempts and oracle availability. Oracle availability cannot decrease. Forced-two/four branches may hit a hard limit before the requested count: report the capped fraction and realized count rather than calling every branch exactly two/four attempts.

Average futures within each state and states within each theorem. Bootstrap whole theorems for 95% intervals. Report acceptance/continuation-value association unconditionally, within attempt count, and within count plus incumbent validity; report acceptance range and residual standard deviation so a nearly constant stopping policy is not presented as a meaningful classifier. These associations are descriptive, and individual noisy negative branch means are not evidence of truly harmful states.

The new whole-policy loop has passed an independent two-theorem runtime audit, including a branch with three future attempts. A separate forced-four smoke passed independent replay of all 20 proofs and controller decisions; `context/future_stop_boundary_audit.json` under the artifact root records 31,000 enumerated stop-uniform sequences with exact agreement, including a requested four-attempt continuation capped at three. These checks precede all main intervention outcomes. A separate final-test protocol will freeze any confirmatory panel and model hashes after development choices are complete.
