---
written_on: 2026-09-20
written_by: agent:claude
---

# Proposal 8: Lean as the training format for the from-scratch model

Follows the [Lean-default decision](../project_strategy/2026-09-20-lean-default.md). Question:
does training the from-scratch model on the Lean rendering of the same cap-6 proofs change
what expert iteration can do?

**Design (suggestion).** Render the campaign's cap-6 pretraining sets with `nd2lean.py`
(one symbol per token; hypotheses referenced by name, so no line-index tokens), train Stage-1
identically (4 layers, d = 256, 6k steps; report sequence-length statistics of both formats),
then run (a) the depth-3 f = 0 dial with two seeds and (b) ladder rung T1 on the ladder's
transfer pool, both against the token-format results already on file and their own frozen
controls. Reward inside the loop: Lean-check the sampled term (batched `lean` calls; measure
the throughput cost against `nd_verify`), with `nd2lean`-inverse agreement checked on a sample.

**Pre-registered predictions.** Held-out in-distribution accuracy within 2 pp of the token
model; depth-3 f = 0 acquisition within the token model's range (0.27–0.36) — not worse;
transfer `L*` = 10, the same as the token model. The mechanism that could make it better:
named hypotheses replace absolute line indices, removing the untrained-index barrier found in
the take-home, so the length-generalisation gap between `abs` and `abs-fixed` (108 vs 0
seven-line proofs) should not exist in the Lean format. The result that would matter: `L*`
above 10, or acquisition of a rule-sequence pattern from f = 0 that the token model never
acquired.

**Budget and stop rule.** ≈ $30 on two 3090 pods; stop if Lean-checking throughput makes a
round exceed 3× the token-format round time (report the ratio and stop).

**Decision rule.** Not worse on all three predictions → Lean becomes the training format for
every subsequent run and the token format is retired. Worse → Lean stays the checker of record
and reporting format only, with the measured gap recorded.
