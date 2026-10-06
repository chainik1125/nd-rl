---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# Episode and controller code audit

## Findings

The current abstraction isolates policy observations from private converted proofs and completion metadata, retrieves the exact stored selected proof, and verifies it against the original theorem. The reward is exactly correctness minus `attempt_cost * (attempts - 1)`, including incomplete attempts. There is no reward normalization in this abstraction; the training loss still needs separate review.

Two concrete defects were reported and fixed by the allocation researcher during review:

- Accepting at an attempt/context limit with the default `forced=False` previously produced a voluntary-stop label. `Episode.decide` now infers forced termination from continuation availability.
- The abstraction previously permitted consecutive attempts without a CONTINUE decision, or repeated decisions without a new attempt. The `awaiting_attempt` state now enforces alternation.

The factorized controller correctly excludes stop likelihood at forced ACCEPT and index likelihood at CONTINUE. Independent execution of `code/.venv/bin/python -m pytest code/tests/test_learned_allocation.py code/tests/test_learned_allocation_controller.py -q` returned **9 passed**. The gradient test confirms zero stopping-logit gradient and nonzero selection-logit gradient at forced ACCEPT. This tests the likelihood helper, not yet a complete rollout/update path.

## Remaining integration checks

1. Each rollout record must retain its continuation-availability mask. The helper accepts a caller-provided forced mask; correct standalone algebra does not ensure the runner supplies it correctly. The episode bookkeeping stores numeric likelihoods, which by itself cannot establish a correct differentiable update.
2. Context reservation must count the actual encoded theorem, candidate separators, candidate token sequences, and decision markers. The full-proof size bound must be chosen before observing a future candidate, unless exact future size is explicitly treated as privileged offline knowledge. A fresh-sampled token bound cannot silently depend on private completion labels.
3. `encode_history` initially defaults to 2048 tokens, while the retrieved checkpoint configuration was reported as 1024. Use the declared experimental cap or label longer execution as a context-extrapolation intervention. Any cap-based candidate filtering changes the evaluation population and must be reported.
4. Validate or guard impossible forced-CONTINUE calls, empty histories, invalid candidate counts, and empty encoded histories at integration boundaries. The helper currently presumes valid inputs.
5. Trace independently loaded model weights, optimizer initialization, head initialization, and training data into each cost/no-cost run manifest. Equal zero-initialized heads are acceptable; inheriting a trained selector is not.
6. Verify the public `rendered_text` construction against raw failing actions once the collector exists. Keeping private fields out of a dataclass does not establish that a string has been stripped of checker-derived content.

## Evidence and scope

Reviewed `code/shared_code/learned_allocation.py`, `code/tests/test_learned_allocation.py`, `code/tests/test_learned_allocation_controller.py`, and `code/experiments/current/learned_compute_allocation_20261006/allocation/controller.py`. Read the actual ALiBi model's `features`/normalization interface in the retrieved source and found the controller's final-state extraction consistent with that interface. No neural training, candidate outcomes, or end-to-end runner was available in this review. This is a protocol-unit approval with integration conditions, not approval of experimental validity.

## Collector review

The subsequently supplied `selection/collect.py` keeps Lean verdicts and environment failure strings private; public `rendered_text` is assembled from sampled IDs including the failing action. Likelihood replay uses the actual state prefix, with the correct next-token offset, and separates sampling temperature from the fixed diagnostic temperature.

Two issues were sent to the selection researcher for resolution:

- **Raw-versus-effective names:** sampled action tokens precede environment-assigned name substitutions, while later actions cite those assigned names from subsequent states. Concatenating raw actions can make a checked valid proof appear to have broken references. Expose sufficient deterministic bookkeeping (effective action names or actual states) to represent the proof being labeled, or explicitly quantify and disclose the limitation. This bookkeeping is separate from semantic verifier feedback.
- **Sampled padding tokens:** the initial `trim` function drops every pad token. The sampling function does not mask pad logits, so pads before EOS can be actual sampled tokens. Removing them changes both replay prefixes and compute accounting. Trim only the suffix after the first EOS, retaining pre-EOS tokens; for no-EOS output retain the complete action-cap sequence. This defect's frequency has not yet been measured.

The split script excludes theorem equivalence under atom bijections and premise permutations against both listed base-training sources, then keeps atom-erased connective skeletons within one selector split. The latter is a split guarantee for the new selector dataset, not a claim that base pretraining never encountered those connective skeletons. Source identity and completeness remain provenance assumptions to verify against the actual checkpoint lineage.

## Independent raw smoke audit

The saved [independent audit](train_smoke_independent_audit.json) replays all 256 candidates in `pools/train_smoke_v2/public_normalized.jsonl`: 239 valid completed proofs, eight invalid completed proofs, and nine incomplete proofs. Effective action rendering and completion status have zero replay mismatches. No sampled PAD tokens occurred. This pool is a quarantined **training smoke test**, subsequently found to contain protected-theorem equivalents; none of these counts establish held-out performance.

For two entire candidates, CPU float32 replay of the actual action-prefix likelihoods agrees with saved CUDA bfloat16 diagnostics to about 0.00014 mean-logprob units per token. This is a numerical spot check, not bitwise reproduction. A raw invalid finished proof contains an explicit conjunction-introduction error: it pairs a `False`-valued hypothesis with `S` while declaring a different first conjunct. Therefore semantic-invalid completed candidates really occur; completion does not exhaust selection difficulty.

The schema-three normalization amendment uses the current action's `Env._eff` for every EOS-ended attempted action irrespective of application success, preserving truncated output raw. This removes an explicit success-conditioned public-rendering branch. The recorded normalization changed zero of these 256 smoke candidates. Future datasets must record their own changed counts.

## Runner and selector integration review

The frozen-pool controller runner uses complete-episode summed decision likelihood, averaged across episodes, with a leave-one-out reward baseline. It omits generator likelihoods because the generator is fixed. This is consistent with the stated controller-only objective. Replay checks compare stored behavior likelihood against the differentiable recomputation before updating weights. Forced and fixed-count decisions omit stop gradients.

Reported and resolved by the allocation researcher: resource records initially omitted generator state-prefix tokens; these are now included separately from controller prefixes and generated tokens. Evaluation now derives context/bound settings from the training checkpoint and rejects silent mismatches. Requested fixed budgets must still be compared with realized budgets when context forces earlier termination.

One selector-evaluation defect was sent to the selection researcher: overflow assigns uniform probabilities but an ordinary argmax then selects canonical candidate zero. This is first-candidate fallback, not random fallback, and differs from the separately reported expected accuracy. Tie/fallback conventions must be aligned before using development accuracy for checkpoint selection.
