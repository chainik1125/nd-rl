---
written_on: 2026-10-06
written_by: agent:codex
status: reviewed_with_runtime_smoke
---

# Whole-policy continuation extension

## Estimand and information boundary

The inspected extension correctly suppresses acceptance at the initial retained state, generates the requested forced future attempts, and then samples the original stopping probabilities. Hard limits override forced continuation. The terminal index distribution is integrated exactly. Old-candidate correct mass is computed over all incumbent IDs; new-candidate mass covers every generated future ID. Private verdicts are still produced only after all public branch generation is complete.

The independent keyed Python stopping RNG samples the same Bernoulli hazards as the policy; it does not need to reproduce torch's particular random draws. The first future generation seed is shared with one-extra mode. Later generation seeds and stopping uniforms are keyed by branch and attempt count, allowing paired comparisons across interventions without one branch's termination consuming randomness intended for another.

Two interpretation details must remain explicit:

- `history_removed` supplies an empty generator history at **every** resumed future attempt, while retaining all candidates for selection. This is a memoryless-generation intervention throughout the branch, not removal of only the initial prefix.
- A requested two/four forced future attempts can be interrupted by a hard limit. Save and report actual future counts and the fraction reaching fewer than requested. Such capped branches are not estimates of an unconstrained “exactly four more attempts” intervention.

Numeric context/action/temperature settings are matched to the checkpoint. A source/protocol fingerprint check is also desirable because unchanged numeric settings do not guarantee unchanged serialization or sampling semantics. Freeze actual snapshots before the substantive panel.

## Generic analysis

The updated analysis preserves whole-theorem resampling and separates unconditional, count-conditioned, and count-plus-incumbent-validity associations. It reports acceptance range and residual standard deviation and avoids interpreting a slope when acceptance barely varies. It labels `new_attempt_valid` as validity of the first forced attempt and separately reports any-new validity and number of new attempts. This resolves the main ambiguity introduced by variable-length branches.

The interpretation remains policy-relative and descriptive. Negative continuation value can follow poor selection among retained valid candidates. A correlation between acceptance and value is not itself a causal explanation for the policy's behavior. In particular, a nearly constant stop-first policy plus an initially weak selector can produce apparently sensible average stopping without state-dependent allocation.

## Bounded substantive panel

Keep the prespecified 64-theorem, counts-one/two, eight-replicate one-extra panel as the broad measurement. A practical whole-policy extension uses 32 of those theorems, both counts, and four replicates. Two/four-forced-attempt checks can use 16 prespecified theorems or states with four replicates. Freeze IDs before inspecting values; do not select states for having negative one-step estimates. Replicate the core panel across an additional independently trained policy when feasible rather than expanding only within one seed.

The development-frozen fixed-count mixtures in `threshold_baselines.py` correctly interpolate selection and inclusion probabilities with saved weights. Matching attempts, generated tokens, and prefills remains three distinct comparisons. The new critic cost convention correctly omits deterministic fixed-one scoring while charging staged critic evaluation on reached candidates. These remain additive proxies, not measured sparse deployment wall time.

## Scope

The earlier one-step smoke does not cover the newly added stopping loop. The whole-policy runtime audit below supplies that additional evidence, subject to its stated boundary limits. No GPU run, final-test outcome access, or protected benchmark access was performed by this reviewer.

## Subsequent runtime audit

The trained-checkpoint `causal_policy_protocol_smoke_v3` now exercises four states, 12 branches, and 18 attempts from two development theorems. One branch generates three future attempts, so the resumed stopping loop is actually exercised. An independent audit rechecked all 18 public transition transcripts in Lean: 14 valid and zero disagreements with saved labels.

The audit independently reconstructed 18 controller inputs and reevaluated their distributions on CPU using the actual saved checkpoint. Maximum probability difference from GPU output is `1.19e-7`. It found zero mismatches in retained candidate identities, controller input lengths, hard-limit flags, keyed generation seeds, stopping uniforms, forced-continuation flags, terminal stopping decisions, or old/new selection-mass arithmetic. Runner and checkpoint hashes match the manifest.

[Runtime audit JSON](whole_policy_smoke_independent_audit.json) and [CPU reproduction script](audit_whole_policy_smoke.py). This validates the one-forced-attempt resumed-policy integration, not a learned-performance claim. The smoke contains no hard-limit termination and requests only one forced attempt, so two/four-forced-attempt and boundary override behavior still need explicit regression coverage before those conditions are relied upon. The reviewer launched no GPU work.
