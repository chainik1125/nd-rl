---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# Forced-path and one-step causal evaluator audit

## Forced-path coupling

For the present protocol, sampling a complete forced-continuation path and integrating stop/index probabilities is valid. Future attempts depend on retained candidate history and fixed model parameters; sampled stopping/index decisions do not change the generation prompt. Generator random seeds come from a separate local Python RNG, while discarded decision draws use the global torch RNG. The checkpoint remains in evaluation mode. Consequently, evaluating all possible first stopping boundaries on the same generated path provides the correct conditional expectation, and averaging generated paths estimates the policy's value.

The independent audit recomputed learned and fixed-1/2/4/8 correctness, attempts, and expected generated tokens on both saved protocol paths, with **zero arithmetic discrepancy**. All **16 public transition transcripts** were rechecked directly in Lean against their original theorems, with **zero label disagreements**.

Greedy-index readouts are legitimate prespecified secondary policies, with uniform treatment of tied maxima. Greedy stopping additionally changes the stopping policy and must be labeled separately from the categorical training/evaluation rule. Costs inferred from the training checkpoint avoid silently evaluating a cost-trained policy at zero cost.

Compute requires different weights: a learned stopper executes its controller at each reached boundary; fixed/count-only stoppers need the selector only when they stop. At one candidate, fixed/count-only selection is deterministic and needs no LM selector call. The root identified the original every-boundary overcharge; I flagged the remaining one-candidate overcharge and missing final-test frozen-protocol guard to the allocation researcher. Full-path measurement expense is distinct from expected deployment expense. Saved earlier smoke quality is valid even where its compute convention predates these corrections.

## Causal probe

Code inspection confirms that the development runner writes all incumbent and continuation states before its separate Lean verification phase. It preserves existing candidate identities under both ordinary and history-removed generation, shares continuation seeds and random name bases across those interventions, and scores the original theorem. It estimates **one additional attempt then acceptance**, not resumption of the complete stopping policy.

The independent smoke audit checked four states, 12 branches, and 16 total attempts. All 16 public proof transcripts were rechecked in Lean: **six valid, zero label disagreements**. Accept-now expectation, old/new correct probability mass, continuation value, and the difference agree with direct calculations to `2.98e-8`. There are no oracle-availability decreases or forced-state branches. All four matched history-removal branch pairs agree on their random seed, name base, and common per-action seed prefix.

The hard-limit logic excludes states flagged forced before branching. The fixed state reserve is still a design allowance rather than a proof of sufficient space for every future intermediate state; actual truncation/context outcomes must be reported. The inspected smoke has no forced states and therefore cannot by itself validate cap-boundary behavior.

## Analysis and interpretation

`analyze_causal.py` joins replicate branches into state means, resamples whole theorems, retains within-state old/new decomposition, and reports acceptance association within attempt-count/oracle-availability strata. This is coherent exploratory analysis. Association remains descriptive; incumbent quality within the any-valid stratum can still confound acceptance-versus-value patterns. State-specific Monte Carlo standard errors are retained separately.

Running the analyzer on the uniform-head two-theorem smoke gives identical theorem-average continuation values of approximately −1/3 and no identifiable acceptance-probability association. A degenerate empirical bootstrap interval here reflects two identical observed theorem means; it is not evidence of precise generalization. In particular, **a uniform selector can lose success when another mostly-invalid candidate is added**, even without learned context-sensitive stopping. A zero-cost policy may learn to stop at attempt one to exploit this initialization handicap.

Use a count-only hazard comparator before describing the policy as adaptive allocation. To match its development marginal stop distribution, set each counter hazard to total learned stop mass at that count divided by total learned reach mass, then freeze the vector. An unweighted mean acceptance probability over forced-path states does not generally match the learned stop distribution. A reward-tuned count-only hazard is merely a mixture of fixed budgets; distinguish that purpose from matching learned count frequencies. Variable context-forced boundaries can produce remaining compute/count mismatches and must be reported.

## Evidence

- [Independent saved audit](path_causal_independent_audit.json) and [CPU reproduction script](audit_path_and_causal.py).
- Saved smoke roots: `allocation/live_path_protocol_smoke_s52006` and `context/causal_protocol_smoke_v2` under sprint artifacts.
- [Analyzer smoke output](causal_smoke_analysis/summary.json).
- Fourteen targeted episode/controller/likelihood/path tests passed during this review.
- No final-test outcomes or protected hidden benchmarks were accessed.
