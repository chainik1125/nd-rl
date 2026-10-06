---
written_on: 2026-10-06
written_by: agent:codex
---

# Proposal coverage and limits on final claims

This review compares the [proposal](../../../../../proposals/learned-compute-allocation/README.md) with the frozen [selection](../selection/method_freeze.md), [joint allocation](../allocation/main_method.md), [causal plan](../context/causal_test_plan.md), and [scope statement](../scope_and_limitations.md). It uses design and already reviewed DEV evidence only; no final TEST outcomes were read. No frozen method was edited.

## Claims that require separate evidence

| Claim | Required evidence and interpretation |
|---|---|
| Training improved proof selection | A reliable held-out gain over the same untrained architecture/interface supports this narrower learning claim, even if a stronger public-feature comparator still wins. Beating random alone does not isolate improvement due to training. |
| The selector meets the proposal’s usefulness/success criterion | Held-out selected-proof accuracy must beat the strongest relevant frozen non-oracle comparator with paired theorem uncertainty and consistent seed results. Improvement over its own untrained control alone does not meet this stronger criterion. Report pooled marginal selection as primary even if a secondary factorized/BCE/frozen-head method performs better. |
| The model learned to allocate computation efficiently | At cost 0.03, improve held-out utility over DEV-selected fixed counts, counter/context rules, and feasible DEV-fitted work-matched mixtures; show allocation varies usefully with public state rather than almost always stopping first. Independently trained fixed1/fixed4 controls distinguish stopping benefit from generator/selector optimization changes, but each has only one seed. |
| The zero-cost model learned when it is done | Require pre-limit harmful continuation, acceptance that preferentially identifies harmful states after count and incumbent-validity controls, and useful held-out selection compared with the relevant forced/simple policies. High acceptance or a negative average continuation effect supplies only part of this evidence. A wrong-sign or uninformative acceptance/value association does not meet the proposal's skill criterion. |
| Accumulated context explains learned stopping | Current generator-history removal can demonstrate a history-dependent change in future generation. It does not by itself show that the stopping head senses harmful context, or that history impairs recognition of an unchanged candidate pool. Marker-induced collapse of the original checkpoint is an initialization/interface result. |
| This is meta-learning | The implemented procedures are supervised proof selection and outcome-trained generation/stopping controllers. There is no explicit task-level inner/outer adaptation objective, held-out within-task learning protocol, or demonstrated cross-task adaptation advantage. In-context conditioning and RL across theorems alone do not establish meta-learning. Use the concrete method names; any broad “meta” terminology must be defined and presented as motivation, not an established result. |

“Model learned when it is done” should therefore be reserved for a state-sensitive, useful decision supported by the continuation-value diagnostics and controls. If only first-attempt concentration and harmful forced continuation appear, the supported statement is that training produced early stopping in an interface where further generation/selection can degrade. This can be a rational response to an initialization handicap without demonstrating the proposed recognition skill.

## Important coverage and comparability limits

The three questions retain independent task-training lineages: no selection checkpoint transfers into joint RL and no cost-trained model initializes zero-cost training. Cost/no-cost share the declared outcome-blind format adapter and fresh uniform heads. This is a matched initialization procedure, not three independently pretrained base models. Three RL seeds cover each primary arm; the causal study uses the prespecified two checkpoints only.

The pooled eight-proof chooser extrapolates from a stepwise 1,024-token proof-state interface to a new list interface with an 8,192-token allowance. Factorized scoring removes the full-list interaction but still uses a new transcript interface. A pooled failure is consequently not a clean test of whether semantic validity is represented in the base model. The frozen 385-parameter head tests whether trained linear readout of frozen representations is useful; it does not demonstrate native recognition without supervised labels. Strong public syntax/features and all-complete mixed-validity strata constrain semantic-recognition interpretations.

Candidate transcripts include deterministic environment transitions and grammar/scope structure while excluding Lean verdicts. Syntactic completion is legitimate public information, but it is predictive. Performance driven by completion is a valid operational result, not sufficient evidence of distinguishing semantically wrong complete proofs. Retain the strong public-feature comparator even when it beats a larger trainable model.

Staged confidence policies are separately trained/calibrated allocation baselines on independent frozen pools. They do not transfer initialization to joint RL and are not the same contextual generation process. Compare each scorer with its own frozen fixed-budget mixtures and label any cross-system comparison carefully. Feature likelihood/entropy replay is required inference work; BCE/frozen-head text scoring has its own forward cost. Frozen-pool unpadded logical work and live executed/padded forward work cannot be merged into a single numerical compute frontier. Neither measures end-to-end latency.

Native categorical selection, greedy selection and oracle availability must remain separate. Existing valid proofs are retained, so oracle availability cannot decrease. Negative native continuation value can reflect redistribution of probability onto invalid candidates even while oracle availability improves. Greedy selection that always chooses the first proof can avoid that dilution without demonstrating useful recognition of additional proofs. Do not describe native categorical harm as inevitable harm from extra reasoning under every deployment rule.

The causal panel forces states after attempts 1 and 2 and averages them equally, even when the policy would already stop. Its effect is on that intervention distribution, not the naturally visited policy-state distribution. The registered primary endpoint is one extra attempt followed by acceptance. The proposal's force-one-then-resume-policy estimand is implemented as a secondary endpoint, with forced-two/four as bounded secondary checks. State the estimand when reporting a number. Theorem intervals condition on fitted checkpoints and saved state/future draws; they are not uncertainty over arbitrary retraining or proof-program families.

## Proposal components not covered by the frozen sprint

The scope statement already records the single nonzero training cost, REINFORCE rather than repeated clipped PPO updates, short training budget, single-seed fixed controls, format adaptation, context extension and structural-but-not-proof-family holdout. The final report should additionally identify these omissions explicitly:

- No separately trained joint-RL costs 0.01 or 0.10; those values are evaluation/calibration utilities only.
- No outcome-RL eight-proof selector experiment; the primary chooser uses marginal valid-index supervision, with separately declared BCE/readout controls.
- No matched fixed-candidate recognition/acceptance study varying added nonselectable relevant history, position and filler. Initial boundary/filler diagnostics are not this trained-controller mechanism experiment.
- No compact-history controller retaining a selected incumbent and latest attempt, and no systematic length/count transfer study up to 16 attempts. Forced-two/four continuation within the eight-attempt cap is not count-limit transfer.
- No registered incremental entropy-prediction study of continuation value beyond count, occupancy, correctness and token-type/length covariates. Entropy logging and its inclusion in a feature scorer do not complete that mechanism analysis.
- No additional TRAIN-learned stopping head over frozen scores. This follow-up was deliberately deferred; DEV-calibrated confidence thresholds remain explicit policies, not a newly trained meta-controller.

These omissions narrow the conclusion rather than invalidate completed tests. The strongest final report will distinguish the primary successes/failures from secondary diagnostics, preserve unfavorable methods and families, and avoid interpreting incomplete optimization or interface failure as a general impossibility result.
