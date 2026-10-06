---
written_on: 2026-10-06
written_by: agent:codex
status: pre_test_development_review
---

# Corrected staged accounting and cost-training audit

## Accounting and comparisons

The v3 feature and three BCE staged reports correctly charge required scoring work. Feature scoring uses unpadded replay P + G; BCE scoring uses its independent theorem-plus-transition input length, with no feature replay. Both include ideal cached generation P + G − A. Fixed one omits scoring; mixtures correctly omit the fixed-one component's scoring while retaining its generation. Thresholds score every reached candidate. Oversized BCE candidates use the declared neutral score without a model forward.

Independent checks covered all **58,880 theorem-policy records** in the four reports: additive totals, mixture linearity, summary means, and fixed-one exemptions. Maximum arithmetic discrepancy was `3.73e-9` in large squared-length attention proxies; work-matching residual was at most `9.1e-13`. The earlier independent permutation audit establishes the inclusion/selection expectation underlying these quantities.

| Development method, cost 0.03 | Threshold accuracy | Same-critic fixed-mixture accuracy at matched total logical tokens | Threshold logical tokens |
|---|---:|---:|---:|
| Public feature predictor | .841291 | .808877 | 14,669.89 |
| BCE seed 0 | .829643 | .797871 | 8,171.11 |
| BCE seed 1 | .827485 | .798565 | 8,886.64 |
| BCE seed 2 | .826551 | .795615 | 8,116.78 |

These are **development-selected** thresholds and mixture weights. Their apparent gains are exploratory. Mixtures randomize the fixed count before seeing a theorem's candidates, with weights frozen on development. This cleanly isolates allocation benefit under a given critic. It is not a comparison against the strongest fixed-budget system across all available critics. For a broad superiority claim, include the development-selected cross-critic fixed-budget frontier, or explicitly limit the claim to the same-critic comparison.

The total is an unpadded model-input-token lower bound, not equal FLOPs or latency. It combines cached generation, dense replay, and dense critic inputs with different attention costs and batch efficiency. The v3 caveat correctly excludes sampler termination cadence and padding. BCE's smaller logical work than the feature model is meaningful under this stated proxy; measured deployment efficiency remains unestablished.

## Statistical freeze

Freeze each cost's threshold and each matching mixture before final outcomes. Do not choose the best final threshold, best seed, or best matching definition afterward. Report accuracy and residual work mismatch together: development matching does not guarantee equal held-out work. Do not refit mixture weights on held-out lengths or outcomes to erase that mismatch.

Use paired whole-theorem differences, stratifying the bootstrap by the two equally weighted families. Average the three BCE training-seed outcomes within theorem for a seed-averaged endpoint; show every seed separately. Such intervals condition on these trained models and do not establish uncertainty over all possible training runs. Candidate orders are already integrated exactly in staged expectations; do not treat eight candidates or many permutations as independent sample units. Report the attempt-reward comparison separately from the equal-logical-work accuracy comparison, because their objectives differ.

## Cost seeds 101 and 102

Both runs completed 200 updates of eight episodes with cost .03. Base/data hashes, all four saved runner hashes, all upstream source hashes, hyperparameters, numerical precision, and loss definition match. Configuration differences are confined to seed, output path, command, process ID, and start time. Every saved checkpoint at updates 50/100/150/200 loads on CPU, contains finite tensors, and matches the run configuration and completed-update count. Episode-derived metrics agree with logs within `6.2e-8`; maximum recorded generation likelihood replay error is zero.

| Last 400 training episodes | Seed 101 | Seed 102 |
|---|---:|---:|
| Mean attempts | 1.0150 | 1.0075 |
| Exactly one attempt | 98.50% | 99.25% |
| Selected correctness | .8425 | .8750 |
| First-attempt correctness | .8425 | .8750 |
| Mean first-boundary acceptance probability | .98329 | .98410 |

This is strong concentration on first-attempt stopping, not yet a held-out adaptive-allocation result. Acceptance probabilities retain some variation, so “exactly constant policy” would overstate the evidence. The training-window variation also includes parameter changes over time and cannot establish within-checkpoint state discrimination. Selected and first-attempt correctness coincide in these windows. Earlier windows have lower selected than first-attempt accuracy, consistent with the selector-dilution concern; this does not uniquely identify its cause.

## Causal protocol

The frozen development panel and paired interventions are adequate for their stated questions: one-extra and resumed-policy estimands remain separate; forced-two/four probes test delayed benefit; all old candidates remain selectable; memoryless generation is applied throughout each resumed branch; hard caps and realized counts are disclosed. The independent forced-four runtime audit has already validated the minimum-extra loop on its smoke panel; the freeze's statement that additional checks are required should link that evidence.

The remaining essential comparisons are already planned outside this causal panel: same-model fixed one and count-only stopping, forced budgets, independently trained fixed four, and format-initialization retention. Keep them visible. Oracle availability and old/new selected-valid mass distinguish added proof supply from selection dilution. If useful new proofs appear but the native selector loses correctness, call it selector failure rather than harmful reasoning. Memoryless-history improvement identifies an interface/history dependence of future generation under these trained weights; it does not by itself identify a semantic interference mechanism.

The states after attempt two are deliberately collected even when the learned policy would have stopped. Report their intervention effect as an off-policy state-panel result, not the policy's naturally encountered continuation value. Two trained zero-cost seeds and the balanced theorem bootstrap are sufficient for the bounded exploratory question; their results cannot certify a general meta-learning mechanism. No additional expensive control is necessary before this declared panel, provided these interpretation limits and existing controls remain in the report.

## Evidence and limits

[Independent reproduction script](audit_staged_v3_cost_training.py) and [raw audit summary](staged_v3_cost_training_independent_audit.json). Read v3 development reports and their frozen policies, current `threshold_baselines.py` and `critic_thresholds.py`, both cost-training episode/metric logs, configurations, saved source snapshots and checkpoints, and `context/causal_method_freeze.md`. This audit did not rerun Lean on training episodes or evaluate the checkpoints on new proofs. No final-test outcome or protected benchmark was accessed.
