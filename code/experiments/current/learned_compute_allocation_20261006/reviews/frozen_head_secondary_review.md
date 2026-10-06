---
written_on: 2026-10-06
written_by: agent:codex
status: fit_cache_and_v4_inference_verified
---

# Frozen-representation scalar-head secondary control

## Information boundary and generation preservation

The extractor consumes public theorem/transition inputs only, using the same factorized construction as `train_bce.py`. It records the final LayerNorm representation at the last nonpadding `<act>` position. Right padding cannot attend backward into this position under the upstream causal mask. No validity, completion, environment error, reference proof, or likelihood feature enters extraction. Training uses all 12,800 training validity labels deliberately; the one terminal head is saved before development labels are read. There is no development checkpoint/hyperparameter selection in the inspected code.

Independently loading the original checkpoint on CPU reproduces the saved pre/post digest `594173e7d67a64f5c4ffa9ac5176e4390770e44ce7d307590474bed3957c8eb5` across all **9,560,832 original parameters**. The extractor freezes them and runs without gradients. The new head contains exactly **385 FP64 parameters** in a separate artifact. Ordinary generation remains unchanged provided inference continues to use the unchanged original model and generation code; the scalar head does not replace its language-model head.

All source/input/cache hashes, canonical candidate joins, train/development theorem separation, and factorized input lengths were checked independently. Maximum factorized lengths are 2,368 training and 2,153 development, below the declared 8,192 limit. This still uses a newly introduced proof-level representation interface rather than the original short state-action training interface.

## Independent numerical reproduction

| Quantity | Reproduced value |
|---|---:|
| Development selected correctness | .84375 |
| Development candidate log loss | .308495937318633 |
| Development Brier score | .09141028890562439 |
| Training regularized objective | .28201677038192047 |
| Maximum absolute training gradient | 2.01898248064e-6 |

Recomputed cached-feature logits differ from saved predictions by at most `7.1e-15`. The objective is mean unweighted BCE plus `0.5e-3 * ||weight||²`, without penalizing the intercept or standardizing features. L-BFGS ends after 101 iterations and 102 optimizer function evaluations; an additional diagnostic closure explains the 103 history entries. The maximum gradient exceeds the prescribed `1e-7` gradient tolerance, which is correctly disclosed. Do not describe this as strict gradient convergence.

The exact stopping branch was not saved. In particular, the last optimizer objective decrease is `2.3286e-9`, larger than the `1e-9` change tolerance; the final duplicate objective is the diagnostic closure. PyTorch also applies `tolerance_change` to directional derivative and step size. Generic termination under the fixed stopping criteria is supported; a specific objective-change stopping claim is not established by the saved history. No refit is needed or recommended.

## Interpretation and remaining check

This is a supervised linear probe of a frozen representation, not zero-shot verification, joint RL, or evidence that fine-tuning cannot improve recognition. Its optimizer and trainable parameter budget differ from the fully fine-tuned critic. Report it as the declared secondary method, including its unfavorable comparisons, without replacing primary selectors.

The extractor's `ln_f(features(...))` is the exact structural input to the original LM head. A CPU FP32 check on two actual factorized inputs found only `3.8e-6` numerical difference between applying the original head to the full sequence versus the selected hidden vector, attributable to matrix-operation shape. This is not an independent GPU/BF16 re-extraction of the caches. Text-only inference requires the trunk pass but no likelihood replay.

## Accepted v4 inference successor

Independent artifact comparison confirms the original cache and fresh GPU v3 extraction are exactly equal for every feature vector, theorem/candidate ID, family, and input length. Fresh v3 and cached successor v4 logits and tie decisions are exactly equal to the saved terminal-head predictions. Development has zero overflow candidates. V4 delegates its entire nonoverflow path to the unchanged original extractor; no fresh v4 GPU pass is claimed.

The prespecified overflow amendment assigns each oversized candidate exactly logit 0/probability .5 without a trunk forward. It retains that candidate, records its full serialized length, and separately charges zero model input tokens. The explicit override prevents the fitted intercept from turning zero-placeholder features into a nonneutral score. Fixed-one still skips scoring; thresholds score each reached in-range candidate. Six focused tests pass independently, including all-overflow zero model calls, mixed fallback label independence, canonical label alignment, original LM-head preservation, and registration before inference/private reads.

The evaluator preserves BF16 extraction, FP32 features, and FP64 scalar readout; guards bind cache source/parameter digests and the terminal head. The separate threshold wrapper checks its frozen source identity, duplicate IDs, neutral-overflow cost/probability consistency, and existing threshold-helper hashes. The comparison validates matched BCE seed/configuration snapshots, rejects duplicate outcomes, and recomputes the frozen public-feature comparator. This bounded source/equivalence review accepts the secondary method for registration. These checks validate the declared batch-eight extraction; finite-pool logical work remains an idealized stopping-policy proxy rather than a measured sequential implementation.

Accepted source hashes: evaluator `86550275e2719e7c48b3d7ef49fe90ee18310d58b0d06788b847caa80f3c81e0`; inference adapter `a58d9e621768e3c8d0723787fbe8ae616c7a079ca8c580e21b9283640e51e898`; threshold wrapper `fc00acbfd42da6f1efbc0dbb4e8222769da30272c747fbcd7a09151202f65276`; comparison `46ca8480c6357d639fdc3f02e7217f5bb357e8565eeb6c23dafe2ac2c062b883`.

Evidence: [independent CPU audit](audit_frozen_head.py), [numeric results](frozen_head_independent_audit.json), [v3/v4 artifact comparison](frozen_head_v4_equivalence_independent_audit.json), the frozen-head method note, extraction/fit/common/inference source and saved artifacts. No research optimizer run, GPU job, final-test outcomes, or protected benchmark was accessed; the focused test suite includes a small synthetic CPU optimizer test.
