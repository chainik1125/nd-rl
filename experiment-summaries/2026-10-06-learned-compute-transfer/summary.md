---
written_on: 2026-10-06
written_by: agent:codex
status: evaluation_in_progress
---

# Learned compute on textbook problems

## Summary

This follow-up evaluates the existing learned-compute checkpoints on textbook72 and Charles's October 2 release. No model is trained or selected on these results. The textbook72 results below are complete and reused from the audited sprint. Charles inference is in progress; no new Charles accuracy is reported yet.

## Completed textbook72 evaluation

| Method | Own pass@1 | Own oracle@8 | Returned proof correct | Mean attempts |
|---|---:|---:|---:|---:|
| Original generator + trained fixed-eight pooled selector | 42.36% | 62.50% | 52.78% | 8.00 |
| Joint model trained with fixed four attempts | 40.97% | 42.36% | 40.80% | 4.00 |
| Joint model with compute penalty | 41.44% | 43.98% | 41.40% | 1.04 |
| Joint model with correctness-only reward | 41.44% | 43.29% | 40.95% | 2.37 |

Each row uses its own generator. The first row samples eight independent proofs from Dan's original r16 checkpoint; pass@1 averages correctness across all eight exchangeable draws. Its first designated candidate alone scores 41.67%. The joint models generate later attempts conditioned on earlier attempts, so their pass@1 is the first proof and oracle@8 is availability along a contextual path. Their oracle values are not the same pool as the first row's oracle.

The fixed-four model returns after four attempts. Its oracle@4 is also 42.36%; its oracle@8 is a supplementary extrapolation. Cost and correctness-only rows use the exact conditional expectation of their learned categorical stop and selection policies. Means cover three training seeds except the fixed-four control, which has one. Two sampled full paths per theorem measure joint-model policies. Expected policy attempts differ from the compute spent generating those complete paths for evaluation.

The correctness-only model may stop at any point up to eight attempts. It has no explicit compute penalty, reward for using all attempts, or coherence reward. These existing checkpoints test whether useful stopping emerged under correctness-only training; they do not implement a separate objective that forces maximum effort.

## Charles evaluation scope

The new primary set contains 187 ordinary proof problems: 87 candidate_v0, 72 candidate_v1, 14 hand-proved, and 14 batch3. The source is Charles's release at Git commit `5d0084add453212e6bea8055d4d28cec790f268c`, fetched on October 6. All 187 problems are retained.

An additional 225 derived entries are evaluated as ordinary sequents and reported separately. Their supplied proof prefixes and extra task instructions are omitted. These are diagnostic sequent projections, not native completion, repair, cloze, or constrained-proof results. They include 144 designated curried twins and 164 source-base clusters. Their uncertainty must account for dependence between related entries; no pooled 412-task headline is used.

The available exposure audit finds one ordinary and one derived sequent matching the known K12 pretraining pool; none match the audited expert-iteration pool, actual r16 training mix, or historical72 under atom renaming and premise permutation. This follow-up audit does not normalize currying; Charles's release documents a separate curry-aware construction audit. Full training exposure is not known. The entire published release is retained, and these are benchmark transfer measurements rather than a claim of pristine unseen data.

## Evidence and reproducibility

- [Working plan and metric definitions](../../code/experiments/current/learned_compute_transfer_20261006/README.md).
- [Task compatibility review](../../code/experiments/current/learned_compute_transfer_20261006/reviews/scope_compatibility_review.md).
- [Preflight model and dataset audit](../../code/experiments/current/learned_compute_transfer_20261006/reviews/transfer_preflight_review.md).
- [Reused textbook72 extraction](../../artifacts/learned_compute_transfer_20261006/historical_comparison.json), guarded by hashes of the closed historical artifacts.
- [New live-model protocol](../../artifacts/learned_compute_transfer_20261006/live/protocol.json).
- [Original completed sprint](../2026-10-06-learned-compute-allocation/summary.md).

## Next steps

Finish the frozen Charles inference, validate the resulting tables against raw records, and render the full comparison figure. The open research question is whether improved selection and productive retries can close the gap between the oracle and the proof the model actually returns.
