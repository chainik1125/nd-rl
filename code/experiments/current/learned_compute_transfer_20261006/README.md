---
written_on: 2026-10-06
written_by: agent:codex
---

# Learned compute: textbook and Charles transfer evaluation

## Summary

Evaluate the already selected learned-compute checkpoints on the historical textbook72 benchmark and Charles's latest published problems. This is an evaluation-only follow-up to the [completed sprint](../learned_compute_allocation_20261006/README.md): there is no further training, development selection, threshold fitting, or alteration of the closed sprint's results. The question is whether more generated proofs improve the proof the model actually returns, and whether a compute penalty or correctness-only reward produces useful stopping.

The historical72 results are reused with their original provenance. New evaluation uses Charles's October 2 release pinned at `5d0084add453212e6bea8055d4d28cec790f268c`, fetched from `origin/charles/main` on October 6. Working checkout: `dmitry/main` at `315f36266bb4dd7feeff87228cb2662f8a36badc`, with the existing sprint changes preserved.

## Frozen scope and comparisons

- **Charles ordinary187:** 87 candidate_v0, 72 candidate_v1, 14 hand-proved, and 14 batch3 problems. These are the primary new results. Keep all problems, including those whose reference proof exceeds the old Lean renderer's name limit.
- **Charles derived225:** evaluate each released sequent as an ordinary proof problem, with no supplied proof prefix or additional instruction. Report these only as sequent-projection diagnostics, separately from the ordinary187. They are not native completion, repair, cloze, or constrained-proof scores. Preserve source nuance and base identity; 144 rows are designated curried twins of other release problems. Do not treat the 412 entries as independent native tasks or pool them into a headline score.
- **Original generator + phase-1 fixed-eight selector:** reuse the original r16 generator and all three frozen training seeds of the pooled chooser, factorized ranker and BCE critic. Report pooled chooser as the requested phase-1 method and the others as comparisons. Reuse the frozen feature and public-text baselines. Candidate sampling seed 94006, eight independent attempts, original sampling precision and caps, temperature 0.8; primary candidate order 77106.
- **Joint generator/controller, compute penalty:** reuse the three frozen cost-0.03 checkpoints, evaluated with their learned stopping and selection distributions.
- **Joint generator/controller, correctness-only:** reuse the three frozen zero-cost checkpoints. This arm is free to stop up to eight attempts; it has no explicit reward for consuming all attempts and no explicit coherence reward.
- **Joint fixed-four control:** reuse the single frozen checkpoint trained to return after four attempts. Its primary returned-proof score is at four attempts. An eight-attempt readout is an extrapolation of this four-attempt-trained model. Retain fixed-one and format-only controls as supplementary comparisons.

New joint-model evaluation samples two full contextual paths per problem, seed 95006, using the existing FP32 inference implementation, maximum eight attempts, temperature 0.8, 16,384-token context, 1,024 emitted tokens per attempt, 256 tokens per action, and 48 actions. Reuse checkpoint-bound settings exactly. Forced full paths measure both the oracle and the exact conditional expectation of each trained stopping/selection policy; measurement compute is greater than expected policy compute.

Freeze dataset hashes, model hashes, inference code hashes, seeds and selectors before reading new model outcomes. Use separate registries and output directories; leave the previous final registries untouched. Public model observations exclude reference proofs, private verdicts and audit labels.

## Metrics and interpretation

For each generator report pass@1, oracle availability through the fixed cap, returned-proof correctness, and expected attempts. Independent-pool pass@1 averages correctness over the eight exchangeable samples; also retain first-candidate correctness. For contextual joint paths, pass@1 is the first proof only. Oracle@8 means a valid proof occurred somewhere in that generator's path; it is not the policy's achieved accuracy and the later contextual attempts are not independent samples. The fixed-four control additionally needs oracle@4.

Use theorem-level summaries after averaging sampling repeats and then training seeds. Show individual training seeds as well as their mean; three seeds do not establish a precise distribution over retraining. Ordinary-set confidence intervals resample problems. Derived projections require base-level grouping and per-nuance reporting because related rows are dependent. Keep paired policy-minus-own-first comparisons for the stopping question. A decline in selected correctness after additional attempts does not by itself establish an intrinsic coherence penalty: generation, categorical selection, and stopping can contribute separately.

## Evidence

- [Task compatibility review](reviews/scope_compatibility_review.md).
- [Pinned release acquisition script](selection/acquire_release.py).
- Artifacts: `artifacts/learned_compute_transfer_20261006/` at the repository root.
- Reused historical metrics: `artifacts/learned_compute_allocation_20261006/analysis/final_historical/selection/metrics.json` and `artifacts/learned_compute_allocation_20261006/allocation/final_historical_summary_fp32_v2.json`.

GPU claims are recorded in `/data/users/dmitry/GPU_CLAIMS.md`: GPUs 1–2 for phase 1 and GPUs 3–7 for joint inference. Freed phase-1 devices can subsequently run remaining joint-model jobs. Other users' jobs are left untouched. Release claims when each queue finishes.

## Next steps

Complete the frozen Charles evaluations, independently audit denominators and model bindings, and write the comparison report and figure under `experiment-summaries/2026-10-06-learned-compute-transfer/`.
