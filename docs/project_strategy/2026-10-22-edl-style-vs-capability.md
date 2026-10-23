---
written_on: 2026-10-22
written_by: agent:claude
papers:
  - donoway2026excessdescriptionlengthlearning
---

# EDL in the project, and whether proof style can inflate it

## Summary

- **Three efforts in the team use excess description length (EDL)**
  [@donoway2026excessdescriptionlengthlearning] to compare pretrained and RL-trained provers. Only Leon's has committed
  results, and those are on Leon's own models: RL's EDL is lower, but EDL tracks the starting loss (Spearman 0.94). The
  two runs on Dan's models (Leon's, configured 2026-10-02, and Charles's `edl_mock`) had no results in git as of
  2026-10-22.
- **EDL scores specific proof strings.** It therefore charges for *which* valid proof a model writes (style), not only
  for *whether* it can write one (capability). A capable model whose valid proofs differ from the targets it is trained
  on will show a high EDL.
- **No EDL measurement in the project isolates this yet.** The clearest evidence comes from teacher-forced agreement,
  not EDL. kalmar-pretrain's K12 control proves 90–94 % of 1,200 held-out theorems greedily, yet picks the Kalmár
  proof's next action on only 23–30 % of states. The paper itself warns that scoring format tokens confounds EDL (§5.5).
- **A cheap experiment, sketched below, would measure the style cost directly.**

## Evidence

### The three EDL efforts

| effort | learners | what they learn online | status |
|---|---|---|---|
| **ndbench, Leon** ([summary on `main`](https://github.com/chainik1125/nd-rl/blob/main/experiment-summaries/2026-09-30-benchmarking/README.md), `code/ndbench/edl.py`) | Leon's base and RL models | Fixed reference proofs of 9 new pattern classes plus a null-control class, 32–64 proofs per class. The learning rate is chosen from six by online code length, at a cost of ln 6. Two data orders. | Done and audited 2026-10-01. RL's EDL is lower on 9 of 9 classes, but RL also starts with lower test loss on 9 of 9. Per nat of test loss removed, RL costs as much as base or more on 7 of 9. The null control is near 0 (base 0.27, RL 0.19 nats per example). Leon added a configuration for Dan's cap-12 seed-0 pair (one data order, null control `dan_heldout`); its results are only in Leon's results store. |
| **`edl_mock`, Charles** ([proposal on `charles/main`](https://github.com/chainik1125/nd-rl/blob/charles/main/experiments/finetuning/edl_mock/proposal.md)) | Dan's base and EI r2 / r4 / r8 (seed 0) | 128 proofs, found by seed 1's ladder, of theorems the seed-0 base never solved. A control stream uses theorems the base solves. Prequential blocks, retrained from the start for each block. | Launched 2026-10-10; no results committed. Pilot: the base's loss on those proofs is 0.081 nats per action token (≈ 29 bits per proof), against 0.038 on its own samples. Full fine-tuning overfits after about 2 passes. |
| **`long_rl_20261014`, Robbie** (`report/export.py` on `robbie/handoff-full-archive`) | each checkpoint, split by proof-length band | Leon's protocol, plus a rule that drops unstable learning rates | Only the exporter is committed; no results. |

Leon's code comment cites the paper as "Elicitation, Not Learning". Its arXiv title is "Excess Description Length of
Learning Generalizable Predictors".

### Why style can inflate EDL

The loss on a target proof y of theorem x splits into two parts:

  −log p(y | x) = −log p(valid | x) − log p(y | valid, x)

The first part is capability; the second is style, meaning which valid proof the model prefers. EDL is the online code
length minus n × the final test loss, so it counts the *systematic* part of a style difference: the part the learner
can learn. Random variation among valid proofs remains in the final test loss and is subtracted.

The direction of the bias depends on the targets:
- **Shared reference proofs** (Leon's design): a model that has drifted away from the reference style looks *worse*.
- **Another model's own proofs** (Charles's stream; stream D_RL in the finetuning plan's M2): the learner is charged for
  the teacher's style, which would read as "new capability".

The paper's §5.5 makes this point for format: scoring the full output "confounds format learning with capability
acquisition". It recommends scoring only designated output tokens or normalising outputs. It treats format as
low-entropy and quickly learned, which makes format an early transient in EDL. A proof strategy is chosen at every
step, so a style gap need not be cheap or transient. That is the case to test.

### Concrete evidence so far (none of it an EDL measurement)

1. **Same theorems, a capable model, a different valid style** (kalmar-pretrain, reviewed;
   [summary](../../experiment-summaries/2026-10-19-kalmar-pretrain-not-learned-far-below-k12-control-in-distribution-replay-fine-tune-matches-ei-jump-at-one-seed/README.md)).
   tf1200 is 1,200 held-out theorems, none in a K12 renaming class. Values are per seed (s0 / s1 / s2):

   | model | agreement with the Kalmár proof's next action | proofs fully agreeing (of 1,200) | greedy proofs Lean accepts |
   |---|---|---|---|
   | K12 control | 0.278 / 0.302 / 0.228 | 0 / 0 / 0 | 0.918 / 0.944 / 0.905 |
   | KAL-D (pretrained on Kalmár proofs) | 0.946 / 0.938 / 0.950 | 170 / 164 / 192 | 0.331 / 0.291 / 0.356 |

   The control proves these theorems but follows the Kalmár proof on under a third of steps. KAL-D follows it on 95 %
   but proves about a third. Any loss on Kalmár targets would rank KAL-D as "closer", although the control is the more
   capable prover.
   - **Caveats:** agreement is greedy-action agreement, not a log-likelihood. Kalmár proofs are also about 15× longer
     than K12's (mean 107.9 against 7.0 ND lines).
2. **Two valid styles of the same theorems** (proof-strategies;
   [summary](../../experiment-summaries/2026-10-18-proof-strategies-13-strategy-taxonomy-k12-reductios-are-templates-textbooks-use-working-reductio-mt-ds/README.md),
   Observation 2). On 990 K12 theorems the shortest-proof search is shorter than the generator's proof on 534. Mean
   pruned lines fall from 7.00 to 4.69, idle case splits from 34.8 % to 0, and proof by cases from 42.5 % to 6.1 %. No
   model has scored these proofs yet.
3. **Leon's likelihoods** (benchmarking). On 43 theorems that RL solves and base never does, base assigns about
   7.0 nats per line less than RL to neutral minimal proofs, and about 13.0 to RL's own proofs. This is confounded: the
   two models' capabilities differ on these theorems, and any model rates its own samples highly.
4. **Charles's pilot:** 0.081 against 0.038 nats per token. Also confounded: the first set is theorems the base never
   solved, the second its own solved samples.

## Next steps

**Ask first.** Leon's numbers for Dan's pair and Charles's `edl_mock` results may already bound part of this. Charles's
control stream comes closest, but its proofs come from another base seed, whose style is similar.

**Experiment sketch: style-swap EDL at fixed capability.** This ran as `style-edl`
([proposal 32](../proposals/2026-10-22-style-edl.md); reviewed result:
[summary](../../experiment-summaries/2026-10-23-style-edl-against-no-detectable-style-cost-at-fixed-capability-kalmar-180-bits-breaks-capability/README.md)).
- **Learner:** the K12 base `stage1_best12_s{0,1,2}_b1200`.
- **Theorems:** held-out K12-distribution theorems that the learner already proves (for example the tf1200 theorems with
  pass@32 ≥ 0.9). Split by renaming class into a 128-proof stream and a 128-proof test set.
- **Streams:** every stream covers the same theorems and every proof is Lean-checked.

  | stream | targets | expected EDL if style inflates it |
  |---|---|---|
  | S_own | the learner's own greedy proofs | ≈ 0 (matched style) |
  | S_seed | another base seed's proofs | small |
  | S_short | shortest-search proofs (minlen) | moderate |
  | S_RL | proofs that EI r8 found for these theorems | the case in question |
  | S_Kalmár | K12-Kalmár proofs | large; also report per action |
  | S_random | prompts paired with other theorems' proofs | ≈ 0 (a gate, as for the paper's random labels) |

- **Measure per stream:**
  - EDL, using Charles's prequential blocks with early stopping;
  - the starting loss;
  - the learner's pass@32 before and after fine-tuning on that stream.

  pass@32 should stay at or above 0.9 throughout. That is what "no capability change" means here.
- **Result that would demonstrate the effect:** EDL rises from S_own through S_Kalmár while pass@32 stays flat.
- **A score that should remove it:** score each target against all valid proofs, not one string:
  −log Σ_{y ∈ V(x)} p(y | x), with V(x) the valid proofs up to a length bound, enumerated by the minlen search. A
  second option is −log P(valid | x), estimated by sampling. If EDL under this score is about the same for every stream,
  the difference was style.
- **Cost:** CPU-scale, by Charles's timings: coding 200 proofs takes about 16 s, and a fine-tune step about 2 s at
  8 proofs. Each stream and seed takes minutes to an hour of laptop CPU. The pass@32 reads take a few GPU-minutes per
  stream, under $2 in total on a 3090.

Related: [kalmar-complexity](../../experiment-summaries/2026-10-20-kalmar-complexity-5443-bit-program-per-step-demand-not-program-length-ranks-kal-d-errors/README.md)
found that a model trained from scratch on Kalmár proofs absorbs 6.3–7.1 Mbit in its first epoch, measured against the
final model's held-out loss (1.7–2.1 Mbit against the loss at the end of epoch 1). The algorithm itself is 5,443 bits.
This is consistent with most absorbed bits being about the target text rather than the skill.
