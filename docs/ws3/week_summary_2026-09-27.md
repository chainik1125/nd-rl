# ws3 week summary, 2026-09-20 to 2026-09-28

**Question:** with the dataset and pre-training fixed, what existing RL library helps most, and
how much lift does RL give over pre-training alone?

**Status of the numbers.**
- **Reviewed:** exp-00 (review PASS) and exp-00c (review PASS WITH CORRECTIONS, all corrections
  applied). Both are merged into `charles/main`.
- **Not reviewed:** exp-00b. Its numbers are the executor's, on branch
  `charles/exp-00b-muon-local-check`, not merged.

Everything below is one seed per run.

## Bottom line

- **TRL works for our custom 3M model,** both `GRPOTrainer` and `SFTTrainer` for expert
  iteration (EI). There is no reason yet to switch library.
- **On held-out problems from the same generator, RL gives a large, verified lift over
  pre-training alone** (exp-00c, 56 minutes of laptop CPU):
  - length-7 average solve rate went from 0.12 to 0.39–0.43 with GRPO, and to 0.55 with EI;
  - EI also moved length 8 from 0 to 0.27.

  Two independent checks found no reward bug, no loophole and no leak.
- **The lift is narrow.** It is one repaired step in the generator's nested-implication proofs.
  Case-split and contradiction problems stay at 0, and **the textbook set does not move** (avg
  0.10 to 0.08–0.12; 0 to 1 of the 43 long problems, always the same one).
- **GRPO vs EI is unresolved.** GRPO wins on the pre-registered metric and EI wins on the rest.
  Settling it needs 3 seeds.

## What training was done on the pre-trained model

No pre-training or re-training of any kind. The pre-trained weights are only ever loaded,
evaluated, and used as the starting point and the KL reference for RL.

| run | checkpoint | training | size | outcome |
| --- | --- | --- | --- | --- |
| **exp-00** (09-23, reviewed PASS) | old `share_pretrain_4l_seed0`, **no longer the fixed one** | GRPO, lr 1e-5, KL 0.01, lengths 3 to 5 from `train.jsonl` | 50 steps, 12,800 rollouts | reward 0.876 to 0.923, KL 0.0017; nothing moved |
| **exp-00b** (09-27, *not reviewed*) | fixed Muon `039_muon-adamw-finish-5pct_s1` (sha256 `418af3f9…`) | same as exp-00 | 12,800 rollouts | inert: 90% of groups all-correct, KL 0.0003 |
| **exp-00c** (09-27/28, reviewed PASS WITH CORRECTIONS) | fixed Muon | GRPO at lr 3e-5, 1e-4, 3e-4 (48 steps, 16 prompts x group 16, KL 0.01); EI 3 rounds with SFT at 1e-4. All on a 198-prompt pool from the train half of `rl_targets` (L7, L8) plus 63 length-6 prompts | about 12,300 samples per arm, 49,338 total | large held-out L7 lift in every arm (below) |
| CPU plumbing tests and smokes | fixed Muon | a few GRPO/EI steps on 8 to 21 prompts | tens of rollouts | wiring only. One finding: a single GRPO step at lr 1e-3 breaks the model |

Totals: 6 real training runs and 75,000 rollouts or samples for training. Separately, 13,296
frozen-model samples went into the exp-00c pool filter. About 46 minutes of CPU training, all
on the laptop: no GPU, no pod, no Tinker, $0 spent. Trained models stay local.

**Outside our runs (context only):** Robbie's own 4-round EI on this checkpoint recipe solved
254 to 291 of 1,108 dev theorems. That comparison is why the fixed checkpoint changed on 09-23.

**Data rules now in force:**
- `rl_targets` is split into train and eval halves by renaming class; the eval half is never
  trained on.
- The test-file renaming classes are banned from file-based RL prompts.
- The fixed `train.jsonl` contains 110 of 256 `test_short` classes, so `test_short` is not a
  clean held-out set for this lineage.

## The fixed checkpoint, before any RL

| evaluation | result | source |
| --- | --- | --- |
| next-token accuracy, 5,000 heldout | 0.9850 (T1 gate passes) | exp-00b |
| greedy solve rate, heldout lengths 2 to 6 | 1.000 / 1.000 / 0.980 / 0.985 / 0.925 (old checkpoint: 0.725 at length 6) | exp-00b, reproduced in exp-00c |
| `rl_targets` eval half, L7 avg solve rate | 0.122 [0.079, 0.166]; 39 / 160 solved at least once in 32 | exp-00c |
| `rl_targets` eval half, L8 | 0 / 40 | exp-00c |
| textbook 72 avg solve rate | 0.10; 0 to 1 of the 43 at reference length 7+ | exp-00b, exp-00c |
| length frontier (avg >= 0.5 at a length and all shorter ones) | 6 | exp-00b |

## RL results (exp-00c, reviewed)

| held-out | frozen | GRPO 3e-5 | GRPO 1e-4 | GRPO 3e-4 | EI |
| --- | --- | --- | --- | --- | --- |
| L7 avg solve rate [95% CI] | 0.122 [0.08, 0.17] | 0.430 [0.37, 0.50] | 0.391 [0.33, 0.46] | 0.333 [0.27, 0.40] | 0.545 [0.48, 0.61] |
| L7 new capability (frozen 0/32, trained >= 0.5) | - | 42 | 30 | 26 | 62 |
| **pre-registered:** L7 paired gain on the 39 frozen-solved | - | +0.273 | **+0.325** | +0.163 | +0.245 |
| L8 avg; solved at least once of 40 | 0; 0 | 0.005; 3 | 0.013; 5 | 0.011; 2 | 0.266; 19 |
| greedy lengths 2 to 6 (frozen 0.978) | - | 0.982 | 0.978 | **0.881** | 0.978 |
| textbook 72 avg (frozen 0.101) | - | 0.113 | 0.119 | 0.084 | 0.108 |

The pre-registered selection rule carries **lr 1e-4** forward. 3e-4 is excluded for forgetting
short proofs. On one seed, 3e-5 and 1e-4 are effectively tied: the ranking flips depending on
the metric.

## Conclusions so far

1. **Library.** TRL `GRPOTrainer` and `SFTTrainer` both work on our model with a thin HF wrapper
   (a model-managed KV cache, `gradient_checkpointing=False`, and `loss_type="nll"` for SFT).
   The CPU laptop is enough for this scale: one arm takes about 10 to 15 minutes.
2. **Lift over pre-training.** It is large and verified in-distribution: held-out generator
   problems at length 7, and at length 8 for EI. None shows out of distribution (the textbook
   set), and none on case-split or contradiction problems. It is one seed, and not an emergence
   claim.
3. **Where to train.** At the frontier. Training at lengths 3 to 5 is inert for this checkpoint
   (exp-00b). Training on length-7 prompts the model sometimes solves moves held-out length 7 a
   lot (exp-00c).
4. **Learning rate.** 3e-5 to 1e-4 at KL 0.01 works. 3e-4 forgets short proofs, 1e-3 breaks the
   model in one step, and 1e-5 is very slow.
5. **Algorithm.** Unresolved. GRPO at 1e-4 leads on the pre-registered metric by 0.08. EI leads
   on L7 avg, new capability and L8, in half the training time.
6. **Evaluation is trustworthy.** Two independent recounts reproduced every number, with 0
   verifier disagreements in 115,127 re-checked rows. One method issue: the eval package's
   `mean_ci` is Wilson-type, not the plan's interval over problems.

## Positive signals worth working on

1. **The frontier lift is real and cheap to reproduce.** It shows up in every arm at 3e-5 to 1e-4
   and needs under 15 minutes of CPU per arm. *Work on:* 3 seeds, to turn it into a claim.
2. **EI reaches length 8.** Length 8 went from 0 to 0.266, and 19 of 40 problems were solved at
   least once, with only 4 length-8 prompts in training. It did this by combining two proof
   shapes from shorter proofs. The frozen model attempted 1,642 proofs nested three boxes deep
   and got none accepted; EI got 1,753 accepted. *Work on:* EI seeds, and whether the next length
   opens up by curriculum.
3. **Short-proof ability survives frontier training** at lr up to 1e-4 and in EI (greedy lengths
   2 to 6 unchanged). The exp-00b worry, that training on easy lengths costs length 7, doesn't
   arise when training at the frontier.
4. **Failed attempts are partly right, and the model knows when it is right.** From exp-00b:
   the valid prefix covers about 0.45 of a length-7 proof, and the model's log-probability
   separates right from wrong with AUROC 0.77 on `rl_targets` and 0.91 on the textbook set. This
   is still the case for line-level signal or best-of-n. It needs your approval as a labelled
   shaped-reward experiment.

**Negative or cautionary signals:**

- **Nothing reaches the textbook set,** and case-split and contradiction templates stay at 0 in
  every arm. The gain is one repaired step in one template family.
- **Diversity collapses:** most newly solved problems end up with a single proof.
- **Selection effects.** The training pool was pre-filtered to prompts the frozen model sometimes
  solves, and the lr rule selects on the eval set. Both are allowed, and both are caveats.
- **Easy-length RL may cost long proofs.** In exp-00b (unreviewed), 50 inert steps at lengths 3
  to 5 lowered 10 of 16 solvable length-7 prompts (sign test p about 0.01).

## Decisions needed from Charles

1. Review and merge exp-00b (its numbers are unreviewed).
2. Whether to report "new capability" as defined, when it is mostly one-step fixes of near
   misses.
3. Whether the pool pre-filter (frozen-model success on the train half) is acceptable as
   standard practice.
4. Switch the evaluator's `mean_ci` to intervals over problems before the Saturday graphs.
5. Whether shaped rewards (valid prefix, partial credit) and hint or prefix methods are in scope
   as labelled experiments.
6. RunPod funding. It is not needed for the next steps, which fit on the laptop, but it is needed
   for any scale-up.
7. The process flag from the exp-00c review: after a surprisingly good result, the queued
   pre-registered arms kept running during the stop-and-check. It cost only laptop CPU and the
   checks were clean. Should future runs pause instead?

## Next (in order)

1. **Seeds.** EI and GRPO at 1e-4 (and 3e-5), seeds 1 and 2, on the same split and pool: 1 to 1.5
   hours of CPU. This settles the lift claim and GRPO vs EI.
2. **Unfiltered pool.** EI and GRPO on the full train half: does the filter or the algorithm
   drive the result?
3. **Beyond the one template.** Longer training, lengths 9 to 10, or a curriculum. Do case-split
   and contradiction problems, or anything on the textbook set, start to move?
4. **Deliverables.** `docs/ws3/summary.md` and the plan section 5 graphs, from reviewed numbers
   only, once the seeds are in.
