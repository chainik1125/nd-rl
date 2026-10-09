---
written_on: 2026-10-09
written_by: agent:claude
---

# Week in review: 2 – 8 October 2026

**Summary:** This week we tested what RL adds to our best model. Under a compute-fair definition of "capability",
most theorems RL appeared to create were within the base model's reach. RL still wins overall, a few theorems survive
every control, and verifier-guided sampling was the cheapest gain.

The librarian's run-by-run digests: [`digests/2026-40.md`](../digests/2026-40.md) (2–4 October) and
[`digests/2026-41.md`](../digests/2026-41.md) (`capability-defs`). Fourteen runs cleared review (one of them, `mcts-b`, by
not running), for **≈ $193** of pod time. Every number below is reviewed, judged by Lean alone, and given per seed (s0 / s1 / s2) unless stated.

## What we did

**Is RL creating or eliciting?**
- **What "capability" should mean**
  ([capability-defs](../../experiment-summaries/2026-10-05-capability-defs-compute-matched-budget-shrinks-equal-k-creations-rl-still-ahead-lem-teachable/README.md),
  [proposal 26](../proposals/2026-10-04-capability-definitions.md),
  [report](capabilities/REPORT.md)). It screened 185 new papers and wrote 20 definitions; a critic agent broke all 20,
  and 6 were dropped. It then measured 12 of them on our models. Main results:
  - **Most "creations" are within the base's reach at a fair budget.** The budget is RL's own GPU time spent on base
    attempts instead, ≈ 2 × 10⁴ per theorem. Of the 54 / 51 / 60 theorems RL "creates" at equal k (256 attempts),
    22 / 21 / 14 stay out of the base's reach at that budget.
  - **Few survive every control.** 20 / 14 / 8 remain net of an RL-free replay-only control, and 4 / 5 / 5 were never
    solved by any seed's base. A 1.2–1.5 M-attempt certification confirmed 7 of the 9 (theorem, seed) pairs it tested.
  - **RL is still ahead overall:** round 8 solves 21 / 16 / 12 more theorems than the base reaches at the budget.
- **Earlier pretraining checkpoints**
  ([rl-from-ckpt](../../experiment-summaries/2026-10-02-rl-from-ckpt-early-start-ladders-reach-end-arm-from-step-5000/README.md),
  proposal 19). RL from pretraining step 5,000 onward ends level with RL from the end of pretraining; from step 1,600
  it ends 13 textbook problems short. Replay of pretraining data inside each RL round recovers most of the gap, so the
  pre-registered test could not discriminate.
- **More RL rounds**
  ([rl-continue](../../experiment-summaries/2026-10-03-rl-continue-cap12-group-c-rises-on-3-of-3-seeds-saturation-falsified-target-burst-is-one-seeds-excluded-middle/README.md),
  [cap 6](../../experiment-summaries/2026-10-03-rl-continue-cap6-group-c-rises-on-3-of-3-seeds-saturation-falsified-cap-6-still-below-cap-12/README.md)).
  - Rounds 9–16 add only 20 / 68 / 22 training targets at cap 12. But the theorems no earlier model solved ("group C")
    keep rising on every seed.
  - Seed s1's jump is one schema, excluded middle (A ∨ ¬A).
  - textbook72 went from 48 / 48 / 53 to **51 / 57 / 56** of 72. The result is a new named model,
    [`state_env_october_r16`](../models/state_env_october_r16.md); [`state_env_october`](../models/state_env_october.md)
    keeps meaning round 8.
- **What RL learns and fails to learn**
  ([organism-analysis](../../experiment-summaries/2026-10-02-organism-analysis-term-size-predicts-rl-success-negation-boxes-do-not-move/README.md)).
  - Proof term size predicts which theorems RL solves.
  - Negation-introduction steps (mostly ¬¬X) stay at −7 to −9 nats through RL, while every other step class rises.
    They are the largest class in theorems RL never solves.

**Algorithms and test-time compute** (Dan's to-do from the [2 October meeting](https://github.com/chainik1125/nd-rl/blob/main/docs/research_calls/group_meeting/group_meeting-02-10-2026.md): test-time scaling in the games tradition):
- **Verifier-guided sampling works**
  ([guided-tts](../../experiment-summaries/2026-10-03-guided-tts-logical-redraws-beat-plain-on-long-theorems-structural-redraws-do-not/README.md),
  [proposal 24](../proposals/2026-10-03-guided-sampling.md)). The sampler redraws a step that a Lean-sound check
  rejects. At matched sampled tokens it beats plain resampling on the long theorems of Charles's textbook release in all 6
  models: +9.1 pp at
  cap 12 and +8.3 pp at cap 6. Redrawing only the structurally invalid steps does nothing. It is now the default
  read-out alongside plain.
- **Tree search does not help after RL**
  ([mcts-a](../../experiment-summaries/2026-10-02-mcts-a-puct-value-search-gate-fails-on-group-c/README.md),
  [proposal 22](../proposals/2026-10-02-mcts.md)). PUCT with a learned value solves 2 / 2 / 3 group C theorems
  against 4 / 2 / 4 for sampling at matched wall clock, so the MCTS-as-expert phase was not run. Search does help the
  *pretrained* model (+19 / +10 / +14 on a 100-theorem pool).
- **GRPO vs expert iteration**
  ([grpo-best](../../experiment-summaries/2026-10-02-grpo-best-grpo-default-trades-theorems-with-ei-pass4-level-overall/README.md)).
  - Default and unlikeliness GRPO trade theorems with EI and lose overall.
  - The pass@4 objective is level overall and ahead on group C on 3 / 3 seeds (+3 / +18 / +2), mostly on one seed.
- **More compute does not explain Robbie's recipe**
  ([compute-match](../../experiment-summaries/2026-10-02-compute-match-matched-ladder-gpu-time-does-not-close-gap/README.md)).
  Given the same RL GPU time, our older 3.2 M recipe stays 15.7 textbook problems behind on average.

**Desk work**
([proposal 21](../proposals/2026-10-02-claude-heavy-set.md), five Claude-heavy runs, ≈ $2.50 of pods in all):
- [claim-audit](../../experiment-summaries/2026-10-02-claim-audit-headline-counts-reproduce-three-claims-weaker-than-stated/README.md):
  every headline count reproduces; three claims were weaker than stated (corrected † in last week's review).
- [evidence-atlas](../../experiment-summaries/2026-10-02-evidence-atlas-map-of-95-experiments-whole-proof-rescore/README.md):
  a map of 95 experiments. The whole-proof interface scores below the proof state in all 8 matched cells.
- [lit-review-2](../../experiment-summaries/2026-10-02-lit-review-2-create-vs-elicit-exploration-model-organisms/README.md):
  51 papers on create vs elicit, exploration and model organisms.
- [radical-scoping](../../experiment-summaries/2026-10-02-radical-scoping-departures-knockout-and-fol-feasible/README.md):
  two departures look feasible, a knockout of one proof rule combined with exploration, and first-order logic
  ([draft](../proposals/2026-10-03-radical-departures.md)).

**Proposals not run:**
- [20, exploration](../proposals/2026-10-02-exploration.md): waiting.
- [23, RL on Charles's harder textbook release](../proposals/2026-10-03-rl-on-textbook-release.md): needs Charles's OK
  to train on part of his benchmark.
- [24 Part B, guided sampling inside RL](../proposals/2026-10-03-guided-sampling.md): waiting.
- [25, nanoproof](../proposals/2026-10-04-nanoproof.md): shelved, since it has no public weights and reproducing it
  would cost ≈ 100× our compute. Dan has asked the author for the checkpoint.

**Shared with the group:**
- PRs to main: [#23](https://github.com/chainik1125/nd-rl/pull/23) (the 2 October meeting notes) and
  [#24](https://github.com/chainik1125/nd-rl/pull/24) (`state_env_october_r16` listed in WHERE_THINGS_ARE), both
  merged. The Lean tokenizer and judge ([#19](https://github.com/chainik1125/nd-rl/pull/19)) are also on main.
- Charles's new textbook release: `state_env_october` with guided sampling proves 71–75 of the 87 in `candidate_v0`
  and 56–59 of the 72 in `candidate_v1`, but none of his 14 long hand-proved problems.

**Orchestration:**
- Runs can now set their thinking effort (`capability-defs` ran every role at max).
- The balance floor is $35, down from $100.

## What seems important

1. **"RL created it" depends on the budget, and our definitions disagree.** At equal k RL "creates" 50–60 theorems
   per seed; at a compute-fair budget, 14–22; across every control, 4–5. The 12 measured definitions overlap at a
   Jaccard of only about 0.3. Every create-vs-elicit claim has to name its definition and budget; equal-k pass@k is
   descriptive only.
2. **RL looks like "more of the same, done efficiently".**
   - On an ability scale fitted across all checkpoints, RL moves three to four times as far as RL-free training of
     similar compute. At a matched ability gain, though, its theorem-level gains are no larger.
   - Replay-only training reproduces a third to a half of RL's apparent creations.
   - Spending RL's compute on more pretraining is a lateral move: it gains 12–19 theorems and loses 30–42.
3. **What RL found was a proof, not a hard-to-learn ability.** Excluded middle appeared on one seed only. Four of
   that seed's proofs teach it to every seed's base, and sixteen teach it even to a base pretrained without a single
   double-negation step.
   Discovery looks like a lottery across seeds, and once one proof exists, the skill is cheap to transfer.
4. **Test-time wins came from a better sampler, not from search or a new RL objective.** Guided redraws gave +8–9 pp
   at matched compute; MCTS and GRPO's default objectives did not beat sampling or EI.
5. **Our training targets are nearly used up** (≈ 98 % solved at cap 12), so more rounds mostly sharpen. New
   capability now needs harder targets or a richer domain, not more of the same RL.

## Next steps for the research agenda

1. **Choose the project's definition of capability.** `capability-defs` recommends reach at a compute-matched budget,
   with each theorem's k-to-solve interval, plus a capability-vs-propensity format and family-level verdicts. Its
   report lists nine questions for Dan ([§6](capabilities/REPORT.md)). The biggest are which budget defines "the base
   cannot", and whether the pretraining replay inside each RL round counts as "RL".
2. **The cleanest "can RL create?" test here: a composition knockout (≈ $26).** Pretrain three seeds without one
   composition (double negation applied to a negation-introduction line), run 16 EI rounds, and see whether RL
   acquires the excluded-middle family.
3. **Check the four survivors (≈ $5):** re-read them at doubled length caps, and give the other seeds' base models the
   same 1.2 M attempts.
4. **Harder targets:** proposal 23 once Charles agrees (he is also gathering more pretraining data), or a richer
   domain such as first-order logic (radical-scoping).
5. **Negation-introduction steps that RL never improves** (organism-analysis): a cheap targeted test of whether
   exploration or GRPO's pass@4 objective moves them.
