---
written_on: 2026-10-01
written_by: agent:claude
status: approved by Dan 2026-10-01; runs `trajectory` (cap 12), then `trajectory-cap6` (cap 6, same configuration)
---

# Proposal 18: how the model organism learns its proofs, from initialisation through pretraining and RL

## Why

best-cap12 is Robbie's 9.56M recipe in the proof-state environment, trained on cap-12 data and then RL'd with our
ladder (`best-state`). It solves 51–52 of the 72 textbook problems, 237–239 of holdout250, and nearly all of the
long pools. That makes it a model organism for the project's question. For a proof it finally finds, when does
that proof become likely: during pretraining, during RL, or only at the end? And do theorems that only RL solves
look different, along the way, from theorems pretraining already solves?

## What gets measured

Train fresh models with the `best-state` recipe and **save the checkpoints along the way.** Then score every
checkpoint, from initialisation through pretraining and every RL round, on a fixed set of theorems.

**1. Log-likelihood of the eventual proof.** For each theorem, fix a target proof. Score it under every
checkpoint, teacher-forced through the proof-state environment: the sum over steps of log p(action | state).
Report:
- the total;
- the mean per step;
- the **worst step** and which step it is. `support-followups` and `lit-measures` found that new proofs hinge on
  one or two very improbable steps, so the question is when that step stops being improbable.

There are two targets:
- **Eventual proof:** the final RL checkpoint's highest-likelihood accepted proof, for theorems it solves.
- **Reference proof:** a fixed known proof (textbook72 reference proofs where they exist; otherwise the pool's
  shortest known ND proof). Every theorem has one, including those the model never solves. It also checks that
  the conclusions do not depend on which proof is tracked.

**2. pass@1, pass@8, pass@64, pass@256** per checkpoint, from 256 samples per theorem with the unbiased
estimator.

## Theorem groups (Dan's split)

Groups are fixed by the final checkpoints of each training run:
- **A. Pretraining solves:** the final pretrained checkpoint solves the theorem at k 256.
- **B. Only RL solves:** the final RL checkpoint solves it and the final pretrained checkpoint does not.
- **C. Unsolved:** the final RL checkpoint does not solve it.

Group membership comes from one set of samples (seed 0). The pass@k curves use a **separate** set (seed 1), so
the groups' endpoints are not inflated by selection. Each training seed gets its own groups, with membership
reported.

## Theorems and models

- **Theorems:** textbook72 (all 72) plus holdout250. Neither was used for any selection, and together they give
  groups of useful size. best-cap12 solves about 51 / 72 and about 238 / 250 after RL, and about 30 and about
  187 frozen, so roughly A ≈ 215, B ≈ 75, C ≈ 30.
- **Models:** 3 fresh seeds of the `best-state` cap-12 recipe (9.56M, `lean_staten`, Stage-1 1,200 s on an A40,
  then the T1 ladder).
- **Pretraining checkpoints:** about 14, log-spaced, from step 0 (initialisation) to the end (≈ 24k steps).
- **RL checkpoints:** the starting checkpoint and the end of each of the 8 rounds.
- **Sanity check:** the final checkpoints should reproduce `best-state`'s best-cap12 numbers within its seed
  spread.

## Figures

1. Log-likelihood of the eventual proof (total, per-step mean, worst step) against checkpoint, one line per group,
   with IQR bands. The x-axis runs through pretraining (log steps), then the RL rounds, with a divider. Then the
   same for the reference proof.
2. pass@1, @8, @64 and @256 against checkpoint, per group.
3. For about 12 example theorems (4 per group), a step-by-step heatmap of log p(action) across checkpoints,
   showing when each step becomes likely.

## Cost and time

- **Pods:** ≈ $20–30 (budget $30).
  - Stage-1: 3 × ≈ 20 min.
  - T1 ladders: 3 × ≈ 7–8 h on A40s, in parallel.
  - Scoring: about 66 checkpoint × 322 theorems × 256 samples. That is about 13 A40-hours; scoring each RL round
    starts as soon as its checkpoint exists.
- **Time:** ≈ 10–12 h end to end.
- **Compute:** recorded per the policy.

## What it could show

- **If group B's worst step only climbs during RL,** RL is teaching a specific move, and the heatmaps show which
  one.
- **If B's proofs were already fairly likely after pretraining** and RL mainly lifts their pass@1, that is
  sharpening: B was within pass@256 reach but just under the line.
- **If group C's reference proofs stay improbable at one step throughout,** that step is the wall, and it names
  the next experiment.
