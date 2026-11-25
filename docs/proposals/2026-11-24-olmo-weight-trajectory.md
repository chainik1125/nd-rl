---
written_on: 2026-11-24
written_by: agent:claude
status: proposal 47, written at Dan's request 2026-11-24. Dan chose the add-ons; the low-rank analysis is paused until he has the initial results, and the behavioural counterpart is a separate proposal (48). Dan approved it on 2026-11-24 at $50 ("If this experiment is sub 50$ we don't need to think much harder about optimization"); launched as `olmo-weights` on agent2 (brief `followups/BRIEF_olmo-weights.md`). No result yet.
---

# Proposal 47: how much the weights move at each stage of training, from pretraining to RL (`olmo-weights`)

## The short version

Dan wants a deliberately simple picture. Line up every released checkpoint of Olmo 3, from the first pretraining step
to the last RL step, and plot how much the weights changed since the previous checkpoint. The chart should make the
switches between kinds of training obvious: pretraining, mid-training, long-context, SFT, DPO and RL.

Weight distance is a crude measure; it is not the same as a change in behaviour (Proposal 48 measures that). But
nobody in this project has looked at the raw picture, and it answers some basic questions:
- Is RL's change to the weights tiny next to SFT's, as its learning rate suggests?
- Does RL move steadily in one direction or wander?
- Which parts of the network does each stage touch?
- How much of the network does RL leave untouched?

Olmo 3 is a rare case where this can be done across the whole pipeline, because AI2 released checkpoints at every
stage. Counted on Hugging Face on 2026-11-24:

| stage | Olmo 3 32B (the pipeline our runs use) | Olmo 3 7B |
|---|---|---|
| pretraining (stage 1) | `allenai/Olmo-3-1125-32B`: 655, every 1,000 steps (0 – 656,000) | `allenai/Olmo-3-1025-7B`: 1,421 (0 – 1,413,814) |
| mid-training (stage 2) | two data "ingredients" × 24, then their merge (`stage2-ingredient1+2-step23842`) | 52 |
| long-context (stage 3) | 14 | 13 |
| SFT | `allenai/Olmo-3-32B-Think-SFT`: 22, from two learning rates (`1e-4-step*`, `5e-5-step*`) | `allenai/Olmo-3-7B-Think-SFT`: 43 |
| DPO | `allenai/Olmo-3-32B-Think-DPO`: final only | `allenai/Olmo-3-7B-Think-DPO`: final only |
| RL | `allenai/Olmo-3.1-32B-Think`: 46, every 50 steps (50 – 2,300) | `allenai/Olmo-3-7B-Think`: 55, every 25 steps |
| RL from the base | — | `allenai/Olmo-3-7B-RL-Zero-{IF,Math,Code}`: 19 / 19 / 29 |

All are stored in bf16: 64.5 GB per 32B checkpoint and 14.6 GB per 7B one.

## The headline chart, and three fixes so it does not mislead

**The chart.** For each pipeline there is one line chart:
- **x:** checkpoints in pipeline order;
- **y:** relative change since the previous checkpoint, ‖θ_b − θ_a‖ / ‖θ_a‖, on a log scale;
- **colour:** by training type, so the line changes colour at each stage;
- **switches between stages** marked with a labelled vertical band, and the interval that crosses a boundary drawn
  as a distinct, highlighted point;
- **merges and run choices** (not training steps) drawn as hollow markers: the stage-2 ingredient merge, and which SFT
  learning-rate run became the released SFT model.

**Fix 1, checkpoint spacing.** Pretraining saves every 1,000 optimizer steps and RL every 25–50, so the raw interval
change mostly shows spacing. The main panel divides by the number of optimizer steps in the interval. A second panel
shows the raw interval change.
- **Stage 1 is sampled as adjacent pairs.** At each sampled location, take checkpoints s and s + 1,000, so every
  stage-1 point is a true 1,000-step change.

**Fix 2, relative size and learning rate.** Plot relative change, not absolute. Where the released configs or papers
give the learning-rate schedule, overlay it, and add a panel of change per step ÷ learning rate. A large share of any
step-to-step change is the learning rate itself.

**Fix 3, known jumps that are not training steps.** Mark the merge, the choice of SFT run, the SFT → DPO interval and
the DPO → RL hand-off. DPO has no released intermediate checkpoints, so SFT → DPO is one large interval: label its
number of steps.

## Add-ons (Dan's selection)

1. **Distance from each stage's start.** Plot ‖θ_t − θ_start‖ / ‖θ_start‖ within each stage. Also report:
   - straightness, (distance at the end) ÷ (sum of the step-to-step changes): 1 means a straight line, near 0 means
     wandering;
   - the cosine between consecutive changes.

   These are computed where the stage's checkpoints are at native spacing (stages 2 and 3, SFT, RL). Stage 1 uses the
   sampled sequence, labelled as such.
2. **Where the change happens.** Keep the top-level chart, plus one breakdown per stage:
   - by module type (embeddings, output head, attention q / k / v / o, MLP gate / up / down, norms);
   - by layer depth, as a heatmap of layer × checkpoint.
3. **How much does not change.** For each interval, report three fractions:
   - weights that are bitwise identical in bf16;
   - weights that changed by at most one bf16 rounding step (1 ulp of the earlier value): "changed very little";
   - weights that changed by more.

   Report overall and by module. bf16 rounding hides small updates (the fp32 training copies are not released), so the
   1-ulp line is the honest floor.
4. **Line it up with `olmo-ifcurve`.** On the 32B RL section, overlay that run's reviewed IFBench pass@1 and pass@8
   at its 9 checkpoints (0.29 → 0.68), and a rug of the 48 reach prompts' first-genuine steps
   ([summary](../../experiment-summaries/2026-11-21-olmo-ifcurve-ifbench-pass1-0p29-to-0p68-sharpening-from-1100-no-abrupt-reach-switch-timing-no-order/README.md)).
5. **Compare RL with `olmo-ifladder`'s LoRA.** That run's rank-16 `lora_main` adapter was trained on the DPO start
   with 16 genuine RL trajectories. It recovered 40–41 of the 48 reach prompts; its control `lora_ctrl`, trained on
   the DPO start's own successes, recovered none
   ([summary](../../experiment-summaries/2026-11-09-olmo-ifladder-lora-on-16-rl-trajectories-elicits-40-41-of-48-if-reach-prompts-control-0/README.md)).
   Both adapters are in the bucket, and both start from the same weights RL started from. For each matrix the LoRA
   touches, compare its change, scale · B·A, with RL's change from the DPO start, θ_t − θ_dpo:
   - the size ratio, ‖ΔW_LoRA‖ / ‖ΔW_RL‖, at step 2,300;
   - the cosine between the two, at every RL checkpoint (does the LoRA point where early RL or late RL went?);
   - the same for `lora_ctrl`, plus a random rank-16 baseline.

   The question is whether a 16-example fine-tune that recovers most of RL's reach moves the weights in RL's direction,
   and at what fraction of RL's size.
6. **The 7B pipeline as well as 32B,** with the same charts. The 7B model adds RL from the base: the three RL-Zero
   runs start from the same base `main` that the 7B SFT started from. Compare how far, how straight and where RL moves
   when it starts from a base model against when it starts from DPO.

**Paused:** whether each stage's total change is low-rank (an SVD of the change matrices). Dan asked to have it
explained after the initial results.

**Separate:** the behavioural counterpart, the KL divergence between checkpoints' predictions, is Proposal 48.

## Which checkpoints

| | 32B | 7B |
|---|---|---|
| stage 1 | about 25 sampled locations, log-spaced early then even, each as an adjacent pair (≈ 50 downloads) | about 30 locations as pairs (≈ 60) |
| stage 2 | ingredient 1: all 24; ingredient 2: every 4th (6); the merge | all 52 |
| stage 3 | all 14 | all 13 |
| SFT | all 22 (both learning rates) | all 43 |
| DPO | final | final |
| RL | all 46 | all 55 |
| RL from the base | — | IF 19, Math 19, Code 29 |
| total | ≈ 164 checkpoints, ≈ 10.6 TB | ≈ 290 checkpoints, ≈ 4.2 TB |

**Hash checks:**
- Which stage-3 checkpoint equals the base's `main`.
- Which SFT checkpoint equals the SFT model's `main`.
- Whether `allenai/Olmo-3-32B-Think`'s 15 checkpoints are the first 750 steps of the 3.1 run, as the Olmo 3 report
  says. If they are, skip them.

## Method

- **Where it runs:** on cheap CPU pods with a disk large enough for two checkpoints. Download with `hf_transfer`,
  stream shard by shard, compute in float32 from the bf16 values, and keep only the previous checkpoint on disk.
- **What leaves the pods:** only statistics. Weights stay on pods.
- **Statistics per interval and per tensor:**
  - ‖Δ‖ and ‖θ‖;
  - the unchanged and 1-ulp fractions;
  - the dot product with the previous interval's change, for the cosine;
  - the dot product with the stage-start offset, for distance from the stage start.
- **Aggregation:** everything else in the charts aggregates these per-tensor numbers.

## What I expect (guesses, not pre-registered readings)

This is descriptive, but writing guesses down first keeps it honest:
- **Size:** RL's change per step is at least 10× smaller than SFT's, and RL's total change from the DPO start is
  smaller than SFT's total.
- **Unchanged weights:** RL leaves most weights within one bf16 ulp of the DPO start.
- **Direction:** RL is straighter than SFT.
- **The LoRA:** `lora_main` points more in RL's direction than `lora_ctrl`, though at a small cosine in absolute terms.

## What it does not show

- **Function:** a large weight change can matter little, and a small one a lot, for example through rescaling
  symmetries around normalisation layers. That is Proposal 48.
- **Training state:** the optimizer state and the fp32 training copies are not released, so "unchanged in bf16" is a
  floor, not the truth.
- **Comparisons across stages per step are approximate.** Batch size, tokens per step, optimizer settings and learning
  rate all differ between stages, so even the per-step comparison is rough.
- **Generality:** one model family, and one run per stage.

## Cost

Downloads dominate: about 15 TB in all. RunPod does not charge for data transfer.
- At an assumed 0.5 GB/s per pod, that is about 8–9 pod-hours, or less on two pods in parallel. Stage 0 measures the
  real throughput.
- Statistics per checkpoint are memory-bound: about 1–2 minutes for 32B on a multi-core CPU pod.
- CPU pods cost about $0.2–0.6/h, plus a few hundred GB of disk for a day.

**Estimate ≈ $10–25; budget $50.** That is a projection, re-costed at stage 0.
