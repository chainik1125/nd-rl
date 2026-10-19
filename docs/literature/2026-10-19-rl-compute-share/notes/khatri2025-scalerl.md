---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - khatriArtScalingReinforcement2025
---

# ScaleRL: what is measured, at what compute, on which bases

Paper: [@khatriArtScalingReinforcement2025] (proposed key; first author Devvrit Khatri).
Source: arXiv 2510.13786v1, the only version on 2026-10-19. Scout's pretraining figures come from Meta's Llama 4
model card (meta-llama/llama-models, main). Ledger: `B-SRL-*`, `B-L4-*`.

## Learnings

What the paper states:

- **Scale.** "more than 400,000 GPU-hours" on GB200.
  - 80 GPUs per run (64 generators, 16 trainers).
  - Ablations cost 3.5–4K GPU-hours each, leave-one-out runs 16K, large runs 30–100K.
- **Flagship runs.** "We run ScaleRL for 100\text{k} GPU hours on an 8B dense model, and 50\text{k} GPU hours on a
  17Bx16 MoE (Scout)." Also: "We trained for 7400 steps for 8B and 7100 steps for Scout".
- **Batch and lengths.** "We sample 48 prompts in each batch, each with 16 generations per prompt" (768 completions).
  - Generation budget 14K (12K think + 2K answer), with a 32K variant.
  - Data: Polaris-53K, "multiple epochs over the training prompts".
- **Recipe.** PipelineRL (8-step off-policy), CISPO, zero-variance filtering, and removal of prompts with pass rate at
  least 0.9. No KL, no critic.
- **Fit.** R_C − R_0 = (A − R_0)/(1 + (C_mid/C)^B) on mean@16 over 1,000 held-out prompts.
  - Table 1: ScaleRL A = 0.610, B = 1.92, C_mid = 2542; 32k A = 0.645; Scout A = 0.710; bs2048 A = 0.645.
  - Seed variance: "\pm 0.02 error margin for asymptotic performance A".
  - Design details "primarily modulate compute efficiency without materially shifting the asymptote".
- **RL vs pretraining.** "For instance, Deepseek-R1-Zero used 100,000 H800 GPU hours for RL training – 3.75% of its
  pre-training compute (Guo et al., 2025)."

My interpretation:

- **Bases.** The 8B dense base is unnamed, so its pretraining is unknown. Scout's card gives "~40T" tokens and
  "| Llama 4 Scout | 5.0M | 700 | 1,354 | 0 |" H100 GPU-hours.
- **Pre-RL stage.** RL starts after an SFT stage of "32B tokens in total" on curated reasoning traces, i.e. from a
  reasoning-SFT checkpoint, not a raw base.
- **Units.** GB200 GPU-hours (RL) and H100 GPU-hours (pretraining) are not commensurate. A ratio needs FLOP or token
  accounting, which is the executor's step.

## Evidence and limitations

- No generated-token totals or mean response lengths are reported.
- One ambiguity: "moving to batch size of 2048 prompts". Elsewhere the paper counts batch in completions (768 = 48 × 16).
- **Create-vs-elicit.** None stated. A is in-distribution mean@16, with no pass@k at large k.
- The 3.75% figure is the authors' estimate about DeepSeek (graded C). It uses R1-Zero's 100K, not the table's 147K
  total ([guo2025-deepseek-r1-compute.md](guo2025-deepseek-r1-compute.md)).

## Connections and questions

- The asymptote rises with model size (Scout) and generation length (32k). If the base drives the ceiling more than RL
  compute does, that bears on the run's RL-share question.
- Pretraining compute is not modelled: "derive predictive “scaling laws” for RL across pre-training compute" is left
  to future work.
- Open: what is the 8B base?
