---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - yuDAPOOpenSourceLLM2025
---

# DAPO-32B: what is and is not disclosed about its RL compute

Paper: [@yuDAPOOpenSourceLLM2025] (proposed key; the verl recipe repo is [@tongDAPORecipe2025])
Source: arXiv 2503.14476v2 (20 May 2025). Also the `BytedTsinghua-SIA/DAPO` README, the HF card `DAPO-Qwen-32B`,
dapo-sia.github.io, and the current `verl-project/verl-recipe` `dapo/run_dapo_qwen2.5_32b.sh`. All fetched 2026-10-19.
Ledger rows: `A-DAPO-*`.

## Learnings

What the sources state:

- **Base and checkpoint.** The base is Qwen2.5-32B *base*, pretrained on 18T tokens. The released `DAPO-Qwen-32B` is the
  full-DAPO run, which "achieves 50 points on AIME 2024 using Qwen2.5-32B base model".
- **Batch.**
  - "For rollout, the prompt batch size is 512 and we sample 16 responses for each prompt."
  - "For training, the mini-batch size is set to 512, i.e., 16 gradient updates for each rollout step."
- **Lengths and KL.** Generation is capped at 20,480 tokens (16,384 plus a 4,096 soft-punish cache). "Therefore, we will
  exclude the KL term from our proposed algorithm."
- **Data.** "DAPO-Math-17K dataset, which consists of 17K prompts". The HF file has 1,791,700 rows (duplicated prompts).
- **Hardware.** "We utilize verl to perform DAPO training using the Qwen-32B-Base model from scratch with 128 H20 GPUs."
  No wall-clock time is given.
- **Dynamic sampling.** DAPO drops groups whose accuracy is 0 or 1. "The sampling cost for each batch is dynamic." The
  factor is not quantified. The recipe draws generation batches of 3 × 512 prompts, at most 10 per step.
- **Conclusions.**
  - Efficiency: the run beats R1-Zero-Qwen-32B "using 50% training steps".
  - Qualitative, from a case study only: RL "gradually gives rise to entirely new modes of reasoning that were initially
    absent". This leans towards "create". Yue finds the base overtakes DAPO-32B on AIME24 at large k (to 1024).

## Evidence and limitations

- **The step count is the key open quantity.** Fig. 1 plots "Training Steps" from 0 to 10,000, and the caption says
  "The x-axis represents the gradient update steps." The DAPO-run metrics in Fig. 7 span 0–5,000. With 16 updates per
  rollout step, the rollout-step count is either the axis value or 1/16 of it. Rollout steps are what set generation compute.
  - [interp.] Under the current recipe (`total_epochs=1` over 1,791,700 rows, at least 1,536 prompts per step), at most
    about 1,166 rollout steps fit. That favours the gradient-step reading.
  - The recipe copy may differ from the paper's run: the branch the README links returns 404.
- **Mean response length appears only as a figure axis.** Fig. 7(a) spans 1,000–4,000 tokens.
- **The over-generation factor for this run is unknown.** Yue's 3–6× figure comes from its own 7B VeRL runs.

## Connections and questions

- Related: [SimpleRL-Zoo note](zeng2025-simplerl-zoo.md) and [Magistral note](rastogi2025-magistral.md). Magistral adopts
  DAPO's Clip-Higher and soft length penalty. Yue: [@yueDoesReinforcementLearning2025].
- Open: settle the step unit from the verl wandb training record linked in the README. It was not machine-readable here.
- Open: wall-clock on 128 H20s.
