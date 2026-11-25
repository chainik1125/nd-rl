---
written_on: 2026-11-25
written_by: agent:claude
papers:
  - heo2025instructionfollowingdimension  # proposed key; arXiv 2410.14516
  - agrawal2025gepa  # proposed key; arXiv 2507.19457
---

# Instruction-following-specific elicitation: a compliance direction in activations (Heo et al.) and prompt optimisation on IFBench (GEPA)

Papers:
- Heo et al., "Do LLMs 'know' internally when they follow instructions?", arXiv 2410.14516v5, ICLR 2025. Full text in part:
  tool-extracted §2–4, §6.3 and Tables 1 and 4.
- Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", arXiv 2507.19457v2, ICLR 2026
  oral. Full text in part: the setup and Tables 1–2. Appendix E.4 (GRPO settings) was not seen.

## Learnings

What the papers claim:
- **Heo et al.**
  - **Setup:** LLaMA-2-7B and 13B chat, Mistral-7B-Instruct-v0.3 and Phi-3-mini. A linear probe on the first-token representation before the response,
    with IFEval checker labels.
  - **Within instruction types:** AUROC on unseen tasks is 0.77 / 0.83 / 0.74 / 0.88.
  - **Across instruction types:** unseen types give "ranging from 0.50 to 0.55, close to chance" (Table 1).
  - **Steering:** adding α·D at the last layer raises success from 0.57 to 0.59, 0.61 to 0.65, 0.58 to 0.64 and 0.71 to 0.74. Random directions give 0.55, 0.54,
    0.56 and 0.63 (Table 4).
  - **What the direction tracks:** it "is more closely related to the phrasing of prompts rather than the inherent difficulty of the task" (abstract).
- **GEPA.**
  - **Method:** module prompts are evolved with the weights frozen.
  - **IFBench, Qwen3-8B** (Table 1): baseline 36.90, GRPO (24,000 rollouts, LoRA) 35.88, MIPROv2 36.22, GEPA 38.61. GEPA "found optimal prompts after just 678 rollouts".
  - **IFBench, GPT-4.1-mini** (Table 2): 47.79 baseline, 55.95 with GEPA plus Merge.

Our interpretation:
- **Both are small effects on IF.**
  - Heo's direction raises compliance by a few points and does not generalise across instruction types.
  - GEPA moves IFBench by 2–8 points, and GRPO with LoRA on 24,000 rollouts did not help Qwen3-8B there.
- **What we expect on Olmo.** A probe or steering metric and a prompt-optimisation metric would most likely show small elicitable shares at `dpo`. That is a
  plausible but low-information outcome, which is why they rank 5 and 7 in the review.
- **Why the probe could still be worth running.** At about 0.1 H200-h per checkpoint for activations, it is cheap if run alongside other work.

## Evidence and limitations

- **Smaller, older models.** All of Heo's models are 3.8–13B chat models from 2023–24, not thinking models.
- **Configuration not seen.** GEPA's GRPO configuration was not checked (App. E.4).

## Connections and questions

- [REVIEW.md](../REVIEW.md), items 5 and 7.
- Dan's `olmo-ifladder` (hand-written instruction ladder) and `olmo-iffu` (60-word hint) are the manual counterparts to GEPA.
