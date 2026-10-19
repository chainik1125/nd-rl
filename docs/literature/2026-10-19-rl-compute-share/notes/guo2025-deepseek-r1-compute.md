---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - guoDeepSeekR1IncentivizesReasoning2025
---

# DeepSeek-R1: RL compute next to DeepSeek-V3 pretraining compute

Paper: [@guoDeepSeekR1IncentivizesReasoning2025]
Source: Nature 645, 633–638 (2025), doi:10.1038/s41586-025-09422-z, with its Supplementary Information (MOESM1 PDF).
arXiv 2501.12948v2 (4 Jan 2026) carries the same supplementary text. Base-model compute is from arXiv 2412.19437v2.
Ledger: `B-R1-*`, `B-V3-*`.

## Learnings

What the paper states:

- **R1-Zero.** GRPO on DeepSeek-V3-Base (671B total / 37B active, 14.8T tokens).
  - Schedule: "with training continuing for a total of 10,400 steps, corresponding to 1.6 training epochs".
  - Batch: "Each training step consists of 32 unique questions, resulting in a training batch size of 512".
  - 16 samples per question, capped at 32,768 tokens, then 65,536 after step 8.2k.
  - "each rollout generates 8,192 outputs, which are randomly split into 16 mini-batches".
  - KL 0.001, reference reset every 400 steps; no critic.
- **Compute (SI §2.4.4).**
  - "For the training of DeepSeek-R1-Zero, we employed 64*8 H800 GPUs, and the process required approximately 198 hours."
  - R1 used the same GPUs for about 80 h.
  - "in H800 GPU Hours | 101K | 5K | 41K | 147K" (R1-Zero, SFT data, R1, total), i.e. "$294K" at $2/GPU-hour.
- **R1 pipeline.** Cold-start SFT, RL stage 1 (32,768 tokens; step count not given), ~800k SFT samples, RL stage 2 (1,700 steps).
- **Base compute (V3 Table 1).** "in H800 GPU Hours | 2664K | 119K | 5K | 2788K". The V3 report adds: "The subsequent
  training stages after pre-training require only 0.1M GPU hours."

My interpretation:

- Both tables count H800 GPU-hours for official runs only. That makes this the cleanest primary pair; the ratio is
  left to the executor.
- Secondary sources mix scopes. ScaleRL puts 100K against pre-training (3.75%); Lambert puts 147K against "2.8M" (~5%).

## Evidence and limitations

- **Uncosted or unclear scope.** The A100 preparatory runs (30B model) are not costed. "$294K" is a rental equivalent.
  Whether the 41K for R1 includes its SFT runs is not stated.
- **Missing quantities.** Mean response length appears only in Fig. 1(b) (axis 0–20,000), with no token totals. The
  R1-Zero prompt subset is not separated from the pool (math 26K, code 17K, STEM 22K, logic 15K, general 66K).
- **Create-vs-elicit.** There is no test.
  - The paper says "pre-trained checkpoints inherently possess substantial potential for complex reasoning tasks".
  - It says the base can "generate plausible solution candidates, from which reinforcement learning can effectively
    identify and optimize high-quality outputs".
  - The largest k is R1's own Pass@64 (90.0% vs Pass@1 79.8%, AIME 2024), with no base curve.
- **Contamination.** V3-Base web data contain "a significant number of OpenAI-model-generated answers". R1 (not R1-Zero)
  also trains on cold-start and R1-Zero-derived SFT data.

## Connections and questions

- [khatri2025-scalerl.md](khatri2025-scalerl.md) quotes the 3.75% figure.
- DeepSeek-V3.2 says its RL budget "exceeds 10% of the pre-training cost".
- Open: do the 101K GPU-hours include evaluation and restarts?
