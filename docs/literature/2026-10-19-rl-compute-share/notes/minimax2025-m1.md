---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - minimaxMiniMaxM1ScalingTestTime2025
---

# MiniMax-M1: RL GPU-weeks against a 456B/45.9B base (MiniMax-Text-01 + 7.5T CPT)

Paper: [@minimaxMiniMaxM1ScalingTestTime2025] (proposed key; author "MiniMax").
Source: arXiv 2506.13585v1, the only version on 2026-10-19. The base model's own report is MiniMax-01, arXiv
2501.08313v1. Ledger: `B-M1-*`, `B-TXT01-*`.

## Learnings

What the papers state:

- **Model.** M1 "comprises 456 billion parameters in total, with 45.9 billion activations and 32 experts", built on MiniMax-Text-01.
- **Base pretraining (Text-01).** Phases: "constant learning rate for 7.2T tokens", "the remaining 3.2T tokens", then a
  1T decay. Long-context stages add 300B + 32B + 26B tokens.
- **Continued pretraining for M1.** "we continue training the MiniMax-Text-01 model with additional 7.5T tokens".
  - The mix is 70% STEM/code/book/reasoning, "while strictly avoiding the use of synthetic data".
  - A long-CoT SFT stage follows; its source is not stated.
- **RL compute.** "Combining hybrid-attention and CISPO enables MiniMax-M1’s full RL training on 512 H800 GPUs to
  complete in only three weeks, with a rental cost of just $534,700."
- **RL setup.** CISPO with the GRPO group advantage, plus DAPO dynamic sampling and length penalty; "There is no KL
  penalty term".
  - Data: ~50K math, ~53K synthetic logic, 30K code, several thousand SWE, 25K general.
  - Generation limits: 40K, then a staged extension to 80K.
- **Lengths.** Eval responses "exceed 20,000 tokens".
  - Fig. 4 step axis: ticks 0–4000.
  - Prompts per step, group size and epochs are not given.

My interpretation:

- **Pretraining compute.** It is given only as cluster size: "the number of H800 GPUs ranges from 1500 to 2500". A ratio
  needs FLOP models for both stages (tokens × active parameters; 512 H800 × 3 weeks × assumed utilisation). That is the
  executor's step.
- **Token totals (my arithmetic, not stated).** The phases sum to 11.4T tokens. With 0.358T of long-context data and
  7.5T of CPT, M1's base saw about 19.3T tokens.

## Evidence and limitations

- **Cost scope.** It is not stated whether "full RL training" covers the 40K stage, the 80K extension, or both. I read
  them as one lineage. The dollar figure is a rental equivalent.
- **Create-vs-elicit.** None stated, and no large-k pass@k. The CISPO comparison on Qwen2.5-32B-base reports pass@1
  against steps only.
- **Contamination.** The CPT up-weights reasoning data and extracted QA pairs. Where the SFT CoTs come from is undisclosed.

## Connections and questions

- ScaleRL ([khatri2025-scalerl.md](khatri2025-scalerl.md)) adopts CISPO.
- DeepSeek-R1 ([guo2025-deepseek-r1-compute.md](guo2025-deepseek-r1-compute.md)) also reports 512 H800 for RL (~198 h,
  against M1's "three weeks").
- Open: GPU-hours for the 7.5T CPT.
