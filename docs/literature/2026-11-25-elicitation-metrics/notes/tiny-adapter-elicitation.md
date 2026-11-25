---
written_on: 2026-11-25
written_by: agent:claude
papers:
  - morris2026tinylora  # proposed key; arXiv 2602.04118
  - sinii2025smallvectors  # proposed key; arXiv 2509.06608
  - sinii2025biasonly  # proposed key; arXiv 2505.18706
  - donowayQuantifyingElicitationLatent
---

# How few trained parameters reproduce a post-training gain: TinyLoRA, RL-trained steering vectors, random-parameter elicitation

Papers:
- Morris, Mireshghallah, Ibrahim, Mahloujifar, "Learning to Reason in 13 Parameters", arXiv 2602.04118v1. Full text in part:
  a fetch tool extracted §3, §4.0.2, §6.2–6.4, §8 and §9.
- Sinii et al., "Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors", arXiv
  2509.06608v4. Full text in part: abstract, §2–3, §5, §7–9, Tables 2–3.
- Sinii et al., "Steering LLM Reasoning Through Bias-Only Adaptation", arXiv 2505.18706v3. Abstract only.
- Donoway et al., "Quantifying Elicitation of Latent Capabilities in Language Models", NeurIPS 2025. Abstract only; it is in
  `references.bib`.

## Learnings

What the papers claim:
- **TinyLoRA** (§4.0.2) keeps LoRA-XS's frozen truncated SVD and replaces the trainable r×r matrix with a small vector that scales fixed random
  matrices. Tying across modules lets it go down to a single parameter.
  - Results on Qwen2.5-7B-Instruct with GSM8K as the task (§6.2): GRPO reaches 91 % at 13 parameters and 95 % at 120, against a 76 % base. SFT reaches 83 % at 13 and 84 % at 120.
  - The abstract: "models trained using SFT require 100-1000x larger updates to reach the same performance".
  - With 196 parameters, the six-benchmark mean rises from 40.3 to 53.2, against 55.2 for full fine-tuning (§6.3). The baseline rows are taken from prior work, not rerun.
  - Their stated limits: math only, and Qwen is about 10× more parameter-efficient than Llama, possibly because of contamination (§8).
  - Interpretation offered as speculation (§8): "the knowledge required to solve the task is already stored in the parameters of the model".
- **Steering vectors trained by RL** (2509.06608). One zero-initialised vector per layer is added to the residual stream and trained with RLOO,
  with the base weights frozen (§2). Mean over six math benchmarks:
  - Qwen2.5-Math-7B: 42.9 (vectors) against 43.8 (full RL) and 24.8 (base).
  - Llama-3.1-8B-It: 25.8 against 26.4 and 21.5 (Tables 2–3).

  The last-layer vector "behaves like first-token biasing at the unembedding" (§9). Prefixing "To" or "Step" recovers about 75 % of
  the last-layer gain for Qwen (§5).
- **Random-parameter elicitation** (Donoway et al. 2025 abstract). "10–100 randomly chosen parameters … can recover up to 50% of
  the performance gap", and "1,000s to 10,000s of parameters can recover 95%". A logistic curve fits gap recovery against parameter count.

Our interpretation:
- **The literature has three parameter-budget elicitation metrics:**
  - RL with a tiny budget (TinyLoRA, steering vectors);
  - SFT with a tiny budget (Donoway et al.);
  - SFT capacity thresholds (Bits That Count).

  Only the SFT versions are affordable on the 32B model. The RL versions would need 5k-token GRPO rollouts.
- **SFT overstates what RL needs.** TinyLoRA's SFT-versus-RL gap means an SFT budget measured on the final model's responses overstates the information RL needs to
  produce the same behaviour. Our capacity ladder is therefore an upper bound on "how few parameters RL needed", and it is the
  checkpoint-to-checkpoint comparison that carries the meaning.
- **Some of the gain may be tokens.** Sinii et al.'s first-token finding warns that part of a tiny adapter's gain may be a bias toward particular opening
  tokens. The token-level footprint (review item 3) can check this on Olmo.

## Evidence and limitations

- **Narrow evidence.** Everything is on 7–8B math models with short answers; nothing is on instruction following or 30B+ thinking models.
- **Not read directly.** The tool-extracted passages were not re-read by me line by line. The bias-only and NeurIPS papers are abstract only.

## Connections and questions

- [REVIEW.md](../REVIEW.md), items 1 and 9.
- [donoway2026-bits-that-count.md](donoway2026-bits-that-count.md).
