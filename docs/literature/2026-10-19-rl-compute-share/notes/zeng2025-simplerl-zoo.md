---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - zengSimpleRLZooInvestigatingTaming2025
---

# SimpleRL-Zoo: small zero-RL runs on open base models, seen as compute

Paper: [@zengSimpleRLZooInvestigatingTaming2025] (proposed key)
Source: arXiv 2503.18892v3 (6 Aug 2025), plus repo `hkust-nlp/simpleRL-reason`, branch `v1` (README and
`train_grpo_math_tune_ray.sh`). All fetched 2026-10-19. Ledger rows: `A-SRZ-*`.

## Learnings

What the paper and repo state:

- **Recipe.** GRPO from the base model, correctness-only reward. The data are GSM8K/MATH difficulty bands
  "with each category containing roughly 8,000 problems". The Qwen2.5 models train on MATH lv3–5; Llama-3.1-8B trains on
  GSM8K + MATH lv1.
- **Shared hyperparameters.** "We use a prompt batch size of 1,024 and generate 8 rollouts per prompt, with a maximum
  rollout length of 8,192 tokens." The mini-batch is 256 and the KL loss is 1e-4 (≤14B) or 1e-3 (32B).
- **Compute, in the README only.** "we utilized 2x8 H100-80G GPUs to train 7B and 14B models for approximately 100
  steps over 15 hours using 8K examples." The 32B run took 1.5 days on 8x8 H100-80G. No GPU time is given for Llama.
- **Bases.** Qwen2.5 was pretrained on 18T tokens. Llama-3.1-8B was pretrained on ~15T tokens, using 1.46M H100-80GB
  GPU-hours. The executor computes the ratios.
- **Conclusion.** For Mistral-Small-24B, the RL model's pass@k lead persists to k = 128 on AIME24 + AMC23. The authors
  read this as "not just reranking": "it enhances the model’s fundamental reasoning abilities". This leans towards
  "create". Yue reaches the opposite conclusion on the Qwen and Llama Zoo checkpoints.

My interpretation:

- [interp.] Under verl semantics, a mini-batch of 256 out of 1,024 prompts gives 4 updates per rollout step.
- [interp.] About 100 steps covers roughly a dozen passes over ~8K problems. The epoch cap `TOTAL_EPOCHS=20` is not
  overridden in the README commands.

## Evidence and limitations

- **Steps per model are not stated.** The figure x-axes run 0–100 for Llama-8B and Qwen-7B and 0–150 for Qwen-14B and
  Qwen-32B (axis ticks, Figs. 1 and 11). The README's "approximately 100 steps" covers 7B and 14B together, which
  conflicts with the 14B axis. The HF cards hold only a license, so each released checkpoint's step is unknown.
- **Training response length is not reported.** Fig. 1 plots eval-time length over six benchmarks. The axis ranges are
  0.5–2.0K (Llama), 0.7–1.0K (Qwen-7B) and 0.5–0.8K (Qwen-32B). These are ranges, not means, and far below the 8,192 cap.
- **The paper's pass@k evidence is on another model.** It uses Mistral-Small-24B (k ≤ 128), which Yue did not test.
  Yue evaluates these checkpoints with k up to 128 or 1024.
- **Contamination flag, raised by the paper itself.** Qwen2.5 bases "already exhibit instruction-following abilities and
  reflective behaviors". This matches the Qwen2.5 report's use of instruct-model synthetic data in pretraining.

## Connections and questions

- Related: [DAPO note](yu2025-dapo.md) and [Magistral note](rastogi2025-magistral.md). Yue: [@yueDoesReinforcementLearning2025].
- Open: which step are the released `hkust-nlp/*-SimpleRL-Zoo` checkpoints? For 14B, 100 vs 150 steps matters.
- Open: the Llama-3.1-8B run's GPU time.
