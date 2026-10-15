---
written_on: 2026-09-08
written_by: agent:codex
---

# Experiment summaries

- [What the GRPO wall was: model diffing depth-3 boxes](2026-10-14-grpo-wall-model-diff/README.md): done 2026-10-15; pretrained models close box runs with a max(0, 2 − r) shortcut, successful RL re-weights blocks 0-2, crosscoders find no new features, a diff-SAE gives a 20-latent handle.
- [R1-style scratchpad reasoning sprint](2026-09-08-r1-cot/README.md): active September 8–9, 2026.
- [Literature for learning beyond initial training data](2026-09-13-astra-literature-review/README.md): completed reading recommendations and experimental-design critique; no new experiment results.
- [Combined pipeline](2026-09-28-combined-pipeline/README.md): Dan's data + 186 pretraining + Leon's RL + guided sampling, 687 → 904 dev theorems at ≥ 7 lines inside the 2,400 s budget; September 27–28, 2026.
- [The combined model, and where its gains come from](2026-09-28-combined-model/README.md) (blog draft): 2×2×2 factorial of data format × pretraining × RL, 3 seeds; naive 142 → best 904 dev theorems, pass@k on held-out and textbook problems; September 29, 2026.
- [Retrying with Lean errors](2026-10-08-retry-ei/README.md): after a failed proof the model sees a Lean error code and the failing line and tries again. Retries are accepted as often as independent samples or less; every variant proves fewer dev theorems than plain expert iteration (771); October 8, 2026.
