---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - matsutaniRLSqueezesSFT2025
---

# RL Squeezes, SFT Expands: trajectory counts across public RL checkpoints, long RL included

Paper: [@matsutaniRLSqueezesSFT2025] (proposed key)
Source: https://arxiv.org/abs/2509.21128v2 (v2, the latest). Ids are in `ledger/C_threads.tsv`.

## Learnings

Design. No RL is run; public Base, RL, SFT and SFT+RL checkpoints are compared (C-RLSQ-models_1p5b, _7b, _14b):
- 1.5B: Qwen2.5-Math-1.5B, its Oat-Zero RL model, R1-Distill-Qwen-1.5B, and Nemotron-Research-Reasoning-Qwen-1.5B (ProRL) as SFT+RL;
- 7B: the same pattern, with AceReason-Nemotron-7B as SFT+RL;
- 14B: Qwen2.5-14B, SimpleRL-Zoo, R1-Distill-14B and AceReason-Nemotron-14B.

The appendix adds Llama-3.1-8B and AceReason-1.1. They describe ProRL as having "prolonged RL training for 2500 steps" (C-RLSQ-prorl_steps) and AceReason as "large-scale RL on mathematics and coding tasks" (C-RLSQ-acereason). Their only training is a control SFT on s1k-1.1.

Measurement. "For each problem and model, we generate M=256 samples using a temperature of 0.6, top_p of 0.95, and a response_length of 16000." (C-RLSQ-samples).
- Unique trajectories are counted by chrF similarity with UPGMA clustering, separately for correct and incorrect outputs (C-RLSQ-method).
- Pass@k up to n = 256 appears only in figures (C-RLSQ-kmax).
- Benchmarks: AIME24, AIME25 and AMC23, plus HumanEval for the 7B models.

Verdict:
- "RL compresses incorrect trajectories, whereas SFT expands correct ones." (C-RLSQ-verdict). This holds for RL started "from either Base model or SFT model", which covers ProRL and AceReason (C-RLSQ-verdict_both).
- "Crucially, we also observe a reduced number of correct trajectories, explaining why Base models outperform RL models in Pass@ k at large k (Yue et al., 2025)." (C-RLSQ-verdict_correct).
- In the step-level graphs, RL steepens the decay of node visitation, degree and centrality "about 2.5 times", while SFT flattens it to about one third (C-RLSQ-decay).

Exceptions the paper reports:
- Llama-3.1-8B: "squeezing of incorrect paths and expansion of correct paths following RL training" (C-RLSQ-llama_expands).
- AceReason-1.1-7B expands correct paths on AIME24/25 but squeezes both kinds on AMC23 (C-RLSQ-ace11_amc, C-RLSQ-ace11_amc_squeeze).

## Evidence and limitations

Interpretation:
- Comparing different checkpoints confounds RL scale with recipe, data and base model. No run is followed across RL steps, so the paper cannot say how squeezing depends on RL length.
- *[review]* Cluster counts are not pass@k; pass@k appears only in figures. App. C.5 does not make them threshold-dependent: there the varied thresholds "yielded identical results".
- Length confound: the Qwen2.5-Math base and Oat-Zero models are evaluated at max_model_len=4096, against 16000 for the others (C-RLSQ-maxlen4096).
- AceReason-Nemotron-14B "does not show performance improvement over the pre-RL DeepSeek-R1-Distill-Qwen-14B on the AIME25 and AMC23 domains" (C-RLSQ-ace14b_caveat). Some SFT+RL comparisons are therefore near-null RL effects.
- Which ProRL revision was evaluated (v1, 2,000 steps; or v2, 3,000 steps) is not stated.

## Connections and questions

- This independently supports [wu2025-invisible-leash.md](wu2025-invisible-leash.md): long RL from distilled bases (ProRL, AceReason) still loses correct-solution diversity at 256 samples. It adds a mechanism-level contrast, SFT adding paths where RL removes them. *[review]* Unclear: the paper's general sentence supports the loss of correct-solution diversity, but its SFT+RL summary and the AceReason-14B appendix speak only of incorrect trajectories or minimal change.
- For the hypothesis: the squeeze appears both in short zero-RL runs (Oat-Zero, SimpleRL-Zoo) and in long or large-scale runs (ProRL, AceReason). *[review]* That is cross-sectional evidence, not a within-run scaling test, and it does not show that narrowing persists at the largest open RL scales tested: the paper reports Llama-3.1-8B and AceReason-1.1-7B expanding correct paths. See [liu2025-prorl-compute.md](liu2025-prorl-compute.md).
