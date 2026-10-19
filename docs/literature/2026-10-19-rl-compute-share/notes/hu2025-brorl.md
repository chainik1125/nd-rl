---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - huBroRLScalingReinforcement2025
---

# BroRL: scaling rollouts per prompt after ProRL's step-scaling plateau

Paper: [@huBroRLScalingReinforcement2025] (proposed key)
Source: https://arxiv.org/abs/2510.01180v1 (the latest version) and the `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` model card. Ids are in `ledger/C_threads.tsv`.

## Learnings

Paper claims:
- Step scaling saturates: "performance plateaus after thousands of steps, with clear diminishing returns from allocating more computation to additional training."
- Rollout width revives the model: "BroRL revives models saturated after 3K ProRL training steps".

Setup:
- Start: the public ProRLv2 checkpoint (3,000 steps, 8,192-token context); continued training at 16,384 context on 64 H100s (C-BRORL-base, C-BRORL-hardware).
- Algorithm: the ProRLv2 recipe (clipped PPO, REINFORCE++-style normalisation, dynamic sampling, clip-higher, reference resets, truncated importance sampling); learning rate scaled by the square root of the batch increase (C-BRORL-algo, C-BRORL-lr).

Table 1 (samples before dynamic-sampling filtering; wall-clock hours on 64 H100s):
- ProRL continuation (N=16, 512 prompts/step):
  - +225 steps: 4.39M samples, 56.3 h.
  - +535 steps: 10.44M samples, 133.8 h. Math falls back from 62.08 to 62.02 (C-BRORL-prorl535_row).
- BroRL (N=512, 128 prompts/step):
  - +107 steps: 98.1 h.
  - +191 steps: 20.04M samples, 173.8 h. Math 63.03, Code 54.20, Reasoning Gym 63.09, against a 61.69 / 52.00 / 61.29 baseline (C-BRORL-brorl191_row, C-BRORL-baseline_row).
- The released BroRL checkpoint trained for 419 steps (model card; C-PRORL2-brorl_mc).

Where the compute goes:
- "The primary performance bottleneck in training models for long Chain-of-Thought (CoT) reasoning via RLVR is the sample generation phase" (C-BRORL-genbottleneck).
- Going from N=16 to N=512 raises generation throughput from 36.5 to 72.4 samples/s and the share of samples kept after dynamic sampling from 41% to 62% (C-BRORL-throughput, C-BRORL-throughput2).

## Evidence and limitations

- No large-k test. Evaluation is "pass@1 with a 32k context length, averaged over 16 independent samples per instance" (C-BRORL-eval). The paper's boundary language comes from a one-step theory and a token-level simulation, in which, at very large N, "RLVR eliminates knowledge shrinkage entirely" (C-BRORL-shrinkage_sim). Nothing here tests create-vs-elicit at large k.
- Effects are small. At equal compute (~140 h) a paired t-test gives Δ = 0.0033 in pass@1 (C-BRORL-ttest), and the authors themselves expect gains "roughly log-linearly with training time" (C-BRORL-ttest_steps).
- Some benchmarks did not improve at all, "suggesting that N=512 might not be large enough for some of the harder problems" (C-BRORL-third_pattern).
- Single lineage: a 1.5B model whose base is distilled from R1 (see [liu2025-prorl-compute.md](liu2025-prorl-compute.md)).

## Connections and questions

- For the hypothesis: BroRL is the clearest within-lab statement that more steps alone stop paying off on this model after about 3k steps. Its lever is exploration width per update, so steps or FLOPs are not the only RL-scale axis.
- Shared author Fang Wu leads Invisible Leash ([wu2025-invisible-leash.md](wu2025-invisible-leash.md)), which re-tests ProRL v1/v2 at k up to 16384 and finds net shrinkage. No study here evaluates the BroRL checkpoint at large k (the cheapest open check).
- Generation dominates RL runtime (C-BRORL-genbottleneck; over 90% per APRIL, C-APRIL-rollout90), which bears on whether to count training FLOPs or total FLOPs.
