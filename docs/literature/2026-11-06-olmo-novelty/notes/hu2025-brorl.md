---
written_on: 2026-11-07
written_by: "agent:claude"
papers:
  - huBroRLScalingReinforcement2025
---

# BroRL: wider rollouts restart pass@1 gains on a plateaued ProRL v2 checkpoint; no reach test

Paper: [@huBroRLScalingReinforcement2025] (proposed key; first author Jian Hu, NVIDIA)
Source: arXiv 2510.01180v1, the only version (checked 2026-11-07); html text.

## Setup card

| | BroRL |
|---|---|
| Model, start | 1.5B; the public ProRL v2 checkpoint (3,000 RL steps at 8,192 tokens) [PR-96, PR-97], itself from distilled R1-Distill-Qwen-1.5B |
| RL share of pretraining | 1.0 × 10⁻² – 4.5 × 10⁻² FLOPs cumulative (rl-compute-share row "ProRL v2 + BroRL") |
| RL steps | + 107 to + 191 steps at 512 rollouts per prompt [PR-98, PR-142, PR-100]; 64 H100s [PR-99] |
| Domains | trains math, code, science, IFEval, Reasoning Gym [PR-96]; evaluates math, code, Reasoning Gym [PR-106] |
| Largest k | 1 (pass@1 from 16 samples) [PR-105]; pass@k only in theory [PR-107] |
| Grading | not described beyond the evaluation protocol (UNVERIFIED) |
| Controls | equal training compute vs ProRL continuation, paired t-test (yes) [PR-103]; format no; guessing no; contamination no |

## Learnings

**Observed.**
- Saturation: BroRL states that step scaling "plateaus after thousands of steps" and that the 3,000-step ProRL v2 checkpoint "had reached such a plateau" [PR-94, PR-127]. From that checkpoint (61.69 math, 52.00 code, 61.29 Reasoning Gym [PR-128]), continuing at 16 rollouts lifts math to 62.08 and Reasoning Gym to 62.10, then lowers them to 62.02 and 61.45 [PR-141, PR-101].
- Rollouts per prompt rise from 16 to 512 [PR-98]. "Revives" means continued pass@1 gains [PR-95]: the +191-step arm reaches 63.03 / 54.20 / 63.09 in 173.8 h [PR-100, PR-102]. The released checkpoint (HF branch) has 419 extra steps, not the paper's arms [PR-148].
- At equal compute the per-instance pass@1 advantage is Δ = 0.0033 (paired t-test over > 10,000 instances, one-tailed p = 6.5 × 10⁻⁷), measured after only ≈ 100 extra steps [PR-103, PR-104, PR-138].
- Pass@k: a one-step mass-balance argument says larger N "directly improves pass@k" [PR-107]; no pass@k beyond k = 1 is measured, at any k [PR-105].
- IF: IFEval is a training family [PR-96], absent from the evaluation [PR-106].
- On some benchmarks neither method gains; N = 512 may be "not large enough" [PR-108]; only N = 16 vs 512 was tried [PR-109].

**Paper interprets.** ProRL's plateau is "not fundamental" but an artefact of too few rollouts; width per prompt is a second scaling axis [PR-129].

**Our interpretation.** BroRL measures accuracy, not reach: a 1–2 point pass@1 change on averages says nothing about problems the base never solves, so it neither supports nor weakens ProRL's boundary claims. It does show that, by ≈ 3K steps, more of the same steps stopped paying on pass@1, which caps how much "prolonged" alone can be credited. Olmo 3's RL uses 8 rollouts per prompt (rl-compute-share table, Olmo 3 rows), far below BroRL's 512; BroRL predicts narrower exploration per prompt for them but tests no boundary.

## Evidence and limitations

Read in full (main text, theory, appendix tables). One run per arm and no training seeds reported, so Δ = 0.0033 has no between-run error bar; the theory assumes a single RLVR step and is checked in a token-level simulator [PR-139, PR-140].

## Connections and questions

- C3: **unrelated**: trains on IFEval-type data but reports no IF result [PR-96, PR-106].
- C7: **extends**: rollouts per prompt is a compute axis our checkpoint trajectory holds fixed [PR-98].
- C1, C2: **unrelated**: pass@1 only [PR-105].
- See [liu2025-prorl-v1-full](liu2025-prorl-v1-full.md) and [nvidia2025-prorl-v2-blog](nvidia2025-prorl-v2-blog.md).
