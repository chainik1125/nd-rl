---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Negative sample reinforcement (NSR) and W-REINFORCE preserve the pass@k spectrum
Paper: The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning, Xinyu Zhu et al., 2025
Source: https://arxiv.org/abs/2506.01347v2 (v2, reviewed)

## Learnings
- Decomposition (§2.2, Eq. 2–4): with r ∈ {−1, +1}, the RLVR objective = L_PSR (raise π on correct samples) + L_NSR (lower π on incorrect samples).
- W-REINFORCE (§5, Eq.): `λ·L_PSR + L_NSR`, with λ = 0.1. λ = 1 is REINFORCE and λ = 0 is NSR (App. E).
- Setup (§3.1): Qwen2.5-Math-7B, Qwen3-4B (non-thinking), Llama-3.1-8B-Instruct; MATH 7,500 problems; 8 rollouts per prompt; eval with 256 samples (64 for Qwen3-4B), k ∈ {1…256}.
- Table 1 (§5; Qwen2.5-Math-7B, pass@1 / pass@256):
  - MATH: base 63.2 / 96.9, PPO 76.6 / 96.3, GRPO 76.3 / 95.5, PSR 74.1 / 91.2, NSR 75.7 / 96.9, W-REINFORCE 76.6 / 96.7.
  - AIME 2025: base 6.1 / 46.7, GRPO 10.3 / 50.0, PSR 11.6 / 43.3, NSR 10.0 / 53.3, W-REINFORCE 10.6 / 56.7.
  - AMC23: base 41.0 / 100.0, NSR 60.9 / 100.0.
- Fig. 4 (Llama-3.1-8B-Instruct): "All the methods underperforms the base model, while NSR retains the most performances"; "Pass@256 drops substantially after RL training" (§3.2).
- Mechanism claim (§4.2, token-level gradient analysis): NSR "suppress[es] incorrect generations and redistribut[es] probability mass toward other plausible candidates, guided by the model's prior beliefs".
- App. E: "When λ=1, Pass@256 drops substantially". Limitations: long NSR training degrades performance.

## Evidence and limitations
- NSR/W-REINFORCE mostly preserves base pass@256 rather than beating it. MATH NSR 96.9 = base 96.9. The only large-k gains over base are on AIME25 (30 problems: +6.6 and +10.0 pp = 2 and 3 problems). No seed counts checked.
- The gains depend on the prior: they help Qwen, while all methods hurt Llama at pass@256.

## Connections and questions
- B2. This is the only objective in the cluster whose loss (Eq. 4) gives a nonzero gradient on groups where every sample fails, because it penalises failures without a group baseline. I did not verify from the code or text that their implementation keeps this with no normalisation (unverified). For us, this means hard or unreached theorems still receive updates, which could push mass toward alternative moves. With a weak 3.2M prior it could equally just spread mass.
- EI analogue: EI is pure PSR (SFT on successes). The paper's PSR row (pass@256 falls below base on MATH and AIME) is the prediction for EI sharpening. That contrasts with our observation that EI expands support, which is worth stating.
- Smallest test: W-REINFORCE (λ=0.1) vs GRPO vs EI-matched attempts from the same checkpoint. Support curve at 400k and the base-unreached set.
