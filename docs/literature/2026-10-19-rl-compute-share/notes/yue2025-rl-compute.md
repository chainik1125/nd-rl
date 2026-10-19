---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - yueDoesReinforcementLearning2025
---

# Yue et al.: how much RL the "elicitation" result rests on

Paper: [@yueDoesReinforcementLearning2025]
Source: https://arxiv.org/abs/2504.13837v5 (v5, 24 Nov 2025, the latest version; read in full, HTML text plus the PDF
text for figure axes). This note extends `../../2026-10-02-lit-review-2/notes/yue2025-doesrl-pass-at-k.md` (the pass@k
standard) with the compute side and does not repeat it.

## Learnings

**What the paper states:**
- **Models.** Math uses SimpleRL-Zoo on LLaMA-3.1-8B and Qwen2.5-7B/14B/32B, plus Oat-Zero-7B and DAPO-32B. Code uses
  CodeR1-Zero-7B and DeepCoder-14B; vision uses Qwen2.5-VL-7B trained with EasyR1. The deep analysis uses its own verl
  runs, and the scaling check uses Magistral-Medium (Table 1; §3–4.6) [Y-yue-models-math, Y-yue-models-deep].
- **Its own runs** use "a prompt batch size of 256 and generate 8 responses per prompt", a maximum of 8,192 tokens, no KL
  and 2,000 training problems (§4.3) [Y-yue-verl-batch, Y-yue-verl-maxlen, Y-yue-verl-kl, Y-yue-verl-train].
  - The runs last 150, 300 and 450 steps (Table 4) [Y-yue-verl-step150, Y-yue-verl-step450], and n = 32 runs 220 steps
    [Y-yue-verl-n32].
  - The response-length axis of Fig. 19 spans 400–1,200 tokens [Y-yue-verl-meanlen].
- **Steps and boundary.** From step 150 to step 450, pass@1 on the training set rises from 26.1 to 42.5 while pass@256
  falls from 66.3 to 64.3 (Table 4).
- **The authors leave the scale question open:** "We leave the question of whether scaling RLVR training can eventually
  surpass the base model to future investigation" (§4.4) [Y-yue-n-open]. Also: "Whether this trend persists as more
  compute, such as pre-training scale budgets, is dedicated to RL training remains a critical question" (§4.6)
  [Y-yue-open-question].

**Our computation** (`../compute_table.csv`; method in `../REVIEW.md`):
- **Ratios.** RL / pretraining FLOPs per model run from 7 × 10⁻⁶ to 3 × 10⁻³; the median geometric midpoint is
  1.2 × 10⁻⁴. The range reaches 3 × 10⁻² only under DAPO's rollout-step reading.
  - Yue's own runs: 1.1 × 10⁻⁵ to 2.0 × 10⁻⁴.
  - SimpleRL-Zoo: 4.6 × 10⁻⁵ to 1.1 × 10⁻³.
  - DAPO-32B: 2.4 × 10⁻⁴ to 3.0 × 10⁻². Its step count is ambiguous by 16×; the caption's gradient-step reading caps it
    at 1.9 × 10⁻³.
- **Gap to the frontier.** That is 3–4 orders of magnitude below the frontier estimate for H2 2025–2026 (central
  0.3–2, grade D) and 2–3 below 2024's (0.01–0.1, grade C).
- **Exploration volume.** Each training problem is sampled 154 (150 steps) to 901 (n = 32) times. That is the same
  order as the evaluation k (128–1,024).

## Evidence and limitations

- **Magistral-Medium is the only near-frontier model** (§4.6, Fig. 9: "little or no improvement at higher k"
  [Y-yue-magistral-result]). Its base size, pretraining and RL compute are all undisclosed, so its ratio cannot be
  computed. It starts from Mistral Medium 3 *Instruct* [A-MAG-start].
- **Unreported RL configurations.** Yue gives no configuration for the EasyR1 run (EasyR1 defaults assumed) and no
  step counts for Oat-Zero or DeepCoder (bounded from GPU-hours and wall-clock).
- **Bases are not clean.** Code, vision and Magistral start from instruct models. Qwen2.5 pretraining includes
  synthetic maths data from Qwen2(-Math)-72B-Instruct [A-Q25-synth]. DeepCoder starts from a model distilled from R1.
- **Attention FLOPs are ignored on both sides.** That undercounts long-context RL relative to pretraining. For a 32B
  dense model, attention adds ≈ 16 % to the forward FLOPs at 8K tokens of context, and the extra grows linearly with
  context (my estimate). At the 16–32K lengths of the largest runs that is at most a factor of ≈ 2, far below the
  10²–10⁴ gaps that matter here.

## Connections and questions

- The paper itself concedes the brief's hypothesis. Its results answer "small RL elicits"; the paper does not claim
  that larger RL does.
- Our create-or-elicit ladders run at 3–10× pretraining FLOPs (`../REVIEW.md` Q5). That is the regime Yue leaves open,
  and in our organism it still gives "needs exposure" for a pattern absent from pretraining. Interplay's exposure
  ablation (2512.07783v1) agrees at ≈ 0.4× pretraining.
- Next: plot equal-k and large-k reach against cumulative RL FLOPs along our existing ladders, which span ratios of
  ≈ 0.2–10 (`../REVIEW.md`, implication 2).
