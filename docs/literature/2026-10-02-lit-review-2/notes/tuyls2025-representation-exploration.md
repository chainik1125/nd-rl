---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - tuyls2025repexp
---

# Representation-based exploration (RepExp) for language models

Paper: [@tuyls2025repexp]
Source: https://arxiv.org/abs/2510.11686v2 (v2 reviewed; html text)

## Learnings
- Question: does RL "promote the discovery of novel behaviors, or simply sharpen those already present in the base model" (Abstract).
- Method: an elliptical (count-like) bonus on the pre-trained model's hidden states. Responses are represented by the averaged token hidden states h̄(x,y). The bonus is h̄ᵀΛh̄, with the inverse covariance updated by the Woodbury identity (method section, Alg. 1). Averaging all tokens is "over 2x more sample efficient" than the last token alone (Fig. 4). In post-training, the bonus is computed against the other rollouts in the batch, with a fresh random projection at each step (post-training section, before Fig. 9).
- Inference time: "over 50% improvement in verifier efficiency" for Qwen-2.5-14b-Instruct on GSM8K, MATH, MBPP+ and Game-of-24 (§1 and RF1).
- RF2, "The benefits of RepExp grow with model strength": "weaker models (e.g., Qwen-2.5-0.5B) experience no benefit or even degradation" (§4.1, RF2).
- RF6 (post-training, Fig. 2): "standard GRPO degrades performance relative to the base model for large values of k", while "policies fit using RepExp preserve or improve pass@k for large values of k".
- Headline: "our post-trained Qwen-2.5-7b-Instruct's pass@80 matches the pass@256 of GRPO on the same model" on AIME 2024 (Abstract).
- RF7 / Fig. 9: under the base model's log-likelihood, RepExp responses "tend to be less likely under the base model", while GRPO shifts mass toward higher base likelihood ("sharpening").

## Evidence and limitations
- **Strength of out-of-support evidence: moderate to weak.** Pass@k at large k is preserved or slightly improved relative to the base model. That rules out the usual GRPO collapse, but it does not show problems moving from unsolvable to solvable under the base model at matched large k. The Fig. 9 "novelty" is lower base log-likelihood of whole responses, which is not the same as zero support.
- **Needs a pretrained prior.** The bonus lives in the pretrained model's representation space. RF2 shows that it fails or hurts on weak (0.5B) models, which is the regime closest to our 3.2M/9.56M models.
- The batch-relative bonus only diversifies among the rollouts actually sampled. Like any sampling-reweighting method, it cannot propose a step the policy never samples.

## Connections and questions
- **Group C.** RepExp is an on-policy reweighting. With k = 32 per target and a hardest step at about −12 nats (p ≈ 6×10⁻⁶), the bonus has almost nothing to reweight. This is our interpretation, consistent with RF2 and with POPE (2601.18779v1): "On hard problems, on-policy RL rarely explores even a single correct rollout".
- **E1.** The elliptical bonus could serve as a novelty prior inside PUCT, either over the proof-state embeddings of the policy or as a visit-count substitute for widening. This is cheap at d = 256. It is a better-founded version of RMaxTS's discrete novelty reward (`deepseek-prover-v15.md`).
- **E2 / E3.** It is not applicable from scratch, because the representations are meaningless at step 0. A representation that does not depend on the model (a hash of the Lean state, i.e. true counts, as in Bellemare 1606.01868v2) is the from-scratch substitute.
- Related earlier notes: `he2025-rewarding-unlikely.md` (the "Unlikeliness" baseline RepExp beats: "RepExp is 2.1-4.1x more sample-efficient than Unlikeliness"), `walder2025-pkpo.md`, `chen2025-passk-training.md`, `karan2025-power-sampling.md` (the Fig. 9 analysis is modelled on Karan and Du).
