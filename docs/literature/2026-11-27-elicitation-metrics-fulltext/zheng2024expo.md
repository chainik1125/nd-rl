---
written_on: 2026-11-27
written_by: agent:claude (reader-implementer for M2, Sonnet 5.5)
papers: [zheng2024expo]
reading_level: full text, all 17 pages incl. appendices A-D (reference list pages 10-13 skimmed only), arXiv v5 (30 May 2025, "Model Extrapolation Expedites Alignment"). Earlier versions (v1-v4, the first titled "Weak-to-Strong Extrapolation Expedites Alignment") not read.
---

# Zheng et al. 2024, ExPO / "Model Extrapolation Expedites Alignment" (arXiv 2404.16792v5): full-text reading note

Source: `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/papers/zheng2024expo/` (PDF + `.txt`). "p. N" = PDF page. Equations came through
the extraction intact; no TeX source needed.

## Method (Sec. 2, pp. 1-4)

- Setup: M0 = SFT checkpoint (theta_0), M1 = the same model after preference alignment (theta_1, DPO or RLHF), omega(theta) = an implicit scalar alignment objective, proxied by a reward model's mean reward on a dev set (p. 2).
- First-order argument: omega(theta_0 + gamma d) ~ omega(theta_0) + gamma grad omega(theta_0) . d, d = theta_1 - theta_0 (Eq. 1-3, p. 2); verified for gamma in [0, 1] by interpolation: interpolated models "generally exhibit intermediate alignment performance", increasing with gamma (Fig. 1, p. 3).
- EXPO (Eq. 6, p. 3): "theta_2 = theta_0 + (1 + alpha) d = theta_1 + alpha d", alpha > 0 chosen at inference time. Uniform over all parameters (the limitations section, p. 10, says per-module alphas are future work). Equivalent to interpolation with weight gamma = 1 + alpha > 1 (p. 4).
- Parameter-change evidence (Table 1, p. 2, zephyr-7b-dpo): SFT -> DPO Frobenius norm 0.0357 (normalised 6.348e-6) against pre-trained -> SFT 0.9882 (1.955e-4). So their DPO delta is about 3.6 % of the SFT delta in norm.

## How alpha is chosen, on what data

- p. 4 ("Hyperparameter Search"): "binary search and grid search with manually tuned intervals (App. B). We select the alpha giving the highest expected reward on the UltraFeedback development set (1K instructions), as calculated by the reward model RM-Mistral-7B." Evaluation then on AlpacaEval 2.0 (805 prompts, GPT-4-Turbo judge) and, for Sec. 3.3-3.4, the same UltraFeedback dev set used for the selection (p. 4, "unless otherwise stated, we report the expected reward score on the UltraFeedback development set"). So in Tables 3, 4 and 6 the number reported is the selection criterion itself (selected and scored on the same 1K prompts, same reward model); only Table 2's win rates (AlpacaEval) and Table 5 (AlpacaEval, MT-Bench) are on a set different from the dev set used to choose alpha (Table 5's alpha comes from the coarser grid described below).
- App. B (p. 15): search is a handful of runs. M10%: alpha 5 and 10 tried, then 7, 8, 9 -> 8. M20%: 2, 4, 1, 3, 2.5 -> 2.5 ("5 searches in total, each taking about 5 min ... about 0.5 GPU hours"). M40%: 0.5, 0.6, 0.4 -> 0.5. For 100 % and the 12 open models: grid [0.1, 0.5] step 0.1 (p. 8), picked per model "using the interval of 0.1 within [0.1, 0.5]"; the selection data for Sec. 4.1 are not stated explicitly (presumably the same dev set and reward model; I did not find a sentence saying so). Table 7 (p. 15) lists the optima: zephyr-7b-alpha 0.3, beta 0.1, tulu2 0.5, snorkel 0.3, llama3-8b-iter 0.3, starling alpha 0.2 / beta 0.5, internlm2 0.5. Note that tulu2 and internlm2 hit the edge of the grid (0.5), so those optima are censored.
- Optimal alpha falls as training grows: 8.0 (10 % of steps), 2.5 (20 %), 0.5 (40 %), 0.3 (100 %) (p. 5): "more training steps lead to larger ||d||, smaller alpha values are required"; large alpha gives "sharp performance drops and abnormal response length increases (e.g., generating gibberish or failing to terminate)" (p. 5, Fig. 4).

## Model pairs, benchmarks, sizes of gains

- Controlled (pp. 4-5, Table 2, zephyr-7b-dpo recipe, UltraFeedback 61K, batch 128, lr 5e-7, AdamW, 8 A100): AlpacaEval 2.0 LC win rate, DPO at 10 / 20 / 40 / 100 % of the steps (478 steps total): 10.4 -> 16.3, 12.9 -> 21.3 (+8.4), 14.6 -> 16.6, 17.3 -> 20.2 (+2.8); raw win rate 5.9 -> 17.9, 8.6 -> 22.7, 12.1 -> 17.7, 14.7 -> 18.0. The 20 %-trained + EXPO (21.3) beats the fully trained model (17.3). Cost: 12 GPU h for full DPO vs about 2.5 h training + 0.5 h search.
- 12 released models (Table 5, p. 8): offline DPO (zephyr alpha/beta, tulu2 7/13/70B), iterative DPO (snorkel, llama3-8b-iter), online RLHF (starling alpha/beta, internlm2 1.8/7/20B), 1.8B-70B. Gains up to +4.5 LC win rate (tulu2-70b) and +10.1 raw win rate (internlm2-20b), +0.37 MT-Bench (llama3-8b-iter); LC gains from +0.3 (internlm2-1.8b) to +4.5 (tulu2-70b), most between +0.7 and +2.4. Sec. 4.2 (Table 6, p. 9): seven other preference algorithms (RRHF, SLiC-HF, IPO, CPO, KTO, R-DPO, SimPO) on Mistral and Llama-3 SFT bases, reward gains +0.02 to +0.71 (KTO-Mistral: no gain).
- No seeds or intervals reported; vLLM decoding "temperature 0.7, top-k 40, top-p 0.9 ... sampling random seed 42" (p. 4): one seed, one sample per prompt. AlpacaEval's judge is a single GPT-4-Turbo pass.

## Failure cases and limits (pp. 6-7, 9-10)

- Data quality (Sec. 3.3, Table 3): with length-biased training data, EXPO gains vanish (optimal alpha 0.2-0.4 instead of 8 / 2.5; "N/A" at 40 %: no gain at alpha = 0.1).
- Training configuration (Table 4): 2 or 3 epochs, higher learning rate reduce the benefit; at 3 epochs "N/A"; AdaGrad best for extrapolation, RMSprop worst. Longer training = larger ||d||, first-order approximation breaks (p. 7).
- Sec. 4.3 (p. 9): "when we attempted with a pre-trained model as M0 and an SFT one as M1, we found that model extrapolation usually cannot improve alignment performance and can even lead to model collapse (e.g., the extrapolated model struggles to generate the EOS token or mistakenly generates special tokens)"; Storm-7B (3 rounds of iterative DPO): "even the very small alpha = 0.1 results in severe model collapse".
- Alignment tax (p. 10, App. D Fig. 6-7): EXPO amplifies the tax on GSM8K etc. for the partially trained models (the DPO models decline on GSM8K, EXPO declines further).
- Selection on a reward model proxy; response lengths grow with alpha (Table 3: 262 -> 541 tokens at M10 %, 294 -> 567 at M20 %), so part of the "gain" may be length; AlpacaEval LC win rate partly controls for this, raw win rate does not.

## What it supports for our question, and what it does not

- ExPO extrapolates from the SFT model through a partially trained aligned model, in a regime with "small" deltas (normalised Frobenius 6e-6) and a monotone-in-steps objective measured by a reward model. It never extrapolates between two aligned checkpoints (theta_1 - theta_1'), and never tests an RL run with optimiser restarts, abandoned branches or non-straight paths. Its own failure analysis says larger or more distorted deltas make the first-order argument fail (p. 7, p. 9).
- It supplies a direct, prior-based setting of alpha: with 20 % of the steps alpha 2.5; with 100 %, 0.3-0.5 (a gain of 2.8 LC points from extrapolating a fully trained model). For us s0500 is about 22 % of the labelled steps (500 of 2,300; about 18 % of the optimiser steps if labels overstate by about 100 after step 250). The nearest ExPO analogue is alpha near 2.5; the step-ratio rule (final / current - 1) would give 3.6 (labels) or 4.5 (corrected). These two are not independent of each other and neither is derived in the paper; the paper only supports "tune alpha on a dev set".
- Its objective is a reward model score, ours is a verifier pass rate on held-out prompts; the transfer of its alpha values is not supported by anything in the paper.

## Ambiguities and unsupported points

1. alpha is a global scalar over all parameters; whether embeddings and norms are extrapolated is not stated (the repository is referenced on p. 1; not read).
2. Dev-set selection and test-set scoring coincide in the ablation tables (3, 4, 6).
3. Section 4.1's selection data and the precision of the extrapolated weights (bf16 rounding) are not stated.
4. The first-order argument is an interpolation check (gamma in [0,1]), then assumed to hold for gamma > 1; the evidence beyond 1 is the alpha curves (Fig. 4), which rise then crash.
5. Single seed, no intervals.

## How our setting differs

32B thinking model with long chains; verifier-scored IFBench (strict); RL (not DPO) stage on top of a DPO model; the "SFT checkpoint" M0 here would be `dpo` (itself preference-trained); 2,300 steps at LR 2e-6 (ExPO's recipe: 478 steps at 5e-7), optimiser restarts, abandoned branch; the delta between `dpo` and `s2300` is 0.364 % of ||theta_dpo|| (a full-model relative figure, not the paper's normalised Frobenius, so not directly comparable); a released bf16 checkpoint must be written to disk to serve a full-rank extrapolation (64.5 GB).

## Not read

v1-v4; the repository; the reference list in detail (pp. 10-13); figures (Fig. 1-4, 6-7) as data. Table 2 uses AlpacaEval win-rate intervals that the paper does not report.
