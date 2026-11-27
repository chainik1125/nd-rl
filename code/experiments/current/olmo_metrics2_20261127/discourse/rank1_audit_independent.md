---
written_on: 2026-11-27
written_by: agent:claude (auditor for M2 `rank1`, Claude Opus 5.5)
papers: [cai2025predictabilityrldynamics, zheng2024expo]
reading_level: >-
  full text of both papers, every page marker read with the Read tool. Cai et al. arXiv 2510.00553v3
  (22 Feb 2026, "Published as a conference paper at ICLR 2026"), 48/48 pages incl. appendices A-E and figure pages
  24-48. Zheng et al. (ExPO) arXiv 2404.16792v5 (30 May 2025, "Model Extrapolation Expedites Alignment"), 17/17 pages
  incl. appendices A-D. Pages rendered to images and inspected: Cai pp. 1, 3, 4, 5, 6, 7, 24, 26, 27, 28, 35 and
  zoomed Figures 3, 4, 7 (MATH-500 panel); ExPO pp. 3, 5. TeX sources not downloaded (equations were legible in the
  rendered pages).
---

## Auditor (Claude Opus 5.5), independent read, 2026-11-27 01:58 PST

Sources I made and used (all under `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/audit/`):
`cai2025predictabilityrldynamics.txt` (48 pages), `zheng2024expo.txt` (17 pages), extracted with the CONTEXT.md command
from the reader-implementer's PDFs; page images in `png_rank1/`; and one small computation from Dan's tracked
per-tensor statistics (`nd-takehome` commit `e8d33db7`, `artifacts/olmo_weights/stats/p32_rl/shard_*/{st,iv}.csv.gz`)
with `rank1_param_groups.py` -> `rank1_param_groups_output.txt` (it reproduces Dan's 0.3642 % total, so the
aggregation is right). I did not open the reader-implementer's notes, any other `discourse/` file, or the 2026-11-25
review notes. Page numbers below are PDF pages ("p. 7" = `=== page 7/48 ===`). "Guess" marks my inferences.

### Numbers from Dan's stats that the answers below use (computed here, 2026-11-27)

| checkpoint | ‖θ_t − θ_dpo‖ | share of ‖Δ‖² in the 448 matrices / lm_head / embeddings / norms | 448-matrix weights bitwise equal / within 1 ulp | per-weight relative change, 448 matrices / lm_head |
|---|---|---|---|---|
| s0100 | 2.65 | 98.97 / 0.97 / 0.06 / 0.000 % | 78.8 / 92.5 % | 0.070 / 0.052 % |
| s0500 | 5.96 | 98.84 / 1.09 / 0.07 / 0.000 % | 64.6 / 85.4 % | 0.158 / 0.123 % |
| s1100 | 10.99 | 97.96 / 1.93 / 0.11 / 0.000 % | 49.0 / 77.5 % | 0.290 / 0.303 % |
| s2300 | 18.91 | 98.43 / 1.45 / 0.12 / 0.001 % | 33.3 / 64.6 % | 0.500 / 0.452 % |

(‖θ_dpo‖ = 5,192.5.) One bf16 ulp is 0.39-0.78 % of a weight's magnitude, so at s0500 the typical 448-matrix weight
moved about a third of an ulp and at s2300 about one ulp. Distance from dpo grows sub-linearly in steps:
‖Δ2300‖/‖Δ500‖ = 3.17 against a label-step ratio of 4.6 (5.5 with the ~100-step label offset).

---

### Q1. Construction

**Cai et al., rank-1 (Section 2.1, p. 3, Eqs. 1-3).**
- Per update matrix: "performing SVD on Δ𝑾, we have: Δ𝑾 = Σ_{i=1}^{r} σ_i u_i v_i^⊤, r = rank(Δ𝑾)" (Eq. 1); "The
  Rank-1 update matrix is then defined by retaining only the largest singular value σ1 and its vectors:
  Δ𝑾^(1) = σ1 u1 v1^⊤" (Eq. 2).
- Rescaling: "To ensure consistency in update strength, we rescale Δ𝑾^(1) to match the L2 norm of Δ𝑾:
  Δ𝑾̂^(1) = αΔ𝑾^(1), α = ‖Δ𝑾‖₂ / ‖Δ𝑾^(1)‖₂" (Eq. 3). **Which norm is never stated.** For a matrix, ‖·‖₂ is
  conventionally the spectral norm, which equals σ1 for both Δ𝑾 and Δ𝑾^(1), giving α ≡ 1 (a no-op). The paper must
  mean the Frobenius (entrywise L2) norm: Fig. 5(a) reports "the fraction of update information captured by the
  unscaled Rank-1 and Rank-1% Subspaces" (12.6-49.1 %, p. 6), and Fig. 4(b) shows that the scale matters. So the
  construction is, in effect, Δ𝑾̂^(1) = ‖Δ𝑾‖_F · u1 v1^⊤: the top singular direction carried at the **full
  Frobenius norm of the whole change** (σ1 cancels). Its spectral norm is ‖Δ𝑾‖_F, i.e. α = ‖Δ𝑾‖_F/σ1 times larger
  than the real top component. If Fig. 5(a)'s "Rank-1 fraction" is σ1/‖Δ𝑾‖_F ("the proportion of the norm of the
  unscaled Rank-1 Subspace ... relative to the total update norm", p. 5), RL's α in the paper is about 4-5.5
  (fractions 18.3-25.6 %); if it is an energy fraction, about 2-2.3. Ambiguous.
- Which parameters: "Δ𝑾 (i.e., the parameter difference between the RL-trained model and the base model)" (p. 2);
  the module analyses list exactly seven module types per layer, "MLP Gate, MLP Up, MLP Down, Attn Q, Attn K, Attn V,
  Attn O" (Fig. 15, p. 35). Embeddings, lm_head and norm vectors are never said to be included or excluded. Indirect
  evidence (guess): in Fig. 7 (p. 24) "Top Rank-100%" gives DIST about 26 % and SFT about 57 % relative accuracy on
  MATH-500. Keeping all singular directions reconstructs Δ𝑾 exactly, so 100 % would be expected unless some changed
  parameters were left at base. DIST and SFT are also the methods whose embeddings moved most (Fig. 5(b), p. 6). So
  the paper probably applied the decomposition only to the per-layer linear modules and left everything else at base.
  That matches the LoRA route, but the paper does not say so.
- All at once or one at a time: one rank-1 per module, all modules at once, each with its own α. "The evaluation
  model is then obtained by adding Δ𝑾̂^(1) to the base model" (p. 3). The sliding-window experiment is the
  exception that shows the default: "For each window, only the Rank-1 updates of the selected modules are injected
  into the base model, while other modules remain unchanged" (p. 7).
- Checkpoint pair: (base before the method, trained model). For the intermediate-checkpoint curve, each checkpoint
  against the same base (Fig. 3(b), p. 4; Fig. 10, p. 26). The base is the RL starting point. For Table 4 the bases
  are "Deepseek-R1-Distill-Qwen-7B (After Distillation)" and "Llama3-8B-Instruct (After SFT)" (p. 19).
- Rank-k%: "only the leading the top k% of singular subspaces are retained" (p. 3). The text does not say whether
  rank-k% is rescaled.
- Scaling variant: "Δ𝑾̂^(1) = λ·αΔ𝑾^(1) ... performance increases rapidly with λ and then saturates, peaking at
  λ≈0.7, slightly below the norm magnitude of the full update" (p. 4, Fig. 4(b)).

**AlphaRL (Section 4.1, p. 7; Appendix E, p. 34, Eqs. 4-10).**
- Inputs: for every module and every saved checkpoint t from the base up to the current stage ("checkpoints between
  the Base model and the current training stage", Table 1 caption, p. 8; "We save the checkpoint after each training
  batch", p. 22), the "scaled Rank-1 vectors, where each vector is scaled by the product of α(t) and σ1(t)" (p. 7),
  i.e. û(t) = α(t)σ1(t)u1(t) = ‖Δ𝑾_t‖_F u1(t) (App. E: "AlphaRL perform the same computation but with the scaled
  vectors û(t) = α(t)σ1(t)u(t)", p. 34). The response is each checkpoint's relative accuracy on MATH-500.
- Fit: features standardized ("each feature is standardized to zero mean and unit variance", p. 34), one-component
  PLS: z1 = Ũ1 w1 (Eq. 7), y(t) = α z1(t) + β + ε(t) (Eq. 8), OLS (Eq. 9), R² (Eq. 10).
- Inversion: "Given a target relative accuracy y*, AlphaRL obtains the corresponding update vector through
  inversion. Finally, this update vector is combined with the left singular vector v1 to form the new Rank-1 update
  for each module" (p. 7). How a scalar target maps back to a d-dimensional vector is not given (guess: along the PLS
  loading, then de-standardized). Which checkpoint's v1 is used is not given. The text calls v1 the "left" singular
  vector, while Eq. 1 and App. E make u the left ("value") vector and v the right ("key") one (p. 34).
- Coefficient: "We use the model's test accuracy on MATH-500 and set the target accuracy to y* = 1" (p. 7). With y
  defined as relative accuracy, "the ratio between the accuracy of the Rank-1 reconstructed model and that of the
  fully trained model" (p. 4), y* = 1 means "reach the fully trained model's MATH-500 test accuracy". So the
  extrapolation length is set from the end-of-run model's score on a benchmark that Table 1 then reports. Reading
  y* = 1 as 100 % absolute accuracy conflicts with "fit the vectors with their relative accuracies" (p. 7). The paper
  calls AlphaRL free of "hyperparameter tuning, or human intervention" (p. 3).

**ExPO (Section 2, pp. 2-3, Eqs. 1-6; Section 3.1, p. 4; Appendix B, p. 15).**
- Anchors: M0 is the pre-alignment checkpoint and M1 the (partially) aligned model initialized from it: "denote its
  SFT checkpoint as M0 (parameterized by θ0), which is used for initializing M1" (p. 2). Formula: "θ2 = θ0 +
  (1 + α)Δθ = θ1 + αΔθ" (Eq. 6, p. 3), with Δθ = θ1 − θ0, full rank and uniform over all parameters ("the simplest
  form of uniform extrapolation", p. 10). It extrapolates **from the initial checkpoint through M1**. It does not
  extrapolate a mid-run segment. In our terms ExPO is dpo + γ(s0500 − dpo). CONTEXT's "(s0100 -> s0500)" is a
  different construction that neither paper evaluates. ExPO only points at it as future work: "leveraging the
  intermediate checkpoints" (p. 9).
- Justification: first-order Taylor, "where we define γ ∈ [0, 1] to ensure that ‖γΔθ‖ remains small" (p. 2), then
  used with γ = 1 + α > 1.
- Choice of α: "We select the α giving the highest expected reward on the UltraFeedback development set (1K
  instructions), as calculated by the reward model RM-Mistral-7B" (p. 4), by manual binary and grid search (App. B,
  p. 15). Optimal α: "8.0, 2.5, 0.5, and 0.3" for the 10 %, 20 %, 40 % and 100 % models (p. 5). For the
  twelve released models: "a simple grid search for the optimal α, using the interval of 0.1 within [0.1, 0.5]"
  (p. 8). The selection data for those is not restated (presumably the same dev set and reward model). Test sets
  (AlpacaEval 2.0, 805; MT-Bench) are separate from the selection set.

### Q2. Evidence

**Cai et al.**
- Models and algorithms (App. B, pp. 20-22): DIST = DeepSeek-R1-Distill-Qwen-7B against Qwen2.5-Math-7B; SFT =
  Qwen3-8B-Base on DeepMath-103K; PPO = Open-Reasoner-Zero-7B against Qwen2.5-7B; Dr.GRPO = Qwen2.5-Math-7B-Oat-Zero;
  RLOO and GRPO = their own Qwen3-8B-Base runs on DAPO-Math-17k (Verl); DAPO at 7B (from R1-Distill), 8B, Llama3.1-8B
  after a cold-start SFT, GLM-9B, Qwen3-14B-Base, and 32B = "DAPO-Qwen-32B trained from the base Qwen2.5-32B" (the
  only 32B model; probably "DAPO*" in the figures, "DAPO(32B)" in Figs. 8 and 11). Also on-policy distillation, DPO
  (Qwen3-8B-Base on Math-Step-DPO-10K), and Spiral self-play on Qwen3-4B-Base. Training to convergence: "until the
  reward/loss converges" (p. 21). RL lr 1e-6, batch 512, n = 8 rollouts, "under Float16 precision" (p. 22).
- Benchmarks: AIME24, AIME25, MATH-500, Minerva, GPQA, GSM8K. AlphaRL evaluation: "32 sampled responses per question
  under temperature T = 0.6, and we report average accuracy" (p. 8). Decoding for the Section 2 rank-1 experiments is
  not stated.
- Seeds and error bars: one training run per configuration. No seeds. No confidence intervals in any table. Fig. 6(c)
  has an unexplained shaded band. Single-checkpoint accuracies are noisy: Fig. 10 (p. 26) has GRPO-full jumping about
  47 -> 72 -> 47 % between neighbouring checkpoints.
- Claim -> support:
  - Rank-1 recovery ">99%" (abstract) and "an average of 99.17%" (p. 2): Fig. 3(a) (p. 4) and Fig. 7 (p. 24), bar
    charts of relative accuracy. No table gives the 99.17 % average. The metric is a ratio of accuracies, not the
    fraction of the gain recovered. Fig. 4(b) shows why that matters: at λ = 0.1 DAPO reads about 49.5 % against a
    45 % base and an 81 % full model, i.e. 61 % "relative accuracy" but about 12 % of the gain. The abstract's
    "recovering over 99% of performance gains" overstates what the metric measures. Near 100 % the two coincide.
  - Rank-1 throughout training: Fig. 3(b) and Fig. 10 (single evaluations, no bands).
  - Only the top direction matters: Fig. 4(a) and Fig. 11 (pp. 5, 26). Every individually rescaled direction helps,
    in decreasing order: bars near the "60" tick of Fig. 4(a) read about 60-64 % against the 45 % base (zoom; the
    ticks label every other bar, so the exact indices are uncertain).
    The paper says this too: "several subspaces ... still contribute substantially" (p. 4).
  - Linear dynamics R² "exceeding 0.96" (p. 2) and "an average R² of 0.914" (p. 2): one module's R² = 0.960 is shown
    (Fig. 6(b)), plus a heatmap (Fig. 15). No table or figure reports 0.914.
  - "predictions of its later state based on early-stage states achieve an average error of less than 5%" (p. 2,
    3): **no table or figure supports it.**
  - RL-specificity: Fig. 3(a) and Fig. 5. **Internal contradiction.** Pp. 3 and 5 say "control experiments with ...
    supervised fine-tuning and distillation ... yielded neither property" and the low-rank property "is completely
    absent in SFT and DIST". Appendix C (p. 23) says "Both On-Policy Distillation and DPO exhibit clear low-rank
    structures in their update patterns, demonstrating that the phenomena identified in this work are not unique to
    RL", and Table 8 (p. 21) gives rank-1 relative accuracy of 56.4 % (online DIST) and 62.25 % (offline DPO). No
    figure or table shows Property 2 (linear dynamics) for SFT or DIST at all.
  - AlphaRL "up to 2.5×" and ">96%": Table 1 (p. 8) and Tables 3-4 (pp. 18-19). The 2.5× is one case, DAPO-8B at
    40 % (53.31 vs 53.38 average). GRPO and RLOO at 40 % retain 95.9 % and 95.5 % (my arithmetic from Table 1), below
    ">96%". As fraction of the remaining gap closed: GRPO 66 %, RLOO 47 %, DAPO 99 % at 40 %; at 20 % (Table 3)
    Qwen3-8B 43 %, Qwen3-14B 33 %, GLM-9B 38 %. The wall-clock saving does not count evaluating every early checkpoint
    on MATH-500 (32 samples) or the SVDs. The MATH column of every AlphaRL row is not held out, because y* is set on
    MATH-500 test accuracy.
  - "the Rank-1 structure ... hold[s] across these diverse model initializations" (p. 18): Table 4 shows AlphaRL
    rows only, no rank-1 reconstruction numbers for R1-Distill or Llama-Instruct. **No table supports the rank-1 half
    of that sentence.** Those are the two cases closest to ours (RL from a post-trained or thinking model).
- Other internal inconsistencies:
  - The abstract has "8 LLMs and 7 algorithms". The introduction has "13 diverse LLMs (ranging from 7B to 32B ...)
    trained with 10 advanced training algorithms" (p. 2). Spiral uses a 4B model (p. 19).
  - Three reports of the DIST/SFT rank-k% experiment disagree. Fig. 3(a)'s groups "70%" and "90%" (DIST about 19 / 26,
    SFT about 29 / 57) equal Fig. 7's "Top Rank-80%" and "Top Rank-100%". Table 8 gives Rank-70% 16.00 / 24.00 and
    Rank-90% 23.25 / 46.75.
  - Table 8's DAPO row is exactly "100.00%" at every rank.
  - Table 4's "Training 20%+AlphaRL" average for R1-Distill reads 48.23 but its six entries average 68.23.
  - "Our model and code will be available" (abstract, p. 1) contradicts "We have already released the complete code"
    (p. 17).
  - "Appendix 7" and "Appendix 15" (pp. 4, 6) are figure numbers.

**ExPO.**
- Controlled experiments: one 7B recipe (zephyr-7b-dpo: Mistral-7B SFT -> DPO on UltraFeedback, lr 5e-7, batch 128,
  478 steps), retrained to 10 / 20 / 40 % of the steps (p. 4). Ablations on data length bias, epochs, lr and optimizer
  (Tables 3-4, pp. 6-7). Twelve released DPO, iterative-DPO and online-RLHF models from 1.8B to 70B (Table 5, p. 8).
  Seven alignment objectives on Mistral and Llama-3 SFT (Table 6, p. 9).
- Evaluation: AlpacaEval 2.0 (805; GPT-4 Turbo judge; raw and LC win rate), MT-Bench (GPT-4 score), and expected
  reward on the UltraFeedback dev set (1K; RM-Mistral-7B, FsfairX-LLaMA3-RM). Sampling: "top-k (k = 40) and nucleus
  sampling (p = 0.9) with a temperature of 0.7 ... presence penalty and frequency penalty to 0.1. We set the sampling
  random seed to 42" (p. 4). One training run per configuration. No CIs in tables. Shaded bands in Figs. 1 and 4 are
  unexplained.
- Claim -> support:
  - "boosts a DPO model trained with only 20% steps to outperform the fully-trained one": Table 2 (p. 5), reward 6.95
    vs 6.16, LC 21.3 vs 17.3 %.
  - "up to 8.4%" LC: Table 2. "up to 4.5% on AlpacaEval 2.0 and 0.37 on MT-Bench" over 12 models: Table 5.
  - 75 % compute saving: Section 3.2 cost paragraph (p. 5).
  - "consistently improves": Table 5 deltas are all positive, but LC deltas of +0.3 (internlm2-1.8b) and +0.7
    (starling-7b-beta) are, by my estimate, inside AlpacaEval's run-to-run noise. Table 6 has KTO-Mistral "N/A" (no
    gain), which the text's "effectively complements various alignment training algorithms" (p. 9) passes over.
- Weak points inside the paper:
  - Table 2's "Reward" column uses the same reward model that chose α. Only the GPT-4 win rates are an independent
    judge.
  - Gains come with longer outputs: raw win-rate gains exceed LC gains, Fig. 4's length curves rise with α, and
    Table 3's lengths go 294 -> 567.
  - Optima are narrow cliffs. For the 20 % model α = 2 helps but α = 4 is "worse than M1" (App. B), and in Fig. 4
    it falls below SFT. Too-large α gives "generating gibberish or failing to terminate" (p. 5).
  - Failure cases (p. 9): pre-trained -> SFT extrapolation "can even lead to model collapse"; Storm-7B collapses
    "with even the very small α = 0.1".
  - App. B's "found it outperformed M0" is presumably M1 (typo).

### Q3. Faithful replication on our checkpoints

**Rank-1 (Cai-faithful), what it looks like.**
- For each of the 448 matrices: ΔW = W_s2300 − W_dpo in fp32 from the bf16 tensors; SVD; add ‖ΔW‖_F · u1 v1^⊤ to dpo
  (λ = 1), all 448 at once. Embeddings, lm_head and norms stay at dpo.
- Serve it as a rank-1 PEFT adapter on dpo: B = √‖ΔW‖_F · u1, A = √‖ΔW‖_F · v1^⊤, lora_alpha = r.
- Score with Dan's sampler and the IFBench strict checker on arm A and reach, against dpo and s2300 from the same
  pipeline (Dan's stored draws, which the 2026-11-25 run reproduced).
- Report "fraction of the gain recovered", (cell − dpo)/(s2300 − dpo), as the headline and Cai's ratio beside it for
  comparability. The ratio is misleading on arm A, where dpo alone already scores 29.0/68.3 = 42 % "relative
  accuracy".

**Departures, and whether each one changes the meaning.**

1. **bf16 dead zone (changes the meaning partly; must be measured before sampling).**
   - dpo's weights are exact bf16 values, so a stored change is round(w + δ) − w. Any |δ| below half an ulp vanishes,
     and larger ones are quantized to whole ulps whose size depends on |w|.
   - The stored Δ is therefore a dead-zone-quantized, |w|-weighted version of the true update. At s2300, 33 % of
     matrix weights are bitwise equal and the typical move is about one ulp. At s0500, 65 % are equal and the typical
     move is about a third of an ulp (table above).
   - Consequences:
     - (a) The top singular direction of the stored Δ is probably still close to the true one (signal concentrates,
       quantization distortion spreads), but this is a guess to check, not a fact.
     - (b) Cai's α = ‖ΔW‖_F/σ1 counts the quantization energy in the numerator, so in our regime the rescaled rank-1
       is likely amplified more than in the paper. The paper's own SFT/DIST cells, where rank-1 gives about 0-2 %
       relative accuracy (far below base), show that over-amplified rank-1 changes break models.
     - (c) Extrapolating a stored early diff amplifies its rounding pattern by γ. This hits ExPO from s0500 hardest.
   - Mitigation: compute σ1, ‖ΔW‖_F, α and the singular spectrum per matrix before any sampling, and estimate a noise
     floor. For example, take the top of a Marchenko-Pastur-like bulk, or compare against the spectrum of a matrix
     with the same per-entry ulp pattern but shuffled signs (guess at method).
   - Cai's open-source RL models (Open-Reasoner-Zero, Oat-Zero, DAPO-Qwen-32B) are, I believe, bf16 releases too
     (unverified), so bf16 per se does not void the comparison. Our per-weight moves are smaller, though.
2. **LoRA serving leaves lm_head, embeddings and norms at dpo (probably the same as Cai, but not neutral here).**
   - lm_head carries 1.45 % of ‖Δ2300‖² and moves 0.45 % per weight, as much as the 448 matrices (0.50 %).
     Embeddings carry 0.12 % and norms about 0 %.
   - Cai is silent on these, and their Fig. 7 hints they left non-module parameters at base (Q1).
   - Effect on the meaning: if rank-1 falls short, part of the shortfall could be lm_head. This is resolved by a
     reference "dpo + full Δ on the 448 matrices only" (a written checkpoint: s2300 with lm_head, embeddings and norms
     copied from dpo), or by verifying that vLLM 0.11's LoRA path can target lm_head for Olmo3 (I have not checked;
     do not assume).
3. **LoRA numerics (a gain in faithfulness, not a loss).**
   - Adding a rank-1 change into bf16 weights would round most of its entries away, because they are sub-ulp just
     like Δ's. A merged-checkpoint route would therefore be unfaithful unless weights were kept in fp32, and fp32 for
     a 32B model does not fit one H200 with KV cache.
   - The adapter adds the change in activation space, where it survives.
   - Keep the 2026-11-25 `lora_check.py` comparison, extended to one rank-1 adapter.
4. **Domain and starting point (changes the prior, not the construction).**
   - Cai's evidence is mostly RL-Zero on math from base models. There the gain is largely an opening-style switch: "using
     only the first 20 tokens - approximately the first sentence - allows the base model to approach the performance
     of the full parameter update" (p. 28, Fig. 14).
   - A mode switch is plausibly one direction. Our RL starts from a DPO thinking model, the domain mix is multi-domain,
     and the reach prompts need constraint satisfaction the start model never showed. The prior for rank-1
     sufficiency is weaker here.
   - The only 32B case in Cai (DAPO*) is also its weakest on hard sets: about 87 % (AIME24) and 91 % (AIME25) relative
     at rank-1, read from Fig. 7 bars.
5. **Decoding** (T 1.0, top_p 1.0, 32,768 tokens vs the paper's T 0.6 on math). No change in meaning, since every cell
   uses the same decoding as its references.
6. **Lineage and the step-label offset** are irrelevant to rank-1 of (s2300 − dpo) and matter for extrapolation:
   - "% of training" for s0500 is 21.7 % by labels and about 18 % with the offset.
   - The step-matched γ is 4.6 by labels and about 5.5 with the offset.
   - The s0100 -> s0500 segment starts with a fresh-optimizer restart (the released line restarts from step 100), so
     its direction carries a restart transient. ExPO's dpo-anchored Δ500 also contains the DPO -> 50 transient.

**ExPO-faithful.** θ2 = dpo + γ(s0500 − dpo), with γ chosen on dev prompts by the strict checker (our analogue of
ExPO's reward-model dev search), written as a bf16 checkpoint (one load per γ).
- Writing in bf16 is a smaller issue than for rank-1, because γΔ is larger than Δ. Departures: domain (verifiable IF
  vs preference judged by reward models) and the bf16 amplification in item 1(c).
- θ2 = s0500 + β(s0500 − s0100) is **not ExPO**. If run, call it "segment extrapolation", not a replication.

**AlphaRL-faithful is not feasible.** It needs accuracies at many early checkpoints and a target equal to the final
model's accuracy (leakage). With dpo, s0100 and s0500 only, the PLS fit has three points and reduces to scaling s0500's
rescaled rank-1. A cheap and honest analogue ("AlphaRL-lite") is dpo + κ · ‖Δ500‖_F u1(500) v1(500)^⊤ with κ chosen on
dev prompts. It is an adapter, so a κ sweep needs no reload. Rank truncation also filters the full-rank rounding pattern
that hurts ExPO in bf16.

### Q4. Weakest points that matter for our use

Cai et al.:
1. Eq. 3's norm is unspecified (the spectral reading makes it a no-op). The effective construction amplifies the top
   direction by α = ‖ΔW‖_F/σ1. "Rank-1 dominance" is therefore a claim about an amplified direction, and the paper
   never shows the unscaled rank-1 except implicitly through λ in Fig. 4(b). There, at the smallest λ, it recovers a
   small fraction of the gain.
2. There is no random-direction control. Fig. 4(a)/11 show that many lower-ranked rescaled directions also lift
   accuracy, so the specificity of "the top direction" is a matter of degree.
3. RL-specificity is confounded with update size. Update norms run offline DIST 21.24, SFT 10.75, online DIST 1.46,
   DPO 0.79, DAPO 0.01 (Table 7, p. 21). Rank-1 relative accuracy rises in exactly that order: 1.5, 2.5, 56.4, 62.25,
   100 % (Table 8). The paper also contradicts itself on specificity (Q2). For us the open question is "small update", not
   "RL", and our update is small (0.36 %).
4. Parameter scope is unspecified, and "Rank-100% < 100 %" (Fig. 7) suggests an omission.
5. Relative accuracy is a ratio, not the fraction of the gain.
6. One run per configuration, no error bars, noisy single-checkpoint evaluations.
7. Property 2's evidence is weak.
   - t-SNE pictures (Fig. 6(a)) cannot show linearity, because t-SNE does not preserve global geometry.
   - In-sample one-component PLS of accuracy on a d ≫ T vector (about 27 checkpoints vs ≥ 4,096 standardized
     features) will give high R² for almost any drifting trajectory paired with a rising accuracy curve. That is my
     reasoning, not a tested claim.
   - There is no held-out R², no null, and no SVD sign alignment.
   - The "< 5 %" prediction error has no support.
8. AlphaRL's target uses the final model's MATH-500 test accuracy, the inversion is underspecified, and the u/v naming
   is inconsistent.
9. Population: mostly 7-8B RL-Zero math with a style-switch gain. One 32B model. No instruction-following task.

ExPO:
1. Evidence for "extrapolate a partially trained model" comes only from 7B **offline DPO** (Section 3.2). The RLHF
   models (Section 4.1) are fully trained and get small α (0.1-0.5). Extrapolating from 20 % of an online-RL run is
   untested in the paper.
2. α is tuned against a length-biased reward model. Gains partly come from length, and optima are narrow with cliffs
   past them. For a thinking model, "failing to terminate" means running to the 32,768-token cap, which is a cost
   risk.
3. The first-order argument assumes a small ‖(1+α)Δθ‖ and a roughly straight path. Our path is far from straight
   (straightness 0.10, consecutive-change cosine about −0.2), and distance grows sub-linearly. Amplifying Δ500 also
   amplifies its oscillating and rounding components.
4. One seed (42), one training run, no CIs. Some reported gains are within judge noise (my estimate).
5. ExPO only claims behaviour (reward) improves. It never claims the extrapolated weights approach the final weights,
   so a weight-space miss would not refute it.

### Q5. Predictions

**(a) Cai rank-1 of (s2300 − dpo), λ = 1, all 448 matrices, added to dpo.**
- Prediction: about s2300's behaviour. Relative accuracy about 99 %, i.e. arm-A pass@1 about 67-68 % and reach about
  69 % of draws. The paper's only 32B case suggests 85-95 % on hard subsets, i.e. reach about 59-66 % (a weak transfer
  from math).
- What would contradict it:
  - Rank-1 recovers less than about half of the gain on arm A or reach, while the full-change reference reproduces
    s2300 and the random control stays at dpo.
  - Or the model breaks (garbled text, cap hits, below dpo). That would show that Cai's Frobenius rescale fails in
    this regime, a result in itself, and the unscaled cell then speaks to the direction.
- A null is interpretable only if all three hold:
  - (i) The adapter path is shown to deliver: a lora_check on the rank-1 adapter, and either the 448-only full
    reference or a rank-16 cell recovering most of the gain on the same path.
  - (ii) σ1 lies clearly above the bf16 noise floor in most matrices. Otherwise "the top direction" is partly rounding
    pattern, and the null says more about bf16 than about RL.
  - (iii) The random control is reported.
  Without (ii), a null is not interpretable as a test of the paper.
- A split result (arm A recovered, reach not) is informative: rank-1 carries a common mode but not the specific
  capability.

**(b) Extrapolating the early change to step 2300.**
- Cai/AlphaRL from about 20 % of training: Table 3 closes 33-43 % of the remaining gap. Transferred loosely:
  arm-A pass@1 about 50-53 % (s0500 41.2 % -> s2300 68.3 %) and reach about 26-32 % (4.5 -> 69.4 %). Cai's AlphaRL
  also uses an end-of-run target, so this is optimistic for a leak-free variant.
- ExPO from 20 % (Table 2): the extrapolated model beats the fully trained one. Prediction: arm-A ≥ 68 % and
  reach ≥ 69 %.
- **The two papers disagree at s0500**, so the cell discriminates between them.
- What contradicts ExPO: the dev-best γ gives held-out arm-A within a few points of s0500, or every γ > 1 is worse.
- A null is weaker than a positive:
  - The grid may miss a narrow optimum (Fig. 4).
  - The bf16 dead zone erased most of Δ500's per-weight signal (65 % of weights unchanged).
  - The curved path predicts failure anyway.
  - A null says "no straight dpo-anchored line through s0500 reaches s2300's behaviour in the tested γ range". It does
    not say "ExPO is wrong".
- A positive (dev-chosen γ reaching s2300 on held-out prompts at similar length and cap-hit rate) would be a strong
  "cheaply recoverable from s0500" result. Checker passes still need reading for genuineness.

### Q6. Controls

**Necessary.**
- **References**: dpo and s2300 (Dan's stored draws, same sampler; the 2026-11-25 run reproduced them; no new cost),
  and s0500 for the extrapolation cells.
- **Random rank-1 of equal norm.**
  - For each of the 448 matrices W (m × n), seed by (fixed global seed, tensor name), draw g ~ N(0, I_m) and
    h ~ N(0, I_n) in fp64.
  - R = ‖ΔW‖_F · (g/‖g‖)(h/‖h‖)^⊤. This matches the Cai-scaled rank-1 matrix by matrix in both Frobenius and spectral
    norm.
  - All 448 at once, as a rank-1 adapter on the same dpo load. One draw (448 independent matrices self-average). A
    second draw only if the first moves anything.
  - Caveat for interpreting it: an isotropic v barely overlaps typical activations, so this control is weak. A null
    from it does not prove the RL direction is special; an effect from it would be alarming.
- **Unscaled rank-1** (σ1 u1 v1^⊤, λ = 1/α). Separates "this direction" from "this direction amplified by α". It is
  essential here because α is probably larger than in the paper (bf16, large matrices).
- **One more rank point**: unscaled rank-16. It has 134 M parameters, the same as the trained 64-example rank-16 LoRA
  (63.5 % on reach), so the truncated real RL change and a trained adapter compare at equal size. Required when rank-1
  falls short (to tell "not low-rank" from "not rank-1"); nice-to-have when rank-1 succeeds.
- **448-only full change** (written checkpoint). Conditional: required if rank-1 or rank-16 falls short, to apportion
  the shortfall to lm_head and the other excluded parameters.
- **For extrapolation**: γ = 1 (s0500, stored) as the baseline, and the dev-chosen γ scored on held-out prompts.
- **Per-cell diagnostics**: mean response length, % of responses hitting 32,768 tokens, % empty or garbled.
- **Weight-space diagnostics on the pod before sampling (no sampling cost)**:
  - α distribution, σ1 against the noise floor, and energy in the top 1 / 16 / 64 directions, per module type and
    depth.
  - Per matrix: cos(Δ500, Δ2300), |u1(500)·u1(2300)| and |v1(500)·v1(2300)|. Property 2's premise is a stable top
    direction, so this tests it without sampling.
  - The weight-space least-squares γ_proj = Δ500·Δ2300 / ‖Δ500‖² (an oracle ceiling for the straight line, never a
    chosen setting).

**Nice-to-have.**
- Negated rank-1 on reach (should be ≤ dpo).
- A mid singular direction (k = 64), Cai-scaled: the paper's own Fig. 4(a) ablation.
- An activation-matched random control (v drawn from the input-activation principal subspace).
- A λ sweep {0.25, 0.5, 1} for the Cai rank-1 on dev prompts. This becomes necessary if the α diagnostics show
  extreme amplification (my threshold guess: median α > 10).
- Rank-1 of Δ500 (Property 1 at an intermediate checkpoint; low signal, since reach at s0500 is 4.5 %).
- AlphaRL-lite with a dev-chosen κ, and rank-16-filtered extrapolation dpo + γ·rank16(Δ500). Both are adapters, so
  cheap, and both denoise the bf16 rounding pattern.
- Segment extrapolation from s0100.
- A matched-norm random extrapolation (dpo + Δ500 + random direction of norm (γ−1)‖Δ500‖).

### Q7. Minimum cells and rough cost

Assumptions:
- H200 at $5.29/h.
- 1,450 tokens/s on both the base and LoRA paths (CONTEXT: 1,335-1,700).
- Mean response length: 4,500 tokens for dpo-like cells, 6,000 for unknown cells, 6,800 for s2300-like or extrapolated
  cells.
- Model load about 7 min. Downloads about 2 min per checkpoint with a stall buffer.
- Top-k SVD by randomized SVD on the GPU. Full spectra only for a sample of about 16 matrices (an exact full SVD of all
  448 is likely 15-45 min, a guess).
- Pod disk: dpo, s0500 and s2300 (194 GB), plus about 65 GB per written checkpoint.

| phase | cell (checkpoint × intervention × prompts × draws) | responses | H200 min | $ |
|---|---|---|---|---|
| 0 | setup, download dpo / s0500 / s2300, ΔW diagnostics, build adapters, write 2 ExPO checkpoints | 0 | 40 | 3.5 |
| 1 | one dpo load (LoRA enabled) | | 7 | 0.6 |
| 1 | C1 Cai rank-1(Δ2300), λ = 1: reach 48 × 8 + arm A 218 × 1 (16-prompt pilot first; stop if > 25 % cap hits) | 602 | 41 | 3.6 |
| 1 | C2 random rank-1, equal norm: reach 48 × 4 | 192 | 10 | 0.9 |
| 1 | C3 unscaled rank-1: reach 48 × 4 | 192 | 11 | 1.0 |
| 1b | conditional: C4 rank-16 unscaled, reach 48 × 4 (if C1 < 80 % of reach gain); C2 on arm A × 1 (if C1's arm-A gain is partial) | 410 | 24 | 2.1 |
| 2 | ExPO dev: dpo + γ(s0500 − dpo), γ ∈ {3.17 norm-matched, 4.6 label-step}, 64 EDL-train prompts × 2 (2 loads) | 256 | 34 | 3.0 |
| 2 | ExPO held-out at the chosen γ: reach 48 × 8 + the other 154 arm-A prompts × 1 | 538 | 42-49 | 3.7-4.3 |
| | **total, unconditional** | 1,780 | ≈ 3.1-3.2 h | ≈ $16-17 |
| | with conditionals and a 25 % runaway buffer | | | ≈ $22-24 |

That is above the $15 split. Ways to trim, in my order of preference:
1. Share the pod session and the dpo load with M1. A LoRA-enabled dpo engine can also serve M1's T cells; check that
   LoRA mode does not change base sampling. Saves about $1.5.
2. Use 32 dev prompts × 2 instead of 64 × 2. Saves about $1.5.
3. Use reach × 4 for C1. Saves about $1.8, with wider intervals.
4. Last resort: commit one step-matched γ without a dev sweep. That tests "naive linear extrapolation", not ExPO.

The rank-1 half alone (phases 0-1) is about $10.

How each coefficient is chosen without the reach prompts:
- Cai rank-1: λ = 1 by the paper's rule, no choice. Unscaled: none. Rank-16: fixed a priori to match the trained
  LoRA. Random control: seed fixed a priori.
- ExPO γ:
  - Grid fixed now from weight-space facts only. 3.17 = ‖Δ2300‖/‖Δ500‖ peeks at the endpoint's norm but at no
    behaviour; 4.6 = the label step ratio, no peek. 5.5 (offset steps) is optional.
  - Chosen by strict pass@1 on the 64 EDL-train prompts (arm-A non-reach and already used for the 64-example LoRA).
    Ties go to the smaller γ. Reject any γ with > 10 % cap hits.
  - Record the choice in a timestamped file before held-out sampling. Score once on the remaining 154 arm-A prompts
    and the 48 reach prompts.
- Any optional λ or κ sweep uses the same dev split and the same rule.
- Intervals: 95 % prompt-bootstrap intervals, paired against dpo and against s2300 on the same prompts, with the
  fraction of the gain recovered as the headline.

Things I could not check: whether vLLM 0.11's LoRA path supports lm_head for Olmo3; the dtype of Cai's checkpoints and
whether they merged their rank-1 into bf16; the cosine between Δ500 and Δ2300 (Dan's stored statistics have distances
from dpo and lag dot products only, not this pair); SVD timings on an H200.

END OF INDEPENDENT READ
