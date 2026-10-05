---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - ruan2024observational
---

# Observational scaling laws: capabilities as a low-dimensional linear space shared across model families

Paper: [@ruan2024observational] (Ruan, Maddison, Hashimoto, NeurIPS 2024)
Source: arXiv 2405.10938v3 (HTML rendering read: abstract, Sec. 1-7, App. B.3, C.1-C.2). Reader L1 screened the
abstract only (`_screen_L1.md`); this is the in-depth reading.

## Learnings

- **Hypothesis (Abstract; Sec. 3.1).** "language model performance is a function of a low-dimensional capability
  space, and model families only vary in their efficiency in converting training compute to capabilities"
  (Abstract). Formally (Sec. 3.1, Eqs. 3-5): σ⁻¹(E_m) ≈ βᵀS_m + α (downstream error E_m), S_m ≈ θ_f log(C_m) + ν_f
  (family f converts compute C_m to capability), B_{i,m} ≈ γ_iᵀS_m for simple benchmarks B with "orthonormal vectors
  γ_i". Because of Eq. 5, "these capabilities are not latent variables to be estimated for each model family, but are
  instead functions of fully observable properties (B)" — S is obtained by PCA of the benchmark matrix (Sec. 3.1;
  App. B.3: mean-centred, no scaling, K = 3).
- **Low rank (Sec. 3.2).** On "21 model families and a total of 77 models" (base models), "the top 3 PCs explaining
  ∼97% of the variance" and "the first PC alone explains nearly 80% of the variation in LM capabilities" (Sec. 3.2,
  Fig. 2); instruction-tuned models: "about 98.6%" (App. C.1). "PC-1 represents the "general capability" as a
  weighted average of all metrics"; PC-2 reasoning, PC-3 programming (Sec. 3.2).
- **Capability ∝ log compute within a family (Sec. 3.3).** PC-1 is linear in log training FLOPs within each family
  "(with R^2>0.9)" (Sec. 3.3, Fig. 3).
- **Link with a floor (Sec. 3.4, Eq. 6).** E_m ≈ hσ(βᵀS_m + α), where h "is the sigmoidal scale that accounts for the
  potential discrepancies in the floor performance"; they "restrict h∈[0.8,1.0], which results in h*=1 in most
  experiments" (Sec. 3.4).
- **Compute-equivalent units (Sec. 3.4, Eq. 8).** Capabilities are re-expressed as "f-equivalent FLOPs": "how many
  FLOPs (C̄_m,f) would it take for a model in a family f to match a model m" (Llama-2 as reference family).
- **Emergence (Sec. 4.1).** With ~100 models, BIG-bench tasks labelled emergent follow smooth sigmoids in the
  capability measure and are forecast from near-random models; "When there are only 5 models across many orders of
  magnitudes of scale, phenomena can appear to be discontinuous, even if the underlying phenomenon is a smooth but
  rapidly varying sigmoid" (Sec. 4.1).
- **Post-training interventions (Sec. 4.3).** They "fit one observational scaling law using base model performance
  on a target benchmark … and then fit another on the performance of models with the post-training intervention"
  (CoT, self-consistency); the gap as a function of capability is the intervention's scaling (Sec. 4.3, Fig. 6).
- **Dynamic range (Sec. 6; App. C.2).** Single benchmarks "saturate quickly for large models … or have completely
  random performance for small models"; PC-1 is usable across ≈ 5 orders of magnitude (Sec. 6, Fig. 8; App. C.4).
- They note the link: the low-rank structure "interestingly connects to the item response theory in
  psychometrics" (Sec. 6). Limitation: extending to "other post-training setups, including scenarios involving
  fine-tuning or more intensive inference-time computation" is future work (Sec. 7).
- Model-subset selection by V-optimality, with "the expected prediction error from using the subset X_M is
  Tr(XᵀX(X_MᵀX_M)⁻¹)" (Sec. 5, Eq. 9).

## Evidence and limitations

- Holdout is weak→strong by FLOPs (47 train / 30 test models below/above Llama-2-7B compute) plus preregistered
  forecasts on 20 later models (Sec. 4).
- Capability measures are PCA of aggregate benchmark scores, not item-level; nothing in the method tests whether a
  post-trained model leaves the base-model capability space — interventions are only modelled as a second sigmoid on
  the same S.
- Fine-tuning and RL are explicitly out of scope (Sec. 7).

## Connections and questions

- **Definition offered:** capability = a model's coordinates S_m in a low-dimensional linear space extracted from
  standardized benchmark scores (PCA), optionally collapsed to one scalar per downstream task and expressed in
  "f-equivalent FLOPs" of a reference family.
- **New vs better access:** not addressed. *Our interpretation:* the framework yields two quantitative versions.
  (a) **Compute-equivalent:** if an RL checkpoint's position lies on the pretraining trajectory S(step), its gain can
  be stated as equivalent pretraining tokens (Eq. 8 analog) — "more of the same". (b) **New direction:** if the
  RL checkpoints need an extra principal direction (residuals of the pretraining-fitted space that are near zero for
  every pretraining checkpoint but grow with RL rounds), RL created a dimension that pretraining does not vary on.
  Their Sec. 4.3 design (separate curves for base and intervened models against base capability) maps onto the
  project's rl-from-ckpt ladders started at several Stage-1 checkpoints: RL gain as a function of the start's
  capability.
- **Null / floor:** the h parameter allows a lower asymptote (random guessing); PC-1 is preferred precisely because
  single benchmarks give random readouts for weak models. No treatment of a model that is at the floor everywhere.
- **Transfer to our setting:** (1) Build S from the pretraining checkpoints (p50…pend, 3 seeds × cap6/cap12) on
  per-theorem-class logit success rates (classes by length, ∨E/¬I boxes, classical-only), with k = 256 counts giving
  the rates; project r1…r16 and GRPO checkpoints. (2) Fit S_m ≈ θ_f log(step) + ν_f with "family" = recipe/seed;
  read RL positions as pretraining-step equivalents (requires extrapolating beyond pend; ladder replay itself
  pretrains — the rl-from-ckpt finding). (3) Test for a new direction: refit PCA with RL checkpoints included and
  check whether an added component has near-zero variance across pretraining checkpoints. Cost: CPU seconds; uses
  existing reads. Failure modes: aggregation into classes hides item-level DIF; ≈ 30 theorems are never solved and
  carry no variance; checkpoints are not independent (3 seeds), so PCA variance shares are descriptive only; the
  fixed-sigmoid link σ⁻¹(E) breaks down for classes at exactly 0 (needs a continuity correction or a binomial model,
  see `polo2024tinybenchmarks.md`, `truong2026irsl.md`).
- Related notes: `davidson2023retraining.md`, `hilton2023singleagent.md` (compute-equivalent gains),
  `burnell2023revealing.md` (factor structure), `schaeffer2023mirage.md`, `hu2023passuntil.md` (emergence and metric
  resolution).
