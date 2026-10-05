---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - truong2026irsl
---

# Item Response Scaling Laws: IRT on checkpoints with empirical-probability (repeated-sampling) responses

Paper: [@truong2026irsl] (Truong, Tu, Schaeffer, Koyejo), "Item Response Scaling Laws: A Measurement Theory Approach
for Efficient and Generalizable Neural Scaling Estimation" (ICML 2026)
Source: arXiv 2606.07616v1 (29 May 2026; HTML rendering; read: abstract, Sec. 1-5, App. C-E; App. A-B figures not read)

## Learnings

- **Factorization (Abstract; Fig. 1).** IRSL "disentangles latent model ability from question characteristics,
  factorizing the scaling law estimation for M models and N questions" from O(M×N) to O(M+N); R_ij ≈ σ(θ_i − z_j)
  (1PL) or σ(d_j(θ_i − z_j)) (2PL).
- **Data (Abstract; Sec. 4.2-4.3).** Pre-training: "6,612 LM checkpoints and 37,682 questions from 10 benchmarks"
  (DataDecide: 25 data mixtures × 14 sizes, 4M-1B, 6-30 checkpoints per run). Test-time: "12 LMs and 120 questions
  from 4 benchmarks with up to 2,500 samples per question"; the 12 are R1-distills, QwQ-32B, Qwen3 and Gemma-3
  (App. E).
- **Beta-IRT (Sec. 3.3).** "replaces the standard Bernoulli loss with the Beta loss", with Beta mean σ(d(θ − z)) and
  φ "a precision parameter controlling the concentration of the Beta distribution around its mean". For repeated
  sampling, the binary tensor M × N × K is averaged: "This tensor is averaged across the sample dimension to yield an
  empirical probability response matrix" (Sec. 3.4).
- **Scaling and pass@k (Sec. 3.4).** "Empirically, we observe that the θ scales linearly with log FLOP" (Eq. 3,
  Fig. 12). Test-time scaling follows from item-level probabilities: pass@k(i, D) = (1/N) Σ_j (1 − (1 −
  σ(d_j(θ_i − z_j)))^k) (Eq. 4).
- **Sample efficiency (Sec. 4.1, Fig. 2).** In simulation, "Beta-IRT achieves reliable calibration with as few as 2
  test takers, requiring 30–60× fewer than Binary-IRT"; "Binary-IRT requires significantly larger sample sizes
  (M≥64)" — with simulated noise ε ~ N(0, 0.01²) that "mimics empirical uncertainty". "The 2PL model typically requires
  more test takers to achieve reliable calibration" (Sec. 4.3, fn. 3).
- **Floor (Sec. 4.3).** "We filter out questions with extremely low pass@1 as they offer no discriminatory power".
- **When it fails (Sec. 4.2; App. C).** On homogeneous benchmarks (BoolQ, HellaSwag, WinoGrande) "Beta-IRT fails to
  capture a predictive trend. We attribute this to benchmark homogeneity"; the Test Information Function is "a measure
  of how precisely a benchmark estimates model ability at each point on the ability scale" (App. C, Fig. 20).
- **Transfer of θ (Sec. 4.2-4.3; App. D).** θ estimated on an easy subset predicts the hard subset's scaling curve;
  convergent validity of θ across benchmarks "ρ = 0.80 between AIME 2024 and AIME 2025 (test-time)" (0.99 for ARC
  Easy/Challenge).
- **Positioning (Sec. 5).** "In human testing, a test-taker sample size of [~]100 is typically insufficient for IRT";
  "human testing increases the number of test takers, whereas LM evaluation leverages empirical probability". Future
  work: "exploring alternative probabilistic models (e.g., Beta-Binomial, zero-inflated models)". Caveat:
  "Difficulties calibrated under one evaluation setup may also not transfer to different conditions".

## Evidence and limitations

- The headline sample-efficiency result is a simulation with very small response noise; real repeated-sampling noise
  is binomial and largest near 0 and 1.
- The Beta likelihood is defined on (0, 1); exact zeros (questions a model never solves) need filtering or a
  zero-inflated / beta-binomial model — the paper filters (Sec. 4.3) and lists those models as future work (Sec. 5).
- Test-time study is small (12 models, 120 questions); 1PL is the primary model there.

## Connections and questions

- **Definition offered:** a checkpoint's ability θ in a 1PL/2PL item-response model fitted to continuous per-item
  success probabilities (token probabilities or repeated-sampling pass@1), shared across items of one construct and
  linear in log compute along pretraining.
- **New vs better access:** not addressed. *Our interpretation:* their transfer experiment is a ready decision rule.
  Estimate θ_r of an RL checkpoint on theorems inside the pretraining-calibrated range, predict its per-attempt
  success on theorems outside it (group C), and compare with observed counts: accurate prediction = RL moved along the
  pretraining axis; systematic under-prediction on a subset = item-level DIF, a candidate new capability. Eq. 4 also
  makes Dan's k a derived quantity: pass@k is a function of the per-attempt θ and item parameters, not a separate
  capability.
- **Null / floor:** handled by exclusion only. *My derivation for censored items* (not in the paper): if theorem j has
  0 successes in N_j attempts at pend, a 95 % bound is σ(a_j(θ_pend − b_j)) < 3/N_j; under the 2PL axis, an RL
  checkpoint with ability gain Δθ = θ_r − θ_pend (estimated on identified items) must satisfy
  logit p_rj ≤ a_j Δθ + logit(3/N_j). Observed logit p̂_rj above that bound, even with a_j at the largest discrimination
  seen on identified items, is axis-inconsistent (positive DIF). Each 10× more pend samples lowers the bound by
  ln 10 ≈ 2.3 logits (logit(3/512) ≈ −5.1; logit(3/10⁵) ≈ −10.4), which turns the project's survivor-style experiments
  ("0 base successes in ≥ 40,000 attempts while EI p ≥ 0.01") into a test that accounts for the RL model's general gain.
- **Transfer to our setting:** our reads are exactly their test-time format (checkpoints × theorems × k = 256 binary
  outcomes). Because attempts are i.i.d. given the checkpoint and temperature, use the **binomial likelihood on the raw
  counts** (exact, handles zeros) rather than Beta on averaged rates; switch to beta-binomial only if x0/x1 replicate
  reads show extra-binomial variance (the batch-size flips in `ATLAS.md` §6 suggest some). Start 1PL, then 2PL (≈ 150
  checkpoints suffice by their simulation, but checkpoints cluster by seed). Check θ ∝ log(Stage-1 step or tokens)
  along pretraining (their Eq. 3), then read r1…r16 as pretraining-step equivalents. Teacher-forced probabilities of
  reference proofs are the analog of their p_correct-choice responses and can extend measurement below the sampling
  floor (p0…p400), linked to the sampled items by concurrent calibration. Cost: CPU minutes; no new sampling except
  optional deeper pend reads on censored theorems. Failure modes: item parameters calibrated at T 0.8 may not hold at
  T 1.0 (their own caveat); the reference-proof probability is a lower bound on success mass, not the success
  probability (route dependence).
- Related notes: `polo2024tinybenchmarks.md`, `hofmann2025fluid.md`, `kazdan2025passk.md` (beta-binomial on counts,
  pass@k forecasting), `schaeffer2025powerlaws.md` (distribution of per-problem p drives pass@k laws),
  `chen2021evaluating.md` (unbiased pass@k).
