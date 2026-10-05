---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - schaeffer2025generative
---

# Pretraining Scaling Laws for Generative Evaluations — pass@k vs compute and vs gold-reference likelihood

Paper: [@schaeffer2025generative]
Source: arXiv 2509.24012v2 (HTML rendering read: abstract, Sec. 1-3.1, 5, 8, App. B). Authors: Schaeffer,
Levi, Miranda, Koyejo.

## Learnings

- **Three covariates for pass@k.** "Our three scaling laws differ in the covariates used: (1) pretraining
  compute, (2) model parameters and pretraining tokens, (3) log likelihoods of gold reference solutions"
  (Abstract). Pythia checkpoints, 14M-12B, ~5 orders of magnitude of compute, C ≈ 6ND; 128 GSM8K and 128
  MATH problems; temperature 1.0 ("We used temperature-only sampling at τ=1.0", Sec. 2).
- **k reshapes the law.** "generative evaluations introduce new hyperparameters (in our setting, k) that act
  as a control lever for scaling behavior" (Abstract). Compute law: −log(pass_B@k)(C, k) = E_0(k) +
  C_0(k)/C^α(k) (Sec. 3.1, Eq. 3); "the irreducible error term E_0(k) falls roughly exponentially with k and
  is effectively 0 by k≈1×10^2" and the exponent rises "from 1.21×10^-1 to 3.75×10^-1" (Sec. 3.1, Fig. 2).
- **Gold-reference likelihood law.** "We calculated the average log-likelihood of these gold reference
  sequences to use to predict pass rates" (Sec. 5, Eq. 5) and fit −log(pass_B@k) = ξ_0(k) +
  K_0(k)·[−log(GoldProb_B)]^κ(k) (Sec. 5.1, Eq. 6). Its parameters "converge to their final values using
  models up to ∼5 orders of magnitude cheaper than the target" (Sec. 5.1, Fig. 7), whereas compute-law
  parameters need checkpoints within ~2 orders ("reliable prediction requires checkpoints within ∼2 orders of
  magnitude of the target", Fig. 3 caption). Predictive accuracy is comparable across laws: "the compute law
  predicts slightly worse for small k and the gold reference law predicts slightly worse for large k"
  (Abstract).
- **Open question they flag — exactly ours.** "it is not immediately obvious why the likelihood of the
  specific benchmark-provided gold reference correlates so strongly with the pass rate", and "(ii) to what
  extent does this signal remain robust under heavy optimization pressure?" (Sec. 8).
- **pass@k as a continuous quantity.** "pass-at-k is a continuous probability derived from the model's
  generative distribution" (Sec. 2).
- **Resolution and adaptive sampling.** "if we draw n samples per problem, then any pass rate on that problem
  below 1/n will likely appear to be 0"; they "drew a minimum of 2^14 samples per model per problem, and then
  continued sampling until 10 successes were obtained or until a maximum of 2^15 samples were drawn" (App. B);
  ~500M samples for GSM8K and ~400M for MATH (App. B).
- **Limitation.** "The primary limitation of this work is its empirical focus on a single model family
  (Pythia)" (Sec. 8).

## Evidence and limitations

- Evidence: Fig. 1-3 (compute law), Fig. 6-7 and 13 (gold-reference law), App. B-D.
- Eq. 5 as rendered averages p_θ(gold | problem) over problems while the sentence says "average
  log-likelihood"; we could not resolve from the HTML text whether the average is over probabilities or
  log-probabilities — check the PDF before reusing the exact form.
- Pretraining only; no RL or fine-tuned checkpoints, so the "optimization pressure" question is open.
- Not checked: Sec. 4, 6, 7, App. E-G.

## Connections and questions

- **Definition offered:** capability on a generative benchmark = pass_B@k as a function of k; its
  predictors are pretraining compute, (N, D), or the likelihood the model assigns to gold reference solutions.
  The k-dependence of the law (E_0(k) → 0 by k ≈ 10^2) is itself a finding.
- **New vs better access:** not addressed for RL, but the paper gives the most directly usable null model
  for our question: on pretraining checkpoints, pass@k is a fixed function of gold-reference likelihood
  (Eq. 6). For RL checkpoints, compare observed pass@k with the pass@k that the pretraining-fitted Eq. 6
  predicts from the RL model's own gold-reference likelihood. RL above the curve = RL raised pass@k without
  raising the likelihood of the reference proof (it found other proofs, or concentrated mass — sharpening /
  access); RL on the curve = RL moved along the same likelihood-capability relation as pretraining. This is
  exactly their open question (ii).
- **Null / floor:** E_0(k) → 0 with k is the "any k solves it" phenomenon in their data; the likelihood
  covariate handles the floor naturally: the random-init model's gold likelihood is ≈ V^(−L) and lies far
  out on the x-axis.
- **Transfer to our setting:** (1) For pretraining checkpoints (if kept) and pend, compute teacher-forced
  log p of each eval theorem's reference proof and per-theorem pass@k from the 256-sample counts; fit Eq. 6
  per k (and a per-theorem analogue). (2) Place r8 and r16 on the same axes. Cost: one forward pass per proof
  per checkpoint plus existing sample counts — cheap. Failure modes: our proofs have many valid variants, so
  reference log p is a lower bound on log p_i that can move opposite to pass@k after RL; a single family /
  dataset (as in the paper) limits generality; the HTML ambiguity of Eq. 5 above.
- Related: schaeffer2023mirage.md (smooth metrics), hu2023passuntil.md (loss as assistant, App. A.2),
  schaeffer2025powerlaws.md, brown2024monkeys.md, jones2025forecasting.md (forward-pass proxy caveat).
