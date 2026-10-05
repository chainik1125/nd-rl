---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/sharpen-expand.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: sharpening vs expansion — selection bits and reshaping bits

Family D (distribution shift). Slug `sharpen-expand`. Notation: `_FRAME.md`.

## 1. Definition, formally

For a theorem t that RL solves, let q_R = π_R(· | t, success) be RL's distribution over its own valid proofs, and
q_B = π_B(· | t, success) the base's.

- **Pure elicitation has an exact form.** With a binary reward, the KL-regularised RL optimum is the base conditioned
  on success: π* ∝ π_B · exp(r / β), which tends to π_B(· | success) as β → 0, with normaliser Z = p_B(t) (Korbak et
  al. 2022). Conditioning **preserves the base's odds between any two correct proofs**: π*(y) / π*(y′) = π_B(y) /
  π_B(y′).
- **Decomposition** (Shenfeld et al., App. A): for any policy q that always succeeds on t,

  KL(q ‖ π_B) = log 1 / p_B(t) + KL(q ‖ q_B).

  The first term, the **selection bits**, is what pure filtering of base samples costs. It is log₂ of the k-to-solve,
  the pass@k view. The second term, the **reshaping bits**, measures how far RL's successful proofs depart from the
  base's own successful proofs.
- **ε-support split:**
  - sharpening mass S(t) = Σ_{y : π_B(y) ≥ ε} π_R(y), over valid y;
  - expansion mass E(t) = Σ_{y : π_B(y) < ε} π_R(y);
  - expansion share ρ(t) = E / (S + E), with ε = 1 / K (K_eval-set as implemented; K_per was the first default).
- **Estimators on the known proofs F(t)** (J1):
  - q̂_θ(y) = π_θ(y) / Σ_{F} π_θ;
  - reshaping bits ≈ KL(q̂_R ‖ q̂_B) restricted to F;
  - ρ̂ as above, restricted to F;
  - **after removing vacuous detours** (see `new-proof-new-theorem`);
  - coverage check: Σ_F π_R / p̂_R.

## 2. Decision rule

*Revised after the critic pass (§9).* The ε-support split degenerates on hard theorems: if p_B(t) < ε, every proof has
π_B(y) < ε, so ρ = 1 for any policy, including pure conditioning. The split therefore cannot be the creation test.

| verdict | rule |
|---|---|
| **gate (theorem level)** | first apply `marginal-bracket` at a declared K. If p_B ≥ 1 / K is certified, the theorem is elicited, whatever ρ says |
| **reshaped beyond the placebo** (the card's own contribution) | the reshaping bits KL(q̂_R ‖ q̂_B) exceed the range the replay-only control (no RL) produces on the same theorems |
| **amplified latent route** | reshaping bits within the placebo's range |
| **ρ** | reported as descriptive only ("RL's routes are ones the base rarely takes"), never as a verdict |

## 3. Null or floor

- The null is exact: pure conditioning has reshaping bits = 0 and preserves odds.
- The selection bits put the random-weights objection in numbers: for random weights, log 1 / p_0 ≈ 10³ bits.
- A placebo for reshaping: the replay-only control ladder (`rl-from-ckpt`), which also changes odds without RL.

## 4. How to compute it here

- **J1** scores (π_B, π_R for every known proof of the hard and calibration theorems; exact 33-base scores for the
  top proofs in stage 2) and **J2** (p_B by sampling).
- Analysis: ρ̂ in `cd_part3.py` (`sharp`); coverage in `cd_bracket.py`. The reshaping KL was not computed (the ρ
  results below made it moot as a decision quantity). CPU minutes.
- **Results** (cap 12, s0 / s1 / s2). Coverage of RL's success mass by the known proofs, median Σ_F π_R / p̂_R: r8
  0.69 / 0.48 / 0.71, r16 0.51 / 0.47 / 0.56. ρ ≥ ½ (r8, x0): **48 / 42 / 46** of the hard theorems r8 solves; net of
  replay 36 / 24 / 27; r16 55 / 59 / 50; redraw Jaccard 0.91–0.93, seed 0.32–0.38. ρ is near 0 or 1 on almost every
  hard theorem (threshold sweep 0.25 → 0.9: 50 / 42 / 48 → 37 / 38 / 36), so it says which side of ε RL's proofs fall,
  and agrees with equal-k at Jaccard 0.83; as the critic said, it is descriptive.

## 5. Sensitivity

- **ε (budget):** ρ is monotone in ε. Report the curve ρ(ε) for ε from 1/256 to 1/K_total.
- **Temperature:** use T 0.8 for both π (the sampler's distributions).
- **Representation:** routes are action sequences in the proof-state interface. A whole-proof model has different
  routes (premise restatements).
- **The detour filter** changes ρ (§6).
- **Renaming:** score per prompt.
- **Noise:** deterministic given F. F's completeness is the main uncertainty (coverage check).

## 6. Failure modes

- **F is a biased sample of the success set.** RL's samples dominate it. If the base's preferred routes are missing, q̂_B
  is wrong. J2's base samples are added to F for that reason.
- **The success set is not just F.** If Σ_F π_R ≪ p̂_R, ρ̂ describes only a corner of R's mass.
- **Vacuous variants** inflate "expansion" (§1 filter). Which differences between proofs are "real" is a choice.
- **Reshaping can come from replay pretraining,** not RL (the placebo bounds this).

## 7. Relations

- The selection bits are `passk-budget`'s log₂ k-to-solve.
- ρ is `new-proof-new-theorem`'s proof axis.
- The total KL(q_R ‖ π_B) is `kl-update-size`'s quantity on the success set.
- The odds-preservation test is the formal core of "RL only elicits" (Korbak), the cleanest *theoretical* definition of
  elicitation found in the literature.

## 8. Literature anchor

- **Korbak, Perez & Buckley 2022 (2205.11275).** The KL-RL optimum is the base reweighted by exp(r / β) (Sec. 4,
  Eq. 5). With a binary reward, conditioning keeps "all other strings … the original probability π0(x) up to Z"
  (Sec. 5). L5 note.
- **Shenfeld, Pari & Agrawal 2025 (2509.04259v1).**
  - The decomposition in App. A (proof of Lemma A.1).
  - "on-policy RL is implicitly biased towards KL-minimal solutions among the many that solve the new task" (abstract).
  - L3 note.
- **Huang et al. 2025 (2503.21878):** the coverage coefficient E_{π*}[π*/π_ref] (Sec. 2.1, Eq. 5). L5 splits it into
  (1/p_B)(1 + χ²), L5's own derivation.
- **Huang et al. 2024, sharpening (2412.01951;** earlier review): "cannot create information that is not already in the
  model".
- **Chen & Foster et al. 2025** (earlier review): the coverage principle.

## 9. Critic's verdict

**Strongest argument (critic): "created" never asked whether pend can prove t.**
- Counterexample: s0 `la_transfer_1100`. pend proves it in 106 of 768 attempts. r8 puts 47 % of its success mass on a
  14-line nested-∨E proof that pend writes with probability e^(−21.8), so ρ = 0.94 at K_per: "created".
- After pruning and detour normalisation, 15 / 30 (s0) and 11 / 30 (s1) calibration theorems are still "created", and
  pend solves every one of them.
- On hard theorems ρ = 1 for every policy: Korbak's pure-conditioning null is "created" on 146 / 152.
- Secondary arguments:
  - "Elicited" (≤ 1 reshaping bit) holds on only 4 / 30 and 9 / 30 calibration theorems (median 3.2 bits). EI has no
    KL term and uses replay, so it breaks the odds anyway.
  - The null's verdict depends on the scoring convention (b0 vs b0 − ln 33).
  - Padding games ρ.

**My answer: accepted.**
- ρ is demoted to a descriptive statistic.
- The theorem-level verdict comes from `marginal-bracket`.
- The card's own contribution becomes the reshaping bits, measured against the replay-only placebo. Pure conditioning
  gives exactly 0, so the reshaping bits measure something k-to-solve cannot: whether RL reorders the base's own
  routes more than non-RL training does.
- Computing the placebo's reshaping bits needs J1 scores under the replay-only control, which were not run. This is
  stated as not computed here.

