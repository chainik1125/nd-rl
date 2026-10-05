---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/compute-equivalent.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: compute equivalence — "RL is worth X× pretraining" (and X× base sampling)

Family C. Slug `compute-equivalent`. Notation: `_FRAME.md`.

## 1. Definition, formally

Two exchange rates, both asking what an alternative use of compute would achieve.

- **(a) Pretraining-compute equivalent** (CEG, Davidson et al.; intrinsic performance, Hilton et al.). Fit performance
  as a function of pretraining compute along the base's own trajectory (14 checkpoints per seed). The equivalent of an
  RL model R is the pretraining compute C′ at which the extrapolated trajectory reaches R's performance; the
  multiplier is M = C′ / C_pend. Performance can be:
  - IRT ability θ (`irt-ability`): set level, threshold-free;
  - per theorem, the logit of p_θ(t), or the log p of a fixed reference proof;
  - a set statistic such as solved@256.
- **(b) Base-sampling equivalent** (Brown et al.). k*(t) = the number of base attempts that matches R's pass@k_eval:
  pass@k*_B(t) = pass@k_eval_R(t). Compare k* × cost per attempt with RL's training compute.

## 2. Decision rule

| verdict | rule |
|---|---|
| **elicited / "equivalent to more of the same"** | M finite and modest (≤ 10×; the bar is a convention to be fixed by Dan), and the per-theorem PT trajectory is rising toward R's level |
| **created** | M undefined: the pretraining trajectory has flattened or is falling below R's level (no amount of extrapolated pretraining reaches R), or M > 100× |
| **neither** | R not better than pend |

For (b): RL is "amortised search" when k* × c_attempt < RL's compute, and does "more than search" when even k* exceeds
RL's compute (this is `passk-budget` again, priced in attempts).

Davidson et al.'s own caveat applies: when no amount of baseline scaling reaches the gain, "the CEG is not meaningful".
That case is exactly "created" here.

## 3. Null or floor

- Random init is the start of the trajectory (step 0).
- The exchange rate needs no k, so Dan's objection is moot. But it inherits the extrapolation's assumptions (log-linear
  in steps over late pretraining).

## 4. How to compute it here

- **(a) on θ** (`out/irt_c12.txt`). θ = α + β ln(step), fitted per seed on steps 3,000 … end. The slope β is 1.11 /
  1.22 / 1.20 per e-fold of steps.

  | cap | r1 | r4 | r8 | r12 | r16 |
  |---|---|---|---|---|---|
  | 12 (s0 / s1 / s2) | 2.0 / 2.1 / 2.6 | 4.4 / 4.1 / 5.0 | **6.2 / 4.7 / 7.1** | 7.6 / 5.7 / 6.7 | **8.6 / 6.7 / 7.5** |
  | 6 | 2.9 / 4.1 / 3.4 | 22.1 / 24.5 / 10.5 | **45.7 / 51.7 / 19.5** | — | **61.1 / 105.5 / 33.1** |

  The ladder's 4,800 fine-tune steps and ≈ 750 M training tokens (r8) are comparable to Stage-1's ≈ 445 M tokens; the
  replay alone is ≈ 476 M tokens (`rl-from-ckpt`). The pend replay-only control reaches θ 0.55–0.70, ≈ 1.6–1.8× PT
  length by the same fit. So 26–37 % of r8's θ gain (0.55 / 2.10, 0.70 / 1.91, 0.64 / 2.18; 27 / 39 / 27 % measured
  from each seed's own pend θ) is what the replay pretraining alone gives.
- **(a) per theorem.** Many theorems sit at 0 / 512 throughout late pretraining, so their slope is unidentified. Use
  the reference-proof log p trajectory (scored at every checkpoint) for those.
- **(b)** needs J2's large-k base counts: k*(t) ≈ ln(1 − pass@256_R) / ln(1 − p_B).
- **(c) the measured counterfactual, J7** (pre-registered): pend trained further on K12 (learning rate re-warmed to
  3 × 10⁻⁴, cosine to 3 × 10⁻⁵) for 28,054 / 31,462 / 33,535 steps, the probe's estimate of the r8 ladder's GPU time
  (the run was faster than the probe, so it used 60,863 A40-seconds, 82 % of the ladders' 73,838 GPU-seconds;
  *[review]* the ladders ran on RTX A6000, not A40, so the match is across GPU classes). Read like RL.
  - Solved of 322 at k 256 (x0 / x1): s0 220 / 214, s1 215 / 208, s2 219 / 224, against pend 232 / 230, 236 / 238,
    234 / 236 and r8 286 / 288, 286 / 284, 293 / 290. A lateral move: it gains 12–19 theorems pend misses and loses
    30–42.
  - θ (IRT, pend ≈ 0): 0.51 / 0.58 / 0.58 (+0.53 / +0.65 / +0.49 over each seed's pend), against replay-only 0.55 /
    0.70 / 0.64 and r8 2.10 / 1.91 / 2.18.
  - It solves 19 / 54, 18 / 51, 16 / 60 of the equal-k set (35 / 35 / 27 %) and 4 / 22, 4 / 21, 0 / 14 of the
    compute-matched set. **This card's set (compute-matched, and J7 fails t): 18 / 17 / 14**, net of replay 17 / 13 /
    8.
  - *[review]* pend had seen K12 for ≈ 3.7 epochs of pairs (`trajectory`'s Stage-1 logs: 3.69 / 3.72 / 3.72), not
    ≈ 20 times: the recipe batches by token budget, ≈ 190 pairs per step, not 1,024. "More compute on the same data
    buys little after ≈ 20 passes" is not supported as written. The θ extrapolation above remains only an exchange
    rate.

## 5. Sensitivity

- **Extrapolation form:** log-linear in steps is assumed. Late pretraining was not flat (`trajectory`: w1 rose 1.8–3.2
  nats from step 8,000 to the end), and the fit window changes M by ≈ 20–30 % (x0-only fit: r8 7.9 / 5.8 / 9.0).
- **What counts as RL compute:** the ladder includes replay pretraining. Use the replay-only control to separate it.
- **Temperature, decoding, representation:** inherited from the performance metric. A guided-read θ gives a different
  M.
- **Renaming, seed:** M varies 4.7–7.1 across seeds at r8, a 1.5× spread.

## 6. Failure modes

- **"Undefined" conflates "flat pretraining" with "different direction".** The CEG measures whether more pretraining
  helps, not whether RL is new. Davidson: "a high CEG might not indicate that the post-training enhancement
  significantly improves performance, but instead indicate that additional training compute doesn't improve
  performance".
- **Extrapolation beyond observed steps** is speculative (Hilton: "we do not think conclusions that depend on the
  precise fitted values of our scaling constants can be drawn with confidence").
- **One model family / size.** True intrinsic performance would need a ladder of model sizes.

## 7. Relations

- θ comes from `irt-ability`.
- (b) is `passk-budget` priced in attempts.
- `chain-reachability` explains why RL compute can buy more than pretraining compute (RL focuses on the frontier).
- `bits-over-null` with an early-checkpoint null is the bits version.

## 8. Literature anchor

- **Davidson et al. 2023 (2312.07413v1).**
  - CEG = "how much additional training compute would be needed to improve performance by the same amount as the
    enhancement" (abstract).
  - "The CEG is given by C′/C" (Sec. 2).
  - When an enhancement enables tasks "impossible for any model without it … the CEG is not meaningful" (Sec. 4).
  - Minerva "reaching a CEG of 30 in STEM benchmarks and 2400 in math benchmarks" (Sec. 5).
- **Hilton et al. 2023 (2301.13442v2):** intrinsic performance = "the minimum compute required to train a model of any
  size in the family to reach the same return" (Sec. 2.1).
- **Jones 2021 (2104.03113v2):** "for each additional 10× of train-time compute, about 15× of test-time compute can be
  eliminated" (Sec. IV-C).
- **Brown et al. 2024 (2407.21787v3):** constant multiplicative sample-budget offsets between models (Sec. 3.2).
- **Ruan et al. 2024 (2405.10938):** f-equivalent FLOPs on observational capability coordinates (L4 note).
- Verified in `_claims_L1.md` / `_claims_L4.md`.

## 9. Critic's verdict

**Strongest argument (critic): M measures sharpening, and ranks pure elicitation above pure creation.**
- θ is calibrated on pretraining checkpoints, so it rides on the ≈ 240 theorems pend already solves. 46 of the theorems
  pend fails were never solved by any pretraining checkpoint, so their difficulty is fixed by the prior.
- Synthetic counterexamples on our items:
  - pend plus *every* theorem it fails at 95 % gives θ +0.6–0.8, M 1.6–2.1× ("elicited");
  - pend with nothing new, but each theorem it solves even once in 512 sharpened to 95 %, gives M 4.1–6.3×, close to
    r8's 4.7–7.1×.
- r8's actual new solves alone give M 1.2–1.5×. Sharpening reproduces 78–104 % of r8's Δθ.
- Secondary arguments:
  - The slope's ±2 SE spans r8 = 2.7–17×, and the fit window moves s0 to 10.3×.
  - The real counterfactual is a longer pretraining run, not an extrapolation.
  - r8's ladder used 17–20× Stage-1's GPU-seconds to be "worth" 4.7–7.1×.

**My answer: accepted. The θ-extrapolation is kept only as a descriptive exchange rate**, with its ±2 SE range and the
caveat that sharpening dominates it. The decision becomes the measured counterfactual (critic's fix, run as **J7**):
- a pretraining continuation from pend that gets the r8 ladder's GPU-seconds;
- RL "created relative to more pretraining" what r8 solves at k 256 that the compute-matched continuation does not.

Pre-registered in `log.md` before launch.

