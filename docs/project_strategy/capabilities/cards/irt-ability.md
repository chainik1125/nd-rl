---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/irt-ability.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: IRT ability — one latent scale for every checkpoint and theorem

Family P (psychometric). Slug `irt-ability`. Notation: `_FRAME.md`.

## 1. Definition, formally

- **The model.** Every (checkpoint m, theorem i) read is a binomial count c_mi of n_mi attempts. A 2PL item response
  model:

  P(one attempt succeeds) = σ(a_i (θ_m − b_i)), with c_mi ~ Binomial(n_mi, that probability),

  where θ_m is the checkpoint's **ability**, b_i the theorem's **difficulty** and a_i its **discrimination**. MAP fit
  with weak priors (θ, b ~ N(0, 4²); log a ~ N(0, 0.5²)).
- **"More of the same" is defined by calibration on pretraining only** (specific objectivity):
  - Item parameters come from the 14 pretraining checkpoints of each seed.
  - Every RL checkpoint is then placed on that scale by MAP with the items fixed.
- **Item-level departure (DIF+).** RL success on item i beyond the 1-D prediction at the RL ability: the one-sided
  binomial tail P(X ≥ c | n, p_pred) < 10⁻³ and a smoothed log-odds residual > ln 10.
- **IRT-created set.** DIF+ items that the base fails (pend 0 / 512).
- **Set-level departure.** The deviance of RL cells under the pretraining items vs under items refitted on all
  examinees. Optionally a 2-D model: does RL load on a second dimension?
- **Ability-as-horizon variant** (METR, ADeLe). The difficulty at which success is 50 %, on an external axis (L_true
  or term size), and the slope.

## 2. Decision rule

*Revised after the critic pass (§9).* The placebo must be **matched on ability gain** Δθ. "Beyond the calibration set"
grows with Δθ for any kind of training, so an unmatched placebo (the original rule) compared RL's Δθ ≈ 2.1 with the
replay-only control's Δθ ≈ 0.6.

| verdict | rule |
|---|---|
| **elicited ("more of the same")** | calibrate items through the start checkpoint S. RL's created rate (DIF+ items S fails, per S-failed item the arm solves at p̂ ≥ 0.05) is within the range of the **pretraining continuation and replay-only arms from the same S at comparable Δθ** |
| **created** | RL's created count / rate exceeds both matched placebos by more than their seed spread, on ≥ 2 / 3 seeds. Set level: RL needs a second dimension that a matched placebo does not |
| **neither** | no ability change |

Items that no pretraining checkpoint ever solves have prior-determined parameters. They are handed to `passk-equal-k`
rather than judged here.

## 3. Null or floor

- Random initialisation sits at the floor of the scale (θ = −25.6 under the prior; it solves nothing). The scale is
  anchored by pretraining progress, not by a k.
- The random-weights objection becomes "init has the lowest ability". There is no k in the model: pass@k = 1 − (1 −
  σ(a(θ − b)))^k, and k only shifts the solved@k boundary by ln k / a logits (L4's derivation).
- The **placebo** (replay-only ladders) is the null for item-level claims.

## 4. How to compute it here

- **Data.** All existing plain reads (cap 12: 72 examinees, 322 items, 11.4 M attempts; cap 6: 69 examinees), plus 12
  placebo / comparison examinees from `rl-from-ckpt`.
- **Code.** `capability_defs/analysis/cd_irt.py`, ≈ 1–2 min of CPU per fit.
- **Results (cap 12, pooled draws, `out/irt_c12.txt`).**
  - **Ability.** θ(pend) ≡ 0; θ(r8) = 2.10 / 1.91 / 2.18 and θ(r16) = 2.46 / 2.35 / 2.24 (s0 / s1 / s2). Pretraining:
    p1600 −3.6 to −3.8, p5000 −1.6 to −2.1, p12000 −0.6 to −1.5.
  - **Placebo examinees.**
    - Replay-only control from pend: θ 0.55 / 0.70 / 0.64.
    - EI from p5000: 1.08 / 1.18 / 1.33.
    - EI from p1600: 0.80 / 0.46 / 0.66.
    - J7, pend's compute-matched pretraining continuation: 0.51 / 0.58 / 0.58.
  - **Fit.** Deviance per cell: RL cells 330.4 under the pretraining items vs 84.8 under items refitted on all; a 2-D
    fit brings them to 66.1 (`out/irt_c12.txt`). The 1-D pretraining scale does not describe RL's item profile.
  - **DIF+ items per RL examinee:** 58–77. Placebo (replay-only, from pend): 105–117. DIF+ alone is therefore not
    evidence of anything RL-specific.
  - **IRT-created** (DIF+ and pend 0 / 512): RL 26 / 26 / 30 at r8 and 33 / 35 / 37 at r16 (an
    earlier fit with one examinee fewer gave 32 / 36 / 35: the DIF counts move by ±2 with optimiser noise). Replay-only placebo:
    4 / 7 / 7. EI from p5000: 28 / 23 / 26.
  - **Redraw floor** (fit on x0 only vs x1 only): created at r8 28 / 26 / 28 vs 32 / 26 / 31.
  - **Spearman** between the 1-D prediction and observed p̂: r8 0.57 / 0.59 / 0.56, r16 0.57 / 0.50 / 0.52 (Q8a: miss).
  - **s1 r16:** 5 of the 6 holdout250 A ∨ ¬A instances are in its top decile of residuals.

## 5. Sensitivity

- **Temperature and decoding:** the counts are plain at T 0.8. A guided read is a different examinee.
- **Thresholds:** 10⁻³ and ln 10 are conventions; the placebo calibrates them.
- **Representation:** the scale is per interface and per cap. Cap-6 and cap-12 items get separate calibrations.
- **Renaming:** each prompt is an item. Renaming classes could be pooled as testlets.
- **Seed and redraw:** the created counts move by 0–6 between draws, and seeds share Jaccard 0.35–0.44.
- **Floors:** items no model solves and init's ability are prior-determined. METR / IRSL's fix (anchor items,
  teacher-forced responses) was not done here.

## 6. Failure modes

- **The 2PL is misspecified even for pretraining** (deviance 27 per cell against ≈ 1 for a fitting binomial model). DIF
  is relative to a wrong model, which is why the placebo matters.
- **Ceiling effects.** Most theorems are near p = 1 at r8, so rankings are noisy there.
- **Extrapolation.** RL abilities lie beyond the pretraining range (θ 1.9–2.5 vs pend ≈ 0), so the 1-D prediction is
  extrapolated.
- **The verdict depends on calibrating with pretraining.** A different calibration population (e.g. including
  replay-only ladders) changes what "more of the same" means.

## 7. Relations

- θ gives `compute-equivalent` (RL in pretraining-step units).
- The IRT-created set agrees most with `sharpen-expand` (Jaccard 0.58 at r8) and `passk-equal-k` (0.50); the
  budgeted definitions agree with each other more (bracket vs best known proof 0.90; REPORT §3.3).
- DIF on a schema's items is `schema-acquisition` seen statistically.
- The horizon variant is `transfer-invariance` along length.

## 8. Literature anchor

- **Polo et al. 2024, tinyBenchmarks (2402.14992):** a multidimensional IRT calibrated on one population and tested on
  shifted models (L4 note, Sec. 4.2–4.4).
- **Hofmann et al. 2025:** a 2PL MAP θ for 61–94 pretraining checkpoints per run; a pooled axis hid a falling ability
  (L4 note, App. D–E).
- **Truong et al. 2026 (IRSL):** IRT on repeated-sampling pass@1, with pass@k as a function of θ (Eq. 4).
- **Kwa et al. 2025 (2503.14499), METR:** 50 % time horizon.
- **Zhou et al. 2025 (2503.06378), ADeLe:** ability = demand level at 50 %.
- **Martínez-Plumed et al. 2016/2019:** IRT over classifiers.
- **Burden et al. 2023 (2309.11975v2):** σ(capability − demand) "allows us to interpret a capability with value x as
  consistently succeeding on the demand with meta-feature x in 50 % of instances" (Sec. 6).
- **Hernández-Orallo et al. 2021:** capability = area under the success-vs-difficulty curve.
- **Wright & Linacre (rasch.org):** specific objectivity.
- Verified in `_claims_L4.md` / `_claims_L2.md`.

## 9. Critic's verdict

**Strongest argument (critic): "created in excess of the placebo" measures how far a checkpoint sits beyond the
calibration set, not RL.**
- Calibrated on p0–p12000, pretraining's own pend "creates" 22 / 32 / 26 items, against RL r8's 26 / 26 / 30.
- From p5000, the replay-only control r8 (Δθ ≈ +2.1) gives 50 / 49 / 54, against EI r2 (Δθ +2.0–2.7) 48 / 41 / 48.
- The original "26–30 vs 4–7" compared a Δθ of 2.1 with one of 0.6.
- Secondary arguments:
  - "pend 0 / 512" is a pass@512 test; 46 never-solved items get prior-fixed parameters.
  - Items pooled over seeds let one seed's base overrule another's (e.g. `la_transfer_372`).
  - The set-level deviances changed once more examinees were added (now 330.4 → 84.8 → 66.1).

**My answer: accepted. I re-derived the matched comparison** (`analysis/cd_irt_matched.py`,
`out/irt_matched.txt`), and it reproduces the critic's counts exactly.
- At matched Δθ, EI's created counts and rates sit inside the range of pretraining continuation and replay-only
  training from the same start.
  - From p12000: EI r2 (Δθ 1.7–2.2) 37 / 46 / 46, rate 0.69–0.93; replay r8 (Δθ 1.6–2.1) 33 / 47 / 48, rate 0.89–0.92.
  - From p5000: EI r2 (Δθ 2.0–2.7) 48 / 41 / 48, rate 0.62–0.86; replay (Δθ ≈ 2.1) 50 / 49 / 54, rate 0.73–1.04.
  - Pretraining's own continuation to Δθ ≈ 0.6 already "creates" 47 / 60 / 54 from p5000.
- **Under the revised rule the IRT definition finds no RL-specific capability on our data: RL looks like more training
  of any kind.**
- The decision rule is changed (§2); the original DIF counts stay in §4 as descriptive numbers, not verdicts.

