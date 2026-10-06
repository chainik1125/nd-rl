# NOISE_FLOOR.md — the smallest difference this project can resolve

**Standing reference. Quote it.** Measured by run `noise-floor` (proposal 11, 2026-09-24/25) from
**four raw pools drawn independently from the control's own generator settings, assembled to the
control's own composition, trained with the control's own schedule, × up to thirteen Stage-1 seeds**.
Every difference between those cells is noise by construction. Numbers below are derived from
`artifacts/nf/summary.json`; per-cell values, sources and the variance decomposition are in
`numbers.md` § noise-floor (§§ N1–N10) and the run write-up is `run_noise_floor.md`.

**What these numbers apply to.** A **3,214,336-parameter from-scratch GPT** in the `lean_seq` Lean
surface format at **cap 6**, `train.py --mode lean_seq --steps 6000 --bs 128`, trained on a
**155,000-record flat 31,000-per-length-2–6 set with depth-3 excluded** — i.e. the control of
`ds-composition`, `ds-generator` and `ds-rendering`, and the model every arm of proposal 10 was
matched against. They are **not** claimed for a pretrained model, a different cap, a different set
size, or a different model scale. A run at a different configuration should measure its own floor;
this one cost **28.3 pod-hours and $13.88**.

## The table

| quantity | cells | observed range over the null cells | pooled sd | **smallest difference worth reporting at n = 2** | as a ratio |
|---|---|---|---|---|---|
| frozen ladder, transfer theorems solved (of 2,285) | 8 | 62 – 265 (4.27×) | 74.06 | **397** (235 % of the mean) | 3.35× |
| frozen ladder, transfer `L*` | 8 | 9 – 10 (1.11×) | 0.3536 | **2** (21 % of the mean) | 1.21× |
| frozen ladder, RL targets solved (of 4,495) | 8 | 714 – 1765 (2.47×) | 391.2 | **2098** (167 % of the mean) | 2.67× |
| `targets_reductio_req` solved at pass@2,000 (of 300) | 8 | 6 – 46 (7.67×) | 13.23 | **71** (232 % of the mean) | 3.32× |
| `r3_1/depth3_req` solved at pass@2,000 (of 300) | 8 | 61 – 242 (3.97×) | 73.27 | **393** (257 % of the mean) | 3.57× |
| held-out greedy, overall (5,000) | 52 | 0.8420 – 0.9630 (1.14×) | 0.03038 | **16.3 pp** | 1.18× |
| held-out greedy, 6-line bin (1,000) | 52 | 0.4210 – 0.9200 (2.19×) | 0.1523 | **81.7 pp** | 2.22× |
| held-out greedy, depth-3 slice (500) | 52 | 0.0080 – 0.9180 (114.75×) | 0.3053 | **not resolvable** | 4.72× |
| held-out greedy, 6-line no-pattern (247) | 52 | 0.6964 – 0.8785 (1.26×) | 0.03667 | **19.7 pp** | 1.24× |
| held-out greedy, 5-line bin (1,000) | 52 | 0.8770 – 0.9560 (1.09×) | 0.01502 | **8.1 pp** | 1.09× |
| held-out greedy, 4-line bin (1,000) | 52 | 0.8920 – 0.9790 (1.10×) | 0.01512 | **8.1 pp** | 1.08× |
| held-out greedy, 3-line bin (1,000) | 52 | 0.9760 – 0.9970 (1.02×) | 0.00486 | **2.6 pp** | 1.03× |
| held-out greedy, 2-line bin (1,000) | 52 | 0.9870 – 1.0000 (1.01×) | 0.003128 | **1.7 pp** | 1.02× |

<!-- bimodality: b = (skew^2 + 1)/kurtosis, > 5/9 is bimodal-consistent -->
| quantity | bimodality b | high-mode proportion (Wilson 95 %) |
|---|---|---|
| frozen ladder, transfer theorems solved (of 2,285) | 0.546 | — |
| frozen ladder, transfer `L*` | 0.818 **bimodal** | — |
| frozen ladder, RL targets solved (of 4,495) | 0.623 **bimodal** | — |
| `targets_reductio_req` solved at pass@2,000 (of 300) | 0.515 | — |
| `r3_1/depth3_req` solved at pass@2,000 (of 300) | 0.875 **bimodal** | — |
| held-out greedy, overall (5,000) | 0.557 **bimodal** | — |
| held-out greedy, 6-line bin (1,000) | 0.677 **bimodal** | — |
| held-out greedy, depth-3 slice (500) | 0.697 **bimodal** | 24/52 = 0.462 [0.333, 0.595] |
| held-out greedy, 6-line no-pattern (247) | 0.603 **bimodal** | — |
| held-out greedy, 5-line bin (1,000) | 0.423 | — |
| held-out greedy, 4-line bin (1,000) | 0.510 | — |
| held-out greedy, 3-line bin (1,000) | 0.498 | — |
| held-out greedy, 2-line bin (1,000) | 0.648 **bimodal** | — |

**MDD** = the smallest difference a two-sample t-test at **n = 2 seeds per arm** has 80 % power to
detect at α = 0.05 two-sided: `5.364 · sd`. It is the **parametric best case**. Non-parametrically an
n = 2 vs n = 2 comparison can never reach p < 0.05 at all — the exact two-sided permutation test's
minimum attainable p is **1/3**. At n = 6 per arm the constant falls to `1.794 · sd`; at n = 13 to
`1.183 · sd`.

## How to use it

1. **Before designing a run**, look up the quantity you intend to read the result off and ask whether
   the effect you expect exceeds its MDD. If it does not, either change the quantity, raise n, or
   say in the pre-registration that the run cannot resolve its own question.
2. **When reporting**, state the difference *and* the floor beside it. "A1's frozen ladder is 205 vs
   the control's 158" is not a finding; "205 vs 158, inside a measured floor of ±235 %" is.
3. **`L*` is the readout to prefer over raw solve counts.** Across eight null cells the frozen
   ladder's solve count moves **4.27×** while `L*` moves by **one point** (9 in seven cells, 10 in
   one). A 1-point `L*` difference is noise; a 2-point one is at the edge of resolution.
4. **Short length bins are precise; the 6-line bin is not.** Held-out accuracy at lengths 2–5 has a
   floor of 1.7–8.1 pp. The 6-line bin and the depth-3 slice are **not resolvable at n = 2 at all**.
5. **For the depth-3 held-out slice, do not report a mean ± sd.** It is bimodal (bimodality
   coefficient 0.697; 24 of 52 cells above 0.44, 13 below 0.11, 15 in between). The estimand is the
   **probability a training run lands in the high mode: 0.462, Wilson 95 % [0.333, 0.595]** for the
   control configuration. An arm differs only if its high-mode proportion is outside that interval,
   which needs ≈ 30+ seeds.

## Three facts that change how runs should be designed

- **≈ 99 % of the variance is the individual training run**, not the data draw and not the seed
  draw. In a balanced 4-pool × 13-seed decomposition the pool variance component comes out
  **negative** (i.e. indistinguishable from zero) on every held-out quantity and on both coverage
  pools, and the seed component is a few per cent of the residual. **Consequence: re-drawing the
  data set is not a stronger replicate than re-seeding Stage-1**, so the cheap replicate (more
  Stage-1 seeds on one set) is also the right one.
- **The sampler and the checker contribute ≈ 0.** The control's frozen ladder measured **158** in
  `ds-composition` and **158** in `ds-generator` from the byte-identical checkpoint on different
  pods and GPU classes; its T1 ladder measured 856 and 890 (+4.0 %). All of the spread above is
  training, not measurement.
- **More seeds are cheap.** A Stage-1 run plus a held-out evaluation is **≈ 6.5 pod-minutes** at
  four concurrent jobs on an A40 — 40 extra models cost $2.1 here. A ladder or a pass@2,000 coverage
  run is 20–200× that, which is why the cheap quantities should carry more seeds than the expensive
  ones rather than the same two.

## What clears the floor today

Of twenty standing findings scored in `numbers.md` § N10, **two survive**: `ds-composition`'s
**A3 (cap 8)** frozen ladder (976 vs 158, 6.18× against a 3.35× floor) and its **`L*` of 11 vs 9**
(+2 against ±1.9). **Six** are outside the null range but below the MDD — `ds-generator`'s G2 writing
**zero** `redreq` proofs and its seed-0 frozen ladder, A3's `redreq` advantage, G1's seed-1
`required@8`, `lean-format`'s `L*` 11 vs 10, and A1's 6-line no-pattern gain — i.e. suggestive, worth
re-measuring at more seeds, not reportable as findings. **The other twelve are inside their floor**,
including every claim that one training set's composition, generator shape or rendering beat
another's in distribution.

Both survivors are the **cap-8 arm**, which is the premise of the sibling run `cap-horizon`; the
training cap remains the only manipulation this project has made that clears its own noise.
