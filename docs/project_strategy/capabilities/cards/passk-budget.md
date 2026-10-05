---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/passk-budget.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: pass@k at a budget tied to RL's compute (Dan's notion (b), made strict)

Family S (sampling). Slug `passk-budget`. Notation: `_FRAME.md`.

## 1. Definition, formally

- **Quantity.** The single-attempt solve probability p_θ(t) of model θ on theorem t under a fixed protocol (plain
  sampling, T 0.8, the read caps `max_steps` 96 / `max_action` 512, Lean alone judges). From n attempts with c
  successes:
  - p̂ = c / n, with a one-sided 95 % Clopper–Pearson interval; with c = 0 the upper bound is ≈ 3 / n.
  - pass@k for k ≤ n by the unbiased estimator 1 − C(n − c, k) / C(n, k) (Chen et al. 2021, Eq. 1).
  - For k > n **no unbiased per-theorem estimate exists**. Beyond the sampled n, use the interval on p, the
    known-proof estimate of `marginal-bracket`, or (for set-level statements only) a fitted distribution of p over
    theorems (beta-binomial or zero-inflated; Kazdan et al. 2025; Schaeffer et al. 2025).
- **k-to-solve** K_θ(t) = 1 / p_θ(t), the expected number of attempts to the first success. **Compute-to-solve** =
  K_θ(t) × the cost of one attempt. Measured for pend on hard theorems: 3.6 ms of A40 time per attempt (J2 chunk 0:
  131,072 attempts in 470 s).
- **The capability statement:** "θ can solve t within budget K" ⟺ p_θ(t) ≥ 1 / K, i.e. at least a 63 % chance of a
  proof in K attempts.
- **The budget is tied to RL's own compute**, converted to base attempts (ladder GPU-seconds divided by the measured
  cost of one base attempt). *Revised after the critic pass (§9).* The headline budget is **K_eval-set**: RL's compute
  spread over the theorems being judged, i.e. "what if RL's GPU-seconds had been spent sampling the base on exactly
  the evaluation set":

  K_eval-set = ladder GPU-s / (N_eval × cost per base attempt),

  with N_eval = 322 (textbook72 + holdout250) and the cost measured on pend (3.6 ms per attempt on an A40, J2):
  19,281 / 21,107 / 23,310 attempts per theorem at r8 and 47,962 / 49,428 / 50,447 at r16 (s0 / s1 / s2). J2 supplies
  it: 16,384 attempts on every hard theorem, and 49,152 more on each that stayed at 0. Two reference budgets are
  reported beside it:
  - **K_per:** RL's compute per *training* target. It depends on how many targets RL trained on, which the judged
    theorems are not: 256 if counted in samples (8 rounds × 32); 777 / 731 / 954 in GPU time at the ladder's read
    cost (6.3–7.5 ms), 1,381 / 1,512 / 1,670 at 3.6 ms.
  - **K_total:** all of RL's compute on one theorem. It can certify elicitation but never creation.
- *[review]* **GPU classes.** The r8 ladders ran on RTX A6000 (`trajectory`'s `compute.json`), and 3.6 ms is an A40
  cost, so K_eval-set divides A6000 seconds by an A40 per-attempt cost. A same-class budget (pend's per-attempt cost
  on an A6000, or K_eval-set in ladder attempts, ≈ 5.6–6.2 × 10³ per theorem) is open.

## 2. Decision rule

**Set level (the headline, after the critic).** Compare the RL model's solves at k_eval = 256 with the base's coverage
at the compute-matched K_eval-set, over the same theorems:

Δ_cov = #{t : R solves t at 256} − #{t : B solves t within K_eval-set}.

- RL **created coverage** if Δ_cov > 0 beyond the seed spread, on ≥ 2 / 3 seeds.
- RL **elicited** if the base's compute-matched coverage meets or exceeds RL's.

The theorems in the difference are the candidate created set. The test can come out either way.

**Per theorem (secondary).**

| verdict | rule |
|---|---|
| **created at K** | R solves t at k_eval, and UB95(p_B(t)) < 0.05 / K |
| **elicited at K** | p_B(t) ≥ 1 / K is certified, by the sampling lower bound or by `marginal-bracket` |
| **neither** | R does not solve t, or B already solves it at k_eval |
| **undetermined** | otherwise |

**Attribution.** The ladder also pretrains (≈ 476 M replay tokens). Every verdict is reported net of the replay-only
control (`rl-from-ckpt`'s c⟨s⟩_pend_r8: the same 8 rounds of fine-tuning with no RL proofs). It solves 16 / 18 / 24 of
the 86 / 77 / 77 hard theorems.

**Continuous version ("bits beyond search"):** β(t) = log₂ p_R(t) − log₂ p_B(t) − log₂ K. It does not agree with the cut
everywhere (§9), so report it as a separate scale.

## 3. Null or floor

- Random initialisation is the null. At init, 96 % of attempts fail to parse; the rl-from-ckpt ladder started at
  step 0 accepted 0 of 287,680 attempts; and init's teacher-forced log p of a reference proof is about −200 nats in
  its worst step alone (trajectory).
- So K_null = 1 / p_0(t) is of order e^(hundreds to thousands), while K_total ≈ e^15.
- **Dan's objection ("some k makes random weights solve everything") is true but irrelevant once K is tied to a real
  compute budget.** Random weights solve nothing within any budget RL itself could afford. The line that matters is
  where the base sits relative to K_per and K_total.

## 4. How to compute it here

- **Data.**
  - Existing per-theorem reads: pend and r8 at x0 + x1 (+ mcts-a x2, and x4 for C), k 256 each; r16 at x1 and x0 (J5).
  - J2 stage A: pend at 16,384 attempts on each seed's hard RL-solved theorems (57 / 59 / 55) and at 4,096 on 30
    calibration theorems per seed; stage A′: 16,384 on the hard theorems no RL read solves (29 / 18 / 22).
  - J2 stage B: 49,152 more (65,536 in all) on the stage-A zeros (30 / 37 / 21). Theorems with a stage-A success keep
    ≈ 17,152 attempts, which already puts p̂ above 1 / K_eval-set. UB95 at 0 / 66,304 = 4.5 × 10⁻⁵ certifies "created"
    only for K ≤ ≈ 1,100.
  - J9: ≈ 1.2–1.4 M more attempts on 3 theorems per seed, enough to certify at K_eval-set (r8).
  - Cost: J1 + J2 ≈ 4.3 A40-hours per seed (≈ 60 % of the r8 ladder's GPU time); J9 ≈ 2.4 A40-hours per theorem
    (21.3 for nine).
- **Analysis.** `cd_part3.py` (the set: `cm`, `cm0`, `cm_recipe`, nets; Δ_cov), `cd_bracket.py` (the bracket),
  `cd_report_tables.py` (tables), a few CPU-minutes.
- **Result (cap 12, r8, draw x0; s0 / s1 / s2).** Set level: r8 at 256 solves 286 / 286 / 293 of 322; the base reaches
  265 / 270 / 281 at K_eval-set; **Δ_cov = +21 / +16 / +12** (r16: +24 / +32 / +14). The replay-only ladder at 256
  solves 239 / 244 / 249. Per theorem: equal-k 54 / 51 / 60 → not within reach at K_eval-set **22 / 21 / 14** (redraw
  Jaccard 0.84–0.87, seed 0.16–0.25) → net of replay 20 / 14 / 8 → 0 base successes 18 / 17 / 8 → no seed's base ever
  solved 4 / 5 / 5 (net 4 / 3 / 5). **Certified created (J9): 7 of the 9 headline candidates** (0 successes in
  1.25–1.51 M attempts each): s0 `la_transfer_2060`, `la_transfer_205`, `la_transfer_1077`; s1 `la_transfer_1648`,
  `la_transfer_2060`; s2 `la_transfer_1833`, `la_transfer_2060`. The other two were found once in ≈ 10⁶ attempts.
  Net of every RL-free control (other seeds' bases, replay-only, the compute-matched continuation) four remain:
  `la_transfer_205` (s0), `la_transfer_2060` (s1, s2), `la_transfer_1833` (s2).
- **Structural limit.** Certifying creation at K_total needs ≈ 60 K_total ≈ 2 × 10⁸ base attempts per theorem. That
  is ≈ 200 A40-hours per theorem, so creation at K_total cannot be certified by sampling. Only elicitation can be
  certified there.

## 5. Sensitivity

- **Temperature and decoding.** p depends on T. Chen et al. find higher T better at large k (Sec. 3.3). Guided
  sampling (step-checked redraws) raised solves by 5–9 pp at matched tokens (`guided-tts`). The protocol must be named,
  and plain and guided are reported separately (AGENT_POLICY).
- **k and thresholds.** K_per and K_total differ by a factor of 4,495 (the number of training targets), and verdicts
  near either line flip. The known-proof estimates of the hard theorems RL solves span 10^−16.8 to 10^−2.3, with
  medians 10^−4.6 / 10^−5.3 / 10^−4.5: right at 1 / K_eval-set. A tenfold budget change moves the set 1.7–2.6×
  (REPORT §3.4).
- **Representation.** The same 29 theorems were unreachable for the whole-proof base in 4 × 10⁵ attempts but reached by
  the proof-state base (`support-state`; *[review]* 3.2 M models, not best-cap12). p is a property of (model, interface, decoder).
- **Renaming, premise order, curried forms.** p is measured per prompt. A capability claim about a theorem should
  average p over its renaming class. The sampler's 33 name bases are already averaged over.
- **Seed and redraw noise.** Binomial sampling error is exact and small at large n. Seed-to-seed variation is large:
  for the k 256 version, the Jaccard of the created sets is 0.68–0.78 between redraws but 0.34–0.39 between training
  seeds (`out/defs_c12.txt`).

## 6. Failure modes

- **Budget choice is a modelling decision.** Should K count sampling only, or sampling plus training? Per target or in
  total? Should it include the replay pretraining that the ladder also does (≈ 476 M tokens, `rl-from-ckpt`)? Each
  choice moves the line by orders of magnitude.
- **Group selection.** If the created set is defined on one draw and p_B is estimated on the same draw, p_B is biased
  down. Define on one draw and estimate on an independent one.
- **Truncation caps** bias p down: 0.27 % of attempts on hard theorems and 1.9 % on calibration theorems were cut off
  in J2 stage A. Re-reading the 6 most-truncated stage-A zeros at doubled caps found one success
  (`textbook_6997656e…`, 2 / 16,384, 0 in ≥ 65,536 at standard caps): the caps are part of the protocol.
- **Population extrapolation cannot settle single theorems.** A beta fit cannot tell impossible from merely hard
  (Kazdan et al. App. D.2).
- **Creation at large K is uncertifiable by sampling** (§4): the rule's creation side is mostly "undetermined".

## 7. Relations

- Generalises `passk-equal-k` (K = k_eval = 256).
- Its elicited side is implied by `marginal-bracket` (the known-proof estimate, with its factor-2 margin).
- `capability-vs-propensity` replaces plain sampling with the best elicitation method within the budget.
- `compute-equivalent` prices RL in pretraining compute instead of in base samples.
- `chain-reachability` explains how RL can be "created" here while every round is elicited relative to the previous
  round.

## 8. Literature anchor

- **Chen et al. 2021 (2107.03374v2).**
  - The unbiased estimator: Sec. 2.1, Eq. 1.
  - The plug-in estimator 1 − (1 − p̂)^k "results in a consistent underestimate" (App. A).
- **Brown et al. 2024 (2407.21787v3).**
  - "With unlimited samples, any model that assigns a non-zero probability to every sequence will achieve perfect
    coverage" and "repeated sampling is only practical if we can improve coverage with a feasible budget" (Sec. 1).
  - Coverage grows over four orders of magnitude of samples (abstract).
- **Schaeffer et al. 2025 (2502.17578v1).** Aggregate pass@k is a power law iff the distribution of p near 0 is one
  (Sec. 3, Thm 3.1–3.2); the left tail has to be fitted to the share of problems in (0, 1 / n) (Sec. 5).
- **Kazdan et al. 2025 (2510.05197v1).** Chen's estimators are "only defined when the number of samples taken for each
  problem is greater than or equal to the number of attempts k" (Sec. 3.1); impossible vs hard (App. D.2).
- **Hu et al. 2023 (2310.03262v3), PassUntil.** Sample until r successes; PU = r / K is the MLE (Sec. 4.1); "designed
  to be applicable when a random baseline achieves P(s)=0" (Limitations).
- **Earlier reviews:** Yue et al. 2025 (equal-k), Chen–Foster coverage (coverage ≈ mass ≥ 1 / k).
- All locations verified in `capability_defs/lit/_claims_L1.md`.

## 9. Critic's verdict

**Strongest argument (critic): "a budget tied to RL's compute" is a menu of budgets, not one line, and the contested
theorems sit inside the menu.**
- K_per at r8 is 256 if RL's compute is counted in samples (8 rounds × 32 per target = k_eval, the equal-k case), 777 at
  6.4 ms per attempt, and 1,381 at 3.6 ms. It also scales with the number of *training* targets (4,495), and the judged
  theorems are not training targets.
- s0's 55 RL-solved hard theorems have known-proof estimates from e^−36.6 to e^−8.0 (median e^−15.7 ≈ 1 / K_total).
  `textbook_b98931f273313e2c32e7` (pend 1 / 17,152, r8 p̂ 0.90) is "created" at K_per and certified elicited at K_total.
- At K_total, certifying creation takes ≈ 2 × 10⁸ zero-success draws per theorem, so there the rule can only return
  elicited or undetermined.
- Secondary arguments:
  - *Attribution:* R is pend + ≈ 476 M replay tokens + RL. The replay-only control solves 16 / 18 / 24 of the 86 /
    77 / 77 hard theorems (e.g. `textbook_418e4b67e7e59d93fa6b` 146 / 512 against r8's 0.16).
  - *Certification arithmetic:* "created at K_per" needs 83k–124k zero-success draws, more than stage B's 65,536 at
    r16 and at 3.6 ms per attempt.
  - *Cut vs β:* the cut ignores p_R beyond one hit in 256 while K grows with RL's spend, so more RL can only weaken a
    verdict (`textbook_5758428c8c74f96345ad` rises 138× from r8 to r16 but can be certified only at r8).
- Smallest surviving change (critic): one set-level comparison at the experiment's own budget. Give base sampling the
  ladder's GPU-seconds over the same 322 evaluation theorems, and call it creation only if RL's solves at k_eval beat
  that compute-matched base coverage by more than seed noise.

**My answer: accepted; the card is revised as the critic proposes (§1–2).**
- The headline budget is **K_eval-set** = ladder GPU-s / (322 × 3.6 ms) ≈ 1.9–2.3 × 10⁴ (r8), 4.8–5.0 × 10⁴ (r16). It
  has no target-count denominator, and J2 stage A plus stage B (65,536 attempts on every RL-solved hard theorem)
  supply it.
- The decision is set level (Δ_cov), with the per-theorem verdicts secondary and every number reported net of the
  replay-only control. K_per and K_total are reported beside it, not used as creation budgets.
- β is kept as a separate continuous scale, not as "the rule without a cut".
- Certification at K_eval-set needs ≈ 60 K zero-success attempts per theorem; J9 buys it for 3 theorems per seed.
