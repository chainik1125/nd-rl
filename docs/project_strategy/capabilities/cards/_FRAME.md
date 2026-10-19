---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/_FRAME.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Shared frame for the definition cards (read this first)

Every card uses the notation and budgets below, so their decision rules can be compared.

## Objects

- **Theorem** t: a propositional sequent. **Proof** y: a sequence of actions (proof-state models, `lean_staten`) or a
  whole Lean proof text (`lean_seq`). **V(t, y) ∈ {0, 1}**: Lean 4 accepts y as a proof of t (Lean alone decides).
- **Model** θ with policy π_θ(y | t): the probability that one plain attempt writes y. For proof-state models this is the
  product over actions of π_θ(action | state), with the names the environment assigns not scored and the sampler's 33
  name bases marginalised (`tj_score.py`).
- **Solve probability** p_θ(t) = Σ_y π_θ(y | t) V(t, y): the probability that one attempt succeeds. It is a sum over
  *every* valid proof, so the probability of any single proof is a lower bound on it.
- **pass@k** = 1 − (1 − p)^k; unbiased estimate from n ≥ k attempts with c successes: 1 − C(n−c, k) / C(n, k)
  (Chen et al. 2021). **k-to-solve** = 1 / p (expected attempts to the first success).
- **Models compared:** π_0 = random initialisation (step 0), π_B = base (end of pretraining, "pend"), π_R = after RL
  (r8 or r16). A **null** is a model that should have no capability: π_0, or a policy uniform over the grammar's legal
  next tokens.

## Budgets (the answer to "at some k even random weights solve it")

Every sampling-based or probability-based rule needs a budget K: "within reach" means p_θ(t) ≥ 1/K, i.e. the model
solves t with probability ≥ 1 − 1/e ≈ 0.63 if given K attempts. *Revised after the critic passes:* the headline budget is
**K_eval-set**, RL's compute spread over the theorems being judged. The others are reported beside it.

| budget | meaning | value (cap-12 r8 / r16, per seed) |
|---|---|---|
| k_eval | the evaluation budget, equal for base and RL (Yue et al.) | 256 |
| K_per | RL's compute per *training* target (depends on how many targets RL trained on) | 777 / 731 / 954 (r8) |
| **K_eval-set** | RL's ladder GPU-seconds / (322 evaluation theorems × 3.6 ms per base attempt) | **19,281 / 21,107 / 23,310 (r8); 47,962 / 49,428 / 50,447 (r16)** |
| K_total | all of RL's compute on one theorem (can certify elicitation, never creation) | 3.5 / 3.3 / 4.3 × 10⁶ (r8); 7.7–9.3 × 10⁶ (r16) |

*Cost basis.* K_per and K_total were pre-registered with the ladder's own read cost (6.4 / 7.5 / 6.3 ms per attempt,
`cd_defs.py` READ_S); K_eval-set uses the measured cost of a base attempt on hard theorems (3.6 ms, J2). At 3.6 ms,
K_per and K_total would be 1.8–2.1× larger (K_per 1,381 / 1,512 / 1,670). *[review]* The ladder GPU-seconds are
RTX A6000 seconds (`trajectory`'s `compute.json`) and 3.6 ms is an A40 cost, so K_eval-set mixes GPU classes.
| K_null | attempts random weights need: 1 / p_0(t) | ≈ e^(hundreds to thousands) |

K_null ≫ K_total for every theorem we have, so tying K to RL's compute settles the random-weights objection: random
weights solve nothing within any budget RL itself could afford. The real question is where the **base** sits relative
to K_eval-set.

## The four verdicts

For a pair (base, RL) and a theorem or family, at a declared budget K:
- **created:** RL solves it at k_eval, and the base cannot within K. "Cannot" is certified only by ≈ 60 K zero-success
  base attempts (UB95 < 0.05 / K). Two cheaper labels: **not within reach** (p̂ < 1 / K over ≥ K attempts; the
  compute-matched set) and, inside it, **not reached** (0 successes in ≥ K attempts). Report both RL-free controls
  beside them: the replay-only ladder and the compute-matched continuation.
- **elicited:** RL solves it, and the base can within K. Certified when the 95 % sampling lower bound on p is ≥ 1 / K
  or the known-proof estimate is ≥ 2 / K (factor-2 margin; see below); a theorem the base found a few times without
  either certificate is "within reach, not certified".
- **neither:** RL does not solve it, or the base already does at k_eval.
- **undetermined:** otherwise.

## Bracketing p_B (used by several cards)

- **Upper bound:** n base attempts with c successes give a one-sided 95 % Clopper–Pearson bound; with c = 0 it is
  ≈ 3 / n.
- **Known-proof estimate:** Σ_{y ∈ F(t)} π_B(y | t), summed over F(t), the distinct Lean-accepted proofs of t that any
  model ever found, including the base's own large-k samples.
  - In exact arithmetic it would be a lower bound.
  - In practice the scorer conditions on canonical names where the sampler conditions on its own, so it is an
    **estimate**. It recovers a median 0.98 / 0.94 / 0.93 of the measured p on the 30 calibration theorems per seed
    (s0 / s1 / s2; 10th–90th percentile 0.75–1.11). Over every theorem where pend's p is measured (64 / 59 / 71) the
    10th–90th percentiles are 0.16–1.34, and the estimate exceeds the sampling UB95 on 3 / 2 / 1.
  - Certificates use a factor-2 margin.
- **Primary per-theorem output:** the base's **k-to-solve interval** [1 / UB, 1 / estimate].

## Families

S sampling / support · L likelihood · N null-relative · E elicitation cost · T transfer / invariance · R reliability ·
P psychometric · C compute equivalence · D distribution shift / update size · M mechanistic · X causal intervention.

## Two rules every card inherits from the critic passes

- **Elicitation methods are per theorem.** Sampling at any T, guided step-checked redraws, prior-only search, and
  renamings are allowed. Anything trained on verifier verdicts for other theorems (fine-tunes, value heads) is RL's
  own mechanism, studied as teachability.
- **Placebos must be matched.** RL counts only beyond RL-free training (the replay-only ladder; the compute-matched
  continuation) at the same ability gain or the same compute.

## Numbers on the cards

Each card's §9 quotes the numbers available when its critic ran (morning of 2026-10-05: J1 stage 1, J2 stage A only).
The final numbers (J1 stage 2, J2 stages A′ / B, J3–J10) are in each card's §4 and in `REPORT.md` §3, which win
where they differ.
