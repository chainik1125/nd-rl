---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/tf-proof-prob.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: teacher-forced probability of specific proofs (Dan's notion (a), made strict)

Family L (likelihood). Slug `tf-proof-prob`. Notation: `_FRAME.md`.

## 1. Definition, formally

For a Lean-valid proof y of theorem t and a model θ, **log π_θ(y | t)** is computed by teacher forcing in the proof-state
environment (`tj_score.py` / `cd_score.py`):
- the product over the proof's actions of π_θ(action | state);
- the names the environment assigns are not scored;
- the sampler's 33 name bases are marginalised;
- T 0.8 (the sampler's distribution) for comparisons with sampling budgets, T 1.0 for "the model's probability".

Three choices make the notion strict.

**(i) Which proof.**

| target | what it asks | caveat |
|---|---|---|
| y_ref: the shortest known proof (`minlen`; an ND-derived upper bound on the minimal length) | does θ know *this canonical* proof? | unselected; the model may take another route (`organism-analysis`: "policy routes around a reference's Or.elim") |
| y_R: RL's own proof (the "eventual" proof = R's most probable accepted sample) | could the base have written RL's proof? | selected by R: inflates log π_R(y_R), so the RL lift on it is partly by construction (`trajectory` obs. 3) |
| y*_B: the known proof the base likes best, argmax over F(t) of π_B(y) | the base's own best route among all proofs anyone found | depends on how complete F(t) is |
| the marginal p_B(t) = Σ_y π_B(y) V(t, y) | the probability that one base attempt proves t | not computable exactly; bounded below by the sum over known proofs (`marginal-bracket`) |

**Only the marginal is the probability that sampling finds a proof.** Every single-proof probability is, in exact
arithmetic, a lower bound on it: p_B(t) ≥ π_B(y) for every valid y.

**(ii) Normalisation.**

| form | what it is | use |
|---|---|---|
| total log π(y) | the probability of producing exactly y | comparable with an attempt budget: y appears in K attempts with probability ≈ 1 − e^(−Kπ) |
| per-step mean | length-normalised; not the probability of any event | ranking proofs of different lengths |
| worst step w1 = min_j log π(a_j \| s_j) | the hardest single decision | a necessary condition: if w1 < −ln K, then y is beyond budget K whatever the other steps |

total ≤ w1 always (all terms ≤ 0). w1 is the right unit for a step-resampling (guided) decoder, and total for plain
sampling.

**(iii) Base vs RL.**
- The RL lift is Δ(y) = log π_R(y) − log π_B(y) on a fixed proof.
- The decision uses the base's absolute level against −ln K, with K from `_FRAME.md`: −ln 256 = −5.5 nats (equal-k);
  −ln K_per ≈ −6.6 to −7.6; **−ln K_eval-set ≈ −9.9 to −10.1 (r8), −10.8 (r16)**, the headline; −ln K_total ≈ −15.0 to
  −16.0 (cap 12, r8 / r16).

## 2. Decision rule

*Revised after the critic pass (§9). The decision is made on the base's best known route, not on RL's proof.*

| level | created | elicited |
|---|---|---|
| **theorem** (the decision) | the base fails at k_eval (0 / 256), and max over F(t) of π_B(y) < 1 / K, where F(t) includes the base's own accepted samples (J2) and RL's proofs; certify with the sampling bound (`marginal-bracket`) | max over F(t) of π_B(y) ≥ 2 / K (or the sum over F(t) ≥ 2 / K; factor-2 margin, since the terms are estimates) |
| **proof** (a descriptive tag, "new route") | log π_B(y_R) < −ln K: RL's own proof is a route the base would not write within K | — |
| **step** (descriptive, "new move") | some step of y_R has log π_B < −ln K | — |

"Neither" means R does not solve t. The proof-level tag is **not** evidence of a new capability: it fires on vacuous
detours (§9), and a new proof of an old theorem is not a new capability to prove the theorem
(`new-proof-new-theorem`).

## 3. Null or floor

- Score the same proof under random initialisation (step-0 checkpoint): log π_0(y) is about −200 nats per step in
  `trajectory` (worst step of the eventual proofs at init: −191 median), so a 10-step proof sits near −10³.
- The random-weights objection is the statement π_0(y) > 0. It is answered by measuring how far below 1 / K the
  number is, not by its sign.
- `bits-over-null` turns this into RL's share of the bits.

## 4. How to compute it here

- **Exists.** Trajectory's scores of the 315 references and the r8 eventual proofs at all 22 checkpoints (init …
  pend, r1 … r8), all three cap-12 seeds; cap-6 twins in `trajectory-cap6`.
- **J1 (this run).**
  - Stage 1: every known proof of each seed's hard and calibration theorems (≈ 1.3 × 10⁵ per seed) under init / pend
    / r8 / r16 at one name base. Each term gets the one-base bound b0 − ln 33 (a lower bound on that proof's marginal). Cost ≈ 47 s per checkpoint per 34,000
    proofs on an A40, ≈ 0.4 GPU-h in all.
  - Stage 2: the top proofs exactly at 33 bases.
- **Final numbers, the post-critic rule** (`cd_part3.py`, `tfmax`; cap 12, r8 draw x0, s0 / s1 / s2): RL solves t,
  pend fails it at k 256, and the base's *best known* proof has π_pend < 1 / K_eval-set (T 0.8; exact 33-base terms
  where J1 stage 2 scored them, the stage-1 bound elsewhere): **29 / 27 / 24**; net of replay 23 / 16 / 17; r16 31 / 39
  / 26. Redraw Jaccard 0.88 / 0.89 / 0.84; seed 0.22 / 0.20 / 0.31. Budget sweep (K = 0.1 / 0.3 / 1 / 3 × K_eval-set):
  42 / 41 / 42 → 34 / 34 / 32 → 29 / 27 / 24 → 26 / 24 / 21. It agrees with the bracket at Jaccard 0.90.
- **The retired proof-level rule** (RL's own eventual proof below 1 / K_total under pend; `out/defs_c12.txt`): 46 / 34 /
  40; the reference proof 49 / 48 / 37. Q6 ("new proof, old theorem": RL's eventual proof below 1 / K_total while the
  known-proof estimate is ≥ 2 / K_total): 22 / 15 / 22 theorems, 41 / 29 / 37 % of the equal-k set (`out/q6.txt`).
  These are "new route" tags, not creation verdicts.

## 5. Sensitivity

- **Temperature.** T 1.0 vs T 0.8 moves a proof's total by several nats. One known proof of a C theorem: −62.4 at T 1.0
  vs −69.0 at T 0.8 under pend s0 (J1 smoke test). Decide at the sampler's T.
- **Decoding.** Plain sampling needs the total; guided redraws make w1 the bottleneck.
- **Name marginalisation.** It moves a total by 2–16 nats (`trajectory`). The scorer conditions on canonical names,
  not on the sampler's own name tokens; this caveat is not quantified (`trajectory` limitations).
- **Representation.** In the proof-state interface the environment writes box-closing tokens (`claim-audit` C2b).
  Whole-proof scores include them. The same proof gets different numbers in the two interfaces.
- **Renaming / premise order.** Score the renaming class, not one prompt.
- **Seed.** The seed SD of a group median is 0.5–0.7 nats (`trajectory`).
- **Thresholds.** A ln K difference of 4,495× between K_per and K_total is 8.4 nats. Verdicts near the line flip.

## 6. Failure modes

- **Route dependence.** A low π_B(y_ref) shows only that the base does not know *that* proof.
- **Selection.** y_R is chosen to be likely under R. Membership in "RL-only" groups was chosen by the base failing,
  which biases base numbers down (`trajectory`, `claim-audit`).
- **Scale drift.** The log-p scale differs between models: the 50 % point moves 9–11 nats between caps
  (`organism-analysis`). A fixed nats threshold is not portable; tie it to K.
- **Single-proof probabilities understate reach.** A theorem can be elicitable through proofs nobody has scored.
- **Name and box-token conventions** change the numbers (§5).

## 7. Relations

- Theorem-level elicited ⟸ `marginal-bracket`'s known-proof estimate ⟸ any single known proof above 2 / K.
- The proof-level version is the measurement behind `new-proof-new-theorem` and `sharpen-expand`.
- The step-level version underlies `schema-acquisition` (new moves).
- Against the init null it gives `bits-over-null`.
- At K = 256 it is the likelihood twin of `passk-equal-k`.

## 8. Literature anchor

- **Jones et al. 2025 (2502.16797v1).** A specific-output probability "can be done in a single forward pass—but may
  not reflect the actual likelihood of producing 'useful' instructions" (Sec. 4.1). They also suggest importance
  sampling (Sec. 4.5).
- **Wu & Hilton 2024 (2410.13211v2).** Naive sampling is "uninformative at distinguishing between small probabilities
  like 10^-10 and 10^-20" (Sec. 2); importance sampling is an unbiased estimator (Sec. 3.1).
- **Schaeffer et al. 2025b (2509.24012v2).** The log likelihoods of gold reference solutions serve as a covariate that
  predicts pass@k (abstract).
- **Ethayarajh et al. 2022 (2110.08420v3).** Pointwise V-information: the log-probability gain on the gold output over
  a null input (Sec. 3, Def. 3.1, Eq. 4).
- **Lin et al. 2023 (2312.01552v1).** Token-level base-rank analysis: 77.7 % of tuned tokens are the base's top choice
  (Sec. 2.2).
- **Project.** `lit-measures` (worst step, cut ln(3/400,000)), `trajectory`, `trajectory-cap6` (Δ_RL vs Δ_PT), and the
  reading-group note (2026-09-17) "Maximize the negative teacher-forced log probability of the proofs generated by
  the fine-tuned model".
- Locations verified in `_claims_L1.md` / `_claims_L3.md`.

## 9. Critic's verdict

**Strongest argument (critic, 2026-10-05).** The proof-level verdict scores RL's route, not a capability, and the new
part can be padding.
- Example: `la_transfer_629`, seed 0, (S → Q) → (P ∨ S) ⊢ ¬(P ∨ S) → ¬(S → Q).
  - pend solves it on 178 of 256 attempts of the same draw, and its 8-step contrapositive (the reference) scores −0.42
    nats.
  - r8's eventual proof wraps that proof in a vacuous detour (`Or.inr n3`, then `Or.elim` on that disjunction). It
    scores −15.64 < −ln K_total = −15.07, so the rule says "created".
  - The worst step (−9.23) is the base's own modus ponens, which scores −0.02 in its own proof.
- Of the 120 (seed, theorem) pairs "created" under the original proof-level rule:
  - pend solves 31 at k 256;
  - in 63 the reference alone is ≥ 1 / K_total;
  - 108 have y_R longer than the reference (median +4–5 actions);
  - *[review]* 26 contain a self-built Or-detour, not 35 (the executor's own `detour()` in `cd_part3.py` also gives
    26; looser rules give 53–55, and no rule the reviewer tried gives 35).
- Secondary arguments:
  - Theorem-level "created" can only be certified by sampling (≈ 60 K samples).
  - The 0.92–0.98 redraw Jaccard is by construction (y_R is fixed from x0); the seed Jaccard of 0.19–0.23 tracks
    routes.
  - Temperature flips 14 of the 120.

**My answer: accepted, and the card is changed.**
- The decision moves to the theorem level, on max over F(t) of π_B(y) (and the sum over F, `marginal-bracket`), with
  F(t) including the base's own large-k samples. The base must also fail at k_eval.
- RL's proof is kept only as a descriptive "new route" tag. Padding can only add proofs to F(t), never lower the
  maximum, so the padded example becomes elicited (−0.42).
- The uncertifiability of theorem-level creation stays as a stated limit: likelihood can certify elicitation, and only
  sampling can certify creation.

