---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/new-proof-new-theorem.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: new proof vs new theorem

Family L / S. Slug `new-proof-new-theorem`. Notation: `_FRAME.md`.

## 1. Definition, formally

Two binary properties for each theorem t that R solves, at budget K:

- **New theorem (NT).** The base cannot prove t within K: certified p_B(t) < 0.05 / K (`marginal-bracket` upper bound).
  *After the critic* the weaker "lower bound < 1 / K" version is deleted; as computed, NT uses the compute-matched set
  (not within reach at K_eval-set) as the uncertified stand-in, and J9 for the certified version.
- **New proof (NP).** *Revised after the critic to a cross-theorem test:* every one of R's accepted proofs of t uses a
  rule set that no base-accepted proof of *any* theorem uses (all pend reads and J2). The original per-theorem version
  (R's success mass on proofs with π_B(y) < 1 / K at least ½) is kept below for reference, as `sharpen-expand`'s ρ:

  ρ(t) = Σ_{y ∈ F(t), π_B(y) < 1/K} π_R(y) / Σ_{y ∈ F(t)} π_R(y) ≥ 1/2,

  computed after **removing vacuous detours** (a self-built disjunction immediately eliminated, an unused box).
  Padding must not count as novelty (`tf-proof-prob` critic).

| | NP: no (RL uses base routes) | NP: yes (RL uses new routes) |
|---|---|---|
| **NT: no** (base can prove t) | sharpened access (elicitation) | new proof of an old theorem |
| **NT: yes** (base cannot) | impossible in principle (a base route within K would make NT false) | **new theorem**: the creation cell |

## 2. Decision rule

*Revised after the critic pass (§9).*
- **NT (new theorem)** must be the strict, sampling-certified form: zero base successes in enough attempts to bound
  p_B below the declared budget. "Not found within the compute-matched budget" is reported as the weaker label it is.
- **NP (new method)** is defined **across theorems**: RL's detour-free route class (its rule set, or its key-step
  family) occurs in **no** proof pend has ever had accepted, on any theorem.
- Measured on proofs of t alone, NP is implied by NT (π_B(y) ≤ p_B) and could never decide anything.

| verdict | rule |
|---|---|
| **new theorem, new method (strongest creation)** | NT ∧ NP |
| **new theorem, known method** | NT ∧ ¬NP (an old method applied to an instance the base cannot finish) |
| **elicited** | ¬NT |
| **neither** | R fails |

## 3. Null or floor

- Both properties are budget-relative; random weights fail NT for every theorem at any feasible K.
- The detour filter is a floor on "new route": a proof is not new just because it is longer.

## 4. How to compute it here

- J1 gives π_B and π_R for every known proof of each seed's hard and calibration theorems (stage 1 at one base; stage
  2 exact for the top proofs).
- J2 gives the sampling upper bound for NT.
- The detour filter is a syntactic pass over the ND proof:
  - an ORI whose disjunction is the major premise of an ORE that immediately follows;
  - unused assumptions (the reads' `pruned` form).
- **First numbers** (existing scores, theorem level; `out/defs_c12.txt`; T 0.8; r8, draw x0):
  - RL's eventual proof is below 1 / K_total under pend for 46 / 34 / 40 theorems, but pend itself solves many of them
    at k 256. Of the 120 (seed, theorem) pairs the `tf-proof-prob` critic examined: 31 solved by pend at k 256, 63 with
    a reference ≥ 1 / K_total, 35 with an Or-detour.
  - So most "new proofs" sit in the "new route, old theorem" or "elicited" cells.
- **Final numbers** (`cd_part3.py`, `npnt`; cap 12, r8 draw x0, s0 / s1 / s2): NT (compute-matched, uncertified) ∧ NP
  (cross-theorem rule set): **5 / 4 / 0**; net of replay 5 / 3 / 0; r16 4 / 11 / 2. Certified NT ∧ NP awaits J9; NT
  alone (compute-matched) is 22 / 21 / 14. Q6 ("new proof, old theorem" on RL's eventual proof): 22 / 15 / 22, i.e.
  41 / 29 / 37 % of the equal-k set (`out/q6.txt`).
- **Cost:** CPU, once J1 / J2 are pulled.

## 5. Sensitivity

- **K** moves both lines. At K_total almost no hard theorem is NT-certified (sampling cannot certify it).
- **Temperature:** T 0.8 for both π terms.
- **Proof set F:** ρ uses F as a stand-in for R's support. The coverage check Σ_F π_R / p̂_R must be high, or ρ is
  undefined.
- **Representation and renaming:** as in `tf-proof-prob`.
- **Noise:** ρ is deterministic given F. NT inherits binomial noise.

## 6. Failure modes

- **What counts as a "different route"** (detours, line order, redundant restatements) is a definitional choice, and ρ
  changes with the filter.
- **Selection.** F is built from RL's own samples, so it over-represents R's routes. That is fine for ρ (it measures R's
  mass), but it means the base's best route may be missing from F. Add the base's own J2 samples.
- **NT is rarely certifiable** (see `passk-budget`), so the creation cell is mostly "undetermined".

## 7. Relations

- The theorem axis is `marginal-bracket` / `passk-budget`; the proof axis is `tf-proof-prob` (route tag) and
  `sharpen-expand` (ρ is the expansion share).
- The four cells are the cleanest way to explain why "RL's proofs are improbable under the base" (reading group,
  2026-09-17) and "RL created a capability" are different claims.

## 8. Literature anchor

- **Yue et al. 2025** (earlier review): base perplexity of RL outputs, Sec. 4.
- **Korbak et al. 2022 (2205.11275):** pure conditioning keeps the base's odds among correct outputs (Sec. 5; L5
  note).
- **Shin et al. 2023 (2303.07462):** novelty = the first point where a sequence leaves the database, read with engine
  quality (L6 note).
- **Silver et al. 2017 (AlphaGo Zero):** stronger play that predicts human moves worse (Extended Data Table 1; L6 note).
- **Reading group 2026-09-17:** "Can we quantify the novelty of a proof, relative to prior proofs seen?"

## 9. Critic's verdict

**Strongest argument (critic): "created" was just NT, and NT as written was either uncomputable or invalid.**
- Every valid proof has π_B(y) ≤ p_B, so once NT is certified, ρ = 1 for any policy that solves t, even the base
  conditioned on success. In Part 3, npnt = cm in every comparison.
- Strict NT at K_eval-set needs ≈ 1.2 M clean base attempts; 0 / 17,152 certifies only up to K ≈ 290.
- Weak NT (LB < 1 / K) treats a lower bound as an upper bound. Counterexample `la_transfer_1728` (s0): pend proves it
  2 / 768 (95 % lower bound 4.6 × 10⁻⁴, certified elicited), yet LB = e^(−15.7) and ρ = 1.00, so the card said
  "created".
- Secondary arguments:
  - NP measured the base's spread, not new routes: detour-free NP holds on 22–23 of 30 calibration theorems that pend
    solves at 2–4.5 % per attempt.
  - Loose bounds reward NT.
  - Distilling base + search passes both tests.

**My answer: accepted.** Weak NT is deleted. NP becomes cross-theorem (the route class is absent from every pend-accepted
proof), which makes the 2 × 2 informative: NT ∧ ¬NP is "new theorem, known method". `cd_part3.py` is changed to match:
NP = RL's rule set absent from the rule sets of all pend-accepted proofs (any theorem, all reads, J2).

