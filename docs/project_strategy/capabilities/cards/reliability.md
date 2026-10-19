---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/reliability.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: reliability — a capability is what the model does reliably when it tries

Family R. Slug `reliability`. Notation: `_FRAME.md`.

## 1. Definition, formally

- **Per theorem.** θ *has the capability* to prove t iff p_θ(t) ≥ τ under a fair protocol (plain sampling, T 0.8, read
  caps), with τ = 1/2 by default (a coin-flip bar: "more often than not").
- **Per family** (a schema, a length bin, a renaming class): the share of members with p_θ ≥ τ, i.e. Cover@τ (Dragoi
  et al.), or the mean p_θ over members.
- **Across training seeds:** the share of seeds whose model meets the bar.
- **The conditional analysis behind it** (Harding & Sharadin): "a machine learning model has a capability to X just
  when it would reliably succeed at doing X if it 'tried'". "Tried" is operationalised as the outputs that are
  attempts at a proof of t (parsable, on-goal), under the best non-gerrymandered background conditions. This makes the
  decoder part of the model.

## 2. Decision rule

*Revised after the critic pass (§9).* Reliability defines **having** a capability (propensity: what the model does
reliably when it tries). Comparing it before and after RL does not by itself separate creation from elicitation.

| verdict | rule |
|---|---|
| **created** | p̂_R ≥ 1/2 (reliable after RL) **and** the base is certified outside its budget: UB95(p_B) < 0.05 / K and the known-proof estimate Σ_F π_B < 2 / K (`passk-budget` / `marginal-bracket`, K = K_eval-set by default) |
| **elicited (reliability gained)** | p̂_R ≥ 1/2 and p_B ≥ 1 / K is certified: RL made a reachable capability reliable |
| **undetermined** | p̂_R ≥ 1/2 and the base's reach is not settled |
| **neither** | p̂_R < 1/2 |

Seed version: count seeds per verdict, and check whether another seed's *base* already meets the bar: a "created on 2 of
3 seeds" theorem that the third seed's pend solves reliably is a pretraining-luck case.

## 3. Null or floor

- Reliability is the classical answer to the random-weights objection. A lucky success is not an ability: "A hopeless
  darts player may, once in a lifetime, hit the bull, but be unable to repeat the performance because he does not have
  the ability to hit the bull" (Kenny via SEP "Abilities", Sec. 4.3).
- Random weights never reach p ≥ 1/2 on any theorem, so under this definition they have no capability at all, at any
  k. No budget is needed: the threshold τ replaces it.

## 4. How to compute it here

- **Free.** p̂ from the k 256 reads, one draw to define and the other to check.
- **Final numbers** (`cd_part3.py`, `rel`; cap 12, r8 draw x0, s0 / s1 / s2): p̂_R ≥ ½ within the compute-matched set
  (base not within reach at K_eval-set, uncertified): **12 / 9 / 5**; net of the replay-only control 10 / 6 / 3; r16
  (x1) 12 / 14 / 8. Redraw Jaccard 1.00 / 1.00 / 0.83; seed Jaccard 0.11 / 0.06 / 0.27. Varying the RL threshold
  q = 0.1 / 0.25 / 0.5 / 0.75 gives 13 / 14 / 11 → 12 / 11 / 8 → 12 / 9 / 5 → 9 / 6 / 5. Strictly *certified* (the
  created row of §2) needs J9 (REPORT §3.8).
- **The retired rule** (p̂_pend < 0.05 at k 256, p̂_R ≥ ½; `out/defs_c12.txt`, Q9): 71 / 62 / 60 at r8, 1.31 / 1.22 /
  1.00× the equal-k set. It is kept as a description of "RL made it reliable", not as a creation test.

## 5. Sensitivity

- **Thresholds:** RL's ½ and the base's budget K. Theorems near either flip; report the (p_B, p_R) scatter, not just
  the count (τ_low = 0.05 was retired after the critic).
- **Temperature:** lowering T raises p for already-likely proofs. Reliability at T 0.8 is not greedy reliability.
  Greenblatt et al. use greedy correctness; ours is sampled.
- **Decoding:** a guided decoder raises reliability on long theorems (`guided-tts`). Under the conditional analysis the
  decoder belongs to the background conditions, so name it.
- **Representation and renaming:** p differs across renamings. The family version over a renaming class is more robust.
- **Noise:** binomial error at n = 256 is ≤ 0.03 on p around 1/2. The seed spread is far larger.

## 6. Failure modes

- **It cannot see latent capability.** A base with p = 0.04 that a trivial fine-tune lifts to 0.9 is "created" here.
  Greenblatt et al. show exactly this for password-locked models.
- **RL's sharpening counts as creation.** Raising p from 0.05 to 0.5 is the most common effect of RL in our data
  (B pass@1 0.001 → 0.56 at cap 12, `trajectory`).
- **Thresholds are conventions.** The SEP's reliability views leave "sufficiently high proportion" open.

## 7. Relations

- The propensity half of `capability-vs-propensity`.
- With τ = 1 / k_eval it becomes `passk-equal-k`.
- With families it is `schema-acquisition`'s and `transfer-invariance`'s reliability criterion.
- Disagreements with `passk-budget` mark sharpening (reliability says created, budget says elicited).

## 8. Literature anchor

- **Harding & Sharadin 2024 (2405.08989v1).**
  - Abstract: "a machine learning model has a capability to X just when it would reliably succeed at doing X if it
    'tried'".
  - Reliability as evidence (Sec. 3.1, Def. 5).
  - The conditional analysis CAMA (Sec. 4.1, Def. 10).
  - "we treat tokenizers, inference procedures, and other scaffolding for p_θ as part of the set of background
    conditions" (Sec. 2.3).
- **SEP "Abilities" (Maier & Kikkert 2025).**
  - The conditional analysis "(CA) S has the ability to A iff S would A if S tried to A" (Sec. 3.1).
  - Jaster's success view (Sec. 3.4).
  - Kenny's darts player (Sec. 4.3).
- **Dragoi et al. 2025, Cover@τ** (earlier review).
- **Greenblatt et al. 2024 (2405.19550v1):** greedy correctness as the metric (Sec. 4.2).
- All verified in `_claims_L2.md`.

## 9. Critic's verdict

**Strongest argument (critic): the original rule labels sharpening as creation.**
- τ_low = 0.05 is a base budget of ≈ 20 attempts, far below k_eval and K_per.
- In s0 r8 x0, 35 of the 71 "created" theorems had a base success in the same 256-attempt read; 40 have pooled base
  p̂ ≥ 1 / K_per (FRAME: "elicited"); on 23 the base's probability of writing RL's own proof is ≥ 1 / K_per.
- Counterexample: `la_transfer_1046`. The s0 base solves 19 / 768 and writes r8's exact proof about once in 58
  attempts; r8 solves 254 / 256, so the rule says "created". The s1 base solves it 447 / 768 (58 %) with no RL at all.
- Secondary arguments:
  - The seed rule rewards pretraining luck: of 33 theorems "created" on 2 of 3 seeds, the third seed's base has
    p̂ ≥ 0.05 on 16.
  - Best-of-N distillation of the base's own samples passes on every theorem with 1/N ≲ p_B < 0.05.
  - My claim "most stable of all definitions" was false: spec_ev / w1_ev redraw 0.93–0.98.

**My answer: accepted.**
- The base side now uses the budgeted bar (§2), and the false stability claim is removed.
- Reliability stays as the right account of *having* a capability (Harding & Sharadin; the SEP's success views). It is
  the propensity half of `capability-vs-propensity`, and its job in a create / elicit rule is the RL side: guarding
  against lucky hits.
- On its own it measures RL converting reachable capability into reliable behaviour. That is useful to report, but it
  is elicitation unless the base is shown to be outside the budget.

