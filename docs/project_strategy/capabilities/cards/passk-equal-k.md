---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/passk-equal-k.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: equal-k support (the field's standard: Yue et al.) — kept as a descriptive label, not a decision rule (critic, §9)

Family S. Slug `passk-equal-k`. Notation: `_FRAME.md`.

## 1. Definition, formally

- Sample k attempts from the base B and from the RL model R on theorem t, with the same protocol (plain, T 0.8, read
  caps, Lean).
- **R created a capability on t at k** if B has 0 successes in k attempts and R has ≥ 1.
- Set level: the share of theorems in the 2 × 2 table (base solves / RL solves) at equal k, and the pass@k curves of B
  and R over k ≤ n (unbiased estimator).
- The "reasoning boundary" claim is that B's curve crosses R's at large k.
- Here k = 256, the project's standard read. This is the A / B / C split of `trajectory`: B = "R solves, base does
  not".

## 2. Decision rule

| verdict | rule |
|---|---|
| **created** | c_B = 0 / k and c_R ≥ 1 / k, defined on one draw and confirmed on an independent draw |
| **elicited** | c_B ≥ 1 / k and R's pass@1 is materially higher (more reliable access to an existing capability) |
| **neither** | R fails |

Set-level reading: a crossover of the curves at some k ≤ n ("the base catches up") is evidence of elicitation; no
crossover up to the largest k is evidence (weak) of creation.

## 3. Null or floor

**None built in.** The budget is k_eval, which is not tied to any cost, so the rule is exposed to Dan's objection: a
larger k for the base turns "created" into "elicited", and it does so for any model including random weights (in
principle). In practice the project answers this with `passk-budget`'s compute-tied K, or by reading B at k far above
R's training budget (`support-curves` used 4 × 10⁵, *[review]* on 3.2 M `lean_seq` cap-6
models, 3,214,336 params, not best-cap12).

## 4. How to compute it here

- **Free.** Every checkpoint has two k 256 draws on textbook72 + holdout250 (r16's x0 draw came from J5).
- **First numbers** (cap 12, `out/defs_c12.txt`):
  - Created at r8: 54 / 51 / 60 of 322 (s0 / s1 / s2; draw x0). At r16 (x1): 61 / 65 / 61.
  - Redraw Jaccard (x0 vs x1): 0.78 / 0.70 / 0.68.
  - Seed Jaccard: 0.34–0.39. r16 redraw Jaccard (J5): 0.78 / 0.78 / 0.68.
  - Net of the replay-only control: 37 / 27 / 33 (it solves 17 / 24 / 27 of the set).
  - What happens to the set under a budget (REPORT §3.2): 22 / 21 / 14 stay out of the base's reach at K_eval-set;
    pend's guided read solves 50 / 55 / 60 % of it; giving the base 512 or ≥ 768 attempts instead of 256 removes
    *[review]* 4–17 theorems (512: 4 / 9 / 11; all reads: 7 / 11 / 17).

## 5. Sensitivity

- **k:** the whole verdict. In `support-curves` (*[review]* 3.2 M `lean_seq` whole-proof base and EI r8, cap 6,
  not best-cap12) the survivor count fell 37 → 29 as base attempts rose from 4 × 10⁴ to
  2 × 10⁵ per temperature. J2 stage A: 24 / 54, 20 / 57 and 28 / 49 of the hard theorems in the set get base
  successes at 16,384 more attempts (`out/extrap.txt`).
- **Temperature and decoding:** T 1.0 rescued 19 theorems T 0.8 missed (`support-curves`, *[review]* the
  same 3.2 M models). Guided redraws change
  solves (+5–9 pp).
- **Representation:** whole-proof vs proof-state flips 28 of 29 survivors (`support-state`, *[review]* 3.2 M
  models: the `lean_seq` whole-proof base vs the SN proof-state base, 3,216,384 params, `lean_staten`; not best-cap12).
- **Renaming:** one prompt per theorem. A renamed instance can flip a borderline theorem.
- **Noise:** borderline theorems (pass@256 ≈ 0.1–0.3 at pend) flip between draws. Selection on one draw biases the
  base's estimate on that draw down (`trajectory` obs. 6).

## 6. Failure modes

- It conflates "rare" with "absent". The created set shrinks without limit as the base's k grows.
- It ignores RL's training compute: RL can spend 10⁴× more samples per theorem in training than k_eval.
- Created sets defined by one draw are partly selection (5–10 % of B flip to A on a redraw; `trajectory-cap6`).
- An RL model that only sharpens (raises pass@1) produces "created" theorems at small k, wherever the base's p ≈ 1 / k.

## 7. Relations

- The special case K = k_eval of `passk-budget`.
- Its created set is a superset of `passk-budget`'s at any K ≥ 256.
- At equal k, the extra theorems R solves are what `reliability` treats as access.

## 8. Literature anchor

- **Yue et al. 2025** (2504.13837v5; earlier review): base models "consistently surpass RLVR models across all
  benchmarks and LLM families as k increases" (Sec. 1, re-verified by me with `quote.py`).
- **Brown et al. 2024** (coverage grows with samples).
- **Kulal et al. 2019** (the origin of pass@k; screened by L1).
- **Dragoi et al. 2025** (Cover@τ; earlier review).

## 9. Critic's verdict

**Strongest argument (critic): "created" is a fact about n, not about the base.**
- A base at 0 / 512 has k-to-solve bounded only above ≈ 170 (UB95 3 / 512). The project's budgets are 10³–10⁴, and
  redraws cannot close the gap.
- Counterexample: `textbook_245a0349` (s0, L 8). pend 0 / 768; r8 246 / 256 and 248 / 256, so "created", confirmed on a
  redraw. But in J2 pend solves it 10 / 16,384 (k-to-solve 1,000–3,200), which certifies it *elicited* at K_eval-set.
  The replay-only control solves it 40 / 256, and s2's pend 505 / 768.
- Typical cases:
  - J2 hits 4 of the 7 confirmed-created theorems it sampled in its first chunk.
  - 17 of 135 confirmed-created theorems already have a base success elsewhere, 13 within ≤ 288 extra attempts; 4
    were solved by pend in RL's own round 1.
- Secondary arguments:
  - The replay-only control passes the same test against pend on 13 / 16 / 19 theorems, most of them inside RL's set.
  - No crossover can occur within n (pend solves 0–1 theorems that r8 misses).
  - The claim in §7 ("the same verdicts as `marginal-bracket`") was false: the frame's created-at-256 rule holds for
    3 of 141 RL-solved hard theorems.

**My answer: accepted.** Equal-k at 256 stays as the field's standard **descriptive label** (the project's "group B"),
and as a comparison with the literature. It is not a create / elicit rule. The decision moves to `passk-budget`'s
compute-matched rule, whose headline is the set-level coverage comparison Δ_cov with per-theorem verdicts beside
it, plus an undetermined row. The false sentence in §7 is withdrawn.

