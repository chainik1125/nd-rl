---
written_on: 2026-11-01
written_by: agent:claude
---

# Pre-registration: proof strategies of the GRPO + thinking-trace models

Written at 11:09 am PT (19:09Z) on 2026-11-01, before any strategy count of a model proof. Its sha256
and the time are recorded in [`log.md`](log.md); this branch cannot be committed to.

**What was seen before writing this.**
- Dan's K12 table, reproduced exactly with his classifier (117 / 117 summary cells).
- The S1g3 corpus mix (below).
- The labels of one base proof: the smoke test of the converter, charles149's first problem, a
  ¬I proof.
- The pass@k numbers in `high_rl_passk_20261030/teaser_numbers.md` and `seeds3.json`.

No label of any other model proof had been computed.

## Observed before registration (not predictions)

S1g3's Stage-1 corpus: 154,630 programmatic search proofs of K12 theorems. Shares are % of
proofs, uses (working).

| strategy | S1g3 corpus | Dan's K12 | Charles187 reference proofs |
|---|---:|---:|---:|
| proof by contradiction, working | 1.47 | 0.04 | 20.86 |
| modus tollens | 0.13 | 0.05 | 11.76 |
| disjunctive syllogism | 1.19 | 0.03 | 14.44 |
| proof of a negation, working | 5.56 | 2.15 | 37.97 |
| proof by cases, genuine | 6.04 | 8.79 | 33.69 |

## Units (fixed now)

Everything below uses Dan's classifier unchanged (taxonomy v1.2, fork `e6bfe0ea`) on the
dependency-pruned ND form of each Lean-accepted proof. "Uses X" means anywhere in the proof.

- **A, per attempt.** For problem p, the share of its k attempts that are Lean-accepted *and* use
  X; this is averaged over the set's problems, as % of attempts. It mixes competence with choice
  and is the emergence curve's main y axis.
- **B, share of accepted proofs.** For each solved problem, the share of its accepted attempts
  that use X, macro-averaged over solved problems. Attempt-weighted pooling is reported beside it.
- **B-matched.** B restricted to the problems both arms solve: the like-for-like change in
  strategy choice, free of which problems get solved.
- **C, problems.** The share of the set's problems with at least one accepted attempt using X, at
  the read's k.
- **Tracked strategies.**
  - working and template proof by contradiction (#10);
  - modus tollens (#8) and disjunctive syllogism (#6);
  - working proof of a negation (#7) and genuine proof by cases (#5);
  - excluded middle (#12), DNE (#11) and ex falso (#9).
- **Order features.** Dead (pruned-away) lines; `gf`, whether the first written step opens the box
  that the root closes (the proof starts from the goal's introduction rule); top-level forward
  lines written before the first box.

## Expectations

- **E1, base on charles149** (S1g3 s0, unit B). The base under-uses the textbook strategies
  relative to Charles's reference proofs, but uses them far more than the corpus does, because
  the problems need them:
  - working #10 in 5–20 %;
  - MT 1–8 %;
  - DS 2–12 %;
  - working ¬I 20–45 %;
  - genuine cases 10–35 %.
- **E2, RL end vs base, B-matched** (L3 end s0 vs base s0, charles149). Strategy choice on the
  problems both solve shifts by less than 10 points for every tracked strategy. The pooled rise
  comes mostly from composition, that is, from which problems get solved.
  - Dead lines per proof fall under RL.
  - No directional prediction for `gf`.
- **E3, per-attempt rates (A).** Every tracked working strategy's rate rises from base to L3 end,
  because pass@1 rises from 39 to 70 %. Working #10 rises at least 1.5×.
- **E4, RL-only problems** (base never solves at k1024; seed 0, and pooled over seeds 0–2). These
  are enriched for classical-only theorems relative to the set (> 22 %). RL's proofs there use
  working #10 or DS in a larger share than its proofs of base-solved problems.
- **E5, emergence vs R** (seed-0 chain, 17 points).
  - Panel (c)'s strategy is chosen by this rule: among working #10, MT, DS, working ¬I and
    genuine cases, the one with the largest base → L3-end fold change in unit A whose base B
    share is < 10 %. A tie or nothing qualifying falls back to the solve rate on the problems the
    base never solves at k1024.
  - Expected shape: flat within noise below R ≈ 1e-3, rising between about 3e-3 and 0.15, then
    flat to ki78.
  - "S-shaped" is called only if a 4-parameter logistic in log10 R beats a straight line in log10
    R by ΔAIC ≥ 2 with its midpoint inside the observed R range. Otherwise it is "monotone, not
    resolved as S-shaped".
  - It tracks the pass@1 gap more closely than the pass@1024 gap: higher Spearman ρ across the 16
    RL checkpoints.
- **E6, t1320s250** (k512). The same direction as E3, with larger per-attempt rises, since the
  13–20-line proofs gain more from RL.
- **E7, against Dan's EI.** Our base already writes working reductios (the corpus has 1.47 %), so
  the base → end fold change is far below Dan's 13–30× in count; we expect ≤ 3× in unit A. Unlike
  Dan's knockout, nothing here starts from zero.

## Validation plan

- **Unit tests.** Hand-built Lean proofs, each Lean-checked, run through converter and classifier.
- **Converter.** The K12 round trip ND → Lean → ND keeps every label. Corpus replays equal
  `plain_e256` on all records.
- **Reviewer's classifier** (`review_ps/cls.py`). Agreement on every distinct model proof:
  uses, working, primary and curried primary.
- **Lean.** Every displayed proof is re-checked.
- **Blind spot-check.** About 100 proofs, stratified by strategy, labelled by an independent
  subagent from TAXONOMY.md alone. Agreement is reported per strategy.
- **Sealed set.** textbook72 is left out, and every displayed problem is checked against its
  renaming classes.
