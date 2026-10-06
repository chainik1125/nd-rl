# Phase 3 — Textbook-shaped curriculum

Two expert-iteration arms from the take-home's Stage-1 model (`ckpts/stage1_abs.pt`) against a pool of 623
textbook-shaped theorems, plus a frozen control with equal attempts (k = 32 × 8 rounds). Outputs: `artifacts/p3/`;
pools: `data/p3/`; numbers: `numbers.md` (Phase 3 section).

**Answer.** A textbook-shaped target pool moves the model on exactly the schemata whose instances it could already
prove occasionally (contraposition, consequentia mirabilis, export, hypothetical syllogism) and on nothing else:
after 8 rounds the EI arms solve 213 and 222 of 623 targets against the frozen control's 161, and the gains are
24/24 on contraposition (frozen 3), 24/24 on consequentia mirabilis (frozen 9), 12 → 19/24 on export with the
precursor injection (frozen 0), while all four De Morgan schemata, all four distribution schemata, constructive
dilemma, disjunctive syllogism, Peirce and negated conditional stay at 0–2 of 24 in every arm. On validation-36 the
`> 6` bin goes 0/24 → 1/24 (contraposition) for plain EI, exactly as in the take-home, and 0/24 → 2/24
(contraposition + export, from round 4) with the precursor injection; the `≤ 6` bin gains consequentia mirabilis
(both arms) and explosion (plain EI). Base reachability of the newly solved validation theorems under the take-home
Stage-1 model (T = 0.8, start-index marginalised): explosion 6.7·10⁻⁴ (rare but present, 57 hits in 10⁵ samples),
contraposition 1.8·10⁻⁴ (14 hits in 10⁵), consequentia mirabilis 9·10⁻⁸ (0 hits in 10⁵), export 6·10⁻⁸ (0 hits in
10⁵). So the two `> 6`-bin wins are one rare-but-reachable theorem and one unreachable-by-sampling theorem, and the
precursor injection is what buys the unreachable one; nothing in the curriculum touches the ORE-over-a-derived-
disjunction shapes that the other 22 theorems need, consistent with Phase 2's P1 result.

## 1. Pools

- **Textbook targets** (`textbook_pool.py`): 26 classical schemata (De Morgan ×4, distribution ×4, contraposition
  and converse, Peirce ×2, disjunctive / hypothetical syllogism, export / import, double-negation forms ×3, explosion,
  excluded middle, consequentia mirabilis, negated conditional ×2, constructive dilemma, modus tollens) instantiated
  with distinct random sub-formulas (depth 1–2, no `( X op X )`), 24 per schema, 623 theorems, disjoint by renaming
  class from validation-36 and from every take-home pool. `minlen.py` labels 338 of them (≤ 8 lines: 188 have a
  ≤ 6-line proof, 118 need 7, 32 need 8); 285 have no ≤ 8-line proof in the bounded search. (`n_lines` = that label,
  99 if unlabelled, used only for per-length tables.)
- **Precursor set** (`precursors.py`): 9,455 verifier-checked ≤ 6-line proofs taken from the Phase-2 generator pool
  (820k classes) whose *theorem* unifies with one of 66 sub-step templates of the schemata (e.g. `~(A&B), A |- ~B`,
  `A, ~A |- B`, `(A&B)>C, A, B |- C`, `~(Av~A), A |- F`), ≤ 400 per template, 23 schemata covered, disjoint by class
  from the targets and validation-36. No proof was written by hand or by a prover; the templates only select
  generator output. Lengths 2/3/4/5/6: 1,326 / 1,666 / 2,100 / 2,757 / 1,606.
- Arms: `ei_textbook` (targets = the pool, retained slice = 20k take-home Stage-1 records), `ei_textbook_precursor`
  (same, plus the 9,455 precursors ×2 in every round's fine-tuning mix), `frozen_textbook` (same attempts, no
  training). Transfer set = the take-home transfer pool (1,638), held-out = the take-home held-out set.

## 2. Results

| round | frozen: targets solved | ei_textbook | ei_textbook_precursor |
|---:|---:|---:|---:|
| 1 (32 attempts) | 153 | 153 | 153 |
| 2 | 155 | 180 | 186 |
| 4 | 159 | 200 | 207 |
| 6 | 160 | 203 | 222 |
| 8 (256 attempts) | 161 | 213 | 222 |

![schemata](figures/phase3_schemata.png)

Per schema at round 8 (theorems solved of 24; schemata not listed are 0–2 in every arm):

| schema | frozen | ei_textbook | ei_textbook_precursor |
|---|---:|---:|---:|
| dn_elim / dn_intro / triple_negation / explosion / modus_tollens | 23–24 | 23–24 | 23–24 |
| hypothetical_syllogism | 22 | 24 | 24 |
| consequentia_mirabilis | 9 | 24 | 24 |
| contraposition | 3 | 24 | 24 |
| export | 0 | 12 | 19 |
| import | 2 | 3 | 4 |
| contraposition_conv / peirce_sequent / negated_conditional_conv / excluded_middle / demorgan_* / dist_* / constructive_dilemma / disjunctive_syllogism / peirce / negated_conditional | 0–2 | 0–2 | 0–2 |

Written-length histogram of distinct accepted target proofs at round 8: frozen {4: 50, 5: 68, 6: 91, 7: 6}; EI
{4: 50, 5: 61, 6: 107, 7: 36}; EI + precursor {4: 49, 5: 62, 6: 109, 7: 44}. Frontier 7 in all three arms; no 8-line
textbook proof appeared. Transfer greedy 0.322 / 0.324 / 0.338, held-out greedy 0.947 / 0.954 / 0.952.

**Validation-36 by bin and round** (greedy through `prove.py` / pass@32 at T = 0.8; Stage 1: 8/36 greedy, 10/36
pass@32, `> 6` bin 0/24):

| round | ei_textbook greedy ≤6 / >6 | pass@32 ≤6 / >6 | ei_textbook_precursor greedy ≤6 / >6 | pass@32 ≤6 / >6 |
|---:|---|---|---|---|
| 1 | 9/12 / 0/24 | 11/12 / 1/24 | 8/12 / 1/24 | 10/12 / 1/24 |
| 2 | 9 / 1 | 10 / 1 | 9 / 0 | 10 / 1 |
| 3 | 9 / 1 | 10 / 1 | 9 / 1 | 10 / 1 |
| 4 | 10 / 1 | 10 / 1 | 10 / 1 | 11 / **2** |
| 5 | 10 / 1 | 11 / 1 | 9 / 1 | 11 / 2 |
| 6 | 7 / 1 | 10 / 1 | 7 / **2** | 10 / 2 |
| 7 | 10 / 1 | 11 / 1 | 10 / 1 | 10 / 2 |
| 8 | 8 / 1 | 10 / 1 | 10 / 1 | 10 / 2 |

The `> 6` solves are contraposition (both arms, every round) and export (precursor arm only, pass@32 from round 4,
greedy at round 6). The 22 other `> 6` theorems are 0 at every round in both arms.

**Base reachability of every newly solved validation theorem** (vs Stage-1 pass@32; `novelty.py` under
`stage1_abs.pt`, T = 0.8, start index marginalised; measured rate from the Phase-1 run at 10⁵ samples):

| theorem | bin | arm(s) | log p_base (T = 0.8) | ≈ samples needed | base hits / 10⁵ |
|---|---|---|---:|---:|---:|
| explosion | ≤ 6 | ei_textbook | −7.3 | 1,500 | 57 |
| contraposition | > 6 | both | −8.6 | 5,600 | 14 |
| consequentia_mirabilis | ≤ 6 | both | −16.2 | 1.1·10⁷ | 0 |
| export | > 6 | ei_textbook_precursor | −16.6 | 1.6·10⁷ | 0 |

## 3. Reading

- The curriculum works where the model already had a foothold: schemata whose instances the frozen model solved a
  few times (contraposition 3/24, consequentia mirabilis 9/24) go to 24/24 within 2–4 rounds; export, which the frozen
  model never solved and the base model reaches at ~10⁻⁸, is lifted by the precursor injection (12 → 19 of 24
  instances; the validation `export` itself from round 4) — the precursors `(A&B)>C, A, B |- C` and `A, B |- A&B` are
  exactly its two sub-steps.
- Everything that needs an `ORE` over a derived disjunction (De Morgan, distribution, dilemma) or a classical
  excluded-middle detour (Peirce, negated conditional) stays at 0–2/24 in every arm, including the precursor arm whose
  set contains 2,000+ De Morgan / distribution sub-step proofs. Phase 2 measured why: within 6 lines the generator's
  "derived ORE" is only ever the degenerate `( X v X )` form, so the precursors cannot contain the non-degenerate
  template and RL has nothing to amplify.
- Two `> 6` validation theorems is the ceiling of this curriculum at this budget: one rare-but-reachable
  (contraposition) and one unreachable-by-sampling (export, via precursors). This matches the take-home's 1–2/24 and
  says the transfer barrier is the theorem *shape*, not the proof length.

## 4. Caveats

- One seed per arm; 623 targets; validation-36 is n = 36 (counts, not rates).
- Precursor weight (×2) and the retained slice were not tuned; the precursor arm also solves 9 more targets than plain
  EI at round 8 (222 vs 213), within seed noise for the solve rate but not for export (19 vs 12).
- The textbook pool includes ≤ 6-line-provable instances (188 of 623: double-negation forms, explosion, modus tollens,
  hypothetical syllogism); they are solved by every arm and inflate the absolute solve rate; the per-schema table is the
  informative one.
