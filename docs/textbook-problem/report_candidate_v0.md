# Report: candidate_v0, 105 new textbook problems, compared with the textbook 72

2026-09-28. Workstream `textbook-problem`.
- **The dataset:** `data/textbook_problem/candidate_v0/` (data card: its `README.md`).
- **The 72:** figures are aggregate counts from the local, gitignored analysis
  (`docs/textbook-problem/textbook72/`). No problem of the 72 is quoted.
- **Status:** a candidate set. It hasn't had the second-reader check (plan task T4), an
  independent recount (CLAUDE.md), or a frozen-model score (T5).

## Summary

- **Size and provenance.** 105 new, real textbook exercises from four openly licensed books.
  Every one has a verified reference proof: 104 found by search, 1 written by hand. None shares a
  renaming class with the 72, the test files, `validation_36`, `heldout`, `transfer`,
  `rl_targets` or `train`.
- **The 72 draw heavily on the same books.** 29 of the 72 appear in the open *forall x* textbooks
  (26 of them in exercise sections), and 16 appear in five or more versions. The other 43 weren't
  found in any open source scanned.
- **Length: more mid-length problems, fewer very long ones.** 59 new problems have 5–10 line
  proofs, against the 72's 30. The median is 8 lines, against 10. Both sets are dominated by 7+
  line proofs: 86 of 105 and 53 of 71.
- **Shape: more volume in most families, not a new mix.** The new set has more negation (24 vs
  13), conditional (22 vs 17), derived case split (12 vs 6) and direct (15 vs 7) problems.
  Classical-only is flat (18 vs 19). 2 of the 9 families reach the plan's 20-per-family target.
- **Named laws: fewer, and none of the 9 missing.** 9 new problems match a named law (the 72: 26).
  None of the 9 laws absent from the 72 could be added: textbooks state them in plain letters,
  which are renamings of `validation_36`.
- **Difficulty for the model is unknown.** The frozen model solves 9 of the 72, all at 6 lines or
  fewer. The 105 haven't been scored yet.

## 1. Sources

### 1.1 candidate_v0

| Book | Version | Licence | Proof exercises transcribed | Duplicate in the set | Sealed overlap | Pool overlap | Kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| *forall x: Calgary* (Thomas-Bolduc & Zach, after Magnus and Button) | `rzach/forallx-yyc` @ `c5cb4154` | CC BY 4.0 | 66 | 4 | 24 | 12 | **26** |
| *forall x: Syracuse* (Rieppel, after Button) | `mrieppel/forallxsyr` @ `46841216` | CC BY-SA 4.0 | 109 | 49 | 13 | 7 | **40** |
| *forall x: Mississippi State* (Johnson, after Calgary) | `loighic/forallx-msu` @ `1c4200da` | CC BY 4.0 | 31 | 14 | 0 | 1 | **16** |
| *The Carnap Book* (Leach-Krouse & Ehrlich), ch. 3–8 | carnap.io/book | CC BY 4.0 | 45 | 10 | 3 | 9 | **23** |
| **Total** | | | **251** | **77** | **40** | **29** | **105** |

**How to read the table:**
- "Transcribed" counts sequents: an interderivable pair counts twice.
- Duplicates keep the first copy in the order Calgary, Syracuse, MSU, Carnap. The other
  locations are recorded in `also_in`.
- "Sealed" means a renaming of one of the 72 (28 distinct) or of a test-file item (12). These are
  counted only, never written.
- "Pool" means a renaming of an item in `validation_36` (10), `train` (17), `heldout` (2) or
  `rl_targets` (1).

**Not transcribed:**
- 14 items in those sections aren't proof tasks ("show this is not a theorem", contingency);
- 12 have more than 4 sentence letters;
- 3 are invalid items from a "prove if valid" exercise.

**How the four books compare:**

| Book | Kept | Median lines | 7+ lines | Classical only | Theorems | Largest families |
| --- | --- | --- | --- | --- | --- | --- |
| Calgary | 26 | 10.5 | 21 | 6 | 4 | case split 7, negation 6, classical 5 |
| Syracuse | 40 | 8 | 35 | 7 | 4 | conditional 12, case split 8, 5 each of direct, negation, classical, derived case split |
| MSU | 16 | 7.5 | 11 | 0 | 2 | derived case split 5, direct 5, conditional 4 |
| Carnap Book | 23 | 10 | 19 | 5 | 3 | negation 12, classical 5, conditional 4 |

- **Syracuse** contributes the most and the widest mix.
- **The Carnap Book** is the main source of negation problems. Its chapters are organized by
  strategy (conditional, nested, indirect derivations).
- **MSU** adds no classical problems.
- **Calgary** loses about half its proof exercises to the 72 or the test files.

**Sources considered and not used yet** (details in `diversification/sources.md`):
- *forall x* versions that only copy Calgary's or Cambridge's exercises: Cambridge, UBC, Magnus,
  Adelaide;
- R³ (licence unverified);
- Avigad et al. (licence unclear);
- DeLancey (licence unresolved; the PDF is behind a download form);
- the Lorain County Remix;
- Pelletier and TPTP (literature, tier B);
- Carnap's shared documents (excluded: `robots.txt` disallows them).

### 1.2 The textbook 72

- **Composition:** 58 `textbook_dev` problems, with Dmitry's `reference_lines`, and 14
  `textbook_train` problems (prompt only).
- **Origin:** per `code/experiments/current/r1_cot/prepare_data.py`, they were drawn from two
  catalogs, `textbook_catalog112_v1` (112 problems) and `textbook_smith_final_v1` ("Smith33").
  Their per-problem citations aren't in this repo.

**What the open textbooks reveal.** All 72 were matched by renaming class against 17,148 formulas
parsed from 14 *forall x* versions plus the Carnap Book:
- **29 of the 72 appear somewhere,** 26 of them inside an exercise section: 22 of the 58 dev and
  7 of the 14 train problems.
- **16 of these appear in five or more versions.** They are the standard exercises every
  *forall x* version reuses.
- **Found per source:** Calgary 18, Syracuse 18, Dortmund 18, Natal 16, MSU 11, Bristol 10,
  Cambridge 9, R³ 9, Lorain 6, Adelaide 5, Carnap 4, Auckland 2, UBC 1, Pécs 1, Magnus 1.
- **These per-source counts are lower bounds.** The parser handles each book's notation macros
  unevenly, and Magnus's and UBC's low counts probably reflect that.
- **The other 43 weren't found** in any open source scanned. Their origin is only in Dmitry's
  catalogs, so asking him (plan task T3) is the only way to recover it.

**Consequence.** The *forall x* lineage was already partly used up by the 72: 28 exercises the
new pipeline transcribed were dropped as renamings of the 72. The new set is built from what the
72 didn't take.

## 2. Problem types

Families are defined in plan section 2. They are measured on the proof found, and "needs"
claims are backed by a restricted search or G4ip. `family` is the first match in the table's
order; a tag is any match.

| | Direct | Conditional | Case split | Case split, derived | Negation | Classical | Refutation | Biconditional | Nested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Family, 72 (71) | 7 | 17 | 15 | 6 | 13 | 13 | – | – | – |
| Family, v0 (105) | 15 | 22 | 17 | 12 | 24 | 15 | – | – | – |
| Tag, 72 | 7 | 17 | 15 | 6 | 13 | 18 | 13 | 10 | 11 |
| Tag, v0 | 15 | 22 | 17 | 12 | 24 | 18 | 22 | 11 | 12 |

- **Classical only** (no intuitionistic proof, decided by G4ip): 72: 19 (26%); v0: 18 (17%).
- **Case splits on a derived disjunction are genuinely needed** (the restricted search fails
  without a timeout) in 72: 6 and v0: 12.
- **"Refutation" is inflated** by using explosion inside case splits. Problems whose goal is ⊥:
  72: 4, v0: 3.
- **Proof patterns** (from the fork's classifiers):

  | Pattern | The 72 | v0 |
  | --- | --- | --- |
  | Reductio (assume the negated goal, derive ⊥, then DN) | 18 | 17 |
  | Case split on a derived disjunction (in the proof found) | 11 | 16 |
  | Box depth 3 or more | 11 | 12 |
  | Case split inside a →-intro box | 7 | 8 |
  | Nested case splits (fork's `ore_nesting`) | 21 | 29 |
  | Chains of →-eliminations (fork's `impe_chain`) | 38 | 76 |

**Family by length** (shortest proof: ≤6 / 7–10 / 11+ lines):

| Family | The 72 | v0 |
| --- | --- | --- |
| Direct | 7 / 0 / 0 | 9 / 6 / 0 |
| Conditional | 7 / 5 / 5 | 6 / 11 / 5 |
| Negation | 3 / 4 / 6 | 3 / 12 / 9 |
| Case split | 0 / 4 / 11 | 1 / 6 / 10 |
| Case split, derived | 0 / 4 / 2 | 0 / 4 / 8 |
| Classical | 1 / 2 / 10 | 0 / 5 / 10 |

In the 72, every problem of 7 or more lines is a case split, classical, negation or long
conditional problem. v0 adds 7–10 line problems that are direct (6), conditional (11) and
negation (12). That begins to separate length from shape, which the 72 couldn't do.

## 3. Lengths

Lengths are the shortest proof found by the fork's `minlen`, in lines including premises. Each
is verified by `nd_verify` and `strict_verify` and is an upper bound on the true minimum.

| Lines | 1–2 | 3–4 | 5–6 | 7–8 | 9–10 | 11–12 | 13–15 | 16+ | 5–10 | 7+ | Mean | Median | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| The 72 (71 with proof) | 4 | 3 | 11 | 8 | 11 | 17 | 8 | 9 | 30 | 53 | 10.5 | 10 | 29 |
| v0 (105) | 1 | 3 | 15 | **34** | 10 | 21 | 14 | 7 | **59** | 86 | 9.7 | 8 | 29 |

**How the lengths were found:**
- **v0:** 100 by whole search; 2 by conjunct splitting (looser bounds); 1 by a 1,800 s search;
  1 trivial identity; 1 by hand (20 lines). An exhaustive search at 19 lines found no proof for
  the hand-written one, so 20 is the minimum in `minlen`'s space.
- **The 72:** 1 problem has no proof (a classical problem, Dmitry's reference 22 lines).
  7 lengths come from conjunct splitting.

**Dmitry's `reference_lines`.** On 40 of 57 dev problems they equal the shortest found. On 17
the search found a shorter proof, and it never needed a longer one. Mean 11.5, median 10. v0's
`reference_lines` is the shortest found, so "≥ 7 lines" means slightly different things in the
two sets. Compare them by shortest length, as above.

## 4. Form and surface

| | The 72 | v0 |
| --- | --- | --- |
| Theorems (no premises) | 16 (22%) | 13 (12%) |
| Premises 0 / 1 / 2 / 3 / 4+ | 16 / 30 / 22 / 3 / 1 | 13 / 41 / 34 / 14 / 3 |
| Sentence letters 1 / 2 / 3 / 4 | 7 / 32 / 25 / 8 | 6 / 35 / 43 / 21 |
| Goal: → / atom / ∨ / ¬ / ↔ / ∧ / ⊥ | 21 / 9 / 14 / 10 / 10 / 4 / 4 | 36 / 20 / 13 / 14 / 11 / 8 / 3 |
| Any ↔ (written as (A→B)∧(B→A)) | 20 | 23 |
| Mean formula size (all formulas) | 6.1 | 5.4 |
| Mean prompt tokens | 28.3 | 26.5 |
| Negation stack of 3 or more | 2 | 1 |

- **More letters and more premises, but smaller formulas.** v0 exercises spread over more, simpler
  premises.
- **Twice as many atom goals (20 vs 9).** Working backward from the goal gives no hint there, so
  these are the forward-work problems `weaknesses.md` asked for.
- **No prompt exceeds the evaluator's cap** of 624 prompt tokens (1024 − 400).

## 5. Proof structure and rules

| | The 72 | v0 |
| --- | --- | --- |
| Subproofs 0 / 1 / 2 / 3 / 4+ | 7 / 13 / 21 / 16 / 14 | 15 / 25 / 37 / 11 / 17 |
| 2+ subproofs | 72% | 62% |
| Box depth 0 / 1 / 2 / 3+ | 7 / 20 / 33 / 11 | 15 / 42 / 36 / 12 |
| Distinct rules per proof (mean) | 4.8 | 4.4 |

Share of proofs using each rule:

| Rule | IMPE | IMPI | ANDI | ANDE1 | ANDE2 | ORI1 | ORI2 | ORE | NEGE | NEGI | BOTE | DN | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| The 72 | 54% | 48% | 39% | 37% | 38% | 38% | 37% | 30% | 58% | 49% | 13% | 30% | 11% |
| v0 | 72% | 46% | 36% | 36% | 48% | 21% | 14% | 27% | 58% | 42% | 17% | 18% | 2% |

Both sets use all 13 rules. v0 leans more on →-elimination and less on ∨-introduction and DN.
Relative to pre-training (`train.jsonl`, where 1.5% of proofs use ORE and 2.1% ∧-elim), both
sets are far out of distribution in the same directions.

## 6. Named laws and overlap with RL training

| | The 72 | v0 |
| --- | --- | --- |
| Problems matching a named law (40 checked) | 26 | 9 |
| Matching one of the fork's 26 textbook schemas (present in the RL pools) | 16 | 4 |
| Fork schemas covered | 17 of 26 | 4 of 26, all also in the 72 |
| The 9 laws missing from the 72 | 0 | 0 |

- **v0 is almost free of the RL pools' schemas.** Only 4 problems share a law with `rl_targets`
  or `transfer`. Results on v0 are therefore less exposed to RL training than results on the 72.
- **The 9 missing laws couldn't be added** (details in `diversification/missing_laws_scan.md`).
  Across 15 open textbooks they occur 142 times but in only 17 distinct forms:
  - 14 are renamings of `validation_36`, `train` or `transfer`;
  - 3 are truth-table exercises, which the plan excludes.

  The only usable instance found is Pelletier's Problem 1 (literature tier), and it hasn't been
  added. The 9 laws are measured by `validation_36`, one plain-letter item each.

## 7. Quality checks and quirks

| | The 72 | v0 |
| --- | --- | --- |
| Reference proof passes `nd_verify` and `strict_verify` at the stated length | 71 of 71 | 105 of 105 |
| Inconsistent premises with a goal other than ⊥ (the goal follows vacuously) | 3 | 2 |
| At least one premise not needed | 2 | 2 |
| Goal among the premises (identity-like) | 1 | 1 (P ⊢ P, Carnap Book 4.1) |

**v0 transcription:**
- **Encoding changes:** letters renamed to P–S in 78 problems; ↔ written out in 23; Carnap's
  grouping convention applied to 3. Every change is recorded per problem.
- **Checks:** 25 random rows were compared with the printed source by Claude, and all were
  correct. The second-reader check is still to do.
- **Hand-written proof:** 1, verified. A copy with a broken citation is rejected by both
  verifiers.

## 8. Difficulty for the model

**The 72:** the frozen checkpoint's results, one seed, k = 32, from the exp-00c dumps. These use
the analysis's primary-type labels, not the plan's families:

| Type | Solved at all | Avg solve rate |
| --- | --- | --- |
| Direct | 3 | 0.35 |
| Conditional proof | 3 | 0.17 |
| Negation intro | 2 | 0.11 |
| Classical | 1 | 0.05 |
| Case split | 0 | 0.00 |

- 9 of the 72 are solved at all, every one at 6 lines or fewer; 63 are at 0.
- Across the four exp-00c RL arms, only two problems of 7+ lines were ever solved.

**v0: not yet measured (T5).** v0 has 59 problems at 5–10 lines, many of them direct,
conditional or negation, which are the shapes the model sometimes solves in the 72. So v0 could
show the difficulty gradient the 72 lack. That's a hypothesis, not a result. Difficulty will be
measured and reported, never used to filter problems.

## 9. Against the plan's v0.1 targets

| Target (plan section 2) | Status |
| --- | --- |
| 200+ new problems | 105 |
| 20+ per family | met for negation (24) and conditional (22) only |
| 60+ at 5–10 lines | 59 |
| The 9 missing laws | 0 (structural reason in section 6) |
| Overlap audit clean | yes: 0 overlaps kept |
| Every problem cited and transcribed as printed | yes (book, commit, file and line, item) |

## 10. Caveats

- **Lengths** are upper bounds from a restricted search space. The 2 conjunct-split lengths are
  the loosest.
- **Families** describe one shortest proof, except "classical" (G4ip, exact) and "derived case
  split" (restricted search).
- **Provenance counts for the 72** depend on how well the parser reads each book's LaTeX. They
  are lower bounds.
- **No numbers here have had the independent recount** CLAUDE.md requires before they go into a
  conclusion.
- **Model numbers** exist only for the 72: one seed, k = 32.

## 11. Next steps

1. **T5:** score the frozen checkpoint once on v0, and report by family and length bin.
2. **Close out T4:** a second reader checks 20 problems per source; an independent recount of
   the audit and verification; `check.py` with the plan's tests (T1).
3. **More sources toward 200:**
   - ask Dmitry for the catalogs, the 73 "final" problems and the missing 22-line proof;
   - decide on R³, Avigad, DeLancey, the Lorain Remix, Pelletier and copyrighted books.
4. **Decisions** (defaults in brackets):
   - the missing laws [accept the gap; add Pelletier P1 to the literature slice];
   - the literature slice in the headline [no];
   - the `completion` slice;
   - an `also_in_validation_36` side slice;
   - the refutation definition;
   - the identity problem [move to a tagged slice].
