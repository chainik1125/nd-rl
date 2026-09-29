# Scan for the 9 laws missing from the textbook 72

2026-09-28. The goal was real exercises that instantiate the 9 schemas in `weaknesses.md`
section 7, in any form the matcher knows: direct, theorem form, conjoined premises or
biconditional.

**Code:** `code/textbook_problem/scan_laws.py`.
**Output:** `data/textbook_problem/law_scan/`. It holds no sealed-set matches: those are counted,
never written.

## Result

**Under the current rules, the open textbooks yield no usable proof exercise for any of the 9
laws.** The literature gives one: Pelletier's Problem 1.

| Source | Formulas parsed | Occurrences of the 9 | Distinct forms | Usable |
| --- | --- | --- | --- | --- |
| 14 *forall x* versions (Calgary, Magnus, UBC, Cambridge, Adelaide, Syracuse, Bristol, MSU, R³, Pécs, Dortmund, Auckland, Natal, Lorain) + the Carnap Book | 17,148 | 142 | 17 | 0 proof exercises; 3 truth-table exercises |
| The Open Logic Text @ `1e960be` | 2,130 | 0 | 0 | 0 (its exercises use metavariables) |
| Pelletier (1986), via TPTP v9.3.1 (P1, P2, P3, P10, P13, P15, P17 fetched) | 7 | 1 | 1 | **1**: P1, (p→q)↔(¬q→¬p), contraposition in both directions as a biconditional. No overlap with any set; tier B. |

**The 17 distinct forms in the textbooks, by law:**

| Law | Distinct forms | Where they overlap |
| --- | --- | --- |
| contraposition | 2 | 1 in `validation_36`; 1 truth-table item |
| contraposition_conv | 1 | `validation_36` |
| modus_tollens | 2 | 1 in `validation_36`; 1 truth-table item (the same formula as the contraposition one) |
| export | 1 | `validation_36` |
| excluded_middle | 2 | 1 in `validation_36`, 1 in `transfer` |
| explosion | 7 | 4 in `train`, 1 in `validation_36`; 2 truth-table items |
| constructive_dilemma | 1 | `validation_36` |
| negated_conditional_conv | 1 | `validation_36` |
| dist_and_over_or_conv | 1 | `validation_36` |

## Why

- **Textbooks state these laws in plain letters,** for example A→B, ¬B ⊢ ¬A. The repetition is
  large (142 occurrences, 17 forms), because every *forall x* version reuses the same examples.
- **`validation_36` holds exactly these schemas in plain letters.** Every plain-letter textbook
  instance is therefore a renaming of a `validation_36` item and is removed by the audit.
- **What would survive** is an instance with compound subformulas, a theorem form or a
  biconditional form. Open textbooks almost never print these as proof exercises.

This also means **the 9 laws are already evaluated, by `validation_36`.** The textbook 72 lack
them; the project's evaluations as a whole don't.

## The 3 truth-table items

The 3 textbook forms with no overlap are all from truth-table exercises. The exercise asks
whether the sentence is a tautology, not for a proof:
- Calgary, truth-table chapter: 2 explosion theorem forms;
- Bristol, TFL chapter: (A→B)→(¬B→¬A), both contraposition and modus tollens in theorem form.

Plan section 3.1 excludes exercises that ask "for something other than a proof, such as a truth
table".

## What was not scanned, and why

- **Carnap shared documents:** `carnap.io/robots.txt` disallows `/shared/`, with a crawl delay
  of 10 s. They are not scanned and not used.
- **DeLancey, *A Concise Introduction to Logic*:** the PDF is behind a download form with a
  one-time token, so it needs a manual download. Its licence is still unresolved
  (`sources.md`, A8).
- **Avigad et al., *Logic and Proof*:** the 16 chapter-3 exercises, as listed by the fetch tool,
  include none of the 9. Not checked by hand.
- **Tier C books** (Teller, Hardegree, Lemmon, Hurley, Copi and others): not accessed. They
  would need Charles's approval. Books built on rules of inference, such as Copi and Hurley,
  often drill modus tollens and constructive dilemma with compound formulas, which is exactly
  the form that would survive the audit.

## Options (decide)

1. **Accept the gap in the textbook set.** The 9 laws are covered by `validation_36`. Report
   that the textbook sets hold no instances of them, and why.
2. **Add Pelletier P1** to the tier B slice.
3. **Allow truth-table tautology items as theorem problems.** That adds 2 laws (contraposition
   and modus tollens in theorem form, one formula) plus explosion. It is a change to plan
   section 3.1.
4. **Approve tier C books** with compound-formula drills, for internal evaluation only.

**Default until Charles decides:** 1 and 2.
