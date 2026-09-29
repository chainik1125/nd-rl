# ND-pack extraction report

Source: `data/nd_pack.pdf` — *The Natural Deduction Pack*, Alastair Carr,
March 2018 (123 pages). Companion to Halbach's *The Logic Manual*.
Extractor: `code/exp/make_ndpack_targets.py` (stdlib + `nd.core` only).
Output: `data/targets_ndpack.jsonl` (68 records).

## Notation seen in the pack

- Connectives: `∧` (and), `∨` (or), `→` (imp), `¬` (not), `↔` (biconditional),
  turnstile `⊢`. Falsum `⊥` does not appear in the propositional problems.
- Atoms: `P, Q, R` and subscripted variants `P1, P2, P3, Q1, Q2`. These were
  remapped onto our vocab `P, Q, R, S` by order of first appearance across
  (premises ++ conclusion).
- Predicate-logic sections additionally use `∀, ∃, =` and predicate/constant
  letters — all first-order, all skipped.

## Counts

| quantity | n |
|---|---|
| Total problems seen (propositional + first-order, incl. worked examples) | ~155 |
| First-order skipped (quantifier / predicate / identity) | 76 |
| Propositional problems considered (transcribed candidates) | 79 |
| — skipped: biconditional `↔` (not in our connective set) | 5 |
| — skipped: > 4 distinct atoms | 3 |
| — dropped: duplicate sequent (after atom remap) | 3 |
| — dropped: failed truth-table validation | 0 |
| **Validated + emitted** | **68** |
| Emitted with a worked solution (`has_solution: true`) | 68 |
| Emitted flagged `classical_only` (needs DN-elim / EM) | 11 |

Notes:

- **"Total seen ~155"** = 68 emitted + 5 biconditional + 3 >4-atoms +
  3 duplicates + 76 first-order. The first-order tally: 2 worked examples
  (universal §3.2, existential §3.3), 6 quantifier/identity items in §4.1
  Core (core2,3,6,8,9,10), 26 in §4.7 Universal, 27 in §4.8 Existential,
  15 in §4.9 Identity. (Metatheory items in §4.10 — "show that any application
  of ¬Intro can be replaced…" etc. — are proof-about-proofs, not sequents, and
  are not counted as problems; four *concrete* sequents from §4.10
  contraposition were transcribed, of which two survived dedup.)
- **0 validation failures.** Every transcribed propositional sequent is
  classically valid, as expected for a textbook. Each emitted record passes a
  brute-force truth-table check in the extractor, and was independently
  re-parsed from its emitted prompt tokens and re-checked — both pass.
- **Biconditional skip rationale:** our connective set is `{~, &, v, >}` with no
  `↔`. Expanding `↔` into `(A>B)&(B>A)` would silently change the problem
  (different formula, different proof), so the 5 biconditional-bearing problems
  (Core 1, Core 7, Negation 14, Negation 21, Negation 28) are skipped, not
  rewritten. (Section 4.5 Biconditional itself, ~13 more problems, is entirely
  biconditional and likewise excluded — those are folded into the "first-order/
  unrepresentable" remainder rather than transcribed.)
- **`classical_only` guesses (11):** the indirect/DN/EM problems — Negation
  15 (¬¬P⊢P), 16 (⊢P∨¬P), 17, 18, 19, 20 (Peirce), 22, 24, 25, 26, 27, plus the
  two surviving contraposition items overlap these after dedup. Best-judgment
  metadata: flagged True when the natural proof essentially needs ¬Elim
  (double-negation elimination) or excluded middle.

## Histograms (emitted records)

`n_atoms`:

| n_atoms | count |
|---|---|
| 1 | 7 |
| 2 | 33 |
| 3 | 27 |
| 4 | 1 |

`n_premises`:

| n_premises | count |
|---|---|
| 0 (bare theorem) | 11 |
| 1 | 52 |
| 2 | 5 |

Spread by source page: p5:1, p11:2, p13:5, p14:17, p15:15, p17:20, p18:6, p25:2.

## Five example records

```json
{"thm": " |- ( ( P > Q ) > ( ( P & R ) > ( Q & R ) ) )", "name": "ndpack_p5_wex1", "n_atoms": 3, "n_premises": 0, "has_solution": true, "classical_only": false, "source_page": 5}
{"thm": "( ( ~ P ) v Q ) |- ( P > Q )", "name": "ndpack_p11_core5", "n_atoms": 2, "n_premises": 1, "has_solution": true, "classical_only": false, "source_page": 11}
{"thm": "P , Q |- ( P & Q )", "name": "ndpack_p13_conj1", "n_atoms": 2, "n_premises": 2, "has_solution": true, "classical_only": false, "source_page": 13}
{"thm": "( P > Q ) |- ( ( ~ Q ) > ( ~ P ) )", "name": "ndpack_p17_neg6", "n_atoms": 2, "n_premises": 1, "has_solution": true, "classical_only": false, "source_page": 17}
{"thm": " |- ( P v ( ~ P ) )", "name": "ndpack_p17_neg16", "n_atoms": 1, "n_premises": 0, "has_solution": true, "classical_only": true, "source_page": 17}
```

(The `prompt` and `human_proof_text` fields are present in the JSONL but elided
here for brevity. `human_proof_text` holds the prose proof-strategy from the
matching worked solution in section 5; the pack's actual derivations are
typeset Fitch/tree figures that do not OCR to clean text, so the prose strategy
is captured verbatim-ish instead of the figure.)

## Method notes

- Each problem was transcribed in (near-)source notation and parsed by a small
  recursive-descent parser (`~` > `&` > `v` > `>`, `>` right-associative;
  `↔` raises and routes the problem to the biconditional-skip bucket).
- Atom remap is by first appearance; >4 distinct atoms ⇒ skip.
- Formulas are built with `nd.core` constructors; `canon_premises`,
  `prompt_tokens`, `theorem_id` are used as-is (not reimplemented).
- Validation: classical truth table, `F` always false, premises-true ⇒
  conclusion-true in every valuation.
- Dedup is on `theorem_id` after canonicalisation; record `name`s are unique.
- Every emitted prompt token is checked against the tokenizer vocab (`STOI`).
```
