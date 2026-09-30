# candidate_v1: 80 more real textbook natural deduction problems (candidate set)

Frozen 2026-09-30. Like candidate_v0, this is a candidate set, not a release: no second-reader
check (plan T4) and no frozen-model score (T5) yet. It is disjoint from candidate_v0 and from
every sealed and RL set (same audit). All counts are aggregate.

## Contents

| File | What |
| --- | --- |
| `problems.jsonl` | 80 problems, each with a reference proof found by search and passing `strict_verify`. |
| `deferred_no_proof.jsonl` | 10 transcribed and audited problems for which the search found no proof (bound 40, 300 s, then conjunct split or 1,800 s). Valid by truth table. **Not part of the set**; kept for hand proofs later. |
| `exclusions.jsonl` | 33 items from the two sources not kept, with reasons. Sealed overlaps are counted only. |
| `transcribe_summary.json`, `proof_summary.json` | Pipeline counts (the transcribe run covers all six sources; the proof run covers the 90). |
| `MANIFEST.json` | sha256 of every file here and of the code that produced them. |

**Format:** identical to candidate_v0 (the textbook_dev fields plus provenance, proof and tags).
The ws3 evaluator reads it unchanged.

## Sources

| Source | Licence | Problems |
| --- | --- | --- |
| Roy, *Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic*, 2023 print edition, third revision July 2025, section 6.2 (systems NDs and NDs+) | CC BY-NC-SA 4.0 (page 2 of the PDF) | 76 |
| Carr, *The Natural Deduction Pack*, March 2018, for Halbach's *The Logic Manual* | none stated | 4 |

**Licence decision pending.** Whether NC-SA is acceptable for evaluation data, and whether to ask
Carr for a licence, are open (`docs/textbook-problem/diversification/people_scan.md`, Decisions).

## How it was built

Same pipeline as candidate_v0 with two PDF readers (`code/textbook_problem/pdf_sources.py`):
pdfminer for Roy, because pypdf drops the glyphs ∼ and `]` from his math font; pypdf for Carr.
Every extracted item was printed and checked against a second extraction. Encoding rules as in the
plan: letters renamed to P–S (Carr's P1, P2, … first read as distinct letters), ↔ written as
(A → B) ∧ (B → A), at most 4 letters. Each row's `encoding_notes` say what was done, including
"starred" for the 36 Roy items whose answer the author publishes in *Answers to Selected
Exercises*.

Funnel: Roy 140 items read → 128 transcribed (12 have 5+ letters) → 29 duplicates, 9 sealed,
4 pool overlaps → 86 candidates → 76 proved, 10 deferred. Carr 104 read → 12 out of scope
(first-order, alternate system ⊢C), 3 with 5+ letters → 89 transcribed → 43 duplicates, 40
sealed, 2 pool → 4 candidates, all proved.

## Compared with the textbook 72 and candidate_v0

**Length** (shortest proof found):

| Lines | 1–2 | 3–4 | 5–6 | 7–8 | 9–10 | 11–12 | 13–15 | 16+ | 5–10 | 7+ | Median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| The 72 (71 proofs) | 4 | 3 | 11 | 8 | 11 | 17 | 8 | 9 | 30 | 53 | 10 |
| candidate_v0 (104) | 1 | 3 | 15 | 34 | 10 | 21 | 14 | 6 | 59 | 85 | 8 |
| candidate_v1 (80) | 0 | 3 | 8 | 19 | 18 | 6 | 9 | 17 | 45 | 69 | 9.5 |

Mean 11.4, max 28. The 16+ bin is large because 26 problems involve a
biconditional, which the format doubles.

**Families** (first match, then every tag):

| | Direct | Cond. | Case split | Case split, derived | Negation | Classical | Refutation | Bicond. | Nested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Family, 72 | 7 | 17 | 15 | 6 | 13 | 13 | – | – | – |
| Family, v0 | 15 | 22 | 16 | 12 | 24 | 15 | – | – | – |
| Family, v1 | 7 | 27 | 19 | 4 | 13 | 10 | – | – | – |
| Tag, v1 | 7 | 27 | 19 | 4 | 13 | 16 | 14 | 26 | 16 |

v0 + v1 together: conditional 49, negation 37, case split 35, classical 25, direct 22, derived
case split 16; five of six families reach the plan's 20.

**Form and proof structure:**

| | The 72 | candidate_v0 | candidate_v1 |
| --- | --- | --- | --- |
| Theorems (no premises) | 16 | 13 | 30 |
| 3+ premises | 4 | 17 | 9 |
| 3–4 letters | 33 | 63 | 42 |
| Goal: → / atom / ∨ / ¬ / ↔ / ∧ | 21 / 9 / 14 / 10 / 10 / 4 (+4 ⊥) | 35 / 20 / 13 / 14 / 11 / 8 (+3 ⊥) | 24 / 7 / 7 / 10 / 26 / 6 |
| 2+ subproofs | 72% | 62% | 70% |
| Uses IMPE / ORI1 / ORI2 / DN | 54 / 38 / 37 / 30% | 72 / 21 / 14 / 18% | 62 / 24 / 28 / 20% |
| Classical-only | – | 18 | 16 |
| Named-law matches (in an RL pool) | 26 | 9 (4) | 15 (9) |

**The nine laws missing from the 72: six are present.** Roy states the standard laws as
theorems, which is the form that survives the validation_36 audit: contraposition and its
converse (T6.14, one biconditional), export (T6.15), the converse of ∧-over-∨ distribution
(T6.26), explosion (E6.15t) and modus tollens (E6.15v). Still absent: excluded middle (Roy's
T6.21 and E6.18e are renamings of a validation_36 item), constructive dilemma, and the converse
of the negated conditional.

## Known gaps and caveats

- **Licences:** 76 problems are CC BY-NC-SA; 4 have none. Decision pending.
- **Published answers:** 36 problems are starred in Roy, so a worked answer exists in print. Recorded
  per problem. This matters if the model has seen the book.
- **Generation cap:** 5 reference proofs exceed 400 tokens, the evaluator's `max_new`. Those
  problems cannot be solved within the default budget however good the model is; raise the cap
  for this set or report the ceiling.
- **The 10 deferred problems** are nine nested biconditionals and one conditional theorem
  (E6.21l). Without a reference proof they have no length bin or family. Score them anyway, as
  their own slice; scoring never uses the reference proof.
- **Search incident:** the main proof run was suspended with SIGSTOP for nine hours; five
  in-flight searches saw their wall-clock timers expire on resume and were rerun alone (log
  2026-09-30). One of the five was proved on the rerun.
- **Transcriptions** were checked by Claude by eye against a second extraction; the second-reader
  check (T4) and the difficulty score (T5) are still to do.
