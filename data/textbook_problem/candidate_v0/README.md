# candidate_v0: real textbook natural deduction problems (candidate set)

Frozen 2026-09-28. This is a candidate set, not a release. It hasn't had the plan's second-reader
check (plan task T4) or a frozen-model score (T5). All counts below are aggregate. The textbook-72
figures come from the local, gitignored analysis (`docs/textbook-problem/textbook72/`).

## Contents

| File | What |
| --- | --- |
| `problems.jsonl` | 105 problems, all with a verified reference proof. 104 were found by search; 1 was written by hand (see Amendments). |
| `exclusions.jsonl` | 58 exercises not kept, with reasons. Overlaps with the sealed sets aren't listed. |
| `transcribe_summary.json`, `proof_summary.json` | Pipeline counts. |
| `MANIFEST.json` | sha256 of every file here and of the code that produced them. |

**Format.** Each row is a superset of `code/ws3/textbook72/textbook_dev.jsonl`:
- the original fields: `canonical_hash`, `name`, `prompt`, `role`, `reference_lines`;
- provenance: `source` (book, licence, commit, file and line, item number, access date),
  `original` (as printed), `encoding_notes`;
- proof: `reference_proof` (passes `strict_verify`), `min_lines`, `min_lines_method`;
- tags: `length_bin`, `family`, `tags`, `proof_features`, `schema`, `schema_in_rl_pool`,
  `classical_only`.

The ws3 evaluator reads the file unchanged.

## Sources

| Source | Licence | Problems |
| --- | --- | --- |
| *forall x: Syracuse* (Rieppel) @ `46841216` | CC BY-SA 4.0 | 40 |
| *forall x: Calgary* (Thomas-Bolduc & Zach) @ `c5cb4154` | CC BY 4.0 | 26 |
| *The Carnap Book* (Leach-Krouse & Ehrlich), ch. 3–8 | CC BY 4.0 | 23 |
| *forall x: Mississippi State* (Johnson) @ `1c4200da` | CC BY 4.0 | 16 |

## How it was built

1. **Transcribe** (`code/textbook_problem/transcribe.py`). Only proof exercises are read. The
   only changes allowed are the plan's encoding rules: letters renamed to P–S, ↔ written as
   (A → B) ∧ (B → A), at most 4 letters. For three Carnap Book items, the book's own rule that
   ∧/∨ bind tighter than → was applied. Every change is recorded in `encoding_notes`.
2. **Audit** by renaming class, using `canonical_theorem` or `canon_key`:
   - 251 sequents transcribed, and 77 duplicates removed.
   - 40 dropped as overlapping a sealed set (28 in the 72, 12 in the test files), counted only.
   - 29 dropped as overlapping `validation_36`, `train`, `heldout` or `rl_targets`.
   - 105 kept.
3. **Prove** (`code/textbook_problem/prove.py`). The fork's `minlen` search: bound 40, 300 s,
   with fallbacks (conjunct split, 1,800 s), then `strict_verify`. It also runs a restricted
   search for case splits on a derived disjunction. Lengths are upper bounds.

## Compared with the textbook 72

**Length** (shortest proof found):

| Lines | 1–2 | 3–4 | 5–6 | 7–8 | 9–10 | 11–12 | 13–15 | 16+ | 5–10 | 7+ | Median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| The 72 (71 proofs) | 4 | 3 | 11 | 8 | 11 | 17 | 8 | 9 | 30 | 53 | 10 |
| candidate_v0 (105) | 1 | 3 | 15 | 34 | 10 | 21 | 14 | 7 | 59 | 86 | 8 |

**Families** (plan section 2; each problem's first match, then every tag that applies):

| | Direct | Cond. | Case split | Case split, derived | Negation | Classical | Refutation | Bicond. | Nested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Family, 72 | 7 | 17 | 15 | 6 | 13 | 13 | – | – | – |
| Family, v0 | 15 | 22 | 17 | 12 | 24 | 15 | – | – | – |
| Tag, 72 | 7 | 17 | 15 | 6 | 13 | 18 | 13 | 10 | 11 |
| Tag, v0 | 15 | 22 | 17 | 12 | 24 | 18 | 22 | 11 | 12 |

**Form and proof structure:**

| | The 72 | candidate_v0 |
| --- | --- | --- |
| Theorems (no premises) | 16 | 13 |
| 3+ premises | 4 | 17 |
| 3–4 letters | 33 | 64 |
| Goal: → / atom / ∨ / ¬ / ↔ / ∧ / ⊥ | 21 / 9 / 14 / 10 / 10 / 4 / 4 | 36 / 20 / 13 / 14 / 11 / 8 / 3 |
| 2+ subproofs | 72% | 62% |
| Uses IMPE / ORI1 / ORI2 / DN | 54 / 38 / 37 / 30% | 72 / 21 / 14 / 18% |
| Named-law matches | 26 | 9 |
| Of the 9 laws missing from the 72 | – | 0 |

## Known gaps and caveats

- **Plan targets not yet met:**
  - 105 problems, against the 200 target;
  - 2 of 9 families have 20 or more;
  - none of the 9 missing laws. Their plain-letter textbook instances are renamings of
    `validation_36` items.
- **Individual problems to know about:**
  - one is the trivial identity P ⊢ P (Carnap Book 4.1).
- **"Refutation"** counts any use of explosion (from ⊥ derive anything). Only 3 problems have
  ⊥ as the goal.
- **Transcriptions** were checked by Claude on 25 random rows. The second-reader check is still
  to do.
- **Difficulty for the model** hasn't been measured (T5).

## Amendments

**2026-09-28: a hand-written reference proof for `textbook_e95c1b499aa32da3b5d3`.** The problem is
*forall x: Syracuse*, derived-rules exercise 4, (Q→P)→R, ¬Q∨¬S ⊢ S→¬(¬R∨¬S).
- **Why by hand:** the search found no proof in 300 s plus 1,800 s, and no published solution
  exists (the Syracuse repo, the author's teaching page and web searches were checked).
- **The proof:** Claude wrote it in the fork's basic rules only, with no derived rules. It is 21
  lines. It assumes S and ¬R∨¬S, derives R by a case split on ¬P∨¬S (inside it, P→Q is proved
  vacuously from ¬P), then closes both cases of ¬R∨¬S with ⊥.
- **Checks:** it passes `nd_verify` and `strict_verify` (21 lines). A copy with one citation
  broken is rejected by both.
- **Caveats:**
  - `reference_lines` = `min_lines` = 21 is an upper bound, not a proven minimum.
  - `requires_derived_ore` is false only because the restricted search was never run for this
    problem, not because it was tested.
- `MANIFEST.json` hashes were updated.

**2026-09-28, later: the hand-written proof shortened to 20 lines.** The new proof first proves
P→Q directly: assume P, get ⊥ by a case split on ¬P∨¬S, then Q. It then derives R and closes both
cases of ¬R∨¬S.
- **Checks:** it passes `nd_verify` and `strict_verify` (20 lines). A second, differently
  structured 20-line proof also verifies.
- **Minimality:** an exhaustive `minlen` search capped at 19 lines finished without timing out
  (716 s, about 58.7 million calls) and found no proof. So 20 is the minimum within minlen's
  search space, the same standard as the other 104 problems.
- **Unchanged:** the tags, the 16+ bin and all counts. `MANIFEST.json` was updated.
