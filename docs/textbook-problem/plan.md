# Textbook problem plan: a larger set of real textbook problems

For Claude Code agents working for Charles. Standing rules (git, Python, experiment folders,
when to stop and ask) are in the root `CLAUDE.md`. Week of Sep 28 to Oct 4, 2026; results go to
the Saturday group meeting. Items marked **(decide)** are Charles's call. Until he answers, use
the stated default and record it in `docs/textbook-problem/questions.md`.

## 1. Scope

> Build a larger set of real textbook natural deduction problems than Dmitry's 72, every one
> machine-checked and tagged by proof shape, so that out-of-distribution capability can be
> measured with less noise and across more proof shapes.

**Why.** The 72 are the project's only out-of-distribution test, and they are too thin to show a
change (the local, untracked `textbook72/overview.md`):
- 10–20 problems per hard category;
- 9 solved and 63 at 0, with nothing in between;
- every 7+ line problem is a case split, classical or deeply nested;
- 9 standard laws never appear.

Everything seen so far, both the pre-training change and RL (ws3 exp-00c), moves the generator's
own problems but never the textbook set.

**Authenticity first (Charles, 2026-09-28).** Every problem is an exercise, example or stated law
from a real textbook, with a citation. There are no synthetic problems for now: no generators,
no templates filled with subformulas, no LLM-invented problems, no renamed variants. A synthetic
diagnostic grid may come later as a separate, labelled set. The design considered on 2026-09-28
is kept in section 9 for that.

**In scope:**
- finding and transcribing exercises;
- reference proofs, the checks and audits below, and tagging;
- a Lean 4 rendering of every problem, checked by Dan's Lean checker (section 3.2);
- choosing *which* real exercises to include so the set covers the thin shapes;
- scoring the frozen checkpoint once per release to measure difficulty.

**Out of scope:** training or RL on the new set; changing the tokenizer, checker or checkpoint;
using the 72 as a source; altering an exercise beyond the encoding rules in section 3.1. If any of
these blocks progress, stop and ask.

## 2. Selection targets

Real exercises can't be made to order, so coverage is a selection goal, not a construction
target. Among the exercises that transcribe cleanly, prefer the ones that fill thin cells. Never
edit an exercise to fit a cell.

Each problem is tagged with a **family** and a **length bin**. Both are measured after
transcription, never assumed. "Needs" below means a restricted search or G4ip rules out every
proof without the feature. "Uses" means only the shortest proof found has it.

| Family | Measured condition | In the 72 |
| --- | --- | --- |
| `direct` | shortest proof has no subproof | 6 |
| `conditional` | uses IMPI; no ORE, NEGI or DN; intuitionistic | 17 |
| `case-split` | uses ORE over a premise disjunction; no derived ORE | ~11 |
| `case-split-derived` | needs a derived ORE (the fork's restricted search fails, no timeout) | 6 |
| `negation` | uses NEGI; no ORE; intuitionistic | 13 |
| `classical` | not intuitionistically valid (G4ip), so every proof needs DN | 19 |
| `refutation` | goal ⊥ or BOTE used | 4–9 |
| `biconditional` | goal is (A→B)∧(B→A) | 10 |
| `nested` | shortest proof has box depth ≥ 3 | 11 |

Families overlap; the `family` field records the first match in the table's order, and `tags`
records all of them.

**Length bins** are set by the shortest proof found, in lines including premises: 3–4, 5–6,
7–8, 9–10, 11–12, 13–15, 16+.

**Default targets for v0.1 (decide):**
- 200 or more new problems;
- at least 20 per family, spread over at least 3 length bins where the sources allow;
- at least 60 problems at 5–10 lines, the middle range the 72 lack;
- as many as the sources give of the 9 missing laws: contraposition, modus tollens, excluded
  middle, constructive dilemma, export, explosion, and the converses.

Report cells the sources can't fill; don't fill them another way.

**Overlap with RL training data.** `rl_targets` and `transfer` hold 760 instances each of the
fork's textbook schemas, and ws3 RL trained on part of `rl_targets`:
- **Renaming classes:** must be disjoint from all of them (section 3).
- **Schemas:** may overlap. Every problem carries `schema` and `schema_in_rl_pool`, and results
  are reported with and without the problems whose schema appears in an RL pool.

## 3. What a problem is

One JSON line, a superset of Dmitry's format so the ws3 evaluator reads it unchanged:

| Field | Meaning |
| --- | --- |
| `canonical_hash`, `name`, `prompt` | As in `textbook_dev.jsonl` (`THM <premises> SEQ <goal> PRF`). |
| `source` | Book, edition, chapter or section, and exercise number. Required. |
| `original` | The exercise as printed, in the book's notation. |
| `encoding_notes` | Every change made to encode it (section 3.1); empty if none. |
| `reference_proof` | A proof in the fork's format that passes `strict_verify`. Required. |
| `reference_lines` | The length of `reference_proof`. |
| `lean_statement` | The Lean 4 theorem statement (`lean.statement(prompt)`). |
| `lean_reference_proof` | `reference_proof` rendered as `lean_seq` tactic text (`lean.lean_proof`). Lean accepts it. |
| `min_lines`, `min_lines_method` | The shortest proof found, and how it was found. |
| `family`, `length_bin`, `tags` | Measured shape (section 2) and the fork's proof patterns. |
| `schema`, `schema_in_rl_pool` | Named-law match, and whether that schema occurs in `rl_targets` or `transfer`. |
| `split` | `eval` or `train_eligible` **(decide)**. |

Hard constraints on every problem:
- It is expressible in the `abs` tokenizer (its 19 symbols and 15 rules, at most 64 lines).
- It fits the evaluator's length cap.
- Its reference proof verifies with `strict_verify` **and** Lean accepts its Lean rendering
  (section 3.2).
- It has at most 8 premises and 64 proof lines, the limits of `lean_seq`.
- It shares no renaming class with `train.jsonl`, `heldout`, `rl_targets`, `transfer`,
  `validation_36`, the test files, the 72, or another problem in the set.

### 3.1 Encoding rules

These are the only allowed changes to an exercise:
- Sentence letters are mapped to P, Q, R, S in order of first appearance.
- A ↔ B is written (A → B) ∧ (B → A).
- ⊤ has no symbol. An exercise that needs it is excluded.
- Parentheses follow the fork's fully bracketed format.

An exercise that doesn't fit is **excluded and logged with the reason**:
- it has more than 4 atoms;
- it uses quantifiers or identity;
- it has more than 64 proof lines;
- it asks for something other than a proof, such as a truth table or a countermodel;
- it is not valid.

The exclusion log is part of each release.

### 3.2 Lean

Every problem ships in Lean 4 as well as the fork's token format, so Lean-format and pretrained
models can be scored on it. Lean is a second checker, not a replacement.

- **Code.** Dan's `lean_seq` tokenizer and `lean_check` checker are copied unmodified into
  `code/textbook_problem/lean_tools/`, pinned to commit `c298be6` of branch
  `dan_lean_seq_tokenizer` by `MANIFEST.json`. `code/textbook_problem/lean.py` loads them and
  renders and checks problems; `lean.py --selftest` verifies the hashes. Never edit the copies.
  To update them, copy again from a named commit and record the new hashes.
- **Toolchain.** Lean 4 core, no Mathlib, installed with `elan`, version 4.34.1 (the version the
  checker was tested with). Record the exact `lean --version` in `notes.md` and in each release
  manifest.
- **Rule.** A reference proof counts only if `strict_verify` and Lean both accept it. On Dan's
  pipeline Lean alone decides; here it does not, because `CLAUDE.md` makes `nd_verify` through
  `strict_verify` the scorer. Any disagreement between the two is a bug. Stop, record the
  problem and both verdicts in `questions.md`, and tell Dan. Never resolve it by picking one.
- **Encoding.** The Lean form comes mechanically from the token form. The section 3.1 rules do not
  change, even where Lean could say more (↔, ⊤, more atoms). A Lean-only slice that keeps such
  exercises would be a separate set, and needs Charles.

## 4. Sources

Record each problem's source. Keep sources separable, so results can be reported by book.

1. **Dmitry's original catalogs.** The 72 came from two catalogs of 145 problems: the
   112-problem `textbook_catalog112_v1` and "Smith33", both on simplex. The 73 problems of the
   lost "final" slice are real textbook problems that were never used for development.
   Recovering the catalogs and their citations from Dmitry is the first step. Exclude the 72.
2. **Openly licensed textbooks.** The default is *forall x: Calgary* (CC BY). Others, such as
   other *forall x* remixes, only after checking their licence.
3. **Other textbooks (decide).** Exercises from copyrighted books may be fine as short
   citations for internal evaluation, but not for publishing the set. Charles decides.

## 5. Task sequence

In order. A gate failing twice means stop and ask.

- **T0 Orient.** Write paths to `docs/textbook-problem/notes.md` for:
  - the textbook 72;
  - `canon_key` and `canonical_theorem`;
  - the fork's rule set and proof format;
  - the evaluator's input format;
  - the ws3 RL pools;
  - the Lean toolchain: install Lean 4.34.1 with `elan` and run `lean.py --selftest` until
    it checks with Lean instead of skipping.

  Gate: every item has a path or a question, and the Lean self-test passes with Lean.
- **T1 Checker and audit tool.** `code/textbook_problem/check.py` takes a JSONL file and
  enforces section 3 for each problem: schema, tokenizer round-trip, reference proof verifies,
  Lean acceptance of the rendered reference proof (`lean.py`), overlap audit, and duplicates
  within the set. It fills `lean_statement` and `lean_reference_proof`, and computes the measured
  fields (family, bin, tags, schema) with `logic.py` and the fork tools. Tests cover:
  - a valid problem;
  - a proof of the wrong goal, rejected by both `strict_verify` and Lean;
  - a problem whose Lean rendering does not check (a disagreement is reported, not passed);
  - a renaming of one of the 72;
  - a renaming of an `rl_targets` problem;
  - a problem that can't be tokenized.

  Gate: the tests pass.
- **T2 Taxonomy. Done 2026-09-27:** `textbook72/overview.md`, local only (gitignored); regenerate
  with `code/textbook_problem/overview.py` then `tables.py`, also local.
- **T3 Sources.** Ask Dmitry for the two catalogs, their citations and the lost final 73.
  Obtain *forall x: Calgary* and list its propositional natural deduction exercises. Gate: a
  source list in `notes.md`, with the number of candidate exercises per source.

  **In progress.** The work is in `diversification/`:
  - `weaknesses.md`: what to add, including the section 7 checklist of the 9 missing laws;
  - `plan.md`: how to collect it (tasks S1–S7);
  - `sources.md`: the source list. S1 has had two passes, and the gate is met.
- **T4 First release, v0.1.** Transcribe, encode (section 3.1), find reference proofs with the
  fork's `minlen`, check, tag, and select toward section 2. Gates:
  - `check.py` passes on every line;
  - the exclusion log is complete;
  - every problem's `original` and `encoding_notes` are filled;
  - 20 random problems are checked against the printed source by a second reader;
  - `stats.py` has written the section 6 metrics and the row in `versions.md`.
- **T5 Difficulty.** Score the frozen checkpoint once on v0.1, with the ws3 evaluator at k=32
  and temperature 1.0. Gate: numbers saved as the baseline for the set.
- **T6 Write-up.** `docs/textbook-problem/summary.md`.

## 6. Metrics tracked for every version

Every version of the set (v0.1, v0.2, ..., and any candidate pool before a release) gets the same
metrics, so the effect of each change can be read off.

**Where they live.**
- `code/textbook_problem/stats.py` computes them from the release JSONL and the exclusion log.
- It writes `data/textbook_problem/<version>/metrics.json`.
- It appends one row per version to `docs/textbook-problem/versions.md`, with the change since
  the previous version.
- The reviewer recomputes them with their own code (section 8).

### 6.1 Size and integrity

- Problems in total, and by source (book), family and length bin, next to the 72's aggregate
  counts.
- The overlap audit against every set in section 3, which must be 0. Duplicates within the set,
  which must be 0.
- `check.py` pass rate, which must be 100%.
- Reference proofs accepted by `strict_verify` and by Lean, each of which must be 100%.
  Disagreements between the two must be 0. Also the Lean version used.
- The exclusion log, counted by reason and by source.

### 6.2 Coverage of the section 2 targets

- Cells of family × length bin with at least 5 problems, and with at least 20. List the empty
  cells.
- Per family, the problems and the number of length bins covered, against the target.
- Problems at 5–10 lines, against the target of 60.
- Which of the 9 missing laws are present, and how many instances of each.
- Rule coverage: which of the 15 rules the reference proofs use, and how often.

### 6.3 Diversity

The unit that counts as "the same problem" is stated for each metric.

- **Renaming classes (`canon_key`).** Every problem is distinct here by construction, so this is
  a check, not a result.
- **Proof shapes.** The dependency-pruned shortest proof, reduced to its rule sequence, citation
  pattern and box tree, with formulas removed. Report:
  - the number of distinct shapes;
  - the effective number: exp(Shannon entropy) and 1/Σp²;
  - the largest share any single shape takes.

  Compute this for the reference proof and the `minlen` proof, and report where they disagree.
- **Schemas.** Distinct named laws, and the share of problems that match none.
- **Distance from training.**
  - The share of proof shapes that never occur in `train.jsonl` or `rl_targets`.
  - The Jensen–Shannon divergence between the set's proof-shape distribution and
    `train.jsonl`'s.
- **Diversity at equal size.** Proof shapes against the number of problems, from random
  subsamples. Report this per source, so books of different sizes can be compared, and to show
  when a source stops adding new shapes.

### 6.4 What each change did

For each version, compared with the previous one:
- the change in every count in 6.1–6.3;
- what each source adds: recompute the effective number of proof shapes and the coverage with
  that source left out;
- how much each exclusion reason removes, as problems lost and proof shapes lost. This is the
  selection effect that encoding into the fork's logic introduces, and it is named in the
  write-up.

### 6.5 Difficulty (once per release, after T5)

- The frozen checkpoint's avg solve rate over problems, with an interval over problems. Break it
  down by family, length bin and source.
- The share at 0, in (0, 0.5), and at 0.5 or more.
- The effective number of proof shapes in each of those three groups. This shows whether the
  hard problems span many shapes or few.
- Every result with and without the problems whose schema appears in an RL pool.
- Agreement with the 72 on overlapping families, in aggregate only.

**Rules for these metrics.**
- Never drop, add or pick a problem because of how the frozen model scores on it. That would
  tune the set to this checkpoint. Difficulty is measured and reported, not filtered on.
- Diversity is reported, not maximised. Selection toward the section 2 cells is allowed. Choosing
  exercises to raise a diversity number beyond that is not.
- Every number states its version, n and source file.

## 7. Deliverables

1. `data/textbook_problem/v0.1/`: the JSONL (with its Lean fields), a `.lean` file of every
   theorem with its reference proof that Lean checks as one file, the exclusion log, and a
   `MANIFEST.json` of hashes and the Lean version.
   If licences restrict it, the file stays out of public places.
2. A table of counts by source, family and length bin, compared with the 72.
3. The frozen checkpoint's avg solve rate by family and length bin on v0.1.
4. `docs/textbook-problem/summary.md`: what was built, how it was checked, known gaps, and how
   ws3 should use it.

## 8. Money and process

- CPU only, $0. Any purchase or API spend means stop and ask.
- Each release goes through the same flow as ws3: proposal, build, independent review, results.
  The reviewer:
  - re-runs `check.py` with a fresh clone, Lean included;
  - re-audits overlap with its own code;
  - checks 20 random problems against the printed source;
  - recomputes the section 6 counts and proof-shape diversity with its own code.
- Log each session in `docs/textbook-problem/log.md`.

## 9. Open questions (decide)

1. ~~What "more elaborate" should mean.~~ Answered 2026-09-28: more real problems, selected to
   cover the thin proof shapes. No synthetic problems for now.
2. Whether the set is evaluation-only, or also gets a `train_eligible` split for later RL.
   Default: evaluation-only for v0.1.
3. Which textbooks are acceptable, beyond CC-licensed ones.
4. The target size and coverage (section 2 defaults).
5. Whether the 72's taxonomy may inform the targets. It is aggregate only and never a
   template, but `CLAUDE.md` currently allows only scoring and overlap audits on the 72.
   Default: yes for aggregate gaps (as in section 2), no for anything per problem.

**Deferred: a synthetic diagnostic grid**, considered 2026-09-28 and not adopted for now. It
would be 9 families × 6 length bins at 8 per cell (~370), with surface-matched pairs, classical
twins (⊢ A vs ⊢ ¬¬A), named laws at 3 complexity levels, and renamed variants. Its purpose would
be to separate proof length from proof shape. If adopted, it is a separate, labelled set, never
mixed into the textbook headline.
