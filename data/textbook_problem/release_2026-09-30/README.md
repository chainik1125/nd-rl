# Textbook natural deduction problems: candidate_v0, candidate_v1 and the unsolved ten

Assembled 2026-09-30 from `data/textbook_problem/candidate_v0/` and `candidate_v1/`, with a Lean 4
rendering of every problem. 194 propositional natural deduction problems transcribed from real
textbooks: 184 with a verified reference proof and 10 valid problems with no proof found. None is a renaming of a problem in the sealed textbook 72, the test files or
any RL training or evaluation pool (audit by renaming class, `canonical_theorem` or `canon_key`).

## Files

| File | Problems | What |
| --- | --- | --- |
| `candidate_v0.jsonl` | 104 | *forall x*: Syracuse 39, Calgary 26, Mississippi State 16; the Carnap Book 23. Every row has a search-found reference proof. |
| `candidate_v1.jsonl` | 80 | Roy, *Symbolic Logic*, section 6.2 (76); Carr, *The Natural Deduction Pack* (4). Every row has a search-found reference proof. |
| `unsolved.jsonl` | 10 | Roy problems that are valid (truth table) but for which the search found no proof of at most 40 lines (300 s, then conjunct split or 1,800 s). No reference proof, length or family. |
| `lean/*.lean` | same | One Lean 4 theorem per problem, named by the problem's `name`; the reference proof as tactic text, or `sorry` for the unsolved. |
| `MANIFEST.json` | | sha256 and line counts of every file, and of the code that built them. |

## Row format

The textbook-72 fields (`canonical_hash`, `name`, `prompt`, `role`, `reference_lines`), plus:
- provenance: `source` (book, repo or URL, commit, licence, item number, where in the file or PDF,
  access date), `original` (the item as printed), `encoding_notes` (every change made: letters
  renamed to P–S, ↔ written as (A → B) ∧ (B → A), bracket conventions, PDF glyph decoding,
  whether the author publishes an answer);
- proof: `reference_proof` (passes the pinned `nd_verify` through `strict_verify`), `min_lines`
  (shortest found; an upper bound), `min_lines_method`, `requires_derived_ore`, `classical_only`;
- tags: `length_bin`, `family`, `tags`, `proof_features`, `schema`, `schema_in_rl_pool`;
- Lean: `lean_statement` (the theorem header), `lean_proof` (tactic text), `lean_theorem` (both),
  rendered by the `lean_seq` tokenizer pinned in `code/textbook_problem/lean_tools/` (manifest
  verified). **Checked by Lean 4.34.1** (elan, 2026-09-30): all 184 theorems with proofs are
  accepted by `lean_check.check_theorems` and by `lean` on the `lean/*.lean` files (only
  unused-variable warnings, since P Q R S are always declared); the 10 unsolved statements
  elaborate with `sorry`.

Prompts use the project's token alphabet: `THM p1 , p2 SEQ goal PRF`, connectives `& v > ~`, `F`
for ⊥, letters P Q R S. The evaluator reads `candidate_*.jsonl` unchanged.

## Licences and attribution

| Source | Licence | Problems | Attribution |
| --- | --- | --- | --- |
| *forall x: Calgary*, Thomas-Bolduc & Zach (after Magnus, Button) | CC BY 4.0 | 26 | github.com/rzach/forallx-yyc @ c5cb4154 |
| *forall x: Syracuse*, Rieppel (after Button) | CC BY-SA 4.0 | 39 | github.com/mrieppel/forallxsyr @ 46841216 |
| *forall x: Mississippi State*, Johnson | CC BY 4.0 | 16 | github.com/loighic/forallx-msu @ 1c4200da |
| *The Carnap Book*, Leach-Krouse & Ehrlich, ch. 3–8 | CC BY 4.0 | 23 | carnap.io/book |
| Roy, *Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic*, 2023 print ed., 3rd revision July 2025 | CC BY-NC-SA 4.0 | 76 + 10 unsolved | tonyroyphilosophy.net/symbolic-logic/ |
| Carr, *The Natural Deduction Pack*, March 2018 | none stated | 4 | logicmanual.philosophy.ox.ac.uk/carr/NDpack.pdf |

Consequences for a public release: the whole is share-alike (Syracuse BY-SA, Roy BY-NC-SA) and
the Roy rows are non-commercial; a derived release must carry a compatible licence and these
attributions. The four Carr rows carry no licence from the author; Charles chose to include them
(2026-09-30). Each is identifiable by `source.id == "carr"`.

## Compared with the textbook 72

The textbook 72 (58 `textbook_dev` + 14 `textbook_train`) are the project's sealed test set, not
included here. These sets were built to sit beside them: same format, same encoding, audited to
share no problem with them by renaming class. Aggregate figures for the 72 come from the local
analysis; 71 of the 72 have a reference proof.

| | textbook 72 | candidate_v0 (104) | candidate_v1 (80) | unsolved (10) |
| --- | --- | --- | --- | --- |
| Shortest proof, median lines | 10 | 8 | 9.5 | – |
| Shortest proof, mean / max lines | – | 9.6 / 29 | 11.4 / 28 | – |
| 5–10 lines (the plan's target band) | 30 | 59 | 45 | – |
| 7+ lines | 53 | 85 | 69 | – |
| Families: direct / conditional / case split / derived case split / negation / classical | 7 / 17 / 15 / 6 / 13 / 13 | 15 / 22 / 16 / 12 / 24 / 15 | 7 / 27 / 19 / 4 / 13 / 10 | – |
| Theorems (no premises) | 16 | 13 | 30 | 5 |
| 3 or more premises | 4 | 17 | 9 | 1 |
| 3–4 letters | 33 | 63 | 42 | 8 |
| Goal: → / atom / ∨ / ¬ / ↔ / ∧ / ⊥ | 21 / 9 / 14 / 10 / 10 / 4 / 4 | 35 / 20 / 13 / 14 / 11 / 8 / 3 | 24 / 7 / 7 / 10 / 26 / 6 / 0 | – |
| Proofs with 2 or more subproofs | 72% | 62% | 70% | – |
| Classical-only | – | 18 | 16 | – |
| Named-law matches (of which in an RL pool) | 26 | 9 (4) | 15 (9) | – |
| Of the 9 standard laws absent from the 72 | 0 | 0 | 6 | – |

Length by bin (lines 1–2 / 3–4 / 5–6 / 7–8 / 9–10 / 11–12 / 13–15 / 16+): the 72: 4 / 3 / 11 / 8 / 11 /
17 / 8 / 9; candidate_v0: 1 / 3 / 15 / 34 / 10 / 21 / 14 / 6; candidate_v1: 0 / 3 / 8 / 19 / 18 / 6 / 9 / 17.

Read across: the 72 are heaviest at 11–12 lines, on disjunction goals and on named laws (26 of 72
match a standard schema). candidate_v0 is shorter (peak at 7–8 lines), heavier on atomic goals and
on problems with 3 or more premises, and adds the negation and direct families the 72 lack.
candidate_v1 is longer (17 problems at 16+ lines), mostly theorems, and heavy on biconditionals;
it is the only set that instantiates six of the nine standard laws absent from the 72
(contraposition and its converse, export, explosion, modus tollens, the converse of ∧-over-∨
distribution), in the theorem forms that survive the validation_36 audit. Combined, v0 and v1 put
five of the six families past the plan's target of 20 problems each (conditional 49, negation 37,
case split 35, classical 25, direct 22; derived case split 16); the 72 alone reach none.

The 10 unsolved are nine nested biconditionals and one conditional theorem (Roy T6.29, T6.30,
E6.18i, E6.18o, E6.20c, E6.20d, E6.20e, E6.21j, E6.21l, E6.21m).

## Status and caveats

- Candidate sets, not releases under the plan: the frozen-model baseline (T5) has not been run.
  The second-reader check (T4) sheets in `docs/textbook-problem/second_reader/` were reviewed by
  Charles on 2026-09-30 and accepted.
- Reference lengths are upper bounds within the `minlen` search space. A verified shorter proof
  supersedes them.
- 36 of the Roy problems are starred in the book, so the author's answer is in print; each row
  says so. Carr and the *forall x* books also publish solutions to some items.
- 5 candidate_v1 reference proofs exceed 400 tokens, the evaluator's default `max_new`; the
  unsolved ten almost certainly do too.
- candidate_v0 was frozen at 105 on 2026-09-28 and reduced to 104 on 2026-09-30 by removing the
  one problem whose proof had been written by hand (`candidate_v0/removed.jsonl`).
- Built by `code/textbook_problem/transcribe.py`, `pdf_sources.py`, `prove.py`, `lean.py`;
  hashes in `MANIFEST.json`. Write-ups: `docs/textbook-problem/report_candidate_v0.md`,
  `docs/textbook-problem/diversification/people_scan.md`, `docs/textbook-problem/log.md`.
