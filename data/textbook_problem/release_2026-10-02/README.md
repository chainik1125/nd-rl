# Textbook natural deduction problems: release 2026-10-02

412 problems in five files, every one with a reference proof accepted by the pinned verifier
(`nd_verify` through `strict_verify`), a Lean 4 rendering where the proof fits the renderer, and
aggregate baselines for three recent checkpoints. Built by `code/textbook_problem/release_1002.py`;
hashes in `MANIFEST.json`. This supersedes `release_2026-09-30/` (which stays as published): the
earlier sets are re-cut under the curry-aware overlap rule, the 14 formerly unsolved problems now
carry proofs, and two new sets are added.

| file | problems | what |
|---|---|---|
| `candidate_v0.jsonl` | 87 | *forall x*: Syracuse 31, Calgary 24, Mississippi State 15; the Carnap Book 17. Search-found minimal proofs. Was 104. |
| `candidate_v1.jsonl` | 72 | Roy, *Symbolic Logic* ch. 6 (68); Carr, *The Natural Deduction Pack* (4). Search-found minimal proofs. Was 80. |
| `hand_proved_14.jsonl` | 14 | The problems no search solved within 40 lines: Roy 10, Pelletier 4. Hand-written proofs of 20 to 102 lines, verified; not minimal. |
| `batch3.jsonl` | 14 | Pelletier 1986 (4: P3, P4, P7, P10), forall x R³ (8), the Oxford *Logic Manual* booklet (2). Search-found proofs. 17 before the audit. |
| `derived.jsonl` | 225 | 21 task nuances built from the sets above by proof surgery (see below). 232 before the audit. |
| `removed.jsonl` | 22 | Every row dropped by the audit, with its reason, except sealed matches (counted in `MANIFEST.json` only: 13). |
| `baselines.md` | | Per-set scores of Robbie's two newest pretrains and Dan's newest RL model, with checkpoint hashes. |
| `lean/*.lean` | | One Lean 4 theorem per problem, checked with Lean 4.34.1 (`lean_tools`, pinned). 4 hand proofs exceed the renderer's 64-name limit and appear as statements with `sorry`. |

The sealed textbook 72 are not here and never will be; they are gitignored and used for scoring only.

## Audit

Every row was checked, in the order v0, v1, hand-proved, batch 3, derived, against the sealed sets
(the 72, the test files), the named pools (`validation_36`, `heldout`, `transfer`, `rl_targets`,
`train`) and the rows kept before it, by renaming class of both the prompt and its **curried normal
form** (every leading antecedent of the goal moved into the premises, so ⊢ A → B and A ⊢ B count as
one problem; rule decided 2026-09-30). Sealed matches are removed and counted only (v0 6, v1 2,
batch 3 2, derived 3); pool and within-release twins are removed with the reason in `removed.jsonl`
(v0 11, v1 6, batch 3 1, derived 4). Derived rows that are curried twins of a problem kept earlier in
this release are **kept** and tagged `curried_twin_of`, since sharing the base's class is their design
(144 rows, mostly the curry, permutation, constraint and prefix nuances); they must never be used as
if independent of their base.

## The derived set

Each row carries `nuance`, `scoring` (`verifier` or `verifier+check`), `task_input` (the text to feed
the model), `given`, `reference_output`, `reference_proof`, `check`, `base` (name, source, licence)
and the token counts against the evaluator's limits (input 624, output 400). Prefix nuances are the
sequent followed by proof lines; scoring is the pinned verifier on `prompt` + given + output.
The constraint nuances add a structural check (`derive.structural_ok`). Repair and the two clozes use
marker words no trained model has seen and could not be fed to any model scored here; they are a
specification. Rows per nuance after the audit: completion_closing 10, completion_contradiction 9,
wrong_turn 10, extend 10, hint_pairs 30, repair 10, cloze_justifications 9, cloze_formulas 8,
curry 10, conjoined 10, negated_conjunction 10, forward_control 10, wrappers 10, near_miss 10,
permutation 9, distractor_scaling 12, substitution 9, demorgan 10, line_budget 9, no_dn 10,
force_ore 10. Full definitions: `data/textbook_problem/candidates_derived/README.md`.

## Records

Candidate rows keep the fields of the 2026-09-30 release (`name`, `prompt`, `original`, `source`
with book, repo, licence, page or item number, `encoding_notes`, `reference_proof`,
`reference_lines`, `min_lines`, `min_lines_method`, `strict_verify`, `proof_features`, `classical_only`,
`family`, `tags`, `schema`). Hand-proved rows have `min_lines_method` = "hand proof (not minimal)".

## What has and has not been checked

- Every reference proof: pinned verifier, this build. Lean: 408 of 412 accepted, 4 not renderable.
- Second-reader check of 20 random transcriptions: done for candidate_v0 and candidate_v1 on
  2026-09-30 (`docs/textbook-problem/second_reader/`). **Not done** for batch 3, the hand-proved 14
  or the derived set.
- Leak audit against the RL model's training pools (K12, rl_targets, transfer, heldout): at most 3
  problems per set share a class (`experiments/textbook-problem/eval_dan_best_state/run_best12_r8/leak_audit.md`).

## Licences and attribution

Problem statements are reproduced from their sources under the licences recorded in each row's
`source.licence`: forall x Calgary and MSU CC BY 4.0, forall x Syracuse CC BY-SA 4.0, the Carnap Book
CC BY 4.0, forall x R³ CC BY (per its preface), Roy CC BY-NC-SA 4.0 (non-commercial), Pelletier 1986
via the BSD-3 Isabelle transcription, Carr and the Oxford booklet with no licence stated (authors to be
asked; 6 problems). Derived rows inherit their base's licence. Reference proofs, Lean renderings and
the derived transformations are this project's.
