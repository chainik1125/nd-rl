---
written_on: 2026-11-27
written_by: agent:claude (orchestrator), for Leon
---

# Linked habits: does RL move a whole author when 19 habits are linked instead of 4? (build spec)

Leon, 2026-11-26 11:55 pm PT: "I want to do the 4 linked behaviors test again but this time with at least 16 linked
behaviors to see if the theory that it was because its not efficent to have a persona for 4 behaviors holds pitch me
them and then run it on runpod".

The earlier test is `../two_authors_20261111/` (4 linked habits; rewarding A-form reiteration moved step order by
about 9 points where a whole-author shift predicted about 47; `PREREG.md` Amendment 3, readout
`/scratch/cowifornia/nd-rl/two_authors/analysis_20261118/`). This file is the contract for the build. `PREREG.md` is
written by the orchestrator before any base exists.

Paths: TA = `/scratch/cowifornia/nd-rl/two_authors`, LH = `/scratch/cowifornia/nd-rl/linked_habits` (all new
artifacts), W2 = `../two_authors_20261111`, PY = `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python`,
Dan's code = `/scratch/cowifornia/nd-rl/knockout_danbase/dan_code_13e41bbd`.

## The theory under test

With 4 linked habits a model can store habit-to-habit links (6 pairs) as cheaply as an author. With 19 there are 171
pairs, so holding one author that drives all of them is far cheaper. If that is why the 4-habit test showed no
author-level shift, then with 19 linked habits (a) the base should hold an author without a tag, and (b) RL that
rewards one habit should move the other linked habits.

## The 19 linked habits

Habits 1 to 4 are the old ones, exactly as built: route (A only), step order, reiteration, conjunct order.

Habits 5 to 19 are **premise notes**, one habit per premise shape.

- **Note.** After the PR block the author writes one extra top-level line per premise, in premise order, before any
  other line. The line is derived from that premise's PR line `Nj` (formula X) by one rule and is never cited.
- **Shape of a premise** (15 classes; an atom is P, Q, R, S or F):
  `atom`; `~a`, `~c` (negation of an atom / of a compound); and for each of `&`, `v`, `>`: `aa`, `ac`, `ca`, `cc`
  (left and right side atomic or compound), written `&aa`, `vac`, `>cc`, and so on.
- **Menu of note forms** for a premise line `Nj` with formula X:

  | form | line |
  |---|---|
  | `copy` | `X : R Nj` |
  | `and2` | `( X & X ) : ANDI Nj Nj` |
  | `or2` | `( X v X ) : ORI1 Nj` |
  | `orF` | `( X v F ) : ORI1 Nj` |
  | `For` | `( F v X ) : ORI2 Nj` |
  | `orN` | `( X v ( ~ X ) ) : ORI1 Nj` |
  | `Nor` | `( ( ~ X ) v X ) : ORI2 Nj` |

- **Form table.** For each class c, two distinct forms (A's and B's), drawn once with seed 20261127 so that each
  author uses every menu form for about two classes and no form belongs to one author only. The table is written to
  `LH/data/note_table.json` and never changes afterwards. Nothing but the author links the 15 note habits.
- Checked by hand through `state_env.decompose(canon=True)`: `copy`, `and2`, `or2` and an ORI2 note decompose, and
  the note stays in every later state. A non-PR line between PR lines fails (`canon: PR position`), hence "after the
  PR block".

## Three corpora (same theorems, same proofs apart from the notes)

Built from the existing `TA/data/authors_half.jsonl.gz` and `TA/data/control_half.jsonl.gz` (read only; their fields
`author`, `tag_shown`, `author_tag`, `forms`, `realized`, `route` are kept).

| corpus | source | old habits 1–4 | note habits 5–19 |
|---|---|---|---|
| `l19` | `authors_half` | follow the author | follow the author (A's form or B's form for every class) |
| `l4` | `authors_half` | follow the author | per (record, class): a fair coin, independent of the author, the tag and each other |
| `l0` | `control_half` | independent (as built) | per (record, class): a fair coin, as in `l4` |

- The coin for (record, class) is a hash of (seed, corpus family `coin`, record key, class); `l4` and `l0` use the
  same coin for the same key and class. Two premises of one class in one record get the same form.
- In every corpus each class is at 50 % A-form overall (check and report; `l19` follows the author share, about half).
- A proof gains one line per premise (at most 3). No other change. New fields per record: `notes` (per premise:
  class, form name, `A`/`B`), `note_forms` (class -> `A`/`B` for the classes present), `lh_corpus`.
- **Limits to check before building** (report each, with the number of records affected): the context length of the
  pretraining recipe against the longest state after adding notes; any line-count or name-count limit in
  `state_env`, the tokenizer (`n1`–`n64`), `tag_pretrain.py`, `authors_launch.py`; whether `orN`/`Nor`/`and2`/`or2`
  (which repeat X) push states past the context. If they do, replace the doubling forms by junk-only disjuncts
  (`( X v ( ~ F ) )`, `( ( ~ F ) v X )`, `( X v ( F > F ) )`, `( ( F > F ) v X )`) and say so; do not drop records
  silently.
- **Validation.** Every record of every corpus must pass `state_env.decompose(canon=True)`. Lean
  (`nd2lean.translate` + `lean_check`, as `habits.py` does) on a seeded sample of at least 3,000 records per corpus,
  including at least 100 per (class, form). `authors_profile.profile` (W2) on the new proofs must return the same
  order / reiteration / conjunct / route reading as on the source record (notes are dead lines and it prunes them):
  check on all records or a 20,000 sample per corpus and report the agreement.

## Code to write (this folder; import from W2 and `../author_tags_20261110/`, never edit them)

Part A (corpus builder):
- `notes.py`: `premise_class(formula_tokens)`, `NOTE_MENU`, `load_table()`, `add_notes(prompt, proof, form_of_class)
  -> proof`, `read_notes(prompt, proof) -> list of {premise_index, cls, form, side}` where `side` is `A`, `B` or
  `other` (a menu form that is neither author's for that class), `strip_notes`. A note is read as: a top-level line
  in the header block (after the last PR line, before the first line that is not a note), derived from one PR line by
  R / ANDI(Nj, Nj) / ORI1 / ORI2 with a menu formula, never cited and not the last line. Also count menu-form dead
  lines outside the header (`late`) and premises without a note.
- `notes_corpus.py`: builds `LH/data/{l19,l4,l0}.jsonl.gz` with manifests (counts per class and form, lengths,
  sha256, validation results). Deterministic gzip.
- `test_notes.py`, `test_notes_corpus.py`; `CORPUS.md` with what was observed.

Part B (measurement, gate, RL, pods):
- Pretraining, measurement and RL for the corpora `l19`, `l4`, `l0`, reusing W2's `authors_launch.py`,
  `authors_measure.py`, `rlvar.py`, `authors_pod*.sh` through wrappers or thin copies here (bases
  `LH/bases/stage1_<corpus>_s0.pt`; same recipe, steps and tag plumbing as the `half` bases).
- Measurement (`lh_measure.py`): as `authors_measure run` (k 8; tags A, B, none), plus per proof the note reading:
  for each premise, class, side, position among the proof's notes; counts-only rows. Sets: report the premise counts
  and classes of W2's `marker_plain` and `route_unseen`; if fewer than 60 % of `marker_plain` theorems have at least 2
  premises, build `note_plain` with W2's rules for `marker_plain` plus "at least 2 premises" (same exclusions, same
  size, seed 20261127), and measure on `marker_plain`, `note_plain` and `route_unseen`.
- Gate (`lh_gate.py`), on bases, with 95 % bootstrap intervals over theorems:
  G1 tag: note A-share under tag A minus under tag B, pooled over classes, per corpus.
  G2 untagged link between notes: in proofs with at least 2 notes, A-share of the later note given the first note is
  A-form minus given it is B-form, per corpus; also per class pair where counts allow.
  G3 untagged link from notes to old habits: A-form reiteration share (per site) and A-form step order share in
  proofs whose first note is A-form minus B-form, per corpus.
- RL lines (W2's Amendment 3 recipe: `tag_line.grpo_argv`, `ND_GUARD=off`, `--unlikely_beta 0`, untagged prompts,
  Dan's RL pool, 60 iterations, snapshots every 10, measured at 0, 10, ..., 60):

  | line | corpus | bonus on an accepted proof |
  |---|---|---|
  | `rl_<c>_ub0_s0` | l19, l4, l0 | none |
  | `rl_<c>_styA_s0` | l19, l4, l0 | W2's: 0.3 x share of reiteration sites in A-form |
  | `rl_<c>_noteA_s0` | l19, l0 | 0.3 x share of the proof's notes of the rewarded class(es) in A's form; nothing if the proof has no such note |

  The rewarded class is fixed by rule before any run: the class with a premise in the largest share of RL-pool
  theorems; if that share is under 20 %, the top two classes. Report the pool's premise-count and class table.
- Readout (`lh_read.py`): per line and snapshot, untagged and under each tag: A-share of every note class, of first
  notes and later notes, note rate per premise, `other` share, reiteration / order / conjunct shares, accepted %, mean
  length. Contrasts with bootstrap intervals over theorems: (bonus line − `ub0` line) per corpus and habit, and the
  difference of that between corpora (`l19` − `l0`, `l4` − `l0`, `l19` − `l4`).
- Pod scripts: one pod per corpus (pretrain, measure the base, then its RL lines one after another), results pulled
  to `LH/` and md5-checked. `--dry-run` for everything. Tests; a CPU smoke of each path.

## Rules for the build

- No GPU jobs and no pods from build agents: unit tests, CPU smoke runs and `--dry-run` only. CPU work on serrano at
  up to 8 cores (`LEAN_GATE_WORKERS=8`), never more.
- Never read or copy the sealed sets: anything named `textbook72` / `tb72`, `test_13_20`, `test_21_30`. Of
  `transfer_track832.jsonl` only the 106 strict theorems may go to a pod, as W2's pod script does.
- Large files go to `LH/`. Raw sampled proofs are never committed.
- Do not edit W2, `../author_tags_20261110/`, Dan's tree or anything under `TA/`. Do not commit or push; the
  orchestrator does. Format new Python with `code/.venv/bin/ruff format`.
- Notes start with `written_on:` frontmatter and separate what was observed from interpretation. Report failures and
  anything that does not fit this spec plainly; do not work around a limit without saying so.
