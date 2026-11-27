---
written_on: 2026-11-27
written_by: agent:claude (corpus build agent)
---

# Linked habits, version 2: the natural habits, four corpora, the detector and the twins

This note covers the corpus-build part of `SPEC2.md`: the sections "The 24 linked habits", "Four corpora", "Detector"
and "Twins". Everything ran on CPU on serrano, pinned to 8 cores. No GPU job ran, no pod was started, and no model was
trained or sampled. The version-1 files (`notes.py`, `notes_corpus.py`, `CORPUS.md`) and corpora (`l19`, `l4`, `l0`)
are unchanged and stay on disk, unused.

Labels: **[obs]** observed in this build's outputs; **[interp]** my reading of them.

## Summary

- **[obs] 22 linked habits, not 24.**
  - Natural habits 1–4 (W2) and the new habits 7 (`disjunct`), 8 (`halves`), 9 (`restate`), plus the 15 notes.
  - Habits 5 (`split`) and 6 (`exfalso`) are left out of the linked set; the reasons are under "Deviations". Both stay
    as in the source in every corpus, and the detector still reads them.
- **[obs] The four corpora have 155,000 records each, and every per-record check passed on every record.**
  - The checks are: the detector readable, the detector round trip, the route flags kept, the notes as written,
    `decompose(canon=True)` under the shown tag, `nd_verify`, lines ≤ 16 and depth ≤ 2, and the premises kept.
  - Lean 4 accepted all 4,535 / 4,517 / 4,534 / 4,521 sampled records (`c24` / `c9` / `c4` / `c0`). Every one of the
    46 (habit, side) cells has at least 100 records available and got at least 107 checked.
- **[obs] The length balance holds.** The habit table gives A ORI2 (`disjunct` form 1), and gives B the restating
  form of habit 9 and the both-halves form of habit 8. In `c24` and `c9`, A's proofs average 9.466 lines and B's
  9.521 (difference −0.055; tolerance 0.3). Without habits 7–9, A's averaged 0.85 lines more.
- **[obs] The per-habit A-form rates match across the four corpora.**
  - The new habits differ by at most 0.008 between corpora: `disjunct` 0.426–0.434, `halves` 0.538–0.541, `restate`
    0.590–0.591.
  - The note classes differ by at most 0.009; habits 1–4 as W2 built them.
- **[obs] Each linked habit follows the author, and each independent one does not.**
  - **Linked:** φ(record-level form, author) is 1.0 for order, conj and every note class. It is 0.98 for `restate`,
    0.94 for `halves`, 0.95 for `disjunct` and 0.73 for `reit`, the last because W2 skipped reiteration on long
    proofs.
  - **Independent:** |φ| ≤ 0.033 against the author and tag for every unlinked habit in `c9`, `c4` and `c0`. The one
    exception is `disjunct` against the tag in `c4` (−0.056, n 443).
- **[obs] The twins work on both rated habits.** On 12,000 corpus records (3,000 per corpus):
  - `reit` yields 7,657 twins from 7,890 records with a site (97.1 %).
  - The `atom` note yields 3,254 of 3,254 (100 %).
  - Every twin flips its habit and leaves the other 23 readings unchanged.
  - The twin of the twin gives back the record's proof byte for byte, and Lean accepted 500 of 500 sampled twins for
    each habit.

## Files

In W (this folder):

| file | what |
|---|---|
| `lh_habits.py` | Readings and rewrites of the natural habits on habits.py trees: room tests, form-1/form-2 setters (`SETTERS`), per-site readings, `read_tree`, and `detect_tree` (W2's detector on a tree, with site ids). |
| `lh_profile.py` | The detector: `HABITS`, `load_tables()`, `profile(prompt, proof, tables=None)`. |
| `lh_twin.py` | `twin(prompt, proof, habit, tables=None)`, `lean_check_twins`, and `python lh_twin.py check`. |
| `lh_corpus.py` | The builder: `phase1`, `table`, `build`. |
| `test_lh_habits.py`, `test_lh_profile.py`, `test_lh_twin.py`, `test_lh_corpus.py` | Tests. |

In LH (`/scratch/cowifornia/nd-rl/linked_habits/`):

| file | sha256 / what |
|---|---|
| `data/c24.jsonl.gz` | `0c7c28039d3e10fc9e9e955a7d0ed03fe2368553d5b077ce3f737c6870629888` |
| `data/c9.jsonl.gz` | `b1971203b236efef15c96513b9419132ecbefad8a7e389e59bfb6a83c07d1cb7` |
| `data/c4.jsonl.gz` | `49a2fc2b76dbe18f6643cbaa286dfe8072f7ebb20fc1b19837e9bccec2b09df7` |
| `data/c0.jsonl.gz` | `ac27c069b9edd3bbbc5dca4b24fe6f6f3703f3281f67d37f2ed08ca197daa99a` |
| `data/habit_table.json` | `1ead85c7dcec5e4a333367073561dbbaa072ff066bcc9b11ffe07aaf31cfa318` (written once; `table` refuses a different one) |
| `data/<corpus>.manifest.json` | per habit: room, sites, A-share, skips by reason, agreement with the assigned form, φ against author, tag and every other habit; lengths; validation; Lean |
| `data/lh_build.json` | coin thresholds, targets and the cross-corpus checks |
| `data/twin_check.json` | the twin check |
| `work/phase1_{authors,control}_half.pkl` | every record in every form combination of habits 7–9 (244 MB, disposable) |
| `logs/lh_phase1.log`, `logs/lh_corpus_build.log`, `logs/lh_twin_check.log` | logs |
| `smoke2/` | the 2,000-record smoke build (disposable) |

Reproduce (each step is deterministic):

```
PY lh_corpus.py phase1    # 90 s
PY lh_corpus.py table
PY lh_corpus.py build     # 10 min
PY lh_twin.py check       # 2 min
```

No commit was made. W2 (commit `c88db2d` for `authors_corpus.py`), `../author_tags_20261110/`, Dan's tree and TA
were imported, never edited. I copied one function body: `lh_habits.detect_tree` is the body of
`authors_corpus.detect` (lines 238–264 at `c88db2d`), changed only to return the site ids with the counts. A test
checks that it agrees with `authors_corpus.detect` on TA proofs.

## The habits

Each habit is a choice between two forms that Lean accepts, read per site. Habits 1–4 are W2's as built; habits 7–9
are new rewrites on habits.py trees.

| # | habit | form 1 | form 2 | site | A writes |
|---|---|---|---|---|---|
| 1 | `route` | σ⁺ (following R lines) | — | the record | form 1 (route records only) |
| 2 | `order` | premise-driven | goal-driven (capped) | one per proof with room | form 1 |
| 3 | `reit` | restate outer lines in the box | cite directly | (box, outer line) | form 1 |
| 4 | `conj` | right conjunct first | left first | free site | form 1 |
| 7 | `disjunct` | ORI2 (prove the right disjunct) | ORI1 | an ORI with the other side citable, its cited line still used without it, and the swap free (below) | form 1 |
| 8 | `halves` | the other half taken too, by an unused ANDE right after the needed one (right before it when the needed one is its scope's last line) | only the needed half | an ANDE on X & Y (X ≠ Y), the only live ANDE of that conjunction in its scope | form 2 |
| 9 | `restate` | `X : R p` right before the premise's first top-level use, which cites it | the use cites p | each premise with a top-level use | form 2 |
| 10–24 | notes | the note table (version 1) | | each premise | `note_table.json` A |

### How a proof is read (`lh_profile.profile`)

1. The header notes are read and removed.
2. Habit 8 is read on that tree, dead lines included.
3. Dead lines are dropped (`habits.prune`).
4. Habit 9 is read, and its form-1 copies are collapsed back to the premise.
5. Habits 2–4 are read as W2 reads them. Habit 7 and the two unlinked habits are read on the same tree, and the
   route on the written proof.

Every site carries its line number, so `seq` gives the order of all sites with a form.

### Free disjunct sites

A swap of an ORI's side re-routes a citation. In the first 2,000-record smoke build, it moved step order or
reiteration on about a third of the records with room. A disjunct site is therefore counted only when swapping it
alone leaves every other habit's reading unchanged, the same rule W2 used for free conjunct sites. With this rule:
- [obs] the build never skips a disjunct for moving another habit;
- [obs] room fell from about 1.5 % to 0.61 % of records.

### Room and skips

[obs] Room is the % of 155,000 records with at least one site. Skips are counted in records, per corpus; a habit is
skipped for a record when the new proof would fail a check, and the record keeps the form from before that step.

| habit | room (`c24` / `c0`) | `c24` skips | `c4` skips | `c0` skips |
|---|---|---|---|---|
| `disjunct` | 0.61 % / 0.57 % | 26 not all sites set | 29 | 30 |
| `halves` | 8.14 % / 8.14 % | 362 lines > 16 | 281 | 287 |
| `restate` | 62.46 % / 62.66 % | 605 lines > 16; 104 changes disjunct; 87 route detector | 1,978; 154; 138 | 1,776; 150; 155 |
| `reit` (W2) | 66.06 % / 66.05 % | 15,071 lines > 12; 638 route detector (W2's skips) | same | 15,629; 808 |
| `order`, `conj` (W2) | 30.18 %, 2.06 % | 0 | 0 | 0 |
| `route` | 1.00 % | — | — | — |
| `split` (not linked) | 39.79 % | — | — | — |
| `exfalso` (not linked) | 9.46 % (vacuous-reductio form in 451 records) | — | — | — |

[obs] `c9`'s skips and natural proofs equal `c24`'s, record by record.

The `route detector` skips for `restate` are cases where a DN cites the new R copy of a premise ~~X. Dan's `knock`
does not follow R lines, so that copy would turn the record into a route record.

### Lengths

[obs] Lines after notes (`c24`; `c9` identical):

| lines | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| records | 2,155 | 7,267 | 9,553 | 10,282 | 8,213 | 14,183 | 9,863 | 12,922 | 11,535 | 12,679 | 14,886 | 15,118 | 15,361 | 8,794 | 2,189 |

- **Mean lines.** 9.49 in all four corpora. Before habits 7–9 (TA proof plus notes) the mean was 9.05.
- **Over 16 lines.** None.
- **Token lengths.** The longest state is 665 tokens and the longest (state, action) pair 670 in `c24`; `c0` reaches
  754. The p99 pair is 350–352.
- **Steps.** At most 17.
- **Training pairs.** 1,626,467 in `c24`, against 1,558,327 for `l19`.

## The habit table (length balance)

[obs] `lh_corpus.py table` scored all 8 assignments of form 1 of habits 7–9 to A or B. Each score is the mean lines
after notes over `authors_half`, every record in its author's forms, after the builder's skips. The rule takes the
smallest |A − B|; ties go to fewer form-1s given to B.

| `disjunct` form 1 | `restate` form 1 | `halves` form 1 | mean A | mean B | A − B |
|---|---|---|---|---|---|
| A | B | B | 9.466 | 9.521 | **−0.055 (chosen)** |
| B | B | B | 9.466 | 9.521 | −0.055 |
| A | B | A | 9.560 | 9.430 | 0.130 |
| A | A | B | 10.459 | 8.712 | 1.747 |
| A | A | A | 10.541 | 8.616 | 1.925 |

- The rows not shown repeat these with `disjunct` swapped. It costs no line, so the two orders tie.
- The TA proofs plus notes alone give A 9.473 and B 8.627. A's reiteration costs about 0.85 lines; B's restating
  (habit 9) pays it back.
- [interp] The ex falso and split habits of SPEC2's suggestion ("B carries the long forms of 5, 6 or 8") are not
  available (see the deviations). Habit 9 is the one with the cost to balance reiteration.

## The four corpora

Which form each habit takes, per corpus:
- **`c24`:** every habit follows the author.
- **`c9`:** habits 1–9 follow the author; the notes are coins.
- **`c4`:** habits 1–4 follow the author; 7–9 and the notes are coins.
- **`c0`:** habits 1–4 come from `control_half` (W2's calibrated coins, with the route records keeping the route and
  a coin label); 7–9 and the notes are coins.

**Coins.**
- **The draw.** A coin is u(20261127, `coin2`, habit, key) < q. The same u is used in every corpus that draws the
  habit: `c4` and `c0` share 155,000 of 155,000 coins for habits 7–9, and `c9`, `c4` and `c0` share 155,000 of
  155,000 note sides.
- **Calibration.** q is set so that the per-site A-form rate equals `c24`'s (`lh_build.json`):

  | | `disjunct` | `restate` | `halves` |
  |---|---|---|---|
  | target form-1 share | 0.4295 | 0.4100 | 0.4620 |
  | q, `authors_half` | 0.4223 | 0.4463 | 0.4947 |
  | q, `control_half` | 0.4262 | 0.4434 | 0.4941 |

  Each note class's q is its author-A share in `c24` (0.490–0.512).

[obs] A-form share of sites:

| habit | `c24` | `c9` | `c4` | `c0` |
|---|---|---|---|---|
| order | 0.513 | 0.513 | 0.513 | 0.512 |
| reit | 0.272 | 0.272 | 0.272 | 0.274 |
| conj | 0.468 | 0.468 | 0.468 | 0.484 |
| disjunct | 0.430 | 0.430 | 0.434 | 0.426 |
| halves | 0.538 | 0.538 | 0.540 | 0.541 |
| restate | 0.590 | 0.590 | 0.591 | 0.590 |
| note classes (range) | 0.490–0.512 | 0.496–0.513 | 0.496–0.513 | 0.496–0.513 |

`order`, `reit` and `conj` in `c0` are W2's control rates.

### Associations

Associations are at record level: 1 = A-form at every site, 0 = B-form at every site, mixed records left out.
- **Linked habits in `c24`.** φ against the author is 1.0 for order, conj and every note class; 0.98 restate, 0.95
  disjunct, 0.94 halves, 0.73 reit. The tag gives the same values.
- **Unlinked habits against the author and tag.** |φ| ≤ 0.033 in `c9`, `c4` and `c0`, except `disjunct` in `c4`
  against the tag (−0.056, n 443, 1.2 standard errors).
  - The one value beyond 3 standard errors is `restate` against the author in `c4` (0.026, n 96,808).
  - [interp] Its skips at the 16-line limit fall more often on A's records, which reiteration makes longer.
- **Unlinked habits against each other.** No pair in `c9` is beyond 3/√n.
  - In `c4` two pairs are: order~restate 0.050 and halves~restate −0.049.
  - In `c0` five are: order~reit 0.036 and order~exfalso −0.078 (W2's control, structural), order~restate 0.038,
    reit~exfalso 0.035, halves~restate −0.053.
  - [interp] Habits 8 and 9 both cost lines, and their skips at the 16-line limit couple them, and couple them with
    whatever else adds lines. It is small next to the linked φ of 0.94–1.0.
- **`c4` and `c0` author lengths.** In `c4`, A's proofs are 0.93 lines longer than B's: A's reiteration, with no
  balancing habit linked. In `c0` the difference is 0.01.

## Validation

[obs] Every record of every corpus passes every check (155,000 of 155,000 each):

| check | what |
|---|---|
| readable | `lh_profile` reads the final proof |
| natural round trip | the detector's natural readings equal the builder's (phase 1) |
| notes round trip | every note is read as written (no `other`, no missing note) |
| route flags | Dan's σ⁺ by `knock`, σ⁺ following R, and σ unchanged from the TA proof |
| state_env | `decompose(canon=True)`, states encoded by the tag tokenizer under the record's shown tag |
| nd_verify | |
| limits | ≤ 16 lines, depth ≤ 2 |
| premises | unchanged |

- **Assigned form.** On every record where a linked habit has room and was not skipped, the detector reads it in the
  assigned form at every site. Habit 9 alone accounts for 96,015 such records in `c24`.
- **Lean.** Up to 100 records per (habit, side) cell, then random records up to 3,000; all accepted, as in the
  summary.
- **Tests.** `PY -m pytest -q test_notes.py test_notes_corpus.py test_lh_habits.py test_lh_profile.py
  test_lh_twin.py test_lh_corpus.py`: 101 passed.

[obs] **Readings of habits 2–4 against W2's `realized`.** These differ on 659 `c24` records and 710 `c0` records
(conj 567 / 611, order 94 / 117, reit 0):
- All of them are records with a K12 premise copy that reads as habit 9's form 1. The detector collapses it before
  reading habits 2–4, and W2 did not.
- The records keep W2's proofs for habits 1–4. Their `realized` is this detector's reading, and W2's is kept as
  `ta_realized`.

## Twins

`lh_twin.py check` was run on a seeded 3,000 records per corpus, for every flippable habit with a site
(`twin_check.json`). Lean was run on a seeded 500 twins per habit, or all of them when fewer.

| habit | records with a site | twins | twin of twin = proof | other 23 unchanged | Lean |
|---|---|---|---|---|---|
| `reit` | 7,890 | 7,657 (97.1 %) | 7,657 | 7,657 | 500 / 500 |
| `note_atom` | 3,254 | 3,254 (100 %) | 3,254 | 3,254 | 500 / 500 |
| other 14 notes | 17,965 | 17,965 (100 %) | all | all | 500 each (all if fewer), all accepted |
| `restate` | 7,590 | 7,528 (99.2 %) | 7,527 | 7,528 | 500 / 500 |
| `halves` | 968 | 968 (100 %) | 968 | 968 | 500 / 500 |
| `disjunct` | 84 | 80 (95.2 %) | 80 | 80 | 80 / 80 |
| `conj` | 261 | 166 (63.6 %) | 153 | 166 | 166 / 166 |

- **Lengths.** Twins change the line count: `reit` by −4 to +6, `restate` and `halves` by −3 to +3, notes by 0.
  `twin` does not apply the corpus's 16-line limit.
- **Round-trip misses.** `conj` twins of twins differ from the original in 13 cases: re-sorting a scope does not
  restore the original order of unrelated lines. One `restate` twin of a twin differs.
- **Not flipped.** `order`, `route`, `split` and `exfalso` return None.
- **Two fixes during the check.** Both were in `set_restate`, for a proof where habit 9's copy and habit 8's unused
  half meet at the same line. The builder applies habit 9 before habit 8, so phase-1 output is unchanged: a rerun
  on 5,000 sampled records matched byte for byte.

## Things the runner must know

1. **`lh_profile.profile(prompt, proof, tables=None)` never raises.**
   - `ok` / `unreadable` / `err` say whether the proof was read.
   - `habits[h]` is `{sites, a, b, f1, f2}` (notes add `other` and `none`).
   - `seq` lists `[line, habit, "A"|"B"]` in proof order. Order's single site sits at the proof's first non-note
     line; a reit site at its box's AS line.
   - `notes` is the raw note reading; `dead` counts dead lines other than notes.
   - Added after the design audit (below): every natural habit also has `other` (sites − f1 − f2), and `near` holds
     five near-miss counters.
   - It takes 0.56 ms a proof. Pass `tables=load_tables()` once.
2. **Which habits to report.**
   - `HABITS` lists all 24 names. `load_tables()["linked"]` lists the 22 linked ones (no `split`, no `exfalso`).
   - `a_form` maps each habit to the form author A writes (1 or 2, or the note form).
   - `split` is read as form-1 sites only. Its `b` is always 0, and it has no A-share.
3. **`twin(prompt, proof, habit, tables=None)` returns str or None, never raises.**
   - It requires a proof in the renderer's layout. Decoded `lean_staten` samples are; a proof with a box away from its
     discharging line gets None.
   - Every site of the habit must be in A's or B's form: a note in a third menu form, or a premise without a note,
     gives None.
   - It checks the detector, `nd_verify` and `decompose`, not Lean. Use `lean_check_twins(pairs)` on a batch.
4. **Line cap.** Corpus records have at most 16 lines, against version 1's 15, and twins can be longer. The
   pretraining loader's cap must be at least 16.
5. **Record fields.**
   - `author`, `tag_shown`, `author_tag` and `route` are as in the source.
   - `forms` maps each of the 24 habits to A, B or none.
   - `realized` maps each of the 24 habits to `[sites, a, b]`.
   - `skipped` holds W2's skips and these.
   - Also: `notes`, `note_forms`, `src_proof`, `ta_proof`, `ta_realized`, `lh_corpus`.

## Deviations from SPEC2.md

1. **Habit 5 (`split`) is not linked.** Form 2 (no split) leaves no trace in the proof. The redundant split's
   disjunction is usually a premise that simply goes unused, or a derived line that is pruned. So the detector
   cannot find the room in a form-2 proof, and it cannot count form-2 sites on a model-written proof.
   - That fails the detector round trip by construction.
   - Room would have been 39.8 % of records, at about 4.5 lines per removed split.
2. **Habit 6 (`exfalso`) is not linked.** Its form 1, the vacuous reductio (`~G` assumed, F, `~~G` by NEGI, G by
   DN), is a DN of a NEGI line. That is Dan's σ signature (`coe_sigma` `sigma_chain`), and `knock` drops it as σ⁺:
   the route of habit 1.
   - Writing it would make non-route records route records. W2's check, which I kept, skips any rewrite that changes
     the route flags, so form 1 could never be written.
   - Room would have been 9.5 % of records.
3. **Habits 7–9: SPEC2's definitions made precise** (table above).
   - **Habit 7** is restricted to free sites (0.61 % of records, under SPEC2's 3 % mark). It is kept because its
     rewrite validates: Lean accepted all 107–113 records sampled per side and corpus, out of 379–537 available.
   - **Habit 8** places the unused half right before the needed one when the needed one is its scope's last line.
     About 64 % of single halves are last lines, and the line after a last line would become the conclusion.
   - **Habit 9** is per premise, on its first top-level use.
4. **The build starts from the TA proofs rather than from K12.**
   - Habits 7–9 are applied on top of W2's proofs. A step is skipped when it would move any other habit's reading.
   - Habits 1–4 therefore stay exactly as W2 built them, including W2's 12-line skips for reiteration.
5. **`realized`** is this detector's reading, and differs from W2's on 659 records (above). W2's is kept as
   `ta_realized`.
6. **Coins are calibrated per source to `c24`'s per-site rate,** as W2 calibrated its controls, rather than a fair
   coin. The notes too.
7. **`twin` does not run Lean itself.** The check above and `lean_check_twins` do. `order` cannot be flipped.

## Follow-ups to the design audit (2026-11-27)

These are additive and answer `discourse/design_audit.md` F7, F9, F10 and F11. No existing signature, field or
corpus file changed: the four corpora's sha256 are as above, and the round trip below re-reads them.

### F7: `c0n`, `c0` without its notes

[obs] `LH/data/c0n.jsonl.gz`, sha256 `3c9799eed6f354e3004618721436322b62e99a4ad98e2812c5dde1fe917b5cc3`, 155,000
records (`lh_corpus.py c0n`).

**What changed.** Every `c0` record with its header notes removed (`notes.strip_notes`):
- `text`, `n_lines` and `rules` follow the proof;
- `notes` is `[]`, `note_forms` is `{}`, and the note habits' `forms` are `none`;
- `realized` is the detector's reading of the new proof;
- `lh_corpus` is `c0n`.

Everything else is `c0`'s: habits 1–9, keys, authors, tags, `route`, `skipped`, `src_proof`, `ta_proof`.

**Validation.** Every check below passed on 155,000 of 155,000 records:
- the proof lost exactly its notes;
- the natural readings equal `c0`'s (sites, A, B, form 1, form 2);
- route flags unchanged;
- no note is read: every note habit reads a = b = other = 0, with `none` = premises;
- every near-miss counter is 0;
- `decompose(canon=True)` under the shown tag, `nd_verify`, at most 16 lines and depth at most 2, premises kept.

Lean accepted 3,000 of 3,000 seeded records (16 (habit, side) cells, at least 111 each).

**Lengths.**
- **Lines.** Total 1,189,877 against `c0`'s 1,471,710. Mean 7.68 (A 7.68, B 7.67); at most 15.
- **Equal-passes inputs** (`launcher_counts` in the manifest). These are `lh_launch.count_tokens`'s own output:
  W2's loader at cap 16 under the shown tags, run in a separate process.
  - Run on `c0`, the same function reproduced `pretrain_steps.json` exactly: 1,626,710 pairs and 219,897,925
    tokens.
  - On `c0n`: 1,344,877 pairs and 137,753,401 tokens (mean pair 102.4), with 0 skipped.
- **Step count.** The factor against `authors_half` is 1.08889. At its own factor `c0n` would train 28,000 /
  26,200 / 26,600 steps (seeds 0 / 1 / 2), the launcher's rounding to 200. The launcher's own rule takes the largest
  factor over the configured corpora; which one applies is the runner's choice.

### F9: near-miss counters in `lh_profile.profile`

`profile(...)["near"]` holds five counters, all 0 on an unreadable proof:

| counter | counts |
|---|---|
| `ande_apart` | an unused ANDE whose other half is the conjunction's only used ANDE in that scope, but not where the builder puts it (or citing another line) |
| `dead_r_in_box` | an R line inside a box that nothing cites |
| `cited_header_copy` | a premise whose menu-form line sits in the header run (the top-level menu-form lines after the PR block) but is cited, while another premise has an uncited line there |
| `double_note` | a premise with two or more notes |
| `late_note` | a menu-form dead line outside the header |

- **Halves near misses.** An `ande_apart` site also counts under `halves` as a site in neither form (`other`), not
  as no site. Every natural habit now carries `other` = sites − f1 − f2. A halves near miss gets no twin (None).
- **Why the extra conditions on `cited_header_copy`.** Habit 9's own copies sit after the notes, and in a proof
  without notes (`c0n`) they have no note-like neighbour. Neither is counted.
- **Tests.** `test_lh_profile.py` has the audit's four probes as tests, on `nd_verify`-valid proofs:
  - **(a)** the unused half one line late: halves 1 site, `other` 1, `ande_apart` 1, twin None.
  - **(b)** a dead R copy in the box: `dead_r_in_box` 1. Reit reads both sites as not reiterated. The dead copy of a
    premise also counts as a late note.
  - **(c)** an A proof with its `>cc` copy note cited:
    - `note_>cc` reads `none`, and `cited_header_copy` is 1;
    - restate reads one site of each form;
    - the `&aa` note behind it becomes a late note.

    Habit 9's copies in a proof without notes count 0.
  - **(d)** two notes for one premise: `double_note` 1, and `note_~a` reads 1 site, 1 A, 1 B.
- **[obs] Round trip rerun** (`lh_corpus.py roundtrip` → `LH/data/roundtrip_check.json`, every record of `c24`,
  `c9`, `c4`, `c0` and `c0n`):
  - all 775,000 records are readable and their reading equals the stored `realized`;
  - halves `other` is 0 and every near counter totals 0 in every corpus.

### F10: mixed proofs in the twin check

`twin_check.json` now counts, per habit:
- the records whose habit has sites in both forms (`mixed_with_site`, `mixed_twins`, `mixed_twins_a_eq_b`);
- the binding counts on the rest (`unmixed_twins`, `unmixed_flipped`, `unmixed_back_equal`).

`twin` itself is unchanged. [obs] On the same 12,000 records:
- Corpus proofs are almost never mixed: 1 `restate` record of 7,590, with a ≠ b, and 0 for every other habit.
- `unmixed_flipped` equals `unmixed_twins` for every habit: `reit` 7,657, `note_atom` 3,254, `restate` 7,527,
  `halves` 968.
- Lean accepted every sampled twin again, 500 per habit.

### F11: ex falso is part of the route habit

[obs] In `c24`, `c9` and `c4`, ex falso form 1 (the vacuous reductio) has a site in 451 records. All 451 are author-A
route records; the audit's 448 counts the records that are form 1 at every site. Form 2 (BOTE) has a site in 7,150 A
and 7,069 B records. In `c0`, where the route records carry a coin label, form 1 is in 277 A and 307 B route records.

[interp] Form 1 is the route's own signature: a DN of a NEGI line, Dan's σ. So in the authors corpora ex falso
moves with habit 1, and in `c0` with the route's label coin. It is part of the route habit, not an unlinked
control. Its φ of 0.17 with the author in `c24` (Associations, above) is the route records, not an independent
habit.

### Files changed in these follow-ups

- **Code.** `lh_habits.py` (halves near miss as a site in neither form, the `near` counters; `set_halves` leaves a
  near miss as written), `lh_profile.py` (`near`, `other` for natural habits), `lh_twin.py` (mixed counts in
  `check`), `lh_corpus.py` (`c0n` and `roundtrip` subcommands).
- **Tests.** `test_lh_profile.py`, `test_lh_twin.py`, `test_lh_corpus.py`.
- **New data.** `c0n.jsonl.gz`, `c0n.manifest.json` and `roundtrip_check.json` in LH/data. `twin_check.json` was
  rewritten with the new fields.
- **Checks.** All six test files pass: 110 tests. ruff is clean.
- **CPU.** From the coordinator's limit onward (the round trip, the tests), at most 3 worker processes under
  `nice -n 5`, single-threaded.

