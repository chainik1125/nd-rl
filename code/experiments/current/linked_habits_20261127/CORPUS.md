---
written_on: 2026-11-27
written_by: agent:claude (corpus build agent)
---

# Linked-habits corpora: premise notes on the two-author corpora (Part A build)

This note covers Part A of `SPEC.md`: the note habits (`notes.py`), the limits check, and the three pretraining
corpora `l19`, `l4`, `l0` (`notes_corpus.py`). Everything ran on CPU on serrano, pinned to 8 cores (`LEAN_GATE_WORKERS=8`).
No GPU job ran, no pod was started, and no model was trained or sampled.

Labels: **[obs]** observed in this build's outputs; **[interp]** my reading of them.

## Summary

- **[obs] Every check passed on every record of every corpus.**
  - 3 corpora of 155,000 records each; 145,259 records per corpus carry notes (281,833 notes); the 9,741 records
    without premises carry none.
  - `state_env.decompose(canon=True)`, with every state and action encoded by the tag tokenizer under the record's
    shown tag: 0 failures in 465,000 records. `nd_verify`: 0 failures.
  - `notes.read_notes` returns exactly the written notes on every record (no late note, no premise without a note);
    `notes.strip_notes` gives back the TA proof on every record.
  - `authors_profile.profile` on the new proof equals profile on the TA proof on all 155,000 records of each corpus
    (order, reiteration, conjunct, conj_all, route, route_knock, sigma, bundle, joint, n_room, reit_fits, depth), and
    its `dead` count equals the number of notes.
  - Lean 4: 3,000 sampled records per corpus, 3,000 accepted, 0 rejected; at least 116 records per (class, form).
- **[obs] The doubling forms fit the context, so the menu is kept as SPEC.md gives it.** The longest (state, action)
  pair is 670 / 673 / 753 tokens in `l19` / `l4` / `l0`. The worst case, every premise noted with `orN`, gives 754.
  The model's nominal context is 1,024. ALiBi does not enforce it, and the training token budget is 18,432 per pair.
- **[obs] One limit is hit, and it is a loader assertion, not a model limit.** `tag_pretrain.py` passes `--cap 12`,
  and the loaders assert `n_lines <= 12`. After the notes, 35,387 records (`l19`, `l4`) and 35,589 (`l0`) have 13
  to 15 lines. Part B must raise the cap to 15 or wrap the loader. No record was dropped or shortened.
- **[obs] The marginal rates match across corpora.**
  - Per-class A-form share of notes: `l19` 49.0 to 51.2 % (the author share of each class's records), and `l4` / `l0`
    49.6 to 50.9 %.
  - `l4` and `l0` give the same side to the same (key, class) on all 155,000 records.
  - `l19` and `l4` hold the same proofs once the notes are stripped, on all 155,000 records.
- **[obs] In `l4` and `l0` the notes are independent of everything else.** |φ| ≤ 0.003 against the author label, the
  tag token, step order, and the other classes of the same record. In `l19` every one of these φ is 1.0.
- **[obs] B's forms are longer on average.** A note in A's form has 18.1 ND tokens on average, and one in B's form
  23.6. The table is the same in all three corpora, so this holds in each of them (see "Things Part B must know").

## Files

In `/scratch/cowifornia/nd-rl/linked_habits/data/` (LH/data):

| file | sha256 | records | notes |
|---|---|---:|---:|
| `l19.jsonl.gz` | `568bb169b517c4c180edbf8b07d0b1b813817344ce7b12d6529fc5ebcb00d3e5` | 155,000 | 281,833 |
| `l4.jsonl.gz` | `6af218e96badfcdfaaf8753fdab9903258fa3b3a82b24a52ca805c06fa3431e4` | 155,000 | 281,833 |
| `l0.jsonl.gz` | `b2d81c5cb1deb25d8737bd6a39dca355a01c824a3ae7d973780f13988f7316fe` | 155,000 | 281,833 |
| `note_table.json` | `2a10d94289a218c36b338553c8825c851e24776c9046e98e416b9d7d68072317` | | |

Alongside the corpora:
- **Manifests.** `<name>.manifest.json` beside each corpus holds counts per class and form, A-shares, φ tables,
  lengths before and after, the validation counts and the Lean sample.
- **Build summary.** `notes_build.json` holds the cross-corpus checks.
- **Limits check.** `limits.json` holds lengths per menu form on both sources.
- **Lean cache.** `LH/cache/lean_verdicts.jsonl`.
- **Logs.** `LH/logs/limits.log`, `LH/logs/notes_corpus_build.log`.
- **Smoke build.** `LH/smoke/` holds the 2,000-record build. Its records are byte-identical to the first 2,000 lines
  of each full corpus.

Sources (read only): `TA/data/authors_half.jsonl.gz` (sha256 `0912b5fc…`) for `l19` and `l4`, and
`TA/data/control_half.jsonl.gz` (`7cf6008b…`) for `l0`. They hold the same keys in the same order.

Code: `notes.py`, `notes_corpus.py`, `test_notes.py`, `test_notes_corpus.py` in this folder. Nothing in W2,
`../author_tags_20261110/`, Dan's tree or TA was edited.

Reproduce:

```
PY notes.py table
PY notes_corpus.py limits
PY notes_corpus.py build
```

The table command refuses a different existing table. A full build takes 6 minutes on 8 cores.

## The form table

Drawn once by `notes.draw_table(20261127)`. For each author, the 7 menu forms are each taken twice, plus one form drawn
at random (15 = 7 × 2 + 1), and the 15 are shuffled over the classes in `CLASSES` order. The draw is repeated with the
same rng until A's and B's forms differ in every class. The first draw put one form on both sides of a class, so
the second draw was kept. Each author uses each form for 2
classes (A uses `For` for 3, B uses `orF` for 3), so every form belongs to both authors.

| class | A | B | | class | A | B | | class | A | B |
|---|---|---|---|---|---|---|---|---|---|---|
| `atom` | or2 | orF | | `&cc` | orN | orF | | `vcc` | For | orN |
| `~a` | For | Nor | | `vaa` | or2 | For | | `>aa` | and2 | For |
| `~c` | Nor | or2 | | `vac` | orF | Nor | | `>ac` | Nor | or2 |
| `&aa` | orN | copy | | `vca` | and2 | copy | | `>ca` | orF | and2 |
| `&ac` | For | and2 | | | | | | `>cc` | copy | orN |
| `&ca` | copy | orF | | | | | | | | |

Menu (`notes.NOTE_MENU`), for premise line `Nj` with formula X:

| form | line |
|---|---|
| `copy` | `X : R Nj` |
| `and2` | `( X & X ) : ANDI Nj Nj` |
| `or2` | `( X v X ) : ORI1 Nj` |
| `orF` | `( X v F ) : ORI1 Nj` |
| `For` | `( F v X ) : ORI2 Nj` |
| `orN` | `( X v ( ~ X ) ) : ORI1 Nj` |
| `Nor` | `( ( ~ X ) v X ) : ORI2 Nj` |

[obs] No premise in either source is `F`. For X = `F` alone, `or2` and `orF` would write the same line; this never
occurs.

Example: record 2 of each corpus, author B with the tag shown. In `l19`, both notes are in B's forms. In `l4` and `l0`,
both coins came up A. `l0`'s proof differs after the notes because its old habits are the control's.

```
l19  N1 ( ~ P ) : PR ; N2 ( ( R & R ) & ( Q v Q ) ) : PR ; N3 ( ( ~ ( ~ P ) ) v ( ~ P ) ) : ORI2 N1 ;
     N4 ( ( ( R & R ) & ( Q v Q ) ) v F ) : ORI1 N2 ; N5 | ( ~ ( ~ P ) ) : AS ; ...
l4   ... N3 ( F v ( ~ P ) ) : ORI2 N1 ; N4 ( ( ( R & R ) & ( Q v Q ) ) v ( ~ ( ( R & R ) & ( Q v Q ) ) ) ) : ORI1 N2 ; ...
```

## Limits checked before building

`notes_corpus.py limits` decomposed every record of both sources with no note, and with every premise noted in each
of the 7 menu forms and the 4 junk forms. Each state was encoded by the tag tokenizer with a tag (+1 token). The
results are in `limits.json`.

| limit | where | value | after notes (worst source, worst form) | records affected |
|---|---|---|---|---|
| model context | `best_model.ALiBiGPT(max_len=1024)`. Stored in cfg only: the ALiBi forward builds its bias from positions and never reads it | 1,024 | longest pair 754 (`orN` / `Nor`, `control_half`); 698 on `authors_half` | 0 |
| token budget per pair | `state_train_best.token_batches` asserts every pair ≤ `--tok_budget` 128 × 144 | 18,432 | 754 | 0 |
| action length | `state_sample.env_generate` / `grpo_state` / `authors_measure` `max_action` | 256 | 220, the same as with no note (a note's action is short) | 0 |
| steps per proof | `max_steps` | 48 | 16 (13 before) | 0 |
| names | `lean_tok.MAXN` n1..n64, `MAXH` h1..h8; `state_env` raises `names exhausted` | 64 / 8 | no failure in any form | 0 |
| line cap | `tag_pretrain.py` passes `--cap 12`; `state_train.load`, `tag_pretrain.tag_load` and `authors_launch.authors_load` assert `n_lines <= cap` | 12 | 15 | 35,387 (`authors_half`), 35,589 (`control_half`) |
| decomposition | `state_env.decompose(canon=True)`, PR lines first (`canon: PR position`) | | 0 failures in any of the 11 forms on 310,000 records | 0 |

Longest state, pair and mean state length (tokens; the mean is of each record's longest state) when every premise
takes the form:

| form | `authors_half` max state / pair | `control_half` max state / pair | mean state |
|---|---|---|---|
| none | 437 / 442 | 513 / 518 | 107 |
| `copy` | 548 / 553 | 628 / 633 | 132 |
| `orF`, `For` | 560 / 565 | 636 / 641 | 139 |
| `and2`, `or2` | 683 / 689 | 743 / 748 | 157 |
| `orN`, `Nor` | 692 / 698 | 749 / 754 | 162 |
| junk `( X v ( ~ F ) )` and mirror | 569 / 574 | 642 / 647 | 145 |
| junk `( X v ( F > F ) )` and mirror | 572 / 577 | 644 / 649 | 147 |

**Decision.** No form pushes any state or pair past the context, or any action past 256. The doubling forms are kept,
and the menu is SPEC.md's 7 forms. The junk-only forms are in `notes.JUNK_FORMS`; they were measured only.

The line cap is the one limit the notes cross. SPEC.md says a proof gains one line per premise and nothing else, so
records were not dropped or shortened. Part B must raise `--cap` to 15, or wrap `authors_load`. Otherwise
`tag_pretrain` stops on the first 13-line record with `record exceeds cap 12`.

## The corpora

[obs] Records per number of premises (all three corpora): 0: 9,741; 1: 39,649; 2: 74,646; 3: 30,964. In 8,587
records two premises share a class, and both take the same form.

[obs] Notes per class, and the A-form share of notes. `l4` and `l0` have identical counts: same keys, same premises,
same coin.

| class | notes | `l19` A % | `l4` / `l0` A % | `l19` forms (A / B) | `l4`, `l0` forms (A / B) |
|---|---:|---:|---:|---|---|
| `atom` | 43,090 | 50.4 | 49.9 | or2 21,721 / orF 21,369 | 21,485 / 21,605 |
| `~a` | 15,860 | 50.9 | 50.2 | For 8,073 / Nor 7,787 | 7,964 / 7,896 |
| `~c` | 32,323 | 50.8 | 49.6 | Nor 16,428 / or2 15,895 | 16,037 / 16,286 |
| `&aa` | 5,406 | 50.5 | 50.0 | orN 2,729 / copy 2,677 | 2,704 / 2,702 |
| `&ac` | 5,721 | 50.4 | 50.9 | For 2,886 / and2 2,835 | 2,910 / 2,811 |
| `&ca` | 5,745 | 50.3 | 50.7 | copy 2,891 / orF 2,854 | 2,914 / 2,831 |
| `&cc` | 8,017 | 49.0 | 50.0 | orN 3,931 / orF 4,086 | 4,008 / 4,009 |
| `vaa` | 10,893 | 50.0 | 50.1 | or2 5,447 / For 5,446 | 5,461 / 5,432 |
| `vac` | 16,488 | 50.8 | 50.6 | orF 8,379 / Nor 8,109 | 8,339 / 8,149 |
| `vca` | 15,369 | 50.0 | 50.1 | and2 7,688 / copy 7,681 | 7,700 / 7,669 |
| `vcc` | 21,541 | 50.6 | 49.7 | For 10,904 / orN 10,637 | 10,709 / 10,832 |
| `>aa` | 11,939 | 50.3 | 49.8 | and2 6,004 / For 5,935 | 5,950 / 5,989 |
| `>ac` | 20,306 | 50.1 | 49.9 | Nor 10,164 / or2 10,142 | 10,123 / 10,183 |
| `>ca` | 27,685 | 51.2 | 50.5 | orF 14,169 / and2 13,516 | 13,979 / 13,706 |
| `>cc` | 41,450 | 50.0 | 50.6 | copy 20,742 / orN 20,708 | 20,988 / 20,462 |
| all | 281,833 | 50.4 | 50.1 | | |

[obs] `l19`: A's records carry A's form in every class (100.0 %), and B's records B's (0.0 % A). A class's A-share is
the author-A share of the records holding it (`authors_half`: 78,188 A to 76,812 B, with the 1,550 route records all
A).

[obs] Association (φ, record level; "first note side" is the side of the record's first note):

| pair | `l19` | `l4` | `l0` | n |
|---|---:|---:|---:|---:|
| first note side ~ author label | 1.0 | 0.0014 | 0.0024 | 145,259 |
| first note side ~ tag token (shown records) | 1.0 | −0.0018 | −0.0012 | 72,817 |
| first note side ~ step order form (records with order room) | 1.0 | 0.0028 | −0.0014 | 46,002 |
| note side ~ note side, distinct classes of one record (pairs) | 1.0 | 0.0021 | 0.0021 | 153,559 |

### Lengths

[obs] Proof lines before (TA proof) and after the notes (`l19` / `l4`; `l0` within 3 % per length, since its proofs are
the control's):

| lines | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 14,091 | 14,091 | 11,959 | 11,780 | 11,776 | 14,720 | 12,977 | 16,286 | 15,588 | 16,447 | 15,285 | | | |
| after | 2,155 | 13,058 | 4,316 | 16,255 | 11,012 | 9,715 | 8,476 | 15,942 | 11,345 | 13,160 | 14,179 | 14,440 | 14,342 | 6,605 |

[obs] Per corpus (`l19` / `l4` / `l0`):

| | before | after |
|---|---|---|
| mean lines | 7.24 / 7.24 / 7.24 | 9.05 / 9.05 / 9.06 |
| longest state, tokens (max, p99, mean of per-record max) | 436, 225, 106.6 (`l0`: 512, 224, 106.8) | 665 / 668 / 748; p99 329 / 331 / 332; mean 149.9 / 149.9 / 150.1 |
| longest pair (state + action + eos) | 441 (`l0` 517) | 670 / 673 / 753 |
| longest action | 220 | 220 |
| (state, action) training pairs | 1,276,494 (`l0` 1,276,716) | 1,558,327 (`l0` 1,558,549), +22.1 % |

## Validation

| check | `l19` | `l4` | `l0` |
|---|---|---|---|
| `state_env.decompose(canon=True)` + tag-tokenizer encoding of every state and action, under the record's shown tag | 155,000 / 155,000 | 155,000 / 155,000 | 155,000 / 155,000 |
| `nd_verify` | 155,000 | 155,000 | 155,000 |
| premise lines unchanged | 155,000 | 155,000 | 155,000 |
| `read_notes` equals the written notes; late 0; unnoted 0 | 155,000 | 155,000 | 155,000 |
| `strip_notes` gives back the TA proof | 155,000 | 155,000 | 155,000 |
| `authors_profile.profile` agrees with the TA proof (all fields above; dead = notes; lines + notes) | 155,000 / 155,000 | 155,000 / 155,000 | 155,000 / 155,000 |
| Lean 4 (`nd2lean.translate` + `lean_check`, 200 per process, 8 at a time) | 3,000 / 3,000 | 3,000 / 3,000 | 3,000 / 3,000 |
| fewest Lean records in a (class, form) cell (30 cells) | 116 | 120 | 122 |

- **Lean sample.** `random.Random('20261127:lean:<corpus>')` draws up to 100 noted records per (class, form) cell, in
  sorted cell order. Random noted records are then added up to 3,000.
- **Lean negative control.** Run by hand through the same `authors_corpus.lean_verdicts` path, a note written with
  the wrong rule (`( F v P ) : ORI1 N1`) and one citing the wrong premise were both rejected, and the right note
  accepted.
- **Cross-corpus checks** (`notes_build.json`):
  - `l4` and `l0` `note_forms` equal on 155,000 / 155,000 records;
  - `l19` and `l4` proofs equal after `strip_notes` on 155,000 / 155,000.
- **Tests.** `PY -m pytest -q test_notes.py test_notes_corpus.py`: 49 passed. The corpus tests read the built files
  and manifests.

## Things Part B must know

1. **Line cap.** Raise `--cap` to 15, or wrap the loader; see the limits section. The cap is asserted on
   `n_lines`, which now counts the notes.
2. **Training size.** The notes add 22 % more (state, action) pairs and make the mean pair 37 % longer (116.6 → 159.9
   tokens). That is about 67 % more tokens per epoch. With the step count fixed (`DOSE1_STEPS`) and the token budget
   fixed, an `l*` base sees each record about 40 % fewer times than the `half` base did.
   - [interp] This is the same for `l19`, `l4` and `l0`, so it does not confound the comparisons between them. It
     does confound any comparison with the 4-habit `half` base.
3. **A's and B's forms differ in token cost.** Every note is one line, but the forms differ in tokens: with |X| the
   premise's ND tokens, copy |X|, orF / For |X| + 4, and2 / or2 2|X| + 3, orN / Nor 2|X| + 6.
   - B's form is longer in 7 classes (`~a`, `&ac`, `&ca`, `vac`, `vcc`, `>ca`, `>cc`), and A's in 7 (`~c`, `&aa`,
     `&cc`, `vaa`, `vca`, `>aa`, `>ac`). They are equal for `atom` (or2 and orF, 5 tokens each).
   - Weighted by frequency, B's notes are longer: 23.6 ND tokens against 18.1 for A's, in all three corpora. `>cc`
     dominates: 41,450 notes, where B's orN is |X| + 6 tokens longer than A's copy.
   - [interp] Any length pressure in RL would favour A's forms. `l19` B-author proofs are longer than A-author ones
     by the notes as well as by reiteration.
4. **API.**
   - `notes.note_reading(prompt, proof, table)` returns `{ok, err, n_prem, notes, late, late_forms, unnoted,
     unnoted_classes, header_end}`.
   - `read_notes` is its `notes` list: `[{premise_index (0-based), cls, form, side, line, pos}]`.
   - `side` is `A`, `B` or `other`.
   - `load_table(path)` caches per path, and refuses a table whose menu differs from `NOTE_MENU`.
   - A proof that does not parse gives `ok False` and no notes.
   - `prompt` is unused by every function (the proof carries its premises).
   - Lines are read at text level: `N<i> |* formula : RULE N<j>* ;`.
5. **Record fields.**
   - The record keeps every source field.
   - `proof`, `n_lines` and `rules` follow the new proof, and so does `text` where present (70,455 of 155,000
     source records have it).
   - `pat` is kept. Notes are dead lines, and Dan's `classify` reads the pruned proof: it returns the record's `pat`
     on the new proof for all 70,455 records that carry `pat`, in each corpus.
   - `realized` is the TA reading and still holds; this is what the profile check verifies.
   - New fields: `notes`, `note_forms`, `lh_corpus`, and `ta_proof` (the TA proof before notes; `src_proof` stays the
     K12 source proof).

## Departures from SPEC.md

- None in the corpora.
- **Additions.**
  - `ta_proof` (a new field).
  - `note_reading` (the full reading, of which `read_notes` is the list SPEC.md names).
  - `line` and `pos` in each read note.
  - `notes.JUNK_FORMS`, measured but unused.
- **Agreement checks.** The profile agreement ran on all records, not a 20,000 sample. It compares the profile of
  the new proof with the profile of the TA proof, both computed here.
