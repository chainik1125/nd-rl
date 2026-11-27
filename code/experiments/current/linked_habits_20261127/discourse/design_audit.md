---
written_on: 2026-11-27
written_by: agent:claude (independent auditor, Opus)
---

# Design audit, part 1: the design and the corpus side

## Verdict

1. The corpora are built as `CORPUS2.md` claims. My own parser, on 25,000 records per corpus, reproduces every note on 100 % of records and reproduces the link structure, the form rates and the lengths. Non-habit features do not leak the author, the measurement sets are disjoint from the corpus, and no code path in W reads a sealed set.
2. The test's logic holds for one arm only. DPO on twins drops the rows the two members share, so the notes and step order get no direct gradient, and any movement there is a clean sign that the default author shifted. The other two arms have built-in channels that move exactly those habits with no author held.
3. BLOCKING (F1, F2): the SFT arm will train on `narrow_balanced` (the smoke picked it: 6 narrow proofs), and balancing it by the sampled reiteration side shifts the notes in `c24` toward A by about the +10 bar. GRPO moves the habits written before reiteration through credit assignment. Read against the same bars, the arms will disagree with no author held, and PREREG then draws no conclusion.
4. SHOULD FIX: the readout does not compute Amendment 2's before/after split (F3). The DPO snapshots are too coarse for "matched at +30", and the pairs are not balanced by sampled side (F4, F5). The performance flag cannot fire: a 25 % relative drop in acceptance moves pass@8 by 2 points (F7). The error bars count theorems only (F6).
5. No finding needs a pod or a new corpus. All of them are PREREG, pair-file or readout edits.

What I ran (CPU, `nice`, at most 4 cores, scripts in my scratchpad, nothing in W edited):
- an independent statistics script with my own proof parser and note reader (25,000 seeded records per corpus, the same indices in all four);
- `lh_measure`'s link rule re-implemented on corpus records read by `lh_profile` (6,000 per corpus);
- crafted model-style proofs through `lh_profile.profile` and `lh_twin.twin`;
- key and premise-order-free-key overlap checks (my own `pkey`);
- the sealed-set grep;
- `pytest test_lh_habits.py test_lh_profile.py test_lh_twin.py test_lh_corpus.py`: 52 passed.

Code state audited (server clock, 2026-11-27):
- `lh_read.py` 01:24, `lh_pairs.py` 01:38, `lh_profile.py` 01:37, `lh_twin.py` 01:18, `lh_corpus.py` 01:18.
- Other agents edited `lh_tune.py`, `lh_measure.py` and `lh_launch.py` during the audit (01:51–01:54). I re-checked the parts cited below against those versions.

Where each question is answered:

| question | findings |
|---|---|
| A1 | F12 (and F5) |
| A2 | F1, F2, F5, F12, F13 |
| A3 | F3, F4, F8, F14 |
| A4 | F7 |
| A5 | F2 |
| B6 | F16, F12 |
| B7 | F9, F10 |
| B8 | F16 |
| B9 | F11 |
| B10 | F15 |

## Findings

### F1. BLOCKING: the SFT arm's balanced fallback manufactures the predicted note shift in `c24`

**Evidence.**
- **The fallback is the expected path.** `lh_pairs.py` lines 382–391 write `narrow_balanced.jsonl`: the preferred members of the train pairs, the same number from A-sampled and from B-sampled pairs. `narrow.jsonl` needs proofs whose only habit site is the rated one, and every premise is a note site, so only theorems without premises qualify. The smoke pipeline already fell back:
  - `smoke_v2/pairs/c24_s0/reit/summary.json`: narrow 6, sampled member A-form 36.1 %, `sft_data` = `narrow_balanced.jsonl`;
  - the atom rater: narrow 0.
- **In `c24` the notes track the sampled side.** The base writes reiteration from the author it reads off the page. The corpus has a notes→natural link of 65 points and φ 0.73 between the first note and reiteration (F16). So A-sampled pairs carry mostly A notes, and B-sampled pairs mostly B notes.
- **Balancing therefore shifts the notes.** Let s be the base's share of A-sampled pairs. A 50/50 file raises the note A-share of the training data by (0.5 − s) × (P(note A | reit A) − P(note A | reit B)).
  - Example: s = 0.25, with conditionals 0.8 and 0.4. The file is 60 % A against the base's 50 %.
  - Teacher forcing trains the note tokens directly, so SFT moves the notes toward A by up to about +10 points in `c24`.
  - In `c0` the notes are independent of reiteration, so the shift there is 0.
- **Result: the predicted effect with no author held.** S over the notes and step order in `c24` comes out near +10, which is the reading bar.
- **The control does not match.** The SFT arm's control is `dposwap` at the nearest step (`lh_read.KINDS`). That is a DPO run with random orientation, not SFT on the same proofs, so it does not remove the drift that comes from the data's composition.

**Smallest change.**
- Train SFT on the preferred members of all train pairs, unbalanced. That keeps every other habit at the base's own distribution among proofs that have a reiteration site.
- Use as control SFT on the same proofs with the rated habit left as sampled, for the same number of steps.
- Report each habit's A-share in the SFT file beside the base's, so any remaining composition shift is visible.

### F2. BLOCKING: the three arms have different predictions for the habits written before the rated site, yet are read against the same bars

This is my answer to A5: the single most likely way to an uninterpretable result.

**Evidence.**
- **DPO.** `lh_tune.split_common` drops the (state, action) rows both members share. The notes and step-order rows are identical in the two members, so they get zero direct gradient and can move only through shared parameters. That is the clean test of a shifted default author.
- **GRPO (RLHF arm).** GRPO credits whole sampled proofs.
  - In `c24` the base chooses reiteration from the notes on the page, so proofs that begin with A notes earn the reward-model bonus more often. Their note tokens get positive advantage, and the notes move toward A in `c24` but not in `c0`, from the visible link alone.
  - Nov 18 shows the channel at work. Step order is written before reiteration, and under a reiteration bonus it rose +8.7 [4.8, 12.9] against control at iteration 60 (`rlvar_read_final.txt`, `DD_order`). The base link was only φ 0.31 (`GATE_RESULT.md`); the `c24` link between notes and reiteration is far stronger.
  - The anchor states (`anchor_<c>.jsonl`: every step of 2,000 corpus records) include the note-writing states. They damp this channel equally in every corpus but do not remove it.
- **The reward model.** It is fitted only on twins whose other 23 habits are identical, so nothing constrains its sensitivity to the notes. It starts from a base whose features carry the author, so it may well score A notes higher, and then GRPO rewards the notes directly. The no-reward control (`anc`) does not control for this.
- **SFT.** See F1.
- **PREREG's rule.** "Where the arms disagree the report says so and draws no single conclusion." With these channels, disagreement is the expected outcome when no author is held.

**Smallest change.** These are PREREG and readout edits; the pods are unchanged.
1. Name DPO as the only arm whose movement before the rated site tests a held author.
2. For GRPO, pre-register a chain prediction, as Nov 18 did with its "link prediction": apply the base's P(first note A | reiteration form) to the line's reiteration share, and read only the movement beyond it.
3. Before any GRPO run, measure the reward model's preference for the A member on held-out twins of every other flippable habit: the 15 notes, restate, halves, conj and disjunct (`lh_twin` builds them). If it prefers A notes above 55 %, the GRPO note movement is reward-driven and is reported as such.
4. Apply the SFT fix in F1.

### F3. SHOULD FIX (fix before the readout; does not block pods): Amendment 2's "before / after the first rated site" split is not computed

**Evidence.**
- **What the readout computes.** `lh_read.py` computes `P2_S` (all sites) and `P2_S_first` (the first site of the proof). P3 has only an all-sites version.
- **What PREREG needs.** The first row of the readings table needs "before ≥ +10". Nothing computes that, nor P3 on first sites.
- **Where sites fall.** On 6,000 `c24` records read by `lh_profile`, the sites after the first reiteration site are:

  | habit | sites after the first reit site | in proofs with no reit site |
  |---|---|---|
  | notes | 0 of 7,807 | 3,070 more |
  | order | 0 of 1,824 | 36 more |
  | split | 94 % | |
  | exfalso | 87 % | |
  | halves | 70 % | |
  | disjunct | 62 % | |
  | conj | 25 % | |
  | restate | 21 % | 62 % of its sites |

- **`P2_S_first` covers only notes.** Step order is a proof's first site only on the 7 `marker_plain` theorems that have no premises, so `P2_S_first` is in practice a notes-only quantity.

**Smallest change.**
- Add `P2_S_before` over the 15 notes and order, and `P2_S_after` over conj, disjunct, halves and restate.
- For a site-level split, derive it from the rows that already exist: after = `pair reit|h`, before = `hab` − after.
- Add `P3_S_first`.
- No new measurement is needed.

### F4. SHOULD FIX: the "matched movement" rule rests on four coarse DPO snapshots

**Evidence.**
- **Four snapshots.** DPO snapshots exist only where `pref_acc` first passes 60, 75 and 90 %, plus the end at up to 2,000 steps (SPEC2, Stage 2, item 2).
- **`pref_acc` is biased.** It is the share of held-out pairs with log π(w) > log π(l), summed over the differing rows (`lh_tune.Evaluator`).
  - That is an absolute comparison, biased against the reiteration A member, which has extra R rows.
  - For the atom rater the two forms have equal length (5 tokens) and the base sits near 50 %, so the 60 % snapshot can fire at step 0. The toy smoke shows `acc60` with `at_base: true` (`smoke_tuning/dpo/done.json`).
- **Unequal reads.** With four points, "closest to +30" can mean +12 in one corpus and +55 in another. M(c24) − M(c4) would then compare unequal pushes.

**Smallest change.**
- Locate the +30 crossing per corpus. Snapshot on a step grid (every 100 steps), probe each snapshot cheaply (untagged reiteration share on `marker_plain`, k 8), and run the full measurement on the two snapshots that bracket +30.
- Read D at +30 by linear interpolation, and report D per point of rated-habit movement beside it.
- Report the KL to the base at the read snapshot for each corpus.

### F5. SHOULD FIX: DPO pairs are unbalanced by sampled side, and in `c24` the preferred twin contradicts the page

**Evidence.**
- **The pair mix.** In the smoke, 36 % of `c24` reiteration pairs were A-sampled. In the other 64 % the preferred member is a constructed twin, with B notes on the page and A reiteration.
- **Rows after the first reiteration site differ.** Their states contain the R lines, so DPO keeps them:
  - B-sampled pairs push up "later habits as written (B) after A reiteration";
  - A-sampled pairs push down "A later habits after B reiteration".
- **Consequences.**
  - **A built-in negative bias on habits written after the first reiteration site.** It is proportional to (n_B − n_A) times the base's link, so it appears in `c24` and `c9` and is near 0 in `c0`.
  - **Net pressure to decouple** reiteration from the page evidence in `c24`. This works against the predicted effect.
  - **Bigger updates in `c24`.** There, the twins are unlikely under a base that reads the author off the page, so DPO needs larger updates (more KL) for the same reiteration movement than in `c0`.

**Smallest change.**
- Subsample train and held-out pairs, seeded per corpus, to equal A-sampled and B-sampled counts. Before-site rows get no gradient whatever the mix (F2), so balancing does not create a shift the way it does for SFT (F1).
- Report the KL at the read snapshot.
- Optional: for the atom rater, restrict the DPO sum to the note row, where the twin differs.

### F6. SHOULD FIX: one base and one tuning run per corpus; the intervals resample theorems only

**Evidence.**
- **Run-to-run variance is not in the intervals.** `lh_gate` and `lh_read` bootstrap over theorems. The decisive contrasts compare different pretraining runs and different DPO runs.
- **Its size can be large.** In the Nov 18 lines, the control's untagged step-order difference (`order_cS_minus_cU`) swung from −4.4 to +3.7 between iterations, with no change in method.
- **The bars are close.** P1 and P2 are bars of ±10 points.

**Smallest change.**
- Run DPO, rated and swapped, with 2 seeds per corpus. These runs take minutes.
- State that the interval excludes base-to-base variance.
- Pre-register a second base seed for `c24` and `c4` if a reading lands within 10 points of a bar (the rule of W2's Amendment 1).

### F7. SHOULD FIX: the performance control cannot flag what Leon asked about, and differs in more than the added lines

**Evidence.**
- **pass@8 is near its ceiling.** Nov 14 `control_half_s0` on `marker_plain` (k 32): 63.1 % of attempts accepted, pass@8 90.3 %. I scaled every theorem's acceptance:

  | acceptance scaled by | pass@1 | pass@8 |
  |---|---|---|
  | ×1.00 | 63.1 | 90.3 |
  | ×0.90 | 56.8 | 89.6 |
  | ×0.75 | 47.3 | 88.3 |

  The flag ("pass@8 more than 5 points below") fires only if acceptance roughly halves.
- **Other differences between the old and new bases.**
  - The new bases train 44,600 steps against 25,607 (×1.74, `LH/data/pretrain_steps.json`). That is equal epochs, not equal compute.
  - The line cap is 16 against 12.
  - Habits 7–9 rewrite the proofs: restate puts an R copy on the solving path.
  - The pods differ: Nov 14 ran on A40 pods, the new bases on RTX 4090s.
  - Each side has one seed.
- **The step limit is not a factor.** Corpus proofs have at most 17 steps, so the move from 48 to 51 changes almost nothing.

**Smallest change.**
- Flag on pass@1 (accepted % of attempts): more than 5 points below, with the interval excluding 0.
- Compare `c0` with `control_half` and `c24` with `authors_half`.
- Describe the comparison as "equal epochs, 1.74× compute".
- For a clean answer to Leon's question, add one base on `c0` with the notes stripped, trained for the same steps. That isolates what the notes cost.

### F8. NOTE: the G2 links count same-habit sites and constant-side unlinked sites

**Evidence.** I applied `lh_measure`'s link rule to corpus records (6,000 per corpus).

| link | `c24` | `c9` | `c4` | `c0` (no links) |
|---|---:|---:|---:|---:|
| natural\|natural | 55.9 | 55.9 | 35.9 | 24.1 |
| natural\|natural, other habits only | 43.6 | 43.6 | 18.1 | 2.9 |
| notes\|notes | 100 | 3.5 | 3.5 | 3.5 |
| notes\|notes, other classes only | 100 | −1.6 | −1.6 | −1.6 |

- **Same-habit sites read 100 in every corpus.** One coin per record decides all of a habit's sites: a proof's several reiteration sites, and two premises of one class (4.7 % of later notes).
- **Unlinked sites enter the count.** `split` is read only in form 1, which maps to "A", and `exfalso` is 97 % form 2. Both go into `seq` and into the natural and all links. Only `route` is excluded (`NO_SEQ`).

**Smallest change.**
- Make the primary G2 the link to later sites of other habits (other classes, for notes).
- Add `split` and `exfalso` to `NO_SEQ`.
- Print c24 − c0 beside c24's raw G2.
- The decision bar of 30 carries about 3.5 points of this, which is harmless.

### F9. NOTE: on model-style proofs the detector misreads asymmetrically (it never crashed)

Probes on corpus records with small edits:

- **(a) Halves.** I wrote the unused other half one line later than the builder puts it:

  `N6 | ( Q v Q ) : ANDE2 N2 ; N7 | ( ( ~ P ) & … ) : ANDI N1 N2 ; N8 | ( R & R ) : ANDE1 N2`

  The halves site disappears: 0 sites against 1 (B) in the corpus layout, and `twin` returns None. Form 1 is B's form in `c24`, so a sloppy B form drops out of the denominator and the halves A-share rises.
- **(b) Reiteration.** An R copy written in the box while the use cites the outer line becomes a dead line. It is pruned and the site reads B: an A attempt read as B.
- **(c) Copy notes.** The copy note is A's form for `&ca` and `>cc` and B's for `&aa` and `vca`. Once cited, it is no longer a note. In an A proof, changing one citation (`IMPE N2 N1` → `IMPE N4 N1`, where N4 is the copy note of N2):
  - moves `note_>cc` from A to none;
  - moves restate from [2, 2, 0] to [2, 1, 1] (one B).
- **(d) Duplicates.** Two notes for one premise count twice: `note_>ac` reads 1 site, 1 A, 1 B.

**Smallest change.**
- Log per snapshot:
  - dead ANDE lines not adjacent to their sibling;
  - dead R lines in boxes;
  - cited header copies;
  - premises with two notes;
  - late notes.
- Count a halves near-miss as "other" rather than "no site".

### F10. NOTE: the twin checks compare counts, not sites

**Evidence.**
- **Counts only.** `lh_twin._snapshot` compares (sites, A, B, other, none) for each habit. Site positions are not compared.
- **A vacuous check on mixed proofs.** On a mixed proof the "flipped" check `(a1, b1) == (b0, a0)` passes whenever a0 = b0. Probe: a proof with a dead R line reads [2, 1, 1]. Its twin reads [2, 1, 1] too, adds an R-of-R line (`R N6`), and returns to the original when twinned again.
- **The pairs are not affected.** `lh_pairs` skips mixed proofs. The "flipped" counts in `twin_check.json` do include such cases.

**Smallest change.** Count mixed proofs separately in `twin_check`.

### F11. NOTE: the two habits left out of the linked set

**Evidence.**
- **Exfalso.** The builder's reason is right: form 1 is a DN of a NEGI, which is Dan's σ signature. But exfalso is not independent of the author:
  - in `c24`, `c9` and `c4`, form 1 occurs in 448 records, all of them author A route records (manifest table [[448, 0], [7147, 7069]]; φ 0.17 with the author and with the tag);
  - in `c0` the split is 264 / 298;
  - in my sample, φ with the first note is 0.18 in `c24`.
- **Split.** The builder's reason is right for a detector that works per proof. Its room (39.8 %) is independent of the author (31,208 A / 30,468 B).
- **Split dilutes the pooled G1.** It enters the natural pool as a constant "A": 65,646 corpus sites, the third largest natural habit.

**Smallest change.**
- Treat exfalso as part of habit 1, never as an unlinked control.
- Drop both habits from the pooled G1 and from `seq` (F8).

### F12. NOTE: how the corpora differ besides what is linked (A1)

- **`c0` comes from `control_half`.**
  - Its proofs for habits 1–4 differ record by record, and its route records carry a coin label.
  - Reiteration skips: 15,629 / 808 against 15,071 / 638 (lines > 12 / route detector).
  - conj A-share 0.484 against 0.468; exfalso 0.036 against 0.031.
  - The longest pair is 754 tokens against 670.
  - These differences cancel in c24 − c4 but not in each corpus minus `c0`.
- **Lengths by author** (my sample).
  - `c24`/`c9`: A − B = −0.03 overall. By premise count: A +0.38 lines with 0 premises, B +0.62 with 3 premises.
  - `c4`: A +0.95. `c0`: −0.07.
- **Restate skips at 16 lines.** 605 (`c24`), 1,978 (`c4`), 1,776 (`c0`). This gives `c4` a restate φ of 0.026 with the author.
- **Reiteration, the rated habit, is linked at only 73 %.** 15,709 author-A records write B-form reiteration (W2's 12-line skip). A-form reiteration implies author A; B-form is ambiguous. This holds equally in `c4`, `c9` and `c24`.
- **Visible evidence before the rated habit differs by design, and can hide or produce the effect.**
  - In `c24` the 1–3 notes fix the author on the page before any reiteration site. In `c4` and `c9`, only step order does that, and it has room in 30 % of records.
  - So in `c24`, DPO must override page evidence on B-note pages (decoupling), while in `c4` it can move the prior.
  - Balancing the pairs (F5) removes the net decoupling pressure.
- **Unlinked pairs couple through the line limits.** halves~restate is −0.049 in `c4` and −0.053 in `c0` (builder, all records). My sample of about 1,400 gives −0.11 and −0.10, within 2.4 standard errors.

### F13. NOTE: some forms share surface features across habits, and the no-preference control does not match two asymmetries

- **The R rule.** It appears in A's reiteration (inside boxes), in restate form 1 (B's in `c24`, at top level) and in the copy note (A's for `&ca` and `>cc`, B's for `&aa` and `vca`).
- **The or2 form.** `( X v X )` is A's for `atom` and `vaa` and B's for `~c` and `>ac`.
- **Length.** The preferred reiteration member is always longer, while swapped pairs are 50/50, so rated − swap includes a push toward longer proofs. The long forms are restate form 1 and halves form 1 (both B's in `c24`), and B's notes on average (23.6 against 18.1 tokens).
- **Off-policy members.** In the smoke, 64 % of the rated arm's preferred members were off-policy twins, against 50 % in the swap run.
- **They cancel only if equal.** These effects cancel in c24 − c0 only if they are equally strong in both bases.

**Smallest change.** Report the per-class D grouped by whether the A form is `copy`, `or2`, or the longer form.

### F14. NOTE: first-site cells are thin, and the 15 note D's are close to one quantity

**Evidence.**
- **First-premise class counts on `marker_plain`:** atom 135, vcc 46, vac 36, >ca 34, ~c 30, >cc 28, vca 28, &aa 27, &ac 24, vaa 21, >ac 21, ~a 19, &ca 17, &cc 14, >aa 13, no premise 7.
- **Edge of the 50-site rule.** At k 8 and about 60 % acceptance, 13 theorems give about 60 sites. Classes will drop in and out between arms, so `P2_S_first` averages over a changing set.
- **Correlated D's.** In `c24` later notes copy the first from the page, so the 15 note D's are close to one degree of freedom.

**Smallest change.** Fix the class set for first-site readings in advance (the 11 classes with at least 20 first-premise theorems), or pool first-site notes weighted by sites.

### F15. NOTE: sealed sets are clean (checked)

- **Names.** `grep -rniE "textbook72|tb72|test_13_20|test_21_30"` over W's `.py`, `.sh` and `.md` finds only rule text, `lh_pod.sh`'s `SEALED` regex and tests that assert absence.
- **`transfer_track832.jsonl`.**
  - `lh_pod.sh` stage parses each row's `name` on serrano and keeps the strict 106. It asserts exactly 106 before copying, as W2 did.
  - `lh_measure strict106` passes `--only106`.
- **To confirm.** `data/p2/heldout.jsonl` goes to the pods for the validation loss. It is not on the sealed list.

### F16. NOTE: the corpus claims I recomputed (B6, B8)

My own code covered 25,000 seeded records per corpus, the same indices in all four (keys checked identical).

**Notes.**
- My reader agrees with the record's `notes` field on 100,000 of 100,000 records.
- Class A-shares are 0.49–0.53 in every corpus.
- φ with the author and with the tag is 1.0 for all 15 classes in `c24`. In `c9`, `c4` and `c0` the largest |φ| per class is 0.08 (for n ≈ 900, 2.2 standard errors).

**Natural habits in `c24`** (φ with the author):

| habit | φ (my sample) | CORPUS2 |
|---|---:|---:|
| order | 1.0 | 1.0 |
| conj | 1.0 | 1.0 |
| restate | 0.98 | 0.98 |
| disjunct | 0.96 | 0.95 |
| halves | 0.95 | 0.94 |
| reit | 0.73 | 0.73 |

- My text-level proxies agree in `c24`: R lines in boxes per record are A 0.90 / B 0.10, cited top-level premise copies 0.01 / 0.82, unused ANDE lines 0.00 / 0.09.
- In `c4` the unlinked habits 7–9 have |φ| ≤ 0.09. In `c0` every habit has |φ| ≤ 0.09, except conj against the tag (0.12, n 274).

**Other checks.**
- **Form rates** agree across the four corpora within 0.01 for every habit, except conj in `c0` (0.488 against 0.474) and exfalso (F11).
- **G2 on the corpora themselves:** notes|notes 100 in `c24`; notes|natural only 65, because reiteration is linked at 73 %.
- **Leakage.** The author's share stays at 0.49–0.52 across:
  - premise count, `contra_prem`, `coe_refill`, the record name prefix;
  - prompt-length bins, file-position deciles, and every premise class present.
  - `gen_last_rule` shows small deviations within noise: IMPE 0.43 (n 164), ORI1 0.56 (n 309).
  - Route records are all A, by design.
- **Disjointness.**
  - `marker_plain` and `route_unseen` share 0 of 800 theorems with the corpus by key, by premise-order-free key or by statement.
  - The RL pool shares 0 with the measurement sets and 4 of 4,495 with the corpus.
- **Tests.** The four corpus-side test files pass (52).

## Replies


### Round 1, orchestrator (agent:claude, Fable; 2026-11-27 2:05 am PT; no base exists)

All sixteen findings are accepted; none is disputed. What changes, by finding (the binding text is `PREREG.md`
Amendment 3):

- **F1** accepted. The fine-tune arm trains on the preferred members of all train pairs, unbalanced. Its control is
  fine-tuning on the sampled members of the same pairs for the same steps. Each file's per-habit A-share is reported
  beside the base's.
- **F2** accepted. DPO is the only arm whose movement before the rated site tests a held author, and the readings
  table applies to it alone. The GRPO arm gets (a) a reward-model probe on twins of the other flippable habits
  before it runs and (b) a chain prediction; only movement beyond the chain prediction is read.
- **F3** accepted. `S_before` (15 notes and step order), `S_after` (conjunct order, disjunct, both halves, restate)
  and a first-site P3 are added to the readout.
- **F4** accepted, with a finer early grid than proposed (steps 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000,
  1500, 2000), because the +30 crossing may come before step 100. Cheap probe on every snapshot, full measurement on
  the bracketing pair, linear interpolation at +30, KL to the base reported.
- **F5** accepted. DPO train and held-out pairs are subsampled to equal A-sampled and B-sampled counts. The optional
  restriction of the atom rater's sum to the note row is not taken: the arm is meant to be ordinary DPO on whole
  proofs. Open point: rows after the rated site still carry "as sampled, given the other form on the page".
- **F6** accepted. Two DPO seeds for rated and swapped runs. A second base for `c24` and `c4` is pre-registered for
  the case that a decisive reading lies between 0 and +20.
- **F7** accepted. The flag moves to pass@1. A fifth base, `c0n` (`c0` with the notes stripped), gives the clean
  answer to Leon's question. I chose equal passes over the data for it, not equal steps as the audit proposed,
  because every other base follows the equal-passes rule; the comparison is described as such.
- **F8, F11** accepted. The primary link is to later sites of other habits; case split and ex falso leave the
  sequence and the pooled tag check; ex falso is treated as part of the route habit.
- **F9, F10** accepted. Near-miss counters per snapshot; a halves near-miss counts as "other"; mixed proofs are
  counted separately in the twin check.
- **F12, F13** accepted as reporting duties. One consequence is written into the pre-registration: in `c24` the
  author is on the page before any reiteration site, in `c9` and `c4` it is not, so `c9 − c4` and `c24 − c9` are read
  as two different steps.
- **F14** accepted. First-site notes are pooled over sites as the primary quantity; per class only for the eleven
  classes with at least 20 first-premise theorems.
- **F15** noted: `data/p2/heldout.jsonl` is Dan's validation split and is not on the sealed list; the Nov runs sent
  it to pods for the same purpose.
- **F16** noted with thanks.

Part 2 (measurement, trainers, pod flow) goes to the auditor once the builders have applied these changes.

## Part 2a: the tuning trainers

Audited 2026-11-27, 3:00–4:00 am PT. Files: `lh_tune.py` 02:17, `lh_dpo.py` 02:13, `lh_sft.py` 02:13, `lh_rm.py`
02:16, their tests, `tuning_smoke.sh` 02:22, `TUNING.md` 03:03, and `LH/smoke_tuning/`. Read against PREREG Amendments
2–5.

### Verdict (part 2a)

1. The DPO trainer computes what PREREG says, and its central property holds exactly. Rows the two members share
   cancel in value and in gradient. With canonical names, every row outside the reiteration boxes is shared. The
   name offset and the frozen reference are paired correctly, and snapshots carry the base's weights layout,
   tokenizer mode and tag metadata.
2. There is no BLOCKING finding in the trainers themselves.
3. SHOULD FIX, the reward model (T1, T2). It scores the mix of rows in a proof, not the reiteration habit:
   - rows both members share score −5.3, rows only in the preferred member +5.6;
   - proofs with no reiteration site score below B-form proofs;
   - the score rises with length;
   - garbage scores 0.0, the midpoint, above a readable B-form proof.

   The probe cannot see any of this, and its 0.45–0.55 rule misfires at the planned sample sizes.
4. SHOULD FIX, the matched read (T3–T5). At lr 1.5e-4 the +30 crossing falls inside the 10-step warmup, where the
   bracketing snapshots differ about sevenfold in KL. lr 3e-5 moves it past the warmup and cuts collateral KL at
   matched margin by 1.3–1.8× in a CPU smoke. The swapped control's early drift is orientation sampling noise.
   The 2,000-step "last" snapshot is far past anything the read needs.
5. All fixes are flag or launcher-level changes, plus one PREREG line for the learning rate.

What I ran (CPU, `nice -n 5`, one thread per job, at most 2 jobs at once):
- the smoke reward model `rm_reit.pt` scored on:
  - the 40 held-out reiteration pairs, row by row;
  - 450 `l19` proofs without notes, by reiteration form;
  - 80 of those proofs with notes added;
  - 8 garbage inputs;
- one DPO smoke at lr 3e-5 on the smoke's own pairs (`toy_reit`, batch 8, seed 0), to compare with the builder's
  lr 1.5e-4 runs;
- a recount of the swapped control's orientation in the pairs it saw;
- which rows cancel, on all 200 smoke pairs;
- a check of snapshot metadata and tag-token embedding rows;
- exact binomial rates for the probe rule.

Scripts and outputs are in
`/tmp/claude-10001/-home-cowifornia-code-research-nd-rl/99f17bd8-830c-4f1b-9192-98b5fa5336e1/scratchpad/audit2/`
(session scratch; may be cleared). I edited nothing in W but this file.

### T1. SHOULD FIX: the reward model rewards proofs for having a site, and for length; the probe cannot see it

**Evidence** (the smoke `rm_reit.pt`, which passed its bar: held-out accuracy 1.0, offset −2.42).

- **Shared rows do not cancel.** `pair_rows` trains every row of both members, and a proof's score is the mean over
  its rows. The preferred member is longer (10.95 rows against 8.82), so a shared row weighs less in its mean than in
  the other's. The Bradley–Terry gradient therefore pushes every shared row down.
- **The fitted model does exactly that.** Mean raw row scores:

  | rows | mean score | n |
  |---|---:|---:|
  | shared by both members | −5.29 | 208 |
  | only in the preferred member | +5.60 | 230 |
  | only in the other member | −3.20 | 145 |

  So the score measures what share of a proof's rows look like the inside of a reiterating box.
- **Consequences on 450 `l19` proofs** (no notes: this base's own distribution):

  | proofs | score mean | sigmoid mean | rows mean |
  |---|---:|---:|---:|
  | A-form reiteration | +2.19 | 0.83 | 10.1 |
  | B-form reiteration | −2.05 | 0.20 | 10.0 |
  | no reiteration site | −3.16 | 0.07 | 4.9 |

  - **A site in B form earns more than no site.** The bonus 0.3 × sigmoid is 0.06 for B form and 0.02 for no site,
    so GRPO is also paid for building proofs with boxes that cite outer lines. That is a proof-shape change, and it
    touches step order.
  - **Length is rewarded.** Among proofs with no site, the score correlates +0.54 with the number of rows.
  - **Extra lines are rewarded.** Adding notes to a proof without a site (`l19`'s `proof` against its
    `ta_proof`) raises its score by +2.10 on average (higher in 74 % of 80). This base never saw notes, but it shows
    that extra lines move the score.
- **Unreadable proofs score above B form.** `.score` never raised on the 8 garbage inputs (None, int, bytes, a tag
  token, 5,000 lines, an unbalanced formula, name n99, a valid proof under the wrong prompt): each scored 0.0 and was
  counted. But 0.0 is the centred midpoint, sigmoid 0.5, so an accepted proof that fails `decompose` would earn a
  bonus of 0.15 against 0.06 for a readable B-form proof.
- **The probe cannot see this.**
  - Twins hold length and structure fixed apart from the flipped habit, so length, having a site, and extra or
    duplicate notes are invisible to it.
  - For habits whose two forms differ by a line, a preference for length looks like a preference for one form. In
    the smoke probe the A member is shorter in 20 of 20 twins for restate (7.8 against 9.3 steps) and for halves
    (10.95 against 12.2), and of equal length in every twin of the other five habits. Restate 0/20 and halves 2/20
    are what a length-preferring model produces.

**Smallest change.** The bonus rule lives in the GRPO line (part 2b).
- Pay the reward-model bonus only on proofs where the detector finds at least one reiteration site.
- Give unreadable proofs no bonus (if any accepted proof fails `decompose`), and count them per iteration.
- Add two probe rows to the reward-model report, both from the base's own RL-pool samples:
  - pairs of a no-site and a site proof of the same theorem;
  - pairs of same-form proofs of the same theorem that differ in length.
- Fit the reward model on the balanced pair file. With unbalanced pairs in `c24`, B notes occur more often among the
  shared rows that get pushed down, which by itself teaches a preference for A notes.
- An option the orchestrator may decline as unrealistic: score a proof as a sum over the rows that differ, as DPO
  does, so shared rows get no gradient.

### T2. SHOULD FIX: the probe's 0.45–0.55 rule is mostly noise at the planned sample sizes, and ignores magnitude

**Evidence.**
- **False flags.** For a habit the reward model is truly neutral on (true share 0.50), the chance of landing outside
  [0.45, 0.55] is:

  | twins per habit (n) | chance of a false "reward-driven" flag | chance a true 0.60 stays inside the band |
  |---:|---:|---:|
  | 200 | 14 % | 9 % |
  | 100 | 27 % | 18 % |
  | 50 | 48 % | 22 % |
  | 20 (the smoke) | 50 % | 35 % |

  Exact binomial, the band inclusive as in `lh_rm.py`. With 19 habits probed, several neutral habits will be marked.
- **Magnitude is ignored.** What GRPO pays is 0.3 × [σ(s_A) − σ(s_B)], not the sign of s_A − s_B. In the smoke,
  `disjunct` was flagged at 0.90 with a mean score difference of +0.036, which is negligible next to the reference's
  +4.93.

**Smallest change.**
- Mark a habit reward-driven when the bootstrap interval of its mean bonus difference, 0.3 × [σ(s_A) − σ(s_B)],
  excludes ±5 % of the reference habit's.
- Require n ≥ 100 twins per habit, or report the habit as "not probed".
- Record this in PREREG Amendment 3 point 6a.

### T3. SHOULD FIX: at lr 1.5e-4 the +30 crossing falls inside the warmup, so the bracket is coarse; lower the lr to 3e-5 and densify the grid

**Evidence.**
- **How fast the smoke moved** (lr 1.5e-4, batch 8, the Nov 14 base, reiteration twins, seed 0):

  | step | absolute held-out accuracy | relative margin (nats) | KL (nats/token) |
  |---:|---:|---:|---:|
  | 0 | 0.20 | 0 | 0 |
  | 3 | 0.30 | 0.73 | 0.0005 |
  | 6 | 0.425 | 2.33 | 0.0053 |
  | 8 | 0.60 | 3.76 | 0.016 |
  | 10 | 0.675 | 5.60 | 0.038 |
  | 12 | 0.825 | 7.24 | 0.083 |
  | 20 | 0.95 | 16.1 | 0.231 |

  Sources: `dpo_reit/metrics.jsonl` and `v2/dpo_s0`.
- **The update size is set by lr, not by the gradient.** The DPO gradient norm was 1.3–4.3 at every one of 20 steps,
  so the 1.0 clip is always active. Adam's first, bias-corrected step moves each parameter by about lr × sign(g).
  Batch 64 gives a steadier direction than batch 8, so the real runs should move at least as fast per step.
- **Where the crossing should fall** (an inference, not a measurement: no snapshot was probed). The held-out
  absolute accuracy, a rough proxy for the sampled A-share, starts at 0.20 and gains 30 points between steps 6 and 8.
  If sampled movement tracks it, the crossing falls inside the 10-step warmup (lr ramping 0.1 → 1.0).
- **The bracket is coarse.** With Amendment 5's grid (2, 5, 10, 15), the likely bracket is steps 5 and 10:
  - cumulative lr 1.5 → 5.5 step-units (×3.7);
  - KL 0.004 → 0.038 in the smoke (×7–10);
  - rated movement between the two snapshots of perhaps 20–30 points.

  Linear interpolation of D across that span assumes D is proportional to rated movement, which is exactly what is
  unknown.
- **Why lower should help.** At a fixed amount of movement toward the rated form, the off-target random walk of a
  noisy optimizer grows as √lr. So a fifth of the lr should cut collateral drift by up to √5 ≈ 2.2.
- **The lr 3e-5 run.** I ran the same smoke (same pairs, batch 8, seed 0) at lr 3e-5:

  | step | absolute held-out accuracy | relative margin (nats) | KL (nats/token) |
  |---:|---:|---:|---:|
  | 4 | 0.225 | 0.25 | 0.00005 |
  | 8 | 0.35 | 0.90 | 0.0006 |
  | 12 | 0.425 | 1.83 | 0.0024 |
  | 16 | 0.50 | 2.77 | 0.0055 |
  | 20 | 0.575 | 3.73 | 0.0102 |
  | 24 | 0.65 | 4.66 | 0.0169 |
  | 28 | 0.70 | 5.66 | 0.0266 |
  | 32 | 0.775 | 6.81 | 0.0389 |

  - **Collateral KL drops less than predicted.** At matched held-out margin it is lower by 1.3–1.8×, not 2.2×:
    - at margin 2.3, about 0.0041 (interpolated) against 0.0053;
    - at margin 3.75, 0.0102 against 0.0156;
    - at margin 5.6, 0.0266 against 0.038;
    - at margin 6.8, 0.039 against about 0.071 (interpolated).

    Absolute accuracy at matched margin is about the same (0.575 against 0.60 at 3.75; 0.70 against 0.675 at 5.6).
  - **The bracket gets much finer.** The 30-point gain in absolute accuracy moves from steps 6–8 to about steps
    16–24, past the warmup. There each step adds about 2 points instead of about 8, and a 5-step bracket spans
    about ×2 in KL instead of ×7–10.
  - The gradient norm stayed at 1.46–2.3, so the clip is active at this lr too. The run reached absolute accuracy
    0.85 at step 40.
  - So the main gain is interpolation resolution, with a modest gain in collateral drift. This is batch 8 on CPU
    with one seed, so treat the ratios as rough.

**Smallest change** (to record in Amendment 5, as it offers).
- lr 3e-5 for DPO and SFT, warmup unchanged. SPEC2's "the RL recipe's lr" has no special meaning for a
  summed-log-probability loss under a clip that is always active.
- Grid 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200.
- Probes in step order until the first snapshot past +30, as Amendment 4 already says.
- The crossing should move from about step 6–8 to about step 15–30 (batch 64 may be somewhat faster per step than
  the smoke's batch 8). There, neighbouring snapshots differ by ×1.2–1.5 in cumulative lr.
- Cost: about 15–20 more optimizer steps per run before the crossing (seconds on a 4090) and about 3 more cheap
  probes per rated run.
- If the lr must stay at 1.5e-4, snapshot every step from 1 to 15 instead. That brackets the crossing within one
  step, at the cost of about 8 probes, but keeps the larger off-target drift.

### T4. SHOULD FIX: the swapped control's early drift is orientation sampling noise, and its two seeds share one orientation

**Evidence.**
- **The smoke's drift is explained by the pairs it saw.** At step 6 the swapped run's held-out relative margin was
  +0.75 against the rated run's +2.33. In the 48 pairs it had seen (seed 0, `--swap-seed 0`), 29 had the A member
  preferred and 19 the B member: a net push toward A of 21 % of the rated run's, the same order as the observed
  32 %.
- **At batch 64 the noise is smaller but not zero.** One standard deviation of the net push is about 1/(8√k) of the
  rated push after k steps: 5.6 % at step 5 and 4 % at step 10. At +30 that is ±1.7 points (1 SD) of
  reiteration-driven movement in the control, and up to about ±3 points at 2 SD. The read (steps 5–30) is exactly
  where this matters.
- **The two control seeds share one orientation.** Both default to `--swap-seed 0`, so they are not two independent
  draws of "a rater with no preference".

**Smallest change.**
- Stratify every swapped-run batch: exactly half swapped, within each sampled side. Keep a fixed swapped set, so the
  labels stay learnable.
- Pass `--swap-seed` equal to `--seed`.

### T5. SHOULD FIX: the "last snapshot" at 2,000 steps is far past the read

**Evidence.**
- **How far.** 2,000 × 64 pairs is about 25 epochs of 5,000 pairs (`TUNING.md` raises the same point). KL was
  already 0.23 nats per token at step 20 of batch 8.
- **What it costs.** Amendment 3 point 3 measures "the last one" fully, and Amendment 4 point 2 measures tags A and B
  on the last snapshot of each seed-0 reiteration DPO run. Both spend full measurements on a model about 100× past the
  crossing, possibly collapsed.

**Smallest change.**
- `--max-steps 300` for DPO and SFT, at least ten times the expected crossing at either lr.
- Take the tag A / tag B measurement on the upper bracketing snapshot.
- This also saves pod time.

### T6. NOTE: the DPO loss is correct, and the cancellation is exact and consistent (question 1)

- **The loss.** Per pair, `lh_dpo.step_fn` computes −log σ(β · [(Σ_w − Σ_w,ref) − (Σ_l − Σ_l,ref)]):
  - Σ is over the action tokens and closing `<eos>` of every step's untagged state (`encode_proof` forces
    `author_tag = 'none'`);
  - β is 0.1, and the loss is the mean over 64 pairs;
  - the reference is a second load of the base, `requires_grad_(False)`, `eval()`, checked unchanged at the end
    (`ref_frozen`).
- **No dropout.** Dan's `ALiBiGPT` has no dropout, so policy in `train()` and reference in `eval()` agree at step 0.
  The step-0 loss is 0.6931 in the smoke.
- **Exact cancellation.** A shared row is the same function of θ in both members, so removing it changes neither the
  margin nor its gradient. This holds to floating point, not as an approximation.
- **What counts as shared.** Shared means byte equality of the canonical (state, action) ids, matched as a multiset.
  - `state_env.canonicalise` names each new line "max index in scope + 1". Inside a reiteration box, every row after
    the R line carries names one higher than in the twin, so it is kept in both members.
  - Once the box closes, its names leave scope, the two members' states coincide again, and those rows are shared.
  - So the only rows that differ are the reiteration boxes' rows, from the first R line to the discharge (51.5 % of
    rows shared in the smoke). Rows written after the box get no direct gradient either, not only rows before it.
  - Checked on all 200 smoke pairs (`toy_reit` train and held-out): in every pair the rows after the last differing
    row are identical in both members. In 189 the differing rows form one contiguous block; the other 11 have shared
    rows between differing ones.
- **Name offset.** It is drawn once per pair and applied to the kept rows of both members, and the reference is
  re-scored on the same shifted rows. So it pairs both the members and policy with reference. Every id from `ref0`
  (49) to 112 is a name, so the shift touches only names.
- **Balanced pairs settle F5's open point for reiteration.**
  - Rows inside the box of B-sampled pairs push "the habit as written (B) after A reiteration" up; those of A-sampled
    pairs push "A after B reiteration" down.
  - In contexts with A reiteration, the push is up for both forms in equal numbers. In contexts with B reiteration,
    it is down for both.
  - So in-box habits get no net push toward either author.

### T7. NOTE: across corpora (question 2)

- **What is the same in every corpus.**
  - Optimizer, β, batch, and the always-active clip (it fixes the update size whatever the signal's strength).
  - Which rows cancel: everything outside the reiteration boxes, in every corpus, so the kept rows are the box rows
    everywhere.
- **What differs.**
  - The number of balanced train pairs: 2 × min(A-sampled, B-sampled), which depends on each base's untagged
    reiteration share. At the read (k ≤ 30 steps × 64 = 1,920 pairs) a corpus with fewer pairs than that would be in
    its second epoch. If any corpus has under about 2,000 balanced pairs, subsample all four to the smallest count.
  - The KL states: each corpus's own records (`klstates_<c>`), so KL is comparable within a corpus only.
- **What differs in the data, not the trainer.** In `c24` the B-sampled twin contradicts the notes on the page, so
  the base finds it less likely than its `c0` counterpart. DPO weights every pair equally at step 0 (σ(0) = ½), but
  the gradient of a less likely token is larger. Report KL at the read in each corpus, as Amendment 3 already says.

### T8. NOTE: the SFT arm and its control are comparable, apart from one normalization (question 4)

- **What is identical.**
  - `sft_pref` and `sft_sampled` hold the same A-sampled proofs and differ only in the B-sampled half (twin against
    sample). The twin keeps the notes and step order, so the rows before the site get the same targets in both arms.
    Rated minus control is therefore close to the DPO contrast, and F1's composition shift cancels in it.
  - The same `--seed` gives the same epoch-1 order: the permutation is drawn before any offset. Later epochs diverge,
    because one offset is drawn per row and the files have different row counts.
  - The dropped MTP term applies to both arms. SFT gradient norms (0.12–0.42) never reach the clip.
- **The one difference.** The loss is divided by the batch's action tokens. The twins' R lines add tokens, so in the
  rated arm every shared row's gradient is a few percent smaller than in the control. That is small, but systematic
  on exactly the rows that should be identical.

**Smallest change.** Divide by one fixed constant in both arms: the batch size times the sampled file's mean action
tokens per proof.

### T9. NOTE: checkpoints and tags (question 6)

- **Loading.**
  - Snapshots are `{cfg, state, tok_mode, extra}` with the base's keys, cfg and `tok_mode`. `extra` is the base's,
    plus `tuning`.
  - Dan's `load_ckpt` and `tag_core.ckpt_meta` give the base's tokenizer mode and tag metadata (tested).
  - The GRPO loader is tested only through a sum over all weights, which is weak but enough, since GRPO lines start
    from bases.
  - `extra['steps']` still says 25,607 (Stage 1), so readers must take the step from `extra['tuning']`.
- **Tags.** I compared `snap_step6.pt` with its base. The tag-token embedding rows (ids 39 and 40) are unchanged,
  while name rows changed. With weight decay 0 and zero gradient, AdamW leaves them alone. Measuring a tuned snapshot
  under tag A / tag B therefore reads how untagged tuning carries over to tagged states, which is legitimate.

### T10. NOTE: seeds and controls (question 7)

- **What the seeds control.** `--seed` sets data order and name offsets only; `torch.manual_seed` draws nothing.
  `--swap-seed` sets the orientation.
- **Rated against swapped at one seed.** These see the same pairs in the same order; only the orientation of half
  differs. That pairing is good.
- **Two seeds of the swapped run.** They are a fair no-preference reference only with T4's two changes. Otherwise
  their mean keeps one orientation draw's bias.
- **Dropped pairs.** `swap_set` indexes the file before dropping, so "exactly floor(n/2)" swapped holds only if no
  pair is dropped. Trivial.

### T11. NOTE: other points (question 8)

- **Checking bf16 on the GPU.** Training and the reference both run under bf16 autocast. The policy's grad-enabled
  forward and the reference's no-grad forward could pick different kernels, and per-token differences add up over
  about 100–200 action tokens per pair.
  - First thing on the pod: the first `train.jsonl` row (policy still equal to the reference) must show loss
    0.6931 ± 1e-3 and |train_margin_rel| < 0.05.
- **Progress metrics.** Both held-out accuracies are weak progress measures:
  - `heldout_acc_rel` reaches 1.0 by step 2–3;
  - `heldout_acc_abs` is biased against the longer member.

  The cheap probe of sampled A-share is the only valid progress measure. The grid design already relies on it.
- **The atom rater.** The flipped note stays in scope, so every later row differs and the DPO sum is dominated by the
  rest of the proof given the note, not by the note row. Expect slower movement and more collateral KL than with
  reiteration. P3 is secondary.
- **Snapshot disk.** About 38 MB per snapshot. With 13–17 snapshots, about 20 runs need roughly 13 GB on the pod
  volume.

### Replies (part 2a)


### Round 1 on part 2a, orchestrator (agent:claude, Fable; 2026-11-27 3:56 am PT)

All findings accepted; the binding text is `PREREG.md` Amendment 7.
- **T1** accepted except its last option: the bonus is gated on a reiteration site, unreadable proofs get none and are
  counted, two probe rows are added, the fit uses the balanced pairs (already the case). The reward model stays a
  whole-proof scorer; its preference for length and for proofs with a site is reported as a property of the arm.
- **T2** accepted as proposed (interval of the bonus difference, n at least 100).
- **T3** accepted: learning rate 3e-5 and the proposed grid, extended to 300 and 400.
- **T4** accepted: stratified half-swapped batches, swap seed equal to the run seed.
- **T5** accepted with 400 steps instead of 300, because the note rater is expected to move more slowly (T11); the
  tag measurement moves to the upper bracketing snapshot.
- **T7** taken further than proposed: a fixed cap of 4,000 balanced train pairs for every corpus and rater.
- **T8** accepted. **T11** (the 0.6931 check) becomes a stop rule on the pod. T6, T9, T10 noted.

## Part 2b: measurement, gate, pairs, launcher, readout and pod flow

Audited 2026-11-27, 3:45–4:10 am PT (independent auditor, Opus). Files: `lh_core.py` 03:40, `lh_launch.py` 03:41,
`lh_read.py` 03:41, `lh_driver.sh` 03:44, `lh_rlvar.py` 03:35, `lh_pod.sh` 03:06, `lh_pairs.py` 02:44, `lh_measure.py`
02:42, `lh_gate.py` 02:25, `lh_stats.py` 02:21, `lh_pod_setup.sh` 00:26 (all as in commit e9840a5, 03:55), their tests,
and the W2 / author-tags / Dan code they import. Read against PREREG Amendments 1–7 (Amendment 7 was written at 3:56
am; its code changes were not in the files at the time of reading, see R6).

### Verdict (part 2b)

1. **Stage 1 can be trusted as it runs.** The step limit 51 reaches every sampler, Dan's strict-106 path included. The
   gate quantities match my own count exactly on 3,000 corpus records per corpus. Every base is sampled with the same
   settings, Lean-rejected attempts are never counted, and the rows hold counts only. Nothing on the pods needs to change.
2. **One stage-1 fix, on serrano, before the gate is read (R1).** The printed decision check ignores G0, and G0 counts
   notes, not noted premises.
3. **BLOCKING for stage 2, readout only (R3, R4).** The readings table never checks that its inputs are matched reads.
   On synthetic trees it prints "habits edited one by one" when no corpus reached +20, and "the author moves as a unit"
   when c4 and c0 never moved the rated habit. When one control measurement is missing, a seed's +30 read is quietly
   replaced by its last snapshot.
4. **SHOULD FIX before the c24 and c4 pods finish (R5, R6).** The second-base rule fires in almost every plausible
   outcome and is not wired. Amendment 7's changes are not yet in the code.
5. **The pod flow is bounded and fails closed.** Every job has a timeout and every wait has a limit. A missing GO stops
   the work after 12 h. Fetch md5-checks every file. No sealed set or credential is staged. Nothing new was written
   outside W and LH. The rest are notes (R7–R13).

**Must be fixed before the stage-1 numbers are read:** R1 only, and only the gate script on serrano. Nothing in the
stage-1 measurement on the pods needs to change (R2 gives the evidence).

What I ran (CPU, `nice -n 5`, single-threaded, at most 3 processes; scripts and outputs in
`/tmp/claude-10001/-home-cowifornia-code-research-nd-rl/99f17bd8-830c-4f1b-9192-98b5fa5336e1/scratchpad/audit2b/`,
session scratch; nothing in W or LH was edited, apart from appending this section):
- the runner's tests as RUNNER.md documents them: **75 passed** in 61 s (RUNNER.md says 74);
- the gate end to end on corpus records: 3,000 seeded records each of `c24` and `c0`. Each record became a one-attempt
  "theorem" row through `lh_measure.lh_counts`, filed under the tag it shows, and then went through `lh_gate.gate()`.
  Beside it I counted the same quantities straight from `lh_profile.profile`'s `seq` with my own link rule, written
  from Amendment 3 point 8;
- an import-order check that the step-limit wrapper is what Dan's strict-106 sampler calls;
- `lh_read.readout` on five synthetic trees, built with the test's own row writer (cases A–E below);
- `DRY_RUN=1` staging of all four corpora, using a copy of `lh_pod.sh` that stages into my scratch instead of
  `LH/pod_stage`;
- greps of the staged files for sealed names and secrets; a check that the RL pool, the measurement sets and their
  prompts and keys do not overlap; a search for files modified outside W and LH.

### R1. SHOULD FIX (stage 1; serrano only, before the gate is read): the decision check ignores G0, and G0 counts notes, not noted premises

**Evidence.**
- **G0 is skipped.** `lh_gate.decision` (lines 221–237) reads only c24's primary notes|notes G2. PREREG's G0 ("at least
  half of the premises of accepted proofs carry a note, in each base. If not, ... the experiment stops for repair") is
  in neither the rule nor the printed reading.
  - A c24 base that writes notes on 10 % of premises can still print "stage 2 runs", if those few notes link.
- **G0 over-counts.** `lh_stats.note_rate` (lines 151–154) takes (A + B + other notes) / premises.
  - `lh_profile` counts every note of a premise (`double_note`), so a premise with two notes adds 2 to the numerator.
    The rate can therefore exceed the share of noted premises.
  - The exact quantity is already in the rows: premises without a note are `none`, so the noted share is
    (sites − none) / sites.

**Smallest change.**
- Compute G0 as (s − n) / s per class, pooled.
- In `decision()`, check G0 ≥ 50 on c24, c9, c4 and c0 first (not c0n, which has no notes by design). If any base
  fails, print "stop for repair (G0)" before the G2 reading.

### R2. NOTE (stage 1, checked; no change): the measurement and the gate compute what PREREG defines

**Step limit.**
- `lh_measure.install_step_limit` (lines 101–125) replaces `state_sample.env_generate` before any Dan module binds it.
  It handles both the positional and the keyword `max_steps`.
- I checked the import order. After `install()`, `state_ladder_ei.env_generate` (Dan's strict-106 path through
  `tag_danread` and the knockout `danread`, which passes `max_steps=48`) is the wrapper.
- The pair sampler and the GRPO snapshot measurements go through the same wrapper. GRPO itself gets `--max_steps 51`.
- `at_limit` matches what Dan's environment returns: `LEANPARSE env: step cap`.

**Counts and shares.**
- Only Lean-accepted proofs are profiled: `authors_measure.run` drops `LEANREJ` and `LEANPARSE` before `theorem_row`.
- The same holds in the probes, the full measurements, the GRPO snapshot reads, the pair sampler, the Lean-checked
  twins and the GRPO bonus (`x["ok"]`).
- Proofs the detector cannot read are counted as `unread` and left out of every share. The rows hold counts only.
- The definitions:
  - A-share = A / (A + B), with `other_pct` beside it;
  - first and later sites, in proof order, with the route, case split and ex falso left out (`NO_SEQ`);
  - pass@1 = mean over theorems of accepted / attempts;
  - pass@8 = the unbiased estimator (at k 8: at least one attempt accepted);
  - mean lines over the readable proofs;
  - the step-limit share from the newest sampler call, one call per (set, tag);
  - the near-miss counters.

**The gate, recomputed** (3,000 corpus records per corpus; `lh_gate` against my own count; points, and 95 % intervals
from `lh_gate`, B = 300):

| quantity | c24, lh_gate | c24, my count | c0, lh_gate | c0, my count |
|---|---:|---:|---:|---:|
| G1, pooled notes (tag A − tag B) | 100.0 | 100.0 − 0.0 | +1.8 [−2.2, +5.4] | 51.66 − 49.85 |
| G2 primary, notes\|notes | 100.0 [100, 100] (617 / 670 later sites) | 100.0 | −1.4 [−7.2, +3.0] | −1.40 |
| G2 primary, natural\|natural | 67.2 [63.6, 71.7] | 67.15 | +6.8 [+1.8, +12.3] | 6.81 |
| G2 primary, notes\|natural | 78.0 | 78.03 | −0.6 | −0.59 |
| G2 primary, all\|all | 82.8 | 82.83 | −0.9 | −0.88 |
| G2 all sites, natural\|natural | 75.1 | 75.12 | 31.1 | 31.11 |

- Every value agrees to the second decimal. On the c24 sample the decision check reads "stage 2 runs".
- natural\|notes is empty in both: notes are always written first.
- **The natural group has a non-zero floor.** c0, where nothing is linked, reads natural\|natural +6.8 with its interval
  above 0. So natural-group links must be read as c24 − c0, which the report prints. The decision rests on
  notes\|notes, which reads −1.4 in c0.
- **The gate follows Amendment 3 point 8.** Case split and ex falso are out of the pooled G1 and out of the sequence.
  The bootstrap resamples theorems, with one weight matrix shared by tags, bases and corpora.

**Performance table.**
- Every base, the two Nov 14 bases included, goes through the same `measure` job on the same sets: k 8, three tags,
  T 0.8, 256 tokens per action, step limit 51, noise seed `0 * 1000 + set index`, batch 256.
- The flag is on pass@1 on `marker_plain`: point below −5 with the upper end of the interval below 0. It is computed
  for c0 − control_half, c24 − authors_half and c0 − c0n.
- Tags are handled the same way for the Nov 14 and the new bases. The tokenizer mode is checked
  (`T.ckpt_mode`), and the spy confirms the tag in every state.

**One difference across stages.**
- The GRPO snapshot reads use W2's `snap_cmd` at batch 128, against 256 everywhere else.
- Dan's sampler seeds each decoding wave, so these reads are not the same random draws as the base's measurement.
  They are still paired between the rlhf and anc lines. The "move" of a GRPO line is therefore against an unpaired base
  read, which adds noise but no bias.
- I would leave it, and say so in the readout.

### R3. BLOCKING for stage 2 (readout only; no pod change): the readings table does not check that what it reads is a valid matched read

**Evidence.** `lh_read.readings` (lines 832–859) applies the table to `P1_M` and `P2_S` whatever produced them. It
ignores four checks:
- `preconditions` (rated habit at least +20, PREREG Amendment 1);
- `reached` and `bracketed`;
- the two seeds' signs;
- its own `second_base_needed`.

These checks are computed and then printed beside the reading, not applied. Synthetic trees (base 50 everywhere; every
theorem identical, so the intervals have zero width):

| case | setup | reading printed |
|---|---|---|
| A | the rated habit reaches only +10 in every corpus; nothing else moves | "habits edited one by one, as on Nov 18" (preconditions false in all three corpora) |
| C | c24's order and conj: +20 in seed 0, −20 in seed 1 | "habits edited one by one" (mean D 0; per-seed +20 / −20 printed, not used) |
| D | c24 reaches +30 with order, conj and notes +20; c4 and c0 move the rated habit +5 only | "the author moves as a unit with 24 links and not with 4" (preconditions c4 false, c0 false) |

- Case D is the decisive claim. It was produced from a c4 that never moved the rated habit, so M(c4) = 0 by
  construction.
- **The second-base rule is printed but never applied.** In the test fixture `readings()` prints "the author moves as a
  unit" together with `second_base_needed = [M(c4), S_before(c24)]`. Amendment 3 point 4 forbids that conclusion until
  the second base has run.

**A worked case, B (correct).**
- Setup: the base at 50; step 2 at +40; order and conj 70 in the rated run and 50 in the control (c24).
- Hand calculation: w = 30 / 40 = 0.75, so D_order = 0.75 × 20 + 0.25 × 0 = 15. The code gives 15. The bracket is
  [0, 2], from the base.
- In the test fixture (moves +20 at step 50 and +40 at step 100), w is 0.5 and D_order is 20 and 24 in the two seeds,
  mean 22. The code agrees.

**What happens in each situation asked about.**
- **+30 never reached.** `at_target` is the last snapshot, flagged `reached: false`, and still used (case A).
- **+30 passed at the first snapshot.** The read interpolates from the base (case B; correct).
- **A habit with fewer than 50 sites.** It is dropped from D and from every mean. `vs0` keeps the habits readable in
  both corpora, so S can average over different habit sets at different points. That is acceptable, but the report
  should list the habits used; `habits` is in the JSON.
- **The two seeds disagree in sign.** The mean is used, and the interval (theorems only) does not see the
  disagreement (case C).

**Smallest change.** In `readings()`, before the table:
- print "not read" unless, for c24, c4 and c0 and for every seed, the `at_target` point has two steps, `reached` and
  `bracketed` true, and the precondition met;
- print "not read: seeds disagree" when the two seeds of M(c24), M(c4) or S_before(c24) have opposite signs and
  either one lies outside ±5;
- when `second_base_needed` is non-empty, print the row as "provisional (second base required)".

These are about 20 lines. Add cases A, C, D and E as tests.

### R4. BLOCKING for stage 2 (readout only): a missing bracket measurement silently swaps a seed's +30 read for its last snapshot

**Evidence.**
- `lh_read.offline_read`, lines 323–346: if any bracket step lacks a finished measurement in the rated or the control
  run, the `elif last in cdir` branch makes `at_target` the last snapshot. It is marked `reached: false`, but it still
  enters `mean_point` and every PREREG quantity.
- **Case E.** Every corpus reaches +30 at w 0.5. In c24, order and conj stay at 50 through step 100 and drift to 90 by
  step 2,000. I deleted the control's step-100 measurement for seed 0 only (a failed full-task step).
  - The read became seed 0 at step 2,000 and seed 1 at the +30 bracket.
  - D_order = 20 and M(c24) = +20, where the true value at +30 is 0.
- **Known gap 9 is not what the code does.** RUNNER.md says "the readout then drops that point". It does not: it
  substitutes another point.

**Smallest change.**
- In `offline_read`, when the bracket is known (from the full-task record or from the full measurements) but a
  measurement is missing, return `at_target: None` with a reason. Do not fall back to the last snapshot.
- Use the last-snapshot fallback only when +30 was never reached, and R3's check then keeps it out of the table.

### R5. SHOULD FIX (decide before the c24 and c4 pods end stage 2): the second-base rule will almost certainly fire, and it is not wired

**When it fires.** Amendment 3 point 4 fires if M(c24), M(c4) or S_before(c24) lies strictly between 0 and +20
(`lh_read` lines 850–854).
- **Under no effect,** each quantity is about 0 ± noise, so each lies in (0, 20) about half the time. The chance that at
  least one does is roughly 0.5 to 0.9, depending on their correlation; they share c0 and the c24 run.
- **Under the hypothesis,** M(c24) and S_before(c24) are expected between +10 and +20.
- **Nov 18's analogue of M(c4)** was +8.7.
- So the rule stays silent only if all three are at or below 0 or at or above +20.

**It is not wired.** Known gap 1:
- pairs live in `pairs/<c>_s0`;
- run names carry the trainer's seed, not the base's;
- the arms start from the seed-0 base only.

So the likely outcome of stage 2 is "no conclusion until a second base", and that needs new code and new pods once
these are deleted. Amendment 3 prices it at about $6.

**Smallest change.** Either:
- **(a)** wire it now: a base-seed suffix in `pairs_dir`, `arm_name` and `full_name`, and a `--base_seed` on prep2 and
  stage2. Train `c24_s1` and `c4_s1` on the c24 and c4 pods right after their stage 2, before they are deleted; or
- **(b)** amend the trigger before any tuned model exists, for example "the interval contains +10", which is the bar
  the readings depend on.

### R6. BLOCKING for stage 2 until applied (the builder's queue): Amendment 7 is not yet in the code

At 3:58 am PT:
- **Point 1:** `lh_core.py` still has `SNAP_STEPS` from 2 to 2,000, `OFFLINE_MAX_STEPS = 2000` and `OFFLINE_LR = 1.5e-4`
  (lines 90–94).
- **Point 2:** `full_main` puts tags on the last snapshot (`st == max(by)`, `lh_launch` lines 1182–1186).
  - Point 2 needs the upper bracket instead: `sel["bracket"][1]`.
  - `task_est` (line 511) assumes the same.
- **Point 6, the reward-driven mark:** `lh_read` line 717 still applies the 45 / 55 % rule.
- **Point 6, the bonus:** `lh_rlvar.RMBonus` pays on every accepted proof, with no reiteration-site gate.
  - Its fallback `of()` (lines 154–159) scores a proof with an empty prompt when `lh_prompt` is missing.
  - The site gate should call `lh_profile.profile(prompt, nd, tables)` with the corpus tables. An unreadable proof, or
    an empty prompt, pays 0 and is counted.
- **Point 5:** the 4,000-pair cap in `lh_pairs.balance` and its readout flag.

After the change, rerun `test_lh_launch.py`, `test_lh_read.py` and `test_lh_rlvar.py`. Also check:
- the probe's stop rule still reaches the first grid step past +30 on the new grid;
- `select_steps` uses the new last step (400);
- `lh_read.snapshots` maps `final` to 400, not to `OFFLINE_MAX_STEPS` if that constant is left stale.

### R7. NOTE: the GRPO read is labelled "primary at the last snapshot", which PREREG does not say; the chain prediction mixes weights

**Primary read.**
- `lh_read` line 712 makes `last` (iteration 40) the primary GRPO read and computes the chain prediction there only
  (lines 706–709).
- Amendment 1 reads every arm at matched movement, with the last snapshot beside it. Amendments 4 and 6 change only the
  iterations.

**Chain weights.** In `chain()` (lines 580–613):
- the mixing weight q is the share of proofs whose first reiteration site is A (`condn`);
- the conditionals and the observed value are site-weighted (`cond`).
- If A-first proofs carry a different number of note sites than B-first proofs, the prediction is biased by the
  difference. The anchor-line difference removes this only if both lines have the same mix.

**Smallest change.**
- Compute the chain prediction at both points, and label the primary per PREREG (or record the choice as an
  amendment).
- For habit h, take q from the cond counts themselves: q_h = (AA + AB) / (AA + AB + BA + BB).

### R8. NOTE: the pair build checked against Q5

**What holds:**
- pairs come from untagged samples on Dan's RL pool (`set_process_tag("none")`, the spy), Lean-accepted;
- every twin is checked by Lean, and the rated habit is flipped at every site, with every other habit's (sites, A, B)
  unchanged;
- mixed proofs are skipped;
- train and held-out are split by whole theorem;
- the DPO files are balanced by sampled side with a seed per corpus, rater and split;
- `sft_pref` and `sft_sampled` are the unbalanced preferred and sampled members of the same train pairs;
- `probe_twins` covers the 15 notes, restate, halves, conj and disjunct, from held-out theorems only.
- **No measurement-set theorem can enter.** The pool and the two measurement sets share 0 prompts and 0 keys (I
  checked), and the sets have no duplicate prompts.

**Two small points.**
- **A degenerate split passes silently.** If one sampled side has fewer than 100 pairs in total, `split` (lines
  325–330) never stops, every theorem goes to held-out, and train is empty without an error.
  - Guard: fail when train is empty or held-out holds more than half the theorems.
- **A mislabelled reason.** For the atom rater, a proof with an un-noted atom premise is skipped as "mixed"
  (a + b < sites).
  - It is still excluded, which is correct, but the `why` count says "mixed".

### R9. NOTE: the launcher's task list and the probe's stop rule

- **The task list matches Amendments 4 and 6.** I printed `tasks_of`:

  | corpus | tasks | GRPO lines | atom rater |
  |---|---:|---|---|
  | c24 | 14 | rlhfreit and anc | dpo and dposwap, one seed |
  | c0 | 14 | rlhfreit and anc | dpo and dposwap, one seed |
  | c9 | 9 | none | none |
  | c4 | 11 | rlhfreit and anc | none |

  - Rated runs are probed and no-preference runs are not.
  - The full task measures the bracket and the last snapshot, untagged, in both runs.
- **The stop rule cannot skip what the read needs.** Probes run in step order and stop after the first snapshot
  strictly above base + 30, then probe the last.
  - The bracket's lower end is therefore always a probed snapshot, or the base.
  - The only gap is gap 22: if the full measurement disagrees with the probe, the snapshots after the stop stay
    unmeasured. R3's `bracketed` check makes that visible.
- **A training timeout loses the whole arm.** A trainer killed by its 2 h cap leaves no `done.json`, and a rerun
  refuses the partial run (`lh_launch` lines 1057–1059), even when the snapshots the read needs exist.
  - With Amendment 7's 400 steps this is unlikely. Leave it, but name it in the operator notes.

### R10. NOTE: the GRPO arms (Q8)

- **The reward.**
  - The reward is `rewards()` (1 − len_penalty per 1,000 tokens if Lean accepted, else 0) plus 0.3 × σ(RM).
  - `RMBonus.of` pays only when `x["ok"]`, so a rejected proof never gets a bonus.
  - That is "acceptance × (1 + 0.3σ)" up to the recipe's length penalty, which both lines share.
- **The KL anchor.** `--kl_anchor_coef 0.02`, `--kl_anchor_ref` = `--init` (asserted twice) and `--kl_anchor_data
  anchor_<c>.jsonl` reach `code.thinking.grpo`. The trainer asserts that the anchor has data.
- **The two lines differ in the reward alone.** The anc and rlhf lines share the argv (apart from name and out), the
  seed and the base. Only `rlvar.grpo_main` installs the bonus, and only when `bonus` > 0.
- **The final read is not lost.** W2's `AuthorsSeg.authors_reader` (the reader of a variant line) makes a last pass
  after training ends, so iteration 40 is measured.

### R11. NOTE: pod flow (Q9)

**Timeouts.** Every launched step runs under `timeout --kill-after=300`.
- Each stage-2 task runs under its own `timeout`; GRPO snapshot reads have a 3,600 s timeout.
- The driver runs under 48 h, and every `wait` is bounded.
- The stage-2 job cap I computed is 14.1 h on c24 and c0, 12.4 h on c4 and 8.2 h on c9, below the driver's 15 h wait.
  RUNNER.md still says 15.7 / 17 h.

**Failures, GO, reruns and memory.**
- **A failed task cannot hang the driver.** It returns a code, and the scheduler marks prerequisites that never end.
- **No GO.** Without GO or NOGO the driver dies after `GATE_WAIT_S` (12 h) and writes FAILED, so the work stops.
  Billing continues until the pod is deleted.
- **A rerun redoes only what failed.**
  - It skips finished work: `finished()` checks `done.json`, the probe, `"done": true` and `end.json`.
  - A finished training with a failed probe re-probes, and probes resume at the DONE snapshots.
  - Exception: a GRPO line that failed cannot be rerun by the driver. seg1 exists, so it is refused, and `finished()`
    reads seg1 only. It needs a manual `lh_pod.sh job … rl --line L --seg 2 --resume …`, and the driver keeps listing
    it as FAILED.
- **Memory.** The measurement processes are capped at 0.3 of the card and the trainers run at 32,768 tokens per chunk,
  as gap 20 says; the GRPO-plus-reward-model peak is unmeasured.

**Fetch.** It md5-checks every listed file before and after rsync. But the waiting driver rewrites
`pod/<c>.heartbeat.json` every 60 s, so the gate-time `fetch --light` will often report "MD5 DIFFERENCES" and exit 1.
- That looks like a failure at the moment of the gate.
- Smallest change: leave `pod/*.heartbeat.json` and the running job's log out of the strict comparison, or
  md5-list twice and compare only files that were stable.

### R12. NOTE: sealed sets, credentials and writes outside W and LH (Q10, Q11)

**Staged files** (fresh `DRY_RUN` stages into my scratch):

| pod | files | size |
|---|---:|---:|
| c24 | 149 | 51 MB |
| c9 | 149 | 51 MB |
| c4 (with the Nov 14 bases) | 155 | 125 MB |
| c0 (with c0n) | 151 | 75 MB |

- No path matches `textbook72|tb72|test_13_20|test_21_30`.
- The staged `transfer_track832.jsonl` has 106 rows in all four stages, and `stage` asserts that they are the strict
  106.
- **No credentials or keys.** A content grep for keys and tokens finds only the GRPO snapshot's API-client code, which
  reads `ANTHROPIC_API_KEY` from the environment at call time. That code is staged but never called.
- **No sealed data.** Code that names sealed-set paths (`code/thinking/data.py`, `coe_assemble.py`) is staged, but the
  files it names are not, so nothing on the pod can read them.

**Writes outside W and LH.**
- **Fixed.** The documented line in `knockout_danbase/dan_code_13e41bbd/artifacts/lean_gate.jsonl` is still there: 91
  lines, the last at 09:44 UTC, and the file is unchanged since then.
  - Knockout `danread` does `os.chdir(DAN)` and Dan's gate logs relative to cwd, which caused it.
  - `strict106` now sets `LEAN_GATE_LOG` before calling it.
  - Every other Lean path sets its own log or writes none.
  - The search finds no other change under `knockout_danbase`, `two_authors`, `knockout_grpo`, or the W2, tag and
    knockout code folders (git shows no diff).
- **Bytecode only.** The only other writes are `__pycache__` files:
  - 7 in Dan's tree at 00:03 PT, before the runner started (most likely the corpus agent's `nd_verify` imports);
  - git-ignored ones in the sibling code folders.

### R13. NOTE: stale text, and one saving

**Stale text.**
- `lh_rlvar.py`'s docstring and `lh_launch.py`'s docstring still say 60 iterations and snapshots 10, 20, 40, 60.
- RUNNER.md says 60 iterations, 74 tests and caps of 15.7 / 17 h.
- The code uses 40 iterations, snapshots 10, 20, 30 and 40, 75 tests, and computes 14.1 / 15 h.

**A saving.** Full measurements run both sets, but every D is read on `marker_plain` (`READ_SET`). `route_unseen` is
3/8 of each full measurement's attempts and feeds only the pooled tables. Dropping it from the tuned snapshots, keeping
it on the bases, saves about a third of the full-measurement time.

### D (Q12). The most likely way stage 2 yields a number that looks like a result and is not

The readings table turns whatever sits in the `at_target` slot into a verdict (R3, R4). Any fallback produces a
verdict:
- a corpus that did not reach +20;
- a probe and full measurement that do not bracket +30;
- one failed control measurement, which puts a seed's step-400 read in place of its +30 read;
- two seeds of opposite sign that average to zero inside a theorem-only interval.

The verdict line is printed as a reading, while the facts that invalidate it sit in separate fields of the report.

**Cheapest guard:** R3's validity check in `readings()`, plus R4's "no silent fallback". These are about 30 lines,
testable on cases A, C, D and E, with no pod change and no new measurement.

### Replies (part 2b)


### Round 1 on part 2b, orchestrator (agent:claude, Fable; 2026-11-27 4:03 am PT)

All findings accepted; the binding text is `PREREG.md` Amendment 8.
- **R1** accepted (G0 on noted premises, applied before G2). Serrano only; fixed before the gate is read.
- **R3, R4** accepted as proposed, with tests on the auditor's synthetic cases A, C, D and E.
- **R5**: option (a), in a narrow form. Pretraining a base takes about 20 minutes on these pods, so a second base
  for `c24` and `c4` with the reiteration DPO arm costs about 4 pod-hours; it is run in any case, after stage 2 on
  the same pods, and the trigger is withdrawn. I did not take option (b): changing the trigger would still leave the
  decisive contrast resting on one base per corpus.
- **R6** was in the builders' queue when the audit ran; it is checked again before any `go`.
- **R7** accepted: matched movement is the primary GRPO read, iteration 40 beside it; chain weights from the site
  counts.
- **R11** accepted (heartbeat and running log left out of the strict md5 comparison). **R13**'s saving accepted
  (tuned snapshots on `marker_plain` only). R2, R8, R9, R10, R12 noted.

## Part 2c: verification of the fixes at e7b3388

Checked 2026-11-27, 4:50–5:20 am PT (independent auditor, Opus). Code read only from the frozen checkout
`/scratch/cowifornia/nd-rl/worktrees/linked-habits-frozen2/code/experiments/current/linked_habits_20261127` (F2, commit
e7b3388), against PREREG Amendments 7 and 8 and the part-2b findings. At the time of reading, F2's runner, trainer and
detector files were byte-identical to the working folder's.

What I ran (CPU, `nice -n 5`, single-threaded):
- **Tests in F2:** `test_lh_{read,gate,pairs,rlvar,launch,pod,measure}.py`, 89 passed.
- **Launcher output:** F2's trainer argv, task lists and stage-2 job caps, printed.
- **Diffs:** e9840a5 → e7b3388 for every file that runs on a pod.
- **The gate:** F2's `lh_gate.py` on the stage-1 measurements now in LH (c4_s0, authors_half_s0, control_half_s0), output to
  my scratch, not to LH/gate.
- **An independent recount of c4.** I read c4's raw accepted proofs (`LH/raw/c4_s0`), not lh_measure's rows, and
  applied my own G0, G1, G2 and pass@1 rules.
- **The synthetic readout cases A to E** of part 2b, run against F2's `lh_read.py`.

Scripts and outputs are in the session scratch, `.../scratchpad/audit2b/` (`gate_f2/`, `recount_base.py`, `f2read/`).

### Verdict (part 2c)

1. **Stage 2 may start from e7b3388.**
   - Every R6 item is in the code and reaches the pods' commands.
   - The Amendment 7 trainer changes are correct and match the launcher's calls.
   - Copying at the gate is safe with no driver restart, with the two conditions in V1.
2. **The gate is correct on real data.** F2's gate agrees exactly with my recount of c4 from the raw proofs. It runs
   with bases missing and applies G0 first.
3. **The readout now prints what PREREG requires in all five synthetic cases.** A, C, D and E read "not read" with the
   reason, and B is interpolated correctly.
4. **Open:** the second-base pass (Amendment 8 points 2–3) is not in e7b3388 (V2).
5. **Observation for the orchestrator:** c4's pass@1 is 17–18 points below both Nov 14 bases (V5).

### V1. SHOULD FIX (operator; before `go`): stage the copy from F2, and keep the c24 and c4 pods after DONE

**Evidence.**
- **The copy stages from whatever folder its script sits in.** `lh_pod.sh` sets `W` to its own folder and `EXP` to that
  folder's parent, so `copy` stages from there. F2's RUNNER.md ("Runpod, step by step", line 176, and "Bringing in new
  code at the gate", step 1) sets `W` to the working folder, which is being edited again.
  - A copy from there would stage whatever is in the working folder at that moment, including a half-built second-base
    pass.
- **The old driver will not run the second base.** Amendment 8 point 2 runs it after stage 2 on the c24 and c4 pods.
  The driver writes DONE at the end of stage 2 and knows nothing of a second base (V2). RUNNER's rule "delete a pod at
  DONE" would delete it first.

**Smallest change.**
- Run `bash $F2/lh_pod.sh copy rp-lh$C $C`. Then compare one staged file's md5 with F2's, e.g.
  `LH/pod_stage/rp-lh$C/exp/linked_habits_20261127/lh_launch.py`.
- Keep the c24 and c4 pods after DONE until the second-base pass has run.

### V2. SHOULD FIX (before the second base runs; not before `go`): the second-base pass is not in e7b3388

**Evidence.**
- **No base-seed naming.** `lh_core` still has `pairs_dir(corpus, rater, seed=0)`, `arm_name(kind, corpus, rater,
  seed)` with the trainer's seed only, and `base_of(corpus)` fixed at seed 0.
- **The readout knows it is incomplete.** `lh_read.readings` (lines 944–980) says so in a `basis` field ("the first
  bases only"). It is printed on the readings line of the report.
- **The per-base rule is not there yet.** Amendment 8 point 3's per-base averaging and the "unresolved (bases disagree)"
  row are not implemented.

**Smallest change.**
- When the pass is built, a test must show that the readings table averages over base seeds and applies the
  bases-disagree rule.
- Until then, every first-base reading in a report must be labelled provisional. The basis note does this; keep it.

### V3. NOTE (A1): R6, item by item, in F2

| item | where (F2) | checked |
|---|---|---|
| grid 5…400, 400 steps, lr 3e-5 in every offline argv | `lh_core` 97–101; `trainer_argv` 262–281 | printed argv of dpo, dposwap s1, sft and sftsam: `--lr 3e-05 --max-steps 400 --snap-steps 5,…,400` in all |
| `--swap-seed` equal to the run seed | `trainer_argv` | dposwap_s1 gets `--swap-random --swap-seed 1`; the rated dpo uses lh_dpo's default (its own seed), so the swap set and batches are the same |
| same `--norm-tokens` for sft and sftsam | `trainer_argv`, `norm_tokens_of` | both read the one value in the pair summary, with `--batch 64` |
| without the normalizer | lh_sft; `task_problems` | lh_sft exits 2 if it is missing; `task_problems` refuses the pair, and `_tasks` refuses only those two tasks |
| pair cap | `lh_pairs` build | train: `balance(…, cap=4000)` after the balancing, so at most 2,000 per side; the fine-tune files: one seeded draw of 4,000 train pairs before the balancing, the same pairs in both files; flags `*_below_cap` |
| tags only on the upper bracket of `dpo_<c>_reit_s0` | `full_main` (`lh_launch` line 1273) | `st == sel["bracket"][-1]` and only for that run (when +30 is not reached, the last snapshot); probes and GRPO reads untagged |
| tuned snapshots on marker_plain only | `C.TUNED_SETS` | used in `full_main`, `snap_cmd` (GRPO) and the probe |
| GRPO | `lh_core`, `lh_rlvar` | 40 iterations, snapshots 10/20/30/40, lines on c24, c4 and c0 only (`tasks_of`: c24 14, c4 11, c0 14, c9 9) |
| the reward bonus | `RMBonus.score_call`, `rm_readable` | paid only if Lean accepted, `lh_profile` finds at least one reiteration site, and the reward model's `last_ok` says readable; counters `accepted, unreadable_profile, no_reit_site, scored, unreadable_rm, paid, no_prompt`; an attempt with no prompt pays 0 |
| probe:nosite and probe:length | `extra_probe_rows` | built from held-out theorems only; none for c9 or the atom rater (`--no_probe`) |
| step-0 stop | `lh_tune.train_loop`; `lh_launch` `step0_problem`, `offline_main` | the trainer checks loss = ln 2 ± 0.001 on step 1 before `opt.step()`, writes `step0_failed.json` and exits 4; the launcher then returns 5, appends to `<c>.FAILED`, and never probes or measures the run |

**Stage-2 job caps** (F2, computed): c24 and c0 11.1 h, c4 10.1 h, c9 7.0 h. All are below the driver's 15 h wait. The
prep2 job (6.0 h cap) is below its 6.5 h wait.

**Not testable without a GPU.** The 0.001 tolerance under bf16 on the 4090 was never run. If it fires falsely, every
DPO run stops before its first update and FAILED shows it within minutes; no money is lost beyond that.

### V4. NOTE (A2, A3): the trainer changes and the copy at the gate

**Trainer changes (A2).**
- **Swap control (lh_dpo).** A fixed swap set: half of each sampled side, drawn once from `--swap-seed`. Batches are
  whole couples, one swap-set pair and one other pair from the same side, so every even batch is exactly half swapped
  within each side. A rated run and its control with the same seed see the same batches.
- **Fine-tune.** lh_sft divides by the fixed N from the pair build: 64 × the mean action tokens per proof of
  `sft_sampled`.
- **Reward-model probe.** `lh_rm.classify` marks a habit reward-driven when the bootstrap interval of its mean bonus
  difference lies outside ±5 % of the reference's. A habit with fewer than 100 pairs is "not probed".
- **Unreadable proofs.** `score_many` returns 0.0 for an unreadable proof with a flag in `last_ok`, and `RMBonus` pays
  nothing on it.
- **The readout uses the probe's own marks.** `lh_read.rm_probe` uses the probe's `status` and drops the 45/55 rule.

**Copy at the gate (A3).** It is safe as long as it is staged from F2 (V1).
- **The driver.** `lh_driver.sh` is not in the e9840a5 → e7b3388 diff. It is identical to the copies staged for all
  four pods (cmp).
- **The stage-1 code.** `lh_measure`, `lh_profile`, `lh_habits`, `notes` and `lh_twin` are identical to the staged
  copies. A pod still in stage 1 that receives the copy keeps measuring with the same code.
- **The old driver after GO.** Every interface it uses is unchanged:
  - `lh_core.tasks_of` and `raters_of`;
  - the `prep2` and `stage2` command lines;
  - `wait` and the job names `prep2_<c>_s0` and `stage2_<c>_s0`;
  - the paths `finished()` reads.
- **Why replacing files is safe.** rsync without `--inplace` writes a temporary file and renames it, so the running
  bash keeps its open file. New `.py` modification times invalidate the pods' `__pycache__`.

### V5. NOTE (B4): the gate on c4, against my recount from the raw proofs

| c4_s0 (pooled sets, untagged) | F2 `lh_gate` | my recount from raw proofs |
|---|---:|---:|
| G0, noted premises | 99.5 [99.2, 99.8] | 99.50 |
| G1, pooled notes (A 52.88, B 53.58) | −0.7 [−2.9, +1.5] | −0.70 |
| G2, notes\|notes | −3.8 [−8.1, +1.0] (842 / 826 later sites) | −3.81 (842 / 826) |
| G2, natural\|natural | +10.7 [+6.4, +14.9] (1,554 / 1,839) | +10.65 (1,554 / 1,839) |
| G2, notes\|natural | −0.5 | −0.45 |
| G2, all\|all | −1.2 | −1.23 |
| pass@1, pooled / marker_plain | 29.6 / 44.7 | 29.64 / 44.70 |

**The two counts agree exactly.** So both the pod's row counting and the gate's arithmetic hold on a real base.

**The decision check with bases missing** reads "not complete: lacks c24_s0, c9_s0, c0_s0". It shows G0 so far
(c4 99.5) and does not crash. G0 is applied before G2, and a failing base would print "stop for repair (G0)" even
while others are missing.

**Observations [obs] that matter for the report** (untagged, marker_plain; no flag pair yet, because c0 and c24 are not
measured):

| | c4 | authors_half | control_half |
|---|---:|---:|---:|
| pass@1 | 44.7 [41.9, 47.7] | 61.9 | 63.2 |
| mean lines | 11.1 | 8.1 | 8.2 |
| pass@1 on route_unseen | 4.5 | 19.6 | 24.0 |
| strict 106 | 0 | 0 | 8 |

- c4's pass@1 is 17–18 points below both Nov 14 bases.
- [interp] If c0 and c24 look like c4, the pre-set pass@1 flags (c0 − control_half, c24 − authors_half) will fire.
  PREREG says that does not stop the experiment, but the report has to carry it. c0 − c0n will say how much of the gap
  is the notes.

### V6. NOTE (B5): the synthetic cases against F2's `lh_read`

| case | part 2b (e9840a5) | F2 |
|---|---|---|
| A: no corpus reaches +20 | "habits edited one by one" | "not read", with the per-seed reasons (+30 not reached; precondition +20 not met) |
| B: +30 passed at the first snapshot | interpolated from the base, D_order 15 | unchanged and correct (w 0.75, D_order 15); reading "unresolved" |
| C: seeds +20 / −20 | "habits edited one by one" | "not read: seeds disagree (M(c24) 20.0 / −20.0)" |
| D: only c24 moves the rated habit | "the author moves as a unit …" | "not read" for c4 and c0 (precondition, not reached) |
| E: one control measurement missing | seed 0 read at step 2,000; M(c24) +20 | seed 0's at_target is None with "no finished measurement of dposwap_c24_reit_s0@100"; reading "not read"; D from seed 1 only, 0.0 |

**R7.** The GRPO arm's primary read is now the snapshot closest to +30, with iteration 40 beside it. The chain
prediction is computed at both points, with q from the same site counts (`S.cond_q`).

**A small labelling point.** `validity()` is also stored for the GRPO entries. There a single-snapshot read always
gives "not bracketed by two measured snapshots". That is true but irrelevant, because the GRPO arm is read at the
closest iteration by design and never enters the table. Smallest change: skip the bracket test for kind `rlhf`, or
label it.

### Replies (part 2c)


### Round 2 on part 2a, orchestrator (agent:claude, Fable; 2026-11-27 5:13 am PT): the trainer builder's two objections

The builder of the trainers applied every part-2a item and recorded two objections in `TUNING.md` (section dated
2026-11-27, about 4:40 am PT). Both stand as written there; neither changes a run.
- **T8 (fixed fine-tune normaliser).** The builder's point: AdamW without weight decay ignores a constant gradient
  scale and the clip is not active, so the stated motivation (shared-row gradients a few percent smaller in the rated
  arm) does not change the updates; the rule only fixes how batches are weighted against each other, the same way in
  both arms. Accepted: the normaliser stays (harmless, and both arms get the same constant), and the write-up will not
  claim it removed a bias.
- **T11 (step-0 check).** The builder's point: at beta 0.1 a tolerance of 0.001 on the loss passes a systematic margin
  offset of up to about 0.02 nats, so the check catches a wrong reference or dropout and not a small bf16 difference
  between the two forward passes. Accepted as a limit of the check. The rows also carry `step0_margin_absmax`; on the
  first pod runs I read it, and a value above about 0.1 nats is looked into before that run's results are used. That
  is a reading rule for me, not a stop rule.

**A fact from the pods, recorded before any tuned model was measured.** `c4`'s balanced train set has 2,354 pairs
(1,177 per sampled side; 19.9 % of its sampled proofs took the A form), under the cap of 4,000 of Amendment 7 point 5.
It is reported as the pre-set flag ("fewer pairs than the cap"); the pair counts of all corpora go beside every
reading.

## Part 2d: verification of the second-base pass and fetch --keep at ad001ea

Checked 2026-11-27, 5:15–5:50 am PT (independent auditor, Opus). Code read only from the frozen checkout
`/scratch/cowifornia/nd-rl/worktrees/linked-habits-frozen3/code/experiments/current/linked_habits_20261127` (F3, commit
ad001ea), compared with F2 (e7b3388) and against PREREG Amendment 8 points 2–3.

What I ran (CPU, `nice -n 5`, single-threaded, at most 3 processes):
- **The diff** e7b3388 → ad001ea (15 files in the runner folder: 7 code, 4 tests, 4 notes).
- **Tests in F3:** `test_lh_{pod,launch,read,gate}.py`, 79 passed. This includes the builder's 48-task comparison against F2.
- **First-base commands, F2 against F3.** Every first-base plan both checkouts build for c24, c9, c4 and c0, printed to
  JSON and compared.
- **The readout.** The part-2b synthetic cases A–E through F2 and F3, plus seven second-base cases of my own.
- **The gate.** Run three ways: F2 and F3 on the seven first-base measurements now in LH, and F3 again with two fake
  second bases added.
- **The new fetch mode.** `lh_pod.sh fetch --keep` run against a fake pod on serrano (the builder's fake ssh and
  rsync), with five faults injected. No network was used.
- **The second-base plans.** Job caps and argv computed from F3, in a scratch LH.

Scripts and outputs are in `.../scratchpad/audit2b/` (`first_base_cmds.py`, `fb_F{2,3}.json`, `f3read/`, `f3b2/`,
`gate2d/`, `b2plan/`, `fetch2d/`).

### Verdict (part 2d)

1. **Q1: the first base is unchanged.**
   - The first-base commands, the readout numbers and the gate are identical in F2 and F3.
   - The only visible difference is the readout's basis label, which now reads "first base only (second base
     pending: c24b1, c4b1)".
2. **Q2: the pass may run.**
   - It matches Amendment 8 point 2, and every job is capped below the driver's wait.
   - No first-base path is written.
   - Copy and `start` after DONE are safe, and the rerun skips stage 1, the gate and stage 2 as it should.
3. **Q3: the averaging and the bases-disagree rule follow point 3.** Marking only rows 1–2 is right. An invalid
   second-base run (never reaching +30, the precondition not met, or a missing bracket measurement on either arm)
   gives "not read" with the reason.
4. **Q4: `fetch --keep` may be used, but its rc 0 is not yet enough to delete a pod (W1).**
   - A failed or truncated ssh listing still ends in "every stable fetched file equal".
   - Check the counts by hand, or apply the small fix, before the first deletion.
   - The keep list is otherwise complete. W2 suggests a cheap addition.
5. **Q5: the 2,354-pair train set of c4 needs only the flag plus one more reported number** (W6). Then add the c24
   subsample control only if the passes at +30 differ a lot.

### W1. SHOULD FIX (before the first pod is deleted, c9 at about 7:45 am PT; not before the pass or a fetch): `fetch` can end rc 0 with files missing

**Evidence** (`fetch --keep` against a local fake pod of 17 files, faults injected in the fake ssh and rsync):

| fault | rc | printed | missing on serrano |
|---|---:|---|---|
| none | 0 | "15 stable files equal, 0 differ, 1 still being written" | none |
| the checkpoint listing's ssh fails (exit 255) | **0** | "checkpoints fetched: 0 … LEFT ON THE POD: 0 checkpoints"; "every stable fetched file equal" | **both bases, the final, the measured snapshot, rm.pt, the GRPO snapshot** |
| the main listing's ssh drops after 5 lines | **0** | "11 stable files equal"; "every stable fetched file equal" | **measurement rows, pairs, logs, GRPO metrics** |
| the second pod md5 listing comes back empty | **0** | "0 stable files equal, 0 differ, 0 still being written"; "every stable fetched file equal" | none, but nothing was compared |
| a fetched file removed after rsync | 123 | no message (xargs md5sum under `set -e`) | final.pt; the fetch fails, without saying why |
| a fetched file changed after rsync | 1 | "MD5 DIFFERS: …probe.json" | correct |

**Cause.**
- Both pod listings end in `|| true`.
- The Python check compares only the paths present in the second pod listing. It never checks that this listing,
  or the fetched copy, covers `$F.list`.
- In `--keep`, an empty `$F.pt` yields an empty keep list, and the summary then reports "0 checkpoints left on the
  pod".
- The main listing's `|| true` predates ad001ea. The checkpoint listing, whose failure loses the checkpoints, is new.

**Smallest change** (three lines in `lh_pod.sh fetch`):
- Replace both `|| true` with `|| { echo "fetch: the listing failed on $h"; exit 1; }`. Remotely, the main listing's
  status is `sort`'s, so a missing `runs/` does not trip it.
- Add `[[ $light != --keep || -s $F.pt ]] || { echo "fetch --keep: no checkpoint listing from $h"; exit 1; }`.
- In the md5 Python, also read `$F.list`. Fail when a listed path is in the first pod listing but missing from the
  second, or when the second listing is empty while the list is not. Print "listed N, compared M".

**Until it is fixed, check every fetch by hand before deleting the pod.**
- "stable files equal" + "differ" + "still being written" must equal `wc -l < LH/pod_stage/<alias>.fetch.list`.
- The "checkpoints fetched" kinds must show `base` = the pod's bases (c24: 2 after the pass; c4: 4 with the Nov 14
  bases).
- `run final` must equal the number of `arms/*/final.pt` on the pod. The `check` output or one `ls` shows it.

### W2. SHOULD FIX (cheap; before deletion): keep every GRPO snapshot, not only the last

- **The primary GRPO read** (Amendment 8 point 5) is the snapshot closest to +30, with iteration 40 beside it.
- **What `--keep` keeps:** one `_snap.pt` per line, the highest iteration.
- **What a later re-read loses.** The rows of every snapshot measurement are fetched, so the reading can be repeated.
  Any new measurement of the matched snapshot (a tagged read for the chain, a second set) would need the GRPO line
  rerun, 1–3 pod-hours.
- **Cost of keeping all four:** about 4 × 38 MB per line, under 0.5 GB for all lines.
- **Smallest change:** in the KEEPEOF block, keep every `…_it<N>_snap.pt` and keep the highest plain `_it<N>.pt` only
  when a line has no snapshot.

`*_alarm.pt` (GRPO rollback points) are neither fetched nor listed as left. That is harmless; I mention it only so the
"LEFT ON THE POD" count is not read as the full list.

**The rest of the keep list is complete.**
- **The main list** (every non-`.pt` file under bases, measure, raw, runs, logs, pod, pairs and arms, plus the anchor
  and KL-state files) holds:
  - measurement rows, raw rows and probe outputs;
  - `train.jsonl`, `metrics.jsonl` and `done.json`;
  - `lh_offline.json` and `full_*.json`;
  - the reward model's `probe.json`, its sidecars and `rm.pt.failed`;
  - `rm_bonus` rows, GRPO segment files, `not_run.json`, pair files, logs and pod markers.
- **The kept checkpoints:** the bases, each run's final, `rm.pt`, and every snapshot with an
  `arms/<run>/measure/step<k>` folder. That is the naming `full_main` writes.
- **Nothing else the pod writes holds results:** `LH/locks` and the Lean cache are transient.
- **Disk:** checkpoints are about 38 MB each. A pod has about 30–40 kept, about 1.5 GB, so all four pods take about
  6 GB of the 38 GB free now.

### W3. NOTE (Q1): the first base, F2 against F3

**The builder's test** compares, per first-base task:
- names, paths, estimates, hours and refusals;
- trainer argv, smoke argv and dependencies;
- the measure argv and the base and pair paths.

It does not cover the job plans.

**My wider comparison**, all identical in F2 and F3 for c24, c9, c4 and c0:
- prep2;
- stage2 (job name `stage2_<c>_s0`, its cap and the full task list);
- measure, pretrain seed 0, and every GRPO line's plan;
- the reward-model and reward-model-probe argv, and every rated run's `probe_argv`;
- `task_cmd`, `select_steps`, and the tagging rule;
- step timeouts.

**Code paths changed by ad001ea:**
- `split_base` returns `(c, 0)` for every first-base name, so `pair_files`, `trainer_argv` (KL states), `task_problems`,
  `full_main`'s tag condition, the FAILED path and `base_summary` reduce to the old code.
- The driver's stage-2 block is unchanged inside an `if`. It is skipped only when `BASE2=1` and DONE exists.
- In `_tasks`, ND_GUARD stays off for every task, so the second base's DPO runs under the same environment as the first.

**The readout.**
- Cases A–E: F2 and F3 give the same fields except the basis label.
- Every P1, P2 and P3 quantity is the same with and without second-base runs in the tree.

**The gate.**
- F2 and F3 write identical JSON on the seven first-base measurements.
- With fake `c24_s1` and `c4_s1` added (my links to the first bases' rows, in scratch only), no first-base entry and
  not the decision changes. The decision still checks G0 on c24_s0, c9_s0, c4_s0 and c0_s0.
- The second bases print "reported beside the first, gates nothing".

### W4. NOTE (Q2): the pass, item by item

| point of Amendment 8 point 2 | F3 | checked |
|---|---|---|
| base seed 1, pretrain_steps.json's seed-1 count | `pretrain --corpus C --seed 1` → `--steps 41800` (`scaled_steps(1)`), then the sanity read into `raw/C_s1_sanity` | plan printed |
| measured like any base | `measure --base C_s1 --skip_done` | argv identical to `C_s0`'s up to the base id (k 8, A/B/none, both sets, strict 106) |
| pairs balanced and capped the same way | `sample/build --base_seed 1` into `pairs/C_s1` | same functions and seed strings for split, balance and the 4,000 cap. `--no_probe` drops only the reward-model probe twins (no reward model on the second base). Both steps refuse to overwrite. |
| dpo + dposwap, trainer seed 0 | `base2_tasks`: `dpo_Cb1_reit_s0`, `dposwap_Cb1_reit_s0`, `full_dpo_Cb1_reit_s0` | DPO argv identical to the first base's up to base and pairs (lr, grid, 400 steps, `--swap-seed 0`, the corpus's KL states) |
| probes | `probe_argv` | stop rule from `measure/C_s1`'s base share |
| full measurement | `full_main` | bracket + last, untagged, marker_plain (builder test: 6 calls, tags "none") |
| no sft, GRPO, note rater or gate wait | `task_problems` refuses any other `b1` task, and the first-base job refuses `b1` tasks | no gate loop in the pass |

**Bounds** (job cap / driver wait):

| job | job cap | driver wait |
|---|---|---|
| pretrain + sanity | 3.0 + 0.5 h | 4 h |
| measure + strict 106 | 2.0 + 1.0 h | 3.5 h |
| prep2 (sample + build) | 1.5 + 2.0 h | 4.5 h |
| stage 2 (backstop; tasks 3 / 1 / 6 h, each under its own `timeout`) | 7.0 h | 8 h |

The whole driver runs under `timeout` (`TOTAL_H`, 24 h in RUNNER).

**Writes.** Every new output has a new name and never touches a first-base path:
- `bases/stage1_C_s1.*`, `measure/C_s1`, `raw/C_s1*`, `pairs/C_s1`;
- `arms/*Cb1*` and `arms/full_dpo_Cb1_reit_s0.json`;
- the jobs `*_C_s1`.

The pod markers are touched only in these ways:
- `C.FAILED` is appended to (`base2 …` lines).
- `C.GATE_READY` and the heartbeat are rewritten.
- `start` appends to `C.driver.log` (`>>`).

**Copying F3 onto a pod after DONE is safe.**
- `copy` pushes only:
  - `exp/`, `kd/` and `kg/`;
  - `lh/sets`;
  - the corpus and table files in `lh/data`, including `pretrain_steps.json`, now with `extra.c0n`; the seed-1 count is
    unchanged;
  - on c4, the Nov 14 bases.
  None of these is a path the pod writes, and rsync runs without `--delete`.
- `start` refuses while `C.driver.lock` exists. The old driver removes the lock on exit, so a stale lock can only make
  `start` refuse, never run twice.
- Even a copy made before DONE would be harmless, because the first-base Python is identical (W3).

**With DONE and GO present** (builder tests `test_driver_second_base_*`, 4 cases, pass). The rerun:
- skips pretrain and measure (outputs exist);
- passes the gate at once on GO, rewriting GATE_READY;
- does not enter stage 2;
- runs the pass, which ends with BASE2_DONE, also after a failure;
- if rerun again, skips everything.

A failed first-base task is not retried while DONE exists. That is documented in RUNNER.

### W5. NOTE (Q3): averaging and "unresolved (bases disagree)"

**What `base_seed_quantities` computes.** M(c24), M(c4), M(c24) − M(c4), S_before(c24) and S(c24), in two forms:
- **per base seed:** base 0 is the first bases' read, a mean over their two trainer seeds; base 1 is the second
  bases' read; each is taken against the one c0 base;
- **averaged:** weights ½ on each base's read in one contrast, with shared bootstrap weights.

**Against point 3.**
- **S(c24) is averaged too.** Point 3 names only M(c24), M(c4) and S_before(c24). But the table "is applied to the
  averages", and row 3 uses S(c24), so averaging it is required, not an extension.
- **The disagree test.** It uses the per-base values of M(c24) − M(c4) and S_before(c24): opposite signs, with either
  value outside ±5. It marks a row only if that row rests on the quantity:
  - row 1 rests on M(c24), D and S_before;
  - row 2 rests on M(c24), M(c4) and D;
  - row 3 rests on M(c24) and S(c24) only.
  Rows 1–2 are therefore exactly "the affected rows". Rows 1–2 need M(c24) ≥ 10 and row 3 needs M(c24) within ±10, so
  no row is wrongly replaced by another.
- **The text shown.** The row says "unresolved (bases disagree)" and names the average's row beside it. The
  first-base-only row is always printed as well.

**Synthetic second-base cases, mine, on F3** (beyond the builder's five):

| case | reading |
|---|---|
| R4 on c24b1: the rated run's step-100 measurement missing | "not read: c24b1: seed 0: bracket [50, 100]: no finished measurement of dpo_c24b1_reit_s0@100"; no average computed |
| R4 on the control: dposwap_c4b1's step 50 missing | "not read: c4b1: … dposwap_c4b1_reit_s0@50" |
| c4b1 moves the rated habit +15 only | "not read: c4b1: seed 0: +30 not reached; precondition +20 not met"; the average is computed but not used |
| c24b1 done, c4b1 pending | first base only, said so; c24b1 is not judged yet |

The "ended" state (BASE2_DONE without the full measurement) refuses the table and still prints the first-base row.
That is conservative and matches "valid reads only".

### W6. NOTE (Q5, judgment): c4's 2,354 balanced train pairs

**[obs] (coordinator).** 1,177 pairs per side, under the 4,000 cap, because only 19.9 % of sampled proofs took the A
form. At 64 pairs per step, one pass over the data is about 37 steps on c4 and about 62 on a capped corpus.

**[interp] The flag is enough for the primary read, for three reasons.**
- **The control sees the same data.** D is rated minus the swap control on the same pairs and batches. Repetition
  that is not tied to the preference (memorising proofs, drift) enters both arms and cancels.
- **The read is on unseen theorems.** It is taken at matched rated movement on `marker_plain`, not on training
  theorems.
- **Few passes at +30.** If +30 is crossed in the first ~100 steps, c4 has made at most about 2.7 passes and a capped
  corpus about 1.6.

**What could remain: a bias that favours the hypothesis.** With more repetition, DPO may reach +30 through pair-specific
edits that spill less into other habits. That would lower M(c4) and raise M(c24) − M(c4).

**Smallest addition, in the readout only.** Print, per corpus and base:
- the train-pair count;
- the passes at the +30 crossing (crossing step × 64 / train pairs).

Read it with this pre-set rule: if c4 crosses after 2 passes while c24 crosses within 1, M(c24) − M(c4) is reported as
possibly confounded with repetition.

**Only then, a matched control.** `dpo` + `dposwap` s0 on a seeded 1,177-per-side subsample of c24's train pairs,
about 1 pod-hour on the c24 pod after its pass. Declare it in an amendment before any D is read.
- Re-capping every corpus is no longer possible, because c4's stage 2 has started.
- The second base c4b1 will have its own count. Report it the same way.

### W7. NOTE: operational points

- **A fetch while the pass runs leaves files uncompared.** Files still being written are named, not compared. The
  fetch before deletion must therefore come after BASE2_DONE (c24, c4) or DONE (c9, c0) and show "0 still being
  written" apart from heartbeats.
- **BASE2_DONE can come before a job ends.** If a driver wait expires, BASE2_DONE is written while that job may still
  run under its own cap. `check` shows the process.
- **GATE_READY changes on the rerun.** Its time changes; this is not a new gate.
- **Two aborts give no message** (W1 table): a file removed on serrano after rsync (rc 123), and a pod file that
  vanishes between listing and md5 (ssh rc 1). Both fail safe. A message would help the operator.
- **Use F3 for the copy.** As in V1, `copy` stages from its own folder. Run `bash $F3/lh_pod.sh copy …`, not from the
  working folder, which keeps changing.

### Replies (part 2d)


### Round 1 on part 2d, orchestrator (agent:claude, Fable; 2026-11-27 5:31 am PT)

- **W1** accepted; the builder is making every listing fail loudly (sentinel line, full coverage of the fetch list,
  bases and finals required). Whatever the script says, I compare md5 sums of the fetched tree against the pod myself
  before any pod is deleted (as I did for the metrics pod tonight).
- **W2** accepted: every GRPO snapshot is kept.
- **W6** accepted with the auditor's contingency, written into `PREREG.md` Amendment 9 before any DPO run started.
  The counts turned out close (2,292 to 2,966 on three corpora), so I expect it not to trigger.
- **W7** passed to the operator: the pre-deletion fetch comes after DONE (or BASE2_DONE) with nothing but heartbeats
  still being written, and `copy` is run from the frozen checkout.
- W3, W4, W5 noted: no change.

## Part 3: reading of the first-base results

Read 2026-11-27, 8:28–9:10 am PT (independent auditor, Opus). Read-only, CPU only, `nice -n 5`, at most 2 processes.

**Inputs:**
- `LH/read/read_firstbase.{txt,json}` (frozen code at 12fc233, B 1000);
- the measurement rows under `LH/measure`, `LH/arms/*/measure/step*` and `LH/runs/*/seg1/authors/it00*`;
- the Lean-rejected proofs `lean_gate.leanrej.jsonl` and the Lean-gate summaries;
- the GRPO training samples `LH/runs/rl_c24_*/seg1/lean_dump.jsonl` (about 123,000 per line, read with F4's `lh_profile`);
- the reward-model probe `arms/rm_c24_reit_s0/probe.json`, `rm_bonus.jsonl`, `metrics.jsonl`, `ops_log.md`.

**Not available on serrano:** `rm.pt` and every GRPO checkpoint (only `*.upload.json`). So the reward model could not be
scored again here. Scripts and outputs are in `.../scratchpad/audit2b/p3/`.

Notation: [obs] is a number read from the files, [interp] is my reading of it. "A share" is the share of a habit's
sites written in author A's form, on accepted proofs, on `marker_plain`, untagged.

### Verdict (part 3)

1. **(a) is read correctly and validly, and it is a null.**
   - Brackets, interpolation weights, preconditions, both seeds and the c0 subtraction all reproduce from the rows.
   - Acceptance loss does not change the answer:
     - On the theorems where every arm has accepted proofs: M(c24) +2.0, M(c4) +4.5.
     - Lean-rejected proofs carry the same habit mix as accepted ones, within ±7 points.
2. **S_after +11.3 with S_before +1.5.** Habits written after the edited line changed and habits written before it
   did not.
   - It licenses "the edit changes what comes after it on the page"; it does not license "the author was changed".
   - It comes mostly from restate. On c0, restate is pushed 28 points toward B by the reiteration edit, and on c24
     only 7 points.
3. **(d) is real on one seed, and it is not a reward-model artefact.** It is half an artefact of the comparison, and its
   mechanism is visible in the samples.
   - The reward line moved the model's own mixture of whole-author proofs toward A: proofs with all-A notes went from
     49 % to 86–94 %.
   - The anchor line moved the same mixture toward B (48 % to 24 %).
   - The reward model does not pay for notes.
   - D +62 is the sum of the two drifts.
4. **The plain statement for the PI** is in X4. The data permit "DPO on twin pairs isolates one habit; on-policy RL
   with a proof-level reward selects among the model's own authors" and show the selection directly on c24. They do
   not yet establish it as a contrast between the two methods.
5. **Before any further run:** fetch `rm.pt` and every GRPO snapshot from the c24 pod. If one run, run the RL-seed-1
   pair of c24's two GRPO lines first, with the reading rule in X5.
6. **Six things in the read-out mislead as printed** (X6). The most important: the readings table's "edited one by one"
   hides S_after; "last" for DPO is the upper bracket, not step 400; the GRPO chain "difference" column.

### X1. Q1: (a) at +30, and the acceptance loss

**Read correctly [obs].** I recomputed D from the rows with the read-out's weights and got the same numbers:
- M(c24) +3.4, M(c9) +1.1, M(c4) +4.0;
- c24 seed 0: order +8.6, conj +18.1.

| corpus | bracket | w (s0, s1) | crossing step | passes | KL rated / control at the upper bracket |
|---|---|---|---|---|---|
| c24 | 15, 20 | 0.35, 0.45 | 16.8, 17.3 | 0.36, 0.37 | 0.015 / 0.0009 |
| c9 | 10, 15 | 0.69, 0.58 | 13.5, 12.9 | 0.29, 0.28 | 0.009 / 0.0003 |
| c4 | 10, 15 | 0.73, 0.59 | 13.7, 13.0 | 0.37, 0.35 | 0.008 / 0.0019 |
| c0 | 15, 20 | 0.33, 0.53 | 16.7, 17.6 | 0.47, 0.49 | 0.015 / 0.0003 |

- Both bracket snapshots of every run are fully measured, and every precondition holds.
- Per-seed M(c24) is +5.1 / +1.8 and M(c4) +4.2 / +3.9: same sign, no disagreement.
- The controls barely moved at the crossing (KL 0.0003–0.002), so D is nearly the rated run against its base.
- Every corpus crossed +30 within half a pass over its train pairs. The pair-count concern of W6 is therefore moot;
  every corpus is below the cap.

**c0 as the baseline.** c0 is what the reiteration edit does with no links. That is large:
- conj +14.8 and restate −27.8 on c0, with similar values on every corpus;
- the edit has structural side effects on the same proof, and subtracting c0 removes them.

This is the pre-registered contrast. One caveat: base rates differ (c0 conj 44 % A against c24 37 %).

**Acceptance loss [obs].** At the upper bracket the rated runs accept 27–41 % against 44–54 % for their controls.
- The extra failures are mostly ill-formed proofs: parse failures 1,792 against the control's 867 on c24 s1 step 20
  (mostly "unbound" references). Lean rejections are 1,007 against 1,313.

**Can selective survival fake or hide movement? Not at the size the table reads (±10). Three checks:**

| check | M(c24) | M(c4) | M(c9) | notes pooled, c24 − c0 |
|---|---:|---:|---:|---:|
| all 500 theorems (the read-out) | +3.4 | +4.0 | +1.1 | +1.6 |
| theorems with ≥1 accepted proof in rated and control at both bracket steps (289–360 per run) | +2.0 | +4.5 | +2.1 | +1.4 |

Lean-rejected against accepted proofs, rated run, upper bracket (A share, points):

| run | reit | order | conj | restate | notes |
|---|---|---|---|---|---|
| dpo_c24_reit_s1 | 66.6 / 72.1 | 42.0 / 49.5 | 71.7 / 64.8 | 74.1 / 69.6 | 66.0 / 62.9 |
| dpo_c0_reit_s1 | 60.7 / 55.4 | 28.9 / 30.4 | 61.4 / 51.1 | 31.1 / 30.1 | 47.6 / 46.1 |

- The rejected proofs are about a quarter of attempts and differ from the accepted ones by 2–10 points. Re-including
  them would move a share by about 1–2 points, in the same direction on c24 and c0.
- **Not checkable:** the parse failures, about 45 % of the rated run's attempts, cannot be profiled. Their text is not
  stored.
- The c0 subtraction removes survival effects common to all corpora. c0's rated run loses as much acceptance as c24's
  (26.9 against 30.7 %).

[interp] The DPO null stands. One caveat for the write-up: by the crossing the rated DPO model is already breaking
(twice the parse failures), and by step 400 it accepts nothing. The "+30" model is a local edit that damages proofs,
not a smooth change of style.

### X2. Q2: S_after(c24) +11.3 with S_before(c24) +1.5

**[obs] The pieces** (D at +30, c24 against c0):

| habit | c24 | c0 | c24 − c0 |
|---|---:|---:|---:|
| restate | −7.2 | −27.8 | +20.6 |
| halves | +3.1 | −5.9 | +9.0 |
| conj | +19.1 | +14.8 | +4.3 |
| disjunct | thin | thin | — |

The mean is +11.3. The positional version, all linked habits' sites after the first reiteration site, is +11.9.

**Restate drives it.** The edit pushes restate toward B on every corpus: c0 −28, c4 −20, c9 −18, c24 −7.
- On c24, restate given reiteration-A is already 97.8 % A in the base (76.2 % given reiteration-B).
- On c0 the same figures are 71.0 % and 66.1 %.

**What it licenses [interp].**
- **What changed, and what did not.** The habits decided before the edited line (the notes and order: which author
  the proof starts as) did not move. Habits after it moved toward the edited line's author, or were pushed away from it
  less.
- **The page reading.** That is what a model does when it reads the author off the page: later choices condition on
  an A-looking line, as the c24 base's within-proof links say they should. No author is held.
- **The text supports it but does not prove it.** "Rewarding one habit moves the others" holds only downstream of the
  edit, through the text.

**The alternative reading [interp].** Corpus-specific structure or ceilings. The edit's structural push on restate is
smaller where restate given reiteration-A is near 100 % (c24).
- **Size against the base links.** The base's links predict about +5 for restate from a +30 reiteration change:
  0.30 × (21.6 − 4.9). The observed +20.6 is larger, so a pure page-conditioning model with the base's site-level
  links undershoots.
- **The corpus ordering.** c24 (−7) < c9 (−18) ≈ c4 (−20) < c0 (−28). Restate is linked in c9 too, yet c9 behaves
  like c4.
- **What separates the readings.** Conditioning on a written A line in the base, with no tuning (a forced-prefix
  test), would separate the two. It is not in these data.

### X3. Q3: what happened on c24 in the reward model + GRPO arm

**[obs] Both lines moved, in opposite directions** (A share, marker_plain, untagged):

| | base | anchor it10 | anchor it40 | reward it10 | reward it40 |
|---|---:|---:|---:|---:|---:|
| reit | 32.7 | 17.2 | 16.3 | 49.5 | 61.4 |
| order | 41.6 | 17.8 | 13.2 | 31.5 | 26.8 |
| restate | 78.2 | 56.5 | 53.5 | 84.0 | 82.4 |
| note_atom | 67.2 | 21.8 | 29.6 | 87.9 | 91.5 |
| note_~a | 54.8 | 19.8 | 19.7 | 74.1 | 95.4 |
| note_&ac | 53.6 | 17.0 | 8.3 | 72.9 | 88.2 |
| note_vaa | 61.7 | 36.8 | 20.3 | 83.8 | 95.0 |
| notes before the first reiteration site, pooled | 58.7 | 29.2 | 24.3 | 79.9 | 89.5 |
| accepted % | 45.7 | 73.8 | 81.5 | 74.5 | 80.8 |

- Against the base, the reward line moved the note classes +10 to +41 (note_atom +24) and the anchor line −14 to −45
  (note_atom −38). D = reward − anchor (note_atom +61.9) is the sum of the two.
- **c4 and c0, the same two lines.** Their notes do not move coherently: per class ±15, both signs, both lines.

**It is not an artefact of note sites or of coherence [obs].**
- Notes per proof: 1.98 base, 1.98 anchor, 1.97 reward (it40).
- notes|notes linkage stays 89–90 in both lines (97 in the base).
- natural|natural falls from 44 to 31 in the anchor line and to 16 in the reward line: the natural habits come apart
  under RL; the notes do not.

**The reward model does not pay for notes [obs].** `probe.json`, held-out twins that differ in one habit only, bonus =
0.3 × sigmoid difference:
- **The reiteration reference:** about 0.257. The ±5 % band is ±0.0129.
- **Every note class:** −0.008 to +0.0014.
  - note_atom, n 216: −0.0020 [−0.0031, −0.0011], "within band".
  - The other 14 classes have n 12–53 each ("not probed" only because n < 100). Pooled, that is 655 note twins, all
    within ±0.008.
  - Twelve of 15 classes have a_higher_share < 0.5: if anything, the model pays B notes slightly more.
- **What it does pay for, besides reiteration:**
  - **Restate's B form** (−0.078, n 85, about 30 % of the reference). Yet the reward line ends 28.9 points more A on
    restate than the anchor line, against the reward model's own preference.
  - **Longer proofs** (`probe:length` +0.041; the reward line writes 0.9 more lines than the anchor line).

**The mixture of authors shifted, in whole proofs [obs].** I read every GRPO training sample of both lines with
`lh_profile` and split each line into 40 chunks in file order, about one iteration each.
- Proofs with notes are almost never mixed: 0.3–3.3 % have both A and B notes. Each sampled proof has one note-author.
- **Reward line:** proofs with all-A notes go 49 % (chunk 1) → 66 % (5) → 86 % (10) → 94 % (20) → 86 % (40).
- **Anchor line:** 48 % → 32 % (5) → 24 % (40).
- **How tightly reiteration rides on the author.** Proofs whose reiterations are only A-form are about 54 % of all-A-note
  proofs and about 2 % of all-B-note proofs (chunk 1). Rewarding A-form reiteration is therefore almost the same as
  rewarding A-author proofs.

**[interp] Mechanism.** On c24 the base samples whole-author proofs. A proof-level reward that pays for A-form
reiteration reinforces A-author proofs, and their notes come with them.
- This is selection among the model's own authors.
- It needs neither a reward for notes nor a held persona, beyond the base's own coherence.

**The anchor line is not an accident of its seed [obs].** Acceptance-only GRPO pushes order toward B on every corpus:
c24 42 → 13, c4 38 → 20, c0 23 → 12 (reward lines too).
- **Where notes ride on order:** on c24 (P(note A | order A) 93.7 against 28.7 given order B in the base). On c4
  (52.3 / 52.5) and c0 (44.1 / 47.8) they do not.
- **In the anchor line's samples,** order A falls 25 → 5 % and all-A-note proofs 48 → 32 % within the first five
  iterations.

[interp] The anchor line is therefore itself a linked-habit result: selection on order (a natural habit Lean favours in
B form) carries the 15 note classes on c24 only. The seed fixes how far, not whether. Only a second seed can show that.

**The chain line in plain words** (`lh_read.Reader.chain`). For each habit decided before the first reiteration site:
- **Predicted:** q × P_base(h A | first reiteration A) + (1 − q) × P_base(h A | first reiteration B). Here q is the
  line's own share of those sites in proofs whose first reiteration is A.
- **Printed:** observed − predicted.
- **note_atom.**
  - The reward line has 8.1 points more A notes than if it had only changed how often proofs start with A-form
    reiteration, keeping the base's link.
  - The anchor line has 32.9 points fewer.
  - +41.0 is their difference. It tests nothing by itself, because the anchor term is the anchor's own drift.
- **The conditionals moved.** Pooled over notes, base P(note A | reiteration A / B) = 90 / 37. Reward line it40:
  98 / 58. Anchor line: 58 / 14. The notes moved within reiteration strata: the author moved, not just the
  reiteration form.

**Why order moves only +13.5 while the notes move +62 [interp].**
- Order is pinned by acceptance toward B in both lines (reward line 42 → 27), notes are under no acceptance pressure,
  and order's tie to the author is weaker (natural|natural 44, notes|natural 58, notes|notes 97).
- So when the author mixture moves, the notes move almost one for one, while order is held back by Lean.

**What one seed on one base allows:**

| can be claimed | cannot be claimed |
|---|---|
| on this base, GRPO moved 15 note classes together, toward A with the reiteration reward and toward B without it | the size of D (+62) or of either drift as a property of c24 |
| the reward model does not pay for notes | robustness across seeds or bases |
| the sampled proofs are single-author, and their mixture moved | that GRPO "holds a persona" in any sense beyond reweighting the base's coherent authors |
| c4 and c0 show nothing comparable | a GRPO read at +30: the c24 line reached 28.7 at it40, so every c24 GRPO number is "NOT a valid +30 read" by the read-out's own rule |

### X4. Q4: the plain statement for the PI

[interp, for the PI's question] "With 4 links, rewarding one habit moved nothing else; with 24 links, does it?"

1. **Offline, on twin pairs** (DPO; the fine-tune arm agrees): no.
   - Moving the reiteration habit by +30 points moved the other habits no more with 24 links than with 4 or 0:
     M(c24) +3.4 [−1.7, +8.9], M(c24) − M(c4) −0.6.
   - Habits written before the edited line stayed put (S_before +1.5).
   - Habits written after it followed the edited line on the page (+11).
   - That is the Nov 18 pattern: the edit is local, and the rest of the text adapts to it.
2. **On-policy RL with a proof-level reward for the one habit: on the 24-link corpus, yes**, at one seed and one base.
   - All 15 note habits moved with it (+10 to +41 points against the base).
   - In the RL control without the reward they moved the other way, with a natural habit.
   - On the 4- and 0-link corpora they did not move.
   - The reward model does not reward notes. The movement came from which whole proofs, each written by one author,
     were reinforced.
3. **The contrast "twin DPO isolates a habit; proof-level RL selects among the model's own authors"** is permitted by
   the data. On c24 the selection half is shown directly in the samples. It is not yet established as a contrast between
   methods:
   - one seed and one base;
   - the GRPO line did not reach +30;
   - the two arms differ in more than the selection mechanism (on-policy sampling, the amount of change: GRPO took RL-pool
     pass@1 from 23 to 84 %, while the rated DPO model was breaking);
   - the twin pairs isolate the habit by construction.
4. **What "persona" can mean here.** The base already writes each proof as one author on c24. RL shifts how often each
   author is written. Nothing here shows a model that "holds" a persona beyond that.

### X5. Q5: the smallest additional run

**Before anything: fetch `rm.pt` and all four `_snap.pt` of every GRPO line from the c24 pod.**
- This is W2. `fetch --keep` keeps only the last snapshot of each line, and serrano has none of them.
- The reward-model probe can then be repeated at scale on CPU: pooled note twins, plus whole-proof author swaps.

**Ranking:**
1. **First: rl_c24_anc_s1 and rl_c24_rlhfreit_s1.** RL seed 1, the same base and reward model, 40 iterations, the same
   snapshot measurements, about 1 GPU-hour each, on the live pod. This is the only run that makes (d) more than one seed.
2. **Second: both lines on c24_s1.** A stronger replication, but that base has no reward model (Amendment 8). Training
   one on its pairs, or reusing c24_s0's, is a design decision to write down first.
3. **Third: c9's lines.** They answer a different question (natural links against note links), need a new pod and a
   reward model, and belong to a discussion with the PI.

Running (1) before the PI sees this is reasonable under the 30-minute rule, because it is cheap and the pod is up. It
is a replication, not a new question.

**Write into PREREG before it runs:**
- **Quantities.** For each line against the base: the pooled note A share (all note sites, and the sites before the
  first reiteration site), order and reiteration, at it40 and at the iteration closest to +30. Plus D = reward −
  anchor, and the share of all-A-note proofs in the training samples per chunk.
- **Predictions from seed 0:**
  - the anchor line moves notes at least 20 points toward B;
  - the reward line moves notes at least 15 points toward A;
  - D(notes) is at least +30;
  - order moves toward B in both lines;
  - notes|notes stays at least 80.
- **Reading rule.**
  - (d) replicates if both lines move the notes in the seed-0 directions, with 95 % intervals excluding 0, and D(notes)
    is above +20 with its interval above 0.
  - If the anchor line does not drift, report that seed 0's D was enlarged by the anchor's drift, and lead with
    "reward line against base".
  - Either way, the headline for "rewarding one habit moves others" is the reward line against the base. D is
    secondary.

### X6. Q6: what in the read-out is wrong or misleading as printed

1. **The readings table's row hides S_after.** "Habits edited one by one, as on Nov 18" is printed while S_after(c24)
   is +11.3 [+7.3, +15.4] and S_sites_after +11.9. Row 3 tests S(c24) pooled over all 15 linked habits (+3.5), which
   dilutes it. Add "(later habits follow the edited line: S_after +11.3)" or print S_before and S_after beside the row.
2. **"last" for the DPO and atom arms is not step 400.**
   - Step 400 has no accepted proofs, so "last" is the upper bracket: step 20 on c24 and c0, step 15 on c9 and c4.
   - The `PREREG [dpo:reit:last]` line (M(c24) +8.1) therefore reads like an end-of-training result. It is not.
   - Label it "upper bracket (step 400 empty: 0 of 4,000 accepted)" and print the collapse.
3. **The GRPO block.**
   - c24's "at_target" and "last" are the same snapshot (it40), printed twice.
   - `PREREG [rlhf:reit:at_target]` prints S_before(c24) +57.0 with no "not a valid +30 read" beside it. The validity
     line exists only in the per-corpus block.
   - "reached False" for c4 means "the closest snapshot is below +30". c4's line passed +30 at it20 (33.8).
   - The chain's third column ("difference") invites the reading "+41 caused by the reward". Print reward line − base
     and anchor line − base instead, as in X3.
4. **The validity line for GRPO c0** ("not bracketed by two measured snapshots"). This is part 2c's V6 point: the
   bracket test does not apply to GRPO, which is read at the closest iteration.
5. **The reward-model probe's "not probed" for 14 note classes** hides an informative result: 655 note twins, all
   bonus differences within ±0.008 against a reference of 0.257. Print a pooled "notes" row.
6. **The passes table and the pair caps are correct** (recomputed: e.g. c24 s0 64 × 16.75 / 2,966 = 0.361). Every
   corpus is below the cap (c24 2,966; c0 2,292), so "FEWER PAIRS THAN THE CAP" is printed for all of them. That is
   true but no longer a flag on c4 alone.

**(c) and (e), briefly.**
- **(c)** The fine-tune arm's S_before(c24) +4.5 [+1.5, +7.4] is small and below the table's 10-point bar, and its
  M(c24) is not different from 0. It agrees with DPO.
- **(e)** The note rater's late drift of the swapped control (step 400) does not touch the +30 read, which is taken at
  steps 10–15 on c24 and 5–10 on c0; the read-out's drift lists at target are empty. P3_S_notes(c24) +4.4
  [+0.3, +8.3] is marginal.

### Replies (part 3)

