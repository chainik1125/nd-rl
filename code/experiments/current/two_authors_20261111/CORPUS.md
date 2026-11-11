---
written_on: 2026-11-11
written_by: agent:claude (corpus build agent), for Leon
---

# Two-author corpora: what was built, and what it shows about the habits

This note covers the build of the four pretraining corpora of `SPEC.md` (sections "The two authors", "Author share",
"Control corpus") and their manifests. Everything ran on CPU on serrano, pinned to 8 cores. No GPU job ran and no
model was trained or sampled.

Labels: **[obs]** observed in this build's outputs; **[interp]** my reading of them; **[guess]** not tested.

## Summary

- **[obs] The build passed every check.**
  - There are four corpora of 155,000 records each.
  - No record was dropped or duplicated, no statement changed, and no premise line was touched.
  - Lean accepted all 122,729 distinct rewritten proofs, with 0 rejections and 0 fallbacks.
  - All 1,550 route records are under author A in both authors corpora, and they are still σ⁺ by Dan's `knock`.
  - Dan's σ and σ⁺ hold for exactly the same records as in the source, in all four corpora.
- **[obs] Step order works completely.**
  - 30.2 % of records have room.
  - Author A is in the premise-driven form at 100 % of them, and author B at 0 %.
- **[obs] Conjunct order is a small habit, and mostly decided by step order.**
  - Only 2.4 % of records have a *free* conjunct site, meaning one where step order does not already fix which conjunct comes first.
  - At free sites, A writes right-first 100 % of the time and B 0.2 %. The B exceptions are records holding two sites that need opposite orders: 4 in `half`, 6 in `p10`.
  - Counted over all sites, the shares are 72.7 % for A and 19.4 % for B, because step order decides the rest.
- **[obs] Reiteration is cut down by the 12-line cap.**
  - A's per-site share is 55.8 % (`half`) and 54.9 % (`p10`).
  - By source length, it is 96 % or more for proofs of 4 to 8 lines, 82 % at 9, 56 % at 10, 11 % at 11 and 0.7 % at 12.
  - 29 % of A's records with room could not take the full form.
  - On the route records themselves the share is 50.1 %, because route proofs are long (mean 8.8 source lines).
- **[obs] The controls match the authors corpora's shares and do not bundle.**
  - Per-site A-form shares (realized against target):
    - `half`: order 51.1 / 51.2 %, conjunct (free sites) 44.7 / 44.1 %, reiteration 27.4 / 27.2 %;
    - `p10`: order 10.5 / 10.6 %, conjunct 8.1 / 8.1 %, reiteration 5.3 / 5.3 %.
  - Every pairwise association among habit forms, tag and route has |φ| ≤ 0.035. The exceptions are conjunct counted over *all* sites: against step order φ −0.16 (n 2,687), and against the route −0.065 (n 5,961), both in `control_half`. Step order fixes the non-free sites.
- **[interp] Of the three markers, step order is the clean one.**
  - Reiteration carries A's mark only on proofs short enough to take it.
  - Conjunct order is rare, and it is only a marker where it is free.

## Files

In `/scratch/cowifornia/nd-rl/two_authors/data/` (TA/data):

| file | sha256 | records | rewritten records |
|---|---|---:|---:|
| `authors_half.jsonl.gz` | `0912b5fc23dc20ed61961cc2156efe8ba10ab0b31ce88aad8c3c0ee251baed55` | 155,000 | 69,747 |
| `authors_p10.jsonl.gz` | `390c2238a100eb9f97ff00ca2a0e9c6d56d24c0e2fae38cab2f26060a67d98d1` | 155,000 | 54,933 |
| `control_half.jsonl.gz` | `7cf6008be1c2d02e2f33c0af9515acb90e8725000f03e544a838b1a2b5ace9c1` | 155,000 | 66,867 |
| `control_p10.jsonl.gz` | `077244a4e99b1699c3bb3b7b5fafad7b56b77995f9623620ac77a3205c6368c1` | 155,000 | 53,246 |

Alongside the corpora:
- **Manifests.** Each corpus has `<name>.manifest.json` beside it.
- **Build summary.** `authors_build.json` holds the timings, the calibration, the state comparison of pitfall 3 and the per-combination totals of pitfall 1.
- **Lean cache.** Every verdict is in `TA/cache/lean_verdicts.jsonl`: 122,729 entries, all accepted.
- **Logs.** `TA/logs/authors_corpus_*.log`.
- **Source.** `knockout_grpo/dan_data/train_k12dose1.jsonl.gz` (`tag_core.SRC_CORPUS`, sha256 `7d74e50e…`).
- **Code and tests.** This folder: `authors_corpus.py` and `test_authors_corpus.py`. Nothing under `../author_tags_20261110/` was edited.

## How a proof is rewritten

Each source proof is parsed into the tree of `habits.py`, in which every box hangs on the line that discharges it. The
habits are then applied in a fixed order:

| stage | author A | author B | built from habits.py |
|---|---|---|---|
| 1 strip | every reiteration written with R is removed, and its citers cite the source | the same | new (`strip_reit`): removes exactly the R lines that make a `reit_sites` site "reiterated" (an R line inside a box, not the box's last line, whose source is outside the box) |
| 2 order | premise-driven normal form | goal-driven normal form, sinking a line only while box depth stays ≤ 2 | `normal_form`; B's goes through `sink_all_capped` (see below) |
| 3 conjunct | right conjunct first at every site | left conjunct first | `conj_sites`, and `topo` with the sites as extra edges (`v_conjunct` swaps only the first site). A site is changed only if the change keeps the proof's step-order form; otherwise it is counted as a conflict |
| 4 reit | each outer line a box uses at its own level is restated once by R after the box's assumption | nothing: stage 1 already left direct citations | `v_reit`, variant "reit" |

**Checks after every stage.** A stage that fails a check is skipped, its habit keeps its previous form, and the
reason goes into the record and the manifest. The checks are:
- at most 12 lines;
- box depth at most 2;
- the premise lines unchanged (at most 3 premises);
- `nd_verify`;
- Dan's `state_env.decompose(canon=True)`, with every state and action encoded by the `lean_staten` tokenizer (this
  is the name limit: n1..n64, h1..h8);
- Dan's σ⁺ (`coe_assemble.knock`) and σ (`coe_sigma.has_sigma`) unchanged from the source.

**Lean.** Every final proof that differs from its source goes to Lean 4 core through `nd2lean.translate` and
`nd2lean.lean_check`, as `habits.py` does, in batches of 200 theorems per Lean process. On a rejection, the last
applied stage would be dropped and the record rebuilt and rechecked. No rejection occurred.

**Departures from `habits.py`.** Each is implemented in `authors_corpus.py`, not patched into `habits.py`.

1. **The depth-capped goal-driven sink (`sink_all_capped`).** `habits.sink_all` can move a box unit into a box at
   depth 1, which puts that unit's own box at depth 3, past K12's cap.
   - [obs] On the first 2,000 records this happened to 60 records (10 % of those with order room), and the skip left
     them in the wrong or neither form.
   - The capped copy makes the same moves wherever none would hit the cap.
   - The order detector uses the same capped normal form.
2. **B's reiteration form removes only the reiterating R lines.** `v_reit` "rfree" removes every optional R line,
   including top-level restatements (3,630 in K12, per HABITS.md), which are not part of the reiteration habit. Those lines are kept
   here, in both authors' proofs.
3. **Every conjunct site is set, not just the first.** A site that conflicts with the step-order form, or with
   another site, is left as it is and counted.
4. **The reiteration rewrite is wrapped, not changed.** It still redirects a box's required last R line to the new
   copy, which is valid. It is skipped when it would make Dan's σ⁺ fire (see the doubts below).

**Detectors** (`detect`). These give every number in the manifests and the tests, and they are what a measurement on
model output should use:
- reiteration per site, on the proof as written (`habits.reit_sites`);
- step order and conjunct sites on the proof with its reiteration stripped;
- a conjunct site counts as **free** when swapping it alone leaves the step-order form unchanged.

## Record format

Each record keeps the source's fields. When the proof changed, `proof`, `text`, `n_lines`, `rules` and `pat` are
recomputed (Dan's `patterns.classify`, which reproduces every source `pat`). The build adds these fields:

| field | value |
|---|---|
| `author` | `A` or `B`. In a control this is a nominal label, drawn independently of everything else, that only picks the tag |
| `tag_shown` | whether the tag token is in this record's states. True iff the first byte of sha256(`20261110:show:<key>`) is even; the same records in all four corpora |
| `author_tag` | the token the states carry: `"A"`, `"B"`, or `"none"` when the tag is hidden |
| `forms` | the form assigned to each habit: `{"order", "conjunct", "reit"}` → `A` or `B` |
| `realized` | the detectors' reading of the final proof: `order` is A, B, none (no room) or neither; `conjunct` is [sites, right-first sites, free sites, free right-first sites]; `reit` is [sites, reiterated sites] |
| `skipped` | present only when a rewrite was not applied: habit → reason (`lines>12`, `route_detector`, `sites need opposite orders`, `lean`, …) |
| `route` | σ⁺ by Dan's `knock`; equals `coe_dose` |
| `src_proof` | the source proof |
| `ta_corpus` | the corpus name |

**How authors and forms are drawn.**
- **`authors_half`.** The author is `tag_core.corpus_tag`: route records are A, and any other record is A iff the
  first byte of sha256(`20261110:<key>`) is even.
- **`authors_p10`.** The author is `tagcorpus.route_share_tags(p = 0.10)`: the route records plus the 13,950 other
  records with the smallest digest are A.
- **Controls.** Habit h is in A-form iff u(seed, `ctrl`, h, key) < q_h, and the label is A iff
  u(seed, `ctrl`, `author`, key) is below the authors corpus's share of A. u is the first 8 bytes of sha256 divided by
  2⁶⁴.
- **Calibration of q.** Each q_h is set by bisection over all eight form combinations, so that the expected per-site
  A-form share equals the authors corpus's realized share. Step order is counted per record with room, conjunct per
  free site, reiteration per site.

## The corpora

[obs] Shares are of the sites with room, after rewriting. Conjunct shares are over **free** sites, with all sites in
brackets.

| | authors_half | authors_p10 | control_half | control_p10 |
|---|---|---|---|---|
| records A / B | 78,188 / 76,812 | 15,500 / 139,500 | 78,357 / 76,643 (label) | 15,544 / 139,456 (label) |
| tag shown A / B | 39,290 / 38,350 (50.3 % / 49.9 %) | 7,882 / 69,758 (50.9 % / 50.0 %) | 39,252 / 38,388 | 7,681 / 69,959 |
| route share A / B | 1.98 % / 0 | 10.0 % / 0 | 0.96 % / 1.04 % | 0.85 % / 1.02 % |
| order: records with room A / B | 24,002 / 22,864 | 4,979 / 41,887 | 23,594 / 23,272 | 4,736 / 42,130 |
| order A-form, A / B | 100 % / 0 % | 100 % / 0 % | 51.1 % / 51.1 % | 10.7 % / 10.4 % |
| conjunct: free sites A / B | 1,704 / 2,167 | 344 / 3,965 | 1,982 / 1,970 | 441 / 4,034 |
| conjunct A-form, A / B | 100 % (72.7 %) / 0.18 % (19.4 %) | 100 % (59.0 %) / 0.15 % (19.1 %) | 42.8 % / 46.6 % | 9.8 % / 8.0 % |
| reiteration: sites A / B | 111,274 / 116,658 | 22,686 / 212,303 | 114,907 / 112,729 | 23,561 / 210,759 |
| reiteration A-form, A / B | 55.8 % / 0 % | 54.9 % / 0 % | 27.5 % / 27.2 % | 5.2 % / 5.3 % |
| skipped, reiteration | A: lines>12 15,071, route detector 638 | A: lines>12 3,183, route detector 125 | lines>12 15,629, route detector 808 | lines>12 3,137, route detector 198 |
| skipped, conjunct (sites need opposite orders) | B: 4 | B: 6 | 5 | 5 |
| skipped, order; Lean fallbacks | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| mean lines A / B (source) | 7.65 / 6.81 (7.02 / 6.98) | 7.81 / 6.82 (7.16 / 6.98) | 7.24 / 7.23 | 6.91 / 6.92 |
| bundle: records with ≥ 2 habits defined | A: 13,159 all A, 10,325 mixed; B: 22,470 all B | A: 2,694 all A, 2,215 mixed; B: 41,141 all B | A label: 3,362 all A, 8,371 all B, 11,393 mixed | A label: 33 all A, 3,944 all B, 672 mixed |

**[obs] Control associations.** These are at record level: 1 when the record is in A-form at every site, 0 when in
B-form at every site, and undefined when it has no room or is mixed. Each cell gives φ, with n in brackets.

| pair | control_half | control_p10 | authors_half (for contrast) |
|---|---|---|---|
| order ~ conjunct (free) | 0.005 (788) | −0.014 (1,299) | 1.0 (713) |
| order ~ reit | 0.035 (45,687) | 0.008 (45,789) | 0.62 (45,696) |
| conjunct ~ reit | 0.009 (967) | −0.028 (1,463) | 0.75 (897) |
| order / conjunct / reit ~ author label | −0.0003 / −0.035 / 0.003 | 0.002 / 0.022 / −0.003 | 1.0 / 1.0 / 0.73 |
| order / conjunct / reit ~ tag token (shown records) | −0.004 / −0.007 / 0.001 | 0.007 / 0.023 / −0.003 | 1.0 / 1.0 / 0.73 |
| order / conjunct / reit ~ route | 0.006 / −0.008 / 0.002 | −0.001 / −0.006 / −0.001 | 0.14 / 0.08 / 0.07 |
| author label ~ route | −0.004 (155,000) | −0.005 | 0.10 |
| conjunct over **all** sites ~ order | −0.16 (2,687) | −0.012 (3,263) | 0.04 |

The manifests also give each pair's 2×2 table and the difference in shares, P(A-form | A) − P(A-form | B). For
example, in `control_half` reiteration against route: 0.0067 (n 102,384).

**[obs] The route records themselves.** In both authors corpora:
- **Step order.** 887 of the 1,550 have room, all in A-form.
- **Conjunct.** 19 free sites, all right-first.
- **Reiteration.** 1,532 of 3,060 sites, 50.1 %. 556 of the route records could not take it (lines>12).
- **Length.** The source proofs average 8.8 lines and A's forms 9.8.

## The pitfalls

### 1. Step order and reiteration are coupled

[obs] On the same 155,000 records, the per-combination totals in `authors_build.json` (`combo_totals`) show the
coupling:

| | order A (premise-driven) | order B (goal-driven) |
|---|---|---|
| reiteration sites | 219,490 | 236,038 (+7.5 %) |
| R lines added by A's reiteration | 122,027 | 123,976 |
| records skipping reiteration at the cap | 29,704 | 32,027 |
| conjunct sites (free) | 5,533 (3,344) | 6,972 (4,620) |

- [obs] The goal-driven form has *more* sites than the premise-driven form. That is the opposite direction from the
  one HABITS.md expected. Sinking puts derivations inside boxes, so their citations of premises become box-level
  citations of outer lines.
- [obs] Either way, a count of R lines would move with step order. The measure must be the share of sites
  reiterated.
- [obs] **Order of application.** Reiteration must come last, and every order reading must strip it first. An R copy
  does not depend on its box's assumption, so the premise-driven normal form hoists it out of the box.
  - Read without stripping, only 10,946 of A's 24,002 order-room records come out premise-driven, and 13,056 read as
    "neither".
- [obs] **R copies also fake conjunct sites.** A copy at the top of a box sits before its conjunct partner. On the
  unstripped proofs, `habits.conj_sites` counts 8,350 sites in A's records (87 % right-first) against 2,944 on the
  stripped ones.
- That is why stage 1 strips first, stage 4 reiterates last, and the order and conjunct detectors work on the stripped
  proof.

### 2. Reiteration costs lines, and the cap takes it from long proofs

[obs] A's reiteration share by **source** length (`authors_half`, records with room):

| source lines | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| A's records with room | 2,173 | 4,561 | 4,981 | 6,000 | 6,511 | 6,837 | 6,824 | 7,070 | 7,021 |
| sites reiterated | 98.8 % | 98.6 % | 99.4 % | 98.5 % | 96.3 % | 82.0 % | 56.2 % | 10.7 % | 0.7 % |
| skipped for the cap | 0 | 0 | 0 | 0 | 5 | 535 | 2,090 | 5,535 | 6,906 |

(Below 100 % at 4 to 8 lines, the gap is mostly the route-detector skips, described under the doubts below.)

[obs] Lengths of the written proofs (`authors_half`, records per length):

| lines | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 7,042 | 6,967 | 4,952 | 4,774 | 4,320 | 6,681 | 5,225 | 8,642 | 8,444 | 10,571 | 10,570 |
| B | 7,049 | 7,124 | 7,007 | 7,006 | 7,456 | 8,039 | 7,752 | 7,644 | 7,144 | 5,876 | 4,715 |

- [obs] Both authors start from the same source (about 7,000 records per length each). A's proofs then average 7.65
  lines (source 7.02) and B's 6.81 (source 6.98).
- [obs] B loses lines because stage 1 removes the 10 % of sites the source had already reiterated.
- [obs] At 12 lines, 69 % of records are A's; at 4 lines, 41 %.
- [obs] 15,071 of A's 51,978 records with room (29.0 %) could not take the full form. The share in `p10` is 29.8 %.
- [obs] The rule was to skip the whole habit, as SPEC.md says. Reiterating only as many sites as fit would raise A's
  share from 55.8 % to an estimated 70.9 %. That figure counts lines only; it was not built or checked
  (`reit_partial_estimate` in the manifest).

### 3. Does the state show the habit where the model chooses?

[obs] For each habit, the build compares the two forms of the same theorem with only that habit changed and the other
two held at B. Each step of one form is compared with the other form's steps:
- **novel:** its state (hypotheses in scope and goal, names stripped) never occurs in the other form;
- **same state, other action:** the other form takes a different action from that same state, i.e. a choice made with
  no trace in the state;
- **same:** the same state and the same action.

This was run on every record of the corpus with room for the habit. The figures are for the A-form side; the B side
is similar (see `state_comparison` in `authors_build.json`).

| habit (others B) | pairs | novel state | same state, other action | differs (sum) | choices per pair | pairs with ≥ 2 choices from a shared state |
|---|---:|---:|---:|---:|---:|---:|
| step order | 46,866 | 64.2 % | 10.6 % | 74.8 % | 1.10 | 9.3 % |
| reiteration | 68,530 | 40.4 % | 12.0 % | 52.3 % | 1.20 | 20.2 % |
| conjunct | 4,446 | 39.4 % | 11.9 % | 51.2 % | 1.00 | 0.1 % |
| all three (A vs B) | 95,480 | 53.6 % | 11.3 % | 65.0 % | 1.18 | 17.4 % |

- With the other habits held at A, the same habits give 77.0 %, 47.4 % and 59.4 % of steps differing. The novel-state
  shares (64 %, 40 %, 39 %) agree with the 65 / 39 / 39 % of HABITS.md (d), measured there on 300 pairs.
- [interp] **Step order** is carried by the state. In 91 % of pairs, the first divergence is the only step where the
  two forms act differently from a state they share; every later difference is in a state the other form never
  shows. A model can continue an author's step order by reading the state.
- [interp] **Reiteration** is only partly carried. In 20 % of pairs the habit is chosen again from a state that B's
  proof also shows. A box closes, its R copies leave the state, and the next box opens from the same state in both
  forms. At those steps only the tag, or the model's prior for the proof, can keep reiterating.
- [interp] **Conjunct order** is one choice per proof. Its later trace lasts only while both conjuncts are in scope.

### 4. What the environment fixes is untouched

- [obs] **Premise lines.** The premise lines are the environment's preamble (`have n<k> : F := h<k>`, in order,
  before any other line). They are never edited: every rewrite renders them first, unchanged. A test compares them
  byte for byte with the source on 4 × 1,500 records, and every local check compares their formulas.
- [obs] **Box closers.** The closing `exact` tokens and the second-branch opener of Or.elim are produced by the
  environment from the box structure.
  - Every box keeps its last line: hoisting never moves it, sinking inserts at the front, and conjunct reordering is
    rejected if a scope's last line changes.
  - Every final proof replays through `state_env` with canonical names.
- [obs] **Other R lines.** Top-level restatements, and a box's required last R line, are kept in both authors'
  proofs.

## What makes me doubt the design

1. **[interp] Reiteration is a short-proof habit in this corpus.**
   - The cap removes it from 11- and 12-line proofs: 10.7 % and 0.7 % of sites.
   - The route records are long, so only 50 % of their reiteration sites are in A's form.
   - A model trained on A can learn "A reiterates when the proof is short", and that is the habit the corpus teaches.
   - On long measurement theorems (`route_unseen` from the long pool), a perfect A model would reiterate little at
     some lengths.
   - The measure should report reiteration by length, or only where A's form fits.
   - Partial reiteration (about 71 %, estimated) or a costless habit in its place are options; neither was built.
2. **[interp] Length is an author cue.** A's proofs are 0.84 lines longer than B's. In `half`, P(A | 12 lines) is
   0.69. The control has the same lengths per habit form, so it matches this for the habits, but the authors corpus
   ties length to the tag.
3. **[obs] Conjunct order is barely a habit.**
   - It has free sites in 3,719 records (2.4 %).
   - At the other 34 to 40 % of sites (depending on the order form), step order fixes it.
   - Counted over all sites, it correlates with step order even in the control (φ −0.16, n 2,687). The measure has
     to use free sites; `detect` gives them.
4. **[obs] Dan's σ⁺ detector is not invariant to reiteration.**
   - `coe_assemble.knock` flags a DN whose cited line is not a premise or an assumption (`patterns.derived_dn`).
   - A DN citing an R copy of a premise ¬¬X therefore counts as σ⁺ ("route").
   - `coe_sigma.has_sigma` follows R chains and does not flag it.
   - The build skipped A's reiteration in the 638 records (`half`) where it would have flipped σ⁺.
   - [guess] A model trained on A may write such proofs. A route measure on its output through `knock` would then
     count them as route use. The measure should follow R chains for σ⁺ as well, or use σ.
5. **[obs] The detector definitions matter for the measurement.**
   - Read with `habits.normal_form` (uncapped sink, as `authors_profile.py` in this folder does), 1,650 of B's
     22,864 order-room records (7.2 %) read as "neither".
   - Order read without stripping reiteration, and conjunct read without stripping, are wrong as shown in pitfall 1.
   - `authors_corpus.detect` is the reading this corpus was built and tested against.
6. **[obs] Loaders must accept `author_tag: "none"`.**
   - The tag experiment's loader (`../author_tags_20261110/tag_pretrain.tag_load`) asserts that every state carries
     a tag token, so it would fail on `"none"`.
   - `TagTokenizer` itself encodes `"none"` as an untagged state.

## Tests

`test_authors_corpus.py`: 19 tests, all passing in 22 s, with this command:

```
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m pytest --confcutdir=. -p no:cacheprovider -q test_authors_corpus.py
```

The suite builds the first 1,500 records twice, each time with a fresh Lean cache, so Lean checks every rewritten
proof inside the test. It checks:
- determinism (byte-identical builds), and that the limited build is a byte prefix of the full one (so `--limit` and
  the hash seed change nothing);
- no record lost or duplicated, and `key`, `thm`, `prompt` and `n_prem` unchanged;
- every proof passes `nd_verify`;
- limits: 12 lines, depth 2, 3 premises, the tokenizer, the premise lines byte for byte;
- every route record is A and σ⁺, and B has none;
- the author draws are the existing rules;
- `realized` equals a fresh detector run, and every habit with room is in its assigned form unless `skipped` says why;
- the tag is shown on about half of each author (|z| < 3);
- control independence of the drawn forms, label and route (|φ| < 3.5/√n), and realized shares within 0.05 of target;
- the manifest counts and hashes match the files;
- unit tests on hand-made proofs: strip / reiterate round trip, order forms, the capped sink against `habits`' own
  (depth 3 against 2), conjunct forms, a Lean negative control, and the Lean fallback, with a forced rejection that
  drops the reiteration stage.

`ruff format` and `ruff check` pass with the repository's configuration.

A second full build into a scratch folder reproduced all four corpora byte for byte.

## Reproduce

```
cd code/experiments/current/two_authors_20261111
CUDA_VISIBLE_DEVICES= LEAN_GATE_WORKERS=8 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python authors_corpus.py --force
```

- The script pins itself and every process it starts to 8 cores; Lean processes are multi-threaded.
- It runs with `PYTHONHASHSEED=0`.
- It refuses to overwrite the corpora without `--force`.
- `--limit N --out-dir DIR` builds the first N records.
- Time on serrano: 106 s for the rewrites and local checks (8 processes), then 218 s of Lean on 122,729 proofs with
  an empty cache. That is 7 min 52 s in all, at a peak of 2.7 GB; with the Lean cache, 4 min 21 s.
