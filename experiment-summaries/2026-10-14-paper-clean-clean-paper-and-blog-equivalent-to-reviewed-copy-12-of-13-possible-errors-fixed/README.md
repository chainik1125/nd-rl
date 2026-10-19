---
written_on: 2026-10-14
written_by: agent:claude
---

# paper-clean: the clean paper and blog for "Did RL create it?" are equivalent to the reviewed marked copy; 12 of the librarian's 13 possible errors were real and are fixed; no printed theorem in a textbook72 class; blog 2,487 words

Status: completed and reviewed 2026-10-14. **No hard-constraint violation, nothing quarantined.** This was an editing
run, with no pod and **$0**. It turned the reviewed reading copy of *Did Expert Iteration Create It?* and its blog post
into clean drafts. The marked copy carried the `create-or-elicit-paper` review's corrections as *[review]* marks. The
clean drafts are in nd-rl at [`docs/papers/create-or-elicit/`](../../docs/papers/create-or-elicit/README.md), and the
marked copy, unchanged, is at [`review_marked/`](../../docs/papers/create-or-elicit/review_marked/README.md).

The reviewer compared clean against marked, number by number and claim by claim, with four sub-auditors and their own
code, and re-derived every consequential row from the data themselves. The verdict, in the review's words:
- **The paper:** the clean version is equivalent to the marked reading copy.
  - Each of the 60 corrections is folded with its meaning.
  - No number changed except the task-2 fixes, and each of those reproduces from the data.
  - No claim is stronger. The one widened claim is item 12's: "kept no proof text" now covers only three comparators.
    It rests on a new Lean re-check, which the reviewer reproduced with an independent renderer: 24 / 24 accepted,
    0 / 24 with the goal negated.
- **The 13 items the librarian listed** as possible errors outside the earlier review: 12 were real errors and are
  correctly fixed. Item 12's note was wrong.
- **textbook72:** no printed theorem is in a textbook72 renaming class under either key.
- **The PDF** builds clean: 35 pages, with only the reading copy's three overfull boxes, each under 2 pt.
- **The blog:**
  - it is 2,487 words;
  - it keeps the four caveats the brief names and every B1–B8 correction;
  - it names the power-sampling models correctly;
  - it claims nothing the paper does not.

  Of its 31 marks, 30 are folded; the 31st (R23) was cut with the seeding paragraph it qualified.
- **Not supported:** nothing. Every claim in the run's write-ups reproduces or differs only in wording.

**For Dan to decide** (the review: "Dan's call; the run disclosed both cuts"): the blog's 2,500-word cut removed two
results that the paper still carries.
- **F1 (should fix):** the one-round seeding null. The paper reports it as a null and as a missed pre-registered
  expectation.
- **F2 (recommended):** the RL-phase likelihood clause, the blog's only reading of Fig. 2's trajectories.

The blog has 13 words of headroom. Restoring both (≈ 38 words) needs ≈ 25 words cut elsewhere.

**What this changes in earlier summaries:**
- **[create-or-elicit](../2026-10-12-create-or-elicit-sigma-knockout-not-acquired-in-16-ei-rounds-needs-exposure-1pct-dose-restores/README.md)**
  says that the plain comparators of power sampling (Phase 2) keep no proof text, so their 1–2 solves could not be
  re-checked in Lean. That holds only for the comparators of the earlier full s0 and s1 bases and of knockout s0's
  base. Two sets of comparators stored their accepted proofs:
  - knockout s1's base at T 1.0: its one solve (`la_transfer_916`);
  - the positive control, on the earlier full s0's base (6 / 17 solved at T 0.8, 13 / 17 at T 1.0; 8 and 15 distinct
    proofs).

  Lean 4.34.1 accepts all 24 stored proofs and rejects all 24 with the goal negated. The reviewer reproduced this
  with their own renderer under Lean 4.34.1 and 4.35.0. The solve counts are unchanged; only their re-checkability
  is new.
- **[create-or-elicit-paper](../2026-10-13-create-or-elicit-paper-draft-and-blog-no-headline-number-wrong-c4-drops-knockout-s1-elicitation-reading/README.md)**
  flagged two defects in the writing run's own files and the §2 schema. All three are resolved, and the review
  confirms each:
  - `LEDGER.md` P8 is now marked superseded by C4;
  - `next_steps.md` item 1b is re-costed at C4's measured 12.35 ms per attempt: 2.4 × 10⁷ attempts take ≈ 82
    A40-hours ≈ $48;
  - §2's negated-conditional schema, which is in `textbook_b98931f2…`'s class, is replaced by words.

  Its reading copy moved to `review_marked/`. Its summary, `docs/STATE.md` and the week-42 digest now link it there.
- **The paper's descriptive rarity factor changed.** By record share, the paper's metric, pretraining's DN and
  ¬-introduction records are 4.2× and 2.9× rarer in K12⁻ than in K12 (13.73 → 3.29 % and 15.21 → 5.19 %, as
  create-or-elicit's summary already gives them), not "three to four times". By line share the old range would have
  held (3.90×, 3.03×). No headline count of the abstract or the intro changed.

Question (brief [`BRIEF_paper-clean.md`](../../docs/proposals/paper/BRIEF_paper-clean.md), authorised by Dan
2026-10-14): can the reviewed paper and blog be made clean without changing any number (other than listed fixes),
strengthening any claim or dropping any reviewed caveat? And are the 13 "possible errors outside the review's list" in
the reading copy's README real? The tasks:
- fold the marks;
- resolve each possible error against the data;
- remove any printed theorem in a textbook72 class;
- bring the blog to ≤ 2,500 words;
- rebuild the PDF;
- have an independent subagent check equivalence.

## Configuration

- **Repository.** Fork `dan-pandori/nd-takehome`, branch `dan_paper-clean`, made by `newrun` from the fork's
  `origin/dan_create-or-elicit-paper` (`1ad2a691`). Ran on agent2, every role at `--effort max`, hard stop 48 h.
  - The first commit, [`11d07d13`](https://github.com/dan-pandori/nd-takehome/commit/11d07d13) (05:27:04 UTC), copied
    the reading copy (nd-rl `7709669:docs/papers/create-or-elicit/`) into `paper/` byte for byte; the reviewer checked
    this. So the run's changes are `git diff 11d07d13 86b705ee -- paper/`.
  - Pre-registration `preregistration/paper-clean.md`, in the same commit: expectations E1–E6, the blog's word-count
    method, no pod planned. It precedes the first text change (`325a05c9`, 05:29:38).
  - DONE at [`8c4b4f64`](https://github.com/dan-pandori/nd-takehome/commit/8c4b4f64) (`PAPER-CLEAN DONE
    2026-10-14T06:40:53Z`). The final commit, [`86b705ee`](https://github.com/dan-pandori/nd-takehome/commit/86b705ee),
    fixes word counts in the write-ups and is the commit the review audited.
  - Deliverables:
    - `paper/`: the clean `main.tex`, `sections/`, `main.pdf`, `blog.md`, `figures/` and `lean/`, plus `CLEANUP.md` and
      `next_steps.md`;
    - `run_paper_clean.md`, `log.md`, `numbers.md` § paper-clean and `STATUS.md`;
    - evidence in `artifacts/paper_clean/`: four check scripts and their outputs, and three equivalence reports.
  - Review: [`review_paper-clean.md`](https://github.com/dan-pandori/nd-takehome/blob/792db421/review_paper-clean.md).
    - [`d7b70d58`](https://github.com/dan-pandori/nd-takehome/commit/d7b70d58): §Recount, written before reading any
      write-up;
    - [`792db421`](https://github.com/dan-pandori/nd-takehome/commit/792db421): §Comparison and §Verdict.
    - The reviewer's kit is in `review_pc/`:
      - own scripts in `kit/`: `numdiff.py`, `sealed.py`, `sigma_polarity.py`, `ndlean.py` (an ND → Lean renderer
        that uses neither `nd2lean` nor `lean_judge`) and `lean_checks.py`;
      - four sub-auditor reports in `phase1/`.
- **Timeline (UTC, 2026-10-14).** Executor 05:24–06:41; reviewer 06:42–07:13; librarian from 07:14.
- **Models.** None was trained or sampled. The task-2 checks re-read existing `create-or-elicit` files on the VPS (CPU,
  one process per check). The models behind the re-checked numbers:
  - items 7 and 9 concern the knockout bases and ladders: best-cap12 recipe, 9,560,832 params, `lean_staten`, from
    scratch on K12⁻, 16 T1 EI rounds, Lean alone;
  - item 10 compares three full seeds with six knockout seeds at r0 (pend) and r16, as labelled in the paper's §3.5
    (the review says §3.6; the label is at `05_results.tex` l.282, in §3.5);
  - item 12 concerns power sampling's plain comparators on the bases:
    - the positive control ran on the earlier full seed s0's base (`ckpts/coe/old/stage1_best12_s0_b1200.pt`, 9.56 M,
      `lean_staten`, from scratch, K12);
    - knockout s1's comparator ran on its base (`ckpts/coe/stage1_ko_s1.pt`, K12⁻);
  - item 1 counts records in the pretraining set K12 (𝒟, 155,000 records) itself.
- **Checks (Lean alone; `nd_verify` not used).**
  - The executor: `nd2lean` plus Lean 4.34.1 through `lean_judge.judge_many`.
  - The reviewer: their own renderer under Lean 4.34.1 and 4.35.0. It was validated on 250 generator proofs from 𝒟:
    250 / 250 accepted, 0 / 250 with the goal negated, 0 / 250 cross-paired, 0 / 150 with DN changed to reiteration.
    1 / 150 survived a shifted reference; its mutated step is itself valid, because the new referent also contradicts
    the premise.
- **Compute and cost.** The budget, $20 / 40 pod-hours, was registered with `podbudget` at 05:24:49. **No pod was
  created: 0 pod-hours, $0.** There was no GPU time, no generated tokens and no training step, so there are no
  results-registry rows.
- **Bucket.** `hf://buckets/dan-pandori/nd-rl/paper-clean/artifacts/paper_clean/` holds 8 files: the four check
  scripts and their outputs.
  - `sigma_goal_polarity`, `gate_rejections`, `power_comparator_recheck` and `tb72_sealed_check`, each as `.py` and
    `.json` / `.txt`;
  - the equivalence reports quote the drafts and are in git only;
  - no checkpoints or data.

## Observations

### Pre-registered expectations, against the reviewer's checks

| | expectation (committed 05:27) | outcome | the reviewer |
|---|---|---|---|
| E1 | 0 marks left; numbers change only by task-2 fixes | hit | `\rv` 60 → 0 (and its definition); blog *[review]* 32 → 0 (31 in the body, 1 in the librarian's header); 0 "[review]" in the PDF text. Number-by-number diff (`kit/numdiff.py`): no number in `sections/` removed. One changed: the abstract's "3–4" → "2.9–4.2". Every inserted number belongs to a task-2 item. In the blog every dropped number belongs to a cut passage, and three changed: the draft date, Fig. 2's alt text ("12 of 15 runs") and "three to four times" → "2.9 and 4.2" |
| E2 | 9–12 of the 13 items real; no headline number changes | hit: 12 real, 1 changed differently | each item checked against the data (table below). No headline count of the abstract or the intro changed; the abstract's descriptive rarity factor did |
| E3 | exactly one printed theorem in a textbook72 class before, 0 after | hit | own parser and keys. The marked §2 schema hits `textbook_b98931f2…` under both keys; the clean copy has 0 of 19 distinct printed sequents in a class |
| E4 | blog ≤ 2,500 words; the four named caveats present | hit | 2,487 words without front matter (2,493 for the whole file); the four caveats are present with their qualifiers. Two cuts remove results the paper carries (F1, F2) |
| E5 | 0 undefined references; ≤ the reading copy's three overfull boxes, all < 2 pt | hit | both copies rebuilt in scratch: exit 0, 0 undefined references, the same three boxes (0.655, 0.959, 1.924 pt), one new cosmetic underfull box; the committed PDF's text equals the rebuild's |
| E6 | first equivalence pass finds 1–6 must-fix items, a second pass 0 | hit: 2 (blog), fixed; then 0 | consistent with the executor's reports. The reviewer's own pass agrees on the paper (0 must-fix). For the blog it differs in judgment on F1; the executor's checker had classed F1 and F2 as accepted cuts |

### The 13 possible errors, checked against the data

| # | the note (reading copy's README) | the clean text | evidence (the reviewer's check unless noted) | verdict |
|---|---|---|---|---|
| 1 | σ is defined "for a positive goal G" | §1: "to prove a goal G" | Table 1 defines σ with no positivity condition. In 𝒟 (K12), 16,703 records contain σ; 1,721 apply it to a negated G (1,585 only to negated goals), as do 1,741 of 17,387 σ lines (10.0 %). The reviewer's own parser gives identical counts | real, fixed |
| 2 | the 123 strict training theorems "require the composition or a variant" | "need the composition, a variant or the lemma route of Table 1" | Table 1's strict-member row says the same (free positive reductio or lemma route); 123 unchanged. Knockout s1's r20 proofs of strict members include 7 by the lemma route (σ 32, other σ⁺ 24; the paper's §3.4) | real, fixed |
| 3 | §1's C4 sentence omits the earlier early-stopped successes | adds "though two of those three had one success each in an earlier, early-stopped sample" | `1246` 1 / 7,168 and `2256` 1 / 9,216 early-stopped; C4 0 / 18,432 on `1019`, `1246`, `2256` | real, fixed |
| 4 | the abstract's last sentence states an interpretation ("EI amplified …") | "In this organism, seeds pretrained with the composition acquired classical families, and within 16 rounds of EI, seeds without it acquired none." | "acquired" is the abstract's own pre-registered indicator (6 / 6 full and 3 / 3 dose against 0 / 6); both scope qualifiers stay. Not stronger | real, fixed |
| 5 | §3.8's "150 re-submitted Lean rejections" | "150 re-submitted texts that the gate had rejected (all by its pre-filter)" | 1,307 of 1,307 `[lean_gate]` lines in the two runs' `logs/` (60 files) have rej = filter: 12,217,252 rejections. Over all of `artifacts/coe*` the reviewer finds 1,368 of 1,368 lines in 121 files, 12,919,685 rejections | real, fixed |
| 6 | "three to four times rarer" | DN 4.2×, ¬-introduction 2.9× (abstract "2.9–4.2", §1, §3, §7, Fig. 0, blog) | record shares K12 → K12⁻: DN 13.725 → 3.292 % (4.17×), NEGI 15.208 → 5.191 % (2.93×). By line share the old range would have held (3.90×, 3.03×) | real for the paper's metric, fixed |
| 7 | knockout s5's base also proved `1142`; s2's x1 draw also proved `1096` | s5's ladder proves "neither `1811` nor `1142` in any" of the four plain r16 draws. s2's ladder proves "`1963` in the main evaluation and `1096` in the other three plain r16 draws (one of which also found `1963`)" | s5 base, early-stopped: `1142` 1 / 3,072 and `1811` 1 / 13,312. s2 at r16: x0 {`1963`}, x1 {`1096`}, C1 standard {`1096`}, C1 doubled {`1096`, `1963`}. s5's ladder: `1246` in 4 / 4 plain draws, `2256` in 3 / 4, `1811` and `1142` in 0 / 4 | real, fixed exactly |
| 8 | App. I implies batch 512 for every read; its script list omits A3b and A5b | C4 at batch 1,024 (both seeds); lists `a3b_member_prior_gained.py` and `round2_extra.py` | C4 `ko_s{1,3}` args: batch 1,024; both scripts exist and are the A3b / A5b code. (Executor: all 29 C1–C3 reads at batch 512) | real, fixed |
| 9 | §6: "the base rate predicts which seeds move" | a zero rate on the 105 family members predicted no movement (s0, s3) and the highest rate the most (s1). Seed 4, with the second-highest rate, never proved a strict training theorem; seed 2, with the lowest non-zero rate, did | knockout base rates: s1 5.36 × 10⁻⁶ (first r9), s4 2.21 × 10⁻⁶ (never by r16), s5 1.41 × 10⁻⁶ (r14), s2 7.47 × 10⁻⁷ (r14), s0 = s3 = 0. s3's base proved `916` (1 / 2,048), so its zero holds on the 105 only | real, fixed |
| 10 | §6: "a few theorems out of hundreds" | "at r0, 4.3 of holdout250's 225 and 4.0 of textbook72's 53 intuitionistic theorems (2.5 of the 53 at r16)" (§3, §6) | ledger P57 (reviewed); the models are labelled in §3.5 | real, fixed |
| 11 | §5: "mastery is not sufficient" lacks the scope | "… for EI at this scale within 16 rounds" | the create-or-elicit review's rewording ("not acquired by this EI within 16 rounds"); no data check needed | real, fixed |
| 12 | the power-sampling positive control's comparators "also kept no proof text" | §7 names the three comparators without stored proofs; App. F: Lean 4.34.1 accepts all 24 stored proofs and rejects all 24 with the goal negated (*this paper*) | stored accepted texts exist only for knockout s1 at T 1.0 (1) and the positive control at T 0.8 (8 distinct) and T 1.0 (15). Executor 24 / 24 and 24 / 24; the reviewer's own renderer 24 / 24 (Lean 4.34.1 and 4.35.0) and 0 / 24 negated | **the note was wrong**; changed differently, and the new claim reproduces |
| 13 | blog: "hand-picked"; "simplest kind" and "routinely"; the power-sampling paragraph's models; Fig. 2's alt text and title | "chosen in advance by a fixed rule"; "fine-tunes on its own successes (rejection-sampling fine-tuning)"; "routinely" gone; "two earlier full runs' bases" and "two knockout bases"; alt text "… 12 of 15 runs", PNG title "… 12 of the 15 runs …" | paper §4: "pre-selected as the most reliable r8 proofs". The power files' `ckpt` fields: `old/stage1_best12_s{0,1}` and `stage1_ko_s{0,1}`; the knockout bases were sampled on the 106. The regenerated PNG shows 12 lines (3 full, 3 dose, 6 knockout) | real, fixed |

The positive control's 17 theorems include 2 textbook72 problems, solved at T 1.0. This is scoring use from
`create-or-elicit`, which is allowed. Their proofs are in the 24 / 24, and no text is printed.

### textbook72

- §2's negated-conditional schema is now a verbal statement of the same law ("a negated implication entails its
  antecedent and the negation of its consequent"), which the brief allows. The review notes that a reader can rebuild
  the class from it (N2).
- **Executor.** `tb72_sealed_check.py` used the `--sealed` method of `make_golden.py` (nd-rl `dan_shared_proof_state`,
  `gen.canon_key` of "premises ⊢ goal") plus a premise-order-invariant key. It checked eight printed theorems; 0 of 8
  are in a class.
- **Reviewer.** Their own scanner covers the sections, `main.tex`, the PDF text, every figure's text, the blog and
  `lean/`. It reads each formula several ways and uses both keys. It finds 19 distinct printed sequents or formulas,
  and 0 are in a class as printed.
  - The positive control is the marked copy: there the §2 schema hits `textbook_b98931f2…` under both keys. The
    reviewer's first extractor missed it, and only the control exposed that.
  - One broader reading hits in both copies (N1). Table 1's lemma-route formula, with its antecedent read as a premise,
    is in `textbook_dd80d9db…`'s class. Under the brief's method, the printed sequent, it is not a hit. The paper cannot
    avoid this law: it is the paper's subject, and the abstract states it in words.
- **Records (F3).** The run's own `artifacts/paper_clean/equivalence_paper.md`, committed to the public fork, prints
  the removed schema (a primary hit). The review advises masking it by ID and keeping it out of the bucket; it is not
  in the bucket. `data/bs/textbook72.jsonl` itself is in the same repository, so this is hygiene, not a leak.
- The fork's root `log.md` prints formulas in two textbook72 classes. These come from take-home commits of 2026-09-15
  to 17 and predate this run.

### The blog

- 3,143 words (marked, including the librarian's 63-word note) → **2,487** (clean, by the pre-registered count:
  `wc -w` without the YAML front matter, including the Lean block, alt texts and captions).
- **The four caveats the brief names are present with their qualifiers:**
  - the 16-round scope;
  - the sampler-specific sharpening null ("we count it as a failure of our sampler");
  - rare, not absent;
  - the power limits ("not for most single-family tests", and the certification limits).
- Every B1–B8 correction is kept. No model label and no post hoc, exploratory or pre-registered label was lost. Every
  new phrasing is in the paper.
- **Marks.** 29 of the 31 are folded in place. R29's definition of "Hardest" moved from Fig. 4's caption into the text
  just before the figure. R23 was cut with its paragraph.
- **F1 (the review: should fix).** The "Seeding didn't help" paragraph is gone. The paper still reports its content:
  - the null, seeded minus control +0 / +0 / +2 / +0 / +0;
  - the missed pre-registered expectation, "≥ 3 seeds", with 1 / 5 observed.

  The paragraph also carried the older-checker caveat on the earlier-organism contrast. The earlier review's B2 held the
  blog to "the same evidence as the paper, including the negative and null results". The reviewer's example
  sentence: "One extra RL round seeded with a rare σ proof changed nothing measurable (an earlier, smaller model under
  an older checker had responded)."
- **F2 (recommended).** The RL-phase likelihood sentence is gone: "one full run and one dose run moved strongly toward
  these proofs; the rest … ended between 8 units lower and 4 units higher". The paper keeps it. It was the blog's only
  reading of Fig. 2's trajectories, and it cuts against an amplification reading.

### The paper, build and Lean

- All 60 marks are folded: 57 word for word, and 3 sentences smoothed, each equal or weaker.
  - §8's word order, and "the experiment's review" (§3.1).
  - §7 Settings: "Doubling the caps moved each strict count" became "With doubled caps, each strict count differed".
    This is the only rewording beyond a fold, and it is weaker; it fits P1.
- `refs.bib` and `lean/` are byte-identical to the marked copy, and the appendix listings are unchanged. Only
  `fig0_overview` (the ratio label) and `blog_fig2_logp` (the title) were regenerated. In each, the pixel rows differ
  only in that band.
- The abstract is 250 words (marked 255, with its five marks).
- All four Lean listings printed in the paper and the blog compile under Lean 4.34.1 and 4.35.0, with no errors and
  only standard axioms (no `sorryAx`): Fig. 1, App. (a), App. (b), and the blog's block (= `lean/blog_example.lean`).
  `lean/blog_example_negctl.lean` is rejected.
- No proof-length claim changed, so no term size needed re-deriving.

### Equivalence: the reviewer's sub-auditors, clean against marked

| part | rows | equivalent | substantive, data-backed fix | must-fix | minor | note |
|---|---|---|---|---|---|---|
| front (abstract, §1, §2, §4, §5, `main.tex`) | 35 | 22 | 9 | 0 | 2 | 2 |
| results (§3, App. I) | 35 | 25 | 9 | 0 | 1 | 0 |
| back (§6–§8, the appendix, `refs.bib`) | 27 | 18 | 7 | 0 | 0 | 2 |
| blog | 70 | 54 | 9 | 1 | 2 | 4 |
| **total** | **167** | **119** | **34** | **1** | **5** | **8** |

The blog's must-fix row is F1, which the reviewer reports as "should fix" and leaves to Dan. In the paper, every
substantive change is one of the task-2 items or the textbook72 schema. The executor ran two passes of its own:
- first: paper 85 changed spans, 0 must-fix, 3 minor; blog 2 must-fix;
- second: 0 must-fix.

The review finds the executor's record consistent with those reports. It differs in judgment on F1, which the
executor's checker had accepted as a cut, as it had F2.

## Rewordings the review requires (records, not the drafts)

- `run_paper_clean.md`, `STATUS.md` and the fork's `paper/README.md` say "31 blog *[review]* marks folded". They
  should read "30 folded; 1 (R23) cut with the seeding paragraph it qualified", as `CLEANUP.md` §6 says. Applied in
  this summary and in nd-rl's new README. The fork's files are unchanged.
- `log.md`'s "No reviewed caveat was cut" overstates; `CLEANUP.md` §4's wording ("Removing these left no reviewed caveat
  without the claim it qualifies") reproduces. The fork's log is unchanged.
- `next_steps.md` presentation proposal 6 says the two missing blog labels no longer fit. They do: 13 words remain.
  Applied, marked *[review]*, in nd-rl's copy.
- F3: mask the removed schema in `artifacts/paper_clean/equivalence_paper.md`, by ID. **Not done**: it is a fork
  record, and the librarian works in nd-rl. It is open, and listed for Dan in the digest.

Optional minor text items in the drafts, not applied so that the clean versions stay exactly as reviewed. The nd-rl
README lists them for Dan:
- App. I names `round2_extra.py`, which prints A5b's 0.0188 by Monte Carlo, beside §3's exact 0.020;
- "early-stopped" is never defined in the paper;
- §5's "a quarter to a third" is approximate beside 2.9× / 4.2×;
- the blog's Fig. 4 caption no longer defines "Hardest";
- the blog's "that only a genuinely classical step can prove" now attaches to the 105 only;
- §6's s4 sentence would be cleaner with "in 16 rounds".

## Copy in nd-rl

The brief asks the librarian for three things: move the marked reading copy to
`docs/papers/create-or-elicit/review_marked/` unchanged, put the clean versions at `docs/papers/create-or-elicit/`, and
add a README note explaining the two. The commits on branch `dan_paper-clean`:
- `dda847b` moves the marked copy by pure rename. The tree at `review_marked/` equals
  `7709669:docs/papers/create-or-elicit/`.
- `1b93c2b` adds the clean versions, hash-identical to the corresponding 45 files of `86b705ee:paper/`: `main.tex`,
  `sections/`, `refs.bib`, `build.sh`, `main.pdf`, `blog.md`, `figures/`, `lean/`, `CLEANUP.md` and `next_steps.md`.
- The librarian's commit after it adds:
  - the README;
  - `next_steps.md`'s header and *[review]* notes;
  - the links from the `create-or-elicit-paper` summary, `docs/STATE.md` and the digest to `review_marked/`;
  - this summary, the index line, the `STATE.md` line, the spend row and the digest update.

Librarian checks:
- **Rebuild.** A scratch tectonic rebuild of the nd-rl copy gives 35 pages, 0 undefined references, the same three
  overfull boxes, and `pdftotext` identical to the committed `main.pdf`.
- **textbook72.** The reviewer's scanner, run on the nd-rl directories, gives 0 printed hits on the clean copy (19
  distinct sequents). On `review_marked/` it hits `textbook_b98931f2…`.
- **Not copied.** These stay on the fork:
  - the working files `OUTLINE.md`, `LEDGER.md`, `REVISIONS.md`, `critiques/`, `notes/`, `scripts/`, `data/` and
    `artifacts/paper_clean/`;
  - the run's own `paper/README.md`, which still says "31 blog marks folded". nd-rl's README replaces it.

## Limitations and next step

- **Scope of the check.** Equivalence is clean against marked. The marked copy's own claims were audited by the
  `create-or-elicit-paper` review (731 claims). This run fixed only the 13 items listed outside that review.
- **The earlier sub-audit is not fully mapped.** The run's `next_steps.md` presentation proposal 1 counts about 79
  W / D rows in that sub-audit. The review's items and the librarian's 13 cover part of them, but no file maps each row
  to its resolution. It names three rows it considers still open in the clean text: I25, I43 and R47. The paper-clean
  review did not check those rows.
- **textbook72.** The count depends on the extraction: 8 theorems in the executor's hand list, 19 in the reviewer's
  mechanical one. The verbal §2 law still identifies its class to a careful reader (N2), and Table 1's lemma-route
  formula is in a class under a broader reading (N1).
- **The blog** omits two results the paper carries (F1, F2) until Dan decides.
- **Next step.** No measurement is needed to settle this run: every open item is editorial. The decisions are Dan's:
  - F1 and F2, which need ≈ 25 words cut elsewhere;
  - the optional minor items and F3;
  - the author list and acknowledgements, and whether to circulate.

  The paper's open scientific question is unchanged: the knockout ladders' σ rate along the run (`next_steps.md` 1b).
  At C4's measured 12.35 ms per attempt it costs ≈ 82 A40-hours ≈ $48 in full, or ≈ $24 at r8 and r16 only. It would
  turn A5's waiting-time null into an observed rate.
