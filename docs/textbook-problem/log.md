# Textbook-problem workstream log

## 2026-09-27 (Charles + Claude)

**Setup.**
- Rewrote the root `CLAUDE.md` as goal-independent conventions and safety rules. Each workstream's
  `plan.md` now carries its goal, scope and budget.
- Created `docs/textbook-problem/plan.md`.

**Ran.** A categorization of the textbook 72 (plan task T2; descriptive only):
- `code/textbook_problem/overview.py` (with `--patch` for timeouts), then `tables.py`.
- Outputs are in `docs/textbook-problem/textbook72/`: `overview.md`, `tables.md`, `problems.csv`,
  `problems.jsonl`.
- It reuses the fork's `minlen.py`, `patterns*.py` and `textbook_pool.py`, copied unmodified at
  `ab629c7d` into `code/textbook_problem/fork_tools/` (sha256 in `MANIFEST.json`). That folder is
  excluded from ruff, like `fork_pin`.
- CPU only, about 50 minutes, $0.

**Learned.**
- **Proofs:** 71 of 72 have a verified proof (all pass `strict_verify`); one classical problem
  has no proof within 1,800 s.
- **Length:** 54 have a shortest proof found of 7 lines or more.
- **Type:**
  - 19 are classical only (G4ip);
  - 21 need a case split in the shortest proof found;
  - 46 match none of the 40 named laws checked;
  - 9 of the fork's 26 textbook schemas are absent, including contraposition, modus tollens and
    excluded middle.
- **Against pre-training:** relative to `train.jsonl`, the 72 use ORE 20× as often and ∧-elim
  18× as often, and 72% have 2 or more subproofs, against 16%.
- **The model:** the frozen model solves 9 of 72 at all, every one 6 lines or fewer with no case
  split. Every model together solves 1 case-split problem and 1 classical problem.
- **Quirks:**
  - 3 problems have inconsistent premises and a goal that isn't ⊥;
  - 2 have redundant premises;
  - some one- and two-line problems fail for surface reasons.
- **Dmitry's `reference_lines`:** they are the shortest found on 40 of 57 dev problems; the 7+
  count (43) is unchanged.

**Next.**
- Charles to answer the plan's section 8 questions.
- T1 `check.py`.
- Agree target counts per family using this taxonomy.

**Open questions.**
- `CLAUDE.md` says the 72's only uses are scoring and overlap audits. This categorization was
  done at Charles's explicit request. Should the rule allow descriptive, aggregate analysis?
- May the new set's target counts per family be informed by the 72's gaps? That is aggregate
  and not a template, but it is a borderline use.

## 2026-09-28 (Charles + Claude)

**Decided.**
- The new set is a designed grid, not a pile, as described in `plan.md` section 2:
  - slices A–G;
  - 9 families × 6 length bins, 8 per feasible cell;
  - a family is verified on the problem, not taken from how it was generated;
  - schema overlap with the RL pools is tagged and reported both ways.
- v0.1 is slices A and D (about 490 problems).
- The task list is renumbered: T3 generators, T4 v0.1, T5 difficulty, T6 v0.2, T7 write-up.

**Next.** T0 notes, T1 `check.py`, then T3 generators, one family at a time.

**Revised the same day.** Charles: "Authenticity matters and no synthetic problems for now."
- The plan is now real textbook exercises only, with citations.
- Coverage of the thin shapes is a selection goal, not a construction target.
- Encoding rules are fixed, and exclusions are logged.
- The grid design moved to `plan.md` section 9 as a deferred option.
- First source step: ask Dmitry for the original catalogs (112-problem catalog + Smith33),
  including the 73 lost "final" problems.

**Sealed set out of git (Charles).**
- `code/ws3/textbook72/*.jsonl` and the per-problem files with proofs are gitignored and
  untracked; restore them with `bash code/ws3/textbook72/fetch.sh`, which checks the hashes.
- `CLAUDE.md` now allows aggregate descriptive analysis of the 72 and forbids committing them or
  their proofs.
- `overview.md` quotes no problem statements.
- Still in git: the old history, and the committed exp-00/00c eval dumps
  (`experiments/ws3/*/run_*/**/textbook72.jsonl.gz`, prompts plus model completions).
- Later the same day (Charles): **all of the 72 categorization is untracked**:
  `docs/textbook-problem/textbook72/`, `overview.py` and `tables.py` are gitignored and stay
  local. `logic.py` and `fork_tools/` stay tracked for `check.py`. The plan quotes only
  aggregate numbers.

**Diversification notes (Charles).** `docs/textbook-problem/diversification/weaknesses.md`: the
72's weaknesses (aggregate only) and what real exercises would fix them, built on Charles's four
leads: forward vs backward, proof strategy, questions about proofs, reused fragments. It proposes
new measured tags (`direction`, `strategy`, `fragments`, `lemma_of`), a first-move strategy
metric, and a `completion` slice (decide). Not yet folded into `plan.md`.
- `diversification/plan.md`: the plan for collecting real exercises from the web. It has source
  tiers (A open textbooks, B cited literature, C copyrighted only with approval), a 9-stage
  pipeline from fetch to human check, provenance fields, what is committed, and tasks S1–S7.
  Nothing has been fetched yet.
- S1 first pass: `diversification/sources.md`. Found 11 open textbook sources (tier A), 3
  literature sources and 5 free-but-not-open ones, recording pinned commits and licences
  (verified or secondary). Finding: the *forall x* family is largely one exercise pool
  (Cambridge ⊂ Calgary), so tier A gives roughly 120–150 exercises before exclusions. The
  200-problem target probably needs DeLancey (licence conflict), Carnap-hosted books, tier B or
  tier C: Charles to decide. Downloads stayed in the session scratchpad.
- S1 pass 2 (`sources.md`, "Pass 2"). Shallow-cloned the *forall x* forks and matched their
  normalized exercises against Calgary: Syracuse (about 47 new), MSU (about 23) and R³ (about 48)
  add most of the new material, and UBC adds nothing. The *forall x* family has 144 distinct
  sequents (an upper bound). Pelletier P1, P3, P10, P13, P15 and P17 found in TPTP
  (SYN040–047). DeLancey's licence now has 3 conflicting claims. The Carnap shared documents
  (4,583) need a decision on their status. Rough total: 250–300 candidates before exclusions.
- **Transcribed and audited the clear pool** (Calgary, Syracuse, MSU, Carnap Book), with
  `code/textbook_problem/transcribe.py`. No proof search yet.
  - **Output:** `data/textbook_problem/candidates/`: `candidates.jsonl` (105 rows),
    `exclusions.jsonl`, `summary.json`. Raw sources are pinned and hashed in the gitignored
    `data/textbook_problem/raw/`.
  - **Format:** rows match `textbook_dev.jsonl` exactly (prompt rendering, `canonical_hash` =
    sha256 of `canonical_theorem`, name `textbook_<hash[:20]>`). This was checked by
    reproducing all 72 with the same code and by round-tripping all 105. The rows add the
    plan's provenance fields and have no `reference_lines` yet.
  - **Funnel:**
    - 251 sequents transcribed (interderivable pairs count twice);
    - 77 duplicates within the set;
    - 40 overlap a sealed set, dropped and not recorded: 28 distinct classes are in the 72, 12
      in the test files;
    - 29 overlap validation_36, train, heldout or rl_targets;
    - 105 kept.
    - Not transcribed: 14 non-proof items, 12 with more than 4 atoms, 3 invalid (the "prove if
      valid" exercises).
  - **Finding:** 28 of the 72 are renamings of exercises in these *forall x* books, so the 72
    lean heavily on this lineage.
  - **Checks:** a random 25 were compared with the printed source by Claude; all correct. A
    second reader is still needed (plan gate T4).
- **Proof search on the 105 candidates** (`code/textbook_problem/prove.py`; CPU, 8 processes,
  45 min, $0).
  - **Output:** `data/textbook_problem/candidates/candidates_proved.jsonl` and
    `candidates_proved_summary.json`.
  - **Proofs:** 104 of 105 have a proof, and all 104 pass `strict_verify`. The one with no proof
    (Syracuse, derived-rules item 4) failed even with 1,800 s. Methods: 100 whole search, 2
    conjunct split, 1 long search, 1 identity.
  - **Shortest length found:** 1–2: 1; 3–4: 3; 5–6: 15; 7–8: 34; 9–10: 10; 11–12: 21; 13–15: 14;
    16+: 6. So 85 of 104 have 7+ lines, and 59 have 5–10 lines. The 72 have 30 at 5–10.
  - **Family (first match):** negation 24, conditional 22, case-split 16, direct 15, classical
    15, case-split-derived 12. Tags: classical 18, refutation 21 (mostly BOTE inside case
    splits), biconditional 11, nested 11.
  - **Named laws:** only 9 problems match one, and 4 of those are schemas in the RL pools.
  - **Read 20 raw proofs:** all correct.
  - **Quirks:** one identity problem (P ⊢ P, Carnap 4.1) passed every audit.
  - **Correction:** before the search I predicted the new set would be easier on average. By
    length it isn't: 82% have 7+ lines, against 75% of the 72. Difficulty for the model still
    needs T5.
- **Froze `data/textbook_problem/candidate_v0/`:** the 105 problems with proofs, exclusions,
  summaries, a README with the comparison against the 72, and a MANIFEST of hashes (code
  uncommitted).
- **Missing-laws scan** (`diversification/missing_laws_scan.md`,
  `code/textbook_problem/scan_laws.py`).
  - **Coverage:** 14 *forall x* versions plus the Carnap Book (17,148 formulas), the Open Logic
    Text (2,130) and 7 Pelletier problems.
  - **In the textbooks:** 142 occurrences of the 9 laws, but only 17 distinct forms. 14 overlap
    `validation_36`, `train` or `transfer`; the other 3 are truth-table exercises.
  - **Usable:** only Pelletier P1, contraposition as a biconditional (tier B).
  - **Why:** textbooks state these laws in plain letters, and `validation_36` holds exactly
    those. The 9 laws are covered by validation_36, not by any textbook set.
- **Corrections recorded in `sources.md`:**
  - the DeLancey "CC BY 4.0" was the website's footer, not the book's licence;
  - `carnap.io/robots.txt` disallows `/shared/` and sets a 10 s crawl delay. Pass 2 fetched 5
    `/shared` pages, and the book pages were fetched 3 s apart. X1 is now excluded.
- **Decide:** the missing-laws options in `missing_laws_scan.md`. Default: accept the gap and
  add Pelletier P1 to the tier B slice.
- **Searched online for proofs of the 2 problems with none.**
  - **candidate_v0, Syracuse derived-rules item 4:** no published proof found. The Syracuse repo
    has no solutions, Rieppel's teaching page links none, and two web searches found nothing.
  - **The 72's one:** not searched. Querying the web with it would send a sealed item to an outside
    service. Dmitry's reference length is 22, so his witness proof is the source: ask him.
- **Hand-wrote the missing proof for candidate_v0's Syracuse derived-rules item 4.**
  - **The proof:** 21 lines, basic rules only. It passes `nd_verify` and `strict_verify`; a copy
    with one broken citation is rejected by both (negative control).
  - **Recorded as:** `min_lines_method` "hand-written", with the length an upper bound. The
    derived-case-split test was not run for this problem.
  - **candidate_v0 now:** all 105 problems have a proof; 86 have 7+ lines; case-split 17; 16+
    lines 7. The README, summaries and MANIFEST were updated.
  - **Still open:** the 72's one unproved problem. Ask Dmitry for his 22-line witness.
- **Shortened the hand-written proof to 20 lines** (two different 20-line proofs verify). An
  exhaustive `minlen` search at bound 19 completed (716 s) with no proof, so 20 is the minimum in
  minlen's space. candidate_v0 was updated (README amendment, MANIFEST); counts are unchanged.
- **Report written:** `docs/textbook-problem/report_candidate_v0.md`. It covers sources,
  problem types, lengths, form, rules, named laws, quirks and model difficulty against the 72
  (aggregate only).
  - **New finding:** 29 of the 72 appear in the open *forall x* textbooks (26 in exercise
    sections; 16 in five or more versions). The other 43 weren't found in any open source.
  - **Git:** `data/textbook_problem/candidates/` (working outputs) is now gitignored, and the
    frozen `candidate_v0/` is committed.

## 2026-09-29: S1 pass 3, academics and their problem sets

- **Ran:** about 40 web searches and 30 page fetches, person by person (logicians who publish
  exercise sets, course problem sets, proof-checker task banks). Written up in
  `diversification/people_scan.md`. No data or code changed; no carnap.io `/shared/` page
  requested.
- **Learned:** one new tier-A book with its own exercises, Tony Roy's *Symbolic Logic* (CC
  BY-NC-SA 4.0, verified on the author's site). The other distinct sets are unlicensed and held
  by individuals: Oxford (Carr's ND Pack, Halbach's booklet, Fritz), Peter Smith (IFL2 answers),
  Slaney (ANU Logic Notes), Klement (Logic Penguin), Zach (Carnap). Tool task banks (Incredible
  Proof Machine, NaDeA, Jape) are open-licensed but of unstated origin. Course problem sets
  (CMU 15-317, MIT OCW 24.241, Edinburgh Inf1, Chalmers) are small or point at copyrighted
  books. Six PDFs are saved in the scratchpad but unread: no PDF text tool in the venv.
- **Next:** Charles decides on NC-SA licences, on emailing authors, and on adding `pypdf`. Then
  read Roy's PDF and the Oxford packs, and count sequents.
- **Open:** does NC count as "open" for tier A? Do tool task banks count as textbook problems?

## 2026-09-29, later: scraped Roy's Symbolic Logic and Carr's Natural Deduction Pack

- **Ran:** downloaded Roy's PDF (author's site, robots allows) and answers; copied Carr's pack.
  Wrote `code/textbook_problem/pdf_sources.py` (pdfminer for Roy, since pypdf drops ∼ and `]`;
  pypdf for Carr) and added the two sources to `transcribe.py`. `uv add pypdf pdfminer.six`.
  Ran the full transcribe + audit into `data/textbook_problem/candidates_v1/` (gitignored).
  Proof search on the 90 new candidates started detached at 15:20 UTC (PID in `prove.pid`,
  log `prove.log`).
- **Learned:** Roy 140 items → 86 new candidates; Carr 104 → 4 (40 sealed overlaps, 43
  duplicates of forall x). The 105 of candidate_v0 are reproduced unchanged. Details in
  `diversification/people_scan.md`, "Scrape" section.
- **Next:** when the search finishes, compare the 90 with the plan's metrics, then decide with
  Charles on the NC-SA licence and whether to fold them into a candidate_v1 release.
- **Open:** `code/pyproject.toml` also carries another session's uncommitted `lean_tools` line;
  stage only the pypdf/pdfminer lines when committing.
- **2026-09-29 ~16:10 UTC: proof search paused** on Charles's request, with 70/90 done.
  `prove.py` writes its output only at the end, so the process group was suspended, not killed:
  `kill -STOP -- -62765` (uv wrapper, python, 8 workers). Resume with `kill -CONT -- -62765`;
  the log is `data/textbook_problem/candidates_v1/prove.log`. If the machine reboots or the
  processes are gone, rerun the command in the log entry above; nothing is lost but time.
- **2026-09-30 01:18 UTC: resumed** (`kill -CONT`); the 20 remaining finished in 3 min because
  the 1,800 s wall-clock timers of 5 in-flight searches had expired during the pause (log shows
  ~35,000 s). Those 5 were rerun alone (01:22–02:00 UTC, `rerun5*`): 1 proved (E6.21f, 19
  lines), 4 no proof within limits. Merged file: `candidates_v1/new_only_proved_merged.jsonl`.
- **Result for the 90 new:** 80 proved (all pass strict_verify), 10 without a proof within the
  search limits (9 are biconditional-of-biconditional theorems: T6.29, T6.30, E6.18i, E6.18o,
  E6.20c, E6.20d, E6.20e, E6.21j, E6.21m; plus E6.21l). Lengths (79 before the rerun): median
  9, mean 11.3, 16+ lines: 16. Families: conditional 27, case-split 18, negation 13, classical
  10, direct 7, derived 4 (+1 case-split from the rerun). **Six of the nine missing laws appear**
  (contraposition, its converse, export, explosion, modus tollens, dist_and_over_or_conv), as
  Roy's theorem/biconditional forms, none overlapping validation_36.
- **Lesson:** never pause `prove.py` with SIGSTOP; its timeouts are wall-clock. Kill and rerun.

## 2026-09-30: candidate_v0 amended to 104; candidate_v1 frozen (80)

- **On Charles's decision:** v0 keeps only search-proved problems. `textbook_e95c1b499aa32da3b5d3`
  and its hand proof moved to `candidate_v0/removed.jsonl`; README tables and MANIFEST updated
  (16+ 7→6, 7+ 86→85, case-split 17→16, Syracuse 40→39). The report and the Claude Doc still
  describe the 105 version; noted in the README.
- **candidate_v1** = the 80 proved of the 90 new (Roy 76, Carr 4), in
  `data/textbook_problem/candidate_v1/` with README, MANIFEST, exclusions and summaries. The 10
  unproved are in `deferred_no_proof.jsonl` there, outside the set. Working outputs remain in
  the gitignored `candidates_v1/`.
- **Next:** commit (staging only my pyproject lines), then Charles's licence decision, T4, T5.

## 2026-09-30, later: second-reader sheets, release folder

- **T4 second-reader check:** sheets of 20 random rows per set written to
  `docs/textbook-problem/second_reader/` (seed 20260930, links to the exact source lines / PDF
  pages). Charles reviewed them and said they look good; recorded as accepted.
- **Release folder** `data/textbook_problem/release_2026-09-30/`: copies of candidate_v0 (104),
  candidate_v1 (76 Roy; the 4 Carr rows in a separate file, held pending his permission) and the
  10 unsolved, each row with `lean_statement` / `lean_proof` / `lean_theorem` from the pinned
  `lean_seq` tokenizer (`lean.py`; manifest verified, self-test passes; **not checked by Lean**,
  none installed). `lean/*.lean` hold one theorem per problem. README with licences and
  attribution, MANIFEST with hashes.
- **Next:** Charles names the public destination; commit first (my pyproject lines only).
- **Lean installed** (elan, Lean 4.34.1, `~/.elan`) on Charles's request. `lean.py --selftest` now
  runs the Lean check. All 184 release theorems with proofs are accepted by `lean_check` and by
  `lean` directly; the 10 unsolved statements elaborate with `sorry`. Release README, `.lean`
  headers and MANIFEST updated to say so.
- **Carr's 4 rows merged into the release's `candidate_v1.jsonl` (80)** on Charles's decision; the
  pending-permission file removed. README says the 4 carry no author licence and are marked by
  `source.id == "carr"`. Merged `.lean` file re-checked: 80 theorems, no errors.

## 2026-09-30, late: batch 3 sources tried (Pelletier, Oxford, MIT OCW, forall x: R³)

- **Ran:** `other_sources.py` (Pelletier from Isabelle's Classical.thy + TPTP; Oxford booklet
  and Fritz PDFs) and three R³ blocks added to `transcribe.py` PARTS; `clean_item` drops a
  stray comma before ∴. Transcribe runs into `candidates_v2/` and `candidates_v3/` (gitignored);
  proof searches 04:26–05:04 UTC. Merged results: `candidates_v3/batch3_proved.jsonl` (21 rows).
- **Learned:** MIT OCW's sets only cite *The Logic Book* (unusable). Oxford has 10 propositional
  ND exercises, 3 new. Pelletier 17 → 9 new; R³ 44 → 9 new (31 duplicate Calgary by renaming
  class; pass 2's string-match count was misleading). Proofs: 17 of 21; unsolved: Pelletier P9,
  P12, P14, P17 (nested biconditionals and a 3-conjunct antecedent). Lengths of the 17: median
  10, max 19; families case split 7, classical 4, direct 2, derived case split 2, conditional 1,
  negation 1; no named-law matches; 5 classical-only.
- **Next:** Charles decides whether to freeze these 17 (+4 unsolved) as candidate_v2 or hold
  them for a larger batch (Lorain 85, Bristol 28).
- **Unsolved set extended to 14:** Pelletier P9, P12, P14, P17 added to
  `candidate_v1/deferred_no_proof.jsonl` and to `release_2026-09-30/unsolved.jsonl` (+ Lean
  statements with `sorry`, file re-checked with Lean: 14 theorems, no errors). READMEs and
  MANIFESTs updated. Not committed.

## 2026-09-30, evening: reference proofs for the unsolved set by decomposition (approach 3)

- **Method:** `scratchpad/decompose.py` (proof of concept, not in the repo yet): apply the
  introduction rules backwards (∧I, →I, ¬I) until the goal is an atom, ∨ or ⊥; search each leaf
  with `minlen` (45 s; 300 s once every letter is decided); if a leaf fails, derive P ∨ ¬P
  (9 lines) and case-split with ∨E; splice, renumber, verify with nd_verify and strict_verify.
- **Machine results on the 14:** proofs for E6.18i (32), E6.18o (33), E6.20d (119), E6.21j (61),
  E6.21l (34), E6.21m (33), P9 (39), P14 (45), P17 (159). Failed: T6.29, T6.30, E6.20c, E6.20e,
  P12 (a leaf timed out even after splits; to inspect).
- **Hand-trimmed, verified (broken-copy control rejected):** E6.18i 30 (minimal by a rule-count
  argument), E6.18o 23, E6.21l 20, P9 26. Saved to `candidates_v3/hand_proofs.jsonl`
  (gitignored); not yet folded into the sets.
- **Lesson:** the search fails on depth, not difficulty; decomposition alone gives proofs 10–30%
  longer than hand ones (re-derived ∧E lines, 9-line LEM); hoisting shared eliminations is the
  obvious next improvement.
- **Five more hand proofs (2026-09-30, late):** E6.21m 29 (machine 33), P14 45 (45), E6.21j 51
  (61), E6.20d 71 (119), P17 73 (159); all pass nd_verify and strict_verify. Assembled with
  `scratchpad/hand5.py` (automatic numbering). `candidates_v3/hand_proofs.jsonl` now holds 9.
  Still without a proof: T6.29, T6.30, E6.20c, E6.20e, P12.
- **E6.20d and P17 vs the 400-token cap:** measured 806 and 1,245 tokens (71 and 73 lines; 11.4
  and 17.1 tokens per line). Probed the verifier for shortcuts: it accepts ∨E/→I boxes whose end
  is the assumption, ∧I of one line with itself, and ⊥E straight to a compound, but not →I
  citing a line outside the box. None of these shortens either proof materially; E6.20d needs
  six biconditional constructions (one per truth-value pattern), P17's formulas alone are
  ~17 tokens a line. Both stay above the cap; they are unsolvable at the default `max_new`.
- **The last five hand proofs:** T6.30 102 lines (1,768 tokens), P12 102 (1,768), E6.20e 51
  (686), T6.29 46 (757), E6.20c 43 (650); all pass nd_verify and strict_verify
  (`scratchpad/hand_rest.py`, with reusable biconditional-construction helpers). All 14 of the
  unsolved set now have a verified reference proof in `candidates_v3/hand_proofs.jsonl`; all 14
  exceed the 400-token generation cap. Not yet folded into the sets.

## 2026-09-30, night: Lean datasets and Lean course repos

- **Ran:** web survey of Lean proof datasets; shallow clones of six Lean course/textbook repos
  plus core `Init/PropLemmas.lean`; regex scanner `candidates_lean/lean_scan.py` (gitignored
  folder) to transcribe propositional statements and audit them by renaming class; `prove.py`
  on the 87 new (120 s / 300 s, 6 procs, 880 s wall).
- **Learned:** the big Lean datasets (miniF2F, Lean Workbook, LeanDojo/Mathlib, ...) have
  essentially no propositional problems, and the propositional Lean datasets (PropL, etc.) are
  synthetic. The usable source is Lean course material: 694 propositional statements, 87 new
  renaming classes, 80 proved (strict_verify ok), 56 at 5–10 lines. Table and caveats in
  `diversification/sources.md`.
- **Next:** Charles decides `questions.md` items 1–2; then hand-check each kept row against its
  file (the scanner misses section-level hypotheses), rerun the 7 unproved with the standard
  300/1800 s limits, and scan LEAN-GitHub for more course repos.

### 2026-09-30 (late): proof-completion problems, proof of concept
- Charles asked for a couple of "finish the proof" problems: a sequent plus a nearly complete proof, the task being to supply the remaining lines. Built two from candidate_v1 reference proofs (Roy E6.18p and E6.15k; never from the sealed 72), each cut with three lines left and two boxes open at the cut. Written to `data/textbook_problem/candidates_completion/poc.jsonl` (gitignored working folder).
- Record fields: `prompt` (the sequent), `given_proof` (the prefix), `completion_prompt` (= prompt + prefix, what a model would be fed), `reference_completion`, `n_given_lines`, `n_remaining_lines`, `open_boxes_at_cut`, and `checks`.
- Checks run with the pinned `nd_verify` and `strict_verify`: prefix + completion accepted (both); prefix + QED alone rejected ("proof ends inside a subproof"); a bogus completion that just reiterates the goal rejected ("bad line cite"). Scoring a completion is therefore just: concatenate prefix and model output, run the usual verifier. No reference completion needed.
- Open design points (not decided): how to choose cut points at scale (fixed k lines from the end vs. "after the last leaf, before the closing rules"); whether to count tokens of the prefix against `max_new` (the prefix is prompt, so no); whether a model trained on whole proofs continues a given prefix at all (it should, since the training format is prompt-then-proof).

### 2026-09-30 (late): "easier backwards than forwards" problems, proof of concept
- Four constructed problems (not from any source; audited by renaming class against the 72, test files, validation_36, heldout, transfer, rl_targets, train, candidate_v0, candidate_v1: no overlap) in `data/textbook_problem/candidates_backward/poc.jsonl`, script `scratchpad/backward_easy.py` (not in the repo yet). Design: goals whose main connectives are →, ∧, ¬ so that introduction rules decompose them deterministically; premises that are inert forward (nothing fires) or form a forward cycle unrelated to the goal.
- Measurement: backward = scratchpad Decomposer (intro rules on the goal, minlen on leaves); forward = a blind forward BFS written for this check (elimination rules, ∧I/∨I within subformulas, open a box on any subformula, close by →I/¬I; no goal information; states expanded until the goal is at top level). This BFS is my own toy, not the pinned search; the fork's minlen is itself goal-directed, so it cannot serve as the forward side.
- Results (tag: backward lines/leaves/secs vs forward BFS states expanded): tower 8 (11 before trimming)/0/0.0 s vs found at 14,846; three-boxes 21/4/0.0 s vs not found at 2,000,000 (30 s); negated-nest 11/1/0.35 s vs found at 936,445 (6 s); red-herrings 12/1/0.0 s vs not found at 2,000,000 (21 s). All four backward proofs pass nd_verify and strict_verify. Elimination-only closure of the premises: 0, 0, 0, 4 formulas; the red-herrings proof cites no premise at all.
- Caveat: a trained model is neither prover; whether these are "easier" for it is an empirical question for a checkpoint run. The forward cost also depends on the subformula space size, which grows with the goal.
- **Correction to "87 new":** "new" was by renaming class only. Moving a theorem's leading
  antecedents into premises (Lean states sequents curried) turns 25 of the 87 into a known
  problem: 9 sealed (counted only), 10 validation_36, 9 our sets, 4 train, 1 heldout (some hit
  several). Left: 62 (59 proved; 7–10 lines 29, 5–6 10, ≤4 7, 11–15 7, 16+ 6; conditional 23,
  direct 14, case-split 9, negation 8, derived 3, classical 2; 3 classical-only). The renaming-
  class audit (here and in v0/v1) does not catch curried twins; whether it should is a question.

## 2026-09-30, night: v0 and v1 re-audited up to currying

- **Ran:** `data/textbook_problem/candidates_audit_curry/audit_curry.py` (gitignored folder;
  hits incl. sealed ones are listed only there). Normal form: move every leading antecedent of
  the goal into the premises, then renaming class as before. Secondary mode also splits
  conjunctive premises. Reference sets: sealed (72 + test files), validation_36, heldout,
  transfer, rl_targets, train; plus twins among v0, v1 and the unsolved 14. The old exact audit
  re-checks at 0 hits.
- **Curry mode:** v0 19 of 104 flagged: 16 match a reference set (sealed 6, validation_36 8,
  train 9, heldout 2, transfer 1; some several), 3 are twins only inside our sets. v1 8 of 80:
  7 reference (sealed 2, validation_36 2, train 6, rl_targets 2), 1 internal only. One twin
  class has 4 members (2 in v0, 2 in v1) and is a sealed twin. Unsolved 14: none.
- **Curry + conjunction split:** v0 24 (21 reference, 9 sealed), v1 18 (15 reference, 6 sealed).
  This mode also catches biconditional symmetry ((A→B)∧(B→A) vs (B→A)∧(A→B) as premises), e.g.
  Roy T6.9 / T6.10 are one class. Splitting a conjunctive premise changes the exercise (no ∧E
  needed), so this mode is informational.
- **Not changed:** v0, v1 and the release folder. Removing flagged rows is Charles's call
  (questions.md item 3).

## 2026-09-30, night: unreleased bundle release_2026-09-30_v2 (PAUSED on Charles's request)

- **Built (first pass, not final):** `code/textbook_problem/build_release.py` →
  `data/textbook_problem/release_2026-09-30_v2/`. Curry-aware audit applied: v0 104 → 87,
  v1 80 → 72, handwritten 14 → 14, Lean 76 → 49 proved + 3 unsolved, lean_core (4) left out.
  All rendered proofs pass strict_verify and Lean 4.34.1; 4 handwritten proofs (>64 lines) can't
  be rendered by lean_seq and ship as statements with `sorry`. Textbook 72 copied to the
  gitignored `textbook72/` subfolder (new `.gitignore` rule `data/textbook_problem/release_*/textbook72/`).
- **Pending, not yet in the folder:** the build script now also drops 2 Lean rows that are
  trivial after encoding (lean4game IffIntro L02, M40001 questions.lean:33); the
  folder still holds the earlier pass. No README yet; MANIFEST's sorry counts are wrong (fixed in
  the script, not rebuilt).
- **Stopped:** the rerun of the 3 unproved Lean rows (prove.py 300/1800 s), killed (PIDs I
  started) at 2 of 3 done, both "no proof"; output was not written (prove.py writes at the end).
  Resume: `cd code && uv run python textbook_problem/prove.py --inp
  ../data/textbook_problem/candidates_lean/rerun_in.jsonl --out
  ../data/textbook_problem/candidates_lean/rerun_out.jsonl --secs 300 --long_secs 1800 --procs 3`,
  then rerun build_release.py (command in its docstring; `--leansrc` is the scratchpad clone dir,
  re-clone at the commits in `candidates_lean/sources_commits.txt` if it is gone), then README.
- **Resumed and finished (same night):** rerun of the 3 unproved Lean rows completed: no proof
  for any (2 × conjunct split 600 s, 1 × 1,800 s), so they stay in `lean_unsolved`. The trivial
  filter is narrowed to ↔-encoding artifacts (2 rows); two rows trivial as printed (P, Q, R ⊢ P
  and P, Q ⊢ P) went through the audit instead and were removed as twins. Final bundle:
  v0 87, v1 72, handwritten 14, Lean 47 + 3 unsolved, removed 38 (+ 20 sealed, counted only),
  textbook72 local copy (gitignored). All 220 proofs pass strict_verify; 216 rendered proofs
  pass Lean 4.34.1; all .lean files compile. Independent re-audit with `audit_curry.py`: 0
  twins. README written. Not committed, not released.
- **Next:** Charles reviews the bundle; questions.md items 1–3; a second-reader check of the
  handwritten and Lean sets; T5 baseline.
- **Summary doc:** "Textbook ND datasets: bundle v2 (2026-09-30)", a Claude Doc at
  https://claude.ai/code/artifact/866b2766-fe37-45b3-a514-eee70fa47bfb (private): sets, comparison
  chart and table, what changed, checks, open decisions.
- **Decision (Charles, 2026-09-30):** curried and uncurried forms are the same problem for the
  overlap audit. Written into plan.md section 3 and questions.md item 3; the bundle already
  applies it, so no rebuild. Questions 1 and 2 set aside; defaults stand.
- **Independent duplicate rescan (same night):** fresh script enumerating all 24 letter
  permutations, premise order ignored, plain and curried forms: 0 exact, hash or name
  duplicates in the bundle; 0 twins within or between our sets; 0 against the 72, the test files
  and the five pools. Sanity: a renamed, reordered, curried copy of a v0 row is caught. The 72's
  source files are untouched; the bundle copy is the source plus `lean_statement`.

### 2026-10-01: derived problems, 30 scorable nuances × 10
- Charles asked for 10 unique problems per scorable nuance from the lists of 2026-09-30/10-01, varying rules and statements. Built `code/textbook_problem/derive.py` (in the repo, uncommitted): loads the 198 pool problems with verified proofs (candidate_v0 104, candidate_v1 80, hand proofs 14), picks bases per nuance by a variety-first greedy (unused base, unseen rule set, unseen source), builds each derived proof by splicing the base proof into new scaffolding (a `PB` builder with `splice`), and checks everything: nd_verify + strict_verify on the full proof; prefix-alone and bogus-completion rejected for prefix tasks; `structural_ok` for the verifier+check nuances; token caps (input 624, output 400); renaming-class audit against the 72, test files, every named pool and the pool itself; Lean 4.34.1 on every rendered theorem (`--lean-only` pass).
- Output `data/textbook_problem/candidates_derived/` (gitignored working folder): 30 jsonl files, 320 rows (hint_pairs 30 = 10 × 3), 175 distinct bases, 7 sources, Lean 320/320, no audit hits; README with the per-nuance table and the scoring conventions; `lean/<nuance>.lean`; `summary.json`.
- New proofs were needed only for `demorgan` (fork minlen, bound 40, 120 s; 10/10 found in 19 s total; run detached per the long-run rule). Every other transformation is proof surgery, so the derived proofs are correct by construction and then verified.
- Design choices worth knowing: whole-proof tasks (repair, cloze, gap, shorten) use marker words BROKEN/CLOZE/GAP/SHORTEN in `task_input`, which no trained model has seen; `open_dead_end` opens a box on the negated goal when the reference does not; `distractor_scaling` is 5 bases × {2, 4}; `permutation` rows share their base's renaming class by design; `contrapositive` prefers a non-negated premise to swap but falls back when all premises are negations.
- Dropped during generation (refilled from other bases): 1 wrapper, 3 permutations and 1 composition over the 400-token output cap; 1 completion over the input cap. No sealed-class hits.
- Next: second-reader read of a sample; decide whether to promote any nuance into a release; commit derive.py; scoring harness for `task_input` (prefix concatenation and the structural checks) on a checkpoint.

## 2026-10-01: batch 4 (Isabelle, Metamath, LEAN-GitHub, more Lean textbooks)

- **Ran:** readers for Isabelle `FOL/ex` and Metamath `set.mm` (cited `$p` only), a filter over
  the LEAN-GitHub parquet (first tactic state, Prop-only binders), LAMR's exercises from the
  published HTML, and a second Lean repo scan (MIL, Mechanics of Proof, FM 2022/23, LoVe 2025,
  tutorials, Robo; Lean 3 `∀ P Q : Prop,` fix). Merged and audited up to currying against the
  sealed sets, the five pools, the v2 bundle and each other: **184 new** (Metamath 118, Isabelle
  33, Lean sources 33); 7 held as not textbook (SMT tests, competition sets, a personal repo).
  Table in `diversification/sources.md`. Proof search on 177 started (PID 18425, 300/1,800 s,
  6 procs); slow: 7 done after 10 min, many are biconditionals.
- **Licences:** Metamath CC0 (file header), Isabelle BSD-3 (COPYRIGHT), HTPI Apache-2.0, LAMR
  none stated.

### 2026-10-01 (later): derived set pruned for conceptual overlap, 30 → 21 nuances
- Charles asked whether the nuances overlap, then to clear out the overlapping ones. Measured first: no derived sequent class appeared in two transformation nuances; base reuse was 61/108/6 bases in 1/2/3 nuances; but conceptually several pairs were the same mechanic (completion_almost_nothing = hint_pairs right; open_dead_end = hint_pairs wrong; red_herring = distractor_scaling k=1; gap ≈ cloze; strengthened_premise ≈ near_miss; contrapositive ≈ negated_conjunction; composition ≈ conjoined; shorten ≈ line_budget; completion_midleaf between two kept cuts).
- Pruned those nine (`derive.DROPPED` with reasons; files moved to `candidates_derived/overlap_dropped/`, not deleted). `wrappers` lost its X → g form (curry's shape); `distractor_scaling` now runs k = 1, 2, 4. Kept nuances grouped in `derive.FAMILIES` (5 families). Regenerated the 21 kept nuances (deterministic; picks shifted because the usage counter no longer sees the dropped nuances), Lean pass 232/232, audit clean. 3 rows dropped and refilled for the output cap (curry, wrappers, permutation).
- Set now: 232 rows (19 × 10 + hint_pairs 30 + distractor_scaling 12 from 4 bases; the fifth base had no 4 jointly satisfiable distractors), 170 bases, 136 used once and 34 twice, no two nuances sharing more than 4 bases. README rewritten with the pruning table; MANIFEST regenerated; Claude Doc updated.
- Charles asked whether the model is told the nuance. It is not: `task_input` is the only input, no instruction channel exists. Checked the pinned tokenizer (`code/ws3/fork_pin/tokenizer.py`): closed vocabulary of 19 symbols + 15 rules + 64 line refs, `encode_prompt` is a dict lookup, so the marker words BROKEN/CLOZE and `?` raise KeyError. Consequence: repair and the two clozes cannot be fed to the pinned checkpoint (specification only, need a widened-vocab model); prefix nuances are encodable (prompt encoder + proof encoder for the given lines) but the evaluator must be changed to seed the decoder with a prefix; the constraint nuances measure default habit, not instruction following. Recorded in the README and the Claude Doc.
- **Proof search finished (177 rows, 300/1,800 s, 6 procs; `candidates_b4/proved.jsonl`, log
  beside it). All found proofs pass strict_verify.
  - Isabelle 33: 22 proved (median 12 lines, max 23; negation 9, conditional 5, case-split 5+1,
    classical 2); 11 unproved, 9 of them ¬¬-wrapped classical biconditionals from
    `Intuitionistic.thy`.
  - Lean additions 26: 23 proved (median 11, max 24; 8 classical-only); 3 unproved (LAMR ch8-4,
    HTPI not_not_iff, a Robo biconditional).
  - Metamath 118: 81 proved (median 13, max 53; 32 at 16+; classical 16; 3 over 400 tokens);
    37 unproved, nearly all biconditional-of-biconditional theorems (imbi1, bibi1, pm5.74, ...).

### 2026-10-02: Robbie's newest pretrains scored on every dataset (eval_scale_pretrain)
- Charles asked to test Robbie's most recent pretrain on the datasets. Newest on the HF bucket: the `scale-*-p1200` family dated 2026-09-30 (d384 9.6M and d768 38M params, three seeds each), tokenizer mode `lean_seq`: the model reads the Lean statement and writes Lean tactic text; the fork at 51604b3 (`lean_tok.inverse`) decodes it to ND. Built `code/textbook_problem/eval_ckpt.py`: loads the run's own `source/pretrain.py` (ALiBiGPT + PeriLN) on the fork clone, samples with the fork's `generate_ids`, judges with the pinned nd_verify through strict_verify, seeds prefix tasks with the lean_seq prefix of the reference proof (open-box headers unavoidable). Pipeline validated on fork heldout (0.977) and rl_targets (0.223 at temp 0.8), matching the run's own dev metric.
- Pre-registered `experiments/textbook-problem/eval_scale_pretrain/proposal.md` (commit 2c5d69c), ran both s0 pretrains on the laptop CPU, k=32, temp 1.0, max_new 400, seed 0, ~11 min and ~23 min. Results in `results.md`; every judged sample dumped; the 72 dump in the gitignored textbook72 folder.
- Headline (avg solve rate / solved at least once): d384: v0 0.111 / 22 of 104, v1 0.050 / 11 of 80, hand14 0 / 0, batch3 0 / 0, lean_v2 0.088 / 11 of 50, derived 0.038 / 20 of 202, textbook72 0.104 / 12 of 72. d768: v0 0.086 / 17, v1 0.058 / 11, hand14 0 / 0, batch3 0.002 / 1, lean_v2 0.091 / 9, derived 0.036 / 17, textbook72 0.129 / 11. Average rates inside the pre-registered bands, any-of-32 counts below them on every set. Derived: only completion_closing (0.20), permutation (0.13), forward_control (0.12) above 0.1; wrong_turn, wrappers, conjoined, negated_conjunction, distractor_scaling, force_ore at zero.
- Stop rule (parse-fail > 0.5) crossed on hand14 (both) and derived (d768): the failures are the `unbound` class (a cited name never introduced), the same class Robbie's lean_gate reports at 16% on his targets; judged model behaviour, runs allowed to finish, documented. 20 raw samples read per checkpoint.
- Lessons: the first d768 launch died at load because only the run's summary had been fetched (`ModuleNotFoundError: pretrain`); the scorer should fetch `source/` and the `.pt` itself. One seed per size: no size claim.
- Next: score the RL-stage `final.pt` of the same runs; three seeds before any comparison; re-audit the derived bases under the curry-aware rule; decide whether to fold the 14 hand proofs and batch 3 into a release.

### 2026-10-02 (evening): Dan's best-state cap-12 RL model scored; seed 2 paused; doc and deck
- Charles: "Dan should've released some new RL models — check there." Found Dan's bucket `dan-pandori/nd-rl` (listed via `https://huggingface.co/api/buckets/dan-pandori/nd-rl/tree/<prefix>`): best-state (done 10-01), trajectory, rl-from-ckpt (done 10-02 10:54), grpo-best (uploading). Scored `best-state/ckpts/bs/ladder/la_T1_best12_s{0,1,2}_r8.pt` (Robbie's 9.56M network, lean_staten proof-state format, 8 EI rounds) with Dan's own `state_eval.py` (branch `dan` 0c492902, worktree in scratchpad) on CPU with Lean 4.34.1, then re-judged every accepted proof with the pinned verifier (`code/textbook_problem/dan_metrics.py`). Pre-registered `experiments/textbook-problem/eval_dan_best_state/proposal.md` (commit 6898cdc) with launch.sh and inputs.
- Seeds 0 and 1 (k=32, temp 1.0): v0 0.343 / 73–76 of 104, v1 0.359 / 49–50 of 80, hand14 0 / 0, batch3 0.31 / 10–12 of 21, lean_v2 0.39 / 31–33 of 50, derived_plain (13 nuances, 132) 0.28 / 66–69, textbook72 0.375 / 47–49 of 72 (aggregate). Against Robbie's d384 pretrain (one seed): 3–6× more problems solved at least once; every problem a pretrain solved the RL model solved; RL gain is at 7–20 reference lines, nothing at 21+; RL proofs padded (13 lines vs 5–10 references). Lean vs pinned verifier disagree on 2 accepted proofs.
- Leak audit (numbers beat pre-registered bands on 4 sets): K12 pool (md5 matches Dan's record), rl_targets, rl_targets_v1, transfer, transfer_v1, heldout vs every set by renaming class: ≤ 3 problems per set. `run_best12_r8/leak_audit.md`.
- Seed 2 stopped at Charles's request (18:10 UTC) by killing the launcher and evaluator PIDs (no SIGSTOP, per the 09-29 lesson), then cancelled outright; nothing of seed 2 was written. Experiment closed with two seeds. Results and the two-seed table in `results.md`.
- Deliverables: Claude Doc "RL vs pretrain on textbook problems" (3 charts, RL-only analysis, nuance ordering) and a 3-slide minimal deck. 20 raw samples read per checkpoint.
- Next: resume seed 2 when asked; score grpo-best when its run is DONE; prefix seeding for the state env; curry-aware re-audit of the derived set.
- Per-problem analysis (Charles: "exactly which questions the pretrain and RL trained models are getting right, at which length"): `code/textbook_problem/eval_analysis.py` → `experiments/textbook-problem/eval_dan_best_state/analysis/{analysis.json, per_problem.csv, per_problem.md}`; the Claude Doc got a "Per-problem analysis" section with a chart by exact reference length and a second tab listing every non-sealed problem with both models' rates (the 72 stay aggregate). Findings: pretrain solved 44 of 269 scraped problems, RL 183 (either seed), pretrain-only 0, containment holds even for the union of both pretrains vs the union of both RL seeds; pretrains agree with each other less (Jaccard 0.61) than RL seeds (0.83); RL holds 79–100 % through 10 reference lines, 55–67 % at 11–13, 12 of 36 at 14–20, 1 of 31 at 21+; the knee at 12 lines matches the flat 2–12 cap of Dan's K12 pretraining pool; all 7 four-premise problems unsolved by every model, and K12 / rl_targets contain only 0–3 premises; ∨E problems 49 of 77 and DN problems 21 of 56 solved by RL (pretrain 4 and 4); RL's shortest accepted proof per problem is at or below the reference on 132 of 183 (padding is per-sample); derived rows follow their base (76 / 20 / 5 / 31). Earlier chart bins had mis-filed 8 Lean-sourced problems with no `reference_lines` field as ≤ 6 lines; corrected (1–6: 47, 21+: 31).

### 2026-10-02 (night): release_2026-10-02 built and published
- Charles asked to publish the new problem sets and confirmed: candidate_v0 and v1 copies, the hand-proved 14, batch 3, the derived set, aggregate baselines, curry-aware audit. He also asked for "the textbook set"; the 72 are sealed and gitignored and were not included (said so).
- `code/textbook_problem/release_1002.py` (curry-aware helpers copied from the v2 bundle's build_release.py so the cut matches it): candidate_v0 104 → 87, candidate_v1 80 → 72 (identical cut to the v2 bundle), hand_proved_14 14, batch3 17 → 14 (2 sealed twins counted only, 1 twin of a v1 row), derived 232 → 225 (3 sealed twins counted only, 4 pool twins removed; 144 rows tagged `curried_twin_of` a kept row, by design). 412 problems; every proof re-verified with the pinned verifier; Lean 408/412 (4 hand proofs exceed lean_seq's 64 names, statements with sorry). `removed.jsonl` 22 rows with reasons; `baselines.md` with the 2026-10-02 scores and checkpoint hashes; README states that batch 3, the hand-proved 14 and the derived set have had no second-reader check.
