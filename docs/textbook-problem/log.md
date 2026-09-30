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
