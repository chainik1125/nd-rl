# Diversification plan: collecting real textbook problems from the web

A sub-plan of `docs/textbook-problem/plan.md`, for its tasks T3 (sources) and T4 (v0.1). The
parent plan's rules hold here unchanged:
- authenticity first, no synthetic problems;
- the encoding rules and exclusion log (parent section 3.1);
- the overlap audit;
- CPU only, $0.

**What to collect** comes from `weaknesses.md` in this folder. **How** is this file. Items marked
**(decide)** are Charles's call. Until he answers, use the stated default and record it in
`docs/textbook-problem/questions.md`.

## 1. Goal

> Find, fetch and transcribe propositional natural deduction exercises from openly available
> textbooks and the published literature, each with a citation and a record of where it came
> from, so the new set can be selected toward the gaps in `weaknesses.md`.

"Scraping" here means **downloading published sources and extracting their exercises**, from
LaTeX source where it exists and from PDF or HTML otherwise. It does not mean harvesting forums,
Q&A sites or homework-help pages. Their authorship and correctness can't be verified, and they
often copy copyrighted books.

**Targets for v0.1 (from `weaknesses.md` section 6):**
- **Middle lengths:** 5–10 line proofs.
- **Strategy classes:** reductio and indirect proof, case split, case split on a derived
  disjunction.
- **Direction:** forward-heavy exercises, whose goal is an atom or a ∨.
- **Missing standard laws:** contraposition, modus tollens, excluded middle, constructive dilemma,
  export, explosion.
- **Lemma and compound pairs:** derived-rule chapters.
- **Completion prompts:** "complete the proof" exercises, only if Charles approves the
  `completion` slice.

## 2. Sources, in order of preference

Every licence below must be **checked at the source before fetching** and recorded in the source
manifest (section 4). The licences given are what to expect, not a verified fact.

| Tier | Source | Expected licence | Why | Format |
| --- | --- | --- | --- | --- |
| A | *forall x: Calgary* (Zach, after Magnus) | CC BY | The plan's default. It has a derived-rules chapter and a proof-strategy chapter. | LaTeX source on GitHub, and Carnap exercise sets |
| A | Other *forall x* remixes (Cambridge, UBC, Magnus's original) | CC BY / CC BY-SA | The same style, with different exercise lists | LaTeX or PDF |
| A | Open Logic Project, sentential logic volume | CC BY | Natural deduction chapters with exercises | LaTeX source on GitHub |
| A | Other open logic textbooks found in discovery (T-S1) | Check each | Coverage and style variety | Varies |
| B | Pelletier's "Seventy-five problems for testing automatic theorem provers" (1986), propositional problems | Published paper, cited problems | Standard, much-cited; strong on classical and case split | PDF, also in TPTP |
| B | TPTP / ILTP propositional problems, **only** those whose header cites a textbook or paper | TPTP terms | Machine-readable, citation in the header | Plain text |
| C | Copyrighted textbooks (e.g. Lemmon, Bergmann–Moor–Nelson, Huth–Ryan, Teller) | Copyright | The classic exercise lists | Only from a copy Charles provides; **(decide)** |
| – | Forums, Q&A sites, course pages without a clear licence, answer-key sites | – | **Excluded** | – |

**Tier rules.**
- Tier A is the default and is reported as its own group.
- Tier B problems are real and cited, but they come from the literature, not from textbooks.
  They carry `source_kind: literature` and are reported separately. **(decide)**: whether tier B
  counts toward the headline. Default: no, a separate slice.
- Tier C is used only for internal evaluation, never published, and only with Charles's approval
  (parent plan section 4).
- **Solutions.** Where a book prints solutions, keep them as `printed_proof` for cross-checking.
  They are never the source of truth; the reference proof is always rechecked with
  `strict_verify`.

## 3. Pipeline

Each stage writes to its own folder, and every stage can be rerun from the one before it.

| Stage | Does | Output |
| --- | --- | --- |
| 1. Discover | Lists candidate sources with URL, licence, access date and exercise count. By hand plus web search, no crawling. | `sources.md` (tracked) |
| 2. Fetch | Downloads each approved source once. Respects `robots.txt` and the site's terms, waits at least 5 s between requests to one host, sets a user agent naming the project, and caches the result. | `data/textbook_problem/raw/<source>/` (gitignored) plus `raw_manifest.json` (tracked: URL, date, sha256, licence) |
| 3. Extract | Finds exercise blocks with a deterministic parser for each source: LaTeX environments first, then PDF text with page numbers. | `extracted/<source>.jsonl`: `original` text, location (chapter, section, number, page) |
| 4. Transcribe | Parses the book's notation into the fork's formula format, applying only the encoding rules in parent section 3.1. | `candidates.jsonl` with `encoding_notes` |
| 5. Validate | Checks classical validity, the atom limit, the tokenizer round-trip and that it is a proof task at all. Everything that fails goes to the exclusion log with a reason. | `candidates.jsonl`, `exclusions.jsonl` |
| 6. Audit | `canon_key` overlap against the 72, `train.jsonl`, heldout, `rl_targets`, `transfer`, `validation_36`, the test files, and within the set. Only aggregate counts are reported for overlaps with sealed sets. | `audit.json` |
| 7. Prove and tag | Runs the fork's `minlen` for the reference proof and shortest length, then `check.py` for family, length bin, strategy, direction and fragments. | `tagged.jsonl` |
| 8. Select | Picks toward the targets in section 1 and `weaknesses.md`, then reports cells the sources can't fill. **Never drops a problem because of the model's score.** | `v0.1/problems.jsonl` |
| 9. Human check | A second reader checks 20 random problems per source against the printed page. | `human_check.md` |

**Code** goes in `code/textbook_problem/scrape/`: one fetch script, and one extractor per source.
Each script records its inputs and outputs in the run folder.

### 3.1 Using an LLM for extraction (decide)

**Default: no LLM at any stage.** Transcription uses deterministic parsers only. LaTeX sources
make this easy; for PDF sources, one hand-written parser per book.

**Allowed if Charles approves:** an LLM may *propose* a transcription of a hard-to-parse exercise.
It is kept only if:
- it round-trips through the parser;
- a human compares it with the printed text;
- `encoding_notes` records "LLM-assisted".

An LLM never supplies a problem, a premise or a goal that isn't printed.

## 4. Records and what is committed

**Every problem carries:**
- `source`: book, edition, chapter or section, exercise number and page;
- `source_url`, `access_date`, `licence`;
- `original`: the exercise as printed;
- `raw_sha256`: the hash of the downloaded file it came from.

A reader can then go from any problem back to the exact file and page.

**What is committed:**

| Committed | Kept local (gitignored) |
| --- | --- |
| `sources.md`, `raw_manifest.json`, the scraper and extractor code, the exclusion log for CC-licensed sources, aggregate audit counts | Raw downloads; any tier C text; per-problem overlap with the 72 or the test files; anything whose licence forbids redistribution |

Tier A problems may be committed with their licence and attribution. Tier B and C problems stay
local until Charles decides (parent plan section 7).

## 5. Tasks

In order. A gate failing twice means stop and ask.

- **S1 Discover.** List candidate sources in `sources.md`, with licence, format and an estimated
  count of propositional natural deduction exercises. Gate: at least 3 tier A sources with
  verified licences, or a note of why not.
- **S2 Fetch tier A.** Fetch the sources, following the stage 2 rules. Gate: `raw_manifest.json`
  is complete, and every hash matches a second download.
- **S3 Extract and transcribe *forall x: Calgary*.** Do this book first, end to end through stage
  7. Gates:
  - the extracted exercise count matches a hand count of the book's exercise lists;
  - 20 random transcriptions match the printed page.
- **S4 The other tier A sources.** Same process as S3, one source at a time.
- **S5 Tier B.** Pelletier's propositional problems, and cited TPTP/ILTP problems, as a separate
  slice. Only if Charles approves tier B.
- **S6 Selection and gap report.** Counts by source × family × length bin × strategy × direction,
  against the targets. A list of the cells no source can fill. Gate: every target has either a
  count or a "not available" line.
- **S7 Hand-off.** The candidates go to parent plan task T4 (`check.py`, release v0.1).

## 6. Risks and how they are handled

| Risk | Handling |
| --- | --- |
| **Leak into the sealed set.** The 72 came from textbook catalogs, so the same exercises will turn up. | The stage 6 audit removes them by renaming class. Only the count is reported. Never inspect which of the 72 matched. |
| **Leak into RL pools.** The fork's `rl_targets` and `transfer` hold textbook schemas. | Renaming classes are removed. A shared schema is tagged `schema_in_rl_pool` and reported both ways (parent plan section 2). |
| **Duplicates across books.** Classic laws appear in every book. | Keep one per renaming class, and record every other book it appears in under `also_in`. Prefer the tier A copy. |
| **Transcription errors.** | Deterministic parsers, the round-trip check, a validity check, and a 20-per-source human check. An invalid "exercise" is logged, never silently fixed. |
| **Licence mistakes.** | Check the licence before fetching, and record it with the date. When unsure, treat the source as tier C. |
| **Load on sites.** | One download per file, a 5 s delay, `robots.txt` obeyed, no crawling beyond the listed URLs. |
| **Too few exercises in the thin cells.** | Report the gap (S6). Don't fill it with synthetic problems; a synthetic grid is a separate, deferred decision (parent plan section 9). |

## 7. Budget and stop rules

- $0: CPU only, free sources, no paid APIs. If an LLM is approved in section 3.1, any API spend
  means stop and ask first.
- **Stop and ask if:**
  - a source requires logging in, paying, or getting around access controls;
  - a licence is unclear for a source we want to publish from;
  - the stage 6 audit finds unexpectedly many overlaps with the test files;
  - fewer than 100 candidate problems survive stage 5 across all tier A sources.

## 8. Open questions (decide)

1. Does tier B (literature problems such as Pelletier and TPTP) count toward the headline, or only
   as a separate slice? Default: a separate slice.
2. Which tier C books, if any, may be used for internal evaluation?
3. May an LLM assist transcription under the rules in section 3.1? Default: no.
4. Should "complete the proof" exercises be collected for a `completion` slice (`weaknesses.md`
   section 3)? Default: record where they are, but don't transcribe them yet.
