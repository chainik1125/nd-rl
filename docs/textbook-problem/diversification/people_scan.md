# S1 pass 3: academics and their problem sets (2026-09-29)

The first two passes (`sources.md`) went book by book. This pass goes person by person: logicians
who teach natural deduction and publish exercise sets, course problem sets, proof checkers with
problem banks, or solution packs. The aim is new propositional sequents that are not copies of
the *forall x* exercise pools, which candidate_v0 has already drained.

**Method.** About 40 web searches and 30 page fetches. Every licence claim below says how it was
seen: *verified* means read on the source's own page through the fetch tool; *snippet* means it
came only from a search-result summary and has not been read on the page. PDFs the fetch tool
could not decode were saved to the session scratchpad and are marked *unread*; `pypdf` and
`pdftotext` are not installed, so they need a PDF tool or a manual read. No carnap.io `/shared/`
page was requested (robots.txt disallows it).

**Tiers** as in `plan.md`: A open licence; B literature; C free to read, no licence; D needs the
author's permission or a manual download. A new column, *distinct from forall x*, is the reason
this pass exists.

## Summary

| # | Person | Resource | Licence | Distinct from *forall x*? | Tier | Next step |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Tony Roy (CSU San Bernardino) | *Symbolic Logic: An Accessible Introduction to Serious Mathematical Logic*, 2023 | CC BY-NC-SA 4.0 (verified) | Yes: own exercise sets, very large | A (NC) | Download PDF; needs a PDF tool. **Best new source found.** |
| P2 | Alastair Carr, Volker Halbach, Peter Fritz (Oxford) | *The Natural Deduction Pack*; Logic Manual *Exercises Booklet*; Fritz's *More Exercises* | None stated (verified: site and GitHub repo) | Yes: Oxford's own problems, with solutions | C/D | Ask Halbach and Carr. |
| P3 | Peter Smith (Cambridge, ret.) | *An Introduction to Formal Logic* 2nd ed., free PDF, plus 40+ pages of ND exercises with answers | Free download; no CC licence found (verified on `logicmatters.net/ifl`) | Yes | C/D | Ask Smith; exercise pages not located (two URLs 404). |
| P4 | Richard Zach (Calgary) | Carnap practice problems IV, "More Practice Problems IV", and course PHIL 279 | Per document, none seen (pass 2) | Partly: pass 2 sampled 26 + 18 items | D | Ask Zach for the documents and a licence. `/shared/` is robots-disallowed. |
| P5 | Kevin Klement (UMass) | Logic Penguin (proof checker with exercise banks); Phil 110 interactive lecture notes with credit exercises | Code GPL-3 (verified); exercise content not stated; notes behind Canvas | Yes: Hardegree-style (Kalish–Montague) derivations | D | Ask Klement whether the exercise bank is separable and under what licence. |
| P6 | Gary Hardegree (UMass) | *Symbolic Logic: A First Course*, earlier edition online, ch. 5 derivations | None stated | Yes: large drill sets | C | Already C2 in `sources.md`. Ask before use. |
| P7 | Frank Pfenning (CMU) | 15-317 Constructive Logic, Homework 1 "Natural Deduction and Constructivity" (tarball) | None stated (verified) | Yes; intuitionistic, so every proof is also classical | C | Unpack `hw1-handout.tar.gz`; expect few problems. |
| P8 | Daniel Clemente Laboreo | *Introduction to natural deduction* (EN/ES/CA) | GNU FDL (snippet; the page cites a licence section 1.4, not read) | Yes | A/B | 22 worked propositional examples (14 explained + 8 extra), no separate exercises. Verify the FDL text. |
| P9 | John Slaney (ANU) | *Logic Notes*, ch. 2–3 propositional ND, with a proof editor; "Exercises 1" has 15 sequents with proofs | None stated (verified on 3 pages) | Yes | C/D | Ask Slaney. Author's name inferred from the URL (`~jks`), not from the page. |
| P10 | Gabriel Uzquiano (USC) | *Introduction to Logic*, online, ch. 5 natural deduction | None stated (verified) | Yes | C | 5 exercises seen. Small. |
| P11 | Joachim Breitner | The Incredible Proof Machine: task bank (`sessions.yaml`, `examples/`) | MIT (verified: repo) | Yes | A | Read the task files; count the propositional tasks. |
| P12 | Jørgen Villadsen et al. (DTU) | NaDeA: exercise list in the paper's section 7 | MIT (snippet) | Yes; mostly first-order | A | Low yield; check the propositional subset. |
| P13 | Richard Bornat, Bernard Sufrin | Jape: natural deduction encodings and example files | GPL-2 (verified: repo) | Yes | A | Find the ND example/conjecture files in `distrib/` or `dev/`. The book's exercises are OUP copyright. |
| P14 | MIT OCW 24.241 (Logic I, 2009; TA Damien Rochford) | 10 problem sets with solutions | CC BY-NC-SA (OCW's standard terms; snippet, not read on the page) | **Probably not usable**: the sets reference *The Logic Book* (Bergmann) section numbers, so the problems may be the book's | C | PDF `hw02` saved, unread. Check whether problems are printed in full. |
| P15 | Edinburgh Informatics 1 (Computation & Logic) | Tutorial 7/8 on propositional ND with solutions | None stated | Yes | C | PDF saved, unread. |
| P16 | Thierry Coquand (Chalmers, DAT060) | Weekly exercise sheets `w1.html`–`w7.txt`, `exercises.txt`; Huth–Ryan solutions PDF | None stated (verified) | Mixed: many are Huth & Ryan's | C | Only the sheets' own problems qualify. |
| P17 | Lawrence Paulson / Mateja Jamnik (Cambridge CST Part IB) | *Logic and Proof* notes with exercises | None stated | Yes; few propositional ND items | C | Low yield. |
| P18 | Michael Genesereth (Stanford) | *Introduction to Logic*, Fitch exercises online | Unknown | Yes | D | Both hostnames fail TLS (certificate is for `logical.stanford.edu`). Open by hand. |
| P19 | Krysia Broda (Imperial) | Pandora e-tutor exercises by difficulty | Unknown | Yes | D | Exercises are inside the Java tool. |
| P20 | Graham Leach-Krouse (Kansas State) | Carnap's own problem sets beyond the Book | Not stated | Yes | D | Ask; his Book is already A5. |
| P21 | Umesh Ahmed, Sumit Gulwani, Amey Karkare (MSR, 2013) | "Automatically generating problems and solutions for natural deduction": a benchmark of textbook ND problems | Paper only | Their *benchmark* is textbook problems; their *generated* problems are synthetic and excluded by the plan | B | PDF saved, unread. Only the cited textbook problems could qualify. |

Not usable and why: *Language, Proof and Logic* (CSLI copyright, exercise files inside the
software); Allen & Hand *Logic Primer* (MIT Press, out of print, not open); Huth & Ryan, Lemmon,
Bergmann, Restall's *Logic: An Introduction*, Halbach's *Logic Manual* itself (all publisher
copyright); academialap.com's 36 solved exercises (commercial site, no licence); Scribd uploads
(provenance unknown).

## Notes per person

**P1 Tony Roy.** The licence line on `tonyroyphilosophy.net/symbolic-logic/` reads "licensed
under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License". CSUSB
ScholarWorks (`scholarworks.lib.csusb.edu/books/1/`) offers the full text (5.4 MB) and *Answers
to Selected Exercises* (7.7 MB), 2023, with the author's note "I am happy for anyone to use this
material". Volume 1 Part I covers sentential derivations (its ND system is Fitch-like, with
derived rules). NC and SA: fine for an internal evaluation set; a public release of the
problems would have to be BY-NC-SA. Exercise counts unknown until the PDF is read.

**P2 Oxford.** `logicmanual.philosophy.ox.ac.uk` lists: `exercises/exercises.pdf` (booklet,
2018/19), `exercises/exercises_peter.pdf` (Fritz, with solutions), `carr/NDpack.pdf` (Carr,
March 2018, problems by connective with a solutions section), sample and collection papers with
solutions. The site says solutions for teachers are withheld from the public. No licence on the
site; the GitHub mirror `Alastair-Carr/Natural-Deduction-Pack` (7 commits, one PDF) has no
LICENSE file. The NDpack and Fritz PDFs are saved, unread.

**P3 Peter Smith.** `logicmatters.net/ifl/` says the corrected 2nd edition is "a freely
downloadable PDF" with "end-of-chapter exercises with detailed worked solutions" on an exercises
page; the March 2020 post "More natural deduction exercises" advertises 40+ pages of worked
answers. No CC statement seen. The exercises page URL was not found (`/ifl/exercises/` and
`/ifl/ifl-exercises/` both 404); find it from the `/ifl/` page by hand.

**P4 Richard Zach.** `richardzach.org/teaching/279-3/` describes problem sets done "on D2L or on
carnap.io". Pass 2 sampled his Carnap documents before the robots rule was checked; they are not
to be fetched again. His 2020 post "Adding online exercises with automated grading to any logic
course with Carnap" is a pointer to how the sets are made, not to a licence.

**P5 Kevin Klement.** `github.com/frabjous/logicpenguin`: GPL-3, JavaScript, Kalish–Montague and
Fitch derivations, *forall x* systems partly implemented. The repo page shows no exercise data
files; the working examples live in his Phil 110 lecture notes, which the syllabus
(`people.umass.edu/~klement/110/`) puts on Canvas. The syllabus forbids AI use *for students
completing the exercises*, which is not about reuse of the problems, but it is a reason to ask
first.

**P7 Pfenning.** `cs.cmu.edu/~fp/courses/15317-f17/assignments.html` lists 11 homeworks as
tarballs; HW1 is the ND one. Intuitionistic propositional logic: any proof is a proof in our
classical system, and some goals may need classical rules only in a different form.

**P8 Clemente.** `danielclemente.com/logica/dn.en.html`: sections "The derivation rules",
"Explained exercises" (5.1–5.14), "Examples, lots of examples" (9.1–9.26, of which 9.23–9.26 use
quantifiers). All are worked, so their proofs are published; that matters if we ever use
reference-proof leakage as a criterion.

**P9 Slaney.** `users.cecs.anu.edu.au/~jks/LogicNotes/exercises1.html` gives 15 sequents with
proofs, for example `(p ∧ q) ∧ r ⊢ s ∨ (q ∨ t)`, `(p → q) → p ⊢ (p → q) → q`,
`(p ∧ q) → ¬r ⊢ (p ∧ r) → ¬q`. Sixteen chapters; ch. 2 "Propositional natural deduction" and ch.
3 "More about propositional logic" have the exercise links. Author and licence not on the pages.

**P11 Breitner.** `github.com/nomeata/incredible`, MIT. The README mentions `sessions.yaml` and an
`examples` folder; the task list is what the tool shows on start. The tasks' own provenance
(written by Breitner or taken from a book) is not stated and must be checked before counting them
as "textbook".

**P14 MIT OCW.** Problem set 2 is described (snippet) as "exercises from sections 3.5E and 3.6E
of *The Logic Book*". If the PDFs only cite section numbers, the problems are Bergmann–Moor–
Nelson's and fall under C5. If they are printed, OCW's CC BY-NC-SA would cover the PDF but not
necessarily the underlying textbook problems. Check `hw02` first.

**P18 Genesereth.** `intrologic.stanford.edu/exercises/exercise_04_04.html` exists (search
index) but the host's certificate is issued to `logical.stanford.edu`, so the fetch tool refuses
it. The exercises are interactive Fitch problems with automated grading; the textbook is Morgan
& Claypool 2012, so assume copyright unless the site says otherwise.

## Dead ends this pass

- `intrologic.stanford.edu` and `ggp.stanford.edu`: TLS certificate mismatch (second time).
- `logicmatters.net/ifl/exercises/`, `/ifl/ifl-exercises/`: 404.
- `logicinaction.org/exercises/`: 404. The site says extra exercises by Dora Achourioti exist;
  URL not found. No licence on the front page.
- Peter Smith's "Logic Café"/Halpin: nothing found under that name.
- Berkeley Phil 12A (Holliday), Stanford Phil 150/151 (Icard), McGill Phil 210 (Hallett), Utah
  PHIL 3200, UT Dallas (Matthew Brown): syllabi only; problem sets are behind the LMS or not
  posted. Brown's course uses Magnus's *forall x*, so nothing new there anyway.
- German-language search for Übungsblätter returned only the same English resources.

## What this pass changes

1. **One large open source: Tony Roy (P1).** It is the only new tier-A book with its own
   exercise sets. NC-SA needs a decision (see below).
2. **The rest is tier C/D by people.** Oxford (P2), Smith (P3), Slaney (P9), Klement (P5) and
   Zach (P4) each hold sets that are distinct from *forall x*, unlicensed, and easy to ask for.
   One email each is cheaper than any further scraping.
3. **Tool task banks (P11–P13)** are open-licensed but of uncertain origin; they count as
   "textbook" only where a task cites one.
4. **Course problem sets** (P7, P14–P17) are small and often just pointers to copyrighted books.

## Decisions for Charles

- **NC-SA licences (Roy; also A8 DeLancey if it resolves to NC):** allowed for internal evaluation
  data? The plan's tier A says "open licence"; it doesn't say whether NC counts. Default until
  decided: allowed for internal use, and flagged in provenance so a public release can drop them.
- **Contact authors?** Proposed list, in order of expected yield: Halbach/Carr, Smith, Zach,
  Klement, Slaney, Leach-Krouse. Each ask: permission to transcribe their propositional ND
  exercises into an internal evaluation set, and whether they would state a licence.
- **Tool task banks:** include the Incredible Proof Machine tasks only if their file cites a
  source per task? Default: no, unless a source is cited.
- **PDF tooling:** add `pypdf` to `code/pyproject.toml` (via `uv add`) so the saved PDFs (Roy,
  Carr, Fritz, MIT hw02, Edinburgh, MSR paper) can be read by script. Default: yes, on request.

## Sources

Fetched (verified): tonyroyphilosophy.net/symbolic-logic/; scholarworks.lib.csusb.edu/books/1/;
logicmanual.philosophy.ox.ac.uk/; github.com/Alastair-Carr/Natural-Deduction-Pack;
logicmatters.net/ifl/; cs.cmu.edu/~fp/courses/15317-f17/assignments.html;
people.umass.edu/klement/ and /~klement/110/; github.com/frabjous/logicpenguin;
logicinaction.org/; danielclemente.com/logica/dn.en.html; classes.matthewjbrown.net/logic/2024-spring/;
cse.chalmers.se/~coquand/LOGIC/logic.html; arxiv.org/abs/2607.04321; github.com/nomeata/incredible;
users.cecs.anu.edu.au/~jks/LogicNotes/ (index, Chapter2, exercises1); github.com/RBornat/jape;
gabriel-uzquiano.github.io/logic/natdedprop.html.

Saved, unread (scratchpad `tool-results/`): Carr NDpack; Fritz exercises_peter; MIT 24.241 hw02;
Edinburgh inf1/cl solutions8; MSR problem-generation paper.

Snippet only: MERLOT and ScholarWorks licence lines for Roy (then verified on the author's site);
OCW licence; NaDeA licence; Bilaniuk GNU FDL (Hilbert-style, not ND, so not listed); Clemente
FDL.

## Scrape of P1 (Roy) and P2 (Carr's pack), 2026-09-29

**Code:** `code/textbook_problem/pdf_sources.py` (readers), wired into `transcribe.py` as sources
`roy` and `carr`. `pypdf` and `pdfminer.six` were added to `code/pyproject.toml`.
**Working output:** `data/textbook_problem/candidates_v1/` (gitignored): `candidates.jsonl` (all
six sources, 195), `new_only.jsonl` (the 90 not in candidate_v0), `exclusions.jsonl`,
`summary.json`, then `new_only_proved.jsonl` from the proof search.

**Sources as fetched.** Roy: `SymLogFullTxt.pdf` from the author's site (robots.txt allows
everything; ScholarWorks put up a Cloudflare challenge, which was not bypassed), 823 pages,
"From first print edition, 2023 (Third revision, July 2025)", licence line on page 2:
CC BY-NC-SA 4.0. Also `SymLogSelAns.pdf` (answers, not read). Carr: `NDpack.pdf`, March 2018,
123 pages, the copy the fetch tool saved. sha256 of all three are in `summary.json`.

**Reading the PDFs.** Roy's math font has no Unicode map for ∼ and `]`; pypdf drops them
silently, which would have turned ¬A ⊢ ¬(A ∧ B) into A ⊢ (A ∧ B). pdfminer keeps them as
`(cid:24)` and `(cid:141)`, which the reader decodes; a layout setting was needed to stop it
interleaving neighbouring items' goals. All 140 Roy items and 104 Carr items were printed and
checked by eye against the page images' text (pypdf's dump, with the tildes known missing).

**What was read.** Roy section 6.2 (sentential ND, systems NDs and NDs+): E6.7, E6.10, E6.13,
E6.15, E6.18, E6.20, E6.21 and theorem lists T6.1–T6.30 (E6.16, E6.19). Skipped: NP game
system (E6.1, which includes an invalid sequent), fill-in-the-justification and find-the-error
exercises, strategy questions, translations, essays, chapter 3 (axiomatic), quantificational
sections. Carr section 4.1–4.6 practice problems, the section 3.1 worked example, the bonus
challenges (as plain sequents) and Contraposition 1–2. Skipped: first-order and identity items,
admissible-rules questions, the alternate system ⊢C.

**Funnel.**

| | Roy | Carr |
| --- | --- | --- |
| Items read | 140 | 104 |
| Out of scope (first-order, ⊢C) | 0 | 12 |
| More than 4 letters | 12 | 3 |
| Transcribed | 128 | 89 |
| Duplicates of an earlier source or within the set | 29 | 43 |
| Sealed overlap (72 or test files; counted only) | 9 | 40 |
| Pool overlap (validation_36, train) | 4 | 2 |
| **New candidates** | **86** | **4** |

Carr's pack is the canonical Oxford drill list: 40 of its 89 sequents are renamings of sealed
problems and 43 more duplicate *forall x* items. Roy's exercises are his own; 86 of 128 survive.

**The 90 new candidates:** 55 sequents, 35 theorems; letters 1/2/3/4: 6/34/31/19; premises
0/1/2/3/4: 35/16/29/9/1; 22 classical-only; 50 use ↔ (rewritten as two conditionals). 38 of the
Roy items are starred, meaning the author's answer is published in *Answers to Selected
Exercises*; this is recorded per problem in `encoding_notes`.

**Licence note.** Roy is CC BY-NC-SA; Carr states none. Both are flagged in `source.licence`.
Whether NC-SA is acceptable is still Charles's decision (section "Decisions").

**Proof search (2026-09-29/30).** 80 of the 90 have a verified reference proof; 10 (nine of
them biconditional-of-biconditional theorems) found none within bound 40 / 300 s + 1,800 s.
Of the 79 proved before a 5-problem rerun: median 9 lines, mean 11.3, max 28; bins 3–4: 3,
5–6: 8, 7–8: 19, 9–10: 18, 11–12: 6, 13–15: 9, 16+: 16. Families: conditional 27, case-split
18, negation 13, classical 10, direct 7, derived case-split 4. Tags: biconditional 26, nested
16, classical 15, refutation 14. Named laws: 20 problems match a schema, 9 of them in the RL
pools; **six of the nine laws missing from the 72 are present** (T6.14 contraposition and
converse, T6.15 export, T6.26 dist_and_over_or_conv, E6.15t explosion, E6.15v modus tollens),
in biconditional or theorem form, with no overlap with validation_36. Details and the merged
rows: `data/textbook_problem/candidates_v1/new_only_proved_merged.jsonl`.
