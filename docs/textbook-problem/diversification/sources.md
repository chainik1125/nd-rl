# Sources: initial discovery (plan task S1)

The first pass of `plan.md` task S1, on 2026-09-28. It was quick and done by hand: web searches,
the GitHub API, and a few single-file downloads to count exercises.

**What was downloaded.** Nothing went into the repo. The chapters that were downloaded to count
exercises are in the session scratchpad only, and the proper download is task S2.

**How to read the licence column.**
- **Verified** means read at the source this session: the repo's licence file as the GitHub API
  reports it, or the book's own page.
- **Secondary** means taken from the Open Logic Project's list of other textbooks
  ([wiki](https://github.com/OpenLogicProject/OpenLogic/wiki/Other-Logic-Textbooks)) and not yet
  checked at the source.

**How to read the counts.**
- They count propositional proof exercises (derive a sequent, prove a theorem, show two sentences
  interderivable) in the chapters that were opened.
- They are rough counts before any exclusions: more than 4 atoms, overlap with the 72 or the RL
  pools, duplicates.
- "Not counted" means the source wasn't opened.

## Tier A: open textbooks

| # | Source | Where (pinned version) | Licence | Prop. proof exercises | Notes |
| --- | --- | --- | --- | --- | --- |
| A1 | *forall x: Calgary*, Thomas-Bolduc & Zach (after Magnus and Button) | [github.com/rzach/forallx-yyc](https://github.com/rzach/forallx-yyc) @ `c5cb41544eb61302fa2d06b8a86c76ed3ba52a7c`, file `forallx-yyc-prooftfl.tex`; [site](https://forallx.openlogicproject.org/); [solutions PDF](https://forallx.openlogicproject.org/forallxyyc-solutions.pdf) | CC BY 4.0 (verified: repo and site) | about 50 | See below |
| A2 | *forall x*, P. D. Magnus (the original) | [github.com/OpenLogicProject/forallx](https://github.com/OpenLogicProject/forallx) @ `b9a872431eb97287fa039103db92a4bf27768b4b`, file `forallx-ch6-proofs.tex`; [fecundity.com/logic](https://www.fecundity.com/logic/) | CC BY 4.0 (verified: repo, and the site's "(Attribution 4.0)") | about 28 | See below |
| A3 | *forall x: Cambridge*, Tim Button (2018) | [forallxcam.zip](https://www.homepages.ucl.ac.uk/~uctytbu/forallxcam.zip) from [Button's OER page](https://www.homepages.ucl.ac.uk/~uctytbu/OERs.html) | CC BY 4.0 (verified: OER page) | 30 | All 30 also appear in A1 (string match after removing whitespace). **Adds nothing new.** |
| A4 | *forall x: UBC*, J. Ichikawa | [github.com/jonathanichikawa/for-all-x](https://github.com/jonathanichikawa/for-all-x) @ `e01f8097b4f6dc5e3b69c94a1dc05d53aca4d0c3`, file `Latex-Files/forallx-ubc-7-SLND.tex` | CC BY 4.0 (verified: repo) | 34 | At least 9 also appear in A1. The crude string match undercounts overlap, so the true number of new exercises is lower than 25. It also has a rules-of-replacement section. |
| A5 | *The Carnap Book*, Leach-Krouse & Ehrlich | [carnap.io/book](https://carnap.io/book), chapters 3–8 | CC BY 4.0 (verified: book page) | 50 (5 + 10 + 5 + 10 + 10 + 10) | See below |
| A6 | *forall x: Adelaide*, A. Eagle (from Cambridge) | [github.com/antonyeagle/forallx-adl](https://github.com/antonyeagle/forallx-adl) @ `22aec473809f866586acfe7bfacc9e59a827e08d` | CC BY 4.0 (verified: repo) | not counted | Probably overlaps with A3. |
| A7 | *forall x: Syracuse*, M. Rieppel (from Cambridge) | [github.com/mrieppel/forallxsyr](https://github.com/mrieppel/forallxsyr) @ `46841216b602e7e9d0fa4069593dfb47e1af0a91` | CC BY-SA 4.0 (verified in pass 2: README) | 66 sequents (pass 2) | See pass 2. |
| A8 | *A Concise Introduction to Logic*, Craig DeLancey (2017) | [Milne Open Textbooks](https://milneopentextbooks.org/concise-introduction-to-logic/) | **Conflicting, 3 claims:** Milne page CC BY-NC-SA; OLP wiki CC BY-SA 4.0; the PDF itself CC BY 4.0 (pass 2) | not counted (PDF only; no PDF-to-text tool installed) | Its own exercise set, not from *forall x*, with proof, conditional-derivation and reductio chapters. Resolve the licence from the PDF's copyright page. |
| A9 | *The Open Logic Text*, Open Logic Project | [github.com/OpenLogicProject/OpenLogic](https://github.com/OpenLogicProject/OpenLogic) @ `1e960beff9ed7835bf3e3f1335e21af3439cd107` | CC BY 4.0 (verified: repo) | not counted | Its natural deduction chapters are mostly metatheory. Expect few concrete exercises. |
| A10 | *Introductory Logic*, Sean Walsh (Carnap) | [carnap.io shared book](https://carnap.io/shared/walsh@g.ucla.edu/book-mainpage.pandoc) | CC BY-NC-SA 4.0 (secondary; the wiki URL returned 404 in pass 2) | not counted | Carnap-based, so its goals are likely machine-readable like A5. |
| A11 | Other *forall x* versions (Bristol, Pitt, MSU, R³, UConn, Leeds, Lorain County, SLU, translations) | Listed on the OLP wiki | CC BY or BY-SA (secondary) | not counted | Mostly copies of A1, A2 or A3's exercise pools. Only worth opening for chapters they wrote themselves. |

**A1, *forall x: Calgary*.**
- **Coverage:** proof exercises from the chapters on basic rules, constructing proofs (strategy),
  additional rules, proof-theoretic concepts, derived rules, and soundness and completeness.
- **The count:** 12 + 8 + 5 + 7 + 4 + 4 + 4 derivations, plus 6 interderivable pairs (12
  sequents), plus "derivation or truth table" items.
- **Also useful for other slices:**
  - "fix the mistake" proofs (2) and "add the citations" proofs (6), for the later probe;
  - it has a "working forward / working backward" strategy chapter.

**A2, Magnus's original *forall x*.**
- The count is chapter 6's propositional items. The rest of its 89 items are first-order or
  syllogisms.
- Most of them reappear in A1.

**A5, *The Carnap Book*.**
- **Machine-readable:** goals sit in `data-carnap-goal` attributes, e.g. `P->Q, Q->R :|-: P->R`.
- **Some goals are hidden:** at least 5 are stored as number arrays, probably to hide them. Read
  those from the rendered page, never by guessing.
- **Many exceed 4 atoms** (T, U, V, W).
- It has its own exercise set, not copied from *forall x*.

## Tier B: literature benchmarks

| # | Source | Where | Licence | Prop. problems | Notes |
| --- | --- | --- | --- | --- | --- |
| B1 | Pelletier, "Seventy-five problems for testing automatic theorem provers", *J. Automated Reasoning* 2 (1986) 191–216; errata in JAR 4 pp. 235–236 and JAR 18(1) p. 135 | [Springer](https://link.springer.com/article/10.1007/BF02432151) | Journal copyright (Springer). Short citation of individual problems is likely fine for internal use: **decide** | 17 propositional (from memory of the paper; **to verify**) | Heavy on classical and case-split problems. Apply the errata. |
| B2 | TPTP problem library, SYN domain, only problems whose header cites a source | [tptp.org](https://www.tptp.org/) | Not stated on the overview page. **To check** on the library's own page | not counted | Includes Pelletier's problems with citations in the headers. |
| B3 | ILTP library v1.1 (Raths, Otten, Kreitz), propositional part | [iltp.de](https://www.iltp.de/) | Not stated on the site | not counted (about 2,800 problems in total, propositional plus first-order) | The domains' origins aren't listed. Only problems with a cited origin qualify. |

## Tier C: free to read, not openly licensed

These can be used only with Charles's approval, and only for internal evaluation (plan section 2).

| # | Source | Where | Licence | Notes |
| --- | --- | --- | --- | --- |
| C1 | Paul Teller, *A Modern Formal Logic Primer* | [tellerprimer.ucdavis.edu](http://tellerprimer.ucdavis.edu/) | Not stated (OLP wiki: "free, not open") | The site returned 403 to an automated fetch; open it by hand. Many derivation exercises, with answers. |
| C2 | Gary Hardegree, *Symbolic Logic: A First Course* | [courses.umass.edu/phil110-gmh](http://courses.umass.edu/phil110-gmh/MAIN/IHome-5.htm) | Not stated | Known for large derivation exercise sets. |
| C3 | Pospesel & Marans, *Arguments: Deductive Logic Exercises* | [arguments.dss.ucdavis.edu](http://arguments.dss.ucdavis.edu/) | Not stated | An exercise book. |
| C4 | Nuel Belnap, *Notes on the Art of Logic* | [pitt.edu/~belnap/nal.pdf](http://www.pitt.edu/~belnap/nal.pdf) | Not stated | |
| C5 | Copyrighted books: Lemmon, Bergmann–Moor–Nelson, Huth–Ryan, Copi, Hurley | – | Copyright | Only from a copy Charles provides. |

**Excluded:** copies on studylib, Scribd and similar sites. Their provenance is unclear, and the
originals above are available.

## What this pass shows

- **The *forall x* family is largely one exercise pool.**
  - A3 is entirely inside A1, most of A2 reappears in A1, and A4 partly overlaps.
  - Counting A1 once, the forall x lineage gives roughly 50–75 distinct propositional exercises,
    not the sum of the columns.
- **Distinct pools found so far:**
  - the forall x lineage (A1 ∪ A2 ∪ A4);
  - Carnap (A5, and probably A10);
  - DeLancey (A8, licence unresolved);
  - the literature (B1–B3).
- **Rough tier A total:** about 120–150 exercises before exclusions. Two exclusions will cut it
  noticeably:
  - more than 4 distinct atoms: common in A1 and A5;
  - overlap with the 72: Dmitry's catalogs likely drew on the same books, since several classic
    laws appear in both. It will be counted in aggregate only at the stage 6 audit.
- **Risk to the plan's targets:**
  - Reaching 200 or more new problems from tier A alone looks unlikely.
  - The stop rule in `plan.md` section 7 (fewer than 100 candidates surviving stage 5) may trigger.
  - **The options:** resolve A8's licence, open A10 and the other Carnap-hosted books, include
    tier B as a separate slice, or approve some tier C books. This is Charles's call.
- **Useful for the `weaknesses.md` axes:**
  - A1 has a strategy chapter that pairs forward and backward working, and a derived-rules chapter
    (lemma and compound pairs).
  - A5 has separate chapters for conditional, nested, indirect and derived-rule derivations, so
    its exercises come with a strategy label from the chapter.
  - B1 is dense in classical and case-split problems.

## S1 gate

The gate is at least 3 tier A sources with verified licences. **Met:** A1, A2, A4, A5 (and A3,
A6, A9).

**Open items for S2:**
- resolve A8's licence and count it;
- verify A7's and A10's licences;
- check TPTP's terms;
- verify B1's propositional count against the paper;
- decode A5's hidden goals from the rendered pages.

## Pass 2 (2026-09-28)

A second quick scan, done the same way as the first. It closed some of pass 1's open items and
added new sources.

**How the fork counts were made.**
- **Method:** each *forall x* fork was shallow-cloned into the scratchpad at the commits below.
  Items were pulled from its practice sections, the connective macros normalized, and exact
  string matches compared. The script is `norm.sh`, in the scratchpad only.
- **What counts:** only sequent-shaped items (premises ⊢ goal). Theorems with no premises aren't
  counted, and items the source marks invalid are dropped.
- **Accuracy:** a different macro or bracket style defeats the match, so overlap is
  **undercounted**. Treat "new" as an upper bound.

### Updates to pass-1 rows

| # | Update |
| --- | --- |
| A1 | 31 sequent-shaped items by the script. The hand count of about 50 in pass 1 also includes theorems and interderivable pairs. |
| A2 | 20 sequent-shaped items, 18 of them also in A1. |
| A3 | 20 sequent-shaped items, all 20 also in A1. |
| A4 | 18 sequent-shaped items, all 18 also in A1. This corrects pass 1's "at least 9", which came from a cruder match. **A4 adds nothing new.** |
| A6 | Adelaide @ `22aec47`: 7 sequent-shaped items, 4 also in A1. |
| A7 | **Licence verified:** the README says "licensed under a CC BY-SA 4.0 license". Syracuse @ `46841216`, file `forallx-04-tfl-prf.tex` and others: 66 sequent-shaped items, 19 also in A1, so **about 47 new**. The largest new *forall x* pool found. |
| A8 | **The licence conflict got worse.** The PDF served by Milne's download link (`https://milneopentextbooks.org/download/a-concise-introduction-to-logic/?tmstv=1672257570`) says "This work is licensed under a Creative Commons Attribution 4.0 International License". That was read through the fetch tool, not by hand. The Milne web page says CC BY-NC-SA, and the OLP wiki says CC BY-SA 4.0. **Resolve by reading the PDF's copyright page by hand**, or by asking the author. Still not counted. |
| A10 | The URL from the OLP wiki now returns 404. The book wasn't found this pass. |
| B1 | **Pelletier located in TPTP v9.3.1.** Headers read `Problem : Pelletier Problem N` and cite `[Pel86]`. SYN040+1 = P1, SYN041+1 = P3, SYN044+1 = P10, SYN045+1 = P13, SYN046+1 = P15, SYN047+1 = P17, all status Theorem. SYN048+1 onward are P18 and later, which are first-order. P2, P4–P9, P11, P12, P14 and P16 are in other files, not found yet. |
| B2 | **TPTP licence still not found.** It isn't on the tptp.org overview or the tptp.tptp.org download page, and not in the SYN040–SYN060 headers. Next place to look: the ReadMe in the TPTP distribution. |

### New sources

| # | Source | Where (pinned version) | Licence | Prop. proof exercises | Notes |
| --- | --- | --- | --- | --- | --- |
| A12 | *forall x: Mississippi State*, G. Johnson | [github.com/loighic/forallx-msu](https://github.com/loighic/forallx-msu) @ `1c4200da893a75bac85d37f97f904a9e2e058a00` | CC BY 4.0 (verified: repo) | 24 sequent-shaped, 1 also in A1 | **About 23 new.** |
| A13 | *forall x: R³*, Davis Smith | [github.com/ProfDavisSmith/forallxR3](https://github.com/ProfDavisSmith/forallxR3) @ `04a5e77dac9875c44816f9a7db38a2ec72cef4de` | CC BY 4.0 (secondary; the API detects no licence file) | 49 sequent-shaped, 1 also in A1 | **About 48 new**, but some are "valid or invalid?" items. Only the valid ones qualify, and each must be checked by hand. Verify the licence. |
| A14 | *forall x: Bristol*, C. Campbell-Moore | [github.com/catrincm/forallx-bris](https://github.com/catrincm/forallx-bris) @ `e9d220e42ed4e9633457dddb71f3f2b732f81394` | CC BY-SA 4.0 (verified: repo) | 0 found by the script | It uses a different exercise markup that the script misses. Open by hand. |
| A15 | *forall x: Pécs* (Hungarian translation) | [github.com/bodri5/forallxPecs](https://github.com/bodri5/forallxPecs) @ `e6b2be067448e3aa14e160585f8cfaf9d06ecdae` | CC BY 4.0 (verified: repo) | not counted | A translation, so expect duplicates only. Low priority. |
| A16 | *forall x: Lorain County Remix*, Loftis, Magnus & Woods | [forallxremix.org](https://forallxremix.org/download/) | **Conflicting:** the download page says CC BY-SA 3.0 Unported; a search snippet for the later *An Open Introduction to Logic* says CC BY-NC-SA 4.0 | not counted | Rewritten with new exercise sets: rules without subproofs first, then conditional proofs, then indirect proofs. Its exercises are sorted by strategy. |
| A17 | *Logic and Proof*, Avigad, Lewis & van Doorn | [github.com/avigad/logic_and_proof](https://github.com/avigad/logic_and_proof) @ `46998b2dad2997975661996daaaac0d179823cc6`; [ch. 3 online](https://avigad.github.io/logic_and_proof/natural_deduction_for_propositional_logic.html) | Repo: Apache-2.0 (verified: API). The page footer says only "©2017, Jeremy Avigad, Robert Y. Lewis, and Floris van Doorn". **Check that Apache covers the text.** | 16 in ch. 3, section 3.6 (listed by the fetch tool; recount by hand) | Uses Gentzen-style natural deduction. One exercise is a word puzzle, not a formula. The Lean chapters may add more propositional exercises. |
| X1 | Carnap shared documents | [carnap.io/shared](https://carnap.io/shared): 4,583 linked documents, 1,127 with proof-like names, from 126 owners | **Per document; none of the 4 sampled states a licence** | Sampled: Zach, "Practice Problems IV", 26 (25 readable); Zach, "More Practice Problems IV", 18 (17 readable); Swartzer, "Practice Derivations for Final Exam Review", 5; Magnus, "6b proofs in SL", 6 | See below. |
| X2 | Stanford *Introduction to Logic* (Genesereth), Fitch exercises | [intrologic.stanford.edu](http://intrologic.stanford.edu/); [exercise index](https://ggp.stanford.edu/intrologic/public/exercises.php) | Not found: the site's TLS certificate didn't match its hostname, so the fetch failed | not counted | Interactive Fitch exercises. Open by hand. |
| X3 | Levin, *Discrete Mathematics: An Open Introduction*, 4th ed. | [discrete.openmathbooks.org](https://discrete.openmathbooks.org/) | CC BY-NC-SA 4.0 (search snippet only) | not counted | Its logic chapter uses informal proofs and rules of inference, not boxed natural deduction. **Likely few usable exercises.** |

**Why X1 (Carnap shared documents) isn't tier A yet.**
- **Why it's valuable:** problems are machine-readable (`data-carnap-goal`), and there are many.
- **Uncertain status:** instructors post their own problem sets there, but some probably copy
  exercises from copyrighted books.
- **The rule for using it:**
  - A document by a known open textbook's author (Zach for *forall x: Calgary*, Magnus) can be
    treated as that book's supplement, but only once Charles agrees.
  - Any other document is tier C until its origin is known.
- **Zach's two "Practice Problems IV" sets:** 44 problems from the *forall x: Calgary* author
  that don't appear in the book's exercise lists (not yet checked against A1).

### What pass 2 changes

- **Distinct *forall x* sequents:**
  - A1–A4, A6, A7, A12 and A13 together have **144 distinct sequent strings** (exact string match,
    an upper bound).
  - Theorems with no premises and the interderivable pairs come on top of that.
  - Of those, A7, A12 and A13 contribute most of the material that isn't in A1.
- **New totals by pool, before exclusions:**
  - *forall x* family: about 144 sequents plus about 30 theorems;
  - Carnap Book A5: 50;
  - Zach's Carnap sets: 44, if approved;
  - Avigad A17: about 15;
  - Lorain County Remix A16, DeLancey A8 and Bristol A14: not counted yet;
  - Pelletier: up to 17.

  That makes **roughly 250–300 candidates before exclusions.** The exclusions (more than 4
  atoms, overlap with the 72 and the RL pools, duplicates) will probably remove a third or more.
  The 200 target now looks reachable if A16, A8 and Zach's Carnap sets come in.
- **Open items, in order:**
  1. resolve the A8 and A16 licences;
  2. decide whether a Carnap document by a textbook's own author counts as tier A;
  3. check A13's licence;
  4. open A14, A16 and X2 by hand;
  5. find TPTP's licence and the remaining Pelletier files;
  6. recount A17 by hand.

### Corrections (2026-09-28, missing-laws scan)

- **A8 (DeLancey).** The "PDF says CC BY 4.0" claim in pass 2 is **wrong**.
  - **What was actually read:** Milne's download link returns an HTML download page, not the
    PDF. That page's footer reads "Except where otherwise noted, content on this site is licensed
    under a Creative Commons Attribution 4.0 License". It describes the website, not the book.
  - **Where that leaves it:** the licence is back to two claims: the Milne page says CC BY-NC-SA,
    the OLP wiki CC BY-SA 4.0.
  - **Access:** the PDF itself sits behind a download form with a one-time token. It needs a
    manual download; getting around the form automatically isn't allowed (`plan.md` section 7).
- **X1 (Carnap shared documents).** `https://carnap.io/robots.txt` has `Disallow: /shared/` and
  `Crawl-delay: 10`.
  - **Breach:** during discovery (pass 2) the `/shared` index and 4 shared documents were fetched.
    That breaks `robots.txt` and `plan.md` stage 2. No more `/shared/` pages will be fetched, and
    X1 is **excluded** as a source.
  - **Book pages:** the Carnap Book pages (`/book/`, allowed) were fetched 3 s apart, not 10 s.
    Future fetches from carnap.io use at least 10 s.
- **Lorain County Remix (A16).** Source found: `github.com/rob-helpy-chalk/openintroduction` @
  `b44f79557a357679fc8e632599cccd1e77c3f327`. The GitHub API detects no licence file. The book's
  download page says CC BY-SA 3.0.

## Status 2026-09-30: what has been transcribed, what has not

**Transcribed and in a candidate set:** A1 Calgary, A7 Syracuse, A12 Mississippi State, A5 Carnap
Book (candidate_v0); P1 Roy, P2 Carr's pack (candidate_v1).

**Counted, found to add nothing or almost nothing:** A2 Magnus (18 of 20 in A1), A3 Cambridge
(all in A1), A4 UBC (all in A1), A6 Adelaide (4 of 7 in A1), A15 Pécs (translation).

**Located but never transcribed** (best first, by expected yield):

| # | Source | Expected new | Blocker | Effort |
| --- | --- | --- | --- | --- |
| A13 | *forall x: R³* (Davis Smith) | about 48 sequents, some "valid or invalid?" | licence secondary only (no LICENSE file); valid-or-not items need hand checking | low: LaTeX repo, same tooling |
| A16 | *forall x: Lorain County Remix* (Loftis) | unknown; its own exercise sets sorted by strategy | licence conflict (BY-SA 3.0 vs BY-NC-SA 4.0) | low–medium: LaTeX repo `rob-helpy-chalk/openintroduction` @ b44f7955 |
| A14 | *forall x: Bristol* (Campbell-Moore) | unknown; 10 of the 72 appear in it | different exercise markup; open by hand | low |
| A8 | DeLancey, *A Concise Introduction to Logic* | unknown; own exercise set with proof, conditional and reductio chapters | licence unresolved (BY-NC-SA vs BY-SA); PDF behind a nonce form, manual download | medium: PDF, now readable with pdfminer |
| P2 | Halbach's *Exercises Booklet* and Fritz's *More Exercises* (Oxford) | unknown; Carr's pack is the same course, so overlap likely | no licence stated | low: PDFs already saved |
| A17 | Avigad et al., *Logic and Proof*, ch. 3 | about 15 | Apache-2.0 on the repo; check it covers the text | low: reST source |
| A11 | Dortmund, Natal, Auckland, Pitt, UConn, Leeds, SLU *forall x* versions | small; Dortmund and Natal each contain 16–18 of the 72, so heavy sealed overlap | licences per repo | low each |
| P3 | Peter Smith, IFL2 exercises | unknown | no CC licence; exercises page not located | low once found |
| P9 | Slaney, ANU *Logic Notes* | 15+ in exercises 1, more in ch. 2–3 | no licence; ask | low: HTML |
| P8 | Clemente, *Introduction to natural deduction* | 22 worked examples (proofs published) | GNU FDL (snippet) | low: HTML |
| P10 | Uzquiano, *Introduction to Logic* ch. 5 | 5 | no licence | trivial |
| P7 | Pfenning 15-317 HW1 | few; intuitionistic | none stated | low: tarball |
| P14–P17 | MIT OCW 24.241, Edinburgh Inf1, Chalmers DAT060, Cambridge CST | small; several point at copyrighted books | mixed | low |
| P11–P13 | Incredible Proof Machine, NaDeA, Jape task banks | unknown | open code licences but task provenance unstated | low |
| B1 | Pelletier P1 (and P2, P4–P9, P11, P12, P14, P16 in TPTP, not yet found) | up to 11 | tier B; TPTP terms unknown | low |
| X1 | Carnap shared documents (Zach, Magnus, Swartzer, Walsh) | 44+ from Zach alone | robots.txt disallows; must come from the authors | email |
| X2 | Genesereth, Stanford *Introduction to Logic* Fitch exercises | unknown | TLS mismatch; copyright unknown | manual |
| P4, P5, P20 | Zach's, Klement's, Leach-Krouse's own problem sets | unknown | behind LMS or robots; ask | email |
| C1–C5 | Teller, Hardegree, Pospesel & Marans, Belnap, copyrighted textbooks | large | not open; Charles's approval | – |

**Not worth opening:** A9 Open Logic Text (metatheory; 0 law hits), X3 Levin (informal proofs),
P21 MSR problem-generation paper (synthetic; only its cited benchmark could count).

### 2026-09-30: R³, Lorain Remix and Bristol opened (cloned into `raw/`, gitignored)

| | R³ (Davis Smith) @ 04a5e77 | Lorain County Remix / *An Open Introduction to Logic* (Loftis) @ b44f795 | Bristol (Campbell-Moore & Stern) @ e9d220e |
| --- | --- | --- | --- |
| Licence, as read in the repo | CC BY (preface: "The CC BY license gives you the right to download and distribute") | CC BY-NC-SA 4.0 (frontmatter of this version; older editions BY-NC-SA 3.0 / BY-SA 3.0) | CC BY-SA 4.0 (`LICENSE.md`, preface) |
| Lineage | Calgary | Magnus, rewritten and expanded | Calgary, "substantial alterations" |
| Rule system | Fitch; adds MT and HS as basic conditional rules; ∨E is disjunctive syllogism; a redefined Dilemma rule (A→B, C→D, A∨C ⊢ B∨D); "returns to Magnus's original" ∨E, ¬I, ¬E; equivalence rules incl. TAUT (A∨A ⇔ A); no ⊥ | Magnus's 11 rules (5 I, 5 E, R); ∨E is disjunctive syllogism; ¬I/¬E by reductio with "for reductio" and "Want:" annotations; derived rules (∨* proof by cases, MT, HS, DN, DeM, MC); rules of replacement; no ⊥ | Reiteration, MP and ∧ first, then →I, ¬I/¬E as contradiction-pair rules (Y ∧ ¬Y), ∨, ↔; ⊥ sections removed; derived rules LEM, DS, MT |
| Proof exercises seen | 44 "give a proof" sequents in the two ND chapters (12 + 12 + 20); pass 2's 49 includes valid-or-invalid items; many use 5+ letters | 85 turnstile items in "Derive the following" parts (`\sststile`), all with premises; answers inline (`\answer`) | 28 turnstile items plus "add citations" and "derivation or truth table" parts; answers inline (`\myanswer`) |
| Other exercise kinds | truth-table checks of the rules; fill in the blank | substitution instances; "is this an instance of the rule"; fill in the blanks; translate then prove | add citations; tautology/contingency by derivation or table |
| Parser work | none (standard `\item ... \therefore`) | `\sststile{}{}` turnstile and set braces | `problemslist`, `\entails` |

### 2026-09-30: Pelletier, the Oxford booklet and MIT OCW tried

- **MIT OCW 24.241 (Logic I, Fall 2009): not usable.** The derivation problem sets (2, 4, 9
  read; PDFs in `raw/mit-ocw/`) only cite *The Logic Book* by section and item number
  ("3.5E: 1d, 2c, …"); set 1 is translation and true/false. The problems are Bergmann's (C5).
- **Pelletier P1–P17: all 17 obtained.** The paper is paywalled, so the statements were read from
  Isabelle's `src/FOL/ex/Classical.thy` (BSD-3; cites the paper and its errata; one lemma per
  problem, numbered) and cross-checked against TPTP v9.3.1 `SYN040+1`, `041`, `044`, `045`,
  `046`, `047` (P1, P3, P10, P13, P15, P17), fetched through TPTP's viewer 10 s apart (raw
  problem URLs 404; no robots.txt). P10 is taken in TPTP's premises form, which is the paper's.
  Audit: 9 new (P3, P4, P5, P7, P9, P10, P12, P14, P17); P1, P8, P13, P15, P16 are already in
  candidate_v1, P2 and P6 are validation_36 forms, P11 is P ↔ P (identity, already in v0 as
  P ⊢ P's twin). Reader: `code/textbook_problem/other_sources.py`.
- **Oxford: 10 propositional ND exercises in total**, not the 60–80 guessed. Halbach's booklet
  has 7 (exercise 6.1) plus one six-premise argument with subscripted letters (excluded);
  Fritz's *More Exercises* has 3 (exercise 6.1), harder ones. After the audit: 3 new (booklet
  iv, Fritz ii and iii); the rest duplicate Carr, Calgary or `train`. No licence stated.
- **Working folder:** `data/textbook_problem/candidates_v2/` (gitignored); proof search on the
  12 started 04:26 UTC.
- **forall x: R³ transcribed** (same day): 44 "give a proof" items in three blocks (nd1 653–666,
  nd2 175–188, 337–358); truth-table and fill-in blocks skipped. Audit: 9 new; 4 renamings of
  `train`/`heldout` items; the other 31 duplicate earlier sources, 20 of them Calgary items that
  are themselves sealed overlaps (R³ is Calgary-derived; pass 2's "1 of 49 in Calgary" was a
  string match, not a renaming-class match). One item had a stray comma before ∴ (nd1 item 10);
  `clean_item` now drops it. Working folder `candidates_v3/`; proof searches started 04:30 UTC.

### 2026-09-30: Lean proof datasets and Lean course repos scanned

Prompted by feedback: "there are lots of datasets on the internet with Lean proofs; find a
natural deduction subset". Working folder `data/textbook_problem/candidates_lean/` (gitignored;
scanner `lean_scan.py`, commits in `sources_commits.txt`).

- **Large Lean datasets hold almost no propositional problems.** miniF2F, ProofNet, PutnamBench,
  Lean Workbook and FormalMATH are competition or undergraduate mathematics. LeanDojo, Herald and
  LEAN-GitHub are mined from Mathlib and GitHub repos; their propositional part is library lemmas
  (Mathlib `Logic/Basic`, core `Init/PropLemmas`) plus whatever course repos they swept in, which
  are scanned directly below. Not scanned: LEAN-GitHub's 28,597 theorems (it may contain more
  course repos than the six read here).
- **Propositional Lean datasets are synthetic, so excluded by the authenticity rule:** PropL
  (Qin et al. 2024; uniformly sampled intuitionistic formulas over 2 atoms, 200k;
  [HF](https://huggingface.co/datasets/KomeijiForce/PropL)), `ergotts/propositional-logic`
  (1,600 generated natural-language arguments, no Lean), LeanReasoner's ProofWriter/FOLIO
  formalisations (127, first-order, rule bases). PropL could serve the deferred synthetic grid.
- **Lean course and textbook repos: real exercises, scanned.** A regex scanner reads `example`,
  `theorem`, `lemma` and lean4game `Statement` headers, keeps those whose binders are `Prop`
  atoms and hypotheses over ∧ ∨ ¬ → ↔ False (Lean precedence; ↔ expanded per §3.1; True
  excluded), renames to P–S, and audits by renaming class against the sealed sets (counts only),
  the five pools and every `candidate*/` file of ours.

| Source | Licence | Prop. statements | Sealed | Pool | Already ours | Dup. | **New** | Proved | 5–10 lines | 11+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *Theorem Proving in Lean 4* (ch. 3, Tactics) | Apache-2.0 | 112 | 30 | 35 | 20 | 12 | **15** | 14 | 8 | 4 |
| *Logic and Proof* (Lean chapters) | Apache-2.0 (repo) | 65 | 7 | 28 | 4 | 20 | **6** | 6 | 5 | 1 |
| *Hitchhiker's Guide to Logical Verification* (LoVe 2024) | none in repo | 55 | 9 | 23 | 3 | 10 | **10** | 10 | 8 | 2 |
| Buzzard, *Formalising Mathematics* 2024, Section01logic | Apache-2.0 | 108 | 23 | 26 | 10 | 26 | **23** | 19 | 11 | 6 |
| Buzzard, M40001 Lean sheets (Lean 3) | none in repo | 214 | 43 | 59 | 22 | 81 | **9** | 9 | 6 | 2 |
| *A Lean Intro to Logic* (lean4game, Trequetrum) | MIT | 134 | 43 | 28 | 5 | 38 | **20** | 18 | 14 | 2 |
| Lean core `Init/PropLemmas` (library lemmas) | Apache-2.0 | 6 | 0 | 1 | 0 | 1 | **4** | 4 | 4 | 0 |
| **Total** | | 694 | 155 | 200 | 64 | 188 | **87** | **80** | **56** | **17** |

Proofs: `prove.py --secs 120 --long_secs 300` (shorter than the 300/1800 used for v0/v1), 80/87
found, all pass strict_verify; 7 without a proof are nested biconditionals and one 5-way
implication chain. Families of the 80: conditional 32, direct 14, negation 12, case-split 11,
case-split-derived 6, classical 5; 10 match a schema in an RL pool. 7 of the 87 are
classical-only.

**Caveats before any of this is used:**
- The scanner ignores section-level `variable (h : A)` hypotheses. That produced 36 invalid
  statements (excluded), and could have produced valid but wrong transcriptions; every kept row
  needs a hand check against its file and line, as the second-reader check does.
- Not every hit is an exercise: some are worked examples or demo lemmas (TPiL `t1`, LoVe `*_Demo`
  files, M40001 lecture files, core library lemmas). Per §1 a stated law or example counts, but
  library lemmas are not textbook problems.
- Lean 3 syntax (M40001, *Logic and Proof*) is only partly parsed; about 1,100 headers were
  rejected as "unexpected" or "non-propositional" without being read.
- A large share of what was found is a renaming of the sealed sets (155 of 694 hits): standard
  laws recur in every course.

### 2026-10-01: batch 4, the sources the Lean scan had missed

Charles asked for all four leads in order. Working folder `data/textbook_problem/candidates_b4/`
(gitignored: readers, merge/audit script, items, exclusions, `sources_commits.txt`). Audit rule:
curried normal form + renaming class, against the sealed sets, the five pools, the v2 bundle and
the batch itself; sealed matches counted only.

| Source | Licence | Read | Propositional | Sealed | Pool | Bundle twin | Batch twin | **New** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Isabelle `src/FOL/ex/` (Propositional_Int, Propositional_Cla, Intuitionistic, Intro, Classical) @ 70839b74 | BSD-3 | 208 lemmas (98 first-order skipped) | 82 | 17 | 6 | 17 | 9 | **33** |
| Metamath `set.mm` propositional section @ ef3d7e99, only `$p` whose comment cites a book (`[WhiteheadRussell]` 203, Margaris, Hamilton, Church, Mendelson, …) | CC0 | 1,016 `$p` (818 uncited skipped) | 197 | 32 | 22 | 20 | 5 | **118** |
| LEAN-GitHub (`internlm/Lean-Github`, 218,866 tactic steps → 21,618 theorems; first tactic state with only `Prop` binders) | Apache-2.0 (dataset); repos vary | 1,365 states with Prop | 102 (19 repos) | 27 | 26 | 17 | 14 | **18** (11 from textbook/course repos, 7 held: lean-smt, compfiles, IMOSL, a personal repo) |
| LAMR, *Logic and Mechanized Reasoning* (Avigad, Heule, Nawrocki), ch. 8 ex. 1, 4 and ch. 9 Lean exercises | © 2021, none stated | 15 (3 unreadable: lowercase `r`) | 12 | 4 | 3 | 2 | 0 | **3** |
| Second Lean repo scan: MIL ch. 3, *Mechanics of Proof* ch. 5, Formalising Mathematics 2022/2023 (Lean 3), LoVe 2025, Massot's tutorials, Robo | Apache / MIT / none | 346 statements | 346 | 67 | 128 | 157 | 5 | **12** (math2001 4, Robo 7, tutorials 1; MIL, FM 2022/23, LoVe 2025: 0) |
| **Total** | | | | | | | | **184**, 177 sent to proof search |

Notes:
- LEAN-GitHub's useful new repo is Velleman's *How To Prove It with Lean* (`djvelleman/HTPILeanPackage`,
  Apache-2.0): 6 new. Brown CS22 / CS1951x and Lewis's `leanclass` add a few course items.
- Metamath citations give PM numbers (`Theorem *4.61 of [WhiteheadRussell] p. 120`), so the 118
  are the first large set of named laws with a textbook citation. Hilbert-style source, but only
  the statements are used. 3-ary ∧/∨ grouped to the left (noted per row).
- Formalising Mathematics 2022 and 2023 are the same sheets as 2024; MIL's logic chapter is over
  real-number predicates; LoVe 2025 repeats 2024.
- The Lean 3 parser gap from the first scan was mostly real first-order or arithmetic statements;
  the fix (`∀ P Q : Prop,` goals) recovered 2.
- Proof search (prove.py 300 s / 1,800 s, 6 procs) started 2026-10-01 on the 177; results to
  `candidates_b4/proved.jsonl`.
