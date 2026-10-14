---
written_on: 2026-10-14
written_by: agent:claude
---

# *Did Expert Iteration Create It?* — paper and blog drafts, clean versions

Drafts for Dan. Nothing here has been posted, submitted or shared. The author line is the brief's placeholder ("Dan
Pandori et al. (author list for Dan to decide)"), and the acknowledgements are left for Dan.

> **Two versions (librarian, 2026-10-14).**
> - **This directory holds the clean versions.** The paper, blog, figures, Lean files and `CLEANUP.md` are run
>   `paper-clean`'s files, byte for byte (fork `dan-pandori/nd-takehome`, branch `dan_paper-clean`,
>   [`86b705ee`](https://github.com/dan-pandori/nd-takehome/commit/86b705ee)). `next_steps.md` carries the review's
>   corrections, marked, and this README is the librarian's.
>   - The reading copy's *[review]* marks are folded into the prose: all 60 in the paper, and 30 of 31 in the blog. The
>     31st (R23) was cut with the seeding paragraph it qualified.
>   - Of the 13 possible errors the librarian listed outside that review, 12 were real and are fixed; the note on item
>     12 was wrong.
>   - No theorem in a textbook72 renaming class is printed.
>   - The blog is 2,487 words.
>
>   The paper-clean review found the clean paper equivalent to the marked copy:
>   - no number changed except the listed fixes, each of which reproduces from the data;
>   - no claim is stronger;
>   - every reviewed caveat is kept.
>
>   The blog keeps the four caveats the brief names, but its length cut removed two results the paper still carries
>   (decision 1 below).
> - **[`review_marked/`](review_marked/README.md) holds the reviewed reading copy, unchanged.** It is nd-rl
>   `7709669`'s copy, moved here by a pure rename in `dda847b`. It shows each correction of the `create-or-elicit-paper`
>   review as a red *[review]* mark, and its README lists them (P1–P17, B1–B8). Keep it as the record of what that
>   review changed. **Do not circulate it:** its §2 prints a theorem in a textbook72 renaming class
>   (`textbook_b98931f2…`), which the clean version removes.
>
> Reviews:
> - [`review_paper-clean.md`](https://github.com/dan-pandori/nd-takehome/blob/792db421/review_paper-clean.md), for the
>   clean versions;
> - [`review_create-or-elicit-paper.md`](https://github.com/dan-pandori/nd-takehome/blob/1ad2a691/review_create-or-elicit-paper.md),
>   for the marked copy.
>
> Summary: [experiment-summaries/2026-10-14-paper-clean-…](../../../experiment-summaries/2026-10-14-paper-clean-clean-paper-and-blog-equivalent-to-reviewed-copy-12-of-13-possible-errors-fixed/README.md).

| file | what |
|---|---|
| [`main.pdf`](main.pdf) | the paper, 35 pages, built by `build.sh` (tectonic 0.17.0; `~/tools/tectonic/tectonic` on the VPS, or `tectonic` on `PATH`) |
| `main.tex`, `sections/*.tex`, `refs.bib` | LaTeX sources. The PDF's section order differs from the file numbers: §1 `01_intro`, §2 `03_setup`, §3 `05_results`, §4 `04_measurement`, §5 `02_related`, §6 `07_discussion`, §7 `08_limitations`, §8 `09_repro`, App. A–J `10_appendix` (App. I is `11_checks`) |
| [`blog.md`](blog.md) | the blog post, 4 figures. 2,487 words: the count excludes the YAML front matter and includes the Lean block, alt texts and captions |
| `figures/` | the paper's figures F0–F7 and the four blog figures (`.pdf` for LaTeX, `.png` to view). Only `fig0_overview` (the 2.9× / 4.2× label) and `blog_fig2_logp` (its title) differ from the marked copy's |
| `lean/` | the blog's simplified Lean σ proof and its goal-negated control, unchanged |
| [`CLEANUP.md`](CLEANUP.md) | the run's record of every change from the marked copy, with evidence: the folded marks, each possible error's outcome, the textbook72 check, the blog cut, the build and the equivalence check. Unchanged from the run |
| [`next_steps.md`](next_steps.md) | the writing run's costed proposals, with the correction of item 1b's cost to C4's measured rate, and paper-clean's presentation proposals. The review's corrections are marked *[review]* |

To compare the two versions, diff the two directories, e.g.
`git diff --no-index docs/papers/create-or-elicit/review_marked/blog.md docs/papers/create-or-elicit/blog.md`.

## Decisions for Dan

None of these blocks reading the drafts.

1. **Restore two blog results?** To reach 2,500 words the run cut two results that the paper still carries. It
   disclosed both cuts (`CLEANUP.md` §4), and the review would restore them:
   - **F1 (the review: should fix): the one-round seeding null.**
     - The paper reports it as a null (seeded minus control +0 / +0 / +2 / +0 / +0) and as a missed pre-registered
       expectation (≥ 3 seeds; 1 of 5 observed).
     - The earlier review's B2 held the blog to the paper's negative and null results.
     - The reviewer's example: "One extra RL round seeded with a rare σ proof changed nothing measurable (an earlier,
       smaller model under an older checker had responded)."
   - **F2 (recommended): the RL-phase likelihood clause.** It read: "one full run and one dose run moved strongly
     toward these proofs; the rest … ended between 8 units lower and 4 units higher". It was the blog's only reading
     of Fig. 2's trajectories. `next_steps.md` presentation proposal 4 also ranks it first.
   - **Room:** the blog has 13 words of headroom. F1 alone (≈ 23 words) needs ≈ 10 words cut elsewhere; F1 and F2
     together (≈ 38) need ≈ 25.
2. **Optional small text fixes** (the review's minor items). They are not applied here, so the clean versions stay
   exactly as reviewed:
   - App. I names `round2_extra.py` for A5b. That script prints 0.0188 by Monte Carlo, while §3 quotes the exact
     0.020. The review suggests "0.019 by Monte Carlo; 0.020 exact".
   - "early-stopped" is never defined in the paper. Its first use is now in §1.
   - §5's "a quarter to a third" is approximate, beside the corrected 2.9× / 4.2×.
   - §6's new sentence says knockout s4 "never proved a strict training theorem". Adding "in 16 rounds" would be
     cleaner.
   - The blog's Fig. 4 caption no longer defines "Hardest"; the text just before the figure does.
   - The blog's "that only a genuinely classical step can prove" now attaches to the 105 only.
   - The two labels of `next_steps.md` proposal 6 fit within the cap.
3. The author list, the acknowledgements, and whether and to whom to circulate.

## Notes (librarian)

- **textbook72.** No theorem printed in the clean paper, blog, figures or `lean/` is in a textbook72 renaming class.
  The brief's method compares each printed sequent with the sealed classes by `gen.canon_key`, as the `--sealed` guard
  does. The run and the reviewer added a premise-order-invariant key, which finds no hit either.
  - The review counts 0 of 19 distinct printed sequents. The librarian re-ran the reviewer's scanner
    (`review_pc/kit/sealed.py`) on this directory and got the same result; on `review_marked/` it finds the removed
    schema.
  - §2 now states the negated-conditional law in words, which the brief allows. A reader can rebuild the class from the
    words (the review's N2).
  - One broader reading hits in both versions: Table 1's lemma-route formula, with its antecedent moved to the premise
    side, is in `textbook_dd80d9db…`'s class. As printed it is not a hit, and the paper cannot avoid this law: it is
    the paper's subject (N1).
- **Not copied; on the fork branch at `86b705ee`:**
  - `artifacts/paper_clean/`: the run's check scripts and outputs, also in the bucket under
    `paper-clean/artifacts/paper_clean/`, and its equivalence reports. Those reports are in git only. The review advises
    masking `equivalence_paper.md`'s print of the removed schema (F3).
  - The writing run's working files: `OUTLINE.md`, `LEDGER.md` (P8 is now marked superseded by C4), `REVISIONS.md`,
    `critiques/`, `notes/`, `scripts/` and `data/`. The figures are regenerated with `python3 paper/scripts/figures.py`
    from the fork's repository root.
  - The run's own `paper/README.md`, which this README replaces. It still says "31 blog marks folded"; the review's
    count is 30 folded and 1 cut.
- **The marked copy's README was written for this directory's old location.** Two things in it now point elsewhere:
  - its summary link is one directory short; use the summary link above;
  - its `git diff d841408 -- docs/papers/create-or-elicit/` now compares with the clean versions. For the changes the
    librarian made in the marked copy, run `git diff d841408 7709669 -- docs/papers/create-or-elicit/`.

  Its list "Not applied: possible errors outside the review's list" is resolved in `CLEANUP.md` §2: 12 items fixed.
  Item 12's note was wrong: the positive control's comparators stored their accepted proofs, as did knockout s1's, and
  Lean accepts all 24 stored proofs and rejects all 24 with the goal negated.
