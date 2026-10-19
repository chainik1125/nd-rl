# Run brief: paper-clean — clean versions of the "Did RL create it?" paper and blog

Run id `paper-clean`. Repository `~/work/paper-clean` (branch `dan_paper-clean` from the fork's
`origin/dan_create-or-elicit-paper`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan
2026-10-14, after he read the drafts ("super interesting").

- **Budget:** **$20 / 40 pod-hours**, only if re-checking a number needs it. Pre-register any pod job in `log.md`.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## Starting point

**The reviewed reading copy** in nd-rl `origin/dan:docs/papers/create-or-elicit/`. Its parts:
- `main.tex`, `sections/`, `refs.bib`, `figures/`, `lean/`, `build.sh`;
- `blog.md`;
- `README.md`, the librarian's notes: the P1–P17 and B1–B8 tables, "Notes for Dan", and the list "Not applied: possible
  errors outside the review's list".

Each reviewer correction in it is marked *[review]* (`\rv{}` in the LaTeX; 60 in the paper, 31 in the blog). The
authorities are:
- the review: `review_create-or-elicit-paper.md` on fork branch `dan_create-or-elicit-paper`, and its `review_coep/`;
- the reviewed results of `create-or-elicit`.

Copy the reading copy into this run's `paper/` as the starting point; the fork branch's own `paper/` is pre-review.

## Tasks

1. **Fold every *[review]* mark into the prose.**
   - Remove the markers and smooth the sentences around them.
   - Keep each correction's meaning exactly: no claim gets stronger, and no caveat is dropped.
   - Remove `\rv` from the LaTeX preamble once it is unused.
2. **Resolve each "possible error outside the review's list"** in the reading copy's README.
   - **Check each against the data** (`create-or-elicit` artefacts, `numbers.md`, the claims ledger); don't take the
     note's word for it.
   - **Fix the text** where it is wrong.
   - **Record each item's outcome** (fixed / already correct / changed differently) in `paper/CLEANUP.md`, with the
     evidence.
3. **textbook72.** §2 prints the negated-conditional family's schema, which shares a renaming class with a sealed
   textbook72 problem. A circulated draft must not print any theorem in a textbook72 class.
   - Replace it with a verbal description, or an instance outside every textbook72 class.
   - Check every theorem printed in the paper and the blog against the sealed classes (the `--sealed` guard in
     `code/tests/fixtures/proof_state/make_golden.py`, nd-rl branch `dan_shared_proof_state`, PR #26, shows the method), and record the check.
   - IDs alone are fine.
4. **The blog: ≤ 2,500 words** (it is 3,080, including the Lean block).
   - Cut wordiness and repetition first.
   - Keep every reviewed caveat a reader needs: the "needs exposure, within 16 rounds of this EI" scope, the
     sampler-specific sharpening null, the rare-not-absent finding, and the power limits.
   - Keep the blog claiming nothing the paper does not.
   - Fix the blog items in the librarian's list: "hand-picked"; the phrases not in the paper; the power-sampling
     paragraph naming its models; Fig. 2's alt text and title.
5. **Rebuild the PDF** (`build.sh`, tectonic in `~/tools/tectonic`): no undefined references, no new overfull boxes.
6. **Equivalence check.** Before finishing, an independent subagent compares the clean versions with the marked
   reading copy, number by number and claim by claim. It confirms that:
   - every number is unchanged, except fixes from task 2, which are listed;
   - no claim is stronger;
   - every reviewed caveat survives.

   Record its report in `paper/CLEANUP.md`.

You may also propose further improvements in `paper/next_steps.md`, such as clarity, structure, or a missing figure.
Apply only presentation changes that keep the paper's claims unchanged; anything else goes in the proposals.

## Deliverables

- `paper/`: clean `main.tex`, `sections/`, `main.pdf`, `blog.md`, `figures/`, `lean/`; plus `CLEANUP.md` and
  `next_steps.md`.
- `run_paper_clean.md` (≤ 200 words), `log.md`, `STATUS.md` line `PAPER-CLEAN DONE <UTC>`, and
  `touch ~/runs/paper-clean/executor.done`.
- **The librarian:**
  - moves the current marked reading copy in nd-rl to `docs/papers/create-or-elicit/review_marked/`, unchanged;
  - puts the clean versions at `docs/papers/create-or-elicit/`;
  - adds a short README note explaining the two;
  - writes the summary as usual.

**For the reviewer:**
- Verify the equivalence: numbers and claims, clean against marked.
- Check each `CLEANUP.md` outcome against the data.
- Confirm no theorem in a textbook72 class is printed.
- Check the blog's word count.

**Rules:**
- Nothing is posted, submitted, shared or sent. These are drafts for Dan.
- Do not contact anyone.
- textbook72 is scoring only.
- At most 4 concurrent processes on the VPS.
