# Run brief: create-or-elicit-paper — write the paper and a blog post from the reviewed results

Run id `create-or-elicit-paper`. Repository `~/work/create-or-elicit-paper` (branch `dan_create-or-elicit-paper` from the
fork's `origin/dan`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan 2026-10-11 (proposal
27).

- **Start:** queued after `create-or-elicit`'s librarian has recorded it.
- **Budget:** **$10 / 20 pod-hours**, for checks the writing turns out to need (pre-register each).
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 96 h; no hurry. Dan wants wide agency here: propose analyses, extensions or more literature that
  would make the paper better.

## Inputs (reviewed results only)

- **`create-or-elicit`:**
  - its librarian summary (nd-rl `experiment-summaries/<date>-create-or-elicit-…`) and its review, whose corrections
    take precedence;
  - its `claims_ledger.md`, figures and `next_steps.md` (fork branch `dan_create-or-elicit`).
- **Earlier reviewed runs:** `capability-defs` (`docs/project_strategy/capabilities/`), `rl-continue`,
  `rl-continue-cap6`, `trajectory`, `rl-from-ckpt`, `support-curves`, `support-state`, `guided-tts`, `mcts-a`,
  `grpo-best`, `compute-match`, `claim-audit`, `evidence-atlas`.
- **Literature:** `docs/literature/` (lit-review, lit-review-2, capabilities, and any review `create-or-elicit` added).
- **Proposal 27** (`docs/proposals/2026-10-11-create-or-elicit-paper.md`), including Dan's outline of the paper.

Use a number only if the claims ledger traces it to a reviewed source. If the paper needs a number that does not exist,
compute it from existing files with a script in the repo. Run a pod only if nothing on disk can answer.

## Deliverable 1: the paper draft

A full research-paper draft in the usual ML structure:
- title;
- abstract (≤ 250 words);
- introduction, with an explicit list of contributions;
- related work (positioned against Abdulsalam, Patel & Saxe 2607.07646, Yue et al., power sampling, Tsilivis et al.,
  Interplay and the elicitation literature);
- the organism and setup;
- the measurement framework (budgets, k-to-solve intervals, the known-proof estimator and its limits, certification,
  RL-free controls);
- results, centred on the causal knockout;
- a short section on test-time compute (guided redraws vs MCTS, before and after RL);
- discussion;
- limitations (9.6 M parameters, propositional logic, near-saturated targets, seed counts);
- reproducibility (code, data, a compute table);
- appendices.

**Format and figures:**
- Write in LaTeX and compile to PDF if a TeX engine fits on the VPS (e.g. `tectonic`, which needs ≤ 1 GB of disk);
  otherwise in Markdown structured for pandoc.
- Regenerate figures from data with scripts kept in the repo.

**Authors and acknowledgements:** use the placeholder "Dan Pandori et al. (author list for Dan to decide)". Add no
other person's name, and leave acknowledgements for Dan.

## Deliverable 2: the blog post

`paper/blog.md`: 1,500–2,500 words for technical readers outside this subfield, with 3–5 figures.
- **Plain language.** Every term is defined where it first appears, and the "so what" comes first.
- **Same evidence as the paper,** including the negative and null results, and no stronger claims than the paper makes.
- Robbie's posts (robbiewmthompson.com) are a reference for tone and length.

## Process

1. **Outline first** (`paper/OUTLINE.md`): the claims, the figure or table behind each, and its ledger entry.
2. **Draft.**
3. **Critique.** At least two "reviewer 2" critique rounds by critic subagents (novelty, soundness, clarity), each
   followed by a revision. Record each critique and response in `paper/REVISIONS.md`.
4. **Audit.** An independent claim-audit subagent checks every number and claim in both documents against the ledger.
   Fix what it finds.
5. **Propose further work.** Analyses, experiments or literature that would strengthen the paper go in
   `paper/next_steps.md`, costed and ranked. CPU-only analyses can be done directly.

## Deliverables

- `paper/` (source, PDF if built, figures and scripts), `paper/blog.md`, `paper/OUTLINE.md`, `paper/REVISIONS.md`,
  `paper/next_steps.md`.
- `run_create_or_elicit_paper.md` (≤ 200 words), `log.md`, `STATUS.md` line `CREATE-OR-ELICIT-PAPER DONE <UTC>`, and
  `touch ~/runs/create-or-elicit-paper/executor.done`.
- **The librarian** copies the paper, blog and figures to nd-rl `docs/papers/create-or-elicit/` and writes the summary
  as usual.

**For the reviewer:**
- Check every number and claim in the paper and the blog against its reviewed source.
- Check that the blog claims nothing the paper does not.
- Check that the limitations section is honest.

**Rules:**
- Nothing is posted, submitted (arXiv or a venue), shared or sent. Both documents are drafts for Dan.
- Do not contact anyone.
- textbook72 is scoring only.
- At most 4 concurrent processes on the VPS.
