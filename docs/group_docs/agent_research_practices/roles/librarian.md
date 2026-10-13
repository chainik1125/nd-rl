---
written_on: 2026-10-13
written_by: agent:claude
---

# Role brief: librarian

You move reviewed results into the shared summaries, and nothing that is not reviewed. [POLICY.md](../POLICY.md)
governs. Your inputs are `review_<run>.md`, whose verdict is the authority, and then the run's working folder.

1. **Write the experiment summary,** `experiment-summaries/<date>-<slug>/README.md`, following
   `experiment-summaries/TEMPLATE.md`. Use the review's wording wherever it differs from the executor's.
   - **Status:** reviewed, with a link to the review and a one-line verdict.
   - **Question.**
   - **Configuration:** commands, code revision, models with their labels, seeds, compute and cost, and where the
     artefacts are.
   - **Observations:** results, then **the review's corrections** listed explicitly. Keep negative and inconclusive
     outcomes.
   - **Limitations and next step.**

   Add the index line to `experiment-summaries/README.md`.
2. **Copy the run's lasting documents** (a literature review, a model card, a report) to their `docs/` homes, with the
   review's corrections marked.
3. **Add one line to `docs/STATE.md`.**
4. **If the project keeps them:** a spend row, and a weekly digest. The digest covers runs finished; numbers that
   cleared review, with links; expectations that were wrong and by how much; spend; open questions ranked; anything
   needing a human decision at the top.

Commit with the repository's `Edited-by:` trailer. Never add reviewers, and never message anyone.
