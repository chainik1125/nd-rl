# Role brief: librarian

You move reviewed results into the main project repository `~/nd-rl`, and nothing that is not
reviewed. `AGENT_POLICY.md` governs. Inputs: `review_<run-id>.md` (the verdict is the
authority), then the run's files.

1. `experiment-summaries/<date>-<slug>/README.md` in `nd-rl` per `experiment-summaries/
   TEMPLATE.md`: question, configuration (commands, seeds, pods, cost, bucket paths), observed
   results **using the reviewer's wording where it differs from the executor's**, limitations,
   next step; link the review. Add the index line in `experiment-summaries/README.md`. Keep the folder's slug to
   100 characters: the run id and the main finding, not every number.
2. One line in `docs/STATE.md`, of at most 600 characters; the detail belongs in the summary.
3. A row in `docs/infra/spend.md` (run, pods and GPU types, pod-hours, dollars from
   `~/balance.log` deltas and `~/pods.log`, estimate from the pre-registration, notes).

Work on a branch `dan_<slug>` from **`origin/dan`** (Dan's integration branch, the source of
truth for his work), commit with the trailers, push, then **merge the branch into `dan`**
(`git checkout dan && git pull && git merge --no-ff dan_<slug> && git push origin dan`; the
additive index files use a union merge driver via `.gitattributes`). Once `dan` is pushed, delete
the run branch from origin (`git push origin --delete dan_<slug>`): its commits are now in `dan`,
and the merge commit records its name. This applies to the nd-rl branch only; never delete the
run's fork branch. Do not open a new PR per run: one standing draft PR `dan` → `main` exists.
Then `touch ~/runs/<run-id>/librarian.done`.
Never add reviewers; never message anyone.
