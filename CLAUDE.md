You are working for Charles in the NDRL repo. The project studies capability emergence in RL using natural deduction proofs.

This file holds only standing conventions and safety rules. The current goal, what is fixed and what varies, tasks, gates, metrics and budget live in the active workstream's `plan.md`. Where the plan and this file disagree, this file wins on safety and the plan wins on scope.

## Start of every session
1. Work out which workstream the task belongs to: the one the user names, or the one whose folder you are working in. If it is unclear, ask.
2. Read `docs/<workstream>/plan.md`, then the latest entries in `docs/<workstream>/log.md` to see where the last session stopped.
3. Check `git status` and the current branch before changing anything.

## Workstream layout
- `docs/<workstream>/`: `plan.md`, `log.md`, `questions.md`, `notes.md`, write-ups. All notes and decisions go here, as Markdown.
- `experiments/<workstream>/<exp-name>/`: one folder per experiment (see Experiments).
- `code/<workstream>/`: the workstream's code.

## Python
- Run Python only as `uv run python` from `code/`, or as `code/.venv/bin/python`. Never bare `python3`.
- On a fresh machine or pod, run `uv sync --locked --extra dev` in `code/` first. Add dependencies through `uv add`, never `pip install`.
- For the pinned model, `PYTHONPATH` is `code/ws3/fork_pin:code/experiments/current/r1_cot/vendor`.

## Git
- Work only on `charles/main` and `charles/exp-<short-name>` branches.
- Never push to, merge into or rebase the repo's `main`. Never force-push. Never touch other people's branches, worktrees or pods.
- Parallel agents each use their own `git worktree`.
- Commit message format: `[<workstream>][<exp-name or main>] what changed`.
- Commit configs, seeds and the exact launch command for every run, before the run.
- Never commit checkpoints, large data, `.env` or any key. Keys live in a gitignored `.env`; never print them. Checkpoints and large artefacts go to the Hugging Face bucket, with their hashes in git.
- Commit or push only when asked, or when the plan's process says to. When asked "is it pushed", check every branch and worktree, not just the current one.

## Data integrity
- **The 72 textbook problems** (Dmitry's 58 `textbook_dev` + 14 `textbook_train`) are a sealed test set.
  - Never train or tune on them or on a renaming of them. Never use them per problem as seeds, templates, few-shot examples, selection targets, or a source for new problems.
  - Allowed: scoring, overlap audits, and descriptive analysis reported in aggregate (counts by category). The analysis itself stays local and untracked (`docs/textbook-problem/textbook72/` is gitignored); aggregate numbers may be quoted in plans and write-ups.
  - They are gitignored. Restore them locally with `bash code/ws3/textbook72/fetch.sh`, which checks the hashes in `MANIFEST.json`. Never commit them, reference proofs of them, or per-problem analysis of them.
- **Test files** (`test_*_prompts.jsonl`, `validation_36*`): score once, report only in aggregate, never inspect verdicts for individual theorems.
- **Leak checks** compare by renaming class (`gen.canon_key`, with `canonical_theorem` as a second opinion), never by string match. Every training set is audited against every evaluation set before it is used.
- **Pinned components** are never edited: `code/ws3/fork_pin/` (run its `fetch.sh`; check it with `fetch.sh --verify`), vendored code, and whatever the plan marks as fixed. Do not retrain a pinned checkpoint.
- **Scoring:** every proof is checked by the pinned `nd_verify` through `strict_verify.py`, never by a custom checker.

## Experiments
- Each experiment gets `experiments/<workstream>/<exp-name>/`.
- Write and commit `proposal.md` before any run: hypothesis, config, a numeric expected result, what we conclude either way, budget and stop rule. A run without one is labelled "not pre-registered".
- During a run, keep `status.md` current. It ends `DONE <UTC>` or `QUARANTINE <UTC> <reason>`.
- After a run, write `results.md`: graphs, raw samples and the conclusion. Read at least 20 raw samples or outputs before reporting on any run.
- Every run writes the metrics the plan requires to its run folder (for training runs, one JSON line per step or round in `metrics.jsonl`).
- Numbers that go into a conclusion need an independent reviewer recount first, as the plan describes.

## Reporting numbers
- Every number cites its source file, its n, and its number of seeds.
- One seed supports "signs of life", not a comparison or a claim. Differences need 3 seeds.
- Give intervals over problems, not over samples. Lead with the average solve rate; pass@k at large k is secondary.
- "Emergence" or "new capability" needs counts of problems the base model never solved and the trained model solves at least half the time, not only a higher average.
- Name every selection effect: pre-filtered pools, picking a config on the eval set, dropped problems.
- Report failures as plainly as successes. Say what was skipped. Never state a result that was not produced and checked.
- Do not state who authored data or code unless it is verified.

## Compute and long runs
- Anything longer than a few minutes runs detached (`nohup … &` then `disown`; no `setsid` on macOS), logs to its run folder, and is checked by polling, not by blocking a tool call.
- Kill only the PIDs you started. Never kill by pattern (`pkill`, `killall`, `pgrep | xargs kill`); other agents share this machine.
- Stop every pod you started as soon as its artefacts are pulled. Never stop or touch pods you did not start.
- Record the cost of every paid run in the workstream's `spend.md`, with a running total.

## Stop and ask when
- A gate in the plan fails twice.
- Anything would spend money beyond the plan's budget, or spend money when the plan names none.
- You want to change a pinned component, or anything the plan marks as fixed or out of scope.
- A result looks surprisingly good. Check for a scoring bug and a train/test leak first.
- An error looks like running out of quota. Confirm it is not a normal process exit first.
- An action would delete or overwrite results, data or someone else's work.

Other questions go in `docs/<workstream>/questions.md`, together with the default you will follow. Then keep working.

## End of every session
Append to `docs/<workstream>/log.md`: the date, what ran, what was learned, what is next, and open questions.
