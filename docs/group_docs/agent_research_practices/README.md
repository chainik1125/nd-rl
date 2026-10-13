---
written_on: 2026-10-13
written_by: agent:claude
---

# Agent research practices

How Dan's agent-run campaign (September–October 2026, about 100 runs) ran experiments with Claude agents. It covers
the standards each run followed and the role briefs the agents worked from, written so other workstreams can reuse
them. Infrastructure is left out: hosts, GPU providers, budgets enforced by scripts and the session driver are
specific to one setup.

## The loop

```
open question → proposer → run brief → executor → reviewer → librarian → experiment summary
```

| role | does | hands over |
| --- | --- | --- |
| [proposer](roles/proposer.md) | turns an open question into a run brief with pre-registered expectations | a proposal note and a run brief |
| [executor](roles/executor.md) | runs the experiment: pre-registers, computes, writes up | raw files, `numbers.md`, a short write-up, a done line |
| [reviewer](roles/reviewer.md) | re-derives every promised quantity **before** reading the write-up, then compares | `review_<run>.md` with a verdict |
| [librarian](roles/librarian.md) | moves only reviewed results into the shared repository | the experiment summary, state line and index entries |

Each role is a separate agent session. The reviewer never shares a session with the executor: independence is the
point.

## Files here

- [POLICY.md](POLICY.md): the standing standards every run follows, each with the reason it exists.
- [roles/](roles/): one brief per role.
- [templates/](templates/):
  - [run brief](templates/run_brief.md);
  - [pre-registration](templates/preregistration.md);
  - [model card](templates/model_card.md);
  - [claims ledger](templates/claims_ledger.md).

## How this fits the repository's conventions

These practices sit on top of [`AGENTS.md`](../../../AGENTS.md), not in place of it:
- Notes carry `written_on` frontmatter and commits carry an `Edited-by:` trailer.
- Working files go in `code/experiments/current/<experiment>/`.
- Human-readable results go in `experiment-summaries/<date>-<slug>/README.md`.

What the practices add:
- an explicit pre-registration before any compute is spent;
- an independent reviewer whose corrections take precedence over the executor's wording;
- a librarian step, so only reviewed numbers reach the shared summaries.

## Where the practices came from

Most standards here were added after a specific bug or near-miss produced a wrong number. [POLICY.md](POLICY.md)
gives each reason. Dan's principle for adding more: **don't scar on the first cut.** One incident gets a direct fix and
a note. A standard or tool is adopted only once a second incident shows it would have helped.
