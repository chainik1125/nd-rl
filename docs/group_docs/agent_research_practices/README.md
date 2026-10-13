---
written_on: 2026-10-13
written_by: agent:claude
---

# Agent research practices

How Dan's agent-run campaign (September–October 2026, about 60 reviewed runs) ran experiments with Claude agents. It
covers the standards each run followed and the role briefs the agents worked from, written so other workstreams can
reuse them.
- **Left out:** infrastructure. Hosts, GPU providers, budget-enforcement scripts and the session driver were specific
  to one setup.
- **Applies to:** experiments that, in `AGENTS.md`'s terms, have a hypothesis to test and take more than about five
  hours of human-equivalent effort. Quick tests and literature notes don't need the full loop.

## The loop

```
open question → proposer → run brief → executor → reviewer → librarian → experiment summary
```

| role | does | hands over |
| --- | --- | --- |
| [proposer](roles/proposer.md) | turns an open question into a run brief with expected results | a proposal note and a run brief |
| [executor](roles/executor.md) | runs the experiment: pre-registers, computes, writes up | raw files, `numbers.md`, a short write-up, a done line |
| [reviewer](roles/reviewer.md) | re-derives every promised quantity **before** reading the write-up, then compares | `review_<run>.md` with a verdict |
| [librarian](roles/librarian.md) | moves only reviewed results into the shared summaries | the experiment summary, state line and index entries |

## Running it by hand

1. **Start each role as a fresh agent session,** given three things: [POLICY.md](POLICY.md), the role brief, and the
   run brief. The reviewer must never share a session with the executor, since independence is the point.
2. **When the executor writes its done line,** whoever launched the run makes the reviewer's phase-1 copy. That is
   the working folder with the write-ups removed: `run_*.md`, `numbers.md`, `log.md`, `STATUS.md`, `next_steps.md`
   and any summaries. **The launcher makes it, not the reviewer,** so the reviewer never sees the write-ups first.
   The reviewer then starts on that copy, and reads the full folder only in phase 2.
3. **After the review is committed,** start the librarian.

## Where a run's files live

Everything for one run lives in its working folder, `code/experiments/current/<run>/`:

| file | written by | what |
| --- | --- | --- |
| `BRIEF.md` | proposer | the run brief ([template](templates/run_brief.md)) |
| `preregistration.md` | executor, before any compute | design and expected results ([template](templates/preregistration.md)) |
| `STATUS.md` | executor | one dated line per milestone, ending in `<RUN> DONE <UTC>` |
| `log.md`, `numbers.md`, `QUESTIONS.md` | executor | dated log; every number with its source and model; questions with defaults |
| `run_<run>.md`, `next_steps.md` | executor | the short write-up; costed proposals for what to do next |
| `review_<run>.md` | reviewer | § Recount, § Compare, § Verdict |

The librarian writes outside it: `experiment-summaries/<date>-<slug>/README.md` (main's `experiment-summaries/TEMPLATE.md`),
the index line, and `docs/STATE.md`.

## Files here

- [POLICY.md](POLICY.md): the standing standards, each with the reason it exists and a link to the evidence.
- [roles/](roles/): one brief per role.
- [templates/](templates/):
  - [run brief](templates/run_brief.md);
  - [pre-registration](templates/preregistration.md);
  - [model card](templates/model_card.md);
  - [claims ledger](templates/claims_ledger.md).

`docs/group_docs/` is gitignored by design, so a new file here needs `git add -f`.

## How this fits the repository's conventions

These practices sit on top of [`AGENTS.md`](../../../AGENTS.md):
- notes carry `written_on` / `written_by` frontmatter;
- commits carry an `Edited-by:` trailer (agents may add `Agent-role:` and `Run-id:` trailers too);
- results go to `experiment-summaries/`.

What they add:
- a pre-registration before any compute;
- an independent reviewer whose corrections take precedence over the executor's wording;
- a librarian step, so only reviewed numbers reach the shared summaries.

## Where the practices came from

Some standards record decisions, each with its reason: Lean alone as judge, guided read-outs, recording compute.
Others were added after a bug or near-miss produced a wrong number; [POLICY.md](POLICY.md) says which.

Dan's principle for adding more: **don't scar on the first cut.** One incident gets a direct fix and an entry in an
ideas list naming its trigger. A standard or tool is adopted only once a second incident shows it would have helped.

## Glossary

- **Base reachability:** whether the model before RL (the base) solves a theorem within a stated number of attempts.
  It's needed to tell "RL solved it" from "RL made it more likely".
- **Curried normal form:** a theorem with every leading antecedent of the goal moved into the premises, so ⊢ A → B and
  A ⊢ B count as one problem.
- **Cut-off fraction:** the share of sampled attempts that hit a length cap (tokens, steps or action length) rather
  than finishing.
- **Frozen control:** the starting model, sampled with the same number of attempts as the trained arm, without
  training.
- **Pool:** a fixed set of theorems used as RL targets or for evaluation.
- **Renaming class:** a theorem up to relabelling its atoms; two theorems in one class are the same problem.
- **Start-index normalisation:** renumbering a proof's line or hypothesis names in order of first appearance, so the
  same proof written with different names counts once.
- **Verdict marker:** a tag a judge attaches to a rejected sample (e.g. `LEANREJ`). It must never reach training text.
