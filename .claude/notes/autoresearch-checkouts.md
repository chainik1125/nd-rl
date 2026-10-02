---
written_on: 2026-09-22
written_by: agent:codex
---

# Autoresearch: separate research and development checkouts

Robbie requested separate full clones so tooling can change without disturbing a running
research agent. These are not Git worktrees.

**Status 2026-10-02: the loops are stopped and this layout is retired.** `autoresearch-pretrain` is
fully merged into `robbie-experiments`, which is now Robbie's working branch (TTRL and every
experiment since 2026-09-28). `nd-rl-2`, `nd-rl-3`, `nd-rl-4` and `nd-rl-5` were deleted;
nd-rl-3's loop session logs were kept in `nd-rl/artifacts/autoresearch/loop-from-nd-rl-3/`. The
`autoresearch-pretrain` branch stays because the dashboard reads results from it. The table below
is the historical layout: to restart a loop, make fresh checkouts with the same roles.

| Checkout | Branch | Role |
| --- | --- | --- |
| `/Users/robbie/repos/nd-rl` | `autoresearch-pretrain` | Researcher: the Fable loop since 2026-09-25 (`--agent fable`); Codex (retired) ran here before. |
| `/Users/robbie/repos/nd-rl-3` | `autoresearch-pretrain` | Researcher: run the Claude loop here (`--agent claude`). |
| `/Users/robbie/repos/nd-rl-2` | `autoresearch-dev` | Development: edit/test orchestration and dashboard here. |

Since 2026-09-23 the two researcher checkouts share one branch and push to it concurrently;
`params.py`/`record.py` catch up before each publish (see [LOOP.md](../../code/experiments/current/autoresearch/LOOP.md)).
Stopping "the loop" to integrate tooling now means stopping both.

Each checkout owns its `.git`, `code/.venv`, tracked code/results, and ignored `artifacts/`.
Do not share artifacts: they contain pod ownership, active session manifests, and per-attempt
outputs. Do not run a second experiment loop from the development checkout.

The development checkout has relative symlinks for `.envrc.local` and `CLAUDE.local.md` to the
research checkout. Credentials have one source of truth. Its ignored `.envrc` loads the shared
credentials through direnv. Do not commit secrets or local paths into experiment registrations.

## Watch live results

From any checkout:

```bash
python3 code/experiments/current/autoresearch/dashboard/serve.py
```

By default the dashboard reads `results/` from `origin/autoresearch-pretrain`, fetching at most
every 20 s. It therefore shows both agents' pushed registrations and results, whichever
checkout serves it. Refresh to see new runs. The header counts keeps per agent since 041.
Only run one dashboard server at a time, because every checkout uses port 7102.

`nd-rl-2/live-results` is an ignored symlink to
`../nd-rl/code/experiments/current/autoresearch/results`, for `--results-dir live-results`. It
only sees the Codex checkout, and only what that checkout has pulled. It is a read-only *usage
convention*: writes through it would change the researcher checkout.

## Move tooling changes into the researcher

Commit and test tooling in `autoresearch-dev`. Stop the research loop before integrating those
commits into `autoresearch-pretrain`; never pull, merge, or cherry-pick into its running checkout.
Use selective cherry-picks for tooling-only commits. Preserve any registered `pretrain.py`
candidate; don't restore it to make the working tree clean. Run `make -C code check` and
`make -C code test` before Robbie restarts the loop from his terminal.

Research results continue to be committed/pushed on `autoresearch-pretrain`; the dashboard's
live alias does not require pulling or merging those commits into development.

## Restart after cancellation

The normal [loop command](../../code/experiments/current/autoresearch/LOOP.md) recognizes exactly
one unfinished, committed preregistration when its candidate still matches. It validates the
hashes, seeds, and absence of an active previous attempt, then gives a fresh Codex session the
existing experiment to finish. Training restarts from scratch; it is not checkpoint recovery.
Old attempt artifacts are archived, not reused as new scores.

At setup, experiment `015_beta2-099` is preregistered but unfinished. Its one-line beta2 edit
remains in the researcher checkout only. The development checkout starts from the committed
incumbent, not that pending candidate. No experiment is launched by creating this checkout.
