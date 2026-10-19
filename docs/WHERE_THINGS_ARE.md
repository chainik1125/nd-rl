---
written_on: 2026-09-30
written_by: agent:claude
---

# Where things are

One page for finding each person's weights, datasets, best models, branches and write-ups. Most
work lives on personal branches and in three Hugging Face stores, not on `main`. Each owner keeps
their own rows current. Everything below was checked on 2026-09-30 unless marked TODO.

## 1. Weights and datasets on Hugging Face

The group has a Hugging Face org, [`ndrl`](https://huggingface.co/ndrl).
[Join it with this link](https://huggingface.co/organizations/ndrl/share/ZpkXbAYdVXIMlJAOcWQNzQpAJVbqVamXJq);
members get write access to every repo in the org, including private ones.

| Owner | Location | Layout | What's there |
| --- | --- | --- | --- |
| Dan | bucket [`hf://buckets/dan-pandori/nd-rl/`](https://huggingface.co/buckets/dan-pandori/nd-rl) | `<run-id>/{ckpts,artifacts,data}/`, plus `registry/<run-id>/` | About 40 runs from Dan's agent campaign (`cap-horizon`, `state-env`, `support-curves`, ...). Datasets live under each run's `data/`, e.g. `cap-horizon/data/kh/train_k12.jsonl.gz`. `registry/` holds the per-run results rows (see §2). |
| Robbie | model repo [`robbiethompson2018/nd-rl-checkpoints`](https://huggingface.co/robbiethompson2018/nd-rl-checkpoints) | `runs/<RUN_ID>_s<seed>/`, `data/<name>/`, `sources/`, `pretrain_poc/`, `previous_attempts/` | Autoresearch runs uploaded by `pod.py`, one directory per seed: `pretrain.pt`, `rl_round_0N.pt`, `final.pt` (includes RL), `summary.json`, `source/pretrain.py`. `data/` holds Robbie's datasets ([DATASETS.md](DATASETS.md)). |
| Robbie | private dataset [`ndrl/nd-rl-autoresearch`](https://huggingface.co/datasets/ndrl/nd-rl-autoresearch) (join the org to see it) | `<checkout>/<run>/`, one folder per local checkout (`nd-rl`, `nd-rl-2` … `nd-rl-9`) | Everything that was in `artifacts/autoresearch/` on Robbie's laptop on 2026-10-01, including runs that never reached `nd-rl-checkpoints` (e.g. the `st-*` runs). Same files per run; JSONL over 1 MB is gzipped. Agent session transcripts (`loop/`) are not included. |
| Leon | TODO | | |
| Charles | TODO | | Charles's branch loads Robbie's run 039 checkpoint from Robbie's repo. |
| Dmitry | TODO | | |

Browse the bucket with `hf buckets ls dan-pandori/nd-rl` (add `-R` to recurse). Download from
Robbie's repo with
`hf download robbiethompson2018/nd-rl-checkpoints --include 'runs/RUN_ID/*' --local-dir checkpoints`,
and from the org dataset with
`hf download ndrl/nd-rl-autoresearch --repo-type dataset --include 'nd-rl-8/RUN_ID/*' --local-dir autoresearch`.

The autoresearch loop's frozen training data is not on HF. It is `data/train.jsonl.gz` in
[`dan-pandori/nd-takehome`](https://github.com/dan-pandori/nd-takehome) at commit `51604b3`
(the `remote.sh` pin on `autoresearch-pretrain`).

[`infra/README.md`](infra/README.md) says all large artifacts go to "the HF bucket". In practice
there are three stores, above; which one new work should use is still open.

## 2. Best models

Moved to [MODELS.md](MODELS.md) (2026-10-18): one row per relevant checkpoint with its weights,
source commit and protocol. Datasets are in [DATASETS.md](DATASETS.md) and the two training
pipelines in [PIPELINES.md](PIPELINES.md).

## 3. Branches

Only `main` is shared. Everything else is a personal working branch; don't assume it's merged or
reviewed. Commit counts are relative to `main` on 2026-09-30.

Each person also keeps one branch as their own main, holding their finished work; start there for
someone's results:

- Leon: `leon/main`. Finished work is merged here from his feature branches.
- Dan: `dan`, his integration branch, which his per-run `dan_<run-id>` branches merge into.
- Robbie: `robbie-experiments`, his write-ups and experiment code. The autoresearch loop's own log
  is `autoresearch-pretrain`.

| Owner | Branch | Purpose |
| --- | --- | --- |
| Dan | `dan` | Integration branch: agent orchestration, proposals, run summaries, registry. Open as PR #16. |
| Dan | `dan_<run-id>` (54) | One branch per agent run, written by his executor/librarian agents and merged into `dan`. Read `dan` rather than these. |
| Robbie | `robbie-experiments` | Writeups, blog drafts and one-off experiment code. |
| Robbie | `autoresearch-pretrain` | The autoresearch loop's log: one pre-registration commit and one result commit per run. Pushed by the loop; never rebased. |
| Robbie | `combined-pipeline`, `robbie_week1`, `autoresearch-dev` | Older work, superseded by the two branches above. |
| Leon | `leon/main` | Leon's main branch: all his finished work merged together. It is built on `dan` (the benchmarks evaluate Dan's models), so it also contains Dan's integration branch and `main`. |
| Leon | `leon/home-rolled-rl` | Home-rolled RL hill-climbing, length generalisation, mid-training. Merged into `leon/main`. |
| Leon | `leon/benchmarking-dan` | `ndbench` benchmarks, EDL, interp measures, dashboard, and their run on Dan's best-state models; `leon/benchmarking` rebased onto `dan`. Merged into `leon/main`. |
| Leon | `leon/rl-dashboards` | RL run dashboard proposal. Merged into `leon/main`. |
| Leon | `leon/nd-verify-fix` | `nd_verify` with the forward box-citation fix (PR #22). Merged into `leon/main`. |
| Leon | `leon/benchmarking` | Superseded by `leon/benchmarking-dan`. |
| Charles | `charles/main` | Workstream 3 experiments (`experiments/ws3/`) and textbook-problem data (`data/textbook_problem/`). |
| Dmitry | `dmitry/cap-comparison-textbook-eval-data` | Textbook eval data for the cap comparison. |

List current branches with `git fetch --prune && git branch -r --sort=-committerdate`.

## 4. Curated write-ups

`main`'s [experiment-summaries index](../experiment-summaries/README.md) stops at the
2026-09-10 imports. Later write-ups live on blogs or on their owners' branches. Branch-based
write-ups each have their own `experiment-summaries/README.md` index.

| Owner | Link | Branch | Covers |
| --- | --- | --- | --- |
| Dan | [experiment-summaries](https://github.com/chainik1125/nd-rl/tree/dan/experiment-summaries), [STATE.md](https://github.com/chainik1125/nd-rl/blob/dan/docs/STATE.md), [weekly digests](https://github.com/chainik1125/nd-rl/tree/dan/docs/digests) | `dan` | His agent campaign: dataset composition, cap horizon, state environment, GRPO, support curves, search expert, ... |
| Robbie | [Pretraining ablations, week of September 21, 2026](https://robbiewmthompson.com/blog/pretraining-ablations/) | | Curated explanation of the week's work. |
| Robbie | [The combined model, and where its gains come from](https://robbiewmthompson.com/blog/combined-model/) (draft, unlisted) | | Curated explanation of the week's work. |
| Leon | [experiment-summaries](https://github.com/chainik1125/nd-rl/tree/leon/home-rolled-rl/experiment-summaries) | `leon/home-rolled-rl` | RL hill-climbing A/B and run 2, length generalisation, representation for length, mid-training on glued proofs. |
| Leon | [benchmarking_20260930](https://github.com/chainik1125/nd-rl/tree/leon/benchmarking/code/experiments/current/benchmarking_20260930) | `leon/benchmarking` | Benchmark plan and runbook. |
| Charles | [experiments/ws3](https://github.com/chainik1125/nd-rl/tree/charles/main/experiments/ws3) | `charles/main` | Per-experiment `status.md` and per-run `report/report.md`. |
| Dmitry | TODO | | |

Read a file off a branch without checking it out:
`git show origin/<branch>:<path>`.
