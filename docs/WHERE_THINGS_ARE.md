---
written_on: 2026-09-30
written_by: agent:claude
---

# Where things are

One page for finding each person's weights, datasets, best models, branches and write-ups. Most
work lives on personal branches and in two Hugging Face stores, not on `main`. Each owner keeps
their own rows current. Everything below was checked on 2026-09-30 unless marked TODO.

## 1. Weights and datasets on Hugging Face

| Owner | Location | Layout | What's there |
| --- | --- | --- | --- |
| Dan | bucket `hf://buckets/dan-pandori/nd-rl/` | `<run-id>/{ckpts,artifacts,data}/`, plus `registry/<run-id>/` | About 40 runs from Dan's agent campaign (`cap-horizon`, `state-env`, `support-curves`, ...). Datasets live under each run's `data/`, e.g. `cap-horizon/data/kh/train_k12.jsonl.gz`. `registry/` holds the per-run results rows (see §2). |
| Robbie | model repo [`robbiethompson2018/nd-rl-checkpoints`](https://huggingface.co/robbiethompson2018/nd-rl-checkpoints) | `runs/<RUN_ID>_s<seed>/`, `sources/`, `pretrain_poc/`, `previous_attempts/` | Every autoresearch run, one directory per seed: `pretrain.pt`, `rl_round_0N.pt`, `final.pt` (includes RL), `summary.json`, `source/pretrain.py`. |
| Leon | TODO | | |
| Charles | TODO | | Charles's branch loads Robbie's run 039 checkpoint from Robbie's repo. |
| Dmitry | TODO | | |

Browse the bucket with `hf buckets ls dan-pandori/nd-rl` (add `-R` to recurse). Download from
Robbie's repo with
`hf download robbiethompson2018/nd-rl-checkpoints --include 'runs/RUN_ID/*' --local-dir checkpoints`.

The autoresearch loop's frozen training data is not on HF. It is `data/train.jsonl.gz` in
[`dan-pandori/nd-takehome`](https://github.com/dan-pandori/nd-takehome) at commit `51604b3`
(the `remote.sh` pin on `autoresearch-pretrain`).

[`infra/README.md`](infra/README.md) says all large artifacts go to "the HF bucket". In practice
there are two stores, above; which one new work should use is still open.

## 2. Best models

A "best" model is a claim about one person's metric and protocol. Two numbers from different rows
of this table aren't comparable unless their protocols match.

| Owner | Model | Where | Metric and protocol |
| --- | --- | --- | --- |
| Robbie | Autoresearch run 198 (`198-claude_lean-muon-no-rms-retest`): ALiBi/NoPE heads, Peri-LN, MTP head, Polar Express Muon | `runs/198-claude_lean-muon-no-rms-retest_s{0,1,2}/` in Robbie's HF repo; source `pretrain.py` at `c891da8` on `autoresearch-pretrain` | Score 681 on the autoresearch harness: 300 s pretraining, then fixed expert-iteration RL, Lean-format fork `51604b3`. `pretrain.pt` is pretraining only; `final.pt` includes RL. |
| Dan | TODO | | Dan's results registry lists checkpoint URIs and headline metrics for his runs: rows in the bucket's `registry/`, described in `experiment-summaries/2026-09-29-results-registry-ckpt-upload-and-results-table/README.md` on `origin/dan`. |
| Leon | TODO | | |
| Charles | TODO | | |
| Dmitry | TODO | | |

## 3. Branches

Only `main` is shared. Everything else is a personal working branch; don't assume it's merged or
reviewed. Commit counts are relative to `main` on 2026-09-30.

| Owner | Branch | Purpose |
| --- | --- | --- |
| Dan | `dan` | Integration branch: agent orchestration, proposals, run summaries, registry. Open as PR #16. |
| Dan | `dan_<run-id>` (54) | One branch per agent run, written by his executor/librarian agents and merged into `dan`. Read `dan` rather than these. |
| Robbie | `robbie-experiments` | Writeups, blog drafts and one-off experiment code. |
| Robbie | `autoresearch-pretrain` | The autoresearch loop's log: one pre-registration commit and one result commit per run. Pushed by the loop; never rebased. |
| Robbie | `combined-pipeline`, `robbie_week1`, `autoresearch-dev` | Older work, superseded by the two branches above. |
| Leon | `leon/home-rolled-rl` | Home-rolled RL hill-climbing, length generalisation, mid-training. |
| Leon | `leon/benchmarking` | `ndbench` benchmarks, EDL, interp measures, dashboard. |
| Leon | `leon/rl-dashboards` | RL run dashboard proposal. |
| Charles | `charles/main` | Workstream 3 experiments (`experiments/ws3/`) and textbook-problem data (`data/textbook_problem/`). |
| Dmitry | `dmitry/cap-comparison-textbook-eval-data` | Textbook eval data for the cap comparison. |

List current branches with `git fetch --prune && git branch -r --sort=-committerdate`.

## 4. Curated write-ups

`main`'s [experiment-summaries index](../experiment-summaries/README.md) stops at the
2026-09-10 imports. Later write-ups live on their owners' branches, each with its own
`experiment-summaries/README.md` index:

| Owner | Where | Covers |
| --- | --- | --- |
| Dan | `origin/dan`: `experiment-summaries/`, `docs/STATE.md`, weekly digests in `docs/digests/` | His agent campaign: dataset composition, cap horizon, state environment, GRPO, support curves, search expert, ... |
| Robbie | `origin/robbie-experiments`: `experiment-summaries/2026-09-2*`, `2026-09-30-*` | Pretraining ablations, combined pipeline, test-time scaling, bigger model/longer pretraining. Autoresearch results: `code/experiments/current/autoresearch/RESULTS.md` on `autoresearch-pretrain`. |
| Leon | `origin/leon/home-rolled-rl`: `experiment-summaries/2026-09-2*` | RL hill-climbing A/B and run 2, length generalisation, representation for length, mid-training on glued proofs. |
| Leon | `origin/leon/benchmarking`: `code/experiments/current/benchmarking_20260930/` | Benchmark plan and runbook. |
| Charles | `origin/charles/main`: `experiments/ws3/<exp>/status.md` and `<exp>/<run>/report/report.md` | Per-experiment status and per-run reports. |
| Dmitry | TODO | |

Read a file off a branch without checking it out:
`git show origin/<branch>:<path>`.
