---
written_on: 2026-10-18
written_by: agent:claude
---

# Relevant model checkpoints

The checkpoints worth starting from, one row per model. Each row gives where the weights are, the
code that produced them, and why the model matters. Scores in different rows use different
protocols; compare them only where the protocol column matches. Datasets are in
[DATASETS.md](DATASETS.md); the training pipelines are in [PIPELINES.md](PIPELINES.md).

Robbie's checkpoints all live in one Hugging Face model repo,
[`robbiethompson2018/nd-rl-checkpoints`](https://huggingface.co/robbiethompson2018/nd-rl-checkpoints),
one directory per seed run: `runs/<run-id>_s<seed>/`. A run directory holds `pretrain.pt`
(pretraining only), `rl_round_*.pt`, `final.pt` (after RL), `summary.json`, the eval files, and
`source/`, the exact files the run executed. Download one run with

```
hf download robbiethompson2018/nd-rl-checkpoints --include 'runs/<run-id>_s0/*' --local-dir checkpoints
```

## Whole-proof models (Robbie)

These write a whole `lean_seq` proof in one pass. Model code is Dan's take-home fork
(`dan-pandori/nd-takehome` at `51604b3`) plus the run's own `source/pretrain.py`.

| model | weights | source | protocol and score | why it matters |
| --- | --- | --- | --- | --- |
| Run 198, pretrain | `runs/198-claude_lean-muon-no-rms-retest_s{0,1,2}/pretrain.pt` | `pretrain.py` at `c891da88` on `autoresearch-pretrain` (sha256 prefix `6bdbe0047ac4`); also in each run's `source/` | autoresearch harness, 300 s pretraining on Dan's cap-6 file, then 4 rounds of the old EI: score 681 | The chosen pretraining recipe: ALiBi/NoPE heads, Peri-LN, an MTP head, orthogonal init, Polar Express Muon, 9.56M parameters. Every later whole-proof model starts from this file. [Summary](https://github.com/chainik1125/nd-rl/blob/autoresearch-pretrain/experiment-summaries/2026-09-21-pretrain-ablations/README.md). |
| Run 590, 198 + the loops EI recipe | `runs/590_control-untied-198_s{0,1,2}/final.pt` | `1dc29530` on `autoresearch-loops`; files in each run's `source/` | loops harness, 1,800 s of RL: 792 dev theorems (790 / 785 / 800) of the 1,108-theorem ladder transfer dev half, 64 samples | The best whole-proof model on short theorems (length 7+, cap-6 pretraining). Its RL recipe (runs 507, 509, 510) is the one the long-RL pipeline uses. [Summary](https://github.com/chainik1125/nd-rl/blob/autoresearch-loops/experiment-summaries/2026-10-08-test-time-loops/README.md). |
| Long RL v5, untied EI | `runs/long-rl-20261014-v5_untied-ei_s{0,1}/final.pt`, with intermediate `rl_round_0NN.pt`; its `pretrain.pt` is in `runs/long-rl-20261014-v2_untied-ei_s{0,1}/` | [`code/experiments/current/long_rl_20261014`](../code/experiments/current/long_rl_20261014/README.md), identical to the v5 runs' `source/` | Lean-only judge, about 7 h of RL on body lengths 7–45: dev 313 / 433 of 843 (64 samples) | The best model on long proofs: pretrained only on proofs of ≤ 6 body lines, extended by RL. The default whole-proof starting point. |
| Long RL v5, looped EI | `runs/long-rl-20261014-v5_looped-ei_s{0,1}/final.pt`; `pretrain.pt` in the matching `v2` run | as above | as above: dev 376 / 372 of 843 at 8 loops | The looped-trunk counterpart (run 512's architecture). Kept for comparison; loops did not buy test-time scaling. |

Long-RL lengths are **body lines**: lines after the premise block, so nd_verify's `n_lines` minus
the premise count. Per-band results for every long-RL run are in the
[long-RL README](../code/experiments/current/long_rl_20261014/README.md).

## Proof-state and thinking models (Dan, Leon)

| owner | model | weights | source | protocol and score | why it matters |
| --- | --- | --- | --- | --- | --- |
| Dan | `state_env_october_r16` ([model card](https://github.com/chainik1125/nd-rl/blob/dan/docs/models/state_env_october_r16.md)) | bucket `hf://buckets/dan-pandori/nd-rl/rl-continue/ckpts/rc/ladder/la_T1_best12_s{0,1,2}_r16.pt` | `dan-pandori/nd-takehome`, branch `dan` (`state_env.py`, `state_sample.py`, `state_ladder_ei.py`) | textbook72 pass@256: 51 / 57 / 56 of 72 (plain sampling, T 0.8, Lean) | Robbie's 9.56M ALiBiGPT recipe on cap-12 proofs in the proof-state environment, then 16 rounds of EI. |
| Leon | GRPO "merged recipe + KL 0" (`grpo_think_vs_mb_kkl0_s{0,1,2}_L3`) | not on Hugging Face: `/scratch/cowifornia/nd-rl/think/runs/grpo_v2/` | [`RECIPE.md`](../code/experiments/current/thinking_traces_20261005/RECIPE.md) | 3 seeds, plain reads, pass@1: 32.1% on 13–20, 74.0% on textbook72 | The group's primary RL recipe from 2026-10-16: the Stage-1 search-trace model S1g3, then GRPO with thinking. |
| Charles | TODO | | | | |
| Dmitry | TODO | | | | |
