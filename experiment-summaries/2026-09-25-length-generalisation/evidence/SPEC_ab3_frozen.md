# RL hill-climb, run 3 ("ab3"): harness specification

Written 2026-09-24 by the harness engineer session (agent:claude) for Leon's workstream 4 ("custom RL
pipeline") of the nd-rl project. It replaces run 2's contract (`/scratch/cowifornia/nd-rl/ab2/frozen/SPEC.md`)
and keeps every rule of it that is not about the budget. This file is the contract between the frozen
harness, the compute meter, the editable RL file, the scorer, the GEAR controller and the agents. Nothing under
`ab3/frozen/` may be edited by an agent.

Run 3 has one arm, GEAR-Evolve: four hill-climbing workers (`w1` ... `w4`) and one auditor share five RTX 3090s
(host GPUs 0-4) under the GPU lock. The agents are in-process sessions on the host and use host paths.

**What changed from run 2.** Runs 1 and 2 gave every experiment a wall-clock budget, so most of the gains came
from sampling faster. Run 3 gives every experiment the same **metered compute budget** (`DEFAULT_FLOPS`,
section "Compute budget"): `frozen/meter.py` charges every `nn.Linear` call and the score depends on the
compute, not the card. The wall clock is only a 3,600 s safety cap. Throughput work (CUDA graphs, compaction,
faster verification) is still welcome because it makes experiments finish sooner, but it no longer raises a
score, except where it removes charged rows (padding) rather than time. The scorer also reports aggregate
pass@1 and pass@4 estimates, which the run-3 controller's promotion score uses. The seed `rl.py` is plain
expert iteration with run 2's speed parts and index fix only (section "Seed").

## The task

Start from the fixed pretrained checkpoint `frozen/pretrain.pt`: 4 layers, d=256, 8 heads, absolute tokenizer,
3.2M parameters, pretrained on cap-6 natural-deduction proofs, no RL. Within the compute budget, train it to
prove longer theorems. Only the scorer measures this. After every experiment it samples the dev set
`dev_L8-11` (1,200 theorems, k = 32) and the frontier set `frontier_L12-14` (a fixed 1,000-theorem subset, k = 64)
at temperature 0.8 and reports, per set, the pass@k count (theorems with at least one of k samples accepted by
`nd_verify` as a proof of the prompted sequent) and the pass@1 and pass@4 estimates from the same samples. The
run-3 controller (`ab3/controller.py`, v18) promotes on

    score = (dev pass@1 + dev pass@4) + 4 x (frontier pass@1 + frontier pass@4)

with each pass@j as expected theorems solved (the harness's per-set mean x the set size; see "Scoring"). The
pass@32 / pass@64 counts (`metric`, `frontier`) stay in every summary. They are secondary.

After the search window closes, the monitor evaluates the best checkpoints on the sealed sets
(`monitor/final_eval_ab3.sh`, set list in `ab3/sets.env`). No agent sees the sealed sets.

## Sets

Run 3 reuses run 2's pools unchanged (PI decision 2026-09-24). File names are constants at the top of
`harness.py` (`TARGETS_FILE`, one constant for the RL pool) and `score_host.py` (`DEV_FILE`, `FRONTIER_FILE`,
`SETS`); `ab3/sets.env` is the monitor's single list of them.

| set | file | who reads the file | who sees the numbers | scored by |
|---|---|---|---|---|
| RL pool | `frozen/data/targets_L6-11.jsonl` (run 2's), 15,738 targets, true length 6-11, no proofs | agents (through `rl.run`'s `targets`) | - | - |
| dev | `ab2/data/dev_L8-11.jsonl` | scorer only | agents (aggregate and per-length) | daemon, k = 32, max_new 512 |
| frontier | `ab2/data/frontier_L12-14.jsonl`, scored on a fixed stratified subset of 1,000 (700 / 200 / 100 at lengths 12 / 13 / 14, seed 20260924) | scorer only | agents (aggregate and per-length) | daemon, k = 64, max_new 512 |
| sealed L12-14 | `ab2/sealed/sealed_L12-14.jsonl` | monitor only | monitor only | `score_host.py` only, max_new 640 |
| sealed L15-16 | `ab3/sealed/sealed_L15-16.jsonl` (run 3's; run 2's 9-theorem file is no longer used) | monitor only | monitor only | `score_host.py` only, max_new 1,024 minus the longest prompt |
| sealed L17-18 | `ab3/sealed/sealed_L17-18.jsonl` | monitor only | monitor only | `score_host.py` only, max_new 1,024 minus the longest prompt |
| matched L6-14 | `ab3/sealed/matched_L6-14.jsonl` | monitor only | monitor only | `score_host.py` only, max_new 640 |
| textbook | `ab/sealed/textbook_dev.jsonl` | monitor only | monitor only | `score_host.py` only, max_new 512 |
| cap-6 train | `frozen/data/train_cap6.jsonl`, 154,990 records with proofs (the pretraining data) | agents | - | - |
| cap-6 held-out | `frozen/data/heldout_cap6.jsonl`, 5,000 records with proofs | harness (damage check) | agents (`heldout_greedy`) | harness |

All pools and sets are meant to be pairwise disjoint by atom-renaming class (`lib/keys.py`). `score_host.py`
takes k from 1 to 256 for every set and clamps every set's `max_new` so that prompt + max_new <= 1,024 tokens
(the GPT's `max_len`). The L15-18 values are placeholders until the monitor measures tokens per line on those
sets' proofs.

## Directory layout

```
/scratch/cowifornia/nd-rl/ab3/       the arm directory (daemon arm "ab3", host GPUs 0-4)
  frozen/                    the frozen files (this SPEC, harness, meter, scorer, lib, data); read-only for agents.
                             The monitor runs the daemon from here (frozen/score_daemon.py -> frozen/score_host.py)
  seed/                      the run-3 seed: rl.py, lib_run2.py (optional run-2 levers), test_seed.py, tests/, runs/
  calib/                     the compute calibration (n092 recipe, graph / no-graph) and the meter validation
  controller.py, test_controller.py, GEAR_README.md, gear_state.json, nodes/    GEAR controller and its state
  reflections.md, decisions.md, ideas.md, digest.md, champion.md, audit/        shared logs
  w1/ ... w4/, auditor         one workspace per agent: rl.py (editable), runs/<name>/
  score_requests/, score_results/   the scoring queue (client <-> daemon)
  logs/                      scores.jsonl (clients), daemon_scores.jsonl (daemon)
  sealed/, monitor/, data_build/, lengthexp/, briefs/, sets.env     monitor only
/scratch/cowifornia/nd-rl/ab2/data/     dev and frontier theorems (scorer only)
/scratch/cowifornia/nd-rl/ab2/sealed/   run 2's sealed sets (monitor only)
/scratch/cowifornia/nd-rl/ab2/locks/    the host-wide GPU locks (shared with run 2)
```

Frozen files, by role:

| file | role |
|---|---|
| `frozen/harness.py` | one experiment: lock a GPU, load the checkpoint, install the meter, import and call `rl.run`, enforce the compute budget and the wall cap, save `final.pt`, score dev and frontier, write `summary.json` |
| `frozen/meter.py` | the compute meter (section "Compute budget") |
| `frozen/score.py` | score client: writes a request, waits for the daemon, prints aggregate counts, writes `score.json` |
| `frozen/score_host.py`, `frozen/score_daemon.py` | the host-side scorer and daemon (run by the monitor; agents never run them) |
| `frozen/gpulock.py` | exclusive per-GPU locks with detached holders and a FIFO queue |
| `frozen/metrics.py` | per-step metrics logger (JSONL) |
| `frozen/lib/` | model, tokenizer, sampler, supervised-loss helpers, verifier, prune, normalize, eval helpers, class keys |
| `frozen/data/` | `targets_L6-11.jsonl`, `train_cap6.jsonl`, `heldout_cap6.jsonl`, `SHA256SUMS` |
| `frozen/pretrain.pt` | the starting checkpoint (`{'state', 'tok_mode': 'abs', ...}`; construct `GPT(tok.vocab_size, 4, 256, 8)`) |
| `frozen/test_meter.py`, `frozen/test_score_host.py` | regression tests of the meter and of the pass@j estimators |
| `frozen/rl.py` | run 2's seed (run 1's n025), for reference only; the run-3 seed is `ab3/seed/rl.py` |
| `frozen/ref/` | reference material from run 2 (the fork's `expert_iter.py` and `grpo.py`, the GEAR paper, the logic spec, ...) |
| `frozen/SHA256SUMS` | `sha256sum` of every file under `frozen/` (see "Build notes") |

## Harness contract (`frozen/harness.py`)

```
python frozen/harness.py --out runs/<name> [--seed 0] [--rl rl.py] [--flops F] [--budget 3600]
                         [--gpu auto|N|env] [--gpus 0,1,2,3,4] [--gpu-wait 14400] [--no-score] [--smoke]
```

0. **GPU.** `--gpu auto` (the default) acquires the first free GPU of `--gpus` through `gpulock.py`, waiting up to
   `--gpu-wait` seconds (default 4 h). It exports `CUDA_VISIBLE_DEVICES=<gpu>` for itself and for the scoring
   subprocesses, and releases the lock when it exits (normally, on a crash, on a timeout, on SIGTERM or SIGHUP;
   if the harness is SIGKILLed the lock holder frees the GPU within 2 s). `--gpu N` locks GPU N. `--gpu env` takes
   no lock and uses `CUDA_VISIBLE_DEVICES` as it is; it is for the monitor and for tests only.
1. Seeds torch and Python. Loads `frozen/pretrain.pt` into `lib.model.GPT(vocab, 4, 256, 8)` with the `abs`
   tokenizer. Copies `rl.py` into `runs/<name>/source/` and records the SHA-256 of `rl.py`, `harness.py`,
   `meter.py`, `score.py`, `metrics.py`, `gpulock.py`, `pretrain.pt`, every file in `lib/` and every file in `data/`
   into `summary.json`. Only `rl.py` is copied and hashed from the agent's side: code it imports from elsewhere
   (e.g. `seed/lib_run2.py`) must be recorded by `rl.py` itself (the seed logs lib_run2's SHA-256).
2. Installs the compute meter with limit `--flops` (default `DEFAULT_FLOPS`), sets SIGALRM, then imports the RL
   module (so import-time code is metered and on the clock) and calls

   ```python
   model = rl.run(model, tok, targets, budget_secs, device, seed, log, metrics, paths)
   ```

   | argument | meaning |
   |---|---|
   | `model` | the loaded GPT, on `device`; train mode allowed |
   | `tok` | `lib.tokenizer.Tokenizer('abs')` |
   | `targets` | list of dicts from `TARGETS_FILE` (`name, thm, key, pkey, prompt, n_lines, L_true, source, schema, n_prem`) |
   | `budget_secs` | the wall-clock safety cap for the whole call (default 3,600 s) |
   | `device` | `'cuda'` |
   | `seed` | int |
   | `log(msg)` | timestamped print |
   | `metrics` | a `metrics.Metrics` instance (schema below) |
   | `paths` | dict: `targets`, `train_cap6`, `heldout_cap6`, `out_dir`, `save_checkpoint(model, tok, path)`, `heldout_greedy(model, tok, n=1000)` (runs the model: charged), `smoke` (bool), **`budget`**: the meter (`used()`, `remaining()`, `frac()`, `limit`, `snapshot()`; `limit` is `inf` for an unmetered run) |

   `rl.run` must return the model to be scored, with the same architecture. It may save intermediate checkpoints
   under `paths['out_dir']` with `paths['save_checkpoint']`.
3. **Budget exhaustion.** Once the charged total reaches the limit, every further `nn.Linear` call and every
   CUDA-graph replay raises `meter.BudgetExhausted` (a `BaseException`, so `except Exception` does not catch it).
   If it propagates out of `rl.run`, the harness scores **the model object it passed to `rl.run`, as it stands**
   (mid-round weights, no final averaging), with `status: ok` and `budget_exhausted: true`. `rl.run` may catch
   it and still return a model built without model compute (e.g. an average of saved weights). Well-behaved code
   polls `paths['budget']` and returns first; the seed returns with at most ~1.5% of the budget unused.
4. **The wall-clock cap.** SIGALRM fires at `budget_secs + 45` and the run is recorded as `status: timeout` with
   no score; returning later than `budget_secs + 15` is also a timeout. The cap is 3,600 s by default; a
   compute-bounded seed run takes about 20 minutes on a power-capped 3090.
5. After `rl.run` returns (or the budget ran out), the meter is uninstalled: everything below is free. `final.pt`
   is saved, and `heldout_greedy` is computed on a fixed 1,000-record slice of `heldout_cap6.jsonl` (every 5th
   record, 200 per length 2-6; greedy decoding). Unless `--no-score` is given, the harness then runs
   `frozen/score.py --set dev --k 32 --seed 0` and `frozen/score.py --set frontier --k 64 --seed 0`, in that order,
   on the same GPU. Before any training, the harness refuses a targets file with duplicate `name` fields.
6. `summary.json` fields:
   - `status`: `ok`, `timeout`, `crash`, or `scoring` (provisional, while scoring runs).
   - `metric`: the dev pass@32 total. `by_len`: dev solved per length. `frontier`, `frontier_by_len`: the same at
     pass@64 on the frontier subset.
   - `pass1`, `pass1_by_len`, `pass4`, `pass4_by_len` (dev) and `frontier_pass1`, `frontier_pass1_by_len`,
     `frontier_pass4`, `frontier_pass4_by_len`: the aggregate estimates, as means over theorems in [0, 1]
     (multiply by `score.n` / `frontier_score.n` for expected theorems solved).
   - `compute`: the meter's snapshot (`limit_flops` (null if unmetered), `used_flops`, `frac`, `sample_flops`
     (charged without autograd), `train_flops` (charged with autograd, 3x included), `linear_calls` (eager and
     traced; replays are counted in `graph_replays`), `optimizer_steps`, `graph_replays`, `exhausted`),
     `flops_budget` (the limit; null if unmetered), `budget_exhausted` (true if `BudgetExhausted` escaped `rl.run`).
   - `heldout_greedy`, `rl_secs`, `score_secs` (`dev_score_secs`, `frontier_score_secs`), `rl_stats` (whatever
     `rl.run` put in `metrics.summary`).
   - `score` and `frontier_score` (the full score JSONs, with the denominators in `n_by_len`), `score_line`,
     `frontier_line`.
   - `shas`, `seed`, `budget` (the wall cap), `gpu`, `gpu_lock`, `gpu_wait_secs`, `worker` (env `AB_WORKER`),
     `started_utc`, `run_name`, `arm`, `smoke`, `targets_file`, `total_secs`, and `reason` on a crash or timeout.
7. `--smoke`: 8 targets, `SMOKE_FLOPS` = 1e12 of compute (about two seed smoke rounds), a 60 s wall cap, 64
   held-out records, and 16-theorem subsets of dev and frontier. For plumbing checks only; a smoke run is never
   promoted. `AB_SCORE_STUB=1` replaces the daemon with an all-zero local result (tests only).

Checkpoints (`final.pt` and anything saved through `save_checkpoint`) have the layout
`{'cfg', 'state', 'tok_mode', 'extra', 'stage', 'seed', 'shas'}`. No `final.pt` is written on a timeout or crash.
Data hashes are cached in `data/SHA256SUMS`, recomputed whenever the file list changes or a data file (or
symlink) was created or changed after the cache was written.

## Compute budget

**The budget.** `DEFAULT_FLOPS = 3.9e15` metered FLOPs per experiment (`frozen/harness.py`). It is 30 rounds of
run 2's n092 recipe (30 is what n092 ran in run 2's same-card replicate: dev 364 / 367, frontier 226 / 229 on two
seeds). Calibration (`ab3/calib/runs/cal_graph`, 2026-09-24): n092's `rl.py` with its CUDA-graph line-verified
sampler, unlimited compute, 1,200 s wall clock on GPU 0 (RTX 3090, power-capped), seed 101: 23 rounds (22 full,
one at 0.87) for 2.9989e15 FLOPs = 1.3039e14 per round, x 30 = 3.9117e15, rounded to 2 significant figures.
Search runs never pass `--flops` or `--budget`; a champion run ordered by the monitor passes a multiple of
`DEFAULT_FLOPS` (and, if it needs it, a longer `--budget`). A run whose `flops_budget` differs from
`DEFAULT_FLOPS` is not promoted.

**What is charged** (`frozen/meter.py`). Every `nn.Linear` call anywhere in the process, `2 x in_features x
out_features` FLOPs per input row (a row is one token position: `input.numel() / in_features`), padding rows
included. A call made while autograd records it (grad mode on and the output requires grad) is charged 3x, for
the backward pass; the backward, the optimizer step and gradient clipping are then free. This covers the policy
and any critic, value head, reference or reward model built from `nn.Linear`, sampling (prefill and every decode
step), training and in-run evaluation (`paths['heldout_greedy']` runs the model, so it is charged). CUDA graphs
are metered: while the current stream is capturing a graph, the Linear calls traced into it are tallied on the
graph (capture itself is free: nothing runs), and every `replay()` charges the tally; eager warm-up steps before
a capture are charged like any other call. For the standard GPT one row costs 6,342,144 FLOPs
(`meter.policy_flops_per_token(99)`; x3 with autograd).

**What is free.** Attention score / value products (`scaled_dot_product_attention`), embeddings, layer norms,
activations, softmax and sampling, the optimizer, the verifier, and all CPU work (verification, data building,
weight averaging on the CPU, checkpointing). Everything after `rl.run` returns (final held-out check, scoring)
is free too.

**What is forbidden** (it bypasses the meter; a node that does it is void, and so is everything built on it):
- `torch.compile`, `torch.jit.script`, `torch.jit.trace` and `torch.cuda.make_graphed_callables` (they raise while
  the meter is installed), and any other compiler or export path (`torch._dynamo`, `torch.export`, Triton or
  custom kernels that do the model's matrix products);
- `F.linear`, `matmul` / `@`, `einsum`, `bmm`, `addmm` or any other product with model weights outside an
  `nn.Linear` module call, calling `Linear.forward` directly, and modules that use their weights that way
  (`nn.MultiheadAttention`, `nn.Transformer*`, `nn.RNN` / `LSTM` / `GRU`, `nn.Bilinear`, convolutions): build
  extra networks from `nn.Linear`;
- `torch.func` transforms (`vmap`, `grad`, `jacrev`, ...) over model calls, nested / packed tensors fed to a
  Linear, backward passes that reuse one recorded forward more than once (`retain_graph=True` with several
  backwards), and anything that reads or changes the meter object, its hooks or its patches;
- capturing CUDA graphs other than with `torch.cuda.CUDAGraph` / `torch.cuda.graph` (the meter raises), and
  launching a captured graph other than through `CUDAGraph.replay` (e.g. `raw_cuda_graph_exec`);
- extra processes or threads that run model compute (in `rl.run`, at import time, or after `rl.run` returns),
  and any second GPU. Only one process on one GPU runs the model.

**Using the budget.** Poll `paths['budget']` (`remaining()`, `frac()`), measure your own FLOP rates in-run (the
meter's `used()` before and after a phase) and return before the limit. With an infinite `limit` (an unmetered
calibration run) stop on the wall clock. The meter cannot tell useful rows from padding: padded fine-tune batches
are 58-61% padding (n092; the seed packs its batches, ~1.4%), and bucket rounding in a graph sampler costs a few
percent. Those are real compute levers; wall-clock speed alone is not.

**Validation** (`ab3/calib/validate_meter.py`, results in `ab3/calib/validate_meter_*.json`). On a fixed
checkpoint and 1,536 prompts, the meter's charge equals `6,342,144 x` the rows each forward / replay processes,
counted independently, to the last digit, for eager and CUDA-graph samplers alike (n092's line-verified graph
sampler and the seed's plain graph sampler). In the calibration pair (n092 with and without its graph sampler, 1,200
s each): 1.628e9 vs 1.566e9 FLOPs per sample (+4.0%: the graph sampler's batch buckets, its warm-up steps and its
one-time self-check), 8.17e11 vs 8.04e11 per fine-tune step (+1.6%: later rounds train on longer proofs), and
per useful token (live rows only) 1.229 vs 1.185 x the per-row cost in the controlled run.

## Seed (`ab3/seed/rl.py`)

Plain expert iteration (run 2's seed, run 1's winner n025: 400-target window, k = 32 at temperature 1.0, 135
fine-tune steps with a fresh AdamW per round, 20,000 cap-6 retention records, 6,000-target length-stratified
subset, hardest-first refills) plus three parts from run 2's n092 line and one efficiency part:
- a CUDA-graph decode step for plain sampling (`graph_generate`; same output distribution as `fast_generate`,
  about 2x faster, falls back to `fast_generate` on any error);
- the capped line-index shift in fine-tuning batches (`index_room` 24);
- sequence packing of the fine-tuning batches (`ft_pack`, on): each step's 128 random records, unchanged, are packed
  whole (first-fit-decreasing, never split or cut, row length between the longest record and 1,024) into rows with
  a block-diagonal causal mask and positions restarting per record. Logits, loss and gradients equal the padded
  batch's (fp32: logits within 2e-4, gradients within 1e-4 relative; bf16: within bf16-vs-fp32 noise). Padding
  falls from 58% of the charged fine-tune rows to about 1.4%, so a fine-tune step costs 2.3x less and the budget
  holds about 63 seed rounds instead of 32-34. `ft_pack=False` restores padded batches;
- stopping driven by the meter: FLOP rates are measured in-run, a round that would not fit is shrunk
  (`min_round_frac` 0.3), sampling stops before a chunk that would eat the round's fine-tune share, fine-tuning
  stops before a step that would not fit, and `margin_frac` (0.3%) of the budget is never planned. The wall
  clock is a second limit.
It also has `max_len_train` (drop targets above a true length before subsetting), `max_proof_lines_train` (leave
found proofs with more written lines out of the fine-tune mix; they still count as solved), and per-round counts
of the sampling window by true length in `metrics.jsonl` (`window_by_len`, `samples_by_len`, `accepted_by_len`,
`solved_by_len_round`, `solved_cum_all_by_len`) and the fine-tune padding fraction (`ft_pad_frac`). Run 2's other levers are in
`ab3/seed/lib_run2.py`, each behind a `CONFIG` switch that is off in the seed (`guided`, `loss='gem'`,
`len_weights`, `schema_cover`, `knapsack`, `temp_schedule`, `avg_last`). Seed results are in "Build notes".

## Scoring

**Client** (`frozen/score.py`, run by the harness or by an agent):

```
python frozen/score.py --ckpt <abs path under ab3>/final.pt [--set dev|frontier] [--k 32|64] [--seed 0] [--tag text] [--smoke]
```

It writes `{id, set, ckpt, k, seed, gpu, cwd, tag, smoke, t}` to `ab3/score_requests/<id>.json` (`ckpt` is the
host-absolute path, `gpu` the caller's `CUDA_VISIBLE_DEVICES`, the host GPU it holds), polls
`ab3/score_results/<id>.json` every 3 s for up to 55 minutes, and prints one line:

```
dev pass@32: solved T/N | L8 a/n8 L9 b/n9 L10 c/n10 L11 d/n11 | secs S
```

It then writes `score.json` (dev) or `score_frontier.json` next to the checkpoint, with every field of the result
(including `pass1`, `pass1_by_len`, `pass4`, `pass4_by_len`), and appends the result to `ab3/logs/scores.jsonl`.
`--k` defaults to the official value (dev 32, frontier 64). Only those, with seed 0 and not smoke, are official
(`official: true`, decided by `score_host.py`). Other `k` (1-256) and seeds are allowed for exploration, but
every request costs GPU time on the requester's GPU (scoring is outside the compute budget; do not use extra
scoring requests as a training signal).

**Daemon** (`frozen/score_daemon.py`, run by the monitor):

```
python frozen/score_daemon.py --arm ab3:/scratch/cowifornia/nd-rl/ab3:0,1,2,3,4
```

- It serves only `set` in {`dev`, `frontier`}; every other set is refused with an error result.
- The checkpoint must resolve, after following symlinks, to a file under the arm directory.
- The request's `gpu` is a host GPU index, which must be in the arm's list. The daemon runs `score_host.py` on
  that GPU, at most once per GPU at a time and at most 5 at a time for the arm.
- Results carry aggregate and per-length numbers only (`solved`, `by_len`, `n_by_len`, `n`, `pass1`,
  `pass1_by_len`, `pass4`, `pass4_by_len`, `k`, `seed`, `secs`, `samples_per_sec`, `ckpt_sha256`, `official`),
  never per-theorem data. Each result is appended to `ab3/logs/daemon_scores.jsonl`.

**Host scorer** (`frozen/score_host.py`, run by the daemon and by the monitor):

```
python frozen/score_host.py --ckpt <path> --set dev|frontier|sealed_L12-14|sealed_L15-16|sealed_L17-18|matched_L6-14|textbook
                            [--k K] [--seed 0] [--smoke] [--full] [--out f.json] [--per-theorem rows.jsonl]
```

- For each theorem it samples k proofs (1 <= k <= 256; default 32 for dev, 64 otherwise) at temperature 0.8 with
  `lib.sample.generate` (batch 384, fixed seed, `max_new` per set as in the Sets table, `MAX_NEW_BY_SET`), and
  verifies each with `nd_verify.verify_text(prompt + ' ' + proof)`. A theorem counts as solved if any sample
  verifies. `frontier` is the fixed 1,000-theorem subset unless `--full`.
- **pass@1 / pass@4** (run 3): for a theorem with c of its k samples verified, pass@1 = c / k and pass@4 =
  1 - C(k - c, 4) / C(k, 4) (the unbiased estimator; null for k < 4). `pass1` / `pass4` are the means over the
  set's theorems and `pass1_by_len` / `pass4_by_len` the means over each length's theorems. They come from the
  same samples as the pass@k count (no extra sampling). Aggregates only.
- It loads checkpoints with `torch.load(weights_only=True)`.
- `--per-theorem` is for the monitor only. It may not write under `ab2/` or `ab3/` except `ab2/sealed/`,
  `ab2/monitor/`, `ab3/monitor/`, `ab3/sealed/` and `ab3/lengthexp/`, and not into run 1's arms.
- Env overrides for tests: `AB2_POOL_DIR` (dev and frontier), `AB2_SEALED_DIR`, `AB3_SEALED_DIR`, `AB2_TEXTBOOK`.

## GPU-lock protocol

- `python frozen/gpulock.py acquire [--gpus 0,1,2,3,4] [--wait SECS] [--run NAME] [--owner PID]` takes an
  exclusive `flock` on `ab2/locks/gpu<i>` (the host-wide lock directory, shared with run 2) for the first free GPU
  and prints `i`. Waiting callers take a FIFO ticket (`locks/queue/`); only the oldest live ticket may take a free
  GPU, and tickets of dead processes are discarded.
  - The flock is held by a small detached holder process (its own session, no stdio). The holder writes its pid
    to `ab2/locks/gpu<i>.pid` and one JSON line into the lock file:
    `{gpu, worker: $AB_WORKER, run, holder_pid, owner_pid, since, host}`.
  - With `--owner PID`, the holder exits, freeing the GPU, within 2 s of that process dying. The harness passes its
    own pid here. Without `--owner`, the lock is held until `release`.
  - `--wait` retries every 5 s. The default of 0 means a single pass: exit status 1 if no GPU is free.
- `python frozen/gpulock.py release <i>` stops that GPU's holder, after checking that the pid really is a gpulock
  holder, and waits until the lock is free.
- `python frozen/gpulock.py status [--json]` lists every GPU as free or as held.
- **Stale locks.** The kernel drops an flock when its holder dies, so a lock whose holder pid is dead is free.
- **The rule for agents.** Run every GPU job through `harness.py --gpu auto`. Do not set `CUDA_VISIBLE_DEVICES`
  by hand, do not use `--gpu env`, and do not run GPU code outside the harness. The one exception: if you must,
  acquire a lock with `gpulock.py acquire` first and release it afterwards. Never release a GPU you do not hold.
- Ad hoc `score.py` calls on your own checkpoints run on the GPU in your `CUDA_VISIBLE_DEVICES`. Make them only
  while you hold that GPU's lock.

## Metrics schema (`frozen/metrics.py`)

Unchanged from runs 1 and 2. `Metrics(out_dir, run_name, arm, wandb_project='nd-rl-hillclimb', t0=None)`. Each
`metrics.log(**fields)` call appends one JSON object to `runs/<name>/metrics.jsonl`. Every record carries `t`,
`elapsed`, `step` (monotone) and `phase` in {`sample`, `train`, `eval`, `summary`}. The seed adds compute fields
to its records (`sample_flops`, `flops_per_sample`, `flops_used`, `frac_used`; per round in
`rl_stats.per_round`). W&B mirroring, if wanted, is done host-side by the monitor.

## Rules for agents

1. Edit only your workspace's `rl.py`, and `ab3/controller.py` under the GEAR-Evolve protocol
   (`ab3/GEAR_README.md`). Never edit anything under `ab3/frozen/` or `ab3/seed/`. The harness records hashes,
   and a run with a modified frozen file is void. A controller step may not remove or weaken the two-seed rule,
   the held-out floor, `--source`, the worker cap, the compute-budget check or the auditor commands.
2. Never read, list, copy or search for anything under `ab3/sealed/`, `ab2/sealed/`, `ab/sealed/` or `ab2/data/`
   (the dev and frontier theorems), or the monitor's directories (`ab3/monitor/`, `ab3/data_build/`,
   `ab3/lengthexp/`, `ab3/calib/` outputs are the monitor's). The RL pool you may use is
   `frozen/data/targets_L6-11.jsonl`. Also stay out of run 1's and run 2's directories except the read-only
   reference files named in your brief, and out of other workers' directories. From other agents' work you may
   read only the shared files: the controller's `frontier`, `history` and `next` output, `nodes/`,
   `reflections.md`, `decisions.md`, `ideas.md`, `digest.md`, `champion.md` and `audit/`. Do not try to reconstruct
   the dev, frontier or sealed sets.
3. **Training data.** Only these are allowed:
   - (a) proofs your own model sampled during this run that `nd_verify` accepts for the prompted sequent, where
     the prompts come from `targets_L6-11.jsonl`;
   - (b) verified by-products of your own samples: a valid proof of a different conclusion from the same premises
     (hindsight relabelling);
   - (c) records from `train_cap6.jsonl`.

   Nothing else is allowed: no hand-written proofs, no proofs written by an LLM, no proofs of length > 6 from any
   file, no proofs from earlier runs or other workers' runs, and no prompts from the dev, frontier or sealed sets.
   `heldout_cap6.jsonl` is for tracking only.
4. **Compute.** All model compute of an experiment happens inside its `rl.run`, in the harness process, on its
   one GPU, through `nn.Linear` modules under the meter (section "Compute budget"). The budget is
   `DEFAULT_FLOPS`: never pass `--flops` or `--budget`, except for a champion run ordered by the monitor.
5. Only experiments launched through `harness.py --gpu auto` count. One experiment per GPU lock. At most five
   experiments run at once; the rest wait in the FIFO queue.
6. Only scorer numbers count. Report the controller's score with the pass@1 / pass@4 estimates and the pass@k
   counts as the harness prints them. Single-seed differences below the controller's tie margin are noise; the
   BEST role needs two seeds (`ab3/GEAR_README.md`).
7. Stop launching experiments at the deadline in your brief. Finish by naming the BEST node and its checkpoint
   paths, and write the report your brief asks for.

## GEAR protocol

Run 3's worker, auditor and controller protocol is `ab3/GEAR_README.md` (controller v18): workers loop `next` ->
copy the parent -> one coherent edit -> harness -> `record --source` -> reflection and `ideas.md` row; a child
that beats BEST by the improvement margin is `PENDING_BEST` until the auditor's second seed confirms it; a node
with `heldout_greedy < 0.85` is never promoted; a run with no pass@1 / pass@4 fields (`UNSCORED`), with a
`flops_budget` other than `DEFAULT_FLOPS`, or from `--smoke` (`OFF_BUDGET`) is never promoted. Every node cites
a source (`record --source <URL>` or `"own idea"`); `ideas.md` is a Markdown table with one row per idea tested
(`| date | worker | idea | source URL | node ids that tested it | verdict |`), appended, never edited by another
worker. The seed `rl.py` opens with a `KNOWN_FAILURES` block summarising runs 1 and 2; check it and `ideas.md`
before proposing an idea.

## Useful facts

- One charged row of the standard GPT costs 6,342,144 FLOPs (x3 with autograd). The budget is about 6.1e8 rows.
- In the n092 calibration 84% of the compute went to fine-tuning (135 steps x 128 records per round; 8.2e11 per
  step, 61% of it on padding) and 16% to sampling (1.6e9 per line-verified sample).
- The seed's rounds (packed) cost 5.1e13 FLOPs early to 6.8e13 late (sampling about 1.5e13 for 12,800 samples;
  fine-tuning 3.7-5.1e13, 2.7-3.8e11 per step, growing as the RL records get longer) and take 25-38 s on a
  power-capped 3090: the budget holds 63 of them in 30-37 minutes (26% of the compute sampling, 74% fine-tuning).
- Run-to-run noise: in run 2 second seeds averaged 154 score points below first seeds and same-code same-seed
  pairs differed by 34-234 points, mostly from card speed (round count). Compute metering removes that source;
  the seed's two seeds (below) differ by 37 score points (1,081 vs 1,044), mostly in frontier pass@1 at L12-13.

## Build notes (2026-09-24)

- Calibration: `ab3/calib/runs/cal_graph` (n092 `rl.py`, graph sampler, GPU 0) and `ab3/calib/runs/cal_nograph`
  (the same with `lv_graph=False`, GPU 1), unlimited compute, 1,200 s, seed 101, `--no-score`. Graph: 23 rounds,
  2.9989e15 FLOPs (sampling 4.76e14, training 2.52e15), 3,087 optimizer steps. No graph: 19 rounds, 2.3542e15
  (3.67e14, 1.99e15), 2,472 steps.
- Seed runs, default budget 3.9e15, run 2's pools (which run 3 keeps). Final seed (`ab3/seed/runs/seed_s101`,
  `seed_s102`; `rl.py` sha256 be380193a3fc..., packed fine-tuning) and, for comparison, the first version with padded
  batches (`ab3/seed/runs/v1_unpacked/`, `rl.py` 0d03d9be70d0...); harness ec6d1b0413c7..., meter a6350b9e0fa3...:

  | seed rl.py | seed | GPU | rounds | compute used | rl secs | fine-tune padding | dev pass@32 (L8 / 9 / 10 / 11) | frontier pass@64 (L12 / 13 / 14) | dev pass@1 / pass@4 | frontier pass@1 / pass@4 | score | heldout_greedy |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | final (packed) | 101 | 0 | 63 | 3.871e15 (99.25%) | 2,172 | 1.4% | 292 (88 / 81 / 60 / 63) | 147 (113 / 34 / 0) | 0.1468 / 0.1933 | 0.06769 / 0.10053 | 1,081.0 | 0.958 |
  | final (packed) | 102 | 1 | 63 | 3.886e15 (99.63%) | 1,829 | 1.4% | 295 (89 / 85 / 65 / 56) | 140 (110 / 30 / 0) | 0.1497 / 0.1997 | 0.05839 / 0.09785 | 1,044.2 | 0.959 |
  | v1 (padded) | 101 | 0 | 34 | 3.886e15 (99.65%) | 1,374 | ~58% | 219 (66 / 65 / 39 / 49) | 37 (30 / 7 / 0) | 0.0938 / 0.1386 | 0.00677 / 0.01400 | 361.9 | 0.942 |
  | v1 (padded) | 102 | 1 | 32 | 3.861e15 (98.99%) | 1,096 | ~58% | 229 (72 / 67 / 42 / 48) | 45 (32 / 13 / 0) | 0.1090 / 0.1499 | 0.00962 / 0.02074 | 432.1 | 0.944 |

  Packing buys twice the rounds (8,505 optimizer steps instead of about 4,400) for the same compute. At equal
  optimizer steps (round 30, `ab3/seed/runs/forget_check/`) packed and padded runs match within seed noise at every
  length: held-out cap-6 greedy L2-6, dev pass@1 L8-11 and the training targets' solve rate L6-11.
  score = (dev pass@1 + pass@4) x 1,200 + 4 x (frontier pass@1 + pass@4) x 1,000, in expected theorems. Neither run
  raised BudgetExhausted; both stopped on the compute planner. The exhaustion test (`seed/tests/rl_exhaust.py`, a
  loop that samples forever, `--smoke`) was scored with `budget_exhausted: true` at 1.0000117e12 of 1e12 FLOPs
  (the call that crosses the limit completes; every later one raises).
- `SHA256SUMS` lists `sha256sum` of every file under `frozen/` except the `SHA256SUMS` files themselves (paths
  `./...`, sorted as `find | sort` does in the en_US locale, `__pycache__` included, as in run 2). It was
  regenerated after the last frozen edit of this build.
