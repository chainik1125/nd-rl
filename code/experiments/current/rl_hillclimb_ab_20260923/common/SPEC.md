# RL hill-climb A/B: shared harness specification

Written 2026-09-23 by the monitor session (agent:claude, Fable 5.1) for Leon's workstream 4
("custom RL pipeline") of the nd-rl project. This file is the contract between the frozen harness,
the editable RL file, the sealed scorer and the agents. Everything under `frozen/` in an arm
directory is a verbatim copy of `common/` and must not be edited by any agent.

## The task

Start from the fixed pretrained checkpoint `pretrain.pt` (4 layers, d=256, 8 heads, absolute
tokenizer, 3.2M params; pretrained on cap-6 natural-deduction proofs, no RL). Maximise the
**dev score**: the number of theorems in a sealed dev set of length 7-9 theorems (n = 1610;
300 / 300 / 1010 at true length 7 / 8 / 9) for which at least one of 32 sampled proofs at
temperature 0.8 is accepted by `nd_verify` as a proof of the prompted sequent. Only the
frozen scorer computes it. The agents never see the dev theorems, only the aggregate counts.

After the search window closes, the monitor evaluates each arm's best checkpoint on sealed
length-10 and longer sets and on a textbook set. No agent sees those either.

## Directory layout of an arm

```
<arm>/
  frozen/            verbatim copy of common/ (harness.py, score.py, metrics.py, lib/, data/, pretrain.pt, ref/)
  <workspace>/       one per worker: rl.py (editable), runs/<name>/ (outputs)
```

Frozen files, by role:

| file | role |
|---|---|
| `frozen/harness.py` | one experiment: load checkpoint, call `rl.run`, enforce the clock, save `final.pt`, score it, write `summary.json` |
| `frozen/score.py` | sealed scorer; prints aggregate counts only; appends to the shared score log |
| `frozen/metrics.py` | per-step metrics logger (JSONL + Weights & Biases) |
| `frozen/lib/` | model, tokenizer, sampler, supervised-loss helpers, verifier, prune, normalize, eval helpers, class keys |
| `frozen/data/targets_L7-9.jsonl` | 3,804 RL targets, true length 7 / 8 / 9 = 1,800 / 1,000 / 1,004 (no proofs) |
| `frozen/data/targets_extra_L7-9.jsonl` | 15,410 further targets, 7 / 8 / 9 = 11,046 / 4,301 / 63 (no proofs) |
| `frozen/data/train_cap6.jsonl` | 154,990 cap-6 supervised records with proofs (the pretraining data), usable for retention |
| `frozen/data/heldout_cap6.jsonl` | 5,000 cap-6 records with proofs, for in-distribution tracking only |
| `frozen/pretrain.pt` | the starting checkpoint (`{'state', 'tok_mode': 'abs', ...}`, no `cfg` key: construct `GPT(tok.vocab_size, 4, 256, 8)`) |
| `frozen/ref/` | reference material: the fork's `expert_iter.py` and `grpo.py`, the GEAR paper, the orchestration proposal, the logic spec |

All four pools and the sealed sets are pairwise disjoint by atom-renaming class (`lib/keys.py`).

## Harness contract (`frozen/harness.py`)

```
python frozen/harness.py --out runs/<name> [--seed 0] [--budget 600] [--rl rl.py] [--no-score]
```

1. Seeds torch and Python; loads `frozen/pretrain.pt` into `lib.model.GPT(vocab, 4, 256, 8)` with the
   `abs` tokenizer; copies `rl.py` into `runs/<name>/source/` and records SHA-256 of `rl.py`,
   `harness.py`, `score.py`, `metrics.py` and every file in `lib/` and `data/` into `summary.json`.
2. Imports the RL module (`rl.py` from the current directory unless `--rl` says otherwise) and calls

   ```python
   model = rl.run(model, tok, targets, budget_secs, device, seed, log, metrics, paths)
   ```

   | argument | meaning |
   |---|---|
   | `model` | the loaded GPT, on `device`, train mode allowed |
   | `tok` | `lib.tokenizer.Tokenizer('abs')` |
   | `targets` | list of dicts from `targets_L7-9.jsonl` (`name, thm, key, prompt, n_lines, L_true, source, schema, n_prem`) |
   | `budget_secs` | wall clock for the whole call (default 600) |
   | `device` | `'cuda'` |
   | `seed` | int |
   | `log(msg)` | timestamped print |
   | `metrics` | a `metrics.Metrics` instance (see schema below) |
   | `paths` | dict with `targets_extra`, `train_cap6`, `heldout_cap6`, `out_dir`; the RL code may read those files |

   `rl.run` must return the model to be scored (same architecture; the scorer loads the state dict into
   `GPT(vocab, 4, 256, 8)`). It may save intermediate checkpoints under `paths['out_dir']` with
   `harness.save_checkpoint(model, tok, path)`.
3. The clock: `budget_secs` from the call. A SIGALRM fires at `budget_secs + 45` and the run is
   recorded as `status: timeout` with no score. Returning after `budget_secs + 15` is also a timeout.
   Well-behaved RL code checks `time.time() - t0` and stops with margin.
4. After return: `final.pt` is saved (`{'cfg', 'state', 'tok_mode', 'stage': 'final', 'seed', shas}`),
   then unless `--no-score` the harness runs `frozen/score.py --ckpt runs/<name>/final.pt` as a
   subprocess on the same GPU and copies its JSON into `summary.json` (`metric` = total solved,
   `by_len` = per-length solved counts). The harness never reads the sealed files itself.
5. `summary.json`: `status` (`ok` / `timeout` / `crash`), `metric`, `by_len`, `rl_secs`, `score_secs`,
   `rl_stats` (whatever `rl.run` put in `metrics.summary`), `shas`, `seed`, `budget`, `gpu`, `started_utc`.
6. `--smoke`: tiny everything (8 targets, k=2, 20-second budget, dev subset of 16), for plumbing checks.

## Sealed scorer (`frozen/score.py`)

```
python frozen/score.py --ckpt runs/<name>/final.pt [--k 32] [--seed 0] [--tag free-text]
```

- Reads `/scratch/cowifornia/nd-rl/ab/sealed/dev_L7-9.jsonl` (never copied into an arm).
- Samples `k` proofs per theorem at temperature 0.8, `max_new` 512, batch 384, with
  `lib.sample.generate` and the fixed seed; verifies with `lib.nd_verify.verify_text(prompt + ' ' + proof)`;
  a theorem is solved if any sample verifies.
- Prints exactly one line: `dev pass@32: solved T/1610 | L7 a/300 L8 b/300 L9 c/1010 | secs S`
  and writes `runs/<name>/score.json` with the same numbers plus `ckpt_sha256`, `k`, `seed`, `secs`,
  `samples_per_sec`. It writes no per-theorem output.
- Appends one JSON line to `/scratch/cowifornia/nd-rl/ab/logs/scores.jsonl` with the arm (derived from the
  cwd), checkpoint path and hash, the numbers, and the harness/rl/lib hashes if `summary.json` exists.
- Any `k` other than 32 is allowed for exploration, but only `k = 32, seed = 0` counts as the dev score.

## Metrics schema (`frozen/metrics.py`)

`Metrics(out_dir, run_name, arm, wandb_project='nd-rl-hillclimb')`. Each `metrics.log(**fields)` call
appends one JSON object to `runs/<name>/metrics.jsonl` and, if W&B is reachable, `wandb.log`s it.
Every record carries `t` (unix seconds), `elapsed` (seconds since the harness started), `step`
(monotone int supplied by the RL code) and `phase` in `{sample, train, eval, summary}`.

| phase | fields |
|---|---|
| `sample` | `round`, `n_prompts`, `k`, `temperature`, `samples`, `accept_rate` (verified samples / samples), `qed_rate` (samples that emitted QED), `targets_solved_round`, `targets_solved_cum`, `new_proofs` (distinct, start-index normalised), `solved_by_len` (`{"7": n, "8": n, "9": n}`), `mean_written_len`, `mean_pruned_len`, `samples_per_sec`, `fail_reasons` (top 5, `[[reason, count], ...]`) |
| `train` | `loss`, `lr`, `grad_norm`, `n_records`, `tokens`, `batch`, optional `entropy` (mean token entropy on the batch), optional `kl_to_init` |
| `eval` | `heldout_greedy` (fraction of a fixed 1,000-record slice of `heldout_cap6.jsonl` solved greedily), optional `targets_greedy` |
| `summary` | free-form totals (`rounds`, `total_samples`, `total_train_steps`, `targets_solved_cum`, `wall_secs`) |

`metrics.summary` is a dict the harness copies into `summary.json`. W&B: project `nd-rl-hillclimb`,
entity from the logged-in account, run name `<arm>/<run_name>`, group `<arm>`, `config` = the
summary's hashes and seed. W&B failures are logged and ignored; the JSONL is the record.

## Rules for agents (both arms)

1. Edit only `rl.py` (and, in the GEAR arm, `controller.py` under the GEAR-Evolve protocol).
   Never edit anything under `frozen/`. The harness records hashes; a run with a modified frozen file
   is void.
2. Never read, list, copy or search for anything under `/scratch/cowifornia/nd-rl/ab/sealed/` or the
   shared logs of another arm. Do not try to reconstruct the dev set. Do not read directories of
   other arms. Your arm directory and `frozen/` are your world.
3. Training data allowed: proofs your own model sampled during this run that `nd_verify` accepts for the
   prompted sequent (targets or extra targets), verified by-products of your own samples (a valid
   proof of a different conclusion from the same premises, i.e. hindsight relabelling), and records
   from `train_cap6.jsonl`. Nothing else: no hand-written proofs, no proofs written by an LLM, no
   proofs of length > 6 from any file, no proofs imported from earlier runs of another arm.
   `heldout_cap6.jsonl` is for tracking only.
4. One experiment per GPU at a time; the GPU is set with `CUDA_VISIBLE_DEVICES` by the worker's brief.
   Sampling batch 384 fits a 3090 alongside training; batch 1024 does not.
5. The per-experiment RL budget is 600 s. Do not change it. Scoring adds about 2.5 minutes.
6. Only `score.py` numbers count. Report the dev score as the scorer prints it. Wilson intervals on
   1,610 theorems are about ±1.5 points at 10 %, and run-to-run training noise is larger than that;
   treat differences under ~15 theorems as noise unless repeated.
7. Stop launching experiments at the deadline in your brief. Finish by naming the best checkpoint
   path and writing the report your brief asks for.

## Useful facts

- Sampling throughput on a 3090 at batch 384 is about 350 samples/s at `max_new` 512 for length 7-14
  prompts; length 7-9 prompts are faster. 1,500 targets × 32 samples take about 2.5 minutes; a
  600-step fine-tune at batch 128 takes about a minute.
- The starting checkpoint solves few dev theorems: at pass@64 it solved 9 of the 1,610 (all at
  length 7). Plain expert iteration from a comparable checkpoint reached roughly 47 % / 46 % / 30 %
  of length 7 / 8 / 9 theorems after 8 rounds × 32 samples over 4,495 targets (about 70 minutes on a
  3090), so the frontier moves; the first rounds are sparse.
- The fork's GRPO (`ref/grpo.py`: on-policy, group-mean baseline, no KL) acquired faster than expert
  iteration on one target family but dropped in-distribution greedy accuracy from 0.87 to 0.56 without
  retained cap-6 data.
- Start-index normalisation: the `abs` tokenizer writes the same proof from N1 or N12; distinct-proof
  counts must renumber first (`lib/normalize.py`'s `norm`).

## Addendum (2026-09-23 00:30 UTC): containers and the scoring service

Each arm's agents run inside a Docker container with GPU access and a single bind mount: the arm
directory at `/work`. The sealed sets, the other arm, the monitor's files and the host home are not
mounted. Inside the container the arm directory is `/work` and `frozen/` is `/work/frozen/`.

Consequences for the files above:

- `frozen/score.py` is a **client**. It writes a request `{id, ckpt, k, seed, gpu, cwd, tag}` to
  `/work/score_requests/<id>.json`, where `ckpt` is the container path of the checkpoint and `gpu` is the
  `CUDA_VISIBLE_DEVICES` index the caller is using, then polls `/work/score_results/<id>.json` (every 3 s,
  timeout 30 min), prints the one aggregate line, writes `score.json` next to the checkpoint and
  appends to `/work/logs/scores.jsonl`. It contains no sealed path and no verifier call.
- The **host-side scoring daemon** (`common/score_daemon.py`, run by the monitor, never inside a
  container) watches each arm's `score_requests/`, maps `/work/...` to the host path, maps the container
  GPU index to the host GPU (arm `gear` → host GPUs 0,1; arm `orch` → host GPUs 2,3), runs
  `common/score_host.py` (the real scorer: sealed dev set, k, seed, temperature 0.8, `max_new` 512,
  batch 384, `lib.sample.generate`, `nd_verify`) as a subprocess on that GPU, and writes the result
  JSON to `score_results/<id>.json`. Results carry `ckpt_sha256`, `k`, `seed`, `solved`, `by_len`, `n`,
  `secs`, `samples_per_sec`. The daemon also appends to `/scratch/cowifornia/nd-rl/ab/logs/scores.jsonl`
  with the arm name.
- `frozen/metrics.py` writes JSONL only. A host-side `common/wandb_sync.py` (run by the monitor)
  tails every `runs/*/metrics.jsonl` under both arms and mirrors new records to Weights & Biases
  (project `nd-rl-hillclimb`, one W&B run per experiment, group = arm). No W&B key enters a container.
- The container image has Python 3 with PyTorch (CUDA), tmux, git and the Claude Code CLI. Agents run
  as an unprivileged user whose uid matches the host owner of `/work`.

## Implementation notes (2026-09-23, harness build; where the code deviates from or pins down the text above)

- **Held-out slice.** `heldout_cap6.jsonl` is length-ordered (1,000 records each at 2 / 3 / 4 / 5 / 6 lines), so
  "the first 1,000 records" would be all 2-line proofs. The fixed slice is every 5th record
  (`recs[::5][:1000]`, 200 per length). `harness.heldout_greedy(model, tok, n=1000)` computes it (greedy,
  batch 384, `max_new` 512, `nd_verify`) and is passed to `rl.run` as `paths['heldout_greedy']` so the RL
  file's per-round number and the harness's final number are the same metric. The harness always computes
  the final one after `rl.run` returns (summary field `heldout_greedy`; also logged as an `eval` record
  with `source: harness`). Smoke uses the first 64 records of the slice.
- **`paths`** also carries `save_checkpoint` (= `harness.save_checkpoint(model, tok, path)`; `import harness`
  from rl.py works too), `heldout_greedy` and `smoke` (bool; the seed rl.py shrinks its own knobs when set,
  e.g. k = 2). In `--smoke` the harness fixes 8 targets and a 20 s budget; sampling k is rl.py's choice.
  The smoke score request is `--k 32 --smoke` (16 theorems).
- **Checkpoints** (`final.pt`, and anything saved through `save_checkpoint`): `{'cfg', 'state', 'tok_mode',
  'extra': {}, 'stage', 'seed', 'shas'}`, readable by `lib.model.load_ckpt`. No `final.pt` is written on
  `timeout` or `crash`.
- **Hashes** (`summary.json` `shas`, full SHA-256): `rl.py`, `harness.py`, `score.py`, `metrics.py`,
  `pretrain.pt`, every file under `lib/` (no `__pycache__`), every file under `data/`. Data hashes are read
  from `data/SHA256SUMS` (sha256sum format, written once) when its file list matches `data/`.
- **Statuses.** `ok` / `timeout` / `crash` as specified; a failed score request (daemon error or 30-min
  timeout) is a `crash` with the client's message in `reason`. Before scoring the harness writes a
  provisional `summary.json` with `status: scoring` so the client can copy the hashes into
  `logs/scores.jsonl`. Extra summary fields: `scored`, `score` (the full score.json), `score_line`,
  `total_secs`, `run_name`, `arm`, `smoke`, `reason`.
- **Score client / daemon interface.** Request: `{id, ckpt, k, seed, gpu, cwd, tag, smoke, t}`. The client
  accepts a result `{status: 'ok' | 'error', error?, solved, by_len, n, n_by_len?, secs, samples_per_sec,
  ckpt_sha256, k, seed}` where `by_len` values are ints (denominators from `n_by_len`, else 300 / 300 / 1010)
  or `{solved, n}` dicts; for smoke results the daemon must supply the denominators (`n_by_len` or dicts).
  `score.json` normalises `by_len` to `{"7": int, "8": int, "9": int}` and adds `metric`, `n_by_len`,
  `official` (k = 32, seed = 0, not smoke). A hidden `--local-stub` flag writes a zero result locally (tests
  only); the harness adds it when `AB_SCORE_STUB=1`.
- **Metrics.** `Metrics(out_dir, run_name, arm, wandb_project=..., t0=None)`; the harness passes its start time as
  `t0`. `log()` raises on a missing/unknown `phase`; `step` defaults to an internal counter. The seed rl.py's
  `sample` records use the global train-step count as `step` (non-decreasing), make `solved_by_len`
  cumulative, and add `round`, `solved_by_len_round`, `sample_secs`, `judge_secs`; `train` records add `round`.
