# RL hill-climb, run 2 ("ab2"): harness specification

Written 2026-09-24 by the harness-build session (agent:claude) for Leon's workstream 4 ("custom
RL pipeline") of the nd-rl project. It replaces run 1's contract
(`/scratch/cowifornia/nd-rl/ab/common/SPEC.md`). This file is the contract between the frozen
harness, the editable RL file, the scorer, the GEAR controller and the agents. Everything under
`frozen/` in the arm directory is a verbatim copy of `common/`, and no agent may edit it.

Run 2 has one arm, GEAR-Evolve. Seven hill-climbing workers (`w1` ... `w7`) and one auditor
(`auditor`) share five RTX 3090s (host GPUs 0-4). The agents are in-process sessions on the host
and use host paths. There are no containers.

## The task

Start from the fixed pretrained checkpoint `frozen/pretrain.pt`. It is run 1's starting point:
4 layers, d=256, 8 heads, absolute tokenizer, 3.2M parameters, pretrained on cap-6
natural-deduction proofs, no RL. Maximise the **dev score**: the number of theorems in the dev
set `dev_L8-11` for which at least one of 32 sampled proofs, at temperature 0.8, is accepted by
`nd_verify` as a proof of the prompted sequent. Only the scorer computes it. Every run also
reports the **frontier count**: the same measure at 64 samples on `frontier_L12-14`, which is longer than
anything in the RL pool. The frontier count is shown next to the dev score but is not the metric.

After the search window closes, the monitor evaluates the BEST checkpoint(s) on the sealed sets
(`sealed_L12-14`, `sealed_L15-16`, textbook) at up to k = 256. A 7,200 s champion run of the final
BEST recipe (`--budget 7200`) is evaluated the same way. No agent sees the sealed sets.

## Sets

| set | file | who reads the file | who sees the numbers | scored by |
|---|---|---|---|---|
| RL pool | `frozen/data/targets_L6-11.jsonl` (copy of `ab2/data/targets_L6-11.jsonl`), true length 6-11, no proofs | agents (through `rl.run`'s `targets`) | - | - |
| dev | `ab2/data/dev_L8-11.jsonl` | scorer only | agents (aggregate and per-length counts) | daemon, k = 32, max_new 512 |
| frontier | `ab2/data/frontier_L12-14.jsonl` | scorer only | agents (aggregate and per-length counts) | daemon, k = 64, max_new 640 |
| sealed L12-14 | `ab2/sealed/sealed_L12-14.jsonl` | monitor only | monitor only | `score_host.py` only, k <= 256, max_new 640 |
| sealed L15-16 | `ab2/sealed/sealed_L15-16.jsonl` | monitor only | monitor only | `score_host.py` only, k <= 256, max_new 640 |
| textbook | `ab/sealed/textbook_dev.jsonl` | monitor only | monitor only | `score_host.py` only, k <= 256, max_new 512 |
| cap-6 train | `frozen/data/train_cap6.jsonl`, 154,990 records with proofs (the pretraining data) | agents | - | - |
| cap-6 held-out | `frozen/data/heldout_cap6.jsonl`, 5,000 records with proofs | harness (damage check) | agents (`heldout_greedy`) | harness |

All pools and sets are meant to be pairwise disjoint by atom-renaming class (`lib/keys.py`).
The pool builder is responsible for that, and the monitor should check it before launch. File
names are constants at the top of `harness.py` (`TARGETS_FILE`, ...) and `score_host.py`
(`DEV_FILE`, `FRONTIER_FILE`, `SETS`). `score_host.py` looks for each sealed file in `ab2/sealed/`
first, then in `ab2/data/`.

## Directory layout

```
/scratch/cowifornia/nd-rl/ab2/       the arm directory (daemon arm "ab2", host GPUs 0-4)
  common/                    the master copy of the frozen files (monitor only)
  frozen/                    verbatim copy of common/ (read-only for agents)
  controller.py              the editable GEAR controller (starts as frozen/gear/controller.py)
  test_controller.py         its test (starts as frozen/gear/test_controller.py)
  gear_state.json, nodes/    controller state and node copies (written only by the controller)
  reflections.md, decisions.md, ideas.md, audit.md    shared append-only logs
  w1/ ... w7/, auditor/      one workspace per agent: rl.py (editable), runs/<name>/
  score_requests/, score_results/   the scoring queue (client <-> daemon)
  logs/                      scores.jsonl (clients), daemon_scores.jsonl (daemon)
  locks/                     GPU locks (gpulock.py)
  data/                      the pool builder's directory, including the dev and frontier theorems (scorer only)
  sealed/                    monitor-only sealed sets
```

Frozen files, by role:

| file | role |
|---|---|
| `frozen/harness.py` | one experiment: lock a GPU, load the checkpoint, call `rl.run`, enforce the clock, save `final.pt`, score dev and frontier, write `summary.json` |
| `frozen/score.py` | score client: writes a request, waits for the daemon, prints aggregate counts |
| `frozen/gpulock.py` | exclusive per-GPU locks with detached holders |
| `frozen/metrics.py` | per-step metrics logger (JSONL) |
| `frozen/rl.py` | the seed `rl.py`: run 1's best recipe (GEAR node n025) with a `KNOWN_FAILURES` block |
| `frozen/lib/` | model, tokenizer, sampler, supervised-loss helpers, verifier, prune, normalize, eval helpers, class keys |
| `frozen/data/` | `targets_L6-11.jsonl`, `train_cap6.jsonl`, `heldout_cap6.jsonl`, `SHA256SUMS` |
| `frozen/pretrain.pt` | the starting checkpoint (`{'state', 'tok_mode': 'abs', ...}`; construct `GPT(tok.vocab_size, 4, 256, 8)`) |
| `frozen/gear/` | the GEAR controller, its test and `README.md` (the worker and auditor protocol) |
| `frozen/ref/` | reference material: the fork's `expert_iter.py` and `grpo.py`, the GEAR paper, the orchestration proposal, the logic spec |
| `frozen/score_host.py`, `frozen/score_daemon.py` | copies of the host-side scorer and daemon, for reference. The monitor runs them from `common/`. Agents never run them |

## Harness contract (`frozen/harness.py`)

```
python frozen/harness.py --out runs/<name> [--seed 0] [--budget 1200] [--rl rl.py]
                         [--gpu auto|N|env] [--gpus 0,1,2,3,4] [--gpu-wait 14400] [--no-score] [--smoke]
```

0. **GPU.** `--gpu auto` (the default) acquires the first free GPU of `--gpus` through
   `gpulock.py`. If all are held, it waits up to `--gpu-wait` seconds (default 4 h), then exits
   with an error. It exports `CUDA_VISIBLE_DEVICES=<gpu>` for itself and for the scoring
   subprocesses, and releases the lock when it exits: normally, on a crash, on a timeout, and on
   SIGTERM or SIGHUP (through try/finally plus an atexit hook). If the harness is SIGKILLed, the
   lock's holder process sees its owner die and frees the GPU within 2 s. `--gpu N` locks GPU N.
   `--gpu env` takes no lock and uses `CUDA_VISIBLE_DEVICES` as it is; it is for the monitor and
   for tests only. The time spent waiting for a GPU does not count against the budget.
1. Seeds torch and Python. Loads `frozen/pretrain.pt` into `lib.model.GPT(vocab, 4, 256, 8)` with
   the `abs` tokenizer. Copies `rl.py` into `runs/<name>/source/` and records the SHA-256 of
   `rl.py`, `harness.py`, `score.py`, `metrics.py`, `gpulock.py`, `pretrain.pt`, every file in
   `lib/` and every file in `data/` into `summary.json`.
2. Imports the RL module and calls

   ```python
   model = rl.run(model, tok, targets, budget_secs, device, seed, log, metrics, paths)
   ```

   | argument | meaning |
   |---|---|
   | `model` | the loaded GPT, on `device`; train mode allowed |
   | `tok` | `lib.tokenizer.Tokenizer('abs')` |
   | `targets` | list of dicts from `targets_L6-11.jsonl` (`name, thm, key, prompt, n_lines, L_true, source, schema, n_prem`) |
   | `budget_secs` | wall clock for the whole call (default 1,200; 7,200 for a champion run) |
   | `device` | `'cuda'` |
   | `seed` | int |
   | `log(msg)` | timestamped print |
   | `metrics` | a `metrics.Metrics` instance (schema below) |
   | `paths` | dict: `targets`, `train_cap6`, `heldout_cap6`, `out_dir`, `save_checkpoint(model, tok, path)`, `heldout_greedy(model, tok, n=1000)`, `smoke` (bool). Run 1's `targets_extra` is gone |

   `rl.run` must return the model to be scored, with the same architecture. It may save
   intermediate checkpoints under `paths['out_dir']` with `paths['save_checkpoint']`.
3. **The clock.** SIGALRM fires at `budget_secs + 45`, and the run is recorded as
   `status: timeout` with no score. Returning later than `budget_secs + 15` is also a timeout.
   Well-behaved RL code checks `time.time() - t0` and stops with margin. The seed stops at least
   20 s early.
4. After `rl.run` returns: `final.pt` is saved, and `heldout_greedy` is computed on a fixed
   1,000-record slice of `heldout_cap6.jsonl` (every 5th record, 200 per length 2-6; greedy
   decoding). Unless `--no-score` is given, the harness then runs `frozen/score.py --set dev --k 32
   --seed 0` and `frozen/score.py --set frontier --k 64 --seed 0`, in that order, on the same GPU.
   The harness never reads a scoring file itself. Before any training, the harness refuses a targets
   file with duplicate `name` fields (the seed `rl.py` keys everything by name).
5. `summary.json` fields:
   - `status`: `ok`, `timeout`, `crash`, or `scoring` (provisional, while scoring runs).
   - `metric`: the dev total. `by_len`: dev solved per length, e.g. `{"8": .., "9": .., "10": .., "11": ..}`.
   - `frontier`: the frontier total. `frontier_by_len`: frontier solved per length (`{"12": .., "13": .., "14": ..}`).
   - `heldout_greedy`, `rl_secs`, `score_secs` (`dev_score_secs`, `frontier_score_secs`), `rl_stats`
     (whatever `rl.run` put in `metrics.summary`).
   - `score` and `frontier_score` (the full score JSONs, with the denominators in `n_by_len`),
     `score_line`, `frontier_line`.
   - `shas`, `seed`, `budget`, `gpu`, `gpu_lock`, `gpu_wait_secs`, `worker` (env `AB_WORKER`),
     `started_utc`, `run_name`, `arm`, `smoke`, `targets_file`, `total_secs`, and `reason` on a
     crash or timeout.
6. `--smoke`: 8 targets, a 20 s budget, 64 held-out records, and 16-theorem subsets of dev and
   frontier (`--smoke` requests). For plumbing checks only. `AB_SCORE_STUB=1` replaces the daemon
   with an all-zero local result (tests only).

Checkpoints (`final.pt` and anything saved through `save_checkpoint`) have the layout
`{'cfg', 'state', 'tok_mode', 'extra', 'stage', 'seed', 'shas'}`. No `final.pt` is written on a
timeout or crash. Data hashes are cached in `data/SHA256SUMS`. The cache is recomputed whenever
the file list changes or any data file (or symlink) was created or changed after the cache was
written.

## Scoring

**Client** (`frozen/score.py`, run by the harness or by an agent):

```
python frozen/score.py --ckpt <abs path under ab2>/final.pt [--set dev|frontier] [--k 32|64] [--seed 0] [--tag text] [--smoke]
```

It writes `{id, set, ckpt, k, seed, gpu, cwd, tag, smoke, t}` to `ab2/score_requests/<id>.json`.
`ckpt` is the host-absolute path, and `gpu` is the caller's `CUDA_VISIBLE_DEVICES`, which is the
host GPU it holds. It polls `ab2/score_results/<id>.json` every 3 s, for up to 55 minutes, and
prints one line:

```
dev pass@32: solved T/N | L8 a/n8 L9 b/n9 L10 c/n10 L11 d/n11 | secs S
```

It then writes `score.json` (dev) or `score_frontier.json` next to the checkpoint and appends
the result to `ab2/logs/scores.jsonl`. The dev set is the default. `--k` defaults to the official value for the set: dev k = 32, frontier k = 64.
Only those, with seed 0 and not smoke, are official (`official: true`, decided by `score_host.py`). Other `k` (1-256) and seeds are allowed for
exploration, but every request costs GPU time on the requester's GPU.

**Daemon** (`common/score_daemon.py`, run by the monitor):

```
python common/score_daemon.py --arm ab2:/scratch/cowifornia/nd-rl/ab2:0,1,2,3,4
```

- It serves only `set` in {`dev`, `frontier`}; every other set is refused with an error result.
- The checkpoint must resolve, after following symlinks, to a file under the arm directory. Run
  1's `/work/...` paths are still mapped.
- `--gpu-map host` (the default) treats the request's `gpu` as a host GPU index, which must be in
  the arm's list. The daemon runs `score_host.py` on that GPU, at most once per GPU at a time
  (`--per-gpu 1`) and at most 5 at a time for the arm (`--per-arm 5`).
- Results carry aggregate and per-length counts only (`solved`, `by_len`, `n_by_len`, `n`, `k`,
  `seed`, `secs`, `samples_per_sec`, `ckpt_sha256`, `official`), never per-theorem data.
- It appends each result to `ab2/logs/daemon_scores.jsonl` (`--log`).

**Host scorer** (`common/score_host.py`, run by the daemon and by the monitor):

```
python common/score_host.py --ckpt <path> --set dev|frontier|sealed_L12-14|sealed_L15-16|textbook
                            [--k K] [--seed 0] [--smoke] [--out f.json] [--per-theorem rows.jsonl]
```

- For each theorem it samples k proofs (1 <= k <= 256; default 32 for dev, 64 otherwise), at
  temperature 0.8, with `max_new` 640 for `frontier`, `sealed_L12-14` and `sealed_L15-16` (proofs of 12+
  lines can exceed 512 tokens) and 512 for `dev` and `textbook` (`MAX_NEW_BY_SET`), batch 384 and a fixed seed, using `lib.sample.generate`. It verifies each proof with
  `nd_verify.verify_text(prompt + ' ' + proof)`. A theorem counts as solved if any sample verifies.
- It loads checkpoints with `torch.load(weights_only=True)`.
- `--per-theorem` is for the monitor only. It may not write anywhere under `ab2/` except
  `ab2/sealed/` and `ab2/monitor/`, and not into run 1's arms.
- Env overrides for tests: `AB2_POOL_DIR` (dev and frontier), `AB2_SEALED_DIR`, `AB2_TEXTBOOK`.

## Budget

The per-experiment RL budget is **1,200 s** (the harness default). Do not change it for search
experiments. SIGALRM stays at budget + 45 s. A champion run passes `--budget 7200`; nothing else
in the harness, the scorer or the seed `rl.py` caps the budget. The seed's round count is bounded
only by the clock. Scoring adds the dev pass (k = 32) and the frontier pass (k = 64, max_new 640). The time on the
stand-in pools is in "Build notes".

## GPU-lock protocol

- `python frozen/gpulock.py acquire [--gpus 0,1,2,3,4] [--wait SECS] [--run NAME] [--owner PID]`
  takes an exclusive `flock` on `ab2/locks/gpu<i>` for the first free GPU and prints `i`.
  - The flock is held by a small detached holder process (its own session, no stdio). The holder
    writes its pid to `ab2/locks/gpu<i>.pid` and one JSON line into the lock file:
    `{gpu, worker: $AB_WORKER, run, holder_pid, owner_pid, since, host}`.
  - With `--owner PID`, the holder exits, freeing the GPU, within 2 s of that process dying. The
    harness passes its own pid here.
  - Without `--owner`, the lock is held until `release`.
  - `--wait` retries every 5 s. The default of 0 means a single pass: exit status 1 if no GPU is free.
- `python frozen/gpulock.py release <i>` stops that GPU's holder, after checking that the pid
  really is a gpulock holder, and waits until the lock is free.
- `python frozen/gpulock.py status [--json]` lists every GPU as free or as held, with the worker,
  run, holder pid, owner pid and held time.
- **Stale locks.** The kernel drops an flock when its holder dies, so a lock whose holder pid is
  dead is free, and the next `acquire` takes it and overwrites the stale info.
- **The rule for agents.** Run every GPU job through `harness.py --gpu auto`. Do not set
  `CUDA_VISIBLE_DEVICES` by hand, do not use `--gpu env`, and do not run GPU code outside the
  harness. The one exception: if you must, acquire a lock with `gpulock.py acquire` first and
  release it afterwards. Never release a GPU you do not hold.
- Ad hoc `score.py` calls on your own checkpoints run on the GPU in your `CUDA_VISIBLE_DEVICES`.
  Make them only while you hold that GPU's lock.

## Metrics schema (`frozen/metrics.py`)

Unchanged from run 1. `Metrics(out_dir, run_name, arm, wandb_project='nd-rl-hillclimb', t0=None)`.
Each `metrics.log(**fields)` call appends one JSON object to `runs/<name>/metrics.jsonl`. Every
record carries `t`, `elapsed`, `step` (monotone) and `phase` in {`sample`, `train`, `eval`,
`summary`}. The per-phase fields follow run 1's SPEC. In the seed `rl.py`, `solved_by_len` and
`solved_by_len_round` have one key per true length present in the pool. W&B mirroring, if wanted,
is done host-side by the monitor.

## Rules for agents

1. Edit only your workspace's `rl.py`, and `ab2/controller.py` under the GEAR-Evolve protocol
   (`frozen/gear/README.md`). Never edit anything under `frozen/` or `common/`. The harness
   records hashes, and a run with a modified frozen file is void. A controller step may not
   remove or weaken the two-seed rule, the held-out floor, `--source`, the seven-worker cap or
   the auditor commands.
2. Never read, list, copy or search for anything under `ab2/sealed/`, `ab/sealed/` or `ab2/data/`.
   `ab2/data/` holds the dev and frontier theorems; the RL pool you may use is
   `frozen/data/targets_L6-11.jsonl`. Also stay out of `common/`, run 1's directories
   (`/scratch/cowifornia/nd-rl/ab/`, except the read-only reference files named in your brief),
   and other workers' directories. From other agents' work you may read only the shared files:
   the controller's `frontier`, `history` and `next` output, `nodes/`, `reflections.md`,
   `decisions.md`, `ideas.md` and `audit.md`. Do not try to reconstruct the dev, frontier or
   sealed sets.
3. **Training data.** Only these are allowed:
   - (a) proofs your own model sampled during this run that `nd_verify` accepts for the prompted
     sequent, where the prompts come from `targets_L6-11.jsonl`;
   - (b) verified by-products of your own samples: a valid proof of a different conclusion from
     the same premises (hindsight relabelling);
   - (c) records from `train_cap6.jsonl`.

   Nothing else is allowed: no hand-written proofs, no proofs written by an LLM, no proofs of
   length > 6 from any file, no proofs from earlier runs or other workers' runs, and no prompts
   from the dev, frontier or sealed sets. `heldout_cap6.jsonl` is for tracking only.
4. Only experiments launched through `harness.py --gpu auto` count. One experiment per GPU lock.
   The lock is exclusive, so at most five experiments run at once, and the rest wait. Sampling
   batch 768 with the compacting sampler fits a 3090 alongside training; batch 1024 was slower in
   run 1.
5. The budget is 1,200 s. Do not change it, except for a champion run ordered by the monitor.
6. Only scorer numbers count. Report the dev score as the scorer prints it, together with the
   frontier count. Single-seed differences below the controller's tie margin
   (`2 * max(8, round(0.6 * sqrt(best)))`, about 24 at a best of 400) are noise. The BEST role
   needs two seeds (below).
7. Stop launching experiments at the deadline in your brief. Finish by naming the BEST node and
   its checkpoint paths, and write the report your brief asks for.

## GEAR protocol (summary; details in `frozen/gear/README.md`)

- Workers loop `next` -> copy the parent -> one coherent edit -> harness -> `record --source` ->
  reflection and `ideas.md` row.
- **Two-seed rule.** A child whose dev metric beats the current BEST by more than the
  improvement margin, `max(8, round(0.6 * sqrt(best)))`, becomes `PENDING_BEST`. It stays out of
  the frontier and cannot be a parent. `next --worker auditor` lists it as a replicate job. The
  auditor runs `nodes/<id>/rl.py` unmodified with the next seed and records the result with
  `replicate --node <id> --summary <path>`. If the two-seed mean still beats the BEST at that
  moment by the margin, the node becomes BEST. Otherwise it is demoted and goes through the
  normal promotion rules with the two-seed mean. Non-BEST promotions (REPLACE_WEAKEST, LEAN,
  DIVERSE) stay single-seed. BEST is sticky: a single-seed child that lands above the BEST, but
  within the margin, does not take the role.
- **Held-out floor.** A node with `heldout_greedy < 0.85` is never promoted (decision `FLOOR`).
  The floor also applies to the two-seed mean.
- The tie margin is 2x the improvement margin. The run-1 bands (v2 share, v3 p1 floor, v7 loss)
  scale with it.

## Research logging

- Every node cites a source: `record --source <URL>`, or `--source "own idea"`. For a crossover,
  give the source of the transplanted idea. The controller refuses a record without one, and
  shows the source in `frontier` and `history`.
- `ab2/ideas.md` is a Markdown table with one row per idea tested:
  `| date | worker | idea | source URL | node ids that tested it | verdict |`. Rows are appended,
  never edited by another worker. A later test of the same idea appends a new row that refers to
  the earlier one. The verdict gives the numbers (delta vs parent1, and the two-seed mean if the
  node was replicated) and a one-clause reading.
- `reflections.md` (one line per experiment) and `decisions.md` (one line per iteration) keep
  run 1's format (README).
- The seed `rl.py` opens with a `KNOWN_FAILURES` block listing what did not help in run 1. Check
  it and `ideas.md` before proposing an idea.

## Useful facts

- The seed (run 1's n025) reached dev 401/1,610 on L7-9 at pass@32 in 600 s: 15 rounds of about
  40 s each (sampling 20-23 s, fine-tune 135 steps). With 1,200 s it has room for roughly 29
  rounds on a 3090, depending on the pool.
- **Target subset.** The new pool may hold 12,000-15,000 targets. The seed `rl.py` draws, per run and
  per seed, at most `CONFIG['max_targets']` = 6,000 of them, `stratify='length'`: equal quotas per true
  length present, with the unused quota of short-supplied lengths (e.g. a few hundred L11s) shared by
  the others. `stratify=None` samples uniformly. A pool within the cap is used whole, shuffled exactly
  as in run 1. With window 400 the first pass over 6,000 targets takes 15 rounds (about 600 s), then
  hardest-first refills (1,200 targets each) run over the same subset.
- Run 1's worst failure modes are in `KNOWN_FAILURES`. Its levers were throughput (the compacting
  sampler, +64) and more, smaller rounds (+50).
- Run 1's run-to-run noise was about 15 theorems on 1,610 at pass@32. Replicates of one recipe
  within the n013 line spread 343-358.

## Build notes (2026-09-24)

- **Pools pending.** `common/data/targets_L6-11.jsonl` is currently a **symlink to run 1's
  `targets_L7-9.jsonl`** (3,804 targets, L7-9), a stand-in for testing. Before launch, replace it
  with a real copy of `ab2/data/targets_L6-11.jsonl`, delete `common/data/SHA256SUMS` if present,
  and re-copy `common/` to `ab2/frozen/`. `ab2/data/dev_L8-11.jsonl`, `frontier_L12-14.jsonl` and
  the sealed files do not exist yet. `score_host.py` fails with a `FileNotFoundError` naming the
  missing file until they do.
- The stand-in test pools live in `/scratch/cowifornia/nd-rl/ab2_selftest/`
  (`pools/dev_L8-11.jsonl` -> `ab/sealed/dev_L7-9.jsonl`, `pools/frontier_L12-14.jsonl` ->
  `ab/sealed/test_L10.jsonl`, `sealed/*` -> `ab/sealed/test_L11-14.jsonl`), used through the
  `AB2_*` env overrides. That directory also holds the self-test arm and can be deleted.
- Measured on GPU 4 with the stand-ins: see the numbers in the harness-build report (dev pass@32 and
  frontier pass@64 at max_new 640 on 1,610 + 907 stand-in theorems).

## Addendum 2026-09-24 03:02 UTC (monitor): frontier scoring cost

Scoring the full 1,935-theorem frontier set at k=64 with max_new 640 took about 10 minutes per experiment,
a third of the experiment's wall clock. From this time the frontier score is computed on a fixed,
stratified subset of 1,000 theorems (700 at length 12, 200 at 13, 100 at 14, drawn once with a fixed seed)
at k=64 and max_new 512. Frontier counts recorded before 03:02 UTC used the full set and are not comparable;
the controller's metric (dev) is unchanged. The full set and the sealed half are scored by the monitor at the end.

## Addendum 2026-09-24 05:00 UTC (monitor): fair GPU queue

`gpulock.py acquire` now takes a FIFO ticket (`locks/queue/<ns>-<pid>.json`) and only the oldest live ticket
may take a free GPU, after a worker waited 41 minutes while later arrivals were served. Tickets of dead
processes are discarded. The harness's `--gpu auto` picks this up automatically for every new launch;
`frozen/SHA256SUMS` was regenerated (gpulock.py and SPEC.md changed, by the monitor).
