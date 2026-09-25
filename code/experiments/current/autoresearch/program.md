# Autoresearch: pretrain workstream

You are an autonomous researcher improving the pretraining stage of a small proof model. Complete
one experiment per session; `loop.py` starts fresh sessions until cancelled. Adapted from Karpathy's
autoresearch (2026) `program.md`; the differences are the metric and the harness split.

**Lean protocol since 098 (2026-09-24).** The model now reads and writes Dan's `lean_seq` Lean
surface form (fork commit `51604b3`, `lean_tok.py`), every sampled proof must pass Lean 4 AND
nd_verify, and TOTAL_SECS is 2400 (1800 for 098-099). Data, pools, RL recipe and metric definition
are unchanged. Scores from runs 000-097 (token format, `abs`) are NOT comparable: they are history
and a source of ideas, not the bar. The bar is the 3-seed mean of the incumbent under the Lean
protocol, starting with the 098 re-baseline of the 081-line recipe. Keep `TOK_MODE = "lean_seq"`;
the harness crashes any other tokenizer. Ideas that won under the token format may not transfer.
From 105 the held-out greedy check covers proof lengths 2-6 (400 each); before it covered 2-5 only,
so `heldout_greedy` values are not comparable across that boundary. The metric is unaffected.

**Two agents run this loop in parallel**: Claude (`claude-opus-5-5`, `$ND_AGENT=claude`) and Fable
(`claude-fable-5-1`, `$ND_AGENT=fable`) since 2026-09-25; Codex (`gpt-6-astra`) ran 041-097. Each
works in its own checkout, both pushing to the same branch. Robbie counts who gets more keeps. If
you are `fable`: be creative. Prefer bold ideas the table has not tried (new mechanisms, not another
nudge to a tuned number), and check the other agent's running registration so you do not duplicate
it.

**Speed work is encouraged (Robbie, 2026-09-25)** Pretraining is wall-clock bound: 101-103 showed
each size step costing 14-24% of steps, and steps turn into score. Only three pure systems runs
exist (001, 002, 003, all before the Muon and Lean incumbents), so this is untouched ground. Leads,
unmeasured: the ALiBi float `attn_mask` keeps SDPA off the flash kernel; no torch.compile since 003;
batches pad to the longest record (mean ~135 tokens, max 537), so length bucketing or packing (with
its data-order risk, see 002); the per-step Newton-Schulz over 21+ matrices is unbatched; the
val-loss pass every 15 s. A faster forward also speeds the harness's RL sampling. `pretrain.steps`
in results.json is your throughput measurement; state the expected step count in the prediction.
Keeps are still decided by the metric: spend the speed on steps, or on a bigger model.

This doesn't need to be 100% of your focus. Do some speed work, and hopefully this allows you to
train bigger or looped architectures on the next round.

`$ND_AGENT` says which one you are. The incumbent is whatever `pretrain.py` is at the branch HEAD,
and either agent can replace it. The scripts pull before every publish; you just
`git pull --ff-only` at the start of your session so you work from the latest incumbent.

## Setup

- `harness.py` is FROZEN. It owns the wall-clock (TOTAL_SECS from process start), the parameter cap
  (PARAM_CAP), the RL stage (RL_ROUNDS rounds of expert iteration, fixed in samples) and the eval.
  The numbers are the constants at the top of the file and are copied into every params.json as
  `protocol` (runs 000-010: 600 s / 240 s / 2 rounds / 32 attempts; since 011: 1200 / 300 / 4 / 64;
  since 098: 1800 / 300 / 4 / 64, since 100: 2400 / 300 / 4 / 64 under the Lean protocol). You never
  edit it. Its sha256 is written into every result; a changed harness invalidates the run.
- `pretrain.py` is the ONLY file you edit. Contract in its docstring. Your budget is PRETRAIN_SECS
  of wall-clock, compile and model construction included, then the harness takes the model.
- Frozen data and verifier come from Dan's fork at the commit pinned in `remote.sh`.
- The metric is `metric` in `summary.json`: dev-transfer theorems solved at true length >= 7 with
  EVAL_K attempts each. Higher is better. The RL stage is what turns pretraining into that number,
  so think about what makes a model a good starting point for expert iteration, not only about
  pretrain loss.

## Loop

For the terminal launcher, setup and logging, see [LOOP.md](LOOP.md). Its task prompt is
[iteration.md](iteration.md). The research scope is pretraining only: keep data and RL fixed.

All commands run from this directory. The scripts need the repo venv (pydantic), so
`PY=$(git rev-parse --show-toplevel)/code/.venv/bin/python` and use `$PY` where `python3` is
written. Run `make -C code check` from the repo root before launching (Ruff + Pyright). Report
failures; do not fix unrelated files. The scripts perform Git operations with normal hooks enabled
and stop on commit/push failure. Do not stage, commit, push, or restore files manually in the loop.

1. Read `results/*/params.json` + `results.json` (all previous experiments, kept and discarded),
   `RESULTS.md`, and `git log`. Also read the
   [pretraining idea backlog](../../../../.claude/notes/ideas.md) at startup. Robbie is excited
   about P01 (looped architectures) and P02 (Muon); these are not mandatory priorities. Include the
   chosen idea ID in the registration description.
2. Pick one idea. Edit `pretrain.py`.
3. Pre-register: `$PY params.py new <NNN> <slug> --seeds 0,1,2` writes
   `results/<NNN>-<agent>_<slug>/params.json` (it appends your `$ND_AGENT` to the id; use the next
   number after the highest existing one) with mechanical fields, `provider`/`model`, `protocol`,
   and `pretrain.patch` filled in, and leaves `TODO`s. If the other agent kept a new `pretrain.py`
   while you were editing, it refuses: re-apply only your idea on top of the new HEAD file and rerun
   it. Fill every TODO: title, description, hypothesis, prediction, and `config` (what your
   pretrain.py actually builds; read the file, do not guess). Use the intended seeds here and at
   launch (e.g. `--seeds 3,4,5` for a control). Both `config.seed` and `protocol.seed` store that
   list; historical runs use a scalar. The cached target histogram is seed-0-only, so
   `targets_sampled` is null for other seeds. The runner commits and pushes these two registration
   files before launching. Committing the prediction BEFORE the run is enforced; params.json is
   never edited after this commit.
4. Run on three seeds at once, one pod each:
   `source "$(git rev-parse --show-toplevel)/.envrc.local" && $PY run_idea.py <NNN>-<agent>_<slug> --seeds 0,1,2`.
   It validates the schema, source hashes, fork and seeds, rejects dirty frozen files, then
   commits/pushes only this experiment's params.json and pretrain.patch. Existing committed
   registrations must match HEAD exactly. It rechecks validation after hooks and before pods. It
   writes the merged `artifacts/autoresearch/<name>/summary.json` (`metric` = mean over seeds,
   `seed_metrics` per seed; status ok only if every seed is ok) plus one dir per seed.
5. Record: `$PY record.py <NNN>-<agent>_<slug> keep|discard|crash|timeout "surprise"` writes
   `results.json` (loss curve, per-round RL numbers, eval histograms, seed metrics) and refuses if
   the run does not match what was pre-registered. It also refreshes `RESULTS.md`, applies the
   supplied decision, and commits/pushes only results.json, RESULTS.md, and pretrain.py. It first
   pulls the other agent's results. If they kept a file scoring at least as high as yours during
   your run, your `keep` becomes a `discard` automatically. A keep replaces pretrain.py whole
   (winner-take-all; never merged with their file). On discard/crash/timeout, pretrain.py becomes
   the incumbent at HEAD. No manual table refresh or Git step.
6. Choose `keep` only if the idea's mean `metric` is above the current baseline's mean `metric`
   (both are 3-seed means). Ties: keep only if simpler. Otherwise choose `discard`. Unrelated
   staged/unstaged work is left alone. If a commit/push fails, stop and report it; rerunning the
   same command can finish publishing without an extra empty commit. A completed record's
   status/observation cannot be rewritten on retry.
7. Verify your pods are terminated, then exit. The launcher starts the next experiment.

## Rules

- **Keep rule.** 3-seed mean strictly above the baseline's 3-seed mean. No noise threshold. One seed
  has sd ~44 at this budget (`SWEEP_s20.md`), so some keeps will be luck; that is accepted (Robbie,
  2026-09-22). Always use seeds 0,1,2 so runs are comparable.
- **Simplicity.** All else equal, simpler is better. An improvement of ~0 with less code: keep. A
  small improvement with 20 lines of hacks: discard.
- **Crash / timeout counts as a failed experiment**, on any seed. If the cause is dumb (typo,
  import), fix and re-run under the same NNN. If the idea is what crashed (OOM, overran the budget),
  log it as `crash` or `timeout` and move on. Never edit the harness to make something fit.
- **No cheating.** No eval or RL code in pretrain.py. No touching the pools. No external weights, no
  hand-written proofs, no calling the verifier from the decoder.
- **Three pods per experiment, one experiment at a time per agent**, so never more than 6 A40s
  across both agents. Pods are named `autoresearch-ndrl-pretrain-<agent>-<NNN>-s<seed>-...`; never
  touch the other agent's pods.
- **Cleanup.** `run_idea.py` terminates its pods after running. On cancellation, `loop.py` stops the
  local processes and terminates only the pods registered to that session. There is no overnight
  cutoff or experiment-count limit.
- **Tags** (multi-select): systems, optimizer, schedule, architecture, tokenizer, data-order,
  augmentation, loss, init, regularization, batch, simplification.
- Do not ask whether to continue within your assigned experiment. If out of ideas, re-read the
  fork's code or combine near-misses. Exit after recording one experiment, or report a blocker.

## Files per experiment: `results/<NNN>-<agent>_<slug>/` (ids before 041 have no agent)

- `params.json`: frozen before the run. `params.py new` fills the mechanical part; you fill the
  rest.
- `pretrain.patch`: `git diff HEAD -- pretrain.py` at pre-registration. The code IS the config; this
  is the ground truth for every run, kept or discarded.
- `results.json`: written by `record.py` from `summary.json` after the run.

### params.json (agent-filled fields marked `<...>`)

```json
{
  "id": "041-claude",
  "slug": "wider-ffn",
  "written_at": "...",
  "base_commit": "<HEAD when written>",
  "provider": "anthropic",
  "model": "claude-opus-5-5",
  "harness_sha": "...",
  "pretrain_sha": "...",
  "patch_lines": 12,
  "title": "<one line>",
  "description": "<two sentences: what changes and why it should help expert iteration>",
  "tags": ["<architecture>"],
  "hypothesis": "<what mechanism would make this work>",
  "p_keep": 0.35,
  "expected_delta": "<+3 to +8 theorems at >=7>",
  "baseline_id": "<001>",
  "baseline_metric": 67,
  "config": {
    "arch": {
      "type": "gpt (fork model.py)",
      "n_layers": 4,
      "d_model": 256,
      "n_heads": 8,
      "d_ff": 1024,
      "pos": "rope",
      "ctx": 1024,
      "norm": "layernorm",
      "act": "gelu",
      "dropout": 0.0,
      "tied_emb": false,
      "n_params": 3210240
    },
    "tokenizer": { "codec": "abs", "maxn": 64 },
    "optimizer": {
      "name": "adamw",
      "lr": 1e-3,
      "schedule": "cosine over wall-clock",
      "warmup_steps": 200,
      "min_lr_frac": 0.1,
      "wd": 0.1,
      "betas": [0.9, 0.95],
      "grad_clip": 1.0
    },
    "batch": { "size": 256, "padding": "to max len in batch", "buckets": false },
    "data": {
      "filter": null,
      "augmentation": "random ref shift (abs)",
      "order": "shuffle",
      "val_slice": 1024
    },
    "precision": "bf16 autocast",
    "compile": false,
    "init": "fork default (normal 0.02)",
    "seed": 0
  },
  "protocol": "<mechanical: budget, pool sizes and proof-length histograms, RL and eval settings>"
}
```

`config` describes what your `pretrain.py` builds. Every key in the template must be filled; add
keys when you add a knob. `n_params` is cross-checked against the harness after the run.

### results.json (written by record.py, you only supply status and surprise)

`commit`, `recorded_at`, `status` (keep|discard|crash|timeout), `metric`, `metric_rl`, `rules`,
`gpu`, `total_secs`, `reason`, `pretrain` (steps, epochs, last_loss, heldout_greedy, secs, n_params,
`curve` = [{secs, step, train_loss, val_loss}, ...]), `rl_rounds` (per round: targets_solved,
sample_acc, lstar, ge histogram, new_proofs, samples_per_sec), `eval` (solved, n, ge, lstar,
heldout_greedy, by_bin), `holdout_metric` (null; scored offline), `surprise`.
