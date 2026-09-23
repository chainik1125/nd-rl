# Autoresearch: pretrain workstream

You are an autonomous researcher improving the pretraining stage of a small proof model. One
long session, one experiment at a time, forever, until a human stops you. Adapted from
Karpathy's autoresearch (2026) `program.md`; the differences are the metric and the harness split.

## Setup

- `harness.py` is FROZEN. It owns the wall-clock (TOTAL_SECS from process start), the parameter
  cap (PARAM_CAP), the RL stage (RL_ROUNDS rounds of expert iteration, fixed in samples) and the
  eval. The numbers are the constants at the top of the file and are copied into every
  params.json as `protocol` (runs 000-010: 600 s / 240 s / 2 rounds / 32 attempts; since 011:
  1200 / 300 / 4 / 64). You never edit it. Its sha256 is written into every result; a changed
  harness invalidates the run.
- `pretrain.py` is the ONLY file you edit. Contract in its docstring. Your budget is
  PRETRAIN_SECS of wall-clock, compile and model construction included, then the harness takes
  the model.
- Frozen data and verifier come from Dan's fork at the commit pinned in `remote.sh`.
- The metric is `metric` in `summary.json`: dev-transfer theorems solved at true length >= 7
  with EVAL_K attempts each. Higher is better. The RL stage is what turns pretraining into that
  number, so think about what makes a model a good starting point for expert iteration, not
  only about pretrain loss.

## Loop

All commands run from this directory. The scripts need the repo venv (pydantic), so
`PY=$(git rev-parse --show-toplevel)/code/.venv/bin/python` and use `$PY` where `python3` is written.
Run `make -C code check` from the repo root before each commit (Ruff + Pyright; the pre-commit hook
runs the same).

1. Read `results/*/params.json` + `results.json` (all previous experiments, kept and discarded),
   `RESULTS.md`, and `git log`.
2. Pick one idea. Edit `pretrain.py`.
3. Pre-register: `$PY params.py new <NNN> <slug> --seeds 0,1,2` writes `results/<NNN>_<slug>/params.json`
   (mechanical fields, `protocol`, and `pretrain.patch` filled in) and leaves `TODO`s. Fill every
   TODO: `provider`/`model` (the agent that is you), title, description, hypothesis, prediction,
   and `config` (what your pretrain.py actually builds; read the file, do not guess).
   Use the intended seeds here and at launch (e.g. `--seeds 3,4,5` for a control).
   Both `config.seed` and `protocol.seed` store that list; historical runs use a scalar.
   The cached target histogram is seed-0-only, so `targets_sampled` is null for other seeds.
   Then
   `git add results/<NNN>_<slug> && git commit -m "autoresearch <NNN> <slug>: pre-registered" && git push`.
   Committing the prediction BEFORE the run is the rule, not a suggestion; params.json is never
   edited after this commit.
4. Run on three seeds at once, one pod each:
   `source ~/repos/nd-takehome/.envrc.local && $PY run_idea.py <NNN>_<slug> --seeds 0,1,2`.
   It validates the schema, requires params.json and pretrain.patch to match committed HEAD,
   checks source hashes, fork and seeds, and rejects dirty frozen files before renting pods.
   It writes the
   merged `artifacts/autoresearch/<NNN>_<slug>/summary.json` (`metric` = mean over seeds,
   `seed_metrics` per seed; status ok only if every seed is ok) plus one dir per seed.
5. Record: `$PY record.py <NNN>_<slug> keep|discard|crash|timeout "surprise"` writes
   `results.json` (loss curve, per-round RL numbers, eval histograms, seed metrics) and refuses
   if the run does not match what was pre-registered. Then `$PY table.py` to refresh `RESULTS.md`.
6. Keep if the idea's mean `metric` is above the current baseline's mean `metric` (both are
   3-seed means): `git add pretrain.py results RESULTS.md`. Otherwise `git checkout pretrain.py`
   and add only `results RESULTS.md`. Ties: keep only if simpler.
   `git commit -m "autoresearch <NNN> <slug>: keep|discard, <metric>" && git push`.
7. Go to 1.

## Rules

- **Keep rule.** 3-seed mean strictly above the baseline's 3-seed mean. No noise threshold.
  One seed has sd ~44 at this budget (`SWEEP_s20.md`), so some keeps will be luck; that is
  accepted (Robbie, 2026-09-22). Always use seeds 0,1,2 so runs are comparable.
- **Simplicity.** All else equal, simpler is better. An improvement of ~0 with less code: keep.
  A small improvement with 20 lines of hacks: discard.
- **Crash / timeout counts as a failed experiment**, on any seed. If the cause is dumb (typo, import), fix
  and re-run under the same NNN. If the idea is what crashed (OOM, overran the budget), log it
  as `crash` or `timeout` and move on. Never edit the harness to make something fit.
- **No cheating.** No eval or RL code in pretrain.py. No touching the pools. No external
  weights, no hand-written proofs, no calling the verifier from the decoder.
- **Three pods per experiment, one experiment at a time.** Never more than 3 A40s at once. A
  pod named `autoresearch-klaus` may belong to another agent: leave it alone.
- **Hard stop.** At 07:00 America/Los_Angeles finish the running experiment, record it, then
  `python3 pod.py list` and `python3 pod.py down --pod <name>` for every pod you own. Pods cost
  money while you think.
- **Tags** (multi-select): systems, optimizer, schedule, architecture, tokenizer, data-order,
  augmentation, loss, init, regularization, batch, simplification.
- **NEVER STOP** to ask whether to continue. If out of ideas, re-read the fork's code, combine
  near-misses, try something radical.

## Files per experiment: `results/<NNN>_<slug>/`

- `params.json`: frozen before the run. `params.py new` fills the mechanical part; you fill the rest.
- `pretrain.patch`: `git diff HEAD -- pretrain.py` at pre-registration. The code IS the config;
  this is the ground truth for every run, kept or discarded.
- `results.json`: written by `record.py` from `summary.json` after the run.

### params.json (agent-filled fields marked `<...>`)

```json
{
  "id": "003", "slug": "wider-ffn", "written_at": "...", "base_commit": "<HEAD when written>",
  "provider": "<anthropic|openai>", "model": "<exact model id you run as, e.g. claude-fable-5-1, gpt-6-astra>",
  "harness_sha": "...", "pretrain_sha": "...", "patch_lines": 12,
  "title": "<one line>",
  "description": "<two sentences: what changes and why it should help expert iteration>",
  "tags": ["<architecture>"],
  "hypothesis": "<what mechanism would make this work>",
  "p_keep": 0.35, "expected_delta": "<+3 to +8 theorems at >=7>",
  "baseline_id": "<001>", "baseline_metric": 67,
  "config": {
    "arch": {"type": "gpt (fork model.py)", "n_layers": 4, "d_model": 256, "n_heads": 8, "d_ff": 1024,
             "pos": "rope", "ctx": 1024, "norm": "layernorm", "act": "gelu", "dropout": 0.0, "tied_emb": false,
             "n_params": 3210240},
    "tokenizer": {"codec": "abs", "maxn": 64},
    "optimizer": {"name": "adamw", "lr": 1e-3, "schedule": "cosine over wall-clock", "warmup_steps": 200,
                  "min_lr_frac": 0.1, "wd": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0},
    "batch": {"size": 256, "padding": "to max len in batch", "buckets": false},
    "data": {"filter": null, "augmentation": "random ref shift (abs)", "order": "shuffle", "val_slice": 1024},
    "precision": "bf16 autocast", "compile": false, "init": "fork default (normal 0.02)", "seed": 0
  },
  "protocol": "<mechanical: budget, pool sizes and proof-length histograms, RL and eval settings>"
}
```

`config` describes what your `pretrain.py` builds. Every key in the template must be filled;
add keys when you add a knob. `n_params` is cross-checked against the harness after the run.

### results.json (written by record.py, you only supply status and surprise)

`commit`, `recorded_at`, `status` (keep|discard|crash|timeout), `metric`, `metric_rl`, `rules`,
`gpu`, `total_secs`, `reason`, `pretrain` (steps, epochs, last_loss, heldout_greedy, secs,
n_params, `curve` = [{secs, step, train_loss, val_loss}, ...]), `rl_rounds` (per round:
targets_solved, sample_acc, lstar, ge histogram, new_proofs, samples_per_sec), `eval` (solved, n,
ge, lstar, heldout_greedy, by_bin), `holdout_metric` (null; scored offline), `surprise`.
