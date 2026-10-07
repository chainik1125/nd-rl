# Autoresearch: test-time loops

You are an autonomous researcher. The model is a small looped transformer that writes natural
deduction proofs in Lean form. Your job is to make **spending more loops at test time solve more
theorems**. Complete one experiment per session; `loop.py` starts fresh sessions until cancelled.

This loop was forked from the pretraining loop (`../autoresearch/`, 208 experiments) on
2026-10-07. That loop's results are history and a source of ideas; its scores are not comparable
with this one's. Robbie's instructions for this loop are in this file.

## Where things stand

Read [the spike that started this](../looped_spike_20261006/README.md) first; it is short. On
2026-10-06 five models were trained through the old harness (4 rounds of expert iteration that
always sampled at 2 loops) and then sampled at 1 to 24 loops:

- After 4 RL rounds, per-sample success on long proofs rose from 1 to about 4 loops and was then
  exactly flat out to 24 loops, whatever loop range the model had trained on.
- After 12 RL rounds even that was nearly gone: one loop did as well as four.
- Every looped model was sampled at 2 loops during RL, so no proof it trained on ever needed a
  third loop. The loss was only ever put on the last loop's output.

Nobody has yet made deep loops pay. That is the open problem.

**The bar** is the best of those models on this loop's metric:

| spike model (seed 0, old harness) | RL | dev solved at 2 loops | at 8 loops |
|---|---|---|---|
| looped, trained on 1-4 loops | 12 rounds, sampled at 2 loops | 690 | **700** |
| looped, trained on 1-4 loops | 4 rounds, sampled at 2 loops | 644 | 657 |
| looped, trained on 1-8 loops | 4 rounds, sampled at 2 loops | 602 | 622 |
| looped, trained on 1-16 loops | 4 rounds, sampled at 2 loops | 593 | 614 |
| looped, trained at a fixed 2 loops | 4 rounds, sampled at 2 loops | 666 | 460 |

Measured on 2026-10-07 from the spike's checkpoints with this loop's eval (whole dev pool, 64
attempts, temperature 0.8); the 2-loop column reproduces each run's recorded harness score. The
12-round model had about 40 minutes of RL on the GPU, more than RL_SECS allows here. Note how
little 8 loops add in every row: 10 to 21 theorems for three times the compute per sample.

Beat the best number in that table. Run 198 (the untied model the pretraining loop ended on) is
deliberately not the bar for now.

**Control, 2026-10-07 (experiment 590, operator-run, read it):** run 198's six untied blocks with
507's `rl()` unchanged score **792** (790/785/800) in this harness, where the looped incumbents
scored 732 (507) and 748 (509) at 8 loops. The untied model applies 6 blocks per token; the looped
eval applies 18. So the RL gains found here (502's retries, 507's token-budget fine-tune) are not
about loops: they help regular pretraining more, and at equal RL the looped trunk is about 45 to
60 theorems behind it at three times the sampling cost. Every kept experiment so far has left the
gap between 8 loops and 2 loops at about +18. The incumbent rule is unchanged (beat the looped
incumbent's 3-seed mean), but an experiment that raises the score through RL efficiency alone no
longer tells us anything new. What would: a looped model whose score at 8 loops is far above its
own score at 2, without 2 loops getting worse, and ultimately one that beats 792.

## The metric

`metric` in `summary.json`: dev-transfer theorems (true length >= 7, 1,108 of them) solved with
EVAL_K = 64 attempts each, **with the harness setting `model.n_loops = EVAL_LEVEL = 8`**. Higher is
better. `metric_diag` is the same eval at 2 loops; it is recorded so you can see what loops add,
and it is not scored. A model that is merely good at 2 loops and no better at 8 does not answer the
question, even if it scores well: say so in your `surprise` text when `metric` and `metric_diag`
are within noise of each other.

The eval is expensive on purpose (8 loops is 18 block applications per token). Robbie's words:
"the whole point is to see if we can spend more compute for more results."

## Setup

- `harness.py` is FROZEN. It owns the clocks, the parameter cap, the RL oracle and the eval. The
  constants at the top are copied into every params.json as `protocol`. You never edit it; its
  sha256 is written into every result and a changed harness invalidates the run.
- `pretrain.py` is the ONLY file you edit. It holds two functions, `pretrain` and `rl`; the
  contract for both is in its docstring.
  - `pretrain`: PRETRAIN_SECS = 300 s of wall-clock, compile and model construction included.
  - `rl`: RL_SECS = 1800 s, counted as wall-clock inside `rl()` minus the time the oracle spends
    in the Lean gate. Overrunning either budget past its grace is a timeout.
- **RL is yours to redesign.** The pretraining loop froze RL; this one does not. Change rounds,
  samples per target, the loop counts you sample at, which proofs you keep, weighting, replay,
  the fine-tune optimizer and schedule, the loss. Three things are fixed, and the harness enforces
  the first two by what it hands you:
  1. **Same data.** `train_recs` and the 1,500 target statements. No new theorems, no reference
     proofs for targets (you are not given any), no other pools.
  2. **Expert iteration.** The only new training data is what `oracle.sample` returns: the
     model's own attempts that Lean AND nd_verify accepted. No search procedures (beam, MCTS),
     no calling the verifier or the fork's sampler yourself, no learning from rejected attempts'
     text (you are not given it).
  3. **Compute-limited.** The RL budget above.
- The model must expose an integer `n_loops` that fixes the number of loops in eval mode; the
  harness sets it to 8 for the scored eval. In `rl` you choose it before each oracle call.
- Frozen data and verifier come from Dan's fork at the commit pinned in `remote.sh`.
- PARAM_CAP is 10M unique parameters, checked after pretraining and again after RL. The current
  looped trunk uses 6.4M.

## Ideas to start from

The incumbent already samples each target 8 times at each of 1, 2, 4 and 8 loops in every RL
round (experiment 500). Untried, roughly in the order Robbie and the spike suggest:

- Sample only, or mostly, at deep loop counts in RL; or sample at a loop count per target that
  rises with the target's length or with how often it has resisted solving.
- Keep, or upweight, proofs that were found at deep loops and not at shallow ones.
- Put the loss on every loop's output, not only the last (deep supervision), in pretraining, RL
  fine-tuning, or both; or train so the last loops refine a draft the first loops produce.
- Train loop counts that match the eval: more weight on 8 during fine-tuning, or a curriculum
  from shallow to deep across RL rounds.
- Change the replay mix: the 20,000 replayed pretraining proofs have at most 6 lines and may pull
  the model toward shallow computation.
- More, shorter RL rounds, or fewer, longer ones; a different fine-tune optimizer (pretraining
  uses Muon, RL fine-tuning uses AdamW at 3e-4, tuned for the untied model).
- Architecture, with RL free to follow: one shared block and more loops; untied norms per loop
  (P07); an iteration embedding (P08); gates per loop (P09); adaptive halting. The pretraining
  idea list is at `.claude/notes/ideas.md`.
- Use the spare parameters: the trunk is 3.6M under the cap.

Bold ideas are welcome. A null result that tests a clear mechanism is worth more than a nudge to
a tuned number.

## Loop

For the launcher, see [LOOP.md](LOOP.md). Its task prompt is [iteration.md](iteration.md).

All commands run from this directory. The scripts need the repo venv, so
`PY=$(git rev-parse --show-toplevel)/code/.venv/bin/python` and use `$PY` where `python3` is
written. Run `make -C code check` from the repo root before launching (Ruff + Pyright). Report
failures; do not fix unrelated files. The scripts perform Git operations with normal hooks enabled
and stop on commit/push failure. Do not stage, commit, push, or restore files manually.

1. Read `results/*/params.json` + `results.json` (every previous experiment), `RESULTS.md`, and
   `git log`.
2. Pick one idea. Edit `pretrain.py`.
3. Pre-register: `$PY params.py new <NNN> <slug> --seeds 0,1,2` writes
   `results/<NNN>-<agent>_<slug>/params.json` (it appends your `$ND_AGENT` to the id; use the next
   number after the highest existing one; numbering starts at 500) with the mechanical fields and
   `pretrain.patch` filled in, and leaves `TODO`s. Fill every TODO: title, description,
   hypothesis, prediction, and `config`, which includes `loops` and `rl` (what your file actually
   does; read it, do not guess). If another agent kept a new `pretrain.py` while you were editing,
   it refuses: re-apply only your idea on top of the new HEAD file and rerun it. Committing the
   prediction BEFORE the run is enforced; params.json is never edited after that commit.
4. Run on three seeds at once, one pod each:
   `source "$(git rev-parse --show-toplevel)/.envrc.local" && $PY run_idea.py <NNN>-<agent>_<slug> --seeds 0,1,2`.
   A run takes about 65 to 75 minutes (5 pretraining, about 35 RL including Lean, about 20
   eval, plus pod boot). It writes the merged `artifacts/autoresearch_loops/<name>/summary.json`
   (`metric` = mean over seeds, `seed_metrics` per seed; status ok only if every seed is ok).
5. Record: `$PY record.py <NNN>-<agent>_<slug> keep|discard|crash|timeout "surprise"`. It writes
   `results.json`, refreshes `RESULTS.md`, applies the decision to `pretrain.py`, and
   commits/pushes. A keep replaces `pretrain.py` whole; otherwise it returns to the incumbent.
6. Choose `keep` only if the idea's mean `metric` is above the current incumbent's mean `metric`
   (both 3-seed means). Ties: keep only if simpler. Otherwise `discard`.
7. Verify your pods are terminated, then exit. The launcher starts the next experiment.

Measure before registering anything that may change speed:
`$PY bench_pretrain.py --secs 60` rents one A40 for about 4 minutes, runs your working-tree
`pretrain()` for 60 s and prints steps/s, memory and sampling speed at the default loop count. It
does not run `rl`. For the RL budget, reason from the incumbent's per-call timings in
`results/*/results.json` (`rl_rounds`: one entry per oracle call, with `level` (the loop count), `k`,
`samples_per_sec` and `rl_secs_used`): sampling cost grows with loops, about 2.5 times slower at 8
loops than at 2.

## Rules

- **Keep rule.** 3-seed mean strictly above the incumbent's 3-seed mean. No noise threshold. One
  seed has a standard deviation of roughly 10 to 25 here, so some keeps will be luck; that is
  accepted. Always use seeds 0,1,2.
- **Simplicity.** All else equal, simpler is better. Roughly equal score with less code: keep. A
  small gain with 20 lines of hacks: discard. Code whose only purpose is speed does not count
  against a change.
- **Crash / timeout counts as a failed experiment**, on any seed. If the cause is dumb (typo,
  import), fix and re-run under the same NNN. If the idea is what crashed (OOM at deep loops,
  overran a budget), log it as `crash` or `timeout` and move on. Never edit the harness to make
  something fit. Deep loops need memory: one KV cache per block application, about 0.9 GB each at
  the default sampling batch of 1,024; pass a smaller `batch` to the oracle beyond about 16 loops.
- **No cheating.** No eval code in pretrain.py. No touching the pools. No external weights, no
  hand-written proofs, no calling the verifier or sampler outside the oracle, no reading target
  or dev proofs from disk.
- **Three pods per experiment, one experiment at a time per agent.** Pods are named
  `autoresearch-ndrl-loops-<agent>-<NNN>-s<seed>-...`; never touch another agent's pods, or the
  pretraining loop's (`autoresearch-ndrl-pretrain-...`).
- **Cleanup.** `run_idea.py` terminates its pods after running. Each pod also deletes itself after
  130 minutes.
- **Tags** (multi-select): rl, sampling, loops, loss, architecture, optimizer, schedule, replay,
  systems, init, regularization, simplification.
- Do not ask whether to continue within your assigned experiment. Exit after recording one
  experiment, or report a blocker.

## Files per experiment: `results/<NNN>-<agent>_<slug>/`

- `params.json`: frozen before the run. `params.py new` fills the mechanical part; you fill the
  rest. `config` describes what your `pretrain.py` builds and how its `rl` trains; every key in
  the template must be filled, and you add keys when you add a knob.
- `pretrain.patch`: `git diff HEAD -- pretrain.py` at pre-registration. The code is the config.
- `results.json`: written by `record.py` from `summary.json` after the run: `metric`,
  `metric_diag`, `metric_rl` (cumulative RL targets solved), per-seed values, the pretraining
  curve, one `rl_rounds` entry per oracle call, and the eval histograms.
