# Autoresearch: test-time thoughts

You are an autonomous researcher. The model is a small transformer that writes natural deduction
proofs in Lean form, and it can take **continuous thoughts** before each proof line: extra
positions whose input is its own previous output, not a token. Your job is to make **spending
more thoughts at test time solve more theorems**. Complete one experiment per session; `loop.py`
starts fresh sessions until cancelled.

This loop is the sibling of the test-time-loops loop (branch `autoresearch-loops`, ids from 500),
which asks the same question of a looped, weight-shared model. Both use the same frozen
`harness.py` apart from three constants (the knob's name and the two levels it is evaluated at),
so their scores are comparable. This loop's ids start at 700. Robbie's instructions for this loop
are in this file.

## Where things stand

Read [the spike that started this](../coconut_spike_20261006/README.md) first; it is short. On
2026-10-06 six ways of training one block of thoughts between the theorem statement and the proof
went through the old harness (4 rounds of expert iteration the agent could not change):

- Before RL, thoughts with feedback helped: 12.9% to 19.9% of the pretrained model's samples
  proved 7 to 14 line targets, against 7.6% for a control whose thoughts carry nothing forward.
  The best two arms supervised the thoughts with a plan (the proof's inference rules).
- After 4 RL rounds the gain was gone: 617 with feedback, 610 for the control. Plan-supervised
  arms fell furthest, and the old harness's RL could not supervise thoughts at all.
- After RL, solved counts were flat in the test-time thought count, 0 through 8, in every arm.
  The models had stopped using their thoughts.

The looped spike found the same shape (`../looped_spike_20261006/README.md`). Nobody has yet
made extra test-time computation pay after RL. That is the open problem.

What is new here, and why it might go differently:

1. **Thoughts before every proof line**, not one block up front. A line starts after a trigger
   token (`;`, or `by`, which also ends the theorem statement). The decisions in a proof are per
   line: which rule, which earlier lines.
2. **RL is yours**, so thoughts can be supervised during RL. The incumbent keeps a plan probe on
   in both stages: at each thinking round a training-only head must read the coming line's
   "move" (the token after its `:=` or `exact`) off the last thought.

The first run of this recipe (700, which then crashed in RL for lack of GPU memory; 701 is the
same recipe with the fine-tune batch run in four length-sorted parts) already showed the pre-RL
effect per line. The pretrained model's share of valid samples on the RL targets, 8 samples per
target, three seeds: 3.6% at 0 thoughts per line, 9.3% at 2, 11.1% at 4, 12.0% at 8, rising on
every seed up to at least 4. Whether any of that survives RL is what this loop is for.

**701's result, the first under this harness: 531 at 8 thoughts per line, 479 at 0** (seeds
510/466, 529/476, 553/496). Thinking is worth 44 to 57 dev theorems on every seed after three RL
rounds, which the spike never saw after RL. In RL sampling the whole gain is the step from 0 to
2 thoughts (after round 1: 30.0%, 36.7%, 37.0%, 36.7% of samples valid at 0, 2, 4, 8) and more
thoughts than 2 add nothing. So thoughts matter here, but more thoughts do not yet buy more.

**The bar.** The incumbent's 3-seed mean `metric`, starting from 701's 531. For scale, the
sibling loop's baseline (500, looped trunk, the same harness) scores 675 at 8 loops and 657 at 2,
with about 5,000 pretraining steps and five RL rounds where this recipe gets about 1,550 and
three; its target is 700. Run 198, untied and with no thoughts, scored 680 under the old harness.
Closing that 145-point gap is mostly a matter of speed; making 8 thoughts beat 2 is the science.

## The metric

`metric` in `summary.json`: dev-transfer theorems (true length >= 7, 1,108 of them) solved with
EVAL_K = 64 attempts each, **with the harness setting `model.n_thoughts = EVAL_LEVEL = 8`**
(thoughts per proof line). Higher is better. `metric_diag` is the same eval at 0 thoughts; it is
recorded so you can see what thoughts add, and it is not scored. A model that is as good with no
thoughts as with 8 does not answer the question, even if it scores well: say so in your
`surprise` text when `metric` and `metric_diag` are within noise of each other.

Robbie's words on the sibling loop apply here too: "the whole point is to see if we can spend
more compute for more results."

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
  samples per target, the thought counts you sample at, which proofs you keep, weighting, replay,
  the fine-tune optimizer and schedule, the loss. Three things are fixed, and the harness enforces
  the first two by what it hands you:
  1. **Same data.** `train_recs` and the 1,500 target statements. No new theorems, no reference
     proofs for targets (you are not given any), no other pools.
  2. **Expert iteration.** The only new training data is what `oracle.sample` returns: the
     model's own attempts that Lean AND nd_verify accepted. No search procedures (beam, MCTS),
     no calling the verifier or the fork's sampler yourself, no learning from rejected attempts'
     text (you are not given it).
  3. **Compute-limited.** The RL budget above.
- The model must expose an integer `n_thoughts` that fixes the thoughts per line in eval mode; the
  harness sets it to 8 for the scored eval and 0 for the diagnostic. In `rl` you choose it before
  each oracle call. What a "thought" is, where the thinking happens and how many positions a
  level stands for are the recipe's to define, as long as a higher level means more test-time
  computation and level 0 means none.
- Frozen data and verifier come from Dan's fork at the commit pinned in `remote.sh`.
- PARAM_CAP is 10M unique parameters, checked after pretraining and again after RL. The
  incumbent uses 9.56M (run 198's six untied blocks), so there is little room; training-only
  modules kept off the model, like the plan probe and the MTP head, do not count.

## The control that sets the real bar (2026-10-07, 16:10 PT)

Run 198's untied six blocks with no thoughts at all score **792** (790 / 785 / 800) in this
harness when given the sibling loop's current RL recipe (its experiment 590, an operator-run
control, recorded there as a discard so it is not their incumbent). Our incumbent 704 is nearly
the same network (pause slots it does not use) with the old RL recipe, and scores 676. So about
115 points are sitting in `rl()` alone, and they have nothing to do with thoughts.

That recipe, from `pretrain.py` on branch `autoresearch-loops` (functions `finetune` and `rl`;
read it with `git fetch origin autoresearch-loops && git show
origin/autoresearch-loops:code/experiments/current/autoresearch_loops/pretrain.py`):

- each round samples every target 8 times, then retries only the failures 24 more times;
- the fine-tune runs in token-budget batches like pretraining, about 2.5 times faster than
  padded batches of 128, so 11 RL rounds fit in 1,800 s (RL solved 1,311 targets);
- their 509 also fine-tunes with Muon.

It is architecture-independent and yours to lift. Two consequences:

1. Porting it is a legitimate experiment and probably the largest single gain available. It also
   makes every later comparison fairer to thoughts with feedback, whose main handicap has been
   how few RL rounds they fit.
2. "Thoughts help" now has an outside bar: a thinking recipe has to beat 792 with the same
   `rl()`, not just beat its own score at 0 thoughts.

## Ideas to start from

The incumbent (experiment 701): n thought positions after every trigger, all made from the
model's output at the trigger and run in one parallel pass; n drawn from 0..8 per training batch;
a plan probe at weight 0.3 in pretraining and RL; RL samples each target 8 times at each of 0, 2,
4 and 8 thoughts per round. Untried, roughly in the order Robbie and the spike suggest:

- **Train without the zero-thought batches.** Drawing n = 0 teaches the model to write proofs
  with no thoughts, which may be why thoughts ended up unused. Train and sample at a fixed n, or
  only at n >= 1, and see whether `metric_diag` falls away from `metric`.
- **Sequential thoughts.** Here a line's n thoughts are computed in one pass from one state.
  Coconut chains them: thought t + 1 is made from thought t's output. That is n passes per line
  and slower; measure it.
- **What the probe reads.** The move token is one choice. Others: the line's rule, the names it
  cites, the formula it derives, the number of lines left; or predict the model's own later
  state. Or no probe at all: is the supervision what makes thoughts matter?
- **Use thoughts in RL.** Sample only, or mostly, at 8; keep or upweight proofs found with
  thoughts and not without; raise the thought count for targets that resist.
- **Where to think.** Only at top-level lines, only at nested `by`, only when the model is
  unsure (entropy of the trigger's next-token distribution), or an emitted think token.
- **Thinking at more depth.** Put the thoughts through extra passes of the blocks (the looped
  loop's idea, applied only at thought positions, where it is cheap).
- **The replay mix.** The 20,000 replayed pretraining proofs have at most 6 lines and may pull
  the model toward not needing thoughts.
- **Speed.** A training step runs one pass per line, several times slower than run 198's single
  pass, so pretraining gets far fewer steps in its 300 s. Anything that buys steps back (rows
  batched by line count, fewer passes, compile) is a legitimate experiment.
- The pretraining idea list is at `.claude/notes/ideas.md`.

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
   number after the highest existing one; numbering starts at 700) with the mechanical fields and
   `pretrain.patch` filled in, and leaves `TODO`s. Fill every TODO: title, description,
   hypothesis, prediction, and `config`, which includes `thoughts` and `rl` (what your file actually
   does; read it, do not guess). If another agent kept a new `pretrain.py` while you were editing,
   it refuses: re-apply only your idea on top of the new HEAD file and rerun it. Committing the
   prediction BEFORE the run is enforced; params.json is never edited after that commit.
4. Run on three seeds at once, one pod each:
   `source "$(git rev-parse --show-toplevel)/.envrc.local" && $PY run_idea.py <NNN>-<agent>_<slug> --seeds 0,1,2`.
   A run takes roughly 70 to 90 minutes (5 pretraining, about 35 RL including Lean, 15 to 25
   eval, plus pod boot); the harness kills it at 5,400 s. The incumbent's measured speed on an
   A40: 5.6 training steps/s (about 1,650 steps in the 300 s, against run 198's 5,363) and, on a
   60-second model whose samples all run to the token limit, 43 samples/s at 8 thoughts. It writes the merged `artifacts/autoresearch_thoughts/<name>/summary.json`
   (`metric` = mean over seeds, `seed_metrics` per seed; status ok only if every seed is ok).
5. Record: `$PY record.py <NNN>-<agent>_<slug> keep|discard|crash|timeout "surprise"`. It writes
   `results.json`, refreshes `RESULTS.md`, applies the decision to `pretrain.py`, and
   commits/pushes. A keep replaces `pretrain.py` whole; otherwise it returns to the incumbent.
6. Choose `keep` only if the idea's mean `metric` is above the current incumbent's mean `metric`
   (both 3-seed means). Ties: keep only if simpler. Otherwise `discard`.
7. Verify your pods are terminated, then exit. The launcher starts the next experiment.

Measure before registering anything that may change speed:
`$PY bench_pretrain.py --secs 60` rents one A40 for about 4 minutes, runs your working-tree
`pretrain()` for 60 s and prints steps/s, memory and sampling speed at the default thought count.
It does not run `rl`. For the RL budget, reason from the incumbent's per-call timings in
`results/*/results.json` (`rl_rounds`: one entry per oracle call, with `level` (the thought
count), `k`, `samples_per_sec` and `rl_secs_used`).

## Rules

- **Keep rule.** 3-seed mean strictly above the incumbent's 3-seed mean. No noise threshold. One
  seed has a standard deviation of roughly 10 to 25 here, so some keeps will be luck; that is
  accepted. Always use seeds 0,1,2.
- **Simplicity.** All else equal, simpler is better. Roughly equal score with less code: keep. A
  small gain with 20 lines of hacks: discard. Code whose only purpose is speed does not count
  against a change.
- **Crash / timeout counts as a failed experiment**, on any seed. If the cause is dumb (typo,
  import), fix and re-run under the same NNN. If the idea is what crashed (out of memory,
  overran a budget), log it as `crash` or `timeout` and move on. Never edit the harness to make
  something fit. Thoughts need sampling memory: each block's KV cache holds the sampler's tokens
  plus `n_thoughts * MAX_THINKS` thought positions per sample, about 1.5 GB per block at 8
  thoughts and the default sampling batch of 1,024. The eval always uses that batch at level 8,
  so a recipe that makes level 8 much bigger has to fit there; in `rl` you may pass a smaller
  `batch` to the oracle.
- **No cheating.** No eval code in pretrain.py. No touching the pools. No external weights, no
  hand-written proofs, no calling the verifier or sampler outside the oracle, no reading target
  or dev proofs from disk.
- **Three pods per experiment, one experiment at a time per agent.** Pods are named
  `autoresearch-ndrl-thoughts-<agent>-<NNN>-s<seed>-...`; never touch another agent's pods, or
  the other loops' (`autoresearch-ndrl-loops-...`, `autoresearch-ndrl-pretrain-...`).
- **Cleanup.** `run_idea.py` terminates its pods after running. Each pod also deletes itself after
  130 minutes.
- **Tags** (multi-select): rl, sampling, thoughts, loss, architecture, optimizer, schedule, replay,
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
