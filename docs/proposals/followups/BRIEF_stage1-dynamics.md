# Run brief: stage1-dynamics — does Stage-1 saturate on long proofs, and is the schedule right?

Run id `stage1-dynamics`. Repository `~/work/stage1-dynamics` (branch `dan_stage1-dynamics`, a
worktree from the fork's `origin/dan`). Role: executor. `AGENT_POLICY.md` (nd-rl canonical copy)
governs; designs are suggestions, the questions are what matter. Pod budget **$15**, ceiling **30
pod-hours**, hard stop 30 h. Balance floor $130 — check `rpbalance` before extending.

Sibling run `lean-judge` is working on the fork's `dan` branch at the same time (the Lean-only
judge: `lean_gate.py`, `lean_judge.py`, and the `nd_verify` imports in `eval_set.py`,
`expert_iter.py`, `ladder_ei.py`, `coverage.py`, `eval_targets.py`, `grpo.py`). **Do not edit those
files**; you need only `train.py` and your own scripts, so the two branches merge cleanly.

Read first: `~/nd-rl/experiment-summaries/2026-09-25-noise-floor-measurement-floor/README.md`,
`NOISE_FLOOR.md`, and `train.py` (the validation loss at lines ≈ 93 and 114–121; the schedule
at ≈ 96).

## Why

Dan looked at the training curves of the 52 noise-floor Stage-1 models
(https://claude.ai/artifact/G8DYSKB8UeE8HmQMAXwkwm) and asked whether we train to saturation. What
was found:

- **`train.py`'s validation loss only measures 2- and 3-line proofs.** It takes the first 2,000
  records of `data/p2/heldout.jsonl`, which is sorted by length (1,000 per length 2–6; the 500-proof
  depth-3 slice is half of the 6-line bin). Every validation curve in the project has measured only
  proofs that every model already solves at 0.99+.
- On that slice the curve is flat by the end (−0.001 over the last 2,000 steps), train falls below
  validation from ≈ step 1,500, and the gap widens to 0.019 by step 6,000 (≈ 4.95 epochs).
- **Final validation loss varies by 1.4 % across the 52 models while 6-line accuracy spans 0.42–0.92
  and the depth-3 slice 0.008–0.918**; correlation ≈ 0. The loss never sees where capability varies.
- The noise floor is ≈ 95 % individual training run, so comparisons *across* runs at n = 2 resolve
  almost nothing on the 6-line bin and nothing on depth-3. Comparisons *within* a run (the same run
  at more steps) do not pay that noise.

## Questions

1. **Saturation.** Past 6,000 steps, do 6-line and depth-3 held-out accuracy keep rising, within
   the same run? Does validation loss on long proofs keep falling, flatten, or turn up (memorisation)?
2. **When is a run's depth-3 mode decided?** The depth-3 slice is bimodal across seeds. Does a
   seed's trajectory fix its mode early, or do low-mode runs cross over with more training?
3. **Schedule.** At an equal 6,000 steps, does warmup-stable-decay (WSD) differ from the current
   cosine (1e-3 → 1e-4)? Prior: little difference at equal steps; say what you expect.
4. **More steps or more data?** Is any gain from longer training a gain from steps, or does it
   depend on repeating the same 155,000 proofs ~20 times?
5. **Is per-length validation loss a usable proxy for per-length accuracy** (unlike the 2k loss)?

## Instrumentation (commit first; the old behaviour must stay the default-compatible)

- Validation loss every 200 steps on the **full** held-out file, logged **per length bin (2–6) and
  for the depth-3 slice**, plus the old first-2,000 number under its own name (`val2k`) so past
  curves stay comparable. Measure the overhead and report it (estimate ≈ 6 % of training time).
- Save checkpoints every 1,000 steps (and at every decay branch point). Upload them as you go.
- A schedule flag (`--sched cosine|wsd`, `--decay_frac`) and `--resume` from a checkpoint (weights,
  optimiser, step, RNG) so decay branches can start from a stable-phase checkpoint.

## Arms (paired seeds: the same seed list in every arm)

- **C — cosine, 6k** (the current recipe, instrumented): reproduces the on-file curves.
- **W — WSD**: warmup 200, constant 1e-3, linear decay to 1e-4 over the last 20 %. One stable run
  to 24,000 steps per seed; decay branches from the stable checkpoints at 4,800 → **6k**, 9,600 →
  **12k**, and the main run's own decay 19,200 → **24k**. W-6k vs C is the equal-budget schedule
  comparison; W-6k / W-12k / W-24k are within-seed training-length comparisons.
- **F — WSD on fresh data, 24k**: the four noise-floor pools (`data/nf/train_p{1..4}.jsonl`,
  620,000 records; de-duplicate across pools by renaming class and report how many remain), so 24k
  steps is ≈ 5 epochs, the same repetition as the baseline. F-24k vs W-24k separates steps from
  repetition. Drop-first if the budget binds.

Seeds: 8 for C and W, as many as fit for F (≥ 4). Model, data (C and W: the control's set
`data/p2/train_depth3_f0_a1.jsonl`), cap 6, batch 128 and everything else as the control.

## Measurements

- The instrumented validation losses, every 200 steps, every run.
- **Greedy held-out accuracy (5,000, k = 1, T = 0) per length and on the depth-3 slice** at every
  saved checkpoint of W's stable runs (the trajectory — the heart of questions 1–2) and at every
  arm's final checkpoint. If full evaluations at every 1,000 steps do not fit, thin the trajectory
  to every 2,000 steps before cutting seeds; say what you did. Hold the sampler batch fixed.
- **Judging:** Lean decides (policy, 2026-09-27). If `lean_judge.py` is on `origin/dan` when your
  evaluations start, use it. Otherwise the existing path counts Lean ∧ `nd_verify`, which differs
  from Lean alone by ≈ 67 per million on held-out greedy (noise-floor gate measurement) — use it,
  label it, and re-judge the stored outputs with `lean_judge.py` once it lands (no GPU needed).
- No ladder, no coverage, no RL in this run.

## Analysis

Within-seed, paired comparisons first (W-6k → W-12k → W-24k on the same seed; W-6k vs C on the
same seed) — that is what avoids the cross-run noise floor. For depth-3, report per-seed
trajectories and the number of seeds in each mode at each step (with Wilson intervals), not means.
Quote `NOISE_FLOOR.md` for any cross-run comparison. Figures: (a) validation loss per length bin
over steps, C vs W vs F; (b) held-out accuracy per length and depth-3 over steps, one line per seed;
(c) per-bin validation loss against per-bin accuracy across checkpoints.

## Pre-registration (`preregistration/stage1-dynamics.md`, before the first pod)

Numbers for each question, with the falsifier: e.g. "long proofs are not saturated at 6k" is dead if
W-12k and W-24k are within ±2 pp of W-6k on the 6-line bin on ≥ 6 of 8 seeds; "the mode is set
early" is dead if ≥ 2 of 8 seeds change mode after step 6,000.

## Budget arithmetic (measure and correct)

One 6k-step run ≈ 21.5 min on an A40 (noise-floor logs); a full greedy held-out evaluation ≈ 1–2
min. C: 8 × 6k; W: 8 × (24k + 1.2k + 2.4k); F: ≈ 4–8 × 24k — ≈ 270k–460k steps, ≈ 16–27 GPU-hours
at one job per GPU, fewer with 2–3 concurrent jobs per GPU (measure). Register first:
`podbudget stage1-dynamics --set 30 15`. Drop order: F's seeds 5–8, then F, then the trajectory
thinned to every 2,000 steps, then W's seeds 7–8.

## Deliverables

`preregistration/stage1-dynamics.md`; `run_stage1_dynamics.md` (≤ 400 words + the three figures);
`numbers.md` § stage1-dynamics (every entry names its source file and the model); `log.md` dated;
`artifacts/sd/summary.json`; the `train.py` changes (default behaviour unchanged except the added
logging) ready to merge into `dan`; `STATUS.md` line `STAGE1-DYNAMICS DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/stage1-dynamics/{ckpts,artifacts}`; pods deleted;
`touch ~/runs/stage1-dynamics/executor.done`. Questions for Dan in `QUESTIONS.md` with your default.
