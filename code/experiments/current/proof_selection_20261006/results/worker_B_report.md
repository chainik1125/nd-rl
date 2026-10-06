---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Worker B report: teacher-forced likelihood scorers, threshold stopping, accuracy vs attempts

Branch `dmitry/selection-sprint` (worktree `/data/users/dmitry/ndrl-selection-sprint`), uncommitted. Generator frozen:
`la_T1_best12_s0_r16.pt`, T 0.8, `max_action` 512, `max_steps` 96, pools from Worker A (`~/runs/proof-selection/pools_v2/`,
seed 0). Timestamped decisions are in `../logs/worker_B.md`. Worker A's learned verifier scores (`verifier_*`, `shallow_lr*`)
were included under a second dev-only freeze (03:15 UTC); see §7.

## Summary

- **Scorer.** Teacher-forced log-probabilities of every emitted action token (and its real `<eos>`) under r16 at T 1,
  with the states recovered by replaying each attempt through `Env(prompt, canon=True, base, assign=True)`. Every one of
  57,536 attempts (1.06 M steps) across the four evaluation pools replays (plus 71,920 rl_targets attempts, also clean) with matching per-step state hashes, matching `done`, and
  byte-identical ND text; teacher-forced and the sampler's own cached decoding agree to bf16 noise. Masking the
  environment-assigned names matters: without it, pooled AUROC on textbook72 drops from 0.69 to 0.53.
- **Likelihood does carry recognition signal beyond completion, but only within a theorem.** Among finished attempts of
  one theorem, `loglik_mean` separates valid from invalid with AUROC 0.79 (holdout250) and 0.69 (textbook72). Pooled
  across theorems, the finished-only AUROC is 0.67 / 0.65. The two numbers do different jobs:
  - The within-theorem number is the one *selection* uses. Picking the highest-likelihood finished attempt beats picking
    a finished attempt at random by +0.017 to +0.040 at k = 8 to 16 on both test sets.
  - The pooled number is the one a *global threshold* sees, so likelihood-thresholded stopping gains little.
  On the hardest theorems (pass rate < 0.25), where the extra attempts matter, within-theorem finished-only AUROC is
  only 0.56 to 0.64, from a handful of theorems.
- **Stopping (H2, intrinsic scores only).** "Stop at the first finished attempt" is the strong baseline: holdout250
  0.904 at 1.6 attempts (fixed-k completion pick at the same cost: 0.865); textbook72 0.521 at 4.1 attempts. It saturates
  at 0.911 (2.6 attempts) and 0.549 (8.8 attempts).
  - Adaptive `completion_loglik` stopping, at the dev-chosen points, beats fixed-k with the same selector at matched
    mean attempts on holdout250 (+0.012 [0.004, 0.021] at 3.9 attempts).
  - Against first-finished it is +0.008 [-0.002, 0.018], and +0.014 [0.004, 0.025] when it spends 8.3 attempts against
    first-finished's 2.6.
  - On textbook72 no dev-chosen intrinsic policy separates from either comparator (CIs span 0).
  - So for intrinsic scores the H2 falsifier ("first-finished matches the learned rule") is essentially met at low
    budgets. Small gains appear only by spending well past first-finished's ceiling.
- **The learned verifier (Worker A) does not beat likelihood or first-finished.**
  - **Recognition (H3).** H3 predicted that the head's AUROC on finished candidates beats likelihood's. On *pooled*
    finished-only AUROC the prediction holds in direction, though it is significant only for the history variant
    (paired theorem bootstrap, verifier - loglik_mean):
    - verifier_ens: holdout250 +0.056 [-0.055, 0.178]; textbook72 +0.070 [-0.057, 0.195].
    - verifier_hist_ens: holdout250 +0.105 [0.009, 0.198]; textbook72 +0.167 [0.055, 0.281].
  - **Within-theorem, the result reverses.** This is the recognition test, because a theorem-difficulty score cannot
    rank a theorem's own candidates:
    - verifier_ens - loglik_mean: holdout250 -0.142 [-0.203, -0.080]; textbook72 -0.091 [-0.195, +0.007], not
      significant on 34 theorems.
    - So, as an interpretation, the head mostly learned theorem difficulty, not which of a theorem's own proofs is
      right. `results/auroc_paired_verifier_vs_loglik.jsonl`.
  - **Stopping, holdout250.** Dev-chosen verifier stopping is within ±0.004 of first-finished.
  - **Stopping, textbook72.** It is worse than first-finished: m = 4 point -0.024 [-0.051, -0.002] at 7.3 attempts.
    Three seeds agree.
- **The threshold is not cost-controlled when the distribution shifts.** The dev-chosen "m = 4" threshold uses 3.9
  attempts on holdout250 but 12.1 on textbook72, because textbook proofs are longer and have lower mean likelihood even
  when easy. Textbook theorems with pass rate around 1 still burn 8 to 16 attempts
  (`figures/stopping_textbook72_allocation.png`). First-finished, by contrast, does allocate by difficulty: about 1
  attempt on easy theorems and N_max on unsolvable ones.

## 1. What was run (commands)

All five pools were scored (rl_targets k16 too: 71,920 attempts, 94 s, all checks 0) so that Worker A has the
per-attempt features for every set.

All from `~/runs/proof-selection`, with `PY=~/venvs/proof-state/bin/python`,
`S=<sprint folder>`, `export PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 ND_SAMPLE_PATH=fast CUDA_VISIBLE_DEVICES=7`.

```bash
# scorers (per pool; ~8-42 s each on one H200)
$S/code/run_scores.sh {textbook72__s0_r16__k64,dev1108__s0_r16__k16,holdout250__s0_r16__k64,rr600__s0_r16__k32}
#   = $PY $S/code/score_loglik.py --pool pools_v2/<stem>.jsonl --out_dir scores
$PY $S/code/validate_scorer.py pools_v2/textbook72__s0_r16__k64.jsonl scores 200     # results/validate_scorer_textbook72.jsonl
# dev: tau grid (union of dev-split quantiles of both dev pools), dev stopping, dev AUROC, exchangeability
$PY $S/code/stopping.py --pool pools_v2/dev1108__s0_r16__k16.jsonl --split dev --write_grid stopping/tau_grid_dev.json
$PY $S/code/stopping.py --pool pools_v2/rr600__s0_r16__k32.jsonl --split dev --grid stopping/tau_grid_dev.json --write_grid stopping/tau_grid_dev.json
for p in dev1108__s0_r16__k16 rr600__s0_r16__k32; do
  $PY $S/code/stopping.py --pool pools_v2/$p.jsonl --split dev --grid stopping/tau_grid_dev.json
  $PY $S/code/stopping.py --pool pools_v2/$p.jsonl --split dev --grid stopping/tau_grid_dev.json --orig_order
  $PY $S/code/auroc.py --pool pools_v2/$p.jsonl --split dev --out $S/results/auroc_${p%%__*}__dev.json
  $PY $S/code/exchangeability.py --pool pools_v2/$p.jsonl --out $S/results/exch_$p.json --fig $S/figures/exch_$p.png
done
# FREEZE logged 02:30 UTC, then test, read once
for p in holdout250__s0_r16__k64 textbook72__s0_r16__k64; do
  $PY $S/code/stopping.py --pool pools_v2/$p.jsonl --grid stopping/tau_grid_dev.json
  $PY $S/code/auroc.py --pool pools_v2/$p.jsonl --out $S/results/auroc_$p.json
  $PY $S/code/curve_table.py --prefix stopping/$p --scorers completion_loglik,completion_loglik_sum,completion_minstep,loglik_mean --md $S/results/curve_table_$p.md
done
$PY $S/code/stopping_headline.py --dev stopping/dev1108__s0_r16__k16__dev stopping/rr600__s0_r16__k32__dev \
   --test stopping/holdout250__s0_r16__k64 stopping/textbook72__s0_r16__k64 \
   --scorers completion_loglik,completion_minstep,completion_loglik_sum,completion_entropy,shortest,loglik_mean,loglik_sum,loglik_min_step,entropy_mean,loglik_mean_full \
   --out $S/results/stopping_headline.json
$PY $S/code/stopping_figs.py --prefix stopping/<stem> --title "..." --loglik completion_loglik --headline $S/results/stopping_headline.json --out $S/figures/stopping_<set>
$PY $S/code/report_tables.py completion,completion_loglik,completion_loglik_sum,completion_minstep,loglik_mean > $S/results/worker_B_tables.md
```

## 2. Scorer (`code/score_loglik.py`) and its validation

The scorer replays the stored attempt and checks the replay:

- **Replay.** Each attempt is replayed with its stored `base` and `assign=True`.
- **Model input.** At each step the input is `tok.encode_toks(env.state_tokens())` (= `state_sample.prompt_ids`, arm SN)
  followed by the raw sampled ids of the action. Sampled pads are kept; the action runs through the real `<eos>`, and no
  `<eos>` is added to a truncated action.
- **Checks.** Every step's state is checked against the collector's `state_sha1`. `done` and the assembled ND text are
  checked byte for byte.
- **Masking.** Positions masked from every aggregate are exactly those `Env._assign_names` overwrites: the name after
  `have`, the lambda binder of an implication/negation box, and the first `Or.elim` branch binder. Environment-inserted
  closures and the second-branch binder are never emitted by the model, so they are never scored. Citation names are
  kept.

Scorers written to `scores/<stem>__<name>.jsonl` (schema §3):

- `loglik_mean` (= `loglik_mean_masked`), `loglik_mean_full` (unmasked), `loglik_sum`, `loglik_min_step`,
  `entropy_mean` (negated).
- `completion_loglik` (done ? loglik_mean : -inf). Derived variants: `completion_loglik_sum`, `completion_minstep`,
  `completion_entropy`.

Per-attempt features for Worker A's `shallow_lr` baseline are in `scores/aux/<stem>__shallow_features.jsonl`.

| check | textbook72 k64 | holdout250 k64 | dev1108 k16 | rr600 k32 |
|---|---|---|---|---|
| attempts / steps scored | 4,608 / 78,051 | 16,000 / 258,109 | 17,728 / 291,109 | 19,200 / 429,479 |
| replay errors, done mismatches, bad state hashes | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| finished attempts with byte-identical ND | 2,332 / 2,332 | 14,092 / 14,092 | 15,466 / 15,466 | 13,521 / 13,521 |
| wall time (replay + forward) | 7.6 s | 24 s | 27 s | 42 s |

Agreement with the sampler (`results/validate_scorer_textbook72.jsonl`, pool and scores only):

- **First action token.** Teacher-forced vs the sampler's left-padded cached prefill: mean |Δ log p| 6e-5, max 0.005
  (top attempts).
- **Every action token.** Teacher-forced vs the sampler's cached one-token-at-a-time path, on 600 steps each of done
  and failed attempts: mean |Δ| 3e-4 / 5e-4; 0.1 % of tokens exceed 0.05; max 0.74 on one failed-attempt token (bf16).
- **Greedy re-decode.** For the 200 highest-likelihood finished attempts, it reproduces the stored action outside
  masked positions in **99.8 %** of steps (89.2 % byte-exact including names). For 200 random finished attempts the
  figure is 87.4 %.

## 3. Stopping simulator (`code/stopping.py`)

The policy follows schema §5:

- **Rule.** Consume attempts in order, stop when the best score so far ≥ τ or at N_max, and return the best-scoring
  attempt so far.
- **Frozen conventions.**
  - Ties go to the earliest consumed attempt; if every score is -inf, the first consumed attempt is returned.
  - R = 50 seeded random orderings per theorem, with accuracy averaged per theorem and then over theorems.
  - The `random` selector is computed exactly (pass rate).
  - N_max ∈ {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}.
  - τ grid = 0..100 % quantiles of each scorer's finite scores on the dev split of dev1108 and rr600 only
    (`stopping/tau_grid_dev.json`, sha256 f159864d…).
- **Policies.**
  - Fixed-k (k = 1..K, the same selector).
  - "Stop at first finished" (completion score, τ = 1, N_max swept 1..K).
  - Oracle (stop at first valid; its fixed-k is pass@k).
  - Fixed-k comparator = the random mixture of adjacent k at the policy's mean attempts.
- **Invariants asserted on every run.**
  - All selectors coincide at k = 1.
  - τ = +inf equals fixed N_max; τ = -inf equals fixed 1 (all with max diff 0).
  - Random fixed-k equals the per-attempt pass rate.
  - Oracle fixed-k is within 0.006 of the unbiased pass@k estimator.
- **Costs.**
  - Attempts.
  - Generated tokens: pool `gen_tokens`, including failed attempts and the eos.
  - Total tokens = prefill + generated + the scorer's own forward pass. For the likelihood family this is one forward
    over every (state, action) pair, counted conservatively although it could be read off the sampling logits for
    free. Completion, random and oracle cost 0 beyond generation.
- **Uncertainty.** Paired theorem-level bootstrap, B = 1000. Displacement (a valid incumbent replaced by a
  higher-scoring invalid later attempt) is tracked per policy.

**Headline protocol.**

- **Choose on dev.** For each policy and target m ∈ {2, 4, 8}, choose (N_max ∈ {8, 16}, τ) on the dev split (both dev
  pools, theorem-weighted) so that dev mean attempts are nearest m. The completion-stop policy instead chooses N_max.
  Freeze, then apply once to test.
- **Report.** The realised test cost, compared with:
  - (a) fixed-k same selector at the realised mean attempts (the fair, cost-matched comparison);
  - (b) "stop at first finished" at the N_max whose mean attempts is nearest;
  - fixed-k at the dev target m, which is in the JSON but is cost-mismatched on textbook72.
- **Curve readings.** These are separate tables (`results/curve_table_*.md`). They read the upper concave frontier of
  all (N_max, τ) points on the test pool itself, so they are optimistic best-of-family readings (two free parameters
  read on test), not dev-selected results.

## 4. Headline: accuracy at mean attempts ≈ 2, 4, 8 (test)

The dev-chosen operating points are in `results/worker_B_tables.md` (generated from `results/stopping_headline.json`;
all scorers are in the JSON). Condensed below. "fixed-same" = fixed-k with the same selector at the realised mean
attempts; "FF" = stop-at-first-finished; brackets are 95 % CIs.

**holdout250** (250 theorems, k = 64; per-attempt validity 0.836; pass@64 0.96)

| target m | fixed-k random | fixed-k completion pick | first-finished (dev N_max) | adaptive completion_loglik (dev τ) | oracle pass@k @ same att |
|---|---|---|---|---|---|
| 2 | 0.836 | 0.880 (@2) | 0.904 @ 1.61 att | 0.905 @ 1.70 (= FF; τ below all finished scores) | 0.877 |
| 4 | 0.836 | 0.894 (@4) | 0.905 @ 1.70 (ceiling 0.911 @ 2.62 with N_max 64) | **0.919 @ 3.93**; vs fixed-same +0.012 [0.004, 0.021]; vs FF +0.008 [-0.002, 0.018] | 0.925 |
| 8 | 0.836 | 0.901 (@8) | (unreachable: FF ceiling 0.911 @ 2.62) | **0.925 @ 8.26**; vs fixed-same +0.007 [0.002, 0.012]; vs FF +0.014 [0.004, 0.025] | 0.940 |

**textbook72** (72 theorems, k = 64; per-attempt validity 0.407)

| target m | fixed-k random | fixed-k completion pick | first-finished (dev N_max) | adaptive completion_loglik (dev τ) | oracle pass@k @ same att |
|---|---|---|---|---|---|
| 2 | 0.407 | 0.459 (@2) | 0.521 @ 4.11 att (dev N_max 13) | 0.526 @ 4.56 (≈ FF) | 0.542 |
| 4 | 0.407 | 0.495 (@4) | 0.526 @ 4.56 (ceiling 0.549 @ 8.81) | 0.563 @ **12.08**; vs fixed-same +0.005 [-0.003, 0.014]; vs FF ceiling +0.015 [-0.027, 0.051] | 0.619 |
| 8 | 0.407 | 0.512 (@8) | (FF ceiling 0.549 @ 8.81) | 0.566 @ 14.51; vs fixed-same +0.002 [-0.001, 0.006]; vs FF +0.018 [-0.026, 0.055] | 0.627 |

Curve readings at exact m (optimistic frontier on the test pool; `results/curve_table_*.md`):

| set | policy | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| holdout250 | fixed-k random | 0.836 | 0.836 | 0.836 |
| holdout250 | fixed-k completion pick | 0.880 | 0.894 | 0.901 |
| holdout250 | stop at first finished (N_max swept) | 0.908 | – (ceiling 0.911 @ 2.62) | – |
| holdout250 | fixed-k completion_loglik pick | 0.889 | 0.908 | 0.918 |
| holdout250 | adaptive completion_loglik (frontier) | 0.912 | 0.922 | 0.929 |
| holdout250 | adaptive loglik_mean, no completion (frontier) | 0.882 | 0.895 | 0.902 |
| holdout250 | oracle pass@k / oracle stop-at-first-valid | 0.900 / 0.948 | 0.925 / 0.960 | 0.939 / – |
| textbook72 | fixed-k random | 0.407 | 0.407 | 0.407 |
| textbook72 | fixed-k completion pick | 0.459 | 0.495 | 0.512 |
| textbook72 | stop at first finished (N_max swept) | 0.486 | 0.520 | 0.547 |
| textbook72 | fixed-k completion_loglik pick | 0.464 | 0.510 | 0.544 |
| textbook72 | adaptive completion_loglik (frontier) | 0.487 | 0.532 | 0.563 |
| textbook72 | adaptive loglik_mean, no completion (frontier) | 0.460 | 0.491 | 0.500 |
| textbook72 | oracle pass@k / oracle stop-at-first-valid | 0.477 / 0.506 | 0.540 / 0.587 | 0.593 / 0.635 |

On tokens, with the same dev-chosen points (`stopping_headline.json`: `test_gen`, `test_tot`):

- **holdout250.** First-finished 504 generated / 3.2 k total tokens per theorem; adaptive completion_loglik at m = 4
  1,086 / 13.8 k. The total includes a conservative teacher-forced forward pass that would be free if read off the
  sampling logits.
- **textbook72.** First-finished 1,222 / 8.7 k; adaptive completion_loglik at m = 4 3,218 / 46 k.

Figures `figures/stopping_{holdout250,textbook72}_acc_vs_cost.png` show three panels (mean attempts, generated tokens,
total tokens), with stars at the dev-chosen points. The per-theorem allocation plots are
`figures/stopping_{set}_allocation.png`; the dev versions are `figures/stopping_dev_{dev1108,rr600}_*`.

Displacement: a valid incumbent replaced by a higher-scoring invalid attempt in 1.0 % of runs (holdout250, m = 4) and
3.6 % (textbook72, m = 4) for completion_loglik. First-finished never displaces.

Mean attempts = 1 is the same mandatory first attempt for every policy, so no policy can beat fixed-1 there (H2's
"every m in [1, 8]" is only testable for m > 1).

## 5. Diagnostics: AUROC for "valid" (`code/auroc.py`, full table in `results/worker_B_tables.md`)

| set | scorer | pooled, all attempts | pooled, finished only | within-theorem, all | within-theorem, finished only [95 % CI] (n thm) |
|---|---|---|---|---|---|
| holdout250 | completion | 0.863 | 0.500 | 0.807 | 0.500 (116) |
| holdout250 | loglik_mean (masked) | 0.776 | 0.669 | 0.860 | 0.788 [0.755, 0.829] |
| holdout250 | loglik_mean_full (unmasked) | 0.755 | 0.633 | 0.835 | 0.725 [0.678, 0.768] |
| holdout250 | loglik_sum | 0.754 | 0.685 | 0.809 | 0.801 [0.761, 0.833] |
| holdout250 | loglik_min_step | 0.681 | 0.629 | 0.750 | 0.743 [0.699, 0.783] |
| holdout250 | entropy_mean | 0.781 | 0.645 | 0.769 | 0.682 [0.634, 0.731] |
| holdout250 | completion_loglik | 0.909 | 0.669 | 0.911 | 0.788 |
| holdout250 | shortest (A's baseline) | 0.880 | 0.560 | 0.862 | 0.642 [0.584, 0.692] |
| textbook72 | completion | 0.916 | 0.500 | 0.839 | 0.500 (34) |
| textbook72 | loglik_mean (masked) | 0.688 | 0.645 | 0.740 | 0.687 [0.596, 0.769] |
| textbook72 | loglik_mean_full (unmasked) | 0.527 | 0.481 | 0.707 | 0.658 [0.563, 0.751] |
| textbook72 | loglik_sum | 0.787 | 0.756 | 0.694 | 0.652 [0.571, 0.737] |
| textbook72 | loglik_min_step | 0.684 | 0.696 | 0.652 | 0.646 [0.552, 0.747] |
| textbook72 | entropy_mean | 0.735 | 0.659 | 0.756 | 0.685 [0.595, 0.777] |
| textbook72 | completion_loglik | 0.941 | 0.645 | 0.900 | 0.687 |
| textbook72 | shortest | 0.952 | 0.714 | 0.855 | 0.537 [0.441, 0.633] |

Dev split (dev1108 / rr600), within-theorem finished-only AUROC: loglik_mean 0.787 / 0.765, loglik_sum 0.801 / 0.750,
entropy 0.722 / 0.727.

By difficulty bucket (`results/auroc_buckets_summary.txt`, loglik_mean, finished only, within-theorem; the n of
theorems with both classes is small):

| set | pass rate < 0.05 | 0.05–0.25 | 0.25–0.5 | 0.5–0.9 | 0.9–1 |
|---|---|---|---|---|---|
| holdout250 | 0.81 (n = 2) | 0.64 (7) | 0.76 (7) | 0.74 (40) | 0.84 (60) |
| textbook72 | 0.56 (n = 4) | 0.64 (8) | 0.61 (7) | 0.78 (7) | 0.78 (8) |

P(valid | finished) in those buckets: holdout250 0.08 / 0.31 / 0.77 / 0.91 / 0.99; textbook72 0.04 / 0.40 / 0.70 /
0.92 / 0.99.

**Answer to "does likelihood carry recognition signal beyond completion?"** Yes, within a theorem: finished-only AUROC
is 0.69 to 0.80, far above completion's 0.5, and it turns into a real selection gain. It is weaker on the hard
theorems, where the recognition headroom is, and it is a poor *global* signal. Pooled finished-only AUROC is 0.65 to
0.69, and the absolute likelihood scale shifts with proof length between sets, which is why global-threshold stopping
gains little. `loglik_sum` has higher pooled finished-only AUROC on textbook72 (0.756) (interpretation: it prefers short
proofs, which is a between-theorem length cue; its within-theorem AUROC is 0.65).

## 6. Exchangeability (reviewer item 12; `results/exch_*.json`, `figures/exch_*.png`)

Validity minus the theorem's pass rate, regressed on the SN base and on attempt index:

| pool | slope on base (z) | slope on index (z) | duplicate action streams |
|---|---|---|---|
| dev1108 | -0.2 | 0.7 | 3 |
| rr600 | 1.8 | -0.05 | 7 |
| holdout250 | 1.2 | 2.0 (+0.0002 per index, i.e. +0.014 over 64) | 16 |
| textbook72 | -0.1 | -1.4 | 6 |

Duplicates are identical short proofs, not a sampling artefact. Original-order vs shuffled (rr600 dev): first-finished
accuracy with the first 16 attempts is 0.794, with the last 16 it is 0.717, and shuffled gives 0.752, with identical
finished and valid rates in both halves. That spread is the noise of a single ordering, not an order effect. No
evidence against random-order averaging, beyond one borderline z = 2.0 among eight tests.

## 6b. Exploratory (post-hoc): count-stop, a threshold-free stopping rule

Designed after the test read of the threshold policies, because their absolute threshold did not transfer. Rule: stop
after the c-th finished attempt (or N_max = 16) and return the best so far by `completion_loglik`; c = 1 is
first-finished. On dev it looked good (dev1108 c = 3: 0.922 at 4.0 attempts vs fixed-same 0.908; rr600 c = 2: 0.783 at
4.9 vs 0.736). It was frozen and then read once on test (`code/stopping_count.py`, `results/countstop_test.{txt,json}`).

| set | c | test att | acc | fixed-same @ att | vs fixed-same | first-finished @ nearest att | vs first-finished |
|---|---|---|---|---|---|---|---|
| holdout250 | 2 | 2.93 | 0.918 | 0.900 | +0.018 [0.009, 0.027] | 0.911 @ 2.62 (ceiling) | +0.007 [-0.001, 0.015] |
| holdout250 | 3 | 4.03 | 0.921 | 0.908 | +0.013 [0.007, 0.021] | 0.911 @ 2.62 | +0.011 [0.002, 0.020] |
| holdout250 | 4 | 5.11 | 0.923 | 0.913 | +0.011 [0.005, 0.017] | 0.911 @ 2.62 | +0.012 [0.002, 0.022] |
| holdout250 | 8 | 9.13 | 0.925 | 0.920 | +0.005 [0.002, 0.009] | 0.911 @ 2.62 | +0.015 [0.004, 0.025] |
| textbook72 | 2 | 6.87 | 0.546 | 0.537 | +0.009 [-0.006, 0.027] | 0.542 @ 6.88 | +0.004 [-0.019, 0.023] |
| textbook72 | 3 | 8.46 | 0.555 | 0.547 | +0.009 [-0.004, 0.023] | 0.548 @ 8.46 | +0.007 [-0.028, 0.036] |
| textbook72 | 4 | 9.67 | 0.560 | 0.549 | +0.011 [-0.000, 0.023] | 0.549 @ 8.81 (ceiling) | +0.012 [-0.027, 0.043] |
| textbook72 | 6 | 11.52 | 0.567 | 0.556 | +0.011 [0.002, 0.021] | 0.549 @ 8.81 | +0.018 [-0.026, 0.054] |

Reading:

- **Cost.** Count-stop keeps the cost predictable: c = 3 spends 4.0 attempts on holdout250 (the threshold rule's
  "m = 4" point spent 3.9 there but 12.1 on textbook72).
- **Against fixed-k.** It beats fixed-k with the same selector by +0.01 to +0.02 on holdout250.
- **Against first-finished.** It beats first-finished by about +0.01 only by spending past first-finished's ceiling.
  On textbook72 every comparison with first-finished includes 0.
- **Status.** Exploratory: one post-hoc design, read once.

## 7. Verifier and learned baselines (Worker A's scores; second freeze 03:15 UTC)

Procedure (`code/run_verifier_stage.sh`):

1. Extend the τ grid from dev-split verifier scores into `stopping/tau_grid_dev_v.json`. Intrinsic entries are
   byte-identical to the first freeze.
2. Run dev stopping and dev AUROC, and make the dev choices with the same rule as §3. Log the freeze.
3. Read test once.

Pre-registered primary: `verifier_ens`. Secondary: `verifier_hist_ens`, `completion_verifier_{,hist_}ens`, `shallow_lr`
(logistic regression on visible features, including this worker's likelihood features) and `shallow_lr_env`. Not
evaluated: `verifier_hist_rank_s0`, which appeared after the freeze.

**AUROC, finished attempts only** (pooled / within-theorem [95 % CI]):

| scorer | holdout250 pooled | holdout250 within | textbook72 pooled | textbook72 within |
|---|---|---|---|---|
| loglik_mean (this worker) | 0.669 | **0.788** [0.748, 0.820] | 0.645 | **0.687** [0.615, 0.767] |
| verifier_s0 / s1 / s2 | 0.704 / 0.742 / 0.705 | 0.638 / 0.640 / 0.631 | 0.726 / 0.682 / 0.748 | 0.600 / 0.579 / 0.585 |
| verifier_ens | 0.724 | 0.646 [0.597, 0.704] | 0.715 | 0.596 [0.514, 0.679] |
| verifier_hist_s0 / s1 / s2 | 0.784 / 0.766 / 0.719 | 0.660 / 0.653 / 0.631 | 0.795 / 0.808 / 0.800 | 0.599 / 0.614 / 0.586 |
| verifier_hist_ens | **0.774** | 0.647 [0.588, 0.708] | **0.812** | 0.622 [0.523, 0.709] |
| shallow_lr | 0.674 | 0.760 [0.711, 0.807] | 0.760 | 0.684 [0.595, 0.769] |

Dev showed the same pattern (dev1108 within-finished: verifier_ens 0.629, hist 0.647, loglik 0.787). Paired
differences, verifier_ens - loglik_mean (theorem bootstrap):

- Pooled: +0.056 [-0.055, 0.178] on holdout250; +0.070 [-0.057, 0.195] on textbook72.
- Within-theorem: -0.142 [-0.203, -0.080] on holdout250; -0.091 [-0.195, 0.007] on textbook72.

**Dev-chosen stopping on test** (full rows for every seed in `results/worker_B_tables.md` and
`results/stopping_headline.json`):

| set | policy | m | att | acc | vs fixed-k same selector | vs first-finished @ nearest att |
|---|---|---|---|---|---|---|
| holdout250 | verifier_ens | 2 | 1.66 | 0.904 | +0.036 [0.021, 0.052] | -0.001 [-0.002, 0.000] |
| holdout250 | verifier_ens | 4 | 2.16 | 0.911 | +0.027 [0.017, 0.042] | +0.003 [-0.003, 0.009] |
| holdout250 | verifier_ens | 8 | 4.75 | 0.911 | +0.010 [0.002, 0.019] | +0.000 [-0.011, 0.011] (FF ceiling) |
| holdout250 | verifier_hist_ens | 4 | 2.24 | 0.911 | +0.024 [0.013, 0.038] | +0.003 [-0.004, 0.011] |
| holdout250 | shallow_lr | 4 | 2.38 | 0.912 | +0.020 [0.011, 0.031] | +0.002 [-0.005, 0.009] |
| holdout250 | completion_loglik (for reference) | 4 | 3.93 | 0.919 | +0.012 [0.004, 0.021] | +0.008 [-0.002, 0.018] |
| textbook72 | verifier_ens | 2 | 4.31 | 0.524 | +0.018 [0.001, 0.037] | +0.001 [-0.002, 0.004] |
| textbook72 | verifier_ens | 4 | 7.34 | 0.520 | +0.003 [-0.016, 0.023] | **-0.024 [-0.051, -0.002]** |
| textbook72 | verifier_ens | 8 | 10.24 | 0.518 | -0.002 [-0.016, 0.014] | -0.031 [-0.069, 0.001] |
| textbook72 | verifier_hist_ens | 4 | 6.60 | 0.532 | +0.022 [0.003, 0.043] | -0.009 [-0.031, 0.015] |
| textbook72 | shallow_lr | 4 | 7.40 | 0.546 | +0.014 [-0.003, 0.033] | +0.002 [-0.030, 0.032] |

Seeds s0 / s1 / s2 at m = 4 vs first-finished: holdout250 -0.002 / +0.000 / +0.004; textbook72 -0.025 / -0.016 /
-0.019, all CIs within about ±0.03.

Fixed-k selection at k = 8, from the curve tables: verifier_ens pick 0.906 / 0.516, completion_loglik pick
0.918 / 0.544, completion pick 0.901 / 0.512 (holdout250 / textbook72).

The verifier's dev thresholds *do* transfer better in cost than likelihood (m = 4: 2.2 / 7.3 attempts against 3.9 /
12.1), consistent with its scores being calibrated to difficulty. Its displacement rate is higher (textbook72 m = 4:
3.7 %; m = 8: 5.7 %).

Figures `figures/stopping_{set}_acc_vs_cost.png` now include fixed-k verifier_ens pick and the adaptive verifier_ens
frontier (aqua). The token panel charges the verifier one forward pass (≤ 1024 tokens) per candidate.

## 8. Limitations

- **What the likelihood is.** T = 1 likelihood is a diagnostic, not the T = 0.8 sampling probability. Masking the
  assigned names is not a marginal over alpha-equivalent actions.
- **Cost accounting.** The likelihood forward is counted as extra cost although it is free at sampling time.
- **Scope.** Frozen-generator threshold stopping on i.i.d. fresh-context pools; this says nothing about accumulated
  context. A single generator seed (s0), a single sampling seed per pool, and R = 50 orderings.
- **Thresholds.** Absolute thresholds from dev (dev1108 + rr600) transfer to holdout250 but not to textbook72 (cost
  overshoot).
- **Dev operating points.** The dev rule only considers N_max ∈ {8, 16}, because dev pools have k = 16 / 32. The m = 2
  and 8 dev choices for completion-based policies often collapse to first-finished (τ below every finished score).
- **Curve tables.** These are optimistic: a frontier read on test.
- **The verifier is Worker A's.** Training choices (early stopping on dev pooled AUROC, clipping, pos-weighting) are
  theirs; dev is therefore slightly optimistic for the verifier, which makes its weak dev and test showing more robust,
  not less.
- **Unpriced scorers.** The total-token axis charges a forward pass only to the likelihood family and `verifier_*`;
  `completion_verifier_*` and `shallow_lr*` (which use the likelihood features) are charged 0 there, so their `tot`
  values in the JSON are understated. The plotted policies are charged correctly.
- **Small n.** textbook72 has 72 theorems, and only 34 have both classes among finished attempts, hence wide CIs.
- **Test exposure.** Before the freeze, an aggregate holdout250 line (validity 0.836, pass@64 0.96) from Worker A's
  collector log was seen while monitoring progress; nothing was chosen from it. Both test sets were also design-visible
  from the prior-pool headroom table (reviewer item 7).
- **Post-freeze change.** After the first test read, the curve-reading method was changed (from linear interpolation
  of the N_max = 64 curve to the frontier over a wider N_max set), and the change is logged. The dev-chosen headline is
  unaffected.
- **Count-stop is post-hoc.** The count-stop rule (§6b) was designed after the test read; it is exploratory. Other
  scale-free rules (for example, thresholds relative to the theorem's own attempts) were not tried.
