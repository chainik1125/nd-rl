---
written_on: 2026-10-30
written_by: agent:claude
---

# Base vs RL pass@k for the high-RL regime paper

This folder holds the figures and numbers for the paper's claim that our setting is in the
high-RL regime by the standard test (Yue et al. 2025). In the low-RL, elicitation regime, RL wins
at small k but the base catches up at large k. In the high-RL, capability-gain regime, RL wins at
every k.

The scripts here are CPU analysis of existing reads and run logs; they sample and train nothing.
They were written on branch `leon/high-rl-paper` at `1abc0ee`. Leon approved the work on
2026-10-30, as relayed by the coordinating agent. The figures go to
[`docs/papers/high-rl-regime/figures/`](../../../../docs/papers/high-rl-regime/figures/).

The reads added on 30 Oct (Charles149 at k1024, the 13-20 subset at k512, base seeds 1-2, and the
low-RL arm's training and reads) ran on ynez's GPUs from branch `leon/high-rl-reads` (commits
`d0cb16d`..`cce4dfc` on `79105ffe`; launcher in `launch/hirl/`), in
`ynez:/scratch/cowifornia/nd-rl/think_hirl/`.

## Regenerate

```bash
cd /home/cowifornia/code/research/nd-rl
ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice code/.venv/bin/python -m \
    code.experiments.current.high_rl_passk_20261030.passk_figs   # about 10 s
```

- **Options.** `--draws N` sets the number of bootstrap draws for the curves (default 2,000; the
  headline numbers use at least 4,000). `--only f1,f2,f3,f4,numbers` runs part of the pipeline,
  and `--figdir DIR` changes where figures go.
- **`--recount-stage1`** recounts the Stage-1 tokens from the data pickles. The counts are
  otherwise cached in [`stage1_tokens.json`](stage1_tokens.json).
- **Outputs.** The script writes PDF and PNG for each figure, plus
  [`numbers.md`](numbers.md) and [`numbers.json`](numbers.json). `numbers.md` holds every number
  quoted below, with its interval.
- **Inputs, all read-only.** Our reads and run logs (paths in the table below) and Dan's bucket
  files in `/scratch/cowifornia/nd-rl/dan_data/`. Those were copied read-only on 2026-10-30 with
  `uvx --from 'huggingface_hub>=1.0' hf buckets cp hf://buckets/dan-pandori/nd-rl/...`.

## Figure 1 (teaser): Charles149 at k up to 1024

```bash
cd /home/cowifornia/code/research/nd-rl
ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice code/.venv/bin/python -m \
    code.experiments.current.high_rl_passk_20261030.teaser   # a few seconds
```

- **Output.** `docs/papers/high-rl-regime/figures/teaser_passk.{pdf,png}`, 5.5 × 2.05 in, plus
  [`teaser_numbers.md`](teaser_numbers.md) / `.json`.
- **Which reads it uses.** Every finished read
  `/scratch/cowifornia/nd-rl/think/runs/eval/v2/charles149_<checkpoint>_s0_plain_k1024/`
  (16 merged chunks of k64, seeds 0–15, run by the ynez queue). A read counts as finished only
  when its `summary.json` exists, so partial reads are skipped. Rows are filtered to the 149
  names in `charles_clean.jsonl`.
- **No code change as reads arrive.** New checkpoints show up on a rerun, as long as each has an
  R in the compute table.
- **The compute table** is [`lowrl_compute.tsv`](lowrl_compute.tsv), a copy of
  `ynez:/scratch/cowifornia/nd-rl/think_hirl/lowrl/compute.tsv` (column `R`, the definition
  below). It matches this note's own values for L2 end, L3 end, ki78 and mini L2 it8. To refresh
  it:
  `scp -i ~/.ssh/id_ynez -o IdentitiesOnly=yes ynez:/scratch/cowifornia/nd-rl/think_hirl/lowrl/compute.tsv code/experiments/current/high_rl_passk_20261030/lowrl_compute.tsv`.
  A read whose checkpoint has no R, or that is malformed or short of k 1024, is skipped and listed in
  `teaser_numbers.md`.
- **(a)** pass@k on a log2 axis.
  - The base is grey and dashed, with a problem-bootstrap band.
  - Each RL checkpoint is coloured by R on one sequential ramp. The log colour scale is fixed at
    5e-5 to 0.5, so colours do not shift as points arrive.
  - Only the final model has a band: the recipe's L3 end (`grpo_think_vs_mb_kkl0`), or the
    largest-R checkpoint if that read is missing.
- **(b)** RL − base against R at k = 1 (filled) and k = 1024 (open), with 95% problem-bootstrap
  bars.
  - The region below 0 is shaded and labelled "RL below base", since a k-1024 point there is the
    elicitation signature.
  - The dotted line is Yue et al.'s median FLOP ratio, 1.2e-4.
- **As of 2026-10-30, 3 pm PT** (base and L3 end only):
  - base: 39.2 / 56.5 / 78.5 (pass@1 / pass@5 / pass@1024);
  - L3 end (R 0.289): 70.3 / 78.9 / 89.3;
  - gap: +31.1 [+26.0, +36.6] at pass@1 and +10.7 [+6.0, +16.1] at pass@1024.

## Three seeds at k1024 (Charles149)

```bash
ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m \
    code.experiments.current.high_rl_passk_20261030.seeds3
```

- **What it compares.** Base `S1g3_s{0,1,2}` against the recipe's L3 end `s{0,1,2}`; each RL seed
  starts from its own base seed, so the bootstrap resamples problems and seed pairs together.
  Output in [`seeds3.json`](seeds3.json).
- **As of 2026-10-30, 9 pm PT** (means over seeds, pass@1 / pass@5 / pass@1024): base 32.6 / 49.7
  / 74.5, RL end 70.5 / 79.0 / 86.6. The gap is +37.9 [+30.0, +46.2] at pass@1 and +12.1 [+7.2,
  +17.9] at pass@1024; every seed is above its base at every k. Pooled over 3,072 attempts per
  side, 9 problems (6.0%) are solved only by RL and none only by the base.

## The low-RL arm on the 13-20-line subset

```bash
ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 code/.venv/bin/python -m \
    code.experiments.current.high_rl_passk_20261030.lowrl_1320
```

- **What it compares.** The teaser's checkpoints against the base on the 250-problem 13-20-line
  sample (`sets/t1320s250_ids.txt` on ynez, seed 20261030), k512, seed 0. Output in
  [`lowrl_1320.md`](lowrl_1320.md) / `.json`.
- **As of 2026-10-31, 1:45 am PT.** The shape matches Charles149: near Yue et al.'s median, RL is within about 2 points of
  the base at k512 (two points slightly below, not resolved). The gap opens from about 1e-2 and
  reaches +16.8 [+12.0, +21.6] at the L3 end. Unlike Charles149, the 16-prompt variant gains
  little at k512 by 32 steps, so the steps-not-FLOPs pattern does not carry over.

## Definitions

- **pass@k.** The unbiased per-problem estimate 1 − C(n − c, k) / C(n, k), averaged over problems
  and then seeds (`code.thinking.stats`). Values are % of the set. Gaps (RL − base) are in points.
- **Bands and intervals.** 95% percentile intervals from `stats`' hierarchical bootstrap. Every
  draw resamples problems once, shared by both arms and every k. It also resamples seeds where an
  arm has several, independently per arm. The exception is textbook72 in the v2 world, where the
  base and RL are paired by Stage-1 seed. With one base seed, the intervals cover problem and
  attempt noise plus RL-seed noise; the base's own seed noise is not in them. A p of 0 means
  below 1 / draws.
- **RL FLOPs / Stage-1 FLOPs (F3's x axis).**
  - The ratio is (6 · trained-token passes + 2 · sampled tokens) / (6 · Stage-1 tokens). The
    parameter count cancels: RL trains the Stage-1 network itself (9.57M parameters in v2, 9.56M
    in v1).
  - Trained-token passes are `trained_tokens` summed over `metrics.jsonl`. Each token is trained
    once per iteration: `--updates 4` splits a batch into 4 minibatches, it does not repeat it.
  - Sampled tokens are `tokens_per_attempt` × prompts × group, summed over iterations.
  - Sums are cumulative per checkpoint, in file order up to the checkpoint's iteration.
    Rolled-back iterations count, because their compute was spent.
  - Excluded: prefill (state tokens), the old-log-prob pass (at most 2 · trained more), KL-anchor
    reference passes, canary scoring and evals.
  - Stage-1 tokens are epochs × (state + completion tokens per epoch) from the data pickle, with
    no padding. That is 440.6M tokens for S1g3 and 440.1–441.3M for T1v1 and T0.
  - The Stage-1 count includes the state tokens; the RL count leaves prefill out. So the ratio
    understates RL relative to Stage-1.
  - Dan's EI points use his `compute.json` the same way: 6 · fine-tune tokens + 2 · generated
    tokens. His fine-tune tokens include the state prompt.
  - The vertical line is Yue et al.'s median RL / pretraining FLOP ratio, 1.2 × 10⁻⁴, from Dan's
    rl-compute-share review on `origin/dan`.
- **Shared k (F3).** The largest k that every checkpoint's read has:
  - 13–20 lines: k 4, because the L3 checkpoint reads are k4;
  - Charles149: k 16;
  - textbook72: k 64.
- **RL-only problems.** Problems RL solves in its first n attempts that the base never solves in
  n. "RL's own rate" is RL's success rate over all its attempts on those problems.

## Figures

| file | what | finding (observed) |
|---|---|---|
| `passk_base_vs_rl` | (a) Charles149, the seed-0 chain from S1g3_s0: base, mini L2 it8, L2 end, L3 it10/20/30, L3 end, ki78. (b) 13–20 lines, 3 RL seeds, base s0. (c) the v1 recipe on textbook72: T1v1 base (k256), GRPO it39/it78, EI round 2 | (a, b): every RL checkpoint is above the base at every k read, from 33k attempts on. (c): v1 GRPO it78 falls below its base at k ≈ 18; EI stays above |
| `gap_vs_k` | RL end (L3, 3 seeds) − base against k, for 4 sets, with the v1 GRPO it78 textbook72 gap as a dashed contrast | Charles149 and textbook72 narrow (+32 → +14 and +35 → +14 points from k 1 to 64) but stay above 0. 13–20 and 21–30 widen (+24 → +34, +7 → +21 points, k 1 → 8). The v1 contrast goes below 0 at k ≈ 18 |
| `gap_vs_rl_compute` | gap at pass@1 (filled) and at the shared k (hollow) against the RL / Stage-1 FLOP ratio. Points: the v2 chain (seed 0), v1 GRPO it39/it78 (own Stage-1), and Dan's EI ladder r1/r4/r8 on textbook72 | Our runs sit at 0.03–0.45× Stage-1 FLOPs and Dan's EI at 0.2–2.0×: 2.4–4.2 orders of magnitude right of Yue's median. The left of the axis is empty, waiting for the low-RL arm |
| `dan_large_k` | (a) support-curves: base vs EI r8, 383 transfer theorems, k up to 10⁴. (b, c) trajectory best-cap12: r0 vs EI rounds 1/4/8, textbook72 and holdout250, 3 seeds, k up to 256 | No crossing up to k = 10⁴ (a) or 256 (b, c). At 10⁴ the base reaches 11.7% and EI 31.6% |

### Adding a point to F3 (the planned low-RL arm)

- **From a run in this layout.** Append it to `CHAIN` and `V2_ARMS`, giving its read stems and
  k, and to `v2_segments`, giving its `metrics.jsonl` stages.
- **From precomputed values.** Pass `--extra-f3 FILE.json`, a list of
  `{"label", "family", "set", "x", "gap1", "gapk"}`, where `set` is `t1320`, `charles149` or
  `tb72`. A new family gets its own marker.

## Key numbers (% of set; pass@1 / pass@5 first)

All intervals are 95% bootstrap ([`numbers.md`](numbers.md)). The v2 base is S1g3_s0 (one seed).
"RL" is the recipe's L3 end (`grpo_think_vs_mb_kkl0`), with 3 seeds unless noted.

| set | base pass@1 / pass@5 | RL pass@1 / pass@5 | gap pass@1 / pass@5 | largest k: base → RL, gap |
|---|---|---|---|---|
| Charles149 (149) | 38.7 / 55.8 | 70.5 / 78.9 | +31.8 [+26.6, +37.1] / +23.1 [+17.3, +29.1] | k 64: 69.8 → 83.4, +13.6 [+6.9, +20.8] |
| 13–20 lines (986) | 8.5 / 23.4 | 32.1 / 57.8 | +23.6 [+19.7, +27.5] / +34.3 [+29.4, +39.0] | k 8: 29.5 → 63.8, +34.3 [+29.4, +38.9] |
| 21–30 lines (1,000) | 0.5 / 2.2 | 7.2 / 19.4 | +6.7 [+4.5, +9.4] / +17.2 [+12.6, +23.0] | k 8: 3.1 → 23.8, +20.7 [+15.3, +27.2] |
| textbook72 (72; base 3 seeds at B256) | 38.6 / 55.0 | 74.0 / 82.3 | +35.4 [+25.2, +45.4] / +27.3 [+18.0, +37.1] | k 64: 73.6 → 87.5, +13.9 [+4.2, +25.0] |

**Earlier and later checkpoints on Charles149, gap against the base s0** (pass@1 / pass@5 /
largest k):
- L2 end (3 seeds): +28.8 / +24.0 / k 64 +16.3 [+10.1, +22.8].
- ki78 end (s0, 117 iterations): +30.2 / +23.9 / k 64 +15.4 [+8.7, +22.1].
- mini L2 it8 (s0, the least-RL read, 0.03× Stage-1 FLOPs): +17.5 at pass@1 and +14.9 at k 16.

**RL-only problems at equal attempts** (seed-0 chain):

| set (n) | RL-only | base-only | RL's own rate on RL-only: median [IQR] | base bound per attempt (95%) |
|---|---|---|---|---|
| Charles149 (64) | 26 (17.4%) | 3 (2.0%) | 75.0% [20.7, 98.4]; 17 of 26 at ≥ 50% | 4.6% |
| 13–20 (8) | 403 (40.9%) | 24 (2.4%) | 43.8% [18.8, 62.5] | 31.2% (n = 8 is weak) |
| 21–30 (8) | 282 (28.2%) | 5 (0.5%) | 25.0% [12.5, 37.5] | 31.2% |
| textbook72 (64, base B256) | 13 (18.1%) | 3 (4.2%) | 39.1% [4.7, 100] | 4.6% |
| v1 textbook72, GRPO it78 vs T1v1 (64) | 4 (5.6%) | 8 (11.1%) | 8.6% [5.1, 16.0] | 4.6% |

**Matched sampled tokens per problem** (`stats.paired_diff_at_tokens`, base s0 against the RL
seeds).
- **Charles149.** RL samples more per attempt: 374–440 tokens against the base's 306. The gap is
  still +24.2 [+18.4, +30.2] points at 1k tokens per problem and +13.6 [+7.0, +20.3] at 16k.
- **13–20.** +20.6 at 1k tokens and +27.9 at 4k.
- **textbook72.** +31.9 at 1k tokens and +16.5 at 16k.
- Targets outside both reads' ranges are clamped and flagged in `numbers.md`.

**v1 contrast (textbook72, 1 seed each).**
- **T1v1_s0 base:** 19.1 / 33.3 / k 64 54.2 / k 256 59.7.
- **GRPO it78:** 31.6 / 38.8 / k 64 48.6.
  - The curves cross at k ≈ 18. The gap at k 64 is −5.6 [−14.2, +2.9].
  - The point estimates cross, but the interval includes 0, so the crossing is not resolved.
- **GRPO it39:** +5.6 [−3.7, +16.2] at k 64, still above the base.
- **Across the four v1 GRPO runs,** the k-64 gap fell from it39 to it78 in 3 of 4: T1v1 s0
  +5.6 → −5.6, T1v1 s1 +6.1 → +4.7, T0 s1 +2.7 → −0.1. The fourth rose: T0 s0 +4.8 → +8.9.
- **EI round 2:** above the base at every k, +13.9 [+6.6, +22.0] at k 64.

**Dan's EI data (F4).**
- **support-curves** (seed 0, T 0.8):
  - base: 0.5 / 1.8 / 4.0 / 7.1 / 11.7 at k 1, 10, 100, 1,000 and 10,000;
  - EI r8: 16.1 / 21.4 / 24.9 / 27.8 / 31.6.
  - Solved at least once: 45 against 121 of 383. This matches Dan's E2′ and E3′.
- **trajectory cap 12** (3 seeds), textbook72, pass@1 → pass@256:
  - r0: 17.1 → 47.7;
  - r1: 29.3 → 59.3;
  - r4: 37.2 → 68.5;
  - r8: 39.2 → 69.0.
- **trajectory cap 12, holdout250:** r0 32.6 → 80.1; r8 80.8 → 95.1.
- **F3 gaps on textbook72** (pass@1 / pass@64, against r0):
  - r1 (0.21× Stage-1 FLOPs): +12.3 / +13.8;
  - r4 (0.94×): +20.1 / +23.0;
  - r8 (2.0×): +22.1 / +24.7.
  - The large-k gap grows with RL compute.

## Read paths

| arm | reads |
|---|---|
| v2 base (S1g3_s0, B64) | `/scratch/cowifornia/nd-rl/think/runs/eval/v2/{charles173,t1320,t2130}_s1g3_s0_plain_k{64,8,8}/rows_plain_B64.jsonl` |
| v2 base on textbook72 (S1g3 s0–2, **B256**) | `runs/eval/rebuild_sanity/tb72_S1g3_s{0,1,2}/rows_plain_B256.jsonl` |
| L2 end, L3 checkpoints, L3 end | `runs/eval/v2/<set>_grpo_think_vs_L2_s*`, `..._grpo_think_vs_mb_kkl0_it{10,20,30}_s*`, `..._grpo_think_vs_mb_kkl0_s*` |
| mini L2 it8, ki78 | `runs/eval/v2/<set>_grpo_think_vs_mini20_L2it8_s0_*`, `..._grpo_think_vs_mb_ki78_s0_*` |
| v2 compute | `runs/grpo_v2/{grpo_think_vs_s0_L2, grpo_think_vs_mb_kkl0_s0_L3, grpo_think_vs_mb_ki78_s0_L3, grpo_think_vs_mini20_s0_L2}/metrics.jsonl` |
| v1 reads | `/scratch/cowifornia/nd-rl/think_serrano_20261011/runs/eval/{tb72_T1v1_s*, tb72_T0_s*}`, `.../eval/matched/{tb72,t1320}_{T1v1,T0,grpoT1v1e*,grpoT0*,eiT1v1e_r2}*` |
| v1 compute | `.../think_serrano_20261011/runs/restart3_grpofix/grpo_{T1v1e,T1v1e_s1,T0,T0_s1}/metrics.jsonl` |
| Dan | `/scratch/cowifornia/nd-rl/dan_data/support-curves/s1_{base,ei}_T08_s0.s{0,1}.jsonl`; `.../dan_data/trajectory/s{0,1,2}_{pend,r1,r4,r8}__{tb72,h250}_x1.jsonl` and `compute.json` |

Filters: Charles149 is `runs/pools/charles_1002/charles_clean.jsonl`. 13–20 is the 986 valid
items (`valid_1320`, the same rule as `matched_eval/figs.py`); the v1 reads are filtered to the
same 986.

## Caveats

- **The v2 base has one seed.** S1g3_s0 is the only Stage-1 model read on Charles149, 13–20 and
  21–30. The 3-seed RL means include RL runs from S1g3_s1 and S1g3_s2, whose bases were not read.
  Panel (a) of F1 and all of F3 use the seed-0 chain for that reason. On textbook72 the Stage-1
  seeds read 43.8, 39.6 and 32.3% pass@1, a spread of 11.5 points.
- **The textbook72 base was read at think budget B256, the RL at B64.** The pre-crash S1g3 models
  (serrano, `eval/matched/tb72_S1g3_s{0,1}`) were read at both budgets. Their B64 and B256 reads
  are within 0–4 points of each other at k 1, 5 and 64, in both directions (s0: 42.9 against
  40.7 at pass@1; s1: 41.6 against 43.0; s1 at k 64: 77.8 against 81.9). That is small against
  the +14-point gap at k 64.
- **Unequal k.** The base reads on 13–20 and 21–30 stop at k 8, and the L3 checkpoint reads at
  k 4 (13–20, 21–30) or k 16 (Charles149). The largest k in the v2 world is 64. A crossing
  beyond k 64 on Charles149 or textbook72 cannot be excluded: the base is still rising there
  (Charles149 +3.5 points from k 32 to 64).
- **The v1 crossing is a point-estimate crossing** with intervals that include 0. The v1 runs
  are single seeds from a different, pre-crash world (serrano). The s0 thinking run had a
  clip-low spike at it48–52 (`thinking_traces_20261005/RL_RESULTS.md`).
- **Selection.** KL 0 (the recipe's L3) was selected on 13–20, 16–20, 21–30 and Charles149.
- **Sets.**
  - textbook72 is sealed: reads only, no tuning.
  - Charles149 is source OOD, 13–20 is longer than any RL training problem, and 21–30 is a
    fixed 1,000-item sample.
  - Dan's holdout250 and support-curves transfer sets are his own and differ from ours.
- **Dan's data comes from other models.**
  - support-curves uses a 3.2M-parameter cap-6 sequence model, and its EI compute was not
    tallied here.
  - trajectory uses best-cap12 (9.56M parameters), 3 seeds, n = 256 from sample seed 1.
  - Some support-curves rows stopped after 50 successes (14 base, 94 EI rows, n ≥ 2,048). For
    k > n they use the plug-in 1 − (1 − c/n)^k.

## Corrections to the 2026-10-28 report (now the notes box in `01_intro.tex`)

1. **The gradient-pass figure was wrong.** That report gave "× 4 PPO updates ≈ 340M token
   passes, gradient passes ≈ 0.8× pretraining". `--updates 4` splits each batch into 4
   minibatches; it does not train each token 4 times. Trained-token passes are 84–86M per seed,
   about 0.19× Stage-1 tokens. The FLOP ratio defined above is 0.29 for the recipe's L3 end.
2. **The v1 crossing is at k ≈ 18, not k ≈ 32.** At k 16 GRPO it78 still leads (44.3 against
   43.8); by k 32 it trails (46.7 against 49.3). The crossing is not statistically resolved:
   at k 64 the gap is −5.6 [−14.2, +2.9].
3. Everything else in that report reproduces: the pass@k values, the RL-only counts and the
   Stage-1 token count.
