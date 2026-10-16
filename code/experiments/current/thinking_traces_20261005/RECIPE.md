---
written_on: 2026-10-16
written_by: agent:claude
---

# Primary RL recipe: GRPO with thinking, verifier (stop), length curriculum, merged L3 + KL 0

## Summary

Leon's decision (2026-10-16): GRPO is the RL method from now on, and expert iteration (EI) is retired. The primary
recipe is the arm `grpo_think_vs_mb_kkl0`, called "merged recipe + KL 0" in the results tables.

The recipe starts from the Stage-1 thinking model S1g3. S1g3 was trained on programmatic search traces at effort
budgets 0, 64 and 256, with the budget shown to the model as an effort token. RL is GRPO with thinking on, at the same
three budgets.

Rollouts use the verifier-guided sampler in its stop form (`grpo_think_vs`). This is the logical arm with no redraws
(`--max_rej 0`). The first step that the environment or `step_check` rejects ends the attempt, and that draw gets a
step penalty of 0.1 on top of the attempt's advantage.

Training is a length curriculum of two stages. Each stage is 39 iterations × 256 targets × 16 attempts, 160k sampled
attempts per stage:

- **L2**, problems of 10–12 lines, with a KL anchor of 0.02 to the Stage-1 model.
- **L3**, problems of 13–15 lines, with the merged recipe:
  - a warmup-stable-decay (WSD) lr cooldown: 1−sqrt over the last 8 iterations, down to 0.3×;
  - a third of each iteration's groups drawn from 3-premise rewrites of the L3 rows;
  - an unlikeliness reward, β 0.25;
  - KL anchor 0, meaning no anchor in L3.

**Observed** (3 seeds, plain reads, pass@1 / pass@5 in % of the set; table below):

| set | pass@1 | pass@5 |
|---|---|---|
| 13–20 | 32.1% | 57.8% |
| 16–20 | 25.9% | 50.5% |
| 21–30 | 7.2% | 19.4% |
| textbook72 | 74.0% | 82.3% |
| Charles149 | 70.5% | 78.9% |

Paired differences against the other arms (3 seeds):

- **Against the merged recipe** (the same recipe with KL 0.02 in L3), the recipe is ahead on the generated sets:
  - pass@1: +1.2 to +1.9 points;
  - pass@5: +2.4 to +3.4 points;
  - the intervals exclude 0.
- **On textbook72 and Charles149**, it is level with the merged recipe.
- **Against EI and EI + verifier**, it is level on the long generated sets (13–20, 16–20 and 21–30).
- **Against EI and EI + verifier at pass@1**, it is ahead on Charles149. On textbook72 it is ahead of EI, and the
  lead over EI + verifier is not resolved.
- **The KL 0 gain shows in plain reads only.** In guided reads it is level, and lower on Charles149 at pass@5 (see
  [Caveats](#caveats)).

## Stage-1 init: S1g3

- **Traces.** Search traces are written by code (`code/thinking/searchtrace.py`); no language model wrote any of them.
  They cover the K12 proofs at levels 0, 64 and 256, mixed 20/40/40, with textbook72 excluded. The build commands
  (`searchtrace` per level, then `code.thinking.data --graded 0:0.2 64:…e64.jsonl:0.4 256:…e256.jsonl:0.4
  --extra-exclude $TB --no-lean --out runs/data/S1g3.pkl`) are in
  [search_traces/README.md](search_traces/README.md).
- **Training.** Dan's best recipe, 24,000 steps: 6 layers, d 384, 8 heads, d_ff 1280. The command for seed `S`, from
  `runs/stage1/S1g3_s0/args.json`:

  ```
  python code/thinking/train.py --data /scratch/cowifornia/nd-rl/think/runs/data/S1g3.pkl \
    --heldout /scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl \
    --out /scratch/cowifornia/nd-rl/think/runs/stage1/S1g3_s$S --seed $S --best_steps 24000
  ```

- **Seeds 0, 1 and 2.** These models were retrained on ynez on 2026-10-13 with `leon/thinking` cd089700, after
  serrano's crash. Their textbook72 pass@1 is 43.8 / 39.6 / 32.3% for seeds 0 / 1 / 2
  ([EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md), "On ynez").

## Stage 2: the GRPO commands

These commands are the `_meta.argv` of the run records:

- `runs/grpo_v2/grpo_think_vs_s{0,1,2}_L2/args.json`;
- `runs/grpo_v2/grpo_think_vs_mb_kkl0_s{0,1,2}_L3/args.json`.

Both are under `/scratch/cowifornia/nd-rl/think/`.

The two stages ran at different commits:

- **L2** is the L2 of the `grpo_think_vs` arm, which also feeds the merged-recipe and v2-start arms. It ran at
  `leon/grpo-v2` fa28f6eb on ynez (2026-10-13).
- **L3** ran at f35d3daf on serrano (2026-10-15).
- **The full reads** ran at 38e12690.

All three commits are ancestors of `leon/main`. The later `grpo.py` changes are opt-in, and tests check that the
default path keeps its digests (`code/tests/thinking/test_grpo_sched.py`, `test_grpo_pilots.py`). The commands run
from the repository root, with `S` the seed and `R=/scratch/cowifornia/nd-rl/think/runs`.

**L2** (10–12 lines, from Stage 1):

```
python -m code.thinking.grpo --init $R/stage1/S1g3_s$S/model.pt --name grpo_think_vs_s${S}_L2 \
  --out $R/grpo_v2/grpo_think_vs_s${S}_L2 --targets $R/pools/curriculum/L2/pool.jsonl \
  --iters 39 --prompts 256 --group 16 --batch 1024 --lr 1.5e-4 --overlong fail --save_every 39 \
  --max_steps 96 --seed $S --keep_rolling 100 --effort --budgets 0,64,256 \
  --canary $R/pools/canary_long.jsonl --canary_n 1000 --canary_every 1 \
  --rollback_on_alarm --max_rollbacks 3 --ban_restore_check --ban_stop 0.005 \
  --kl_anchor_coef 0.02 --kl_anchor_ref $R/stage1/S1g3_s$S/model.pt \
  --monitor --arm logical --max_rej 0 --rej_penalty 0.1
```

**L3** (13–15 lines, from L2's last checkpoint):

```
L2CK=$R/grpo_v2/grpo_think_vs_s${S}_L2/ckpts/grpo_think_vs_s${S}_L2_it39.pt
python -m code.thinking.grpo --init $L2CK --name grpo_think_vs_mb_kkl0_s${S}_L3 \
  --out $R/grpo_v2/grpo_think_vs_mb_kkl0_s${S}_L3 --targets $R/pools/curriculum/L3/pool.jsonl \
  --iters 39 --prompts 256 --group 16 --batch 1024 --lr 1.5e-4 \
  --lr_schedule wsd --decay_frac 0.2 --min_lr_frac 0.3 --decay_shape sqrt --unlikely_beta 0.25 \
  --overlong fail --save_every 39 --max_steps 96 --seed $S --keep_rolling 100 --effort --budgets 0,64,256 \
  --canary $R/pools/canary_long.jsonl --canary_n 1000 --canary_every 1 \
  --rollback_on_alarm --max_rollbacks 3 --ban_restore_check --ban_stop 0.005 \
  --kl_anchor_coef 0 --kl_anchor_ref $L2CK \
  --monitor --arm logical --max_rej 0 --rej_penalty 0.1 \
  --targets_extra $R/pools/curriculum_3p/L3_3p.jsonl --targets_extra_frac 0.33
```

**Every value that differs from [`grpo.py`](../../../thinking/grpo.py)'s defaults:**

| flag | grpo.py default | L2 | L3 | what it does |
|---|---|---|---|---|
| `--init` | (required) | `stage1/S1g3_sS/model.pt` | `grpo_think_vs_sS_L2_it39.pt` | start weights |
| `--targets` | (required) | `pools/curriculum/L2/pool.jsonl` (5,000 rows, 10–12 lines) | `pools/curriculum/L3/pool.jsonl` (5,000 rows, 13–15 lines) | RL pool |
| `--budgets` | 16,64,256 | 0,64,256 | 0,64,256 | think budgets, one per group |
| `--effort` | off | on | on | show the budget as an effort token |
| `--iters` | 280 | 39 | 39 | one stage = 160k attempts = one EI round |
| `--lr` | 5e-5 | 1.5e-4 | 1.5e-4 | peak lr (the 10-step warmup is the default) |
| `--overlong` | mask | fail | fail | an attempt the cap ends counts as a failure |
| `--max_steps` | 48 | 96 | 96 | step cap (eval's value) |
| `--arm` | plain | logical | logical | verifier-guided sampling |
| `--max_rej` | 10 | 0 | 0 | no redraws: the first rejected step ends the attempt ("stop") |
| `--rej_penalty` | 0 | 0.1 | 0.1 | step penalty on the rejected draw |
| `--kl_anchor_coef` | 0 | **0.02** (L2 only) | **0** (passed explicitly; grpo.py's default) | KL(π_ref ‖ π) anchor; at 0 the reference is not loaded |
| `--kl_anchor_ref` | none | the Stage-1 model | the L2 checkpoint (unused at coefficient 0) | π_ref |
| `--lr_schedule wsd` | constant | — | **L3 only** | warmup-stable-decay |
| `--decay_frac` / `--min_lr_frac` / `--decay_shape` | (wsd: 0.2 / 0.1 / linear) | — | **L3 only**: 0.2 / 0.3 / sqrt | constant to iteration 31, then 1−sqrt over 8 iterations to 0.3× |
| `--unlikely_beta` | 0 (off) | — | **L3 only**: 0.25 | unlikeliness reward (He, Fried & Welleck 2025) |
| `--targets_extra` / `--targets_extra_frac` | none / 0 | — | **L3 only**: `pools/curriculum_3p/L3_3p.jsonl` (1,843 rows) / 0.33 | a third of each iteration's groups from 3-premise rewrites |
| `--canary` | none | `pools/canary_long.jsonl` | the same | collapse canary on long-state steps |
| `--rollback_on_alarm` | off | on | on | on an alarm, restore the last good checkpoint and halve the lr (up to 3 times) |
| `--ban_restore_check` | off | on | on | a think-ban alarm alone does not roll back onto a restore point that is over the limit too |
| `--ban_stop` | 0 (off) | 0.005 | 0.005 | end the stage when the think-ban mass exceeds this |
| `--monitor` | off | on | on | gradient-source logging (dashboards) |
| `--save_every` / `--keep_rolling` | 35 / 4 | 39 / 100 | 39 / 100 | checkpointing only |
| `--seed` | 0 | 0, 1, 2 | 0, 1, 2 | |

Some flags are passed with grpo.py's default value: `--prompts 256 --group 16 --batch 1024 --canary_n 1000
--canary_every 1 --max_rollbacks 3`. Others are left at their defaults:

- clip 0.2 / 0.28 (clip-higher), with dual clip 3;
- 4 updates per batch;
- sampling temperature 1.0;
- `--kl_anchor_n 512`;
- `--rej_penalty_scope diverge`, which has no effect without redraws.

## How to launch it (`line.sh`)

The launcher is [`grpo_v2/launch/line.sh`](grpo_v2/launch/line.sh). Its installed copy,
`/scratch/cowifornia/nd-rl/think/runs/grpo_v2/line.sh`, is byte-identical to the committed file. It has two
constraints:

- It runs the code of the worktree `/scratch/cowifornia/nd-rl/think/nd-rl-v2` and refuses to start unless that
  worktree is on `leon/grpo-v2` (the `WT` line).
- It runs with the venv `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv`.

To run it from `leon/main`, change `WT` and the branch check. The arm is a knob pilot of the merged recipe, so it
needs the L2 of `grpo_think_vs` first:

```
cd /scratch/cowifornia/nd-rl/think/runs/grpo_v2
bash line.sh GPU SEED grpo_think_vs                       # L2 (and its own L3 and reads: the v2-start arm)
KNOB_FULL=1 bash line.sh GPU SEED grpo_think_vs_mb_kkl0   # L3 of the recipe from that L2, then the full reads
```

- `grpo_think_vs_mb_kkl0` refuses to start in either of two cases:
  - `grpo_think_vs_sSEED_L2_it39.pt` is missing;
  - the last guard line of that L2's log is not at `lr x1 (0 rollbacks)`.

  It runs the preflight on the pools first: no overlap with any evaluation set, by prompt, renaming key or class.
- **The reads.**
  - Without `KNOB_FULL=1` the arm does only the knob-screening reads: plain, on t1320 k16, t2130 k8 and charles173
    k64.
  - With `KNOB_FULL=1` it does the usual full reads: plain and guided on t1320, textbook72 and charles173, plus
    hand14 and t2130. A finished L3 is skipped, and so is any read that already exists.
  - The seeds were run in exactly this order: first the screening reads, then `KNOB_FULL=1` for the rest.
- **Seeds 0, 1 and 2.** The runs went through `pull_queue.sh`, one worker per GPU, taking job lines such as
  `grpo_think_vs_mb_kkl0:1` from a shared file:

  ```
  [KNOB_FULL=1] bash pull_queue.sh GPU JOBFILE
  ```

  Examples are `runs/grpo_v2/jobs_kl0_seeds.txt` and `jobs_kl0_full.txt`.
- **Smoke test.** `SMOKE=1 bash line.sh 0 0 ARM` is a tiny CPU dry run of the whole line.
- Details: [grpo_v2/README.md](grpo_v2/README.md) ("Merged recipe", "Launch") and
  [grpo_v2/launch/README.md](grpo_v2/launch/README.md) (item 9, the knob arms).
- **Timing.** The L3 stages of seeds 1 and 2, with their screening reads, ran in parallel from 7:52 to 10:23 am on
  serrano (2026-10-15), about 2.5 h.

## Evidence

All numbers come from § 1 (arm table) and § 3 (paired differences) of
[matched_eval/v2_results_ynez.md](matched_eval/v2_results_ynez.md), generated by `v2_results.py`. Those are the
"ynez world" tables: Stage-1 models and pools rebuilt on ynez, never pooled with serrano's pre-crash results.

How the results were read:

- **Read settings.** Plain sampling at T 0.8, think budget B64. The guided reads use the logical arm, with up to 10
  redraws.
- **Seeds.** 3 seeds per arm. Arms are paired by their L2 / Stage-1 init.
- **Intervals.** 95% hierarchical-bootstrap intervals, over seeds and problems.

The sets:

| set | items | k |
|---|---|---|
| 13–20 (sealed) | 986 | 16 |
| 16–20 (the part of 13–20 longer than any training problem) | 616 | 16 |
| 21–30 (a fixed sample of the sealed set) | 1,000 | 8 |
| textbook72 | 72 | 64 |
| Charles149 (the clean part of Charles's 173-problem release) | 149 | 64 |

The arms compared:

- **Merged recipe:** the same recipe with KL 0.02 in L3.
- **GRPO + verifier (stop):** the v2-start arm, `grpo_think_vs`; its L3 is plain GRPO with the stop sampler.
- **EI + verifier:** guided EI.
- **EI:** plain EI.

Both EI arms use the same curriculum, one k-32 round per stage, from the same S1g3 seeds.

**pass@1, % of the set (plain reads, 3 seeds)**

| arm | 13–20 | 16–20 | 21–30 | textbook72 | Charles149 |
|---|---|---|---|---|---|
| **merged recipe + KL 0 (this recipe)** | **32.1** [28.2, 36.1] | **25.9** [22.1, 30.3] | **7.2** [5.0, 10.0] | 74.0 [64.7, 82.6] | 70.5 [64.2, 76.3] |
| merged recipe (KL 0.02) | 30.4 [26.4, 34.5] | 24.0 [19.8, 29.0] | 6.0 [3.9, 8.7] | 74.7 [65.5, 83.1] | 70.4 [64.0, 76.5] |
| GRPO + verifier (stop), v2 start | 29.2 [24.6, 33.2] | 22.4 [18.4, 25.9] | 5.6 [3.5, 7.5] | 74.8 [65.1, 83.8] | 70.2 [63.5, 76.7] |
| EI + verifier | 31.1 [27.5, 34.8] | 25.1 [21.1, 29.3] | 6.7 [5.3, 8.2] | 70.3 [61.0, 78.6] | 63.7 [55.8, 71.0] |
| EI | 31.5 [28.7, 34.1] | 25.5 [22.7, 28.2] | 7.4 [6.2, 8.6] | 68.4 [58.9, 77.4] | 62.2 [55.2, 68.8] |

**pass@5, % of the set (plain reads, 3 seeds)**

| arm | 13–20 | 16–20 | 21–30 | textbook72 | Charles149 |
|---|---|---|---|---|---|
| **merged recipe + KL 0 (this recipe)** | **57.8** [52.9, 62.2] | **50.5** [45.3, 55.7] | **19.4** [14.8, 25.2] | 82.3 [73.6, 90.0] | 78.9 [72.6, 84.8] |
| merged recipe (KL 0.02) | 55.4 [49.8, 60.6] | 47.8 [41.4, 54.4] | 16.0 [11.2, 21.9] | 83.1 [74.5, 90.6] | 79.8 [73.7, 85.3] |
| GRPO + verifier (stop), v2 start | 53.0 [47.3, 58.0] | 45.1 [39.4, 50.3] | 15.0 [10.6, 19.0] | 81.7 [72.6, 89.6] | 78.7 [72.4, 84.4] |
| EI + verifier | 57.4 [52.9, 62.2] | 50.7 [44.9, 56.8] | 18.4 [14.9, 22.3] | 80.7 [72.2, 88.4] | 75.9 [68.9, 82.1] |
| EI | 57.9 [53.7, 61.9] | 51.2 [46.0, 56.4] | 19.5 [16.7, 22.3] | 77.9 [69.1, 85.9] | 73.8 [66.9, 80.0] |

**Paired differences, this recipe minus each arm** (points, plain reads, 3 seeds paired by init, pass@1 / pass@5; §
3). The intervals of differences are much narrower than those of the levels above, so read these, not the gaps
between the rows above.

| minus | 13–20 | 16–20 | 21–30 | textbook72 | Charles149 |
|---|---|---|---|---|---|
| merged recipe | +1.7 [+0.8, +2.6] / +2.4 [+0.7, +4.0] | +1.9 [+0.6, +3.1] / +2.6 [+0.2, +4.9] | +1.2 [+0.6, +1.8] / +3.4 [+2.0, +4.7] | −0.7 [−2.8, +1.4] / −0.8 [−4.2, +2.5] | +0.1 [−1.7, +2.1] / −0.8 [−2.6, +1.1] |
| GRPO + verifier (stop) | +2.9 [−0.4, +5.7] / +4.8 [+1.1, +7.9] | +3.5 [−0.2, +6.5] / +5.4 [+0.6, +9.5] | +1.6 [+0.3, +3.0] / +4.3 [+1.1, +7.5] | −0.8 [−3.2, +1.6] / +0.6 [−2.4, +4.0] | +0.4 [−2.3, +3.2] / +0.2 [−2.5, +2.8] |
| EI + verifier | +1.0 [−2.5, +4.7] / +0.3 [−3.8, +4.7] | +0.8 [−4.5, +5.5] / −0.2 [−6.6, +5.6] | +0.5 [−1.8, +3.4] / +1.0 [−4.6, +7.9] | +3.7 [−1.3, +8.7] / +1.6 [−3.9, +7.2] | +6.9 [+1.0, +13.2] / +3.1 [−2.2, +8.8] |
| EI | +0.6 [−1.9, +3.7] / −0.1 [−3.6, +3.6] | +0.4 [−2.9, +4.1] / −0.7 [−6.1, +4.5] | −0.2 [−2.1, +2.4] / −0.2 [−4.3, +4.9] | +5.6 [+0.8, +10.9] / +4.4 [−1.2, +10.3] | +8.4 [+3.6, +13.1] / +5.1 [+0.5, +10.4] |

**Guided reads** (verifier redraws at test time), this recipe minus the merged recipe (§ 3):

| set | pass@1 | pass@5 |
|---|---|---|
| 13–20 | +0.0 [−1.5, +1.5] | −0.4 [−2.8, +1.7] |
| 16–20 | +0.5 [−1.6, +2.4] | +0.2 [−2.7, +2.8] |
| textbook72 | −1.0 [−4.7, +2.2] | −0.8 [−4.7, +2.5] |
| Charles149 | −1.6 [−3.6, +0.3] | −2.6 [−4.9, −0.6] |

**Also observed:**

- **hand_proved_14** (20–102-line hand proofs) is at the floor for every arm. This recipe reads 0.3 / 1.1%.
- **At matched sampled tokens per problem** (§ 5; Charles149, against EI):
  - +7.3 [+2.6, +12.4] points at 1k tokens;
  - +1.7 [−3.2, +7.2] at 8k.

  On 16–20 the recipe is level with EI at 1k–8k tokens.
- **How KL 0 was chosen** (§ 7, [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md), "Knob pilots"). It was one of nine
  single-seed L3 knob pilots on the merged recipe, at seed 0. The KL anchor was the only knob with a dose response on
  the long sets: 0 did better than 0.02, which did better than 0.1. The run-to-run noise band from the "nothing
  changed" rerun is about 1–1.5 points. KL 0 was then confirmed at seeds 1 and 2:
  - every generated set improved at every seed, at pass@5;
  - Charles149 was level.

## Caveats

Observed facts and interpretation are kept apart; interpretation is marked.

- **Selection.**
  - KL 0 was chosen on the generated sets (13–20, 16–20, 21–30) and on Charles149. The knob screening read only
    those sets, and the seed-1 and seed-2 confirmation used the same ones. So the gains on them are partly selected.
  - textbook72 was not read in the screening or the confirmation and was not used for selection. Its reads came
    after KL 0 had held at 3 seeds (2026-10-15, 11:06 am to 12:06 pm). On textbook72 the recipe is level with the
    merged recipe: −0.7 / −0.8 points, with both intervals including 0.
  - The merged recipe's own three components come from single-seed L3 pilots (seed 2). Those pilots had full reads,
    textbook72 included.
- **The KL 0 gain is in plain reads only.** In guided reads it is level with the merged recipe on 13–20, 16–20 and
  textbook72. On Charles149 at guided pass@5 it is lower: −2.6, with an interval of [−4.9, −0.6].
- **Tuning effort is unequal.**
  - GRPO had about 15 single-seed tuning pilots: the 5 L3 recipe pilots, the in-context feedback pilot and 9 knob
    pilots (one of them the noise control). Before those came arm changes such as redraw → stop and the half-rate lr.
  - EI had about 4 pilots: `rr`, `3p`, `ret0`, `t10`, and also `r2`. They started on 2026-10-15 at 8:09 pm. Their
    reads are on disk (`runs/eval/v2/*_ei_think_v_k*`) but were not analysed when this note was written.
  - So the EI rows above are close to untuned.
- **Against EI.**
  - On the long generated sets (13–20, 16–20, 21–30) the recipe is level with both EI arms. Every interval includes
    0.
  - At pass@1 it is ahead on Charles149, against EI (+8.4) and against EI + verifier (+6.9).
  - On textbook72 it is ahead of EI (+5.6). Against EI + verifier the lead is +3.7 [−1.3, +8.7], which is not
    resolved.
  - At pass@5, only Charles149 against EI excludes 0 (+5.1).
- **The gain over the merged recipe is small:** +1–2 points at pass@1 and +2–3 at pass@5 on the generated sets.
- **Rollbacks.** The recipe's safeguards rolled back some L3 runs:
  - seed 0 once, on the canary at it 29, restored to it 25, with the lr then halved;
  - seed 1 twice, on the think ban at it 35 and it 38, both times restored to it 30. Its final weights are it 30 plus
    one iteration at lr × 0.25;
  - seed 2 not at all.

  Of the merged recipe's L3 runs, only seed 2 rolled back, once. All three L2 runs ended with 0 rollbacks.
- **Reproducibility.**
  - GPU runs are not bit-reproducible; the rerun noise band is about 1–1.5 points.
  - The pools used here, L2, L3 and the 3-premise rewrites, were rebuilt on ynez. Their builders and rebuild note are
    on branch `leon/curriculum-rebuild-ynez`, which is not merged into `leon/main`.
  - The canary builder's ynez-rebuild option (`--skip_missing`) is on `leon/rebuild-canary-ynez`, also not merged.
  - The pools themselves live only on `/scratch` (`runs/pools/`).
- **Interpretation (untested).** The KL dose response suggests that, in L3, an anchor to the L2 model limits how far
  the policy moves toward the long, 3-premise problems. No run tests this directly; for example, KL 0 in L2 has not
  been run.

## Next steps

- Analyse the EI knob pilots' reads, so that the GRPO-vs-EI comparison is less one-sided in tuning.
- KL 0 in L2 as well. Leon (2026-10-15): this waits.
- Merge the pool builders (`leon/curriculum-rebuild-ynez`) and the canary option (`leon/rebuild-canary-ynez`), so
  that the recipe's inputs can be rebuilt from `leon/main`.
- Let `line.sh` take its code worktree as a setting instead of the fixed `nd-rl-v2` on `leon/grpo-v2`.
