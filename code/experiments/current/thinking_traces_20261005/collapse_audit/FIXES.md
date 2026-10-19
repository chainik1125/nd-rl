---
written_on: 2026-10-08
written_by: agent:claude
---

# Seed-0 collapse: the GRPO safeguards and the E1 script

This implements the monitoring and prevention items of the audit's §5 ([README.md](README.md)) in
`code/thinking/grpo.py`, plus the instrumentation E1 needs.

**Code.** On `leon/thinking` at `9b0e203`, uncommitted:
- changed: `grpo.py`, `ckpt.py`;
- new: `canary.py`, `grpo_io.py`, `code/tests/thinking/test_grpo_guard.py`, and
  [build_canary.py](build_canary.py) in this folder.

Development and tests ran on CPU only. Nothing was trained on a GPU, and E1 has not been run.

## Flags

Training is unchanged unless a flag is set. On the tiny end-to-end run, default flags give the same
weights at every iteration as HEAD's `grpo.py`, and the same value for every metric that already
existed. Two things do change by default:

- **Two new alarms are on**, as the audit asked: the capped-attempt alarm and the relative
  clip-low alarm. They add alarm lines and alarm checkpoints only. Pass 0 to turn either off.
- **Checkpoints are larger.** Every checkpoint now carries its training state, so it is about
  115 MB instead of 38 MB.

| flag | default | what it does |
|---|---|---|
| (always on) | | Logs `capped_frac`, `capped_by_budget` and `capped_kinds`: the share of attempts that a cap ended (action length, step count or context), whatever `--overlong` does with them. `overlong_frac` still counts only masked attempts, so it stays 0 under `fail`. |
| `--alarm_capped` | 0.05 | Alarms when any budget's capped share exceeds this. |
| `--clip_low_rel` | 2.0 | Alarms when clip-low exceeds this multiple of its median over the last 20 iterations (`clip_low_median`), after 5 iterations of warm-up, and only above 0.001. The absolute 1% alarm stays. |
| `--canary FILE`, `--canary_n`, `--canary_every`, `--canary_mult` | off, 1000, 1, 5 | Teacher-forces reference steps from long proofs after each update (see below). Alarms when the action NLL, P(`:`) or the top wrong-token probability exceeds `canary_mult` × its value before the first iteration (`canary_base.json`). |
| `--rollback_on_alarm`, `--max_rollbacks` | off, 3 | On any alarm: saves the alarm checkpoint, restores the last good checkpoint, halves the lr and continues. Details below. |
| `--resume CKPT` | | Continues a run exactly from a checkpoint's training state. `--init` is still the weight-only start. |
| `--start_iter S` | 0 | Numbers the iterations S+1 … S+`--iters`. Sampling seeds and checkpoint names follow the numbering. |
| `--kl_anchor_coef β`, `--kl_anchor_ref PATH`, `--kl_anchor_n`, `--kl_anchor_data` | off, –, 512, the canary file | Adds β · KL(π_ref ‖ π_θ) at every AdamW step (see below). Logs `kl_anchor` and `grad_norm_policy`, the gradient norm before the anchor's term is added. |
| `--dump_batches DIR` | off | Saves each iteration's training batch and AdamW state (see below). |
| `--targets_extra FILE`, `--targets_extra_frac f` | off, 0 | Draws round(f × `--prompts`) groups per iteration from a second target file, with its own curriculum. Logs `extra_*`, `main_pass1` and `curriculum_extra`. |

### Rollback

- The checkpoint it restores is the newest rolling or permanent checkpoint saved in an iteration
  that had no alarm, or saved after a rollback.
- A start checkpoint, `<name>_it<s>_rolling.pt`, is written first, so there is always something
  to roll back to.
- It restores the weights, AdamW, the curricula, the RNGs, the step count and the alarm
  histories. The iteration number keeps counting.
- `row['rollback']` records each rollback. A rollback that the cap blocks is printed.

### Resume

- **What is saved.** Each checkpoint holds the AdamW state, the iteration, the curricula, the RNG
  states and the rest of the loop state (`LoopState`). They sit under a top-level `train_state`
  key, outside `extra`, so EI and the effort expansion never copy them into the checkpoints they
  write.
- **Usage.** Re-run the same command with `--resume CKPT` added; the run continues to iteration
  `--start_iter` + `--iters`.
- **Bookkeeping.** Rows already logged for later iterations move to `metrics.pre_resume.jsonl`,
  and the old `args.json` is kept as `args.pre_resume.json`.
- **Exactness.** On CPU a resumed run gives bit-identical weights, AdamW state and metrics to a
  straight one: 2 + 2 against 4 iterations on the tiny model, and 1 + 1 against 2 on the real it39
  checkpoint. On GPU the state is restored exactly, but nondeterministic CUDA kernels can still
  make the runs differ.

### Batch dumps

- **File.** One per iteration, `DIR/<name>_it<i>_batch.pt` (see `grpo_io.py`).
- **Per item:**
  - the prompt, generated ids and loss mask;
  - the advantage, in float64;
  - the sampling policy's log-probs of the masked tokens;
  - the budget, the attempt and the minibatch.
- **Per attempt:** status, reward and mask. **Per group:** budget, target, source and whether it
  was skipped.
- **Per AdamW step:** the lr, the gradient norm and the anchor KL.
- **Optimizer state** before the iteration's first step, in bfloat16. The full-precision state is
  in the previous iteration's checkpoint.
- **Size:** an estimated 45 MB per iteration at E1 scale. The AdamW part is 38 MB (measured).
- `load_batch` and `unpack_items` read a file back.

## The canary file

**File.** `/scratch/cowifornia/nd-rl/think/runs/pools/canary_long.jsonl`, 9.7 MB, built by
`build_canary.py`; the report is `canary_long.jsonl.meta.json`. Each row is one reference proof
and keeps only its steps with ≥ 10 hypothesis lines (`:` tokens in the flat state). States and
actions are the Stage-1 data path's (`state_env.decompose`). A test checks that they match
`progtrace.trace_for` → `data.trace_pairs`.

**Source.** `runs/pools/long_cands.jsonl` (2,720 proofs) and `long2_cands.jsonl` (2,600 proofs),
13–20 lines each.
- 1,868 proofs have no state with ≥ 10 lines and are dropped.
- That leaves **3,452 proofs and 15,212 steps** (1,737 from `long_cands`, 1,715 from
  `long2_cands`).

| hypothesis lines in the state | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|
| steps | 7,794 | 4,445 | 2,043 | 711 | 185 | 32 | 2 |

**Exclusion check.** Renaming classes are compared with `data.class_keys`: the atom-renaming key,
or the premise-order-invariant key.

| check | sets covered | candidates hit |
|---|---|---|
| fresh index (`test_21_30.seal.build_index`) | hard12_test_1k and the hard12_test file it was drawn from; test_13_20 and its matched 1,000-item subset; test_21_30; textbook72 dev and train; both 600-item calibration sets | 0 |
| existing index `runs/pools/test_13_20/work/excl_index.pkl` (`test_13_20/excl.py`), counting only its sealed-set files | hard12, test_21_30, textbook72 | 0 |
| duplicate classes within the two pools | — | 0 |

The kept rows were re-checked against both indexes after writing: 0 hits.

**Canary and anchor split.** The trainer splits the file by proof, ordered by a hash of the proof's
name, so the split is the same in every run and every seed.
- The first proofs, up to `--canary_n` steps, form the canary set: 1,000 steps from 235 proofs.
- The rest (14,212 steps) form the KL anchor's pool, so the anchor never trains on the steps the
  canary reads.

**What it measures.**
- **Layout:** B0, i.e. `state <e0> <think> </think>` → `action <eos>`.
- **Distribution:** the sampler's masked action distribution at T = 1.
- **Names:** shifted by the audit's per-proof base, which is drawn from [0, 32].
- **Output:** the action NLL; P(`:`) at non-`:` targets, and its mean log; and the largest mean
  probability any single token gets at positions where it is not the target (`wrong_max`, plus
  the token itself).

**Values on existing checkpoints** (1,000 canary steps, CPU; observed):

| checkpoint | action NLL (nats) | P(`:`) at non-`:` targets | top wrong token |
|---|---|---|---|
| s0 it0 | 0.075 | 0.0004% | 0.62% `(` |
| s0 it39 | 0.116 | 0.0011% | 0.39% `(` |
| s0 it49 | 0.160 | 1.94% | 1.94% `:` |
| s0 it50 | 0.911 | 20.8% | 20.8% `:` |
| s1 it39 | 0.145 | 0.0001% | 0.50% `(` |
| s1 it49 | 0.149 | 0.0002% | 0.53% `(` |

**Reading (interpretation).** Take the base at it39, as E1 does, with the default 5× threshold:
- The P(`:`) alarm fires at s0 it49, which is 1,800× its it39 value.
- The generalised score is 4.96× at it49, just under the threshold, and fires at it50 (53×).
- The NLL alarm fires at it50 (7.8×).
- Seed 1 moves at most 1.3× on all three between it39 and it49.
- Measured from it0, s0 it39's P(`:`) is only 2.7× in these ≥ 10-line states, so an it0-based
  canary would not have fired by it39. The audit's 8× was for ≥ 12-line states.

The generalised score alone would have caught the drift one iteration later than P(`:`) did.
That is why the P(`:`) alarm is kept beside it.

## KL anchor, and β for E1

**Definition.** KL(π_ref ‖ π_θ) is the mean, over the action positions, of Σ_v π_ref(v)(log
π_ref(v) − log π_θ(v)).
- Both distributions are masked by the sampler's action ban (`sample.phase_bans`) and tempered at
  `--temperature`.
- `--kl_anchor_n` steps are drawn from the anchor pool at every AdamW step.
- The anchor's gradient is added before the gradient is clipped.

**Scale** (512 anchor steps, π_ref = s0 it0, CPU; observed):

| checkpoint | KL from π_ref (nats per position) | gradient norm |
|---|---|---|
| s0 it39 | 0.121 | 0.77 |
| s1 it39 | 0.165 | 0.91 |
| s0 it49 | 0.160 | 3.09 |
| s0 it50 | 0.79 | 41.4 |

For comparison, the policy gradient's norm in seed 0 is 0.018–0.020 per AdamW step over it30–47
and 0.030 over it48–50 (`metrics.jsonl`).

**Choice: β = 0.02.**
- At the restart, β · ‖∇KL‖ = 0.015, about 0.8× the policy gradient's norm, so RL still drives
  the update.
- In the it49 state the anchor's term is 0.062, 2× the policy gradient at that point. In the it50
  state it is 0.83, 28× — so the anchor grows sharply exactly as the attractor forms.
- The anchor's steps all have ≥ 10 hypothesis lines, which hard12 states almost never reach. It
  therefore constrains in-distribution behaviour only weakly.
- The cost: it also pulls long-state actions back toward it0 in general, and the KL of
  0.12–0.16 nats that both seeds have built up by it39 is ordinary drift.
- If arm (c) lowers the hard12 batch pass@1, the next value to try is 0.005–0.01.

## E1 script

`/scratch/cowifornia/nd-rl/think/runs/collapse_e1/e1.sh` has not been run. Launch it with:

```
mkdir -p /scratch/cowifornia/nd-rl/think/runs/collapse_e1/logs && \
nohup bash /scratch/cowifornia/nd-rl/think/runs/collapse_e1/e1.sh GPU_A GPU_B GPU_C \
  > /scratch/cowifornia/nd-rl/think/runs/collapse_e1/logs/e1.log 2>&1 &
```

**Arms.** Each restarts seed 0 from `grpo_T1v1e_it39.pt` with `--init`, for iterations it40–it55
(`--start_iter 39 --iters 16`). They run in parallel, one GPU each:

| arm | settings |
|---|---|
| a | the original `args.json` settings, passed explicitly |
| b | the same, at lr 7.5e-5 |
| c | the same, plus `--kl_anchor_coef 0.02 --kl_anchor_ref grpo_T1v1e_it0.pt` |

**Every arm:**
- runs the canary every iteration (1,000 steps);
- writes batch dumps to `<arm>/dumps`;
- saves a checkpoint, with its AdamW state, every iteration (`--save_every 1`).

**Then a 13–20 eval** of the it55 checkpoint, at B0 and B256:
- set: `runs/eval/matched/sets/t1320_n1000.jsonl`;
- flags: `--k 8 --arms plain --batch 1024 --seed 0 --temperature 0.8 --max_action 512
  --max_steps 96 --max_rej 10 --min_think 0`, as in `followup_20261007/lib.sh`.

**Guards.**
- Every launch waits for a free GPU (under 500 MiB used and no compute process), for up to 3 h.
- Training runs under `timeout 10800` with `runs/watchdog.py` (expected 1 h; OVERRUN at 1.5 h).
  Each eval runs under `timeout 5400`.
- The script stops if an input is missing or if `grpo.py` lacks the new flags.
- It records the commit, `git status` and the hashes of the changed files in `base_commit.txt`.
- Re-running it skips finished trainings and evals, and resumes a crashed arm from its newest
  checkpoint with `--resume`.

**Outputs** (under `runs/collapse_e1/`):
- `{a,b,c}/metrics.jsonl`, whose `canary` field holds the per-iteration values;
- `{a,b,c}/ckpts/e1_<arm>_it40…55.pt`;
- `{a,b,c}/dumps/`;
- `eval/<arm>_it55/`;
- `logs/`.

Disk use is about 2.6 GB per arm. Cost, from the audit's estimate: about 35 GPU-min of training
and 15 GPU-min of eval per arm, so about 2.5 GPU-h in total. The canary and the anchor add a few
seconds per iteration.

**Known deviations from the original run.**
- Fresh AdamW moments and the 10-step warmup.
- A fresh curriculum: the original run saved neither optimizer nor curriculum state.
- The canary's base is the it39 policy, not it0.

**What to read.**
- **(a) vs (b).** Does P(`:`) in the canary rise several-fold above its it39 value by about
  it49 in (a)? Does lower lr slow it, by roughly 2×, in (b)?
- **(c).** Do P(`:`) and the top wrong-token probability stay flat, and how much hard12 batch
  pass@1 does the anchor cost?
- **If (a) drifts.** The dumps and AdamW states let you replay each update from the previous
  checkpoint and split it by token class (§2.3b, §6 of the audit).

## Checks run

**Static checks.** `make -C code check` is clean: ruff, the format check and pyright report no
errors.

**Tests.** `code/tests/thinking`: 372 passed and 1 skipped. The skip is not from these changes.
- The 71 existing tests in `test_grpo.py` pass unchanged.
- `test_grpo_guard.py` adds 27 tests:
  - the capped share under `--overlong fail`, and its alarm;
  - the relative clip-low and canary alarms;
  - the canary rows against `trace_pairs`, and the canary layout, name shift and split;
  - the canary metrics against a direct computation;
  - the KL anchor: 0 when π_ref is the policy, with a finite gradient; its value, its linearity
    in β, and a descent step that lowers it;
  - an exact resume (2 + 2 = 4 iterations);
  - rollback, which restores the weights and AdamW state and halves the lr;
  - the batch-dump round trip, and that the dumped old log-probs are the sampling policy's;
  - long proofs mixed into the pool.

**One-off checks (not kept).**
- Default flags against HEAD's `grpo.py` on the tiny run, with `--overlong mask` and with
  `--overlong fail --len_target_coef 0.2`: the weights match at every iteration, and so do the
  pre-existing metrics.
- A CPU smoke of the E1 trainer command on the real it39 checkpoint: 2 iterations at a tiny batch,
  with the canary, the anchor against it0, the dumps and the Lean gate. A resume from it40 gave
  bit-identical it41 weights, AdamW state and metrics.

**E1 script.** Checked with `bash -n` only.

## Limits

- The canary scores only the B0 action. Long-context drift inside think blocks, which the audit
  found worse at B256, is not measured.
- After a rollback, the curriculum and RNG states are restored too. The next iterations therefore
  redraw the pairs the checkpoint would have drawn, with new sampling seeds, at the halved lr.
- No `--targets_extra` file exists yet. A 13–16-line pool for it needs its own exclusion check. It
  must not reuse the canary's proofs, or the canary would read training data.
