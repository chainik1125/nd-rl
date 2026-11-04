---
written_on: 2026-11-04
written_by: agent:claude
---

# Pre-registration: GRPO on Dan's K12⁻ knockout bases

Frozen before any GPU run of a line. Its sha256 goes into `log.md` first. The only GPU work
before it was the base check (Dan's own round-1 draw, below) and a 10-iteration pilot on the full
base, the reference base. No knockout line had run.

## Question

Dan asked: "It would be cool to know if GRPO can actually learn proof-by-contradiction off the
0% proof-by-contradiction pretrain (where EI failed to learn it)."

The test runs GRPO on **his** knockout bases. These are the create-or-elicit `stage1_ko_s0–s2`,
trained by state_train.py at 13e41bbd on K12⁻ (`train_k12ko`, no σ proofs). His T1 expert
iteration on these bases reached 0, 3 and 0 of the 106 strict σ theorems by round 16 (x 5.84–5.93
on his rule), against his bar of 14. Everything else is held fixed: the base, the problems (his
4,495 rl_targets), the environment and its caps, the held-out set, the measure and the bar. So
the algorithm is what differs.

## Bases (checked; not rebuilt)

His checkpoints come from his bucket. Their md5s equal his registry rows (`log.md`).
- `stage1_ko_s0–s2`: lean_staten, 6 × 384 ALiBi, 9,560,832 parameters.
- `stage1_full_s3`: the reference, on full K12.

**Base check, with his code.** `danread.py` repeats his round-1 transfer draw: his
`state_ladder_ei.generate` / `eval_set.judge`, k 32, T 0.8, 256 tokens per action, 48 steps, his
noise seed and batch 1024, on all of transfer_track832.

| base | strict 106 (ours / his) | transfer 832 (ours / his) |
|---|---|---|
| ko_s0 | 0 / 0 | 303 / 317 |
| ko_s1 | 0 / 0 | 305 / 313 |
| ko_s2 | 0 / 0 | 317 / 327 |
| full_s3 | 17 / 13 | 320 / 319 |

The knockout bases solve none of the 106 on either draw. full_s3 lands at 17 against his 13, and
the transfer counts differ by 1–14 theorems: our RTX 3090 sampling does not reproduce his A40
stream exactly, so every count here is a fresh draw from the same distribution. His six full
bases ranged from 2 to 19 at round 1.

## Algorithm: one GRPO recipe for every line

- **Code:** `code.thinking.grpo` at f35d3daf (Leon's pipeline, the knockout_grpo L3 code). It
  carries the `snap_iters` patch and the `danbase_log` patch, which only logs the sampled
  generated and prefill tokens, the update's state and action tokens, and peak memory. Dan's
  lean_staten checkpoints load as they are: vocabulary 111, no think tokens, so `--budgets 0` and
  no `--effort`.
- **Flags** (`common.grpo_argv`):
  - Sampling: 256 prompts × group 16 per iteration, T 1.0.
  - Optimiser: lr 1.5e-4, constant after a 10-iteration warmup; PPO clip 0.2 / 0.28 and dual
    clip 3; 4 minibatch steps per iteration.
  - Reward and loss: unlikeliness β 0.25; `--overlong fail`; `--arm logical --max_rej 0
    --rej_penalty 0.1`; KL anchor 0.
  - Environment: Dan's caps, `--max_action 256 --max_steps 48`.
  - Guards: the canary guard with rollback (max 3).
  - Checkpoints: a model-only snapshot every 10 iterations, and a rolling checkpoint with the
    full training state every 5.
- **The think-ban guard is moot here.** There are no think blocks: `ban_think` was 0 throughout
  the pilot, and there were no alarms.
- **Where it departs from Dan's EI.** These differences are the algorithm itself:
  - on-policy policy gradient instead of fine-tuning on accumulated verified proofs;
  - no pretraining replay (his EI mixes 20,000 K12 records into every round);
  - training samples at T 1.0 (his at 0.8); every read samples at 0.8, as his did;
  - 4,096 rollouts per iteration, against his 143,840 per round.

## Lines

| line | base | problems | seed | x grid (his ladder) | role |
|---|---|---|---|---|---|
| dko_s0, dko_s1, dko_s2 | ko_s0–s2 | Dan's rl_targets (4,495; 2.7 % strict σ-covered) | 0–2 | la_ko_s0–s2 | **primary** |
| dkoour_s0, dkoour_s1, dkoour_s2 | ko_s0–s2 | our L2_pool (5,000, 10–12 lines; 39.5 % strict σ-covered) | 0–2 | la_ko_s0–s2 | secondary: does GRPO learn σ at all when the problems reward it? |
| dfull_s3 | full_s3 | Dan's rl_targets | 3 | la_full_s3 | reference |

- **Our pool.** L2_pool is the knockout_grpo L2 pool. Every prompt builds a state in Dan's
  environment, and none overlaps the 106 or transfer_track832 by key.
- **Launch order:** dko_s0, dko_s1, dko_s2, dfull_s3, dkoour_s0, dkoour_s1, dkoour_s2.

## Measurement: on his chart

**Reads.**
- **When:** the base (iteration 0) and every 10-iteration snapshot.
- **How:** his code (`danread.py`): `state_ladder_ei.generate` and `eval_set.judge`, T 0.8,
  256 / 48.
- **What:** k 64 on the 106 strict σ members, noise seed `seed*1000 + it + 500`, batch 256.
- **Kept per read:**
  - the first 32 attempts, which form Dan's k 32 draw;
  - all 64, which give E32 (unbiased pass@32);
  - the solved members' σ flags.

**x.** D is the base's Stage-1 tokens, from his registry (`dan_axis.json`).
- **His rule (primary):** 2 × (generated + re-prefilled state tokens) of sampling, plus 6 ×
  (state + action tokens) of the update, divided by 6D.
- **Ours (knockout_grpo's R):** (6 × trained + 2 × sampled generated tokens) ÷ 6D, with prefill
  excluded.
- **Every metrics row counts,** including rolled-back iterations. The canary and the reads are
  excluded, as he excluded his eval phase.

**Primary measure: his.** It is the cumulative count of the 106 solved by k 32 draws, against
his bar of **14**. To match his number of draws to compute, there are 16 matched draws, one per
round of the same-seed ladder (`merge.py`):
- round 1 is the base read;
- round r ≥ 2 is the latest read with x_dan ≤ his x_r;
- the count is the union of the matched draws' solved members.

**Reaching his bar.**
- **First crossing:** the x_dan, and x_ours, of the matched read whose union first reaches 14.
- **Reached:** a crossing within the line's budget.
- **Not reached by x:** the union stays below 14 to the end.

**Secondary measures.**
- the snapshot first-32 count and E32 per read;
- the σ share of the distinct Lean-accepted training proofs, by Dan's `coe_sigma.flags`
  (σ = sigma_chain or sigma_strict on the written proof) on each segment's LEAN_GATE_DUMP;
- pass@1 reward, entropy and the canary.

## Stop rule and budget

A line ends at the first of three events.
- **Compute:** x_dan reaches his same-seed ladder's round-16 x, which is one ladder (5.80–5.93).
- **Time:** 15 GPU-h of its segments' wall time.
- **Reached and held:** his measure has crossed, and the snapshot first-32 count is ≥ 14 at 3
  consecutive reads, the last at ≥ 2 × the first's x_dan.

**Pilot.** It ran on the full base, 10 iterations on temescal's RTX 3090.
- Speed: 42 s per iteration; x_dan 0.0081 and x_ours 0.0017 per iteration; peak 8.4 GB.
- A read (k 64 on the 106, batch 256) takes 95 s and 3.7 GB.

**Projection.**
- A full-length line runs about 720 iterations: 8.4 h of training plus about 72 reads. That is
  about 9–10 GPU-h.
- Seven lines, with the reference stopping early, come to about 60 GPU-h.
- The limit is 120 GPU-h in total. I stop and ask if the projection passes it.

## Machines and supervisor

`sup.py` runs on serrano until every line has ended. Every 90 s it checks each line:
- the host is reachable;
- the process is alive;
- the log is growing (no progress for 25 min counts as dead);
- no other user's process is on its GPU.

**Respawning.**
- A dead line is respawned as a new segment, from its newest checkpoint with training state.
- The segment can go on any GPU with no process of anyone, on any machine:
  - serrano: any of its 5 GPUs;
  - temescal: at most 2 of ours;
  - ynez: when it is reachable.
- When another user's process appears on a line's GPU, the line moves at its next rolling
  checkpoint.

**Copies.**
- **Backups:** each rolling checkpoint is copied one way to a backup on another machine, keeping
  the last 2, with the segment's metrics, reads and dump.
- **Respawns:** a respawn starts a new segment directory. Nothing is ever copied back into a
  live directory. The superseded rows and reads (lost iterations) are dropped by `merge.py` and
  logged.

## Predictions

These are made before any knockout line is read.

| | prediction | probability |
|---|---|---|
| P1 | primary: no dko seed reaches his bar (14, cumulative) within one ladder | 70 % |
| P2 | primary: σ stays under 1 % of the accepted training proofs on all three dko seeds | 75 % |
| P3 | secondary: at least 1 of 3 dkoour seeds reaches his bar within one ladder | 35 % |
| P4 | secondary: σ exceeds 5 % of the accepted training proofs on at least 1 dkoour seed | 50 % |
| P5 | reference: dfull_s3 reaches his bar at its first matched draw after the base (round 2) | 80 % |
| P6 | pass@1 reward on Dan's pool passes 0.8 within 50 iterations on every dko seed | 70 % |
