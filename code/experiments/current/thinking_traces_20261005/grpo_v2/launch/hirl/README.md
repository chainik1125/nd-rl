---
written_on: 2026-10-30
written_by: agent:claude
---

# think_hirl: large-k reads and the low-RL arm for the high-RL regime paper (ynez)

Leon approved this queue on 2026-10-30, as relayed by the coordinating agent. It runs on ynez from
the clone `/scratch/cowifornia/nd-rl/think_hirl/nd-rl-hirl`, on branch `leon/high-rl-reads`.
The branch starts at `79105ffe`, the commit `grpo_think_vs_mini20` trained at. The clone is not
pushed. The data tree is `/scratch/cowifornia/nd-rl/think_hirl/`, holding copies (md5-checked) of
the serrano checkpoints, pools and sets it needs. ynez's older tree `think/` is only read: its
venv's python and `~/.elan`.

The analysis this serves, and the RL / Stage-1 FLOP ratio, are in
`code/experiments/current/high_rl_passk_20261030/README.md` on `leon/high-rl-paper`.

## Files

| file | what |
|---|---|
| `hirl_plan.py` | writes `queue/jobs.jsonl`: the J1-J4 jobs in priority order (appends new ids only) |
| `hirl_queue.py` | `worker GPU`: one per GPU, GPU-gated, on the shared job file; `status`; `merge GROUP` |
| `hirl_compute.py` | writes `lowrl/compute.tsv` (RL / Stage-1 FLOP ratios of the low-RL checkpoints) |
| `../watchdog.py` | per job: STALL / EXIT / OVERRUN lines in `logs/watchdog.log` |

`code/thinking/grpo.py --save_steps` (opt-in) writes model-only checkpoints after listed AdamW steps,
and a compute row per step in `steps.jsonl`. It leaves training unchanged:
`code/tests/thinking/test_grpo_save_steps.py` checks it against `test_grpo_pilots.py`'s c4b8548
`plain` digests.

## Reads

Every read is `code.thinking.eval` with line.sh's `read1` flags: plain arm, B64, T 0.8,
`--batch 1024 --max_action 512 --max_steps 96 --max_rej 10`.

- **Chunks.** A large-k read is made of k64 chunks, each one eval call with `--seed` = the chunk
  index. Chunks are merged per theorem in seed order into `eval/final/<read>/`, which holds
  `rows_plain_B64.jsonl` (k = the sum), `summary.json` (eval.summarize's `all` cell, recomputed)
  and `chunks.json` (provenance).
- **Reproduction chunk.** The base S1g3_s0 and the L3 end read chunk 0 on charles173 in its
  three files. That is the existing `charles173_*_plain_k64` read's exact command. The merge keeps
  only the 149 Charles149 theorems.
- **Charles149** is `charles_clean.jsonl`; its prompts equal the charles173 rows of the same name.
- **t1320s250** is 250 of the 986 valid 13–20 theorems (`matched_eval/figs.py:valid_1320`):
  `sorted(random.Random(20261030).sample(sorted(valid), 250))`, rows in `t1320_n1000.jsonl`
  order. The ids are in `sets/t1320s250_ids.txt` and the rule in `sets/t1320s250_meta.json`.
- **J3** reads are single eval calls, named as line.sh names them.

## Queue

- **Job file.** `queue/jobs.jsonl` lists the jobs in priority order. A worker takes the first job
  that has no state file and whose inputs exist; train-job dependents wait until the train job
  is `done`. State files are `queue/state/<id>.json`. Delete a failed one to requeue it.
- **Editing.** Edit `jobs.jsonl` only under `flock queue/jobs.lock`.
- **Concurrency.** `queue/slots_gpuN` sets the concurrent jobs on GPU N. A train job runs alone.
- **Stopping.** `queue/STOP` or `queue/STOP_gpuN` stops new launches; a worker exits once it is
  idle.
- **Gate.** Checked right before every launch: no compute process on the GPU except the worker's
  own jobs, and under 500 MiB in use besides them. `--free_for MIN` adds a free streak before a
  GPU's first job.
- **Claiming.** A worker claims its GPU with `mkdir locks/gpuN`.
- **Bounds.** Every job has a hard `timeout` (chunks: 4x the estimate, at least 30 min) and its
  own watchdog.
- **No overwrites.** A partial output dir is moved aside to `*.crashed_<UTC>`.
- **Status.** `STATUS.md` is rewritten every 5 min and after each job.

    ssh -i ~/.ssh/id_ynez -o IdentitiesOnly=yes ynez 'cat /scratch/cowifornia/nd-rl/think_hirl/STATUS.md'

## The low-RL arm (J2)

- **`grpo_think_vs_mini20_kdense_s0_L2`.** `grpo_think_vs_mini20_s0_L2`'s argv with the same
  files (copies), `--iters 8` instead of 20, and `--save_steps 1,2,3,4,6,8,12,16,24,32`. Each
  iteration has 4 AdamW steps, so iteration i ends at step 4i.
- **`grpo_think_vs_mini20_kp16_s0_L2`.** The same, with `--prompts 16` instead of 256 (16 × 16
  attempts an iteration; the 4 minibatch steps stay), `--iters 12`, and `--save_steps`
  through 48.
- **No other changes.** Both keep the KL anchor's 512 steps per AdamW step, the 10-step warmup and
  the safeguards.

## Blocks, in queue order

- **T and J1.** T is the first base chunks (timing). J1 is the base and the L3 end on Charles149
  at k1024 and on t1320s250 at k512.
- **J2.** The two training runs above, on GPU 1.
- **J3.** S1g3 s1 and s2 get `s1g3_ckreads`'s reads. textbook72 is read at B64 k64 for S1g3 s0-s2.
- **J4a.** Charles149 at k1024 along the RL axis, in log-spaced RL-compute order.
- **J4b.** t1320s250 at k256 along the RL axis.
- **J4c.** J4b extended to k512, last in the queue. It reuses J4b's chunks 0-3 and adds 4-7.

## Results so far (2026-10-30)

- **Reproduction.** The existing `charles173_s1g3_s0_plain_k64` and
  `charles173_grpo_think_vs_mb_kkl0_s0_plain_k64` reads reproduce bit-for-bit on a ynez 3090.
- **mini20 rerun.** Iteration 1's rollouts are identical to the original's. The first update
  differs at float precision, so later iterations are a statistical replicate; see
  `STATUS_notes.md` on ynez.
