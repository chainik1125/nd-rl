---
written_on: 2026-11-10
written_by: agent:claude (orchestrator)
---

# Final-test plan for the fable_rl hand-back (fixed before any test result exists)

Written 2026-11-10 at 11:20 am PT, after the hand-back and before the first sealed run.
Checkpoints: `../final/final_s{0,1,2}.pt` and `../final/final_abl.pt` (md5 verified against `../final/MD5SUMS`).
Solver: `../final/solver` (commit a0914b4 per `COMMIT`). One run of each set per checkpoint and mode; no second try.

Choices that were mine to make:

1. **No wall-clock cut inside the solver** (`SOLVER_MAX_SECONDS=0`) in every run. The solver's default budget
   ends attempts by time; the reference models were read with no such cut, and compute is reported as generated
   tokens per attempt. The hand-back says the default budget roughly halved the success rate on its own long
   statements, so the default would measure the clock and not the method.
2. **Full mode (headline), K = 32, seeds 0/1/2 on the 832 and textbook72.** Labels `fable_full_s{0,1,2}`.
3. **Plain mode, K = 32, seeds 0/1/2 on all four sets**; the driver's own 30-minute projection rule decides
   between the full long sets and the 320 + 320 subsamples. Labels `fable_plain_s{0,1,2}`. The ablation
   checkpoint the same way, label `fable_ablplain_s0` (plain only: it has no verifier).
4. **Full mode on the long sets: the fixed 320 + 320 subsamples at K = 8**, labels `fable_fullL_s{0,1,2}`.
   Reason: only one temescal GPU is free and jobs run one at a time; the hand-back puts 320 theorems x 32
   uncut full-mode attempts at over an hour per set per seed (about 7 to 8 GPU-hours in all). K = 8 still gives
   unbiased pass@1 and pass@5 per theorem; the cost is noisier per-theorem estimates, which the paired
   bootstrap over theorems carries. Chosen on time alone, before any result.
5. Order: 2, then 3, then 4, sequentially on one temescal GPU. Seeds pooled with `score.py --labels`.
6. Extra row, from the earlier ruling on solver-written premise lines: theorems with more than 3 premises are
   also reported on their own.

Scores come from `recheck.py` (Dan's Lean check against our statements), never from the solver's verdict.

## Incident, 2026-11-10 (written at 2:33 pm PT)

The step `fable_fullL_s2` (seed 2, full mode, long-set subsamples) was launched at 2:22 pm PT on
temescal GPU 0. At 2:30 pm another user's process started on that GPU and our watchdog killed our
job (the never-share rule). No score from this label had been computed or read. The partial rows
file and log were renamed (`*.killed_1430.*`) and the step was rerun unchanged (same checkpoint,
seed, K and settings) by re-running `run_final_queue.sh`. This is a restart of an interrupted run,
not a second test.
