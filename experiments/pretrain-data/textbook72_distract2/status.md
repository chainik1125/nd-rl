# status: textbook72_distract2

- 2026-10-09 21:30 UTC-7: proposal committed (632a2c9). Dan's worktree re-created at ~/.cache/nd-rl/fork_dan (0c492902);
  checkpoints la_T1_best12_s{0,1,2}_r8 and stage1_best12_s0_b1200 fetched (s0_r8 sha256 a992485e… matches the
  2026-10-02 record). Sampler smoke test: 7 of 8 K12 samples verified.
- Distracted-72 builder running (minimal length first, then the shortcut check, 10 s each). First build without the
  minimal-length step placed distractors on 59 of 72 (13 "failed": shortcut flagged against the non-minimal reference);
  discarded, rebuilt.
- 21:50: distracted set built: 71 of 72 placed (1 problem: every draw either unsatisfiable or a shortcut; left out,
  the comparison runs on the 71), shortcut status no_shortcut 44 / unknown 27 (minlen timeout at 10 s); minimal length
  found for 45, reference used for 15, timeout 11. Premises after distraction: 2: 16, 3: 30, 4: 22, 5: 3 (25 cross
  to four or more). Launched: Robbie d384 and d768 (eval_ckpt.py, k 32, T 1.0), Dan la_T1_best12_s0_r8 and
  stage1_best12_s0_b1200 (env sampler, k 32, T 1.0). Seeds 1 and 2 of Dan's r8 follow.
- 22:40: Charles asked to pause. All four launched jobs had already finished (Robbie d384 and d768; Dan r8 seed 0 and
  Stage-1); nothing was killed, nothing is running. Dan's r8 seeds 1 and 2 NOT launched. Results in results.md.
PAUSED 2026-10-10 05:40 UTC (seeds 1 and 2 of Dan's r8 pending; relaunch with the command in
code/pretrain_data/textbook72_distract_eval_dan.py for --ckpt la_T1_best12_s1_r8 and _s2_r8)
