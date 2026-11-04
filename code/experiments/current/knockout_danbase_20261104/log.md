---
written_on: 2026-11-04
written_by: agent:claude
---

# knockout_danbase: run log

GRPO on Dan's own K12⁻ knockout bases (create-or-elicit `stage1_ko_s*`), the bases on which his
EI did not recover σ, so that only the RL algorithm differs. Asked by Dan ("It would be cool to
know if GRPO can actually learn proof-by-contradiction off the 0% proof-by-contradiction pretrain
(where EI failed to learn it)"); approved by Leon ("run this too also some of the machines have
been going down so just monitor the gpus and respawn dead processes on new gpus"). Branch
`leon/knockout-grpo`. Artifacts: `/scratch/cowifornia/nd-rl/knockout_danbase/` (KD below).

All times UTC.

## 2026-11-04

- **20:12Z — machines.** serrano: 4 GPUs busy with other users (marie on 3, cipeng on 1), over
  the PSU count of 3, so nothing launches there until ≤ 2 are busy. ynez: no route to host.
  temescal: up, frankyang on GPUs 0 and 1, GPU 2 free (≤ 2 of ours allowed there).
- **20:15Z — bases, not rebuilt.** Dan's checkpoints are in his bucket; copied read-only with
  `hf buckets cp` to `KD/bases/`; md5 equal to his registry's `ckpt_saved` rows:
  - `stage1_ko_s0.pt` 9ee0c620f8b749a3a1956a3f93be7750
  - `stage1_ko_s1.pt` 3b0f4ba929029870a2e13057ab10e063
  - `stage1_ko_s2.pt` 1518b58e67cd96df4d4a3fe13720fce9
  - `stage1_full_s3.pt` 3a3c86380be766eeb79f979b7cd1e0af
  All from state_train.py at 13e41bbd, lean_staten, recipe "best" with its 1,200 s budget (the CLI
  says 6,000 steps; the checkpoints record about 24,000: 24,248 / 23,977 / 24,153 for ko_s0–s2),
  ko on `data/coe/train_k12ko.jsonl`, full on `data/kh/train_k12.jsonl`. Seeds 0–2 are those of his la_ko_s0–s2 ladders; full_s3 is
  his la_full_s3's base.
- **20:3xZ — serrano rule changed (Leon, via the coordinator):** "you can use up to all of the
  gpus on serrano". The ≤ 3-busy PSU count is lifted; any serrano GPU with no other user's
  process, up to all 5, checked continuously, never shared. 200 W caps and GPU settings
  untouched; LEAN_GATE_WORKERS=8 per job. A serrano hang or reboot is a downed host for the
  supervisor and is reported.
- **20:2xZ — Dan's code and data.** His create-or-elicit code is on his fork (read-only clone
  `/scratch/cowifornia/nd-rl/think/nd-takehome`, branch `origin/dan_create-or-elicit` at
  624982dc). `git archive 13e41bbd` to `KD/dan_code_13e41bbd/` (the ladder run's commit), plus
  `data/coe/{transfer_track832.jsonl, sigma_heldout_strict.txt, heldout1000.jsonl,
  families.json}` and `coe/{coe_sigma.py, coe_rounds.py}` from the branch tip. rl_targets md5
  365450be (4,495), sigma_heldout_strict md5 388afa0f (106). His ladder settings (registry):
  rounds 16, k 32, T 0.8, max_action 256, max_steps 48, batch 1024, retain 20,000.
- **20:24–20:49Z — base check** (serrano GPU 0, gated; `basecheck.sh`, `danread.py`): his round-1
  transfer draw with his code, seed, batch 1024 and set. Strict 106: ko_s0–s2 0 / 0 / 0 (his
  0 / 0 / 0), full_s3 17 (his 13). Transfer 832: 303 / 305 / 317 / 320 (his 317 / 313 / 327 /
  319). 281–304 s each. CUDA allocator OOM warnings at batch 1024 (17.5 GB), no failure.
- **20:3xZ — GRPO path** (two read-only searches). Dan's own `grpo_state.py` (his fork,
  grpo-state / grpo-best) checkpoints only at round boundaries, without RNG state; Leon's
  `code.thinking.grpo` at f35d3daf loads lean_staten checkpoints as they are (vocabulary 111, no
  think tokens: budgets collapse to 0, the think-ban bans are empty) and writes rolling
  checkpoints with the full training state, which the supervisor's respawns need. Chosen: f35d3daf
  + the knockout `snap_iters` patch + `danbase_log_f35d3daf.patch` (logging only: sampled
  generated and prefill tokens, the update's state and action tokens, peak memory). Code tree
  `KD/snap/f35d3daf_danbase`. No other change; Dan's code is used unchanged for the reads.
- **20:33–20:41Z — pilot** (temescal GPU 2, RTX 3090; `pilot.sh`): 10 iterations on full_s3 with
  his rl_targets, the PREREG recipe. pass@1 reward 0.396 → 0.801; entropy 0.060 → 0.038; groups
  without signal 1 → 27–36 %; no alarm, `ban_think` 0 throughout (the think-ban guard is moot
  without think blocks); canary nll 0.063 → 0.073. 42 s per iteration (median), peak 8.4 GB;
  x_dan 0.0081 and x_ours 0.0017 per iteration. A read of its iteration-10 snapshot (`danread.py
  --only106 --k 64 --batch 256`): 95 s, 3.7 GB; first-32 28 of 106, E32 27.6. Our GRPO snapshots
  load in Dan's `load_ckpt` (cfg identical to the base).
- **Pools.** Every prompt of rl_targets and L2_pool builds a state in Dan's environment; neither
  overlaps the 106 or transfer_track832 by key.
- **PREREG.md frozen:** sha256 `dac8e2df131008116990730917170c2915e46c7f81b41bd24a21939662dcdb31`. Projection: about 720 iterations per full-length
  line (one ladder, x_dan 5.8–5.9), 9–10 GPU-h each; about 60 GPU-h for the seven lines.
