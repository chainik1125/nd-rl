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
- **20:46Z — supervisor started** (`sup.py` on serrano, under `timeout 96h`, every 90 s;
  `KD/sup/`). dko_s0 → serrano GPU 0, dko_s1 → temescal GPU 2. My error, fixed at 20:53Z: the
  launch command backgrounded a whole `&&` list, so each launch waited for its 60 s timeout (the
  jobs started all the same); now the job alone is backgrounded.
- **20:50:45Z — respawn test** (coordinator asked for respawns; done once on a live line, which
  the coordinator has since ruled out: later tests use dummy jobs). I SIGKILLed dko_s1's GRPO
  process at iteration 6 (as an OOM kill would). dline wrote `end.json` failed (rc −9). It exposed
  a bug, fixed: the dead segment's GPU still counted as ours, so the respawn waited while the
  pending dko_s2 took temescal GPU 2 (20:51Z). I stopped dko_s2 at iteration 8 (20:57Z, `STOP`;
  it restarts from its base as seg2, merge.py drops seg1) and the supervisor respawned dko_s1 as
  seg2 on temescal GPU 2 from its iteration-5 rolling checkpoint (20:58Z); it resumed at iteration
  6, no logged iteration lost. Also fixed: an orphaned GRPO of a dead segment (dline gone) is
  killed before a respawn.
- **21:02Z — standby supervisor and lease** (coordinator: the supervisor must not be a single
  point of failure). `sup.py --standby` on temescal (under `timeout 96h`) writes a heartbeat every
  90 s and takes over only when serrano has not answered ssh for 10 min **and** serrano's lease
  (its heartbeat, renewed every pass on both machines) is older than 10 min; it then acts with
  epoch + 1 from serrano's last copied state. Every action on a line (launch, respawn, move,
  stop) checks the lease first; a supervisor that sees a higher epoch on its peer yields and
  exits; a supervisor that starts acting, or sees a host answer again, ends our segments its
  state does not know. Tested on a dummy tree (simulated hosts): passive while serrano answers,
  passive while the lease is fresh, takes over when both fail, and serrano's old supervisor
  yields to the higher epoch. Host locality now follows the running machine (gpu.py, merge.py).
- **ynez** rejoins the GPU pool on its own: every pass polls it, and a free ynez GPU (no process
  of anyone) is used like any other after the code and data are copied there; the never-share
  check runs every pass on every host and every 60 s inside each segment.
- **20:58:02Z — dko_s0 seg1 OOM** (serrano GPU 0): GRPO and the iteration-10 read shared the GPU
  and the caching allocators filled it ("Tried to allocate 950 MiB … 924 MiB free"); the
  supervisor respawned it as seg2 from its iteration-10 rolling checkpoint (20:59Z; 1 logged
  iteration lost). Fix (21:05Z): segments run GRPO and reads with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`; a read
  is capped at 30 % of the GPU (`torch.cuda.set_per_process_memory_fraction`) and uses batch 128
  (a different but equivalent draw); a read that fails 3 times is marked `it<i>.failed` (a
  gap). Both running lines were moved onto the new segment code at their next rolling checkpoint
  (`STOP_AT_CKPT`): dko_s1 seg3 (21:10Z) and dko_s0 seg3 (21:12Z), from iteration 20, 0 lost.
- **21:05Z — process matcher narrowed** (coordinator: never touch other processes). The
  supervisor now acts only on processes from `ps -u $USER` whose command line is a python running
  `dline.py --line <dko_s*|dfull_s3|dkoour_s*> --seg <k>` with cwd this folder, or `-m
  code.thinking.grpo … --name <tag> --out KD/runs/<tag>/seg<k>` with cwd our code tree; it kills
  by PID. This also fixes a bug: the old liveness check (`pgrep -f` over ssh) matched its own
  shell, so a dead segment could look alive (deaths were still caught by `end.json` and the
  25-min stall rule).
- **21:17:39–21:18:0xZ — my error: two supervisors on serrano for about 20 s.** A failed lint
  check skipped the `pkill` of a `&&` chain but not the restart after it. The older one was
  stopped; neither acted in that window (no events, state unchanged). Fix: a supervisor takes an
  exclusive lock (`KD/sup/sup.lock`) and a second one exits at once (tested), and restarts go
  through `KD/sup/restart_sup.sh` (stop, wait until gone, start). First crossings of Dan's bar are
  now supervisor events.
- **21:14–21:40Z — my error: reads failed.** `numbers.py` (then the analysis script, created in
  this folder at about 21:14Z) shadowed Python's `numbers` module for `danread.py`, which runs
  from this folder, so torch failed to import and every read failed: dko_s0 it 30 / 40, dko_s1
  it 30 / 40 / 50, the first snapshot (it 10) of dfull_s3, dko_s2 seg2, dkoour_s0 and dkoour_s1,
  and the base reads (it 0) of dfull_s3, dkoour_s0, dkoour_s1 (21:26Z) and dkoour_s2 (21:35Z).
  Fix (21:40Z): renamed to `tables.py` on every machine, the `.failed` markers removed so the
  readers retry the snapshots (kept), and the four base reads run once with the segments'
  settings (`baseread.sh`, on each line's own GPU; 21:4xZ). The reader now also retries a
  missing base read. No training was affected.
- **21:39Z — dfull_s3 (reference) at the bar at round 1:** its base read (k 64 on the 106, first
  32) solved 15 ≥ 14; the snapshot first-32 counts were 28 / ~31 / 34 at iterations 10 / 20 / 30
  (x_dan 0.08–0.23), so the hold rule was met and the supervisor stopped it (21:58Z, 33
  iterations, 0.54 GPU-h). P5 (reach at round 2) misses: it was at the bar before RL.
- **22:17Z — dkoour_s2 moved** (temescal GPU 1): its segment saw another user's process
  (BenPham) on its GPU, ended at the next rolling checkpoint (it 60) and was respawned as seg2;
  by then BenPham's process was on GPU 0 only and GPU 1 had no other process, so seg2 went back
  to GPU 1 (22:23Z; 0 iterations lost).
- **22:3xZ — GPU-h of an ended segment** now uses its end time (recorded by the supervisor, and
  the segment's last files are collected when it ends); dfull_s3's had kept counting.
- **23:5xZ — merge.py clarification** (no effect on any stop decision; the supervisor uses the
  running-line rule): a matched draw for round r exists only if the line ran to his x_r. For an
  ended line the code had re-used its last read for every later round (dfull_s3 showed 16 draws
  though it ended at x_dan 0.26); now an ended line has draws up to the last x_r it reached.
- **23:50Z — status:** x_dan 1.0–1.35 on the six knockout lines (3 h of training); his measure
  0–1 on every line; σ in 0.04–0.06 % of the accepted training proofs on Dan's targets and 0.00 %
  on our σ-rich pool. dkoour_s1 moves off ynez GPU 3 (frankyang's process appeared).
  Projection: dko_s0 (serrano, the slowest) reaches 15 GPU-h near x_dan 4.9 (the GPU-h cap ends
  it before his round-16 x); the others reach his x_16 at 13–15 GPU-h; about 85 GPU-h in all.
- **23:55Z — dkoour_s1 waiting.** It ended at its iteration-145 rolling checkpoint after
  frankyang's process appeared on ynez GPU 3; no GPU qualified (serrano 1–4 marie / cipeng,
  ynez 1 and 3 frankyang, temescal at its cap of 2 of ours). Coordinator: keep the temescal cap
  at 2 (a third job there put the primary dko_s1 at risk in the past); dkoour_s1 waits for the
  first qualifying GPU on any host.
- **00:17Z — dko_s2: a move requested** (agentexperts' process on ynez GPU 0). It hung: the
  segment (started before 00:3xZ) waits for a *rolling* checkpoint, and iteration 160 alarmed
  (an alarm checkpoint instead). By 00:29Z the other process had gone; the coordinator asked to
  cancel moves no longer needed, so the move was cancelled (00:30Z; no iteration lost; dko_s2
  stays on ynez GPU 0). The supervisor now cancels such moves itself, completes a move from an
  alarm checkpoint after 8 min, and dline's moves count alarm checkpoints.
- **00:30Z — coordinator's rules added to the supervisor** (tested on a dummy state first):
  primary lines (dko_*) get GPUs before secondary ones (dkoour_*); a waiting primary line
  preempts the lowest-priority running secondary line at its next checkpoint; GPUs listed in
  `KD/sup/reserved.json` are never launched onto, moved onto or reclaimed, and do not count in a
  cap (an unreadable file reserves everything).
- **00:30:44Z — ynez GPU 2 reserved for circuit_stage3** (Leon approved a circuit-diffing stage-3
  run; coordinator): `reserved.json` {"ynez": [2]} (copy: `reserved_ynez2.json`). dkoour_s0 got an
  availability stop and ended at its iteration-170 rolling checkpoint (00:37:07Z, 0 lost); GPU 2
  (GPU-d6327ec5-dcea-859e-51cf-875c92cbc0a8) had no process of ours at 00:37:23Z. dkoour_s0
  waits behind the primary lines with dkoour_s1.
- **Deadline (Leon, via the coordinator):** done by 11:00Z, no later than 16:00Z; a dkoour line
  not at its cap by 15:00Z is recorded at its last read as "stopped at the deadline". Primary ETAs
  (00:31Z rates): dko_s1 about 10:30Z (his x_16), dko_s0 about 11:50Z (15 GPU-h at x_dan about
  4.7), dko_s2 about 12:20Z (15 GPU-h at x_dan about 5.8).
- **04:20Z — dko_s0 moved and came back** (sudarsankg's process on serrano GPU 0): it ended at
  its iteration-340 rolling checkpoint (04:28Z) and waited; the supervisor preempted dkoour_s2
  (temescal GPU 1) for it (04:28:52Z); by 04:31Z serrano GPU 0 was free again and dko_s0
  resumed there as seg4 (0 iterations lost). The preemption was then no longer needed: its
  `STOP_AT_CKPT` was removed before dkoour_s2 reached a checkpoint, and the supervisor now
  cancels a preemption whose primary no longer waits (04:3xZ).
- **04:36:56Z — global cap of 4 GPUs** (Leon, via the coordinator: never more than 4 of our GPUs
  at once across serrano, ynez and temescal, counting every GPU with any of our processes,
  other agents' included, and reserved GPUs). `sup.py` counts it live every pass (our GPU
  processes on the three hosts, reserved GPUs, our lines' assigned GPUs; an unreadable host
  counts its last-seen ours), refuses any launch, move or respawn at 4, and when over 4 ends the
  lowest-priority running line (secondary first) at its next checkpoint. Tested on a dummy state
  (5 in use → dkoour_s2 stopped; at 4 no launch; a released reservation frees a slot; an
  unreadable host fails closed). Applied at once: 5 in use (serrano 0; temescal 1, 2; ynez 0, 2),
  so dkoour_s2 ends at its next checkpoint and waits.
- **04:42–04:44Z — at the cap.** dkoour_s2 ended at its iteration-350 rolling checkpoint
  (04:42:57Z, 0 lost) and waits; dko_s0 (sudarsankg back on serrano GPU 0 since 04:33Z) ended at
  its iteration-345 checkpoint (04:42:27Z) and resumed as seg5 on temescal GPU 0 (04:44:01Z, 0
  lost). In use at 04:44Z: temescal 0 (dko_s0) and 2 (dko_s1), ynez 0 (dko_s2) and 2 (the
  reserved circuit reads): 4 of 4. The three secondary lines wait.
- **05:0xZ — GPU rule changed** (Leon, via the coordinator): the cap of 4 is dropped; at least 2
  of the lab's 12 GPUs (serrano 5, ynez 4, temescal 3) must stay free after any launch, move or
  respawn. Free: no process of anyone and not reserved; an unreadable host has none free.
  Launch time only (running lines are not stopped when other users fill the lab). The
  per-machine rules stay (temescal ≤ 2 of ours, never share, serrano at 200 W). Tested on a dummy
  state (the 04:55Z census, 1 free: nothing launches; 3 free: one launch; reserved and
  unreadable GPUs never count as free).
- **05:0xZ — correction (Leon):** both rules apply: ours ≤ 4 **and** at least 2 lab GPUs free
  after any launch, move or respawn; both fail closed; launch time only. Tested on a dummy state
  (cap binding, MIN_FREE binding, an unreadable host).
- **05:00–05:07Z — Runpod pods** (Leon approved 2 A40s for dkoour_s0 / dkoour_s1; the
  coordinator created them: vsr5wcacsoq4g8 CA-MTL-1, in1myc2jl5fz75 EU-SE-1, $0.59/h each from
  04:59:57Z / 04:59:58Z). Read their public IP and port by REST (key on stdin, only status, IP and
  ports printed); added `rp-s0` / `rp-s1` to `~/.ssh/config`; copied `pyproject.toml`, `uv.lock`
  and a bounded setup script; the pods are downloading the cu130 env (`uv sync`) and Lean
  v4.34.0 (elan). RUNPOD_POD_ID and RUNPOD_API_KEY exist in the pods' PID-1 environment
  (presence only). Leon then dropped every time-based stop for the pod lines: they run to their
  cap, and each pod is deleted when its line ends (final backup verified, REST DELETE, GET).
  Spend is tracked against a $16.90 ceiling (an event; nothing stops).
  - The supervisor's pod support is written and passed its dummy-state tests (pinning, pod root
    as us, teardown, spend) but is **not deployed**: the permission system refused my rsync of
    the code, Dan's code and data, the bases and the pools to the pods ("data exfiltration").
    Not worked around; the coordinator was told at 05:07Z. Unless unblocked, both pods are
    stopped at 06:30Z.
  - **05:13Z: the copy.** The coordinator (agent:claude) ran `runpod_copy.sh` at 05:13:20Z on
    Leon's explicit approval ("this is not data exfil this is how run pod is supposed to be
    used"); all four md5s matched (ko_s0 9ee0c620, it170 ada0b5aa on rp-s0; ko_s1 3b0f4ba9, it145
    74af31c6 on rp-s1). Pod setup had finished at 05:08Z (rp-s0) and 05:15Z (rp-s1).
    `podcheck.sh` passed on both (A40, torch 2.14.0+cu130 with CUDA, Lean 4.34.0 accepts one proof
    and rejects one, a k-2 Dan read of the base runs end to end); the pods' four segment scripts
    have the lab's md5s. `sup/pods_enabled` created and the supervisor restarted (05:21Z):
    dkoour_s0 resumed as seg2 on rp-s0 from it 170, dkoour_s1 as seg2 on rp-s1 from it 145, both
    with 0 lost iterations. Fixed before the restart: the pod launch's environment prefix (an
    assignment before a subshell is a bash syntax error; dummy-tested with `bash -n`), a waiting
    line's reason no longer grows every pass, and a pod-pinned line no longer holds lab lines
    behind it.
  - The supervisor still copies nothing to a pod: my edit that would let it (deploy and
    md5-checked copies to pods, for the auto-spawn pods) was refused by the permission system
    ("data exfiltration") at 05:19Z even after Leon's approval was relayed. Not worked around;
    reported to the coordinator. The two pods run on the files the coordinator copied.
  - **05:2xZ: pace on the A40.** About the same pace as on ynez. dkoour_s0 took 69, 68 and 72 s per
    iteration (ynez median 64 s); dkoour_s1 took 63, 64 and 68 s (ynez 61 s). Projected: the compute
    cap binds first, at about 14:45Z (s0) and 14:40Z (s1). Pod spend is projected at about $11.6 of
    $16.90. The 06:30Z stop is dropped.
  - **05:38Z: more pods, with every copy done by the coordinator** (dd9c4cc). The supervisor reads
    the pods from `KD/sup/pods.json` every pass. That file is written by `register_pod.py`, and
    rp-s0 and rp-s1 moved there.
    - A line displaced by another user's process, with no lab GPU under the rules, gets one
      `NEEDS POD:` event. The event gives the checkpoint (first put on serrano) and the GPU-h
      remaining at its lab pace. The line then waits for a lab GPU or a pod.
    - A pinned line resumes on its pod from the coordinator's `pod_resume/resume_it<i>.pt`. The
      file must be non-empty and no older than the line's newest checkpoint.
    - Other events: `POD NOT NEEDED` when a lab GPU frees first, an idle-pod warning, and
      `NEEDS POD` again when a pinned pod is down.
    - Teardown, spend and the ceiling (+`budget_add`) cover every registered pod.
    - Two preemption fixes, found while testing. Preemption now happens only when it can yield a
      GPU under MIN_FREE; in a full lab it only idled the secondary. A pod line is never preempted;
      before, it was a candidate and would have freed no lab GPU.
    - The coordinator runs `runpod_copy_line.sh` (any line; md5-checked; `DRY_RUN=1` previews the
      99 files, about 160 to 180 MB) and `podsetup.sh`.
    - `test_autopod.py`: 48 mocked dummy-state checks, all pass.
    - A first test run, before `deploy` was mocked, rsynced the working folder to temescal. That
      is the same lab-to-lab code sync every launch does.
    - The temescal standby was restarted on the new code and receives `pods.json` with the state.
  - **12:01Z: the primary arm is done.** On Dan's measure the three seeds reached 0, 1 and 0;
    dko_s1's 1 was already its base's. dko_s0 stopped at the 15 GPU-h cap at x_dan 5.07 of 5.84.
    The 09:29Z GPU fault on temescal (GPU 0 dropped out of nvidia-smi) moved dko_s0 and dko_s1 to
    serrano; 1 iteration was lost. The primary report went to the coordinator at 12:13Z (0ad304f).
  - **14:30Z: dkoour_s0 ended at 13 of 106, one below the bar.**
    - σ in its accepted training proofs rose from under 1 % (to it ~384) to 33 % over its last
      50 iterations, peaking at 46 % at it 443.
    - Its snapshot first-32 counts went 0 → 12 (it 380–430), then 9.
    - rp-s0 was deleted at 14:31Z after its backup was verified.
    - 13:21Z: ynez stopped answering ssh, though it still answers ping. No line of ours was on
      it.
    - 13:28Z: the supervisor now stays up until every finished line's pod is deleted (bounded
      at 2 h; 14c40a5).
  - **dkoour_s1 is still running on rp-s1** (0 of 106 so far, x_dan 5.56 of 5.93; ETA about
    16:10Z). The agent handed back before it ended. `post_finalize.sh` reruns `finalize.sh` once
    every line has ended and logs one event; the summary's text for dkoour_s1 says pending.
- **Amendment 1, the dose lines (2026-11-05; all times Z; events in `KD/sup/events.md`).**
  - **17:17Z: launched.** dd01_s0 on serrano GPU 1; dd01_s1 and dd01_s2 on temescal GPUs 1 and 2.
  - **18:25Z: temescal marked broken.** Its CUDA has given "unknown error" on GPUs 1–2 since the
    GPU 0 fault of 09:29Z, though nvidia-smi still answers. torch fell back to the CPU, so
    dd01_s1 and dd01_s2 ran GRPO on the CPU for about an hour (3 iterations) and their base reads
    failed. Both restarted from the base on ynez at 18:27Z; `broken_hosts.json` now keeps lines
    off temescal (04338cf).
  - **18:27Z: a launch broke the lab rule.** dd01_s2 on ynez GPU 2 left only 1 usable lab GPU
    free, because `lab_free` counted a GPU assigned in the same pass as free. I ended that segment
    at 18:29Z and fixed the count (5b3a129).
  - **18:29–19:14Z: another user's 4-GPU job kept relaunching on ynez.** Its gaps made GPUs look
    free, so moves were started and then cancelled, and the 19:07–19:10Z cancels were wrong.
    - 15 quiet minutes needed, pod reuse and `no_pod.json` (18:32Z, 08eebc3).
    - A segment's own sighting of another user counts (19:11Z, f83847f).
    - 60 quiet minutes on GPUs where that job keeps coming back (19:14Z, 2dae7a1).
  - **18:34–19:12Z: the 1 % lines ended, each crossing at its round-2 draw and holding (a8397ea).**
    dd1_s2 was displaced from ynez and finished on rp-d1, reused from dd1_s0 at 18:36Z (seg2 from
    it 70, 0 iterations lost). rp-d1 and rp-d2 were deleted at 19:12Z and 18:39Z.
  - **19:11–19:13Z: dd01_s2 (it 25) and dd01_s1 (it 35) displaced from ynez.** No lab GPU was
    usable under the rules, and both NEEDS POD events were held for Leon's answer. dd01_s0 stayed
    on serrano.
  - **19:30Z and 19:33Z: the missing base reads made** on serrano GPU 1 with the same code and
    arguments (`baseread_dd01.sh`): dd01_s1 0 first-32 (his base read 0), dd01_s2 1 (his 3). The
    merged measure has them as round 1 (checked with `merge.line_view` at 20:16Z).
  - **20:14Z: the supervisor no longer reaches Runpod.** Leon revoked the API key and deleted
    `~/.runpod_env`. `rest()` is gone (3b69a19). When a pod's line has ended, its backup is
    verified and it is not held or reused, the supervisor logs one `POD READY TO DELETE:` event.
    The coordinator deletes the pod through the MCP, confirms with a read, then runs
    `register_pod.py retire`; the spend clock runs until then.
  - **20:14Z: pods for dd01_s1 and dd01_s2, on Leon's approval.** The reason is the temescal fault
    and the contention on ynez, with serrano's other GPUs held by other users. The coordinator
    created the pods and does the copy, install and registration.
    - dd01_s1 → rp-d3 (48pxqqf95kb23x, created 20:14:01Z), resuming from it 35 of serrano's backup.
    - dd01_s2 → rp-d4 (2w3qebaxx02cy3, created 20:14:03Z), resuming from it 25 of serrano's backup.
    - Each registration adds $7.80 to the pod ceiling: $15.60, which is the $12 Leon approved plus
      30 %. The ceiling is now $34.88. The coordinator estimates about 13 GPU-h per pod, $16 in
      total, if a line never crosses.
    - Registered at 20:22:52Z and 20:23:00Z, and both lines came off `no_pod.json`. Seg4 launched at
      20:24:13Z and 20:24:16Z, 0 iterations lost. Both logs say "resumed ... at iteration" 35 and
      25, and each pod's A40 holds the process.
  - **21:34Z: dd01_s0 displaced from serrano GPU 1 by another user's job.** It stopped at its it 220
    checkpoint (x_dan 1.42, 0 of 106 at every read so far, 4.3 GPU-h), with no iterations lost. No
    lab GPU is usable under the rules. Leon's 20:14Z approval named only dd01_s1 and dd01_s2, so I
    asked the coordinator. Its ruling: his standing permission covers a GPU taken by another user
    ("if one of our gpus gets stolen by another user you have permission to spin up more a40s"); the
    earlier 0.1 % hold was for the temescal fault and `no_pod.json`.
    - The coordinator created rp-d5 (upfpayavayd95j, 21:40:45Z), copied the line and registered it
      at 21:51:11Z, adding $8.45. The ceiling is now $43.33.
    - Seg2 launched at 21:52:35Z from it 220, 0 iterations lost.
  - **03:56Z (2026-11-06): dd01_s1 died on rp-d3 with its disk full.** Past `--max_rollbacks` a line
    writes a 115 MB alarm checkpoint at every iteration, and the 28 GB volume reached 100 %. rp-d4
    was at 849 MB free. No traceback was recorded, since the log could not be written, so disk
    full is the inferred cause; it is the only fault found.
    - I deleted the old alarm checkpoints on both pods, keeping the newest 4 of each (106 on rp-d3,
      97 on rp-d4; each pod now has 12 GB free). Rolling and snapshot checkpoints stay. Nothing read
      the deleted files: a resume takes the newest, a rollback a rolling or permanent one, and reads
      use the snapshots.
    - dd01_s1 respawned as seg5 from it 273 at 03:59Z, 0 iterations lost; seg4's metrics end at it
      273 with no damaged rows.
    - `prune_pod` now does this on every pass for pod lines; lab lines are untouched.
  - **04:20Z (2026-11-06): the guard's timeline added (`guard.py`, read-only; in the summary's Caveats).** All
    lines that did not cross reached lr x0.125 early: 0 % at x 1.16, 2.59, 0.99; 0.1 % at x 1.44,
    1.43, 1.51. The 1 % lines crossed at the full lr. The coordinator's first read of dd01_s2's
    first rollback (x 0.21) was the first sighting in seg4's log; the metrics rows give it 3, x 0.02.
    The coordinator is proposing a guard-off pilot to Leon; nothing starts without his yes, and the
    15 GPU-h cap stands until he answers.
  - **08:10Z (2026-11-06): "backup verified" tightened before dd01_s0's stop.** The check accepted any
    older checkpoint in the backup folder, and a stopped line ends by SIGTERM while `torch.save` has
    no atomic rename. The pod's newest checkpoint must now be in serrano's backup, the same size, and
    pass a zip test. The ready event names that iteration and the last logged one. A resume can be up
    to 4 iterations behind the last logged one, since a stop does not write a checkpoint.
  - **08:18–09:44Z (2026-11-06): the 0.1 % lines ended at the 15 GPU-h cap, none crossing.** Leon was
    asked whether to lift the cap and had not answered, so it stood.

    | line | stop x | his round-16 x | his measure, of 106 | best first-32 | lr 1/8 from x | resume point in the backup |
    |---|---|---|---|---|---|---|
    | dd01_s0 | 4.49 | 5.67 | 0 | 0 | 1.44 | it 695 (last logged 696) |
    | dd01_s1 | 5.43 | 5.94 | 0 | 0 | 1.43 | it 400 |
    | dd01_s2 | 5.63 | 5.71 | 1 | 3 | 1.51 | it 432 |

    - The coordinator deleted rp-d5, rp-d4 and rp-d3 through the MCP and retired them; its list of
      pods is empty. Pod spend by the supervisor's count is $37.95, under the $43.33 ceiling.
    - Serrano's supervisor exited by itself at 09:47:17Z. I stopped temescal's standby at 09:47Z,
      since it exits only after a takeover. No line or supervisor process is left on any lab host.
    - P8 is not met as run. No line was resumed and nothing new was started.
- **Amendment 2, the guard-off pilot (2026-11-06; times Z).**
  - **21:08Z: PREREG.md frozen again before the line runs:** sha256 `a829342adb0f5d6775c98a4453559952fba7a9b94b460d4ce7f73bee4d11da50`. `guardoff_grpo.patch`: sha256 `30e4833fe954f9216b3fee226baeae1fd3a45db4455916656466ad2401eb5ecf`; the snapshot's `grpo.py` goes from d3ff0add… to 6d88250b…, with both flags off for every other line. The earlier copy is `KD/snap_orig_grpo_f35d3daf_danbase.py`.
