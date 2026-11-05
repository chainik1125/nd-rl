---
written_on: 2026-11-04
written_by: agent:claude
---

# base_saturation_20261105: running log

Times are PST (UTC−8), with UTC in brackets.

- **11:03 pm (07:03Z).** The coordinator created 5 A40 pods: `rp-sat-a`, `rp-sat-b025`, `rp-sat-b4`,
  `rp-sat-c` and `rp-sat-d1`. Their env (cu130 venv from the lockfile, Lean 4.34.0) was set up by
  07:13Z with knockout_danbase's `podsetup.sh`.
- **11:10 pm.** Worktree `/scratch/cowifornia/nd-rl/worktrees/base-saturation`, on the new branch
  `leon/base-saturation` from `leon/circuit-diffing` `32bc1da`. It is a separate worktree so the
  circuit-diffing worktree's branch stays as it is: the circuit agent still works there.
- **11:16 pm (07:16Z).** The coordinator ran `runpod_copy.sh` for all five pods, and every md5
  matched. The pods have a CPU quota of 8, 96 visible cores and 503 GB RAM; `rp-sat-a` has 15 GB of
  disk free.
- **11:15 pm.** PREREG written (before any job).
- **11:18–11:19 pm (07:18:34–07:18:54Z).** PREREG committed (`994790c`), then all five launched
  with `launch.sh`:
  - `rp-sat-d1`: D1 L2.
  - `rp-sat-c`: C.
  - `rp-sat-b4`: B 1,248.
  - `rp-sat-b025`: B 78.
  - `rp-sat-a`: A.
  - The pods run the pre-lint `anneal.py` from the staging at 07:15Z, md5 in
    `stage/md5_rp-sat-a.lab`. It differs from `994790c`'s only in the operand order of one assert,
    the `shape` label string in its saved JSON, and line wrapping. `stage1_to0.py` is AST-identical.
- **11:20 pm.** First signs. A's base val loss is 0.0692 (action 0.0772), equal to Stage 1's logged
  final 0.0692. C runs at about 20 steps/s, so about 40 min for 48k steps. D1's Lean gate uses 7
  workers on the 8-CPU quota.
- **11:40 pm.** PREREG correction, before any read of a new model. Seed 0's CPT − base at pass@1024
  is +9.4 [+4.7, +14.8], not "+13.5" (base 78.5%, CPT 87.9%, RL 89.3%, from `table.py` on the
  existing reads). The writing time is also fixed.
- **11:50 pm (07:50Z).** PREREG amendment 1 (val-loss context, `1dc80bb`).
- **12:01 am Nov 5 (08:01Z).** The first agent handed back (harness limit). A second agent:claude
  session took over at 08:02Z: watch, pull, merge, table, val loss, teardown and write-up. Pod
  state then: A's and B0.25's reads at chunks 13–14 of 16; B4 training (step about 700 of 1,248);
  D2's L2 started 07:58:48Z on `rp-sat-c` (C finished 07:58:48Z, 40 min); D1's L2 at iteration
  14 of 39.
- **12:04 am (08:04Z).** B0.25's read finished on `rp-sat-b025` (16 chunks, about 2–3 min each).
  Pulled and merged (`final/charles149_sat_B78_s0_plain_k1024`, 149 problems). `merge.py`'s
  closing print asked for `pass_at_k_pct['5']`, which `summary.json` does not hold (powers of 2
  only); the merge itself had finished. The print now omits pass@5 (`table.py` gives it).
  `table.py` gained descriptive pairs, marked † (each arm against base and CPT), beyond the
  PREREG's list.
- **12:06 am (08:06Z).** A queued `go.sh` waits about 10 min after the previous one ends. The
  watchdog, started by the first `go.sh` after it took the GPU lock, inherits the lock's file
  descriptor (fd 9), so the `flock` stays held until the watchdog exits, 10 min after the last
  `go.sh` ends. On `rp-sat-b025` the queued C read (chunks 8–15) therefore starts about 08:14Z,
  not 08:04Z. The same delay applies on `rp-sat-a` (C chunks 0–7) and on `rp-sat-d1` (D2 chunks
  8–15). Cost: about $0.10 per hand-over. Stopping the idle watchdog by hand was not done (the
  permission classifier refused it).
- **12:09 am (08:09Z).** A's read finished on `rp-sat-a` at 08:08:54Z. Pulled and merged
  (`final/charles149_sat_A_s0_plain_k1024`). `table.py` gained RL − A and RL vs A one-arm-only
  counts (†, descriptive).
- **12:10 am (08:10Z).** The second agent session also had to hand back (harness limit). It
  stopped its event monitor; poll counters at hand-back: `rp-sat-a` 39, `rp-sat-b025` 39,
  `rp-sat-b4` 3, `rp-sat-c` 5, `rp-sat-d1` 3. The pods' `watchdog.sh` was not edited to close
  fd 9. `rp-sat-d1`'s watchdog is already running with the lock's fd, so an edit would not shorten
  the D1-to-D2 hand-over, and rewriting a script bash is executing is not a clean change.
  `figure.py` (pass@k curves, `ndbench` style) is written and tested on the three old reads.
- 2026-11-05T10:25:53Z coordinator: D2 read chunks 8-15 launched on rp-sat-c (queued behind its chunks 0-7) instead of on rp-sat-d1: D2's L3 finished at 10:24Z, before D1's L3, so this needs no copy and frees rp-sat-d1 after D1's read. Same checkpoint (it39 on rp-sat-c) and read settings.
- **Reconstructed from the pods' `events.log` and stage logs** (no agent session watched live
  between 12:10 am and about 3:30 am). All times are the pods' own:
  - **11:58 pm (07:58:48Z).** C finished (48,000 steps, 40 min). D2's L2 started on `rp-sat-c`.
  - **12:36 am (08:36Z).** C's read finished: chunks 8–15 on `rp-sat-b025` at 08:32Z, chunks 0–7 on
    `rp-sat-a` at 08:36Z. `rp-sat-b025` ended at 08:34Z and `rp-sat-a` at 08:38Z.
  - **12:55 am (08:55:35Z).** D1's L2 finished (97 min) after 1 rollback: a canary alarm at it 29,
    restored to it 25, lr ×0.5. D1's L3 started.
  - **1:15 am (09:15:59Z).** D2's L2 finished (77 min) after 1 rollback: a canary alarm at it 3,
    restored to it 0, lr ×0.5. D2's L3 started.
  - **1:30 am (09:29:43Z).** B4's read finished. B4 itself had trained in 69 min. `rp-sat-b4`
    ended at 09:31Z.
  - **2:24 am (10:24:38Z).** D2's L3 finished (68 min) after 2 rollbacks, final lr ×0.25:
    - a canary alarm at it 3, restored to it 0;
    - a think-banned-mass alarm at it 30, restored to it 25;
    - banned-mass alarms at it 37–39, not rolled back because the restore point (it 35) had the
      same mass.
  - **2:37 am (10:37:10Z).** D1's L3 finished (101 min) after 3 rollbacks, final lr ×0.125:
    - canary alarms at it 20, 28 and 29, restored to it 15, 25 and 25;
    - a 4th alarm at it 30, not rolled back (`--max_rollbacks 3` reached).
  - For comparison, the paper's RL s0 had 0 rollbacks in L2 and 1 in L3 (final lr ×0.5), from
    `think/runs/logs/v2_grpo_think_vs_*_s0_L{2,3}.log`.
  - **2:43 am, then 2:53–3:12 am.** D2's read chunks 0–7 finished at 10:43:46Z. Chunks 8–15 ran
    10:53:44–11:12:31Z, also on `rp-sat-c`, as the coordinator line above says.
  - **3:16 am.** The D2 read was merged into `final/` (149 problems × 1,024, seeds 0–15).
    `rp-sat-c` ended at 11:17Z.
  - **3:24 am (11:24:04Z).** D1's read finished (all 16 chunks on `rp-sat-d1`) and was merged at
    3:28 am. `rp-sat-d1` ended at 11:29Z. All five pods are deleted.
  - **Cost.** Pod-hours were a 1.58, b025 1.51, b4 2.46, c 4.23 and d1 4.43: 14.21 h × $0.59 =
    **$8.38**, against the $18.49 ceiling.
- **3:20–3:45 am (11:20–11:45Z), a third agent:claude session (write-up).** No GPU was used.
  - **Val loss.** `valloss.py` (CPU, `taskset -c 0-3`, `CUDA_VISIBLE_DEVICES=`) scored D1 and D2
    from the pulled L3 it39 checkpoints, about 2 min each:
    - D1: val w1 0.0719, Charles149 ref w1 0.0949;
    - D2: val w1 0.0707, Charles149 ref w1 0.0699.
  - **Table.** `table.py` regenerated from the six merged reads plus the three existing reads.
    Before any edit, its output was identical to the earlier `table.md`. I then added these
    descriptive † pairs, chosen after the data:
    - RL − CPT, RL − B0.25, RL − B4 and RL − C;
    - RL vs C one-arm-only counts.

    Each `paired_diff` call uses bootstrap seed 0, so the existing rows did not change.
  - **Figure.** `figure.py` wrote `passk.{pdf,svg,png}`. I checked the PNG: the legends sit clear
    of the curves.
  - **Checks.** `make -C code check` passed (ruff, format, pyright 0 errors).
  - **Readings** (details in README.md):
    - H-anneal holds.
    - "Anneal is not the story" fails.
    - H-undertrained fails; the C clause fails by one problem (87.2 vs 87.9).
    - The dose plateau holds; "turns over" fails.
    - "H-RL holds" fails on the boundary, and "H-RL fails" holds on the boundary. D2 − C at
      pass@1024 is +3.4 [+0.00, +6.7], and the side clause on pass@1 holds.
  - **Write-up.** README.md here and `experiment-summaries/2026-11-05-base-saturation/`.
