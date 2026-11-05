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
