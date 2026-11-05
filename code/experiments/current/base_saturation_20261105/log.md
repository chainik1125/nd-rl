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
