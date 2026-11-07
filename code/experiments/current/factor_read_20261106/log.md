---
written_on: 2026-11-06
written_by: agent:claude
---

# factor_read: run log

Diagnostic read for Leon's natural-deduction RL project, under his standing 30-minute rule (about 2 GPU-h,
hard stop 3 GPU-h, ynez GPU 0). No training. Branch `leon/factor-read` from d78d12a. Outputs under
`/scratch/cowifornia/nd-rl/factor_read/`. All times PT (UTC−8).

## 2026-11-06

- **9:10 pm — start.** ynez GPU 0 free (4 MiB, no process). GPUs 1–3 belong to another user and are left alone.
- **9:20 pm — reference proofs, CPU only.** `run_minlen.py` at bound n_lines + 2, 60 s, 8 processes on serrano: all 106 found in 26 s. `make_refs.py`: all 106 replay in the env, pass the verifier, flag σ, and pass Lean. 0 fallback, 0 dropped. 15 have two σ DN lines (all negated_conditional). `refs.jsonl` sha256 7074e5ea166ceb1e9f0cb76db4fb290fc4a65a971cdef4aad91f25bf599d1c77.
- **9:30 pm — 19 checkpoints confirmed to load** on the CPU (`make_ckpts.py`, `ckpts.json`): all lean_staten, 9,560,832 parameters. Three RL ends are alarm checkpoints (dko_s1, dd01_s1, dd01_s2). dko_s0's and dko_s1's are earlier than their lines' last iterations (500 of 580, 630 of 676).
- **Check 5 (unit tests), before any GPU read:** `test_factor_lib.py` 10 of 10 pass, including every frozen reference replaying at name bases 0, 11, 32 and passing its own classifiers.
- **PREREG.md frozen, sha256 `a0b59c63694bf93fcf62e5f3806fc905005410515bcff8d218fcd7ee966af623`.** No GPU read before this line.
