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
- **Check 1 (9:33 pm, GPU):** an empty forced prefix reproduces `env_generate` byte for byte: 64 draws at batch 16 and 96 at batch 64 (0 differ), same end counts, prompt tokens and waves.
- **Check 2 (GPU):** the forced state equals the env's at all 424 boundaries (106 theorems × 4), against Dan's `decompose` with names shifted (0 differ) and the scorer's rows (0 differ), across all 33 name bases. On 70 sampled, non-reference attempts replayed as forced prefixes, 0 differ.
- **Check 3 (GPU), k 4000, one fixed name base:** sampled frequency against exp(log p): 7.80 % vs 7.99 % (z −0.44), 21.20 % vs 20.12 % (z +1.71), 50.73 % vs 50.68 % (z +0.06). The scorer's full forward against the sampler's cached left-padded path, 256 rows: mean difference 0.018 nats (0.008 in log10) per action, largest 0.26 nats; largest difference in probability 0.020. So bf16 rounding sets C's precision at about 0.01 in log10 per action.
- **First checkpoint (`stage1_dose01_s2`):** A 92.9 s, pick 3.7 s, reach 42.7 s, NEGI 3.7 s, DN 5.0 s, C 9.6 s: 157 s. 19 checkpoints project to about 50 min, more if RL ends write longer attempts. Under the 3 GPU-h cap.
- **Check 4:** the unforced read reproduces `$KD/collected/ynez/dd01g0_s2/seg1/reads/it0000.json` at seed 2500, batch 128: the same 2 solved, the same first-32 solve, proofs byte identical, end counts (4,131 syntax / 2,652 done / 1 step cap), prompt tokens (3,371,032) and waves (407) all equal. Same host and GPU model. **Pass.**
- **Classifier QC on that read:** 0 replay mismatches, 0 disagreements with Dan's σ flag.
- **PREREG amendment 1 (name tokens) frozen, sha256 `7ba46f8576db09927c192f088c3ec7e3cfa3903c316a080d934f269340ba5201`.** Written after that one base read and before any RL end. Only that base's C is rerun.
