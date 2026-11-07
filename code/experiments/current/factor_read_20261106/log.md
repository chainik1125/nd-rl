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
- **Check 6, knockout control (9:42 pm, C on the 3 + 3 bases):** mean log10 P(DN group) at base, all tokens: 0 % −1.81 (seeds −1.58, −1.99, −1.85), 1 % −0.42 (seeds −0.30, −0.15, −0.82). Lower at 0 %, every 0 % seed below every 1 % seed. **Pass.** The name-free values differ only in the 4th decimal.
- **9:42 pm — full read started,** ynez GPU 0, `launch_full.sh`, timeout PID 1159532, watchdog PID 1159533. 280 s of GPU time used before it. Coordinator told.
- **PREREG amendment 2 frozen, sha256 `690439678dae5fa2b1a73265e275bb6cc445ad1544b8e0c839d4c796553f0f50`.** The coordinator corrected the list while the read was on the first base (true ends for dko_s0 and dko_s1; no RL-end result read or seen) and asked for the groups to be labelled by their tokens. Both true ends load; x_dan 5.07 and 5.93 confirmed from `numbers.json`.
- **11:29 pm — first process ended, rc 0, all 19 checkpoints.** Its reads of the stale ends were renamed on ynez to `dko_s0_it500` and `dko_s1_it630` (`rename_secondary.py`, which checks what each directory holds and refuses to run twice).
- **11:29–11:46 pm — the true ends** `dko_s0_it580_snap.pt` and `dko_s1_it676_alarm.pt` read on ynez GPU 0, same settings and gates (Amendment 2).
- **GPU in all: 7,591 s = 2.11 GPU-h,** against the 3 GPU-h cap: A 4,780 s, reach 2,236 s, C 246 s, DN 119 s, pick 93 s, NEGI 75 s, checks 42 s. Nothing else used GPU 0 during the read; the gate never stopped it.
- **Watchdog:** OVERRUN (I gave it 1 h expected; the read took longer because RL-end attempts are about six times longer than base attempts) and EXIT (the first process ending) both benign, coordinator told. No STALL.
- **Allocator warnings:** CUDACachingAllocator "memory allocation failed with OOM" at batch 1024 during reach, on some RL ends. The allocator retried and the runs continued. No draw lost: every row has 6,784 draws.
- **Classifier QC on all 21:** 0 replay mismatches, 0 disagreements with Dan's σ flag.
- **Against the lines' own reads at the same iteration** (first 32, solved in 64, E32), here vs theirs: dd1_s0 (28, 29, 28.3) vs (28, 31, 29.4); dd1_s1 (20, 20, 19.5) vs (19, 19, 19.0); dd1_s2 (19, 20, 19.5) vs (19, 20, 19.47); the zeros match on dko_s0, dko_s2, dd01_s1 and the pilot. Bases: dose1_s1 12 vs 13, ko_s1 0 vs 1. These are fresh draws on another host, so small differences are expected.
- **Control rerun on the final data (check 6) passes,** and `analyze.py` writes `results/results.json`, `results/tables.md`, `results/control.json`.
- **Serrano copy verified against ynez** by `rsync -anc` over reads, ledger and refs: no difference. The one difference is `summaries/`: ynez held only the seven partial summaries from the 9:42 pm control, and the 21 full ones were built on serrano at 11:46 pm.
- **`results/checks.json`:** the rsync from ynez overwrote serrano's copy, which had check 4 on it. Re-ran the check 4 comparison on serrano (CPU, from the stored read): the same pass, written back.
- **Post-hoc, not pre-registered (`posthoc.py`, `results/posthoc.md`):** written after I had read the frozen tables, because they raised questions the frozen tables could not answer: the close-box collapse, the long attempts, B and C disagreeing on pick and reach, and how often the model finishes from a forced pick. These are counts of the existing draws only, no new read or GPU.
- **A label to know:** `tables.md` prints dd01_s2's x as 5.60 (read at it 430), from that read's `ckpt.json`. The line's last row is 5.63, as Amendment 2 says. Nothing is computed from this number.
- **11:57 pm — ynez clean.** `/scratch/cowifornia/nd-rl/factor_read` deleted on ynez, after the logs were pulled to serrano. No process of ours, no lock, GPU 0 at 4 MiB and 0 %, watchdog PID 1178160 gone. GPUs 1–3 never touched. The originals stay on serrano.
- **A flaw in a frozen table, found while writing up.** `analyze.py`'s bottleneck takes the first of the lowest stages, so a tie at 0 of 64 between pick and reach ⊥ counts as pick. The frozen table's "pick on 104–106 of 106" is mostly ties: pick is alone lowest on 23–39, tied on 67–83, and reach ⊥ is 0 of 64 on 63–78. I left the frozen code and table as they are and added post-hoc table (j) with the ties shown. The summary reads from (j) and from C, which has no such ties.

## 2026-11-07: Amendment 3, the matched-compute control (post-hoc)

- **12:15 am — the coordinator asked for it after seeing the results,** under Leon's standing rule for fairness controls: read seed-matched checkpoints at the 1 % ends' x, with no frozen reading applied and the frozen tables and code untouched.
- **Checkpoints.** All seven copied to `/scratch/cowifornia/nd-rl/factor_read/matched/ckpts` with an md5 check (`copy_matched.sh`), including one file read-only from temescal, under the coordinator's override; nothing else was done there. All load on the CPU, 9,560,832 parameters, lean_staten, `grpo_init` the expected base and `grpo_iter` the expected iteration (`make_matched.py`, `ckpts_matched.json`). Each x recomputed from the line's metrics rows, equal to `numbers.json`'s curve where the curve has that iteration, including the pilot's 0.663 from its own rows.
- **Found on the way, before any read:** at the snapshot the guard had already halved the lr on all four guarded matched lines, and `dko_s0_it60_snap.pt` is the state just after a rollback to it 55. Serrano has no dd01_s1 or dd01_s2 checkpoint nearer the match; only their alarm ends are backed up.
- **PREREG Amendment 3 frozen, sha256 `f56da4c983c955da5e14ec1569e7ca600bd7bb38f6461f3c2e9916df298220c9`.** No read of these seven before this line.
- **12:16 am — first matched checkpoint** (`m_dko_s2_it100`), ynez GPU 0, gate passed, own lock and ledger (`launch_matched.sh`): 503 s, so the seven project to 2,500–3,200 s, under the 3,600 s cap. Coordinator told.
- **12:25–12:59 am — the other six.** GPU in all: 2,509 s = 0.70 GPU-h of the 1 GPU-h cap (it100 503, dko_s0 320, dko_s1 289, dd01_s0 323, pilot 510, dd01_s1 it35 271, dd01_s2 it25 293). The gate never stopped it. The first watchdog's EXIT at 12:26 am was its own process ending, which is benign. No STALL or OVERRUN.
- **Classifier QC on all seven:** 0 replay mismatches, 0 disagreements with Dan's σ flag. The read agrees with the lines' own reads at these iterations, where they exist: first 32 is 0 and solved in 64 is 0 on all five.
- **Analysis:** `matched.py` writes `results/matched.md` and `results/matched.json`. `posthoc.py` adds tables (k) C's name-free pick by schema, (l) σ by iteration window and (m) the any-formula reductio at the pick step. (k) reproduces −9.52 for dko_s0, and (m)'s X = T equals (h)'s pick count on all 21. The frozen tables and code are unchanged.
- **What it shows, in short (post-hoc, no frozen reading):**
  - At matched x the 0 % lines never choose the reductio (pick 0, any formula 0.0 %).
  - Their reach ⊥ rises as much as the 1 % lines' does.
  - They finish from a forced pick less often: 24.9 / 10.2 / 19.7 against 25.9 / 30.2 / 36.6 %, with the gap at DN after the box.
  - So the summary's earlier "0 % ends finish from a forced pick about as often as 1 % ends" holds only at unmatched compute. The summary is corrected.
- **1:03 am — ynez clean.** Reads, ledger and logs pulled to serrano and checksum-verified; `/scratch/cowifornia/nd-rl/factor_read` deleted on ynez. The finished watchdog (PID 1185424) was stopped by its exact PID. No process of ours, no lock, GPU 0 at 4 MiB and 0 %. The serrano copies (`$FR/matched/`) stay.
