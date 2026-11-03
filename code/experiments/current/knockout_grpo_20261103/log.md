---
written_on: 2026-11-03
written_by: agent:claude
---

# knockout_grpo_20261103: run log

Times are UTC (PT = UTC − 8 after 2026-11-01).

## 2026-11-03

- **2026-11-03T06:16:32Z** — PREREG.md frozen before any GPU job: sha256 `f35fc350aedb40bd0337d01e3257c16ba19d93d87cd151089ec4a0e582c8b2a8`.
- Before it (CPU only, Nov 3 05:00–06:15Z): sigma.py positive control on K12 (16,703 σ, 16,961 σ⁺:
  Dan's numbers); S1g3 σ⁺ 3,838 (2.48 %); Dan's bucket files fetched read-only
  (`dan_data/`, sha256 equal to his MANIFEST); refill_pool.py (258,494 eligible); search traces of
  the candidates at 79749ce (reproduces S1g3's traces byte for byte on 2,990 / 2,990); assemble.py
  plan / write; the S1g3 pickle command reproduces S1g3.pkl (md5 d1f29544); ko / d01 / d1 pickles
  built and σ-verified (0 / 155 / 1,546).
- **06:26–07:14Z** — 9 bases trained (serrano 2, ynez 5, temescal 2; ynez 16 min, temescal 18 min,
  serrano 25–29 min each; Stage-1 tokens 441.1M (ko_s0) against S1g3_s0's 440.6M). Base reads (k 256,
  n 256 per theorem, seed 0), E32 of 106: full 22.34 / 26.31 / 29.33; d1 14.39 / 14.39 / 32.63;
  d01 12.29 / 6.67 / 5.21; ko 0.73 / 0.36 / 1.04.
- **Bar** (pre-registered rule): median of the full bases' E32 = **26.31** of 106 (24.8 %);
  Dan-matched single-draw bar 27.
- **At the bar before RL** (no line): full_s1 (26.31), full_s2 (29.33), d1_s2 (32.63).
- **Pilots** (L3 from grpo_think_vs_s0_L2_it39, 10 iterations, mean of it 2–10): ynez clean
  135.2 s/it, ynez with the online read 173.6, temescal with the read 163.1, serrano with the read
  199.6 (serrano's clean kkl0 L3 runs: 158–186). R per L3 iteration 0.0040. A co-located k 64 read
  takes 190–230 s. Iteration 1 is identical in the ynez and temescal pilots (same seed).
- **T_L3** (pre-registered rule): median clean L3 s/it over the machines = 135.2 (ynez; temescal
  ≈ 127 and serrano ≈ 156 after removing the read's 1.28× slowdown); L2 = 4,744 s (the full arm's
  three L2 runs, ynez). T_L3 = round((54,000 − 4,744) / 135.2) = **364**; c_max = 403.
- **Lines** (9): ynez ko_s0, ko_s1, ko_s2, d01_s2; temescal d01_s0, d01_s1; serrano full_s0, then
  d1_s0, d1_s1 (max 2 of ours on serrano). Each line: `line.py --t_l3 364 --bar 26.31`.
- **07:30Z** — gate report sent; 9 lines launched (`line.py --t_l3 364 --bar 26.31`).
- **07:36Z** — full_s0: E32 26.73 at c 1 (R 0.0029), 28.23 at c 2 (R 0.0063): stop rule met, line
  done at c 3; serrano's GPU went to d1_s1.
- **07:40–07:50Z** — coordinator's CPU checks.
  - Leakage of the 106 (Dan's canon_key / canon2): 0 in the 136,650 kept S1g3 records, 0 in the
    dose σ records (155 / 1,546), 0 in the refill, 0 in all of S1g3, 0 in L2_pool / L3_pool /
    L3_3p (and the canary).
  - Strict σ-covered share of the RL pools (Dan's rule, `pool_sigma.py`): L2 39.5 %, L3 44.5 %,
    L3_3p 34.5 %; Dan's rl_targets 2.7 % (123 / 4,495).
  - health.py: reward, entropy, no-signal / all-fail / all-correct (estimated) every 25
    iterations in KO/STATUS.md, with entropy-collapse and falling-reward alerts.
- **2026-11-03T07:49:31Z — Amendment 1 (coordinator, after the PREREG hash; the primary crossing is unchanged).**
  - Hold window: a line keeps running until it has been at or above the bar for ≥ 3 consecutive
    evaluated checkpoints and the latest is at ≥ 2 × the R of the first of that streak (a
    checkpoint below the bar resets the streak). The pre-registered stop (≥ bar and ≥ 1.1 × R) is
    replaced; the primary crossing (the first checkpoint ≥ bar, interpolated in log R) stays.
    Both the first crossing and the held crossing are reported; a first crossing that does not
    hold is flagged.
  - Lines the first rule already stopped continue with `line.py --resume_line`: from the newest
    rolling checkpoint (exact training state), with the later snapshots, reads and curve rows
    moved aside (`.rule1`), so each curve is one trajectory. Running controllers keep the first
    rule in memory, so they are continued this way when they stop.
  - full_s0 already meets the window (c 1–3 at 26.73 / 28.23 / 29.31, R 0.0029 → 0.0101, 3.5 ×);
    it is not resumed.
- **07:46Z — incident.** Four init reads (it0 checkpoints through the snapshot read path, n 64)
  were started co-located on the lines' GPUs. A third process per GPU ran out of memory (GRPO peaks
  at 12.4 GB, the line's read at 7.4 GB): the GRPO processes of d1_s0 (c 4) and d1_s1 (c 3) died
  (lines EXIT rc 1); on ynez ko_s0's c 5 read failed once and was retried; the init reads were
  killed. No other line was hit. The init reads were rerun alone on serrano's two freed GPUs
  (`initread2.sh`); d1_s0 and d1_s1 are then resumed with `--resume_line` (from their it0
  rolling checkpoints, i.e. a restart; their few evaluated checkpoints are moved aside).
- **07:50–07:55Z — init reads** (it0 rolling checkpoints, the snapshot read command: k 64, batch
  512, seed 0; alone on serrano GPUs 1 / 4). E32: full_s0 21.12 (base n 256 22.34, c 1 26.73);
  d1_s0 13.42 (14.39, c 1 18.66); d01_s1 5.94 (6.67, c 1 7.36); ko_s0 1.38 (0.73, c 1 1.25). The
  inits read near their bases, not near their first checkpoints: the early jumps are not a read
  artifact. n 64 read noise (subsampling the n 256 base reads): sd 0.8–1.0 (ko 0.5).
- **07:53Z** — line2_d1_s0 (resumed from its it5 rolling checkpoint; snapshot c 6 moved aside) and
  line2_d1_s1 launched on serrano. Every other line has a queued line2 job (`--resume_line`,
  depending on its line job).
- **2026-11-03T07:58:59Z — ko ×2 extension (exploratory, coordinator: "keep it on one seed").** One knockout seed
  gets a second L3 with twice the budget, as ki78 lengthened L3: `line.py --x2 --t_l3 767` (c_max
  806 = 2 × 403), L3 from that seed's own L2 it39 with the same args and seed, WSD over 767
  iterations; its L2 part is the original line's. Its seed is chosen when it launches (the first
  GPU free after the ko lines' L2 ends): among the ko seeds not yet at the bar, the one with the
  highest mean E32 over its latest 3 checkpoints, ties to the lower seed, as Dan extended the
  knockout seed that was making progress. If every ko seed has reached the bar, it is not run.
