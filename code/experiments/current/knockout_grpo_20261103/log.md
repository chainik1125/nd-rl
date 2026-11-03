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
