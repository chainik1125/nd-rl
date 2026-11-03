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
- **2026-11-03T08:34:36Z — Amendment 2 frozen** ([`amendment2.md`](amendment2.md), sha256 `bc7f0c75db5dfa0468f2989b4c391fb0c67a95b991b14048f9a0ea6e445fb898`): the RL-pool control
  (Dan's 4,495 rl_targets as the pool, 0 hits with the 106; only the pool changes). full_s0_dp
  launches first; the ko-type bases follow the rule in the amendment.
- **08:25–08:50Z — trace-leak check** (`trace_leak*.py`; at each record's trained level and over
  the e64 / e256 files). No ko proof or narration applies DN to a NEGI-built line (narration DN
  citations in ko: hypotheses 2,022, premises 662, planned-box hypotheses 414, other 1; full
  2,509 / d1 1,036 / d01 99 on NEGI-built lines). But the classical-detour plan `|- G :=
  Classical.byContradiction |- ( ~ ( ~ G ) )` is in 637 ko records at their trained level (515 with
  the NEGI half), 949 in the e64 ∪ e256 files (full 2,881 / d1 1,525 / d01 722 trained-level).
  Reported to the coordinator, who approved ko_strict (Amendment 3) and the pool control
  (Amendment 2).
- **2026-11-03T08:43:21Z — Amendment 3 frozen** ([`amendment3.md`](amendment3.md), sha256 `bda6eb52761d78af4cebf75cb568d4ab26a1fbaf715246130b4c61f03ccb99c1`): ko_strict (949
  records swapped; verified 0 σ⁺ in proofs, 0 detour plans in either level's narration).
- **2026-11-03T09:08:32Z — Amendment 2, ko-type bases:** GPUs freed while ko_strict × our pool was
  still undecided (kos_s0 / kos_s2 lines just started; kos bases E32 0.00 / 0.33 of 106 at n 256,
  against ko's 0.73 / 1.04), so by the amendment's rule the Dan's-pool lines run on the ko_strict
  bases: kos_s2_dp (ynez), kos_s0_dp (temescal), kos_s1_dp once its base is trained.
- full_s0_dp: crossed the bar and held (stopped at c 7, line done at c 9). First crossing R
  0.0088 (pre-registered log-R interpolation between c 2, E32 25.02 at R 0.00641, and c 3, 26.48 at
  R 0.00918); the hold window c 3–c 7 completed at c 7 (R 0.0212; c 6 at R 0.01832 was just short
  of 2 × 0.00918). Two messages to the coordinator gave R 0.021 (the hold point) and 0.027 (the last
  checkpoint) as "the crossing"; corrected at 10:40Z.
- **10:25Z — ko_strict status** (interim message to the coordinator): kos_s0 E32 0 → 10.1 by R
  0.10, with σ's share of distinct Lean-accepted training proofs 0.3 % (L2 it 1) → 17 % (L2 it 37)
  → 31 % (L3 it 4); kos_s2 σ share 0 → 10 % by L2 it 33, E32 ≤ 1.0; kos_s1 σ share ≤ 0.1 %, E32
  ≤ 0.5. The three kos_*_dp lines: E32 ≤ 0.94, σ share 0–0.8 %, DN in 4–12 % of accepted proofs.
  Projection: about 27 GPU-h used, about 72 more if all six run to the cap (≈ 100 in all).
- **10:37Z — σ labels of accepted held-out proofs** (`sigma_proofs.py`, Dan's classifier): every
  Lean-accepted attempt on the 106 is σ in kos_s0 c 40 (224 attempts, 11 theorems), ko_s1 c 32
  (1,479, 48), full_s0_dp c 9 (906, 33), kos_s2 c 32 (2, 2) and kos_s1_dp c 32 (2, 1).
- **10:42:40Z — kos_s1: think-ban stop in L3 iteration 2** (`STOP it 2: think banned mass 0.00649 >
  0.005`; c 41, last read c 40 at R 0.1089, E32 0.00, best 0.5). Its L2 had raised the think-ban
  alarm on 17 of its 39 iterations (from iteration 18; mass 0.0010–0.0021, under the 0.005 stop;
  1 rollback, 16 with the restore point itself above 0.001), and L3's restore point (it 0) was
  already above the alarm level (L3 it 1 0.00297, it 2 0.00649), so the recipe could not roll
  back. (Corrected 11:05Z: my first count, 34, counted ALARM lines, two per alarmed iteration.) By the pre-registered stop rule the line ends there: "not reached by R 0.109".
  - My error: the queued `line2_kos_s1` (`--resume_line`, queued for crash recovery) relaunched L3
    at 10:44:49Z, because line.py's resume guard only skipped hold-window stops. I killed it at
    10:46Z before its first iteration (no checkpoint, no read), restored `metrics.jsonl`,
    `args.json` and the Lean dump to the original run (the resume's files kept as
    `*.resume_killed.*`; its log lines remain at the end of `kos_s1_L3.log`), and wrote
    `evals/kos_s1/think_ban_stop.json`.
  - Fix (deployed to all three machines, md5 8594826d): `--resume_line` now returns when
    `line_done.json` exists without `STOP_LINE` (a think-ban stop or the cap), and line_done
    records `think_ban_stop`. The other five kos lines' line2 jobs stay queued for crash recovery.
  - serrano GPU 1 is idle; nothing new is launched without the coordinator.
- **11:05Z — think-ban guard events per line** (`think_ban.py`, distinct iterations): alarms
  only on the ko_strict bases — kos_s1 19 of 41 iterations (1 stop), kos_s2 5 of 46, kos_s2_dp
  4 of 60; 0 on every other line (full, d1, d01, ko, kos_s0, kos_s0_dp, kos_s1_dp, full_s0_dp).
  Coordinator: show kos_s1 as censored by the guard (distinct marker), not as evidence either way.
