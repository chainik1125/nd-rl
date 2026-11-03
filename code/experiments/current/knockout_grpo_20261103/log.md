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
- **11:15Z — caption facts corrected** (coordinator): the narration-plan share is now the
  trained-level count, 637 of 154,630 ko records (0.41 %; 515 with the NEGI half), with the
  any-level 949 (0.61 %) given as ko_strict's removal count. Dan's x ÷ our R is 4.98–5.41 at
  round 1 (sampling only: his x counts the re-read state prefill, ours does not; his round-1
  crossings are drawn on the floor) and 1.53–1.90 at every later round; the caption's "1.5–5.2"
  had pooled the two. numbers.md lists the ratio per ladder.
- **11:59:09Z — kos_s2_dp: think-ban stop in L3 iteration 62** (c 101; mass 0.0035–0.0057 over
  iterations 59–62; last read c 100 at R 0.240, E32 0.00; best 0.9). Final by the stop rule;
  reported as censored at R 0.240. The fixed resume guard let `line2_kos_s2_dp` exit at once.
  ynez GPU 2 idle. analyze.py now reads think-ban stops from the stage logs (`think_ban.json`),
  since this line's line_done.json predates the `think_ban_stop` field.
- **12:15Z — think-ban guard check** (coordinator's three questions; CPU, logs and dumps only).
  Rule: `sample.phase_bans` (f35d3daf:112 = fa28f6eb:99) never samples `<eos>`, `<think>`,
  `<pad>` or the effort tokens inside a think block; `ban_think` is the mean full-vocabulary
  π_T mass on them at the scored think positions (grpo.py:837), alarm 1e-3, stop 0.005. No think
  text is stored (no `--dump_batches`; the Lean dump keeps the proof only), so the banned-mass
  contexts cannot be shown from existing files. Per line (iterations / alarms / max 5-iteration
  σ share of accepted training proofs): kos_s0 100 / 0 / 42 %, kos_s2 76 / 28 / 42 %, kos_s1
  41 / 19 + stop / 0.1 %, kos_s2_dp 101 / 29 + stop / 0.3 %, kos_s0_dp 106 / 10 / 0.3 %,
  kos_s1_dp 90 / 0 / 0.2 %. Within a line ban mass and σ share share a trend (Spearman 0.25–0.90
  on levels) but not their changes (−0.02 to 0.22 on first differences). Leon's full-base L3 runs
  with this recipe alarm too (kkl0_s1 2 / 39, vs_mb_s1 6 / 39, vs_s1 12 / 39, kg64_s1 15 / 39
  with max 5.7e-3, ki78_s0 22 / 78); no L2 run alarmed except kos_s1 (17) and kos_s2 (3). L2
  iteration-1 mass on our pool is 2.4–3.5 × Dan's pool's on the same kos base. Proposed a CPU
  scoring / sampling diagnostic for the token and context breakdown; waiting for approval.
  train_sigma.py now stops where a dump ends (kos_s1's killed resume left gate lines past it).
- **12:20–12:35Z — diagnostic A** (coordinator-approved; ynez CPU, 8 threads, nice, no GPU;
  `ban_diag.py` from the f35d3daf snapshot; results `ban_diag.json`): 500 ko_strict corpus records
  at effort 64 / 256 (half with DN in the proof), 54,699 think positions, teacher-forced, T 1.0.
  Mean banned mass on corpus narration: base_kos_s1 1.13e-5, kos_s1 L2 it39 9.4e-6, kos_s1 L3
  it2 (stop) 1.03e-5, kos_s2_dp L3 it6 1.30e-5, kos_s2_dp L3 it62 (stop) 8.37e-5 (46 % nested
  `<think>`; top positions start a goal line after a ¬¬ hypothesis line), kos_s0 L3 it61
  4.44e-5, ko_s1 L2 it32 1.00e-5. `Classical.byContradiction` contexts: 1–2 % of the mass in
  every kos checkpoint (ko_s1 8 %). kos_s1's own-sample mass (2.1e-3, 6.5e-3) is not elicited by
  corpus text, so locating it would need diagnostic B (not run).
- **13:07:01Z — kos_s0_dp: think-ban stop in L3 iteration 105** (c 144; mass 0.0047–0.0057 over
  iterations 102–105; last read c 141 at R 0.376, E32 0.00, best 0.5). Censored at R 0.376;
  `line2_kos_s0_dp` exited at once. temescal GPU 1 idle. Running: kos_s0 (E32 19.8 at R 0.438),
  kos_s2 (12.1 at R 0.335), kos_s1_dp (0.0 at R 0.261).
- **13:14:51Z — kos_s1_dp: think-ban stop in L3 iteration 86** (c 125; mass 0.0023–0.0055 over
  iterations 82–86; its pass@1 on Dan's pool was 91 % with think blocks down to 10.7 / 18.6 tokens
  at budget 64 / 256; last read c 112 at R 0.261, E32 0.00, best 0.8). Censored at R 0.261; all
  three Dan's-pool ko_strict lines are now censored (R 0.240, 0.261, 0.376), E32 ≤ 0.9 throughout.
  serrano has no line running. Running: kos_s0, kos_s2.
- **13:53:58Z — kos_s2: think-ban stop in L3 iteration 76** (c 115; mass 0.0046–0.0055 over
  iterations 74–76; last read c 112 at R 0.441, E32 12.96, best 13.4; σ share of accepted training
  proofs 40–46 % in its last L3 iterations). Censored at R 0.441 while rising: unlike the four
  earlier stops, this line was finding σ. ynez GPU 0 idle; only kos_s0 runs (temescal).
- **2026-11-03T14:13:50Z — Amendment 4 frozen** ([`amendment4.md`](amendment4.md), sha256
  `7121dcc149038258f6b7d984f889a9b7226a0380f1683999f309348da0444dc5`): guard-relaxed continuation
  (secondary, exploratory) of the five censored ko_strict lines from their last rolling
  checkpoints, `--ban_stop 0.05` and nothing else changed, with a health stop (reward or E32 below
  0.5 × peak for 25 iterations; E32 rule once its peak is ≥ 3.0). Proposed by the coordinator to
  Leon; launches 14:35Z unless halted. `line.py --gr` (setup tested on serrano at 14:12Z: the
  kos_s1_gr and kos_s1_dp_gr runs and curves were seeded, nothing launched); `mkjobs_gr.py`.
- **14:35:32–14:36:07Z — Amendment 4 launched** (no halt by 14:35Z): gr_kos_s1_gr (serrano GPU 1,
  resumed L3 it 0), gr_kos_s1_dp_gr (serrano GPU 0, it 85), gr_kos_s2_gr (ynez GPU 1, it 75),
  gr_kos_s2_dp_gr (ynez GPU 2, it 45), gr_kos_s0_dp_gr (temescal GPU 1, it 100); each with
  `--ban_stop 0.05` (checked in args.json) and a crash-recovery line2gr job. serrano: 2 of ours +
  1 other user's (GPU 2).
- **15:52:01Z — kos_s1_dp_gr: relaxed think-ban stop in L3 iteration 121** (mass 0.0611 > 0.05;
  c 160; last read c 158 at R 0.362, E32 0.00). Its mass grew 0.0035 (it 84) → 0.0498 (it 120),
  doubling about every 10 iterations, with the training reward flat at 0.86–0.89. At 15:58Z the
  other continuations' mass: kos_s2_dp_gr 0.017–0.026 and kos_s0_dp_gr 0.013–0.017 (rising),
  kos_s2_gr 0.003–0.004, kos_s1_gr 0.002–0.003; kos_s0 (primary) 0.0027–0.0029.
- **19:06:58Z — kos_s2_dp_gr: relaxed think-ban stop in L3 iteration 180** (mass 0.050 > 0.05;
  c 219; last read c 210 at R 0.685, E32 0.00, best 0.9 before the continuation). Its mass had
  peaked at 0.044 (it 115), eased to 0.026 (it 139) and climbed again; reward flat at 0.87–0.90.
  At 19:09Z: kos_s0 (primary) E32 25.0 at c 251 (R 1.22), just under the bar; kos_s2_gr 13.8 at
  R 0.99; kos_s1_gr 0.0 at R 0.44; kos_s0_dp_gr 0.0 at R 0.80.
- **19:15:18Z — kos_s0: think-ban stop in L3 iteration 234** (mass 0.00502 > 0.005; c 273; last
  read c 251 at R 1.224, E32 25.02, best 25.02, just under the bar 26.31). Its L3 had used all
  three rollbacks by iteration 89 (lr × 0.125 from then on), and from about iteration 89 nearly
  every iteration alarmed, so grpo saved alarm checkpoints (with training state) instead of
  rolling ones: the last rolling checkpoint is iteration 85 (c 124). All six ko_strict lines are
  now censored under the primary rule. temescal GPU 2 idle.
- **19:22Z — Amendment 4, kos_s0: wording deviation (this line only; coordinator's Option B).**
  kos_s0_gr resumes from `kos_s0_L3_it234_alarm.pt`, the checkpoint written at its stop, not from
  its last rolling checkpoint (iteration 85). Reason: grpo writes an alarm checkpoint instead of a
  rolling one on alarmed iterations, and kos_s0 alarmed on nearly every iteration after 89, so
  its last rolling checkpoint is 149 iterations before the stop with another lr history (× 0.25,
  2 rollbacks); the amendment asks what the line would have done with `--ban_stop 0.05`, and the
  alarm checkpoint carries the exact training state at the stop. Checked in its `train_state`:
  iteration 234, LoopState lr_scale 0.125, rollbacks 3 (of 3), n_step 892, last_good it 85;
  AdamW, curricula and RNGs present; saved args identical to the run's (T 364, lr 1.5e-4, ban_stop
  0.005, now overridden to 0.05). `line.py --gr_from` (md5 d5ba470d) names the checkpoint.
