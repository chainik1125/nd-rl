# Review exp21_s2 (exp21 recipe, seed 2; third seed)

## Recount (written before reading preregistration / numbers.md / log.md for this run)

### Recount of `/work/exec/runs/exp21_hot_focus_s2` (tools/recount.py)

- status `ok`, seed 2, budget 600, gpu `0:NVIDIA GeForce RTX 3090`, started_utc 2026-09-23T03:50:02Z, rl_secs 576.7, smoke False
- frozen hashes: 18 checked, mismatched none, not in SHA256SUMS none, frozen files absent from summary none; source/rl.py sha == summary rl.py sha: True
- summary metric 300, by_len {'7': 96, '8': 91, '9': 113}, heldout_greedy 0.883
- score.json: solved 300, by_len {'7': 96, '8': 91, '9': 113}, k 32, seed 0, official True, smoke False; final.pt sha matches: True; score == summary metric: True; by_len sum 300
- metrics.jsonl: 33 records {'sample': 5, 'train': 26, 'summary': 1, 'eval': 1}; step monotone True; sample rounds [1, 2, 3, 4, 5]; sum(samples) 221568; last targets_solved_cum 762
  - vs rl_stats: rounds 5 (=), total_samples 221568 (=), targets_solved_cum 762 (=), total_train_steps 1277 (max step 1278)
  - harness eval heldout_greedy 0.883 == summary: True
  - r1: n_prompts 1343 k 16 T 0.8 samples 48048 accept 0.0124 solved_round 103 cum 103 by_len {'7': 102, '8': 1, '9': 0} new 118
  - r2: n_prompts 1343 k 16 T 0.8 samples 48048 accept 0.0701 solved_round 354 cum 354 by_len {'7': 269, '8': 85, '9': 0} new 287
  - r3: n_prompts 1341 k 16 T 0.8 samples 45392 accept 0.145 solved_round 549 cum 552 by_len {'7': 374, '8': 162, '9': 16} new 225
  - r4: n_prompts 1278 k 16 T 0.8 samples 64752 accept 0.2171 solved_round 616 cum 692 by_len {'7': 467, '8': 192, '9': 33} new 163
  - r5: n_prompts 1256 k 4 T 0.8 samples 15328 accept 0.2535 solved_round 603 cum 762 by_len {'7': 526, '8': 199, '9': 37} new 75
  - train records 26: loss first 0.0624 last 0.0430; n_records last 11464
- found_targets.jsonl: 868 rows, 868 raw-distinct, 868 distinct by my normaliser; 762 targets solved, by L_true {7: 526, 8: 199, 9: 37}; names not in target pools: 0; thm field matches pool: 868/868
  - re-verified with nd_verify against the pool prompt: 150/150 accepted
- other files: ['round_01.pt', 'round_02.pt', 'round_03.pt', 'round_04.pt', 'round_05.pt']
textbook solved cum by round [0, 1, 11, 71, 120] schemas [('contraposition', 102), ('export', 18)]

source/rl.py byte-identical to exp21_hot_focus_s0. **Hard constraints:** pass. **Gate 1:** pass.

## Compare (after reading preregistration/exp21_s2.md, numbers.md, log.md)

Gate 0: prereg commit d708dcf at 2026-09-23T03:44:23Z; started_utc 03:50:02Z. **Pass.**

| claim (source) | reviewer's independent value | verdict |
|---|---|---|
| dev 300 (96/91/113), heldout 0.883 | 300, 96/91/113; 0.883 | reproduces |
| textbook cum 0/1/11/71/120 (5 rounds) | 0/1/11/71/120; contraposition 102, export 18 | reproduces |
| "not falsified" (falsified if no ignition by r3 or ≤ 290) | 11 solves by r3; 300 > 290 | reproduces **if** 11 counts as ignition. This prereg gives no threshold, while exp21 s0 and exp19 s1 used "≥ 20 by round 3". By that earlier bar the run would be falsified. Threshold drift; state it |
| three-seed mean 313.7 (323/318/300) | 313.7; SD ≈ 12 | reproduces |

Pattern across the two 3-seed recipes: exp19 328/301/309 (312.7), exp21 323/318/300 (313.7). **They are indistinguishable**, and the gen-stream focus (the only difference) has no detectable effect.

## Verdict

**REJECT** (300 < 324). Heldout 0.883 is among the lowest of the textbook family.
